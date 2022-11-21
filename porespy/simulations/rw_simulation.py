import numpy as np
from tqdm import tqdm
import porespy as ps
from scipy import constants as C
import dask
import dask.array as da
from functools import wraps
from time import perf_counter


__all__ = [
    'calc_gas_props',
    'compute_steps',
    'rw',
]


def timer(func):
    # This decorator function prints the execution time of the decorated function
    @wraps(func)
    def wrap_func(*args, **kwargs):
        t1 = perf_counter()
        result = func(*args, **kwargs)
        t2 = perf_counter()
        print(f'Function {func.__name__!r} executed in {(t2 - t1):.4f}s')
        return result
    return wrap_func


def calc_gas_props(P, T, MW, mu):
    r"""
    Uses kinetic theory of gases to computes mean free path, average velocity,
    and diffusion coefficient of a molecule at given thermodynamic conditions

    Parameters
    ----------
    P : float
        The gas pressure in units of [Pa]
    T : float
        The gas temperature in units of [K]
    MW : float
        The molecular weight of the diffusion gas molecule in units of [kg/mol]
    mu : float
        The dynamic viscosity of the gas mixture in units of [Pa.s]

    Returns
    -------
    mpf : dict
        A dictionary with the following keys. This can be passed directly to the
        ``compute_steps`` function as kwargs (i.e., **kinetics).

        ======= =============================================================
        key     description
        ======= =============================================================
        mfp     float, Mean free path of the gas in units of [nm]
        v_rel   float, The kinetic velocity of the gas in units of [nm/s]
                relative to other moving gas particles, calculated using
                kinetic gas theory
        Db      float, The theoretical bulk phase diffusion coefficient in
                units of [m\u00b2/s], calculated using kinetic gas theory
        P       float, the pressure of the simulation in units of [Pa],
                as passed in
        T       float, the temperautre in units of [K]
        MW      float, the molecular weight of the diffusing molecule in
                units of [kg/mol]
        mu      float, the viscosity of the gas mixture in which the
                molecular is diffusion in units of [Pa.s]
        ======= =============================================================

    Notes
    -----
    In order to perform a Knudsen random walk, we need the mean free path.
    The gas velocity and bulk diffusion coefficients are also calculated
    from kinetic theory, as these are necessary if we wish to directly compute
    the effective diffusivity later.

    The mean free path is defined as:
        $$ \lambda = \frac{\mu}P \sqrt{(\frac{\pi RT}{(2*MW)})} $$
    The gas velocity is defined as:
        $$v =  4 \sqrt{\frac{RT}{\pi*MW}} $$
    And bulk diffusion is defined as:
        $$ D_b=\frac{1}{2} \lambda v $$

    """
    mfp = mu / P * np.sqrt(np.pi * C.R * T / (2 * MW)) / 1e-9
    # path_length = mfp / resolution
    v_rel = np.sqrt(2) * np.sqrt(8 / np.pi * C.R * T / MW) / 1e-9
    Db = 1 / 2 * mfp * v_rel * (1e-9) ** 2
    kinetics = {}
    kinetics['mfp'] = mfp
    kinetics['v_rel'] = v_rel
    kinetics['Db'] = Db
    kinetics['T'] = T
    kinetics['P'] = P
    kinetics['MW'] = MW
    kinetics['mu'] = mu
    return kinetics


def compute_steps(
    mfp,
    v_rel,
    Db,
    ndim,
    voxel_size=1,
    n_steps=10000,
    steps_per_mfp=10,
    **kwargs
):
    r"""
    Helper function to compute the step size information in a form suitable for the
    ``rw`` function

    Parameters
    ----------
    mfp : float
        The mean-free path of the walker in units of [nm]
    v_rel : float
        The relative velocity of the walker in units of [nm/s]
    Db : float
        The bulk diffusivity of the walker in units of [m2/s]
    ndim : int
        The dimension of the image which will be passed into the random walk
        simulation. Must be '2' or '3'.  The number of dimensions of an image can
        be obtained with ``im.ndim``.
    voxel_size : float
        The resolution of the image in units of [nm/voxel]
    n_steps : int
        Number of steps to take before exiting the walk.  A higher number
        is needed if the void space is highly tortuous so the walkers can
        probe all the space, but this takes longer to run.
    steps_per_mfp : int
        The number of steps that each mean free path should be broken into. Since
        the walkers check if they have collided with solid after each step, this
        number controls how accurately these collisions are resolved. A high
        number may be more accurate but slows the simulation since walkers move
        less distance on each step.
    kwargs
        All additional keyword arguments are ignored, but are included in the
        returned dictionary. This means they will be passed onto ``rw``. For
        instance, if ``n_walkers`` is included it will not be used by this function,
        but means that it need not be explicitly passed to ``rw``.

    Returns
    -------
    A walk object : dict
        A dictionary object with following keys:

        =============== ======================================================
        attribute       description
        =============== ======================================================
        step_size       float, The distance a walker should take on each step
                        in units of [voxels]. This should not be more than 1
                        voxel since this would cause walker to penetrate walls.
                        It should also be much less than ``path_length`` so
                        that walker does not overshoot the desired mean-free
                        path by too much.
        path_length     float, The distance a walker should travel in [voxels]
                        before randomly choosing another direction.  If a
                        walker hits a wall before this distance is reached it
                        also changes direction.
        time_step       float, The time it takes to complete each step [s]
        steps_per_mpf   as passed in
        voxel_size      as passed in
        n_steps         as passed in
        mfp             as passed in
        v_rel           as passed in
        Db              as passed in
        =============== ==========================================================


    Notes
    -----
    In order to simulate Knudsen diffusion, the walkers need to take many small
    steps over the course of each mean free path, during which they don't change
    direction unless they hit a wall. In a regular random walk, each step is one
    voxel and the walkers change directions after each step. For this reason,
    when we tell the simulation to take 30000 steps, but to take 50 steps per
    mean free path, we are simulating for 600 mean free path lengths, which is
    equivalent to 600 steps in a regular random walk.

    A few requirements here:

    'step_size' is a calculated value, but cannot exceed 1 voxel.
        If it does, the walkers could theoretically walk through walls which
        is not ideal :).
    'step_size' must be less than or equal to the path_length.
        The default value is 50 steps per path_length - more steps and the
        precision of the walkers increases, but more steps are required to
        simulate the same period of time
    if 'path_length' << 1 voxel, then either the image is too coarse, or
    Knudsen diffusion is likely to be insignificant in your image, in which
    case it is suggested to use knudsen=False

    """
    walk = kwargs
    walk['voxel_size'] = voxel_size
    walk['n_steps'] = n_steps
    walk['steps_per_mfp'] = steps_per_mfp
    walk['mfp'] = mfp
    walk['v_rel'] = v_rel
    walk['Db'] = Db
    walk['path_length'] = walk['mfp'] / walk['voxel_size']
    path_length = walk['path_length']
    step_size = path_length / steps_per_mfp
    if step_size > 1:
        raise ValueError('Step size exceeds 1 voxel, try larger steps_per_mfp')
    walk['step_size'] = step_size
    walk['time_step'] = step_size * voxel_size / (walk['v_rel'] * ndim)
    return walk


def rw(
    im,
    path_length,
    n_steps=1000,
    step_size=None,
    n_walkers=1000,
    n_write=1,
    seed=None,
    start=None,
    edges='periodic',
    mode='random',
    **kwargs,
):
    r"""
    Perform a random walk on an image using the given parameters.

    All arguments are in units of voxels. Use the ``compute_steps`` helper function
    if necessary to convert physical sizes to voxels.

    Parameters
    ----------
    im : ndarray
        The image of void space in which the walk should occur, with ``True``
        values indicating the phase of interest.
    path_length : float
        The length of the mean-free path in units of [voxels]. When walkers
        travel this distance without hitting a wall they are assigned a new
        direction vector and they travel along a new path.
    step_size : float
        The length of each step taken by the walkers in units of [voxels]. This
        value should be some fraction of the ``path_length``, and effectively
        controls the resolution of walk. If ``step_size`` is too large, then
        walkers will not accurately detect collsions with walls. A small value
        will give more accurate results but means that longer run times are
        required to probe the same amount of pore space. Given that the solid
        is approximated as a rough voxel surface, this accuracy is probably not
        justified. If not provided then it is assumed to be 10% of the path length.
    n_steps : int
        Number of steps to take before exiting the simulation. A high number
        is needed if the void space is highly tortuous so the walkers can
        probe all the space, but this takes longer to run.
    n_walkers : int
        Number of walkers to use.  A higher number gives less noisy data but
        takes longer to run.
    n_write : int
        The number of steps to take between recording walker positions in
        the `path' array. Default is 10, in which case all steps are recorded.
        Values larger than 1 enable longer walks without running out of memory
        due to size of `path`. A value of 1 is really only necessary if a
        visualization of the walker paths is planned, in which case `n_walkers`
        should be lower.
    start : ndimage
        A boolean image with ``True`` values indicating where walkers should
        start. If not provided then start points will be selected from the
        void space at random.  If ``n_walkers`` > ``start.sum()``, then multiple
        walkers will start from the same location, including if ``start.sum()`` is
        1 (i.e. a single pixel).
    seed : int
        A seed value for the random number generator, which ensures repeatable
        results. Default is ``None``, in which case each simulation will be
        different.
    edges : string
        How walkers should behave when walking past an image boundary.
        Options are:

        ============= ========================================================
        Option        Description
        ============= ========================================================
        'periodic'    (default) When a walker exits a edge of an image, it
                      continues as if the image were tiled so it sees the
                      properties of the opposite side of the image.
        'symmetric'   When a walker exits the edge of an image, it continues
                      as if the image were flipped so it sees the properties
                      similar to the edge of just exited.
        ============= ========================================================

    mode : string
        How walkers choose their directions.  Options are:

        ============= ========================================================
        Option        Description
        ============= ========================================================
        'random'      (default) Walkers choose a random angle and walk in that
                      direction until they hit a wall or reach their
                      ``path_length``.
        'axial'       Walkers follow the cardinal axes of the image.
        ============= ========================================================

    Returns
    -------
    paths : ndarray
        A numpy ndarray of size ``[n_steps, n_walkers, im.ndim + 1]``. The
        [x, y, [z], step_num] of the walker at each step is recorded in the
        final axis.
    """
    # Parse input arguments
    step_size = path_length/100 if step_size is None else step_size
    n_write = int(n_write) if n_write > 1 else 1
    # Initialize random number generator
    rng = np.random.default_rng(seed)
    # Allocate empty array for path info
    path = np.zeros([int(n_steps / n_write), n_walkers, im.ndim + 1])
    # Generate the starting conditions:
    if start is None:
        start = im
    start = _get_start_points(start, n_walkers, rng)
    # Get initial direction vector for each walker
    x, y, z = _new_vector(N=n_walkers, L=step_size, ndim=im.ndim, mode=mode, rng=rng)
    loc = np.copy(start)  # initial location for each walker
    path[0, :, :-1] = loc.T  # save locations to path
    path[0, :, -1] = 0  # Add step number
    i = 0
    with tqdm(range(n_steps - 1), **ps.settings.tqdm) as pbar:
        while i < (n_steps - 1):
            # if image is 2D, this excludes the z coordinate
            new_loc = loc + np.array([x, y, z])[:im.ndim, :]
            # has any walker gone past its mean free path?
            temp = np.sum((new_loc - start) ** 2, axis=0)
            check_mfp = np.around(np.sqrt(temp), decimals=5) > path_length
            if np.any(check_mfp):
                # Find the walker indices which have gone past their mfp
                inds_mfp = np.where((check_mfp == True))
                # Regenerate direction vectors for these walkers
                x[inds_mfp], y[inds_mfp], z[inds_mfp] = \
                    _new_vector(N=len(inds_mfp[0]),
                                L=step_size,
                                ndim=im.ndim,
                                mode=mode,
                                rng=rng)
                # Update starting position of invalid walkers to current position
                start[:, inds_mfp] = loc[:, inds_mfp]
                # Re-update new location with walkers that changed direction
                new_loc = loc + np.array([x, y, z])[:im.ndim, :]
            wrapped_loc = _wrap_indices(new_loc, im.shape, mode=edges)
            # has any walker passed into solid phase?
            check_wall = im[wrapped_loc] == False
            if np.any(check_wall):  # If any walkers have moved into solid phase
                inds_wall = np.where((check_wall == True))  # Get their indices
                # Regenerate direction vectors for walkers who hit wall, but wait
                # till next step to apply direction. This imitates having them
                # hit the wall and bounce back to original location.
                x[inds_wall], y[inds_wall], z[inds_wall] = \
                    _new_vector(N=len(inds_wall[0]),
                                L=step_size,
                                ndim=im.ndim,
                                mode=mode,
                                rng=rng)
                # Reset mean free path start point for walkers that hit the wall
                start[:, inds_wall] = loc[:, inds_wall]
                # Walkers that hit a wall return to previous location,
                # with a new direction vector
                new_loc[:, inds_wall] = loc[:, inds_wall]
            loc = new_loc  # Update location of each walker with trial step
            i += 1  # Increment the step index
            # every stride steps we save the locations of the walkers to path
            if i % n_write == 0:
                path[int(i / n_write), :, :-1] = loc.T  # Record new position of walkers
                path[int(i / n_write), :, -1] = i  # Record index of current step
            pbar.update()
    return path


@timer
def rw_parallel(im,
                walk,
                n_walkers=1000,
                start=None,
                edges='periodic',
                mode='random',
                stride=1,
                chunks=None,
                cores=None,
                seed=None,):
    """
    This is a wrapper function for 'rw', which splits the walkers into groups
    and performs parallel processing on them.
    TODO: Write as decorator??

    Parameters
    ----------
    im : same as in rw
    walk : same as in rw
    n_walkers : same as in rw
    start : same as in rw
    edges : same as in rw
    mode : same as in rw
    stride : same as in rw
    chunks : int
        The number of chunks to split the walkers into for parallel processing
    cores : int
        The number of cores to use. Defaults to all cores.
    seed : int
        A starting seed to generate RNG. Is used to generate separate seeds for
        each parallel process. A default of None generates a non-reproducible
        simulation (RNG generated from system entropy)

    Returns
    -------
    The updated walk Results object with the path attribute:
    path : ndarray
        A numpy ndarray of size ``[n_steps, n_walkers, im.ndim]``. The
        [x, y, [z]] location of the walker at each step is recorded in the
        final column.

    In addition to all previous attributes, and the parameters of 'rw_parallel'
    added to it.
    """
    # Leave cores alone if specified, else set to # of cores in machine
    cores = cores if cores else ps.settings.ncores
    # Leave chunks alone if specified, else set to # of cores
    chunks = chunks if chunks else cores
    if cores > 1:
        ps.settings.tqdm['disable'] = True
    # the number of walkers to dispatch to each process/chunk
    walkers_per_chunk = n_walkers // chunks
    # Generate an independent random state (seed) for each chunk
    ss = np.random.SeedSequence(seed)
    # Generate a starting seed for each chunk/parallel process
    states = ss.spawn(chunks)
    #  Save function parameters into walk:
    walk.im = im
    walk.porosity = ps.metrics.porosity(im)
    walk.n_walkers = n_walkers
    walk.stride = stride
    walk.mode = mode
    walk.edges = edges

    n_steps = walk.n_steps
    delayed_walks = []

    for i in range(chunks):
        path = da.from_delayed(dask.delayed(rw)(im=da.from_array(im),
                                                walk=walk,
                                                n_walkers=walkers_per_chunk,
                                                edges=edges,
                                                mode=mode,
                                                seed=states[i],
                                                stride=stride,
                                                start=start,
                                                _parallel=True),
                               shape=(n_steps / stride, walkers_per_chunk, im.ndim),
                               dtype='float32')
        delayed_walks.append(path)
    # Before computing, concatenate all returned 'path' dask arrays into single array
    paths_delayed = da.concatenate(delayed_walks, axis=1)
    path = paths_delayed.compute()
    walk.path = path
    return walk


def _new_vector(N=1, L=1, ndim=3, mode='random', rng=None):
    r"""
    Generate a new set of displacement vectors

    Parameters
    ----------
    N : int
        The number of vectors to generate
    L : float
        The length of the vectors.  All will be the same length.
    ndim : int (2 or 3)
        The number of dimensions for the vectors
    rng : Generator
        A random number generator as created by np.random.default_rng(seed).
        Using this ensures repeatability Default is None, which generates
        random numbers from system entropy (ie not reproducible)
    mode : string
        How to determine the angle of the vectors.  Options are:

        ========= ============================================================
        mode      description
        ========= ============================================================
        'random'  Angles are continuous between 0 and 2pi
        'axial'   Angles follow the axial directions of the image
        ========= ============================================================

    Returns
    -------
    vectors : ndarray
        A numpy ndarray containing the ``[x, y, z]*L`` values for each ``N``
        vectors.
    """
    if rng is None:
        rng = np.random.default_rng()
    if mode == 'random':
        # Generate random theta and phi for each walker
        # This was taken from https://math.stackexchange.com/a/1586185
        u, v = np.vstack(rng.random((2, N)))
        q = np.arccos(2 * u - 1) - np.pi / 2
        f = 2 * np.pi * v

        # Convert to components of a unit vector
        if ndim == 3:
            x = np.cos(f) * np.cos(q)
            y = np.cos(q) * np.sin(f)
            z = np.sin(q)
        else:
            x = np.cos(f)
            y = np.sin(f)
            z = np.zeros(N)
        # x, y, z = x * ndim ** 0.5, y * ndim ** 0.5, z * ndim ** 0.5
        # x, y, z = x * ndim, y * ndim, z * ndim
    elif mode == 'axial':
        if ndim == 2:
            options = np.array([[0, 1], [1, 0],
                                [0, -1], [-1, 0]])
            # options = np.array([[1, 1], [1, -1],
            #                     [-1, -1], [-1, 1]])
            # x, y = options[np.random.randint(0, len(options), N), :].T
            x, y = options[rng.integers(0, len(options), N), :].T
            z = np.zeros(N)
        if ndim == 3:
            options = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0],
                                [0, 0, -1], [0, -1, 0], [-1, 0, 0]])
            # options = np.array([[1, 1, 1], [1, 1, -1], [1, -1, -1], [1, -1, 1],
            #                     [-1, -1, -1], [-1, -1, 1], [-1, 1, 1], [-1, 1, -1]])
            # x, y, z = options[np.random.randint(0, len(options), N), :].T
            x, y, z = options[rng.integers(0, len(options), N), :].T
        # x, y, z = x / ndim ** 0.5, y / ndim ** 0.5, z / ndim ** 0.5
    else:
        raise Exception(f'Unrecognized mode {mode}')
    temp = np.array((x, y, z)) * L
    return temp


def _get_start_points(im, n_walkers=1, rng=None):
    r"""
    Generates the requested number of random starting points constrained to
    locations where ``im`` is ``True``

    Parameters
    ----------
    im : ndarray
        A boolean array with ``True`` values indicating valid potential
        starting points
    n_walkers : int
        The number of starting points to generate. These will be chosen
        randomly from the potential sites in ``im``.
    rng : Generator
        A random number generator as created by np.random.default_rng().
        Using this ensures repeatability. Default is None, which generates
        random numbers from system entropy (ie not reproducible).

    Returns
    -------
    points : ndarray
        An array of shape ``[ndim, n_walkers]`` with coordinates for each
        walker's starting position

    """
    # Find all voxels in im that are not solid
    options = np.where(im)
    if rng is None:
        rng = np.random.default_rng()
    inds = rng.integers(0, len(options[0]), n_walkers)
    # Convert the chosen list location into an x,y,z index
    points = np.array([options[i][inds] for i in range(im.ndim)])
    return points.astype(float)


def _wrap_indices(loc, shape, mode='periodic'):
    r"""
    Given a set of array indices, wraps them to lie within the given shape

    Parameters
    ----------
    loc : ndarray
        The array of indices to wrap, of shape ``[ndim, nlocs]``
    shape : array_like
        The shape of the domain within which the indices are confined
    mode : string
        Indicates how wrapping should occur. Options are:

        ============= ========================================================
        Option        Description
        ============= ========================================================
        'periodic'    When a walker exits a edge of an image, it continues as
                      if the image were tiled so it sees the properties of the
                      opposite side of the image.

        'symmetric'   When a walker exits the edge of an image, it continues
                      as if the image were reflected so it sees the properties
                      similar to the edge it just exited.
        ============= ========================================================

    Returns
    -------
    new_loc : ndarray
        A numpy ndarray of the same shape as ``loc`` with indices that
        extended beyond ``shape`` wrapped accordingly.

    """
    shape = np.array(shape)
    loc = np.array(loc)
    temp = np.zeros_like(loc)
    if mode == 'periodic':
        for i in np.arange(shape.size):
            temp[i] = np.round(loc[i]) % (shape[i])
        # temp = periodic_range(shape, loc, temp)
    elif mode == 'symmetric':
        for i in range(len(shape)):
            x = np.around(loc[i])
            x = x + (x < 0)  # Offset negative numbers by 1
            x = np.abs(x)  # Deal with abs since now symmetric
            N_wrap = x // shape[i]  # Find number of times indices wrap
            N_even = 1 - N_wrap % 2  # Note even vs odd number of wraps
            # Deal with even and odd numbered wraps differently
            temp[i] = x - N_even * shape[i] * N_wrap + \
                      (1 - N_even) * ((shape[i] - x % shape[i]) - x - 1)
    else:
        raise Exception(f'Unrecognized mode {mode}')
    inds = tuple(temp.astype(int))
    return inds
