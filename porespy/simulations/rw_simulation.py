import numpy as np
from tqdm import tqdm
import porespy as ps
from scipy import constants as C
import dask
import dask.array as da
from porespy.tools import Results
from functools import wraps
from time import perf_counter


def timer(func):
    # This function shows the execution time of
    # the function object passed
    @wraps(func)
    def wrap_func(*args, **kwargs):
        t1 = perf_counter()
        result = func(*args, **kwargs)
        t2 = perf_counter()
        print(f'Function {func.__name__!r} executed in {(t2 - t1):.4f}s')
        return result

    return wrap_func


def calc_kinetic_theory(
    P=101325,
    T=298,
    MW=0.032,
    mu=1.9e-5,
    display=True
):
    r"""
    Computes mean free path, average velocity and diffusion coefficient.

    Uses kinetic theory and gas properties to compute the mean free path,
    relative velocity and theoretical bulk diffusion coefficient of a gas
    in free space. The defaults are set to simulate oxygen diffusion at STP
    (101.3 kPa and 298 K)

    Parameters
    ----------
    P : float
        The gas pressure in units of [Pa]
    T : float
        The gas temperature in units of [K]
    MW : float
        The molecular weight of the gas molecule in units of [kg/mol]
    mu : float
        The dynamic viscosity of the gas in units of [Pa.s]
    display: bool
        Whether to print out the results

    Returns
    -------
    A Results object with named attributes:
        mfp : float
            Mean free path of the gas in units of [nm]
        v_rel : float
            The kinetic velocity of the gas in units of [nm/s] relative to other
            moving gas particles, calculated using kinetic gas theory.
        Db : float
            The theoretical bulk phase diffusion coefficient in units of [m\u00b2/s],
            calculated using kinetic gas theory.

    Notes
    -----
    In order to perform a Knudsen Random Walk, we need the mean free path.
    The gas velocity and bulk diffusion coeffiecients are also calculated
    from kinetic theory, and are necessary if we wish to directly compute
    the effective diffusivity using the included 'effective_diffusivity' function.

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
    if display:
        print('_' * 60)
        print('*Kinetic Theory Parameters*')
        print('mean free path:'.ljust(35), f'{mfp:0.5}', 'nm')
        print('mean relative velocity:'.ljust(35), f'{v_rel / 1e9:0.5}', 'nm/ns')
        print('Theoretical open-space diffusion:'.ljust(35), f'{Db:.5}', '[m\u00b2/s]')
        print('_' * 60)
    kinetics = Results()
    kinetics.mfp = mfp
    kinetics.v_rel = v_rel
    kinetics.Db = Db
    return kinetics


def compute_steps(
        kinetics,
        ndim,
        n_steps=1000,
        resolution=1,
        steps_per_path=50,
        knudsen=True,
):
    r"""
    Sets up the steps, path length and step sizes used in the random walk
    simulation

    Parameters
    ----------
    kinetics : Results
        the return value from 'calc_kinetic_theory'
    ndim : int
        The dimension of the image which will be passed into the random walk
        simulation. Must be '2' or '3'
    n_steps : int
        Number of steps to take before exiting the walk.  A high number
        is needed if the void space is highly tortuous so the walkers can
        probe all the space, but this takes longer to run.
    resolution : float
        The resolution of the image in units of [nm]
    steps_per_path: int
        The number of steps to split each path-length into. More steps
        increases accuracy, but also reduces the time interval over which
        the simulation is completed for a given number of steps.
    knudsen : bool = True
        Determines whether the simulation will consider knudsen diffusion, or
        if it should do a regular random walk
        TODO: test knudsen=False -> may currently be broken

    Returns
    -------
    A walk object : Results
        with named attributes as follows:

        step_size : float
            The distance a walker should take on each step in units of [voxels]. This
            should not be more than 1 since this would makes steps longer than a
            voxel, and should also be less than ``path_length``.
        time_step : float
            the time it takes to complete each step [s]
        path_length : float
            The distance a walker should travel in [voxels] before randomly
            choosing another direction.  If a walker hits a wall before this
            distance is reached it also changes direction.
        n_mfps : float
            the number of mean free paths that each walker will undergo
            during the simulation.
        The parameters of the function are also saved into the object for
        convenience later:
            ndim : as passed in
            resolution : as passed in
            n_steps : as passed in
            steps_per_path: as passed in
            knudsen : as passed in
            mfp : extracted from kinetics object
            v_rel : extracted from kinetics object
            Db : extracted from kinetics object

    Notes
    -----
    In order to simulate knudsen diffusion, the walkers need to take many small
    steps over the course of each mean free path, during which they don't change
    direction unless they hit a wall. In a regular random walk, each step is one
    voxel and the walkers change directions after each step. For this reason, when
    we tell the simulation to take 30000 steps, but to take 50 steps per mean free path,
    we are simulating for 600 mean free path lengths, which is equivalent to 600 steps
    in a regular random walk.

    A few requirements here:

    'step_size' is a calculated value, but cannot exceed 1 voxel.
        If it does, the walkers could theoretically walk through walls which
        is not ideal :).
    'step_size' must be less than or equal to the path_length.
        The default value is 50 steps per path_length - more steps and the
        precision of the walkers increases, but more steps are required to
        simulate the same period of time
    if 'path_length' << 1 voxel, then either the image is too coarse, or
    knudsen diffusion is likely to be insignificant in your image, in which
    case it is suggested to use knudsen=False

    """
    walk = Results()
    walk.ndim = ndim
    walk.resolution = resolution
    walk.n_steps = n_steps
    walk.steps_per_path = steps_per_path
    walk.knudsen = knudsen

    if not knudsen:
        walk.mfp = resolution
        path_length = 1  # voxel
        step_size = path_length  # each step completes an entire path
        walk.step_size = walk.path_length = step_size
    else:
        walk.mfp = kinetics.mfp
        walk.v_rel = kinetics.v_rel
        walk.Db = kinetics.Db
        path_length = walk.path_length = walk.mfp / resolution

        step_size = walk.step_size = path_length / steps_per_path
        if step_size > 1:
            raise ValueError(f'Step size cannot exceed one voxel: {step_size}')
        walk.time_step = step_size * resolution / (walk.v_rel * ndim)
        walk.n_mfps = n_steps * step_size / path_length
    print('_' * 80)
    print('*Step information*')
    print(f'Number of partial steps: {n_steps}')
    print(f'Step size: {step_size:0.4f} voxels | {step_size * resolution:0.4f} nm')
    print(f'Path length: {path_length:0.2f} voxels | {walk.mfp:0.2f} nm')
    if knudsen:
        print(f'Number of path lengths in simulation: {walk.n_mfps:0.2f}')
    print('_' * 80)
    return walk


def rw(im,
       walk,
       n_walkers: int = 1000,
       stride: int = 1,
       seed=None,
       _parallel=False,
       start=None,
       edges='periodic',
       mode='random',
       same_start=False):
    r""" Perform a random walk on an image.

    Walkers that hit a wall move 0 distance for that step

    Parameters
    ----------
    im : ndarray
        The image of void space in which the walk should occur, with ``True``
        values indicating the voids.
    walk : Results
        The Results object return by 'compute_steps'. Contains passed and
        computed attributes required for the simulation
    n_walkers : int
        Number of walkers to use.  A higher number gives less noisy data but
        takes longer to run.
    stride : int
        The number of steps to take between recording walker positions in
        the `path' variable. Must be >= 1. Default is 1, in which case all
        steps are recorded. Values larger than 1 enable longer walks without
        running out of memory due to size of `path`.
    start : ndimage
        A boolean image with ``True`` values indicating where walkers should
        start. If not provided then start points will be selected from the
        void space at random. In some cases it is necessary or interesting
        to start all walkers a common point or at a boundary.
    same_start : bool
        Default is False. If True, all walkers will start from the same
        randomly generated location within the image. If `start` is also
        provided, this will select one point from the subset of locations
        in `start`
    seed : int
        A seed value for a random number generator, which ensures repeatable
        results. Default is None, in which case each simulation will be
        different.
    _parallel :
        Indicates if this function is being called in parallel. For internal use
        when this function is called from rw_parallel
        TODO: is there a way to get rid of this?
    edges : string
        How walkers should behave when walking past an image boundary.
        Options are:

        ============= ========================================================
        Option        Description
        ============= ========================================================
        'periodic'    When a walker exits a edge of an image, it continues as
                      if the image were tiled so it sees the properties of the
                      opposite side of the image.

        'symmetric'   When a walker exits the edge of an image, it continues
                      as if the image were flipped so it sees the properties
                      similar to the edge of just exited.
        ============= ========================================================

    mode : string
        How walkers choose their directions.  Options are:

        ============= ========================================================
        Option        Description
        ============= ========================================================
        'random'      Walkers choose a random angle and walk in that direction
                      until the high a wall or reach their ``path_length``.

        'axial'       Walkers follow the cardinal axes of the image.
        ============= ========================================================

    Returns
    -------
    path : ndarray
        A numpy ndarray of size ``[n_steps, n_walkers, im.ndim]``. The
        [x, y, [z]] location of the walker at each step is recorded in the
        final column.
    """

    rng = np.random.default_rng(seed)
    if not _parallel:
        # Save settings to walk object for easier passing between functions
        # if parallel is True, this means 'rw' is being called from 'rw_parallel',
        # where these attributes have already been stored.
        walk.im = im
        walk.n_walkers = n_walkers
        walk.stride = stride
        walk.mode = mode
        walk.edges = edges
        walk.porosity = ps.metrics.porosity(im)

    step_size = walk.step_size
    n_steps = walk.n_steps
    path_length = walk.path_length

    path = np.zeros([int(walk.n_steps / stride), n_walkers, im.ndim])

    # Generate the starting conditions:
    if start is None:
        start = _get_start_points(im, n_walkers, rng, same_start=same_start)
    else:
        start = _get_start_points(start, n_walkers, rng, same_start=same_start)
    # Get initial direction vector for each walker
    x, y, z = _new_vector(N=n_walkers, L=step_size, ndim=im.ndim, mode=mode, rng=rng)
    loc = np.copy(start)  # initial location for each walker
    path[0, :] = loc.T  # save locations to path
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
                # Walkers that hit a wall return to previous location, with a new direction vector
                new_loc[:, inds_wall] = loc[:, inds_wall]
            loc = new_loc  # Update location of each walker with trial step
            i += 1  # Increment the step index
            # every stride steps we save the locations of the walkers to path
            if i % stride == 0:
                path[int(i / stride), :] = loc.T  # Record new position of walkers
            pbar.update()
    # if this is the main function being run, we add path to the walk variable
    if not _parallel:
        walk.path = path
        return walk
    else:
        # if 'rw' is being called from 'rw_parallel', just return the path, so
        # it can be concatenated with the paths return by other processes
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
                seed=None,
                same_start=False):
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

    same_start : same as in 'rw'
        TODO: might be generating a single starting point for each process

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
                                                same_start=same_start,
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

    temp = np.array((x, y, z)) * L
    return temp


def _get_start_points(im, n_walkers=1, rng=None, same_start=False):
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
    same_start : bool
        If ``True``, this will force all the walkers to start from the same
        point voxel in the image. Otherwise, they start at randomly chosen
        locations throughout the void space of the image.

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
    # From this list of options, choose one for each walker
    if same_start:
        index = rng.integers(0, len(options[0]))
        inds = np.ones(n_walkers, dtype=int) * index
    else:
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
                      as if the image were flipped so it sees the properties
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
