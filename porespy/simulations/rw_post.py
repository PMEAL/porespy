import sys
import numpy as np
import porespy as ps
import matplotlib.pyplot as plt
from collections import namedtuple
from rw_simulation import _wrap_indices
from scipy.stats import linregress


def rw_to_displacement(walk, inplace=True):
    r"""
    Compute the displacement of each walker from their starting point in [nm]

    Parameters
    ----------
    walk :
        The random walk Results object, as returned from 'rw' or 'rw_parallel'
    inplace : bool
        If 'True', the function will add the calculated displacements to the
        walk object.

    Returns
    -------
    displacements : Displacement
        A namedtuple with four computed results:
        the total displacement and the directional values, in the form (d, dx, dy, dz).
        These are also added as attributes to the walk object if inplace is `True`.

    Notes
    -----
    The returned result contains the progressive displacement at each step,
    not just the final value. This is needed for plotting the mean-squared
    displacement as a function of time.
    """

    path = walk.path
    resolution = walk.resolution
    # Get the total displacements in each direction and overall
    dx = path[:, :, 0] - path[0, :, 0]  # all x's subtract initial x position
    dy = path[:, :, 1] - path[0, :, 1]
    try:
        dz = path[:, :, 2] - path[0, :, 2]
    except:
        dz = np.zeros(np.shape(dx))
    d = (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5

    # Scale the displacements from voxels to nm
    d = d * resolution
    dx = dx * resolution
    dy = dy * resolution
    dz = dz * resolution
    if inplace:
        walk.displacement = d
        walk.displacement_x = dx
        walk.displacement_y = dy
        walk.displacement_z = dz
    Displacement = namedtuple('Displacement', 'displacement displacement_x displacement_y displacement_z')
    dt = Displacement(displacement=d, displacement_x=dx, displacement_y=dy, displacement_z=dz)
    return dt


def effective_diffusivity(walk, inplace=True):
    r"""
    Calculates the effective diffusivity from the random walk simulation

    Uses a kinetic theory computation to get effective diffusivity, which requires
    having computed the mean free path and relative velocity as returned by the function
    'calc_kinetic_theory'

    Parameters
    ----------
    walk : Results
        The walk object returned from `rw` or `rw_parallel` with required attributes:
        path : ndarray
            The paths of the walkers as computed by the ``rw`` function
        stride : int
            The number of steps taken between recording walker positions in
            the `path' variable. Must be >= 1. Default is 1, in which case all
            steps are recorded. Values larger than 1 enable longer walks without
            running out of memory due to size of `path`. This must be the same
            as what was used to simulate the random walk.
        porosity : float
            The fraction of the image that is pore space. Can be computed with
            ps.metrics.porosity
        time_step : float
            The time elapsed per step of each walker in seconds [s]. Can be calculated as
            `step_size / v_rel` where `v_rel` is the average gas particle velocity,
            as calculated by `calc_kinetic_constants`.
        resolution : float
            The size of the voxels, in nm/vx, to convert the path lengths into
            physical units
    inplace : bool
        If True (default), the diffusivities get added to the walk object

    Returns
    -------
    Deff:
        A namedtuple with the following attributes:
        eDiff: Effective diffusivity in all directions
        eDiff_x: Effective diffusivity in X dimension
        eDiff_y: Effective diffusivity in Y dimension
        eDiff_z: Effective diffusivity in Z dimension
    """
    porosity = walk.porosity
    n_steps_nominal = walk.n_steps
    s = np.arange(1, n_steps_nominal/walk.stride) * walk.stride
    d, dx, dy, dz = rw_to_displacement(walk, inplace=False)

    d2 = d[1:, :] ** 2
    dx2 = dx[1:, :] ** 2
    dy2 = dy[1:, :] ** 2
    dz2 = dz[1:, :] ** 2

    D = porosity * np.mean(d2, axis=1) * 1e-18 / (2 * walk.im.ndim * s * walk.time_step)
    Dx = porosity * np.mean(dx2, axis=1) * 1e-18 / (2 * s * walk.time_step)
    Dy = porosity * np.mean(dy2, axis=1) * 1e-18 / (2 * s * walk.time_step)
    Dz = porosity * np.mean(dz2, axis=1) * 1e-18 / (2 * s * walk.time_step)
    if inplace:
        walk.eDiff = D
        walk.eDiff_x = Dx
        walk.eDiff_y = Dy
        walk.eDiff_z = Dz
    Diffusion = namedtuple('Diffusion', 'eDiff, eDiff_x eDiff_y eDiff_z')
    Deff = Diffusion(eDiff=D, eDiff_x=Dx, eDiff_y=Dy, eDiff_z=Dz)
    return Deff


def compute_tau(walk, inplace=True):
    """
    Computes the tortuosity obtained from the random walk simulation

    Uses a statistical expected displacement method, instead of kinetic theory
    Parameters
    ----------
    walk :
        Results object returned by 'rw' or 'rw_parallel'

    Returns
    -------
    tortuosity : Tortuosity
        a namedtuple with attributes:
        tau : image tortuosity
        ff : the formation factor (equal to tau/porosity)
        tau_x : tortuosity in x-dimension
        tau_y : tortuosity in y-dimension
        tau_z : tortuosity in z-dimension
    """
    s = np.arange(1, walk.n_steps / walk.stride) * walk.stride
    porosity = getattr(walk, 'porosity', ps.metrics.porosity(walk.im))
    mfp = walk.mfp
    step_size = walk.step_size
    res = walk.resolution
    dim = walk.im.ndim

    expected_msd = (mfp * step_size * res)**2 * (s / (mfp * step_size * res))
    expected_msd1D = expected_msd / dim

    if not hasattr(walk, 'displacement'):
        d, dx, dy, dz = rw_to_displacement(walk, inplace=False)
    else:
        d, dx, dy, dz = walk.displacement, walk.displacement_x, walk.displacement_y, walk.displacement_z

    d2 = (d ** 2).mean(axis=1)
    dx2 = (dx ** 2).mean(axis=1)
    dy2 = (dy ** 2).mean(axis=1)
    dz2 = (dz ** 2).mean(axis=1)

    tau = expected_msd/(d2[1:])
    tau_x = expected_msd1D/(dx2[1:])
    tau_y = expected_msd1D/(dy2[1:])
    tau_z = expected_msd1D/(dz2[1:])
    formation_factor = tau / porosity
    if inplace:
        walk.tau = tau
        walk.ff = formation_factor
        walk.tau_x = tau_x
        walk.tau_y = tau_y
        # if dim == 3:
        walk.tau_z = tau_z
    Tortuosity = namedtuple('Tortuosity', 'tau ff tau_x tau_y tau_z')
    tortuosity = Tortuosity(tau=tau, ff=formation_factor, tau_x=tau_x, tau_y=tau_y, tau_z=tau_z)
    return tortuosity


def calc_probed(walk):
    r"""
    Calculates the percentage of the void space that the random walk has probed.

    Parameters
    ----------
    walk : Results
        The walk object returned from 'rw' or 'rw_parallel' with required attributes:

        im : ndarray
            The original binary image used to simulate the random walk
        path : ndarray
            The path of the walkers as returned by the random walk function
        edges : str
            How the edges were dealt with during the random walk, either
            ``'periodic'`` or ``'symmetric'``.

    Returns
    -------
    percent_probed : float
        The percentage of void voxels that have been occupied by at least one walker
        at any point during the random walk
    probing_frequency : float
        The frequency of probing. This should be equal to n_walkers * n_steps / number of void space voxels

    """
    path = walk.path
    im = walk.im
    porosity = getattr(walk, 'porosity', ps.metrics.porosity(walk.im))
    im_blank = np.zeros(im.shape, dtype=int)
    for i in range(len(path[:, ...])):
        locs = np.around(path[i, ...], decimals=0).astype(int)

        locs = _wrap_indices(locs.T, im.shape, mode=walk.edges)
        im_blank[locs] += 1

    probed = np.count_nonzero(im_blank) / im_blank.size  # number of voxels through which the path has passed
    percent_probed = probed / porosity
    probing_frequency = np.sum(im_blank) / (im_blank.size * porosity)
    print('_'*60)
    print(f'{percent_probed*100:0.4}% of the pore space was probed.')
    print(f'On average, each pore voxel was visited {probing_frequency:0.4} times.')
    print('_'*60)

    return percent_probed, probing_frequency

# plotting functions:


def plot_diffusion(walk, ax=None):
    """
    Plots tortuosity as a function of step #

    This function will recompute diffusion with 'effective_diffusivity' if the 'eDiff'
    attribute has not already been assigned to the walk object.

    Parameters
    ----------
    walk:
        The 'walk', returned by the 'rw' or 'rw_parallel' functions. In this function,
        the required attributes are:

    ax : Axes
        A matplotlib Axes handle for the axes onto which to place the plot

    Returns
    -------
    ax : Axes
        The matplotlib Axes handle with the diffusion plot drawn on it.
    """

    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    s = np.arange(0, walk.path.shape[0]) * walk.time_step * walk.stride
    if not hasattr(walk, 'eDiff'):
        print('Running diffusion calculation...')
        diff, diffx, diffy, diffz = effective_diffusivity(walk, inplace=False)
    else:
        diff, diffx, diffy, diffz = walk.eDiff, walk.eDiff_x, walk.eDiff_y, walk.eDiff_z
    ax.plot(s[1:], diff, '.', label=f'Diffusion: {diff[-1]:0.4} m\u00b2/s')
    ax.plot(s[1:], diffx, '.', label=f'Diffusion in X: {diffx[-1]:0.4} m\u00b2/s')
    ax.plot(s[1:], diffy, '.', label=f'Diffusion in Y: {diffy[-1]:0.4} m\u00b2/s')
    if walk.im.ndim == 3:
        ax.plot(s[1:], diffz, '.', label=f'Diffusion in Z: {diffz[-1]:0.4} m\u00b2/s')
    if walk.porosity > 0.9:
        ax.plot([0, s[-1]], [walk.Db, walk.Db], '-', label=f'Bulk diffusion: {walk.Db:0.4} m\u00b2/s')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Diffusion Coefficient [m\u00b2/s]')
    ax.set_title(f'{walk.im.ndim}D Diffusion Random Walk Simulation')
    ax.legend()
    return ax


def plot_tau(walk, ax=None):
    """
    Plots tortuosity as a function of step #

    This function will recompute tau with 'compute_tau' if the 'tau' attribute has not already
    been assigned to the walk object.

    Parameters
    ----------
    walk:
        The 'walk', returned by the 'rw' or 'rw_parallel' functions. In this function,
        the required attributes are:

            n_steps: int
                The number of steps taken in the simulation
            stride: int
                The number of steps taken between saving walker location
            tau
            ff
            taux
            tauy
            [tauz]

    ax : Axes
        A matplotlib Axes handle for the axes onto which to place the plot

    Returns
    -------
    ax : Axes
        The matplotlib Axes handle with the tortuosity plot drawn on it.

    """
    s = np.arange(1, walk.n_steps / walk.stride) * walk.stride
    if not hasattr(walk, 'tau'):
        tau, ff, taux, tauy, tauz = compute_tau(walk, inplace=False)
    else:
        tau, ff, taux, tauy, tauz = walk.tau, walk.ff, walk.tau_x, walk.tau_y, walk.tau_z
    if not ax:
        fig,ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(s, tau, label=f'Final tau: {tau[-1]:0.3f}')
    ax.plot(s, taux, label=f'Final tau in X: {taux[-1]:0.3f}')
    ax.plot(s, tauy, label=f'Final tau in Y: {tauy[-1]:0.3f}')
    if walk.im.ndim == 3:
        ax.plot(s, tauz, label=f'Final tau in Z: {tauz[-1]:0.3f}')

    ax.set_ylabel('Tortuosity')
    ax.set_xlabel('Number of Steps')
    ax.set_ylim([0, 2 * tau[-1]])
    ax.set_title(f'{walk.im.ndim}D '
                 f'{"Knudsen" if walk.knudsen else "Regular"} Tortuosity From '
                 f'{"Pearsonian" if walk.mode=="random" else "Axial"} Random Walk Simulation')
    ax.legend()
    return ax


def msd_vs_nsteps(walk, ax=None):
    r"""
    Plot mean-squared displacement in voxels vs number of steps

    Parameters
    ----------
    walk : Results
        The 'walk', returned by the 'rw' or 'rw_parallel' functions. In this function,
        the required attributes are:

        path : ndarray
            The paths of the walkers as computed by the ``rw`` function.
        path_length : scalar
            The distance a walker travelled before choosing another direction.
        stride : int
            The spacing to use between points when computing displacement. Higher
            values result in smoother data but may lose features.
    ax: [optional]
        a handle to a matplotlib axes object

    Returns
    -------
    ax : matplotlib axis handle
        The handle to the plotted data

    """
    stride = walk.stride
    n_steps = walk.n_steps
    step_size = walk.step_size

    if not hasattr(walk, 'displacement'):
        d, dx, dy, dz = rw_to_displacement(walk, inplace=False)
    else:
        d, dx, dy, dz = walk.displacement, walk.displacement_x, walk.displacement_y, walk.displacement_z

    d2 = (d ** 2).mean(axis=1)
    dx2 = (dx ** 2).mean(axis=1) * walk.im.ndim
    dy2 = (dy ** 2).mean(axis=1) * walk.im.ndim
    dz2 = (dz ** 2).mean(axis=1) * walk.im.ndim

    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(10,6))
    s = np.arange(0, n_steps, stride)

    lines = [(s, d2, f'MSD -> Tau: {walk.tau[-1]:0.3f}', 'r'),
             (s, dx2, f'ASD0: -> Tau: {walk.tau_x[-1]:0.3f}', 'g'),
             (s, dy2, f'ASD1: -> Tau: {walk.tau_y[-1]:0.3f}', 'b'),
             (s, dz2, f'ASD2: -> Tau: {walk.tau_z[-1]:0.3f}', 'k')]
    for x, y, label, c in lines[:walk.im.ndim+1]:
        ax.plot(x, y, c, label=label)
        plot_regress((x, y, label, c), ax=ax)
    # plot_regress((s, d2, 'msd'), (s, dx2, 'asd0'), (s, dy2, 'asd1'), (s, dz2, 'asd2'), ax=ax)
    ax.set_xlabel('Number of Steps [#]')
    ax.set_ylabel('Mean Squared Displacement [nm\u00b2]')
    ax.legend()
    return ax


def plot_regress(*args, ax):
    """
    Calculates the linear regression of two 1D arrays, and plots onto the provided plot
    Parameters
    ----------
    args : tuple
        A list of tuples, each of the form: (x, y, label, colour)
    ax : The matplotlib handle to an Axes object onto which to plot the regression results

    Returns
    -------

    """
    for x,y,l,c in args:
        res = linregress(x, y)
        ax.plot(x, res.intercept + res.slope*x, '--', color=c, label=f'Regression: R\u00B2 -> {res.rvalue**2:0.4f}')


def rw_to_image(walk, tiled=True, mode='count', walkers=None, shrink=True, threshold=0.005):
    r"""
    Adds the paths taken by random walkers to the supplied image

    Parameters
    ----------
    shrink : bool
        If True, the image is unpadded to a point such that it zooms into the space
        surrounding the walk.
    walkers : list
        a 1D array or list of the indices of the walkers to plotted. The default
        is None, in which case all walkers are plotted
    walk : Results
        The 'walk', returned by the 'rw' or 'rw_parallel' functions. In this function,
        the required attributes are:

        path: ndarray
            The paths of the walkers as computed by the ``rw`` function.
        im : ndarray
            The image into which the walker paths should be inserted.
        edges : str
            How the edges were dealt with during the random walk, either
            ``'periodic'`` or ``'symmetric'``.
    tiled : bool
        If ``True`` (default) the image is tiled as many times as needed to
        accomodate the distance taken by the walkers.  If ``False`` the
        walkers' locations are wrapped to remain within the image. The tiling
        is done either periodically or symmetrically depending on ``edges``.
    mode : str
        How to plot the walkers. Option are:

        =========== ==========================================================
        Option      Description
        =========== ==========================================================
        'count'     Each voxel will contain the number of times it was visited

        'step'      Each voxel will contain the step number at which is was
                    last visited
        =========== ==========================================================

    Returns
    -------
    ims : tuple
        A tuple containing the walker paths and the void image. If ``tiled``
        was set to ``True`` then the returned images may be larger than
        ``im`` depending how far the walkers reached in each dimension.

    Notes
    -----
    It is possible to plot the paths for only selected walkers using
    ``rw_to_image(path=path[:, [1, 3, 5], :]`` for example.

    """
    path = np.copy(walk.path)
    im = walk.im
    edges = walk.edges

    if tiled:  # Convert image to appropriate size
        N_min = []
        N_max = []
        for i in range(im.ndim):
            imin = np.amin(path[..., i])
            N_min.append(int((np.abs(imin // im.shape[i])) * im.shape[i]))
            imax = np.amax(path[..., i])
            N_max.append(int(np.ceil(imax / im.shape[i]) * im.shape[i]))
        N_min = np.max(N_min) * np.ones(len(N_min)).astype(int)
        N_max = np.max(N_max) * np.ones(len(N_max)).astype(int)
        max_shape = max(max(N_min), max(N_max))
        path[..., :] = path[..., :] + max_shape
        padmode = 'wrap' if edges == 'periodic' else 'symmetric'
        # padding = tuple([(N_min[i], N_max[i]) for i in range(im.ndim)])
        padding = tuple([(max_shape, max_shape) for i in range(im.ndim)])
        im = np.pad(im, padding, mode=padmode).astype(bool)
    im_blank = np.zeros_like(im, dtype=int)
    if not walkers:
        walkers = np.arange(0, path.shape[1])
    for i in range(len(path[:, ...])):
        locs = np.around(path[i, walkers, :], decimals=0).astype(int)
        if not tiled:
            locs = _wrap_indices(locs.T, im.shape, mode=edges)
        else:
            locs = tuple(locs.T)
        if mode == 'step':
            im_blank[locs] = i + 1
        elif mode == 'count':
            im_blank[locs] += 1
    if threshold:
        im_blank, pad_excess = remove_blank_space(im_blank, threshold=threshold)
        im = remove_blank_space(im, pad_excess)[0]
    return im_blank, im


def imsave3D(walk, filename, tiled=True, mode='count', walkers='all', threshold=0.005):
    im_path, im_voids = rw_to_image(walk, mode=mode, tiled=tiled, walkers=walkers, threshold=threshold)
    im_path = im_path + (~im_voids.astype(bool)) * im_path.max() * 0.1
    downsample = True if sys.getsizeof(im_path) > 5e8 else False
    ps.io.to_vtk(im_path, filename=filename, downsample=downsample)


def imshow_rw(walk, ax=None, tiled=True, mode='count', walkers=None, threshold=0.005):
    """
    This is a plotting wrapper for 'rw_to_image'

    Plots the image of the walkers overlaid on the image of pore space.
    Only implemented for 2D images. 3D images should use the 'imsave3D' function
    which exports the walk to a vtk file which can be opened in Paraview.

    Parameters
    ----------
    walk : Results
        The 'walk', returned by the 'rw' or 'rw_parallel' functions. In this function,
        the required attributes are:

        path: ndarray
            The paths of the walkers as computed by the ``rw`` function.
        im : ndarray
            The image into which the walker paths should be inserted.
        edges : str
            How the edges were dealt with during the random walk, either
            ``'periodic'`` or ``'symmetric'``.
    ax : A handle to the matplotlib axes for the image to be shown on. If left
        blank, a new ax object is generated.
    tiled : bool
        If ``True`` (default) the image is tiled as many times as needed to
        accommodate the distance taken by the walkers.  If ``False`` the
        walkers' locations are wrapped to remain within the image. The tiling
        is done either periodically or symmetrically depending on ``edges``.
    mode : str
        How to plot the walkers. Option are:

        =========== ==========================================================
        Option      Description
        =========== ==========================================================
        'count'     Each voxel will contain the number of times it was visited

        'step'      Each voxel will contain the step number at which is was
                    last visited
        =========== ==========================================================

    walkers : list
        a 1D array or list of the indices of the walkers to plotted. The default
        is None, in which case all walkers are plotted

    Returns
    -------
    ax :
        a handle to the matplotlib axes on which the image was plotted.
    """
    if walk.im.ndim == 3:
        print('Not implemented for 3D images')
        return None
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    im_path, im_voids = rw_to_image(walk, mode=mode,  tiled=tiled, walkers=walkers, threshold=threshold)
    im_path = im_path + (~im_voids.astype(bool)) * im_path.max() * 0.1
    ax.imshow(im_path.squeeze(),
               origin='lower',
               cmap=plt.cm.magma,
               interpolation='none')
    ax.patch.set_facecolor('black')
    ax.set_title(f'{walk.im.ndim}D '
                 f'{"Pearsonian" if walk.mode=="random" else "Axial"} Random Walk Simulation Visualization')
    plt.axis('off')
    return ax


def remove_blank_space(im, pad_width=None, threshold=0.005):
    """
    Somewhat crude functions which removes blank or almost blank space
    from an image (2d or 3d).

    Only works symmetrically - once it encounters part of an image which passes the
    threshold for having walkers in it, it stops unpadding all edges.

    Parameters
    ----------
    im : image from which to remove blank space
    pad_width :(optional) int
        The amount of voxels to strip from the edges of the image. If this is provided,
        `threshold` will be ignored and `pad_width` will be removed from all faces of the image.
    threshold : float
        A measure of how much to strip away from the image. 0< threshold < 1:
        A value of 0 will only strip completely blank space. A value of 1 will strip everything.

    Returns
    -------
    The image, with white 0 values stripped from edges
    pad_width : The number of voxels stripped from each edge
    """
    if pad_width:
        im = ps.tools.unpad(im, pad_width)
        pad_width_total = pad_width
    else:
        high = im.max()
        t = 50
        mask = ps.tools.get_border(im.shape, mode='faces', thickness=t)
        pad_width_total = 0
        while True:
            if im[mask].sum() / (2 * np.sum(im.shape)) <= threshold * high:
                im = ps.tools.unpad(im, t)
                pad_width_total += t
            elif t > 1:
                t -= 1
            else:
                break
            mask = ps.tools.get_border(im.shape, mode='faces', thickness=t)
    return im, pad_width_total
