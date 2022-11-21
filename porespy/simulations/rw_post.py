import sys
import numpy as np
from porespy.tools import Results
import matplotlib.pyplot as plt
from rw_simulation import _wrap_indices
from scipy.stats import linregress


def steps_to_displacements(paths, voxel_size, **kwargs):
    r"""
    Compute the displacement of each walker from their starting point in [nm]

    Parameters
    ----------
    paths : ndarray
        The paths followed by each walker, as returned from the ``rw`` function.
    voxel_size : float
        The size of each voxel in units of [nm/voxel]

    Returns
    -------
    displacements : dataclass
        A tuple-like with four computed results stored as attributes:

        ========== ==========================================================
        attribute  description
        ========== ==========================================================
        total      The total displacement of the walker
        x          The x-component of the displacement
        y          The y-component of the displacement
        z          The z-component of the displacement
        step_num   The step number corresponding to the given displacement
        ========== ==========================================================

    Notes
    -----
    The returned result contains the progressive displacement at each step,
    not just the final value. This is useful for plotting the mean-squared
    displacement as a function of time.

    """
    # Get the total displacements in each direction and overall
    dx = paths[:, :, 0] - paths[0, :, 0]  # all x's subtract initial x position
    dy = paths[:, :, 1] - paths[0, :, 1]
    if paths.shape[-1] == 4:
        dz = paths[:, :, 2] - paths[0, :, 2]
    else:
        dz = np.zeros(np.shape(dx))
    d = (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5
    step_num = paths[:, 0, -1]
    # Scale the displacements from voxels to nm
    dt = Results()
    dt.total = d * voxel_size
    dt.x = dx * voxel_size
    dt.y = dy * voxel_size
    dt.z = dz * voxel_size
    dt.step_num = step_num
    return dt


def effective_diffusivity(displacements, im, time_step, **kwargs):
    r"""
    Calculates the effective diffusivity based on physical displacements of random
    walkers

    Parameters
    ----------
    displacements : ndarray
        The displacements the walkers
    im : ndarray
        The image on which the random walk was conducted.
    time_step : float
        The time elapsed per step of each walker in seconds [s]. Can be calculated as
        `step_size / v_rel` where `v_rel` is the average gas particle velocity,
        as calculated by `calc_kinetic_constants`.

    Returns
    -------
    Deff : Results object
        A dataclass with the following attributes:
        Deff : Effective diffusivity in all directions
        Deff_x : Effective diffusivity in X dimension
        Deff_y : Effective diffusivity in Y dimension
        Deff_z : Effective diffusivity in Z dimension

    """
    d, dx, dy, dz, n = displacements
    d2 = d[1:, :] ** 2
    dx2 = dx[1:, :] ** 2
    dy2 = dy[1:, :] ** 2
    dz2 = dz[1:, :] ** 2

    n_write = np.diff(n)[0]
    n_steps = np.amax(n) + n_write
    s = np.arange(1, n_steps/n_write) * n_write
    dim = 3 if dz.any() else 2

    porosity = im.sum()/im.size
    D = porosity * np.mean(d2, axis=1) * 1e-18 / (2 * dim * s * time_step)
    Dx = porosity * np.mean(dx2, axis=1) * 1e-18 / (2 * s * time_step)
    Dy = porosity * np.mean(dy2, axis=1) * 1e-18 / (2 * s * time_step)
    Dz = porosity * np.mean(dz2, axis=1) * 1e-18 / (2 * s * time_step)

    Deff = Results()
    Deff.Deff = D
    Deff.Deff_x = Dx
    Deff.Deff_y = Dy
    Deff.Deff_z = Dz
    Deff.step_num = n
    return Deff


def tortuosity_rw(displacements, mfp, step_size, voxel_size, **kwargs):
    """
    Computes the tortuosity obtained from the random walk simulation

    Uses a statistical expected displacement method, instead of kinetic theory
    Parameters
    ----------
    displacements : ndarray
        The displacements of walkers
    mfp : float
        The mean free path of the walkers in units of [nm]
    step_size : float
        The size of the step taken by each walker on each step, in units of [voxels]
    voxel_size : float
        The resolution of the image in units of [nm/voxel]

    Returns
    -------
    tortuosity : dataclass
        A dataclass-like object with the following attributes:

        ========== ==========================================================
        attribute  description
        ========== ==========================================================
        tau        image tortuosity
        tau_x      tortuosity in x-dimension
        tau_y      tortuosity in y-dimension
        tau_z      tortuosity in z-dimension
        ========== ==========================================================
    """
    d, dx, dy, dz, n = displacements
    d2 = (d ** 2).mean(axis=1)
    dx2 = (dx ** 2).mean(axis=1)
    dy2 = (dy ** 2).mean(axis=1)
    dz2 = (dz ** 2).mean(axis=1)

    n_write = np.diff(n)[0]
    n_steps = np.amax(n) + n_write
    s = np.arange(1, n_steps/n_write) * n_write
    res = voxel_size
    dim = 3 if dz.any() else 2

    expected_msd = (mfp * step_size * res)**2 * (s / (mfp * step_size * res))
    expected_msd1D = expected_msd / dim
    t = Results()
    t.tau = expected_msd/(d2[1:])
    t.tau_x = expected_msd1D/(dx2[1:])
    t.tau_y = expected_msd1D/(dy2[1:])
    t.tau_z = expected_msd1D/(dz2[1:])
    t.step_num = n
    return t


def calc_probed(im, paths, mode='symmetric'):
    r"""
    Calculates the percentage of the void space that the random walk has probed.

    Parameters
    ----------
    im : ndarray
        The original binary image used to simulate the random walk
    paths : ndarray
        The path of the walkers as returned by the random walk function
    mode : str
        How the edges were dealt with during the random walk, either
        ``'periodic'`` or ``'symmetric'``.

    Returns
    -------
    percent_probed : float
        The percentage of void voxels that have been occupied by at least one
        walker at any point during the random walk
    probing_frequency : float
        The frequency of probing. This should be equal to
        ``n_walkers * n_steps / im.sum()``

    """
    porosity = im.sum()/im.size
    im_blank = np.zeros(im.shape, dtype=int)
    for i in range(len(paths[:, ...])):
        locs = np.around(paths[i, :, :-1], decimals=0).astype(int)
        locs = _wrap_indices(locs.T, im.shape, mode=mode)
        im_blank[locs] += 1
    # number of voxels through which the path has passed
    probed = np.count_nonzero(im_blank) / im_blank.size
    percent_probed = probed / porosity
    probing_frequency = np.sum(im_blank) / (im_blank.size * porosity)
    f = Results()
    f.percent_probed = percent_probed
    f.probing_frequency = probing_frequency
    return f

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
        ax.plot([0, s[-1]], [walk.Db, walk.Db], '-',
                label=f'Bulk diffusion: {walk.Db:0.4} m\u00b2/s')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Diffusion Coefficient [m\u00b2/s]')
    ax.set_title(f'{walk.im.ndim}D Diffusion Random Walk Simulation')
    ax.legend()
    return ax


def plot_tau(taus, ax=None):
    """
    Plots tortuosity as a function of step #

    Parameters
    ----------
    taus : ndarray
        The tortuosity of each walker as computed by the ``tortuosity_rw`` function
    ax : Axes
        A matplotlib Axes handle for the axes onto which to place the plot

    Returns
    -------
    ax : Axes
        The matplotlib Axes handle with the tortuosity plot drawn on it.

    """
    tau, taux, tauy, tauz, step_num = taus
    n_write = np.diff(step_num)[0]
    n_steps = np.amax(step_num) + n_write
    dim = 2 if np.all(np.isinf(tauz)) else 3
    s = np.arange(1, n_steps/n_write) * n_write

    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(s, tau, label=f'Final tau: {tau[-1]:0.3f}')
    ax.plot(s, taux, label=f'Final tau in X: {taux[-1]:0.3f}')
    ax.plot(s, tauy, label=f'Final tau in Y: {tauy[-1]:0.3f}')
    if dim == 3:
        ax.plot(s, tauz, label=f'Final tau in Z: {tauz[-1]:0.3f}')

    ax.set_ylabel('Tortuosity')
    ax.set_xlabel('Number of Steps')
    ax.set_ylim([0, 2 * tau[-1]])
    ax.legend()
    return ax


def msd_vs_nsteps(displacements, ax=None):
    r"""
    Plot mean-squared displacement in voxels vs number of steps

    Parameters
    ----------
    displacements : ndarray
        The displacments of the walkers in units of [nm]
    path_length : scalar
        The distance a walker travelled before choosing another direction.
    ax: [optional]
        a handle to a matplotlib axes object

    Returns
    -------
    ax : matplotlib axis handle
        The handle to the plotted data

    """
    d, dx, dy, dz, step_num = displacements
    d2 = (d ** 2).mean(axis=1)
    dx2 = (dx ** 2).mean(axis=1)
    dy2 = (dy ** 2).mean(axis=1)
    dz2 = (dz ** 2).mean(axis=1)

    n_write = np.diff(step_num)[0]
    n_steps = np.amax(step_num) + n_write
    dim = 3 if dz.any() else 2
    # s = np.arange(1, n_steps/n_write) * n_write
    s = np.arange(0, n_steps, n_write)

    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    lines = [(s, d2, 'Total', 'r'),
             (s, dx2, 'X', 'g'),
             (s, dy2, 'Y', 'b'),
             (s, dz2, 'Z', 'k')]
    for x, y, label, c in lines[:dim+1]:
        ax.plot(x, y, c, label=label)
        plot_regress((x, y, label, c), ax=ax)
    ax.set_xlabel('Number of Steps [#]')
    ax.set_ylabel('Mean Squared Displacement [nm\u00b2]')
    ax.legend()
    return ax


def plot_regress(*args, ax):
    """
    Calculates the linear regression of two 1D arrays, and plots onto the provided
    plot

    Parameters
    ----------
    args : tuple
        A list of tuples, each of the form: (x, y, label, colour)
    ax : The matplotlib handle to an Axes object onto which to plot the regression results

    Returns
    -------

    """
    for x, y, l, c in args:
        res = linregress(x, y)
        ax.plot(x, res.intercept + res.slope*x, '--',
                color=c,
                label=f'Regression: R\u00B2 -> {res.rvalue**2:0.4f}')


def rw_to_image(paths, im, edges='symmetric', tiled=True, color_by='count'):
    r"""
    Adds the paths taken by random walkers to the supplied image

    Parameters
    ----------
    paths : ndarray
        The paths of the walkers as computed by the ``rw`` function. To plot a
        limited number of walkers just index into this array like
        ``paths[:, [1, 4, 8], :]``
    im : ndarray
        The image into which the walker paths should be inserted
    edges : str
        How the edges were dealt with during the random walk, either
        ``'periodic'`` or ``'symmetric'``.
    tiled : bool
        If ``True`` (default) the image is tiled as many times as needed to
        accomodate the distance taken by the walkers.  If ``False`` the
        walkers' locations are wrapped to remain within the image. The tiling
        is done either periodically or symmetrically depending on ``mode``.
    color_by : str
        How to plot the walkers. Option are:

        =========== ==========================================================
        Option      Description
        =========== ==========================================================
        'count'     Each voxel will contain the number of times it was visited
        'step'      Each voxel will contain the step number at which is was
                    last visited
        'visited'   A 1 is placed in any voxel which was visited
        =========== ==========================================================

        In all cases the non-active phase is label with -1.

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
    paths = paths.copy()
    im2 = im.copy()
    if tiled:  # Convert image to appropriate size
        N_min = []
        N_max = []
        for i in range(im2.ndim):
            imin = np.amin(paths[..., i])
            N_min.append(int((np.abs(imin // im2.shape[i])) * im2.shape[i]))
            imax = np.amax(paths[..., i])
            N_max.append(int(np.ceil(imax / im2.shape[i]) * im2.shape[i]))
        N_min = np.max(N_min) * np.ones(len(N_min)).astype(int)
        N_max = np.max(N_max) * np.ones(len(N_max)).astype(int) - N_min
        max_shape = max(max(N_min), max(N_max))
        paths[..., :-1] = paths[..., :-1] + max_shape
        padmode = 'wrap' if edges == 'periodic' else 'symmetric'
        padding = tuple([(max_shape, max_shape) for i in range(im2.ndim)])
        im2 = np.pad(im2, padding, mode=padmode).astype(bool)
    im_blank = np.zeros_like(im2, dtype=float)
    im_blank[~im2] = -1
    for i in range(paths.shape[0]):
        locs = np.around(paths[i, :, :-1], decimals=0).astype(int)
        if not tiled:
            locs = _wrap_indices(locs.T, im2.shape, mode=edges)
        else:
            locs = tuple(locs.T)
        if color_by == 'step':
            im_blank[locs] = paths[i, :, -1][0]
        elif color_by == 'count':
            im_blank[locs] += 1
        elif color_by == 'visited':
            im_blank[locs] = 1
    return im_blank
