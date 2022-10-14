from rw_simulation import *
from rw_post import *
from dask.distributed import Client
import matplotlib.pyplot as plt


def tortuosity_rw(im,
                  resolution,
                  n_walkers=2000,
                  n_steps=5000,
                  steps_per_path=50,
                  knudsen=True,
                  start=None,
                  edges='symmetric',
                  mode='random',
                  stride=5,
                  cores=None,
                  chunks=None,
                  same_start=False,
                  P=101325,
                  T=298,
                  MW=0.032,
                  mu=1.9e-5,
                  seed=None):
    r""" Perform a random walk on an image to get tortuosity

    This is a convenience function which completes the whole set-up,
    simulation and post-processing steps in one go. If more customization is
    desired, the functions are available separately as well.

    Parameters
    ----------
    im : ndarray
        The image of void space in which the walk should occur, with
        ``True`` values indicating the voids.
    resolution : float
        The resolution of the image in units of [nm]
    n_walkers : int
        Number of walkers to use.  A higher number gives less noisy data
        but takes longer to run.
    n_steps : int
        Number of steps to take before exiting the walk.  A high number
        is needed if the void space is highly tortuous so the walkers can
        probe all the space, but this takes longer to run.
    steps_per_path: int
        The number of steps to split each path-length into. More steps
        increases accuracy, but also reduces the time interval over which
        the simulation is completed for a given number of steps.
    knudsen : bool = True
        Determines whether the simulation will consider knudsen diffusion,
        or if it should do a regular random walk
    stride : int
        The number of steps to take between recording walker positions in
        the `path' variable. Must be >= 1. Default is 1, in which case all
        steps are recorded. Values larger than 1 enable longer walks
        without running out of memory due to size of `path`.
    cores : int
        The number of cores to use. Defaults to all cores.
    chunks : int
        The number of chunks to split the walkers into for parallel
        processing
    start : ndimage
        A boolean image with ``True`` values indicating where walkers
        should start. If not provided then start points will be selected
        from the void space at random. In some cases it is necessary or
        interesting to start all walkers at a common point or at a boundary.
    same_start : bool
        Default is ``False``. If ``True``, all walkers will start from the same
        randomly generated location within the image. If `start` is also
        provided, this will select one point from the subset of locations
        in `start`
    seed : int
        A seed value for a random number generator, which ensures
        repeatable results. Default is ``None``, in which case each simulation
        will be different.
    edges : string
        How walkers should behave when walking past an image boundary.
        Options are:

        ============= =========================================================
        Option        Description
        ============= =========================================================
        'periodic'    When a walker exits a edge of an image, it continues
                      as if the image were tiled so it sees the properties
                      of the opposite side of the image.

        'symmetric'   When a walker exits the edge of an image, it
                      continues as if the image were flipped so it sees
                      the properties similar to the edge of just exited.
        ============= =========================================================

    mode : string
        How walkers choose their directions.  Options are:

        ============= =========================================================
        Option        Description
        ============= =========================================================
        'random'      Walkers choose a random angle and walk in that
                      direction until the high a wall or reach their
                      ``path_length``.

        'axial'       Walkers follow the cardinal axes of the image.
        ============= =========================================================

    P : float
        The gas pressure in units of [Pa]
    T : float
        The gas temperature in units of [K]
    MW : float
        The molecular weight of the gas molecule in units of [kg/mol]
    mu : float
        The dynamic viscosity of the gas in units of [Pa.s]

    Returns
    -------
    walk : Results
        Returns a Results object containing the following named attributes:

        path : ndarray
            A numpy ndarray of size ``[n_steps, n_walkers, im.ndim]``. The
            [x, y, [z]] location of the walker at each step is recorded in
            the final column.
        tau : ndarray
            A 1D numpy array of the overall tortuosity at each step
        ff : ndarray
            A 1D numpy array of the formation factor tortuosity at each
            step (equal to tau/porosity)
        tau_x : ndarray
            A 1D numpy array of the tortuosity in x-direction at each step
        tau_y : ndarray
            A 1D numpy array of the tortuosity in y-direction at each step
        tau_z :
            A 1D numpy array of the tortuosity in z-direction at each step
        eDiff:
            A 1D numpy array of the effective diffusivity in all directions
            at each step
        eDiff_x:
            A 1D numpy array of the effective diffusivity in x-direction
            at each step
        eDiff_y:
             A 1D numpy array of the effective diffusivity in y-direction
             at each step
        eDiff_z:
             A 1D numpy array of the effective diffusivity in z-direction
             at each step
        displacement: ndarray
            A 2D array of the magnitude of each walker displacement at each
            step
        displacement_x : ndarray
            A 2D array of the displacements of each walker at each step
            in x-direction
        displacement_y : ndarray
            A 2D array of the displacements of each walker at each step
            in y-direction
        displacement_z : ndarray
            A 2D array of the displacements of each walker at each step
            in z-direction
        step_size : float
            The distance a walker should take on each step in units of
            [voxels]. This should not be more than 1 since this would
            makes steps longer than a voxel, and should also be less than
            ``path_length``.
        time_step : float
            the time it takes to complete each step [s]
        n_mfps : float
            the number of mean free paths that each walker will undergo
            during the simulation.
        mfp : float
            Mean free path of the gas in units of [nm]
        path_length : float
            The distance a walker should travel in [voxels] before
            randomly choosing another direction.  If a walker hits a wall
            before this distance is reached it also changes direction.
        v_rel : float
            The kinetic velocity of the gas in units of [nm/s] relative to
            other moving gas particles, calculated using kinetic gas
            theory.
        Db : float
            The theoretical bulk phase diffusion coefficient in units of
            [m\u00b2/s], calculated using kinetic gas theory.

    """

    kin = calc_kinetic_theory(P, T, MW, mu)
    walk = compute_steps(kinetics=kin, n_steps=n_steps, ndim=im.ndim, resolution=resolution,
                         steps_per_path=steps_per_path, knudsen=knudsen)
    walk = rw_parallel(
        im,
        walk,
        seed=seed,
        n_walkers=n_walkers,
        edges=edges,
        mode=mode,
        stride=stride,
        start=start,
        chunks=chunks,
        cores=cores,
        same_start=same_start
    )

    rw_to_displacement(walk)
    compute_tau(walk)
    effective_diffusivity(walk)
    calc_probed(walk)

    fig, ax = plt.subplots(2,2, figsize=(16,12))
    ax = ax.ravel()
    if walk.porosity > 0.90:
        s = np.arange(0, n_steps / stride) * walk.time_step * stride
        ax[0].plot([0, s[-1]],
                   [walk.Db, walk.Db], '-',
                   label='Theoretical Bulk Diffusion in Free Space')
    plot_diffusion(walk, ax=ax[0])
    plot_tau(walk, ax=ax[1])
    msd_vs_nsteps(walk, ax=ax[2])
    if im.ndim == 2:
        imshow_rw(walk, ax=ax[3])
    plt.show()
    return walk


if __name__ == '__main__':
    import numpy as np
    client = Client(n_workers=20)
    # im = ps.generators.blobs([100, 100])
    im = np.ones([100, 100])
    n_walkers = 5000
    n_steps = 5000

    stride = 20
    seed = 2357
    B = tortuosity_rw(
        im,
        resolution=5,
        n_walkers=n_walkers,
        chunks=20,
        n_steps=n_steps,
        seed=seed,
        stride=stride,
        mode='axial'
    )

    # im = np.ones([100,100]).astype(bool)
    # im = ps.filters.fill_blind_pores(im, surface=True)
    # resolution = 20
    # tortuosity_rw(im, resolution)
    client.shutdown()
