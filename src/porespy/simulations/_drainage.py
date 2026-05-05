import inspect
from typing import Literal

import numpy as np
import numpy.typing as npt

from porespy.filters import (
    fftmorphology,
    find_trapped_clusters,
    seq_to_satn,
    trim_disconnected_voxels,
    find_disconnected_voxels,
    dilate,
)
from porespy.metrics import pc_map_to_pc_curve
from porespy.tools import (
    Results,
    _insert_disk_at_points,
    _insert_disk_at_points_parallel,
    _insert_disks_at_points_parallel,
    get_edt,
    get_strel,
    get_tqdm,
    make_contiguous,
    parse_steps,
    settings,
    ps_round,
)

__all__ = [
    "drainage",
    # The following are reference implementations using different techniques
    "drainage_dt",
    "drainage_conv",
    "drainage_dt_conv",
    "drainage_bf",
]


edt = get_edt()
tqdm = get_tqdm()
strel = get_strel()


def drainage_bf(
    im,
    inlets=None,
    outlets=None,
    dt=None,
    steps=None,
    smooth=False,
):
    r"""
    Performs a distance transform based drainage simulation using distance transform
    thresholding for the erosion step and brute-force sphere insertion for the
    dilation step.

    Parameters
    ----------
    im : ndarray
        The boolean image of the void space on which to perform the simulation
    inlets : ndarray (optional)
        A boolean array with `True` values indicating the inlet locations for the
        invading (non-wetting) fluid. If not provided then access limitations will
        not be applied, meaning that the invading fluid can appear anywhere within
        the domain.
    outlets : ndarray (optional)
        A boolean array with `True` values indicating the outlet locations through
        which defending (wetting) phase would exit the domain. If not provided then
        trapping of the wetting phase is ignored.
    dt : ndarray, optional
        The distance transform of the void space. This is optional, but providing
        it if it is already available saved some time.
    steps : scalar or array_like
        Controls which sphere sizes to invade. If an `int` then this many steps
        between 1 and the maximum size are used. A `tuple` is treated as the start
        and stop of the integer values. A `list` or `ndarray` is used directly. If
        `None` (default) then each unique value in the distance transform is used.

    Returns
    -------
    results : Dataclass-like object
        An object with the following attributes:

        =========== ================================================================
        Attribute   Description
        =========== ================================================================
        `im_seq`    An ndarray with each voxel indicating the step number at which
                    it was first invaded. -1 indicates uninavded, either due to
                    the applied `steps` not spanning the full range of sizes in the
                    image, or due to trapping, while 0 indicates residual invading
                    phase.
        `im_size`   A numpy array with each voxel containing the radius of the
                    sphere, in voxels, that first overlapped it. `inf` indicates
                    uninavded, either due to the applied `steps` not spanning the
                    full range of sizes in the image, or due to trapping, while 0
                    indicates residual invading phase.
        =========== ================================================================

    Notes
    -----
    The sphere insertion steps will be executed in parallel if
    `porespy.settings.ncores > 1`
    """
    if settings.ncores > 1:
        func = _insert_disk_at_points_parallel
    else:
        func = _insert_disk_at_points
    im = np.array(im, dtype=bool)
    if dt is None:
        dt = edt(im)
    dt = dt.astype(int)
    bins = parse_steps(steps=steps, vals=dt[im], descending=True)
    im_seq = -np.ones_like(im, dtype=int)
    im_size = np.zeros_like(im, dtype=float)
    nwp = np.zeros_like(im, dtype=bool)
    seeds_prev = np.zeros_like(im)
    desc = inspect.currentframe().f_code.co_name  # Get current func name
    for i, r in enumerate(tqdm(bins, desc=desc, **settings.tqdm)):
        seeds = dt >= r
        if inlets is not None:
            seeds = trim_disconnected_voxels(seeds, inlets=inlets)
        edges = seeds * ~seeds_prev * im
        coords = np.vstack(np.where(edges))
        if coords.size > 0:
            nwp = func(
                im=nwp,
                coords=coords,
                r=int(r),
                v=True,
                smooth=smooth,
            )
        nwp[seeds] = True
        mask = nwp * (im_seq == -1)
        im_size[mask] = max(r, 1)
        im_seq[mask] = i + 1
        seeds_prev = np.copy(seeds)

    # Deal with trapping
    if outlets is not None:
        trapped = find_trapped_clusters(
            im=im,
            seq=im_seq,
            outlets=outlets,
            conn="min",
            method="labels",
        )
        im_seq[trapped] = -1
        im_seq = make_contiguous(im_seq, mode="symmetric")
        im_size[trapped] = -1
    results = Results()
    results.im_seq = im_seq * im
    results.im_size = im_size * im
    return results


def drainage_dt_conv(
    im,
    inlets=None,
    outlets=None,
    dt=None,
    steps=None,
    smooth=False,
):
    r"""
    Performs a distance transform based drainage simulation using distance transform
    thresholding for the erosion step and fft-based convolution for the dilation
    step.

    Parameters
    ----------
    im : ndarray
        The boolean image of the void space on which to perform the simulation
    inlets : ndarray (optional)
        A boolean array with `True` values indicating the inlet locations for the
        invading (non-wetting) fluid. If not provided then access limitations will
        not be applied, meaning that the invading fluid can appear anywhere within
        the domain.
    outlets : ndarray (optional)
        A boolean array with `True` values indicating the outlet locations through
        which defending (wetting) phase would exit the domain. If not provided then
        trapping of the wetting phase is ignored.
    dt : ndarray, optional
        The distance transform of the void space. This is optional, but providing
        it if it is already available save some time.
    steps : scalar or array_like
        Controls which sphere sizes to invade. If an `int` then this many steps
        between 1 and the maximum size are used. A `tuple` is treated as the start
        and stop of the integer values. A `list` or `ndarray` is used directly. If
        `None` (default) then each unique value in the distance transform is used.

    Returns
    -------
    results : Dataclass-like object
        An object with the following attributes:

        =========== ================================================================
        Attribute   Description
        =========== ================================================================
        `im_seq`    An ndarray with each voxel indicating the step number at which
                    it was first invaded. -1 indicates uninvaded, either due to
                    the applied `steps` not spanning the full range of sizes in the
                    image, or due to trapping, while 0 indicates residual invading
                    phase.
        `im_size`   A numpy array with each voxel containing the radius of the
                    sphere, in voxels, that first overlapped it. `inf` indicates
                    uninvaded, either due to the applied `steps` not spanning the
                    full range of sizes in the image, or due to trapping, while 0
                    indicates residual invading phase.
        =========== ================================================================

    Notes
    -----
    The distance transform will be executed in parallel if
    `porespy.settings.ncores > 1`
    """
    im = np.array(im, dtype=bool)
    if dt is None:
        dt = edt(im)
    dt = dt.astype(int)
    bins = parse_steps(steps=steps, vals=dt[im], descending=True)
    im_seq = -np.ones_like(im, dtype=int)
    im_size = np.zeros_like(im, dtype=float)
    desc = inspect.currentframe().f_code.co_name  # Get current func name
    for i, r in enumerate(tqdm(bins, desc=desc, **settings.tqdm)):
        seeds = dt >= r
        if inlets is not None:
            seeds = trim_disconnected_voxels(seeds, inlets=inlets)
        if not np.any(seeds):
            continue
        nwp = dilate(im=seeds, r=r, method='conv', smooth=smooth)
        mask = nwp * (im_seq == -1)
        im_size[mask] = max(r, 1)
        im_seq[mask] = i + 1

    # Apply trapping as a post-processing step if outlets given
    if outlets is not None:
        trapped = find_trapped_clusters(
            im=im,
            seq=im_seq,
            outlets=outlets,
            conn="min",
            method="labels",
        )
        im_seq[trapped] = -1
        im_seq = make_contiguous(im_seq, mode="symmetric")
        im_size[trapped] = -1
    results = Results()
    results.im_seq = im_seq * im
    results.im_size = im_size * im
    return results


def drainage_conv(
    im,
    inlets=None,
    outlets=None,
    dt=None,
    steps=None,
    smooth=False,
):
    r"""
    Performs a distance transform based drainage simulation using fft-based
    convolution for both the erosion and dilation steps

    Parameters
    ----------
    im : ndarray
        The boolean image of the void space on which to perform the simulation
    inlets : ndarray (optional)
        A boolean array with `True` values indicating the inlet locations for the
        invading (non-wetting) fluid. If not provided then access limitations will
        not be applied, meaning that the invading fluid can appear anywhere within
        the domain.
    outlets : ndarray (optional)
        A boolean array with `True` values indicating the outlet locations through
        which defending (wetting) phase would exit the domain. If not provided then
        trapping of the wetting phase is ignored.
    dt : ndarray, optional
        The distance transform of the void space. This is optional, but providing
        it if it is already available save some time.
    steps : scalar or array_like
        Controls which sphere sizes to invade. If an `int` then this many steps
        between 1 and the maximum size are used. A `tuple` is treated as the start
        and stop of the integer values. A `list` or `ndarray` is used directly. If
        `None` (default) then each unique value in the distance transform is used.

    Returns
    -------
    results : Dataclass-like object
        An object with the following attributes:

        =========== ================================================================
        Attribute   Description
        =========== ================================================================
        `im_seq`    An ndarray with each voxel indicating the step number at which
                    it was first invaded. -1 indicates uninvaded, either due to
                    the applied `steps` not spanning the full range of sizes in the
                    image, or due to trapping, while 0 indicates residual invading
                    phase.
        `im_size`   A numpy array with each voxel containing the radius of the
                    sphere, in voxels, that first overlapped it. `inf` indicates
                    uninvaded, either due to the applied `steps` not spanning the
                    full range of sizes in the image, or due to trapping, while 0
                    indicates residual invading phase.
        =========== ================================================================
    """
    im = np.array(im, dtype=bool)
    if dt is None:
        dt = edt(im)
    dt = dt.astype(int)
    bins = parse_steps(steps=steps, vals=dt[im], descending=True)
    im_seq = -np.ones_like(im, dtype=int)
    im_size = np.zeros_like(im, dtype=float)
    desc = inspect.currentframe().f_code.co_name  # Get current func name
    for i, r in enumerate(tqdm(bins, desc=desc, **settings.tqdm)):
        se = ps_round(int(r), ndim=im.ndim, smooth=True)
        seeds = ~fftmorphology(~im, se, "dilation")
        if inlets is not None:
            seeds = trim_disconnected_voxels(seeds, inlets=inlets, conn='min')
        if not np.any(seeds):
            continue
        nwp = dilate(im=seeds, r=r, method='conv', smooth=smooth)
        mask = nwp * (im_seq == -1)
        im_size[mask] = max(r, 1)
        im_seq[mask] = i + 1

    # Apply trapping as a post-processing step if outlets given
    if outlets is not None:
        trapped = find_trapped_clusters(
            im=im,
            seq=im_seq,
            outlets=outlets,
            conn="min",
            method="labels",
        )
        im_seq[trapped] = -1
        im_seq = make_contiguous(im_seq, mode="symmetric")
        im_size[trapped] = -1
    results = Results()
    results.im_seq = im_seq * im
    results.im_size = im_size * im
    return results


def drainage_dt(
    im,
    inlets=None,
    outlets=None,
    dt=None,
    steps=None,
    smooth=False,
):
    r"""
    Performs a distance transform based drainage simulation using distance transform
    thresholding for the erosion step and a second distance transform for the
    dilation step.

    Parameters
    ----------
    im : ndarray
        The boolean image of the void space on which to perform the simulation
    inlets : ndarray (optional)
        A boolean array with `True` values indicating the inlet locations for the
        invading (non-wetting) fluid. If not provided then access limitations will
        not be applied, meaning that the invading fluid can appear anywhere within
        the domain.
    outlets : ndarray (optional)
        A boolean array with `True` values indicating the outlet locations through
        which defending (wetting) phase would exit the domain. If not provided then
        trapping of the wetting phase is ignored.
    dt : ndarray, optional
        The distance transform of the void space. This is optional, but providing
        it if it is already available save some time.
    steps : scalar or array_like
        Controls which sphere sizes to invade. If an `int` then this many steps
        between 1 and the maximum size are used. A `tuple` is treated as the start
        and stop of the integer values. A `list` or `ndarray` is used directly. If
        `None` (default) then each unique value in the distance transform is used.

    Returns
    -------
    results : Results object
        A dataclass-like object with the following attributes:

        =========== ================================================================
        Attribute   Description
        =========== ================================================================
        `im_seq`    An ndarray with each voxel indicating the step number at which
                    it was first invaded. -1 indicates uninvaded, either due to
                    the applied `steps` not spanning the full range of sizes in the
                    image, or due to trapping, while 0 indicates residual invading
                    phase.
        `im_size`   A numpy array with each voxel containing the radius of the
                    sphere, in voxels, that first overlapped it. `inf` indicates
                    uninvaded, either due to the applied `steps` not spanning the
                    full range of sizes in the image, or due to trapping, while 0
                    indicates residual invading phase.
        =========== ================================================================

    Notes
    -----
    The distance transforms will be executed in parallel if
    `porespy.settings.ncores > 1`
    """
    im = np.array(im, dtype=bool)
    if dt is None:
        dt = edt(im)
    dt = dt.astype(int)
    bins = parse_steps(steps=steps, vals=dt[im], descending=True)
    im_seq = -np.ones_like(im, dtype=int)
    im_size = np.zeros_like(im, dtype=float)
    desc = inspect.currentframe().f_code.co_name  # Get current func name
    for i, r in enumerate(tqdm(bins, desc=desc, **settings.tqdm)):
        seeds = dt >= r
        if inlets is not None:
            seeds = trim_disconnected_voxels(seeds, inlets=inlets, conn='min')
        if not np.any(seeds):
            continue
        tmp = edt(~seeds)
        nwp = tmp < r if smooth else tmp <= r
        mask = nwp * (im_seq == -1)
        im_size[mask] = max(r, 1)
        im_seq[mask] = i + 1

    # Apply trapping as a post-processing step if outlets given
    if outlets is not None:
        trapped = find_trapped_clusters(
            im=im,
            seq=im_seq,
            outlets=outlets,
            conn="min",
            method="labels",
        )
        im_seq[trapped] = -1
        im_seq = make_contiguous(im_seq, mode="symmetric")
        im_size[trapped] = -1
    results = Results()
    results.im_seq = im_seq * im
    results.im_size = im_size * im
    return results


def drainage(
    im: npt.NDArray,
    pc: npt.NDArray = None,
    dt: npt.NDArray = None,
    inlets: npt.NDArray = None,
    outlets: npt.NDArray = None,
    residual: npt.NDArray = None,
    steps: int = None,
    conn: Literal["min", "max"] = "min",
    min_size: int = 0,
    smooth: bool = True,
):
    r"""
    Simulate drainage using image-based sphere insertion, optionally including
    gravity [1]_.

    Parameters
    ----------
    im : ndarray
        The image of the porous media with ``True`` values indicating the
        void space.
    pc : ndarray, optional
        Precomputed capillary pressure transform which is used to determine
        the invadability of each voxel. If not provided then it is calculated
        as `2/dt`.
    dt : ndarray (optional)
        The distance transform of ``im``. If not provided it will be calculated,
        so supplying it saves time.
    inlets : ndarray, optional
        A boolean image the same shape as ``im``, with ``True`` values
        indicating the inlet locations. If not specified then access limitations
        are not applied so invading phase can appear anywhere within the domain.
    outlets : ndarray, optional
        A boolean image with ``True`` values indicating the outlet locations.
        If this is provided then trapped voxels of wetting phase are found and
        all the output images are adjusted accordingly.
    residual : ndarray, optional
        A boolean array indicating the locations of any residual invading
        phase. This is added to the intermediate image prior to trimming
        disconnected clusters, so will create connections to some clusters
        that would otherwise be removed. The residual phase is indicated
        in the capillary pressure map by ``-np.inf`` values, since these voxels
        are invaded at all applied capillary pressures.
    steps : int or array_like (default = 25)
        The range of pressures to apply. If an integer is given then the given
        number of steps will be created between the lowest and highest values in
        ``pc``. If a list is given, each value in the list is used in ascending
        order. If `None` is given then all the possible values in `pc`
        are used.
    conn : str
        Controls the shape of the structuring element used to find neighboring
        voxels when looking at connectivity of invading blobs.  Options are:

        ========= =============================================================
        Option    Description
        ========= =============================================================
        'min'     This corresponds to a cross with 4 neighbors in 2D and 6
                  neighbors in 3D.
        'max'     This corresponds to a square or cube with 8 neighbors in 2D
                  and 26 neighbors in 3D.
        ========= =============================================================

    min_size : int
        Any clusters of trapped voxels smaller than this size will be set to not
        trapped. This argument is only used if `outlets` is given. This is useful
        to prevent small voxels along edges of the void space from being set to
        trapped. These can appear to be trapped due to the jagged nature of the
        digital image. The default is 0, meaning this adjustment is not applied,
        but a value of 3 or 4 is recommended to activate this adjustment.

    Returns
    -------
    results : Results object
        A dataclass-like object with the following attributes:

        ============ ===============================================================
        Attribute    Description
        ============ ===============================================================
        `im_seq`     An ndarray with each voxel indicating the step number at
                     which it was first invaded by non-wetting phase. -1 indicates
                     uninvaded, either due to the maximum pressure being too low,
                     or trapping, while 0 indicates residual.
        `im_pc`      A numpy array with each voxel value indicating the capillary
                     pressure at which it was invaded. `inf` indicates uninvaded,
                     either due to the maximum pressure being too low, or due to
                     trapping.
        `im_snwp`    A numpy array with each voxel value indicating the global
                     value of the non-wetting phase saturation at the point it
                     was invaded. -1 indicates uninvaded, either due to the maximum
                     pressure being too low, or trapping, while 0 indicates
                     residual.
        `im_size`    A numpy array with each voxel containing the radius of the
                     sphere, in voxels, that first overlapped it. `inf` indicates
                     uninvaded, either due to the maximum pressure being too low,
                     or trapping, while 0 indicates residual.
        `im_trapped` A numpy array with ``True`` values indicating trapped voxels.
                     If `outlets` was not provided it will be all `False`.
        `pc`         1D array of capillary pressure values that were applied.
        `swnp`       1D array of non-wetting phase saturations for each applied
                     value of capillary pressure (``pc``).
        ============ ===============================================================

    References
    ----------
    .. [1] Chadwick EA, Hammen LH, Schulz VP, Bazylak A, Ioannidis MA, Gostick JT.
       Incorporating the effect of gravity into image-based drainage simulations on
       volumetric images of porous media.
       Water Resources Research. 58(3), e2021WR031509 (2022).
       doi: `10.1029/2021WR031509 <https://doi.org/10.1029/2021WR031509>`_

    Examples
    --------
    `Click here
    <https://porespy.org/examples/simulations/reference/drainage.html>`_
    to view online example.

    """
    im = np.array(im, dtype=bool)

    if (outlets is not None) and (inlets is not None):
        outlets = outlets * im
        if np.sum(inlets * outlets):
            raise Exception("Specified inlets and outlets overlap")

    if dt is None:
        dt = edt(im)

    if pc is None:
        # Cast `dt` to float64 to avoid precision loss in `2.0/dt`. With float32
        # `dt`, voxels at integer `dt` boundaries miss the `pc <= P` threshold
        # by ~1e-8, putting them out of sync with integer-`dt` step sequences.
        pc = 2.0/dt.astype(np.float64)
    pc[~im] = 0  # Remove any infs or nans from pc computation

    if isinstance(steps, int):  # Use values in pc for invasion steps
        mask = np.isfinite(pc) * im
        Ps = np.logspace(
            np.log10(pc[mask].min()),
            np.log10(pc[mask].max()*1.05),
            steps,
        )
    elif steps is None:
        Ps = np.unique(pc[im])
    else:
        Ps = np.unique(steps)  # To ensure they are in ascending order

    # Initialize empty arrays to accumulate results of each loop
    im_pc = np.zeros_like(im, dtype=float)
    im_seq = np.zeros_like(im, dtype=int)
    trapped = np.zeros_like(im, dtype=bool)
    if residual is not None:
        im_seq[residual] = 1
    if (outlets is not None) and (residual is not None):
        trapped = find_disconnected_voxels(
            im=im * ~residual,
            inlets=inlets,
            conn=conn,
        )
        trapped += find_disconnected_voxels(
            im=im * ~residual,
            inlets=outlets,
            conn=conn,
        )
    im_seq[trapped] = -1
    nwp_mask = np.zeros_like(im, dtype=bool)
    seeds_prev = np.zeros_like(im, dtype=bool)

    desc = inspect.currentframe().f_code.co_name  # Get current func name
    for step, P in enumerate(tqdm(Ps, desc=desc, **settings.tqdm)):
        # Perform erosion to find all locations invadable at current pressure
        seeds = (pc <= P) * im
        # Trim locations not connected to the inlets
        if inlets is not None:
            seeds = trim_disconnected_voxels(im=seeds, inlets=inlets, conn=conn)
        if not np.any(seeds):
            continue
        # Dilate the erosion to find locations of non-wetting phase
        edges = seeds * (~seeds_prev)  # Isolate edges to speed up inserting
        coords = np.where(edges)  # Find (i, j, k) coordinates of edges
        radii = dt[coords]  # Extract sphere sizes to insert at each new location
        nwp_mask = _insert_disks_at_points_parallel(
            im=nwp_mask,
            coords=np.vstack(coords),
            radii=radii.astype(int),
            v=True,
            smooth=smooth,
            overwrite=False,
        )
        nwp_mask[seeds] = True  # Fill in center in case spheres did not reach
        # Connect residual to invasion front
        if residual is not None:
            if np.any(nwp_mask):  # Add residual blobs to invasion front if touching
                nwp_mask = join_residual_and_invasion_front(
                    im=im,
                    pc=pc,
                    dt=dt,
                    residual=residual,
                    nwp_mask=nwp_mask,
                    seeds_prev=seeds_prev,
                    P=P,
                    conn=conn,
                )
        # Find trapped wetting due to presence of residual
        if all([inlets is not None, outlets is not None, residual is not None]):
            # Find any wetting phase which is pinned between residual and invading
            # front, and set it to uninvaded
            nwp_mask = trim_disconnected_voxels(
                im=nwp_mask * ~trapped,
                inlets=inlets,
                conn=conn,
            )
            trapped += find_disconnected_voxels(
                im=im * ~nwp_mask * ~residual,
                inlets=outlets,
                conn=conn,
            )
            trapped[residual] = False
            nwp_mask[trapped] = False  # Set nwp in trapped regions to 0
            im_seq[trapped] = -1

        mask = nwp_mask * (im_seq == 0) * im
        if np.any(mask):
            im_seq[mask] = step + 1
            im_pc[mask] = P
        # Add new locations to list of invaded locations
        seeds_prev = np.copy(seeds)

    # Set uninvaded voxels to inf and -1
    mask = (im_seq == 0)*im
    im_pc[mask] = np.inf
    im_seq[mask] = -1

    # Update images with residual
    if residual is not None:
        im_pc[residual] = -np.inf
        im_seq[residual] = 0

    # Analyze trapping as a post-processing step if no residual
    if (outlets is not None) and (residual is None):
        trapped = find_trapped_clusters(
            im=im,
            seq=im_seq,
            outlets=outlets,
            min_size=min_size,
            method="labels" if len(Ps) < 100 else "queue",
            conn=conn,
        )
        trapped[im_seq == -1] = True  # TODO: This probably isn't necessary?
        im_pc[trapped] = np.inf  # Trapped defender only displaced at Pc -> inf
        im_seq[trapped] = -1

    # Initialize results object to collect data
    results = Results()
    results.im_snwp = seq_to_satn(seq=im_seq, im=im, mode="drainage")
    results.im_seq = make_contiguous(im_seq, mode='symmetric')
    results.im_pc = im_pc
    results.im_trapped = trapped

    if trapped is not None:
        results.im_seq[trapped] = -1
        results.im_snwp[trapped] = -1
        results.im_pc[trapped] = np.inf

    pc_curve = pc_map_to_pc_curve(
        im=im,
        pc=results.im_pc,
        seq=results.im_seq,
        mode="drainage",
    )
    results.pc = pc_curve.pc
    results.snwp = pc_curve.snwp
    return results


# The following functions are helpers to make the drainage code more concise
def join_residual_and_invasion_front(
    im,
    pc,
    dt,
    nwp_mask,
    residual,
    P,
    seeds_prev,
    conn,
):
    # Find nwp pixels connected to residual
    temp = trim_disconnected_voxels(
        im=residual,
        inlets=nwp_mask,
        conn=conn,
    )
    if np.any(temp):
        # Trim invadable pixels not connected to residual
        seeds = (pc <= P) * im  # Find full set of invadable seeds again
        seeds = trim_disconnected_voxels(
            im=seeds,
            inlets=temp,
            conn=conn,
        )
        # Convert to just edges
        coords = np.where(seeds * (~seeds_prev))
        radii = dt[coords].astype(int)
        nwp_mask = _insert_disks_at_points_parallel(
            im=nwp_mask,
            coords=np.vstack(coords),
            radii=radii.astype(int),
            v=True,
            smooth=True,
            overwrite=False,
        )
    return nwp_mask


# %%
if __name__ == "__main__":
    from copy import copy

    import matplotlib.pyplot as plt

    import porespy as ps
    ps.visualization.set_mpl_style()

    cm = copy(plt.cm.plasma)
    cm.set_under('k')
    cm.set_over('grey')
    bg = "white"
    plots = True

    # %% Run this cell to regenerate the variables in drainage
    seed = np.random.randint(100000)  # 12129, 61227
    im = ps.generators.blobs(
        shape=[1000, 1000],  # [1000, 1000]
        porosity=0.75,  # 0.75
        blobiness=2.5,  # 2.5
        seed=4,  # 4
        periodic=False,  # False
    )
    # im = ~ps.generators.random_spheres(
    #     [600, 600],
    #     r=15,
    #     clearance=15,
    #     seed=1,
    #     edges='extended',
    #     phi=0.2,
    # )
    im = ps.filters.fill_invalid_pores(im)
    inlets = np.zeros_like(im)
    inlets[0, :] = True
    outlets = np.zeros_like(im)
    outlets[-1, :] = True
    dt = edt(im)

    imb = ps.simulations.imbibition_dt(
        im=im,
        inlets=outlets,
        outlets=inlets,
    )
    residual = imb.im_seq == -1

    steps = 25
    pc = ps.filters.capillary_transform(
        im=im,
        dt=dt,
        sigma=0.072,
        theta=180,
        rho_nwp=1000,
        rho_wp=0,
        g=0,
        voxel_size=1e-4,
    )

    # %% Run different drainage simulations
    drn1 = ps.simulations.drainage(
        im=im,
        pc=pc,
        inlets=inlets,
        steps=steps,
        min_size=5,
    )
    drn2 = ps.simulations.drainage(
        im=im,
        pc=pc,
        inlets=inlets,
        outlets=outlets,
        steps=steps,
        min_size=5,
    )
    drn3 = ps.simulations.drainage(
        im=im,
        pc=pc,
        inlets=inlets,
        residual=residual,
        steps=steps,
        min_size=5,
    )
    drn4 = ps.simulations.drainage(
        im=im,
        pc=pc,
        steps=steps,
        inlets=inlets,
        outlets=outlets,
        residual=residual,
        min_size=5,
    )

    # %% Visualize the invasion configurations for each scenario
    if plots:
        fig, ax = plt.subplot_mosaic(
            [['(a)', '(b)', '(e)', '(e)'],
             ['(c)', '(d)', '(e)', '(e)']],
            figsize=[12, 8],
         )
        tmp = np.copy(drn1.im_seq).astype(float)
        vmax = tmp.max()
        tmp[tmp < 0] = vmax + 1
        # tmp[tmp == 0] = np.nan
        tmp[~im] = -1
        ax['(a)'].imshow(tmp, origin='lower', cmap=cm, vmin=0, vmax=vmax)

        tmp = np.copy(drn2.im_seq).astype(float)
        vmax = tmp.max()
        tmp[tmp < 0] = vmax + 1
        # tmp[tmp == 0] = np.nan
        tmp[~im] = -1
        ax['(b)'].imshow(tmp, origin='lower', cmap=cm, vmin=0, vmax=vmax)

        tmp = np.copy(drn3.im_seq).astype(float)
        vmax = tmp.max()
        tmp[tmp < 0] = vmax + 1
        # tmp[tmp == 0] = np.nan
        tmp[~im] = -1
        ax['(c)'].imshow(tmp, origin='lower', cmap=cm, vmin=0, vmax=vmax)

        tmp = np.copy(drn4.im_seq).astype(float)
        vmax = tmp.max()
        tmp[tmp < 0] = vmax + 1
        # tmp[tmp == 0] = np.nan
        tmp[~im] = -1
        ax['(d)'].imshow(tmp, origin='lower', cmap=cm, vmin=0, vmax=vmax)

        Pc, Snwp = ps.metrics.pc_map_to_pc_curve(
            pc=drn1.im_pc, seq=drn1.im_seq, im=im, mode='drainage')
        ax['(e)'].semilogx(Pc, Snwp, 'b->', label='drainage')

        Pc, Snwp = ps.metrics.pc_map_to_pc_curve(
            pc=drn2.im_pc, seq=drn2.im_seq, im=im, mode='drainage')
        ax['(e)'].semilogx(Pc, Snwp, 'r-<', label='drainage w trapping')

        Pc, Snwp = ps.metrics.pc_map_to_pc_curve(
            pc=drn3.im_pc, seq=drn3.im_seq, im=im, mode='drainage')
        ax['(e)'].semilogx(Pc, Snwp, 'g-^', label='drainage w residual')

        Pc, Snwp = ps.metrics.pc_map_to_pc_curve(
            pc=drn4.im_pc, seq=drn4.im_seq, im=im, mode='drainage')
        ax['(e)'].semilogx(Pc, Snwp, 'm-*', label='drainage w residual & trapping')

        ax['(e)'].legend()
