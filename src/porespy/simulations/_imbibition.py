import inspect

import numpy as np
from numba import njit, prange

from porespy.filters import (
    erode,
    fftmorphology,
    find_trapped_clusters,
    seq_to_satn,
    trim_disconnected_voxels,
    find_disconnected_voxels,
)
from porespy.metrics import pc_map_to_pc_curve
from porespy.tools import (
    Results,
    _insert_disk_at_points,
    _insert_disk_at_points_parallel,
    _insert_disks_at_points_parallel,
    get_tqdm,
    get_edt,
    make_contiguous,
    parse_steps,
    ps_round,
    settings,
)

tqdm = get_tqdm()
edt = get_edt()


__all__ = [
    'imbibition',
    'imbibition_dt',
    'imbibition_dt_conv',
    'imbibition_conv',
    'imbibition_bf',
]


def imbibition_bf(
    im,
    inlets=None,
    outlets=None,
    dt=None,
    steps=None,
    smooth=False,
):
    r"""
    Performs a distance transform based imbibition simulation using distance
    transform thresholding for the erosion step and brute-force sphere insertion
    for the dilation step.

    Parameters
    ----------
    im : ndarray
        The boolean image of the void space on which to perform the simulation
    inlets : ndarray (optional)
        A boolean array with `True` values indicating the inlet locations for the
        invading (wetting) fluid. If not provided then access limitations will
        not be applied, meaning that the invading fluid can appear anywhere within
        the domain.
    outlets : ndarray (optional)
        A boolean array with `True` values indicating the outlet locations through
        which defending (non-wetting) phase would exit the domain. If not provided
        then trapping of the non-wetting phase is ignored.
    dt : ndarray, optional
        The distance transform of the void space. This is optional, but providing
        it if it is already available save some time. Also, it can be converted to
        integer type or round to fewer decimal places to reduce the number of unique
        sphere sizes to insert if `steps=None`.
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
    ``porespy.settings.ncores > 1``
    """
    if settings.ncores > 1:
        func = _insert_disk_at_points_parallel
    else:
        func = _insert_disk_at_points
    im = np.array(im, dtype=bool)
    if dt is None:
        dt = edt(im)
    dt = dt.astype(int)

    bins = parse_steps(steps=steps, vals=dt[im], descending=False)

    im_seq = -np.ones_like(im, dtype=int)
    im_size = np.zeros_like(im, dtype=float)
    nwp = np.zeros_like(im, dtype=bool)
    seeds_prev = np.zeros_like(im)

    desc = inspect.currentframe().f_code.co_name  # Get current func name
    for i, r in enumerate(tqdm(bins, desc=desc, **settings.tqdm)):
        # Perform erosion using dt
        seeds = (dt <= r)*im
        if np.any(seeds):
            # Perform dilation using bf
            edges = seeds * ~seeds_prev * im
            coords = np.vstack(np.where(edges))
            nwp.fill(False)
            if coords.size > 0:
                nwp = func(
                    im=nwp,
                    coords=coords,
                    r=int(r),
                    v=True,
                    smooth=smooth,
                )
            nwp[(~seeds)*im] = True
            wp = (~nwp)*im
            # Trim disconnected wetting phase
            if inlets is not None:
                wp = trim_disconnected_voxels(wp, inlets=inlets, conn='min')
        else:
            wp = np.copy(im)
        mask = wp*(im_seq == -1)
        im_size[mask] = r
        im_seq[mask] = i + 1
        seeds_prev = seeds

    if outlets is not None:
        trapped = find_trapped_clusters(
            im=im,
            seq=im_seq,
            outlets=outlets,
            conn='min',
            method='labels',
        )
        im_seq[trapped] = -1
        im_seq = make_contiguous(im_seq, mode='symmetric')
        im_size[trapped] = -1

    results = Results()
    results.im_seq = im_seq*im
    results.im_size = im_size*im
    return results


def imbibition_dt_conv(
    im,
    inlets=None,
    outlets=None,
    dt=None,
    steps=None,
    smooth=False,
):
    r"""
    Performs a distance transform based imbibition simulation using distance
    transform thresholding for the erosion step and fft-based convolution for
    the dilation step.

    Parameters
    ----------
    im : ndarray
        The boolean image of the void space on which to perform the simulation
    inlets : ndarray (optional)
        A boolean array with ``True`` values indicating the inlet locations for the
        invading (wetting) fluid. If not provided then access limitations will
        not be applied, meaning that the invading fluid can appear anywhere within
        the domain.
    outlets : ndarray (optional)
        A boolean array with ``True`` values indicating the outlet locations through
        which defending (non-wetting) phase would exit the domain. If not provided
        then trapping of the non-wetting phase is ignored.
    dt : ndarray, optional
        The distance transform of the void space. This is optional, but providing
        it if it is already available save some time. Also, it can be converted to
        integer type or round to fewer decimal places to reduce the number of unique
        sphere sizes to insert if `steps=None`.
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
    The distance transform will be executed in parallel if
    ``porespy.settings.ncores > 1``
    """
    im = np.array(im, dtype=bool)
    if dt is None:
        dt = edt(im)
    dt = dt.astype(int)
    bins = parse_steps(steps=steps, vals=dt[im], descending=False)
    im_seq = -np.ones_like(im, dtype=int)
    im_size = np.zeros_like(im, dtype=float)
    desc = inspect.currentframe().f_code.co_name  # Get current func name
    for i, r in enumerate(tqdm(bins, desc=desc, **settings.tqdm)):
        # Perform erosion using dt
        seeds = dt >= r
        if np.any(seeds):
            # Perform dilation using convolution
            se = ps_round(r, ndim=im.ndim, smooth=smooth)
            wp = im*~fftmorphology(seeds, se, mode='dilation')
            # Trim disconnected wetting phase
            if inlets is not None:
                wp = trim_disconnected_voxels(wp, inlets=inlets)
        else:
            wp = np.copy(im)
        mask = wp*(im_seq == -1)
        im_size[mask] = r
        im_seq[mask] = i+1

    # Apply trapping as a post-processing step if outlets given
    if outlets is not None:
        trapped = find_trapped_clusters(
            im=im,
            seq=im_seq,
            outlets=outlets,
            conn='min',
            method='labels',
        )
        im_seq[trapped] = -1
        im_seq = make_contiguous(im_seq, mode='symmetric')
        im_size[trapped] = -1

    results = Results()
    results.im_seq = im_seq*im
    results.im_size = im_size*im
    return results


def imbibition_dt(
    im,
    inlets=None,
    outlets=None,
    dt=None,
    steps=None,
    smooth=True,
):
    r"""
    Performs a distance transform based imbibition simulation using distance
    transform thresholding for the erosion step and a second distance transform
    for the dilation step.

    Parameters
    ----------
    im : ndarray
        The boolean image of the void space on which to perform the simulation
    inlets : ndarray (optional)
        A boolean array with `True` values indicating the inlet locations for the
        invading (wetting) fluid. If not provided then access limitations will
        not be applied, meaning that the invading fluid can appear anywhere within
        the domain.
    outlets : ndarray (optional)
        A boolean array with `True` values indicating the outlet locations through
        which defending (non-wetting) phase would exit the domain. If not provided
        then trapping of the non-wetting phase is ignored.
    dt : ndarray, optional
        The distance transform of the void space. This is optional, but providing
        it if it is already available save some time. Also, it can be converted to
        integer type or round to fewer decimal places to reduce the number of unique
        sphere sizes to insert if `steps=None`.
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
    The distance transforms will be executed in parallel if
    ``porespy.settings.ncores > 1``
    """
    im = np.array(im, dtype=bool)
    if dt is None:
        dt = edt(im)
    dt = dt.astype(int)
    bins = parse_steps(steps=steps, vals=dt[im], descending=False)
    im_seq = -np.ones_like(im, dtype=int)
    im_size = np.zeros_like(im, dtype=float)
    desc = inspect.currentframe().f_code.co_name  # Get current func name
    for i, r in enumerate(tqdm(bins, desc=desc, **settings.tqdm)):
        # Perform erosion using dt
        seeds = dt >= r
        if np.any(seeds):
            # Perform dilation using dt
            tmp = edt(~seeds)
            wp = ~(tmp < r) if smooth else ~(tmp <= r)
            wp[~im] = False
            # Trim disconnected wetting phase
            if inlets is not None:
                wp = trim_disconnected_voxels(wp, inlets=inlets)
        else:
            wp = np.copy(im)
        mask = wp*(im_seq == -1)
        im_size[mask] = r
        im_seq[mask] = i+1

    # Apply trapping as a post-processing step if outlets given
    if outlets is not None:
        trapped = find_trapped_clusters(
            im=im,
            seq=im_seq,
            outlets=outlets,
            conn='min',
            method='labels',
        )
        im_seq[trapped] = -1
        im_seq = make_contiguous(im_seq, mode='symmetric')
        im_size[trapped] = -1
    results = Results()
    results.im_seq = im_seq*im
    results.im_size = im_size*im
    return results


def imbibition_conv(
    im,
    inlets=None,
    outlets=None,
    residual=None,
    dt=None,
    steps=None,
    smooth=True,
):
    r"""
    Performs a distance transform based imbibition simulation using fft-based
    convolution for both the erosion and dilation steps

    Parameters
    ----------
    im : ndarray
        The boolean image of the void space on which to perform the simulation
    inlets : ndarray (optional)
        A boolean array with `True` values indicating the inlet locations for the
        invading (wetting) fluid. If not provided then access limitations will
        not be applied, meaning that the invading fluid can appear anywhere within
        the domain.
    outlets : ndarray (optional)
        A boolean array with `True` values indicating the outlet locations through
        which defending (non-wetting) phase would exit the domain. If not provided
        then trapping of the non-wetting phase is ignored.
    dt : ndarray, optional
        The distance transform of the void space. This is optional, but providing
        it if it is already available save some time. Also, it can be converted to
        integer type or round to fewer decimal places to reduce the number of unique
        sphere sizes to insert if `steps=None`.
    steps : scalar or array_like
        Controls which sphere sizes to invade. If an `int` then this many steps
        between 1 and the maximum size are used. A `tuple` is treated as the start
        and stop of the integer values. A `list` or `ndarray` is used directly. If
        `None` (default) then each unique value in the distance transform is used.
    smooth : boolean
        If `True` (default) then the spheres are drawn without any single voxel
        protrusions on the faces.

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
    """
    im = np.array(im, dtype=bool)
    if dt is None:
        dt = edt(im)
    dt = dt.astype(int)
    bins = parse_steps(steps=steps, vals=dt[im], descending=False)
    im_seq = -np.ones_like(im, dtype=int)
    im_size = np.zeros_like(im, dtype=float)
    desc = inspect.currentframe().f_code.co_name  # Get current func name
    for i, r in enumerate(tqdm(bins, desc=desc, **settings.tqdm)):
        # Perform erosion using convolution
        se = ps_round(r, ndim=im.ndim, smooth=True)
        seeds = ~fftmorphology(~im, se, mode='dilation')
        if np.any(seeds):
            # Perform dilation using convolution
            se = ps_round(r, ndim=im.ndim, smooth=smooth)
            wp = im*~fftmorphology(seeds, se, mode='dilation')
            # Trim disconnected wetting phase
            if inlets is not None:
                wp = trim_disconnected_voxels(wp, inlets=inlets)
        else:
            wp = np.copy(im)
        mask = wp*(im_seq == -1)
        im_size[mask] = r
        im_seq[mask] = i+1

    if outlets is not None:
        trapped = find_trapped_clusters(
            im=im,
            seq=im_seq,
            outlets=outlets,
            conn='min',
            method='labels',
        )
        im_seq[trapped] = -1
        im_seq = make_contiguous(im_seq, mode='symmetric')
        im_size[trapped] = -1
    results = Results()
    results.im_seq = im_seq*im
    results.im_size = im_size*im
    return results


def imbibition(
    im,
    pc=None,
    dt=None,
    inlets=None,
    outlets=None,
    residual=None,
    steps=25,
    min_size=0,
    conn='min',
    smooth=False,
):
    r"""
    Performs an imbibition simulation using image-based sphere insertion

    Parameters
    ----------
    im : ndarray
        The image of the porous materials with void indicated by ``True``
    pc : ndarray
        An array containing precomputed capillary pressure values in each
        voxel. This can include gravity effects or not. This can be generated
        by `capillary_transform`. If not provided then `2/dt` is used.
    dt : ndarray (optional)
        The distance transform of `im`.  If not provided it will be
        calculated, so supplying it saves time.
    inlets : ndarray
        An image the same shape as `im` with `True` values indicating the
        wetting fluid inlet(s).  If `None` then the wetting film is able to
        appear anywhere within the domain.
    outlets : ndarray, optional
        A boolean image with ``True`` values indicating the outlet locations.
        If this is provided then trapped voxels of non-wetting phase are found and
        all the output images are adjusted accordingly.
    residual : ndarray, optional
        A boolean mask the same shape as `im` with `True` values
        indicating to locations of residual wetting phase.
    steps : int or array_like (default = 25)
        The range of pressures to apply. If an integer is given then logarithmically
        spaced steps will be created between the lowest and highest pressures in
        `pc`. If a `list` is given, each value in the list is used directly, in
        order.
    min_size : int
        Any clusters of trapped voxels smaller than this size will be set to not
        trapped. This argument is only used if `outlets` is given. This is useful
        to prevent small voxels along edges of the void space from being set to
        trapped. These can appear to be trapped due to the jagged nature of the
        digital image. The default is 0, meaning this adjustment is not applied,
        but a value of 3 or 4 is recommended to activate this adjustment.
    conn : str
        Can be either `'min'` or `'max'` and controls the shape of the structuring
        element used to determine voxel connectivity when assessing trapping. The
        default is `'min'` which imposes the strictest criteria, so that voxels
        must share a face to be considered connected.

    Returns
    -------
    results : Result Object
        A dataclass-like object with the following attributes:

        =========== ===========================================================
        Attribute   Description
        =========== ===========================================================
        im_pc       An ndarray with each voxel indicating the step number at
                    which it was first invaded by wetting phase.
        im_seq      A numpy array with each voxel value indicating the sequence
                    at which it was invaded by the wetting phase.  Values of -1
                    indicate that it was not invaded, either because it was
                    trapped, inaccessbile, or sufficient pressure was not
                    reached.
        im_snwp     A numpy array with each voxel value indicating the global
                    non-wetting phase saturation at the point it was invaded.
        im_trapped  A numpy array with ``True`` values indicating trapped
                    voxels if `outlets` was provided, otherwise will be `None`.
        pc          1D array of capillary pressure values that were applied
        snwp        1D array of non-wetting phase saturations for each applied
                    value of capillary pressure (``pc``).
        =========== ===========================================================

    Notes
    -----
    The simulation proceeds as though the non-wetting phase pressure is very
    high and is incrementally lowered. Then imbibition occurs into the smallest
    accessible regions at each step. Closed or inaccessible pores are assumed
    to be filled with wetting phase.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/simulations/reference/imbibition.html>`_
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

    if isinstance(steps, int):
        mask = np.isfinite(pc)*im
        Ps = np.logspace(
            np.log10(pc[mask].max()),
            np.log10(pc[mask].min()*0.95),
            steps,
        )
    elif steps is None:
        Ps = np.unique(pc[im])[::-1]
    else:
        Ps = np.unique(steps)[::-1]  # To ensure they are in descending order

    # Initialize empty arrays to accumulate results of each loop
    im_pc = np.zeros_like(im, dtype=float)
    im_seq = np.zeros_like(im, dtype=int)
    trapped = np.zeros_like(im)
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

    desc = inspect.currentframe().f_code.co_name  # Get current func name
    for step, P in enumerate(tqdm(Ps, desc=desc, **settings.tqdm)):
        invadable = (pc <= P)*im  # This means 'invadable by non-wetting phase'
        # Using FFT-based erosion to find edges.  When struct is small, this is
        # quite fast so it saves time overall by reducing the number of spheres
        # that need to be inserted.
        # TODO: This can be made faster if I find a way to get only seeds on edge,
        # so less spheres need to be drawn
        edges = (~erode(invadable, r=1, smooth=False, method='conv'))*invadable
        nwp_mask = np.zeros_like(im, dtype=bool)
        if np.any(edges):
            coords = np.where(edges)
            radii = dt[coords].astype(int)
            nwp_mask = _insert_disks_at_points_parallel(
                im=nwp_mask,
                coords=np.vstack(coords),
                radii=radii,
                v=True,
                smooth=smooth,
                overwrite=True,
            )
            nwp_mask += invadable
        if inlets is not None:
            nwp_mask = ~trim_disconnected_voxels(
                im=(~nwp_mask)*im,
                inlets=inlets,
                conn=conn,
            )*im

        # Deal with impact of residual, if present
        if (residual is not None) and (outlets is not None):
            # Remove any wp which is blocked by previously trapped nwp
            nwp_mask = ~trim_disconnected_voxels(
                im=(~nwp_mask)*(~trapped),
                inlets=inlets,
                conn=conn,
            )
            trapped += find_disconnected_voxels(
                im=im*(nwp_mask)*(~residual),
                inlets=outlets,
                conn=conn,
            )
            trapped[residual] = False
            nwp_mask[trapped] = True
            im_seq[trapped] = -1

        mask = (nwp_mask == 0) * (im_seq == 0) * im
        if np.any(mask):
            im_seq[mask] = step + 1
            im_pc[mask] = P

    # Set uninvaded voxels to -inf and -1
    mask = (im_seq == 0)*im
    im_pc[mask] = -np.inf
    im_seq[mask] = -1

    # Add residual if given
    if residual is not None:
        im_pc[residual] = np.inf
        im_seq[residual] = 0

    # Check for trapping as a post-processing step if no residual
    if (outlets is not None) and (residual is None):
        trapped = find_trapped_clusters(
            im=im,
            seq=im_seq,
            outlets=outlets,
            min_size=min_size,
            method='labels' if len(Ps) < 100 else 'queue',
            conn=conn,
        )
        im_pc[trapped] = -np.inf
        im_seq[trapped] = -1

    im_seq = make_contiguous(im_seq, mode='symmetric')
    satn = seq_to_satn(im=im, seq=im_seq, mode='imbibition')

    # Initialize Results object to collect data
    results = Results()
    results.im_snwp = satn
    results.im_seq = im_seq
    results.im_pc = im_pc
    results.im_trapped = trapped

    if trapped is not None:
        results.im_seq[trapped] = -1
        results.im_snwp[trapped] = -1
        results.im_pc[trapped] = -np.inf

    pc_curve = pc_map_to_pc_curve(
        pc=im_pc,
        im=im,
        seq=im_seq,
        mode='imbibition',
    )
    results.pc = pc_curve.pc
    results.snwp = pc_curve.snwp
    return results


@njit(parallel=True)
def _insert_disks_npoints_nradii_1value_parallel(
    im,
    coords,
    radii,
    v,
    overwrite=False,
    smooth=False,
):  # pragma: no cover
    if im.ndim == 2:
        xlim, ylim = im.shape
        for row in prange(len(coords[0])):
            i, j = coords[0][row], coords[1][row]
            r = radii[row]
            for a, x in enumerate(range(i-r, i+r+1)):
                if (x >= 0) and (x < xlim):
                    for b, y in enumerate(range(j-r, j+r+1)):
                        if (y >= 0) and (y < ylim):
                            R = ((a - r)**2 + (b - r)**2)**0.5
                            if (R <= r)*(~smooth) or (R < r)*(smooth):
                                if overwrite or (im[x, y] == 0):
                                    im[x, y] = v
    else:
        xlim, ylim, zlim = im.shape
        for row in prange(len(coords[0])):
            i, j, k = coords[0][row], coords[1][row], coords[2][row]
            r = radii[row]
            for a, x in enumerate(range(i-r, i+r+1)):
                if (x >= 0) and (x < xlim):
                    for b, y in enumerate(range(j-r, j+r+1)):
                        if (y >= 0) and (y < ylim):
                            for c, z in enumerate(range(k-r, k+r+1)):
                                if (z >= 0) and (z < zlim):
                                    R = ((a - r)**2 + (b - r)**2 + (c - r)**2)**0.5
                                    if (R <= r)*(~smooth) or (R < r)*(smooth):
                                        if overwrite or (im[x, y, z] == 0):
                                            im[x, y, z] = v
    return im


# %%

if __name__ == '__main__':
    from copy import copy

    import matplotlib.pyplot as plt
    import numpy as np
    from edt import edt

    import porespy as ps
    ps.visualization.set_mpl_style()

    cm = copy(plt.cm.plasma)
    cm.set_under('k')
    cm.set_over('grey')
    steps = 50

    i = np.random.randint(1, 100000)  # bad: 38364, good: 65270, 71698
    # i = 59477  # Bug in pc curve if lowest point is not 0.99 x min(pc)
    # i = 38364
    im = ps.generators.blobs(
        shape=[1000, 1000],  # [1000, 1000]
        porosity=0.75,  # 0.75
        blobiness=2.5,  # 2.5
        seed=4,  # 4
        periodic=False,  # False
    )
    im = ps.filters.fill_invalid_pores(im)

    inlets = ps.generators.faces(im.shape, inlet=0)
    outlets = ps.generators.faces(im.shape, outlet=0)
    dt = edt(im)
    pc = ps.filters.capillary_transform(im=im, dt=dt, voxel_size=1e-4)

    drn = ps.simulations.drainage(im=im, pc=pc, inlets=inlets, outlets=outlets, steps=steps)
    residual = drn.im_trapped

    imb1 = imbibition(im=im, pc=pc, inlets=inlets, steps=steps, min_size=5)
    imb2 = imbibition(im=im, pc=pc, inlets=inlets, outlets=outlets, steps=steps, min_size=5)
    imb3 = imbibition(im=im, pc=pc, inlets=inlets, residual=residual, steps=steps, min_size=5)
    imb4 = imbibition(im=im, pc=pc, inlets=inlets, outlets=outlets, residual=residual, steps=steps, min_size=5)

    # %%

    fig, ax = plt.subplot_mosaic(
        [['(a)', '(b)', '(e)', '(e)'],
         ['(c)', '(d)', '(e)', '(e)']],
        figsize=[12, 8],
     )
    tmp = np.copy(imb1.im_seq).astype(float)
    vmax = tmp.max()
    tmp[tmp < 0] = vmax + 1
    # tmp[tmp == 0] = np.nan
    tmp[~im] = -1
    ax['(a)'].imshow(tmp, origin='lower', cmap=cm, vmin=0, vmax=vmax)

    tmp = np.copy(imb2.im_seq).astype(float)
    vmax = tmp.max()
    tmp[tmp < 0] = vmax + 1
    # tmp[tmp == 0] = np.nan
    tmp[~im] = -1
    ax['(b)'].imshow(tmp, origin='lower', cmap=cm, vmin=0, vmax=vmax)

    tmp = np.copy(imb3.im_seq).astype(float)
    vmax = tmp.max()
    tmp[tmp < 0] = vmax + 1
    # tmp[tmp == 0] = np.nan
    tmp[~im] = -1
    ax['(c)'].imshow(tmp, origin='lower', cmap=cm, vmin=0, vmax=vmax)

    tmp = np.copy(imb4.im_seq).astype(float)
    vmax = tmp.max()
    tmp[tmp < 0] = vmax + 1
    # tmp[tmp == 0] = np.nan
    tmp[~im] = -1
    ax['(d)'].imshow(tmp, origin='lower', cmap=cm, vmin=0, vmax=vmax)

    Pc, Snwp = ps.metrics.pc_map_to_pc_curve(
        pc=imb1.im_pc, seq=imb1.im_seq, im=im, mode='imbibition')
    ax['(e)'].semilogx(Pc, Snwp, 'b->', label='imbibition')

    Pc, Snwp = ps.metrics.pc_map_to_pc_curve(
        pc=imb2.im_pc, seq=imb2.im_seq, im=im, mode='imbibition')
    ax['(e)'].semilogx(Pc, Snwp, 'r-<', label='imbibition w trapping')

    Pc, Snwp = ps.metrics.pc_map_to_pc_curve(
        pc=imb3.im_pc, seq=imb3.im_seq, im=im, mode='imbibition')
    ax['(e)'].semilogx(Pc, Snwp, 'g-^', label='imbibition w residual')

    Pc, Snwp = ps.metrics.pc_map_to_pc_curve(
        pc=imb4.im_pc, seq=imb4.im_seq, im=im, mode='imbibition')
    ax['(e)'].semilogx(Pc, Snwp, 'm-*', label='imbibition w residual & trapping')

    ax['(e)'].legend()


    # %%
    # i = 50591
    # voxel_size = 1e-5
    # steps = 50
    # im = ps.generators.blobs([100, 100], porosity=0.6, seed=1)
    # dt = edt(im).astype(int)
    # steps = ps.tools.parse_steps(steps=50, vals=dt, mask=im, pad=(1, 1))
    # imb7 = imbibition_dt(im=im, dt=dt, steps=steps, inlets=None, smooth=True)
    # imb8 = imbibition_dt(im=im, dt=dt, steps=steps, inlets=None, smooth=False)
    # imb9 = imbibition_conv(im=im, dt=dt, steps=steps, inlets=None, smooth=True)
    # imb10 = imbibition_conv(im=im, dt=dt, steps=steps, inlets=None, smooth=False)

    # # assert np.all(imb7.im_seq == imb9.im_seq)
    # # assert np.all(imb8.im_seq == imb10.im_seq)

    # fig, ax = plt.subplots(2, 2)
    # ax[0][0].imshow(imb7.im_seq/im)
    # ax[0][0].set_title('dt, smooth')
    # ax[0][1].imshow(imb8.im_seq/im)
    # ax[0][1].set_title('dt, not-smooth')
    # ax[1][0].imshow(imb9.im_seq/im)
    # ax[1][0].set_title('fft, smooth')
    # ax[1][1].imshow(imb10.im_seq/im)
    # ax[1][1].set_title('fft, not-smooth')
