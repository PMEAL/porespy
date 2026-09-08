import heapq as hq
import inspect
import logging
from typing import Literal

import numpy as np
import numpy.typing as npt
from numba import njit

from porespy.filters import find_small_clusters, find_trapped_clusters, seq_to_satn
from porespy.tools import (
    Results,
    _insert_disk_at_points,
    get_edt,
    get_tqdm,
    make_contiguous,
    settings,
)

logger = logging.getLogger(__name__)
tqdm = get_tqdm()
edt = get_edt()


__all__ = [
    'qbip',
    'ibip',
    'injection',
]


def qbip(
    im: npt.NDArray,
    pc: npt.NDArray = None,
    dt: npt.NDArray = None,
    inlets: npt.NDArray = None,
    outlets: npt.NDArray = None,
    maxiter: int = None,
    return_sizes: bool = False,
    return_pressures: bool = True,
    conn: Literal['min', 'max'] = 'min',
    min_size: int = 0,
):
    r"""
    Simulates non-wetting injection using a priority queue, optionally
    including the effect of gravity
    """
    im = np.atleast_3d(im == 1)
    n_sites = im.sum()
    if maxiter is None:  # Compute number of pixels in image
        maxiter = n_sites

    if inlets is None:
        inlets = np.zeros_like(im)
        inlets[0, ...] = True
    inlets = np.atleast_3d(inlets)

    if dt is None:
        dt = edt(im)
    dt = np.atleast_3d(dt)

    if pc is None:
        pc = 2.0/dt
    pc = np.atleast_3d(pc)

    # Record centers in heap-pop order.  Negative entries mark the final center
    # in each pressure batch, so no separate per-center step array is needed.
    index_dtype = np.int32 if im.size <= np.iinfo(np.int32).max else np.int64
    inv_order = np.empty(n_sites, dtype=index_dtype)
    count, step = _qbip_inner_loop(
        im=im,
        inlets=inlets,
        pc=pc,
        order=inv_order,
        maxiter=maxiter,
        conn=conn,
    )
    logger.info(f"Exiting after {step} steps")

    # Draw the spheres after traversal so queue operations and rasterization
    # can be profiled and optimized independently.
    inv_seq = np.zeros_like(im, dtype=int)
    inv_pc = np.zeros_like(im, dtype=float)
    if return_pressures is False:
        inv_pc *= -np.inf  # This is a flag to the numba-jit function to ignore it
    inv_size = np.zeros_like(im, dtype=float)
    if return_sizes is False:
        inv_size *= -np.inf  # This is a flag to the numba-jit function to ignore it
    max_radius = int(np.max(dt))
    if max_radius <= np.iinfo(np.uint8).max:
        depth_dtype = np.uint8
    elif max_radius <= np.iinfo(np.uint16).max:
        depth_dtype = np.uint16
    elif max_radius <= np.iinfo(np.uint32).max:
        depth_dtype = np.uint32
    else:
        depth_dtype = np.uint64
    im_depth = np.zeros_like(im, dtype=depth_dtype)
    squared_distance = np.arange(max_radius**2 + 1)
    ceil_distance = np.ceil(np.sqrt(squared_distance)).astype(depth_dtype)
    sequence, pressure, size, drawn, skipped = _draw_qbip_spheres(
        order=inv_order[:count],
        dt=dt,
        pc=pc,
        seq=inv_seq,
        pressure=inv_pc,
        size=inv_size,
        im_depth=im_depth,
        ceil_distance=ceil_distance,
    )
    logger.info(f"Drew {drawn} spheres and skipped {skipped} contained spheres")
    # Reduce back to 2D if necessary
    sequence = sequence.squeeze()
    pressure = pressure.squeeze()
    size = size.squeeze()
    pc = pc.squeeze()
    im = im.squeeze()

    # Convert invasion image so that uninvaded voxels are set to -1 and solid to 0
    sequence[sequence == 0] = -1
    sequence[~im] = 0
    sequence = make_contiguous(im=sequence, mode='symmetric')
    # Deal with invasion pressures and sizes similarly
    if return_pressures:
        pressure[sequence < 0] = np.inf
        pressure[~im] = 0
    if return_sizes:
        size[sequence < 0] = np.inf
        size[~im] = 0
    # Deal with trapping if outlets were specified
    if outlets is not None:
        logger.info('Computing trapping and adjusting outputs')
        trapped = find_trapped_clusters(
            im=im,
            seq=sequence,
            outlets=outlets,
            conn=conn,
            method='queue',
        )
        trapped = trapped.squeeze()
        if min_size > 0:
            temp = find_small_clusters(
                im=im,
                trapped=trapped,
                min_size=min_size,
                conn=conn,
            )
            trapped = temp.im_trapped
        pressure = pressure.astype(float).squeeze()
        pressure[trapped] = np.inf
        sequence[trapped] = -1
        sequence = make_contiguous(im=sequence, mode='symmetric')
        size = size.astype(float)
        size[trapped] = np.inf

    # Create results object for collected returned values
    results = Results()
    results.im_seq = sequence
    results.im_snwp = seq_to_satn(sequence, im=im)  # convert sequence to saturation
    if return_pressures:
        results.im_pc = pressure
    if return_sizes:
        results.im_size = size
    return results


@njit
def _qbip_inner_loop(
    im,
    inlets,
    pc,
    order,
    maxiter,
    conn,
):  # pragma: no cover
    # Store only entry pressure and a flat index in the heap.  Radius and
    # coordinates are recovered after popping to keep frontier entries small.
    inds = np.where(inlets*im)
    bd = []
    _, ylim, zlim = im.shape
    stride0 = ylim * zlim
    max_conn = conn == 'max'
    for i, j, k in zip(inds[0], inds[1], inds[2]):
        ind = i * stride0 + j * zlim + k
        bd.append((pc[i, j, k], ind))
    hq.heapify(bd)
    # Note which sites have been added to heap already
    processed = inlets*im + ~im  # Add solid phase to be safe
    count = 0
    step = 1  # Total step number
    for _ in range(1, maxiter):
        if len(bd) == 0:
            break
        pts = [hq.heappop(bd)]  # Put next site into pts list
        while len(bd) and (bd[0][0] == pts[0][0]):  # Pop any items with equal Pc
            pts.append(hq.heappop(bd))
        for pt in pts:
            ind = pt[1]
            order[count] = ind
            count += 1
            i = ind // stride0
            rem = ind - i * stride0
            j = rem // zlim
            k = rem - j * zlim
            _push_valid_neighbors(
                bd=bd,
                processed=processed,
                pc=pc,
                i=i,
                j=j,
                k=k,
                stride0=stride0,
                max_conn=max_conn,
            )
        order[count - 1] = -order[count - 1] - 1
        step += 1
    return count, step


@njit
def _draw_qbip_spheres(
    order,
    dt,
    pc,
    seq,
    pressure,
    size,
    im_depth,
    ceil_distance,
    smooth=True,
):  # pragma: no cover
    _, ylim, zlim = seq.shape
    stride0 = ylim * zlim
    draw_pressure = pressure[0, 0, 0] > -np.inf
    draw_size = size[0, 0, 0] > -np.inf
    drawn = 0
    skipped = 0
    step = 1
    for item in order:
        end_of_step = item < 0
        ind = -item - 1 if end_of_step else item
        i = ind // stride0
        rem = ind - i * stride0
        j = rem // zlim
        k = rem - j * zlim
        r = int(dt[i, j, k])
        # im_depth is a conservative distance to the edge of the union of
        # earlier spheres.  This sphere is redundant when its radius fits at
        # its center without crossing an earlier sphere's boundary.
        if im_depth[i, j, k] >= r:
            skipped += 1
        else:
            _insert_qbip_sphere(
                seq=seq,
                pressure=pressure,
                size=size,
                im_depth=im_depth,
                ceil_distance=ceil_distance,
                i=i,
                j=j,
                k=k,
                r=r,
                step=step,
                value_pc=pc[i, j, k],
                value_size=dt[i, j, k],
                draw_pressure=draw_pressure,
                draw_size=draw_size,
                smooth=smooth,
            )
            drawn += 1
        if end_of_step:
            step += 1
    return seq, pressure, size, drawn, skipped


@njit
def _get_axial_extent(distance_squared, ceil_distance, smooth):
    if smooth:
        if distance_squared <= 0:
            return -1
        return int(ceil_distance[distance_squared]) - 1
    if distance_squared < 0:
        return -1
    extent = int(ceil_distance[distance_squared])
    if extent**2 > distance_squared:
        extent -= 1
    return extent


@njit
def _insert_qbip_sphere(
    seq,
    pressure,
    size,
    im_depth,
    ceil_distance,
    i,
    j,
    k,
    r,
    step,
    value_pc,
    value_size,
    draw_pressure,
    draw_size,
    smooth,
):  # pragma: no cover
    xlim, ylim, zlim = seq.shape
    radius_squared = r**2
    for x in range(max(0, i - r), min(i + r + 1, xlim)):
        dx = x - i
        yz_extent = _get_axial_extent(
            radius_squared - dx**2, ceil_distance, smooth)
        if zlim > 1:
            for y in range(max(0, j - yz_extent), min(j + yz_extent + 1, ylim)):
                dy = y - j
                z_extent = _get_axial_extent(
                    radius_squared - dx**2 - dy**2,
                    ceil_distance,
                    smooth,
                )
                for z in range(max(0, k - z_extent), min(k + z_extent + 1, zlim)):
                    dz = z - k
                    distance_squared = dx**2 + dy**2 + dz**2
                    depth = r - ceil_distance[distance_squared]
                    if im_depth[x, y, z] < depth:
                        im_depth[x, y, z] = depth
                    if seq[x, y, z] == 0:
                        seq[x, y, z] = step
                    if draw_pressure and (pressure[x, y, z] == 0):
                        pressure[x, y, z] = value_pc
                    if draw_size and (size[x, y, z] == 0):
                        size[x, y, z] = value_size
        else:
            for y in range(max(0, j - yz_extent), min(j + yz_extent + 1, ylim)):
                dy = y - j
                distance_squared = dx**2 + dy**2
                depth = r - ceil_distance[distance_squared]
                if im_depth[x, y, 0] < depth:
                    im_depth[x, y, 0] = depth
                if seq[x, y, 0] == 0:
                    seq[x, y, 0] = step
                if draw_pressure and (pressure[x, y, 0] == 0):
                    pressure[x, y, 0] = value_pc
                if draw_size and (size[x, y, 0] == 0):
                    size[x, y, 0] = value_size


@njit
def _push_valid_neighbors(
    bd,
    processed,
    pc,
    i,
    j,
    k,
    stride0,
    max_conn,
):  # pragma: no cover
    xlim, ylim, zlim = processed.shape
    if not max_conn:
        if (i > 0) and not processed[i - 1, j, k]:
            processed[i - 1, j, k] = True
            hq.heappush(bd, (pc[i - 1, j, k], (i - 1) * stride0 + j * zlim + k))
        if (i + 1 < xlim) and not processed[i + 1, j, k]:
            processed[i + 1, j, k] = True
            hq.heappush(bd, (pc[i + 1, j, k], (i + 1) * stride0 + j * zlim + k))
        if (j > 0) and not processed[i, j - 1, k]:
            processed[i, j - 1, k] = True
            hq.heappush(bd, (pc[i, j - 1, k], i * stride0 + (j - 1) * zlim + k))
        if (j + 1 < ylim) and not processed[i, j + 1, k]:
            processed[i, j + 1, k] = True
            hq.heappush(bd, (pc[i, j + 1, k], i * stride0 + (j + 1) * zlim + k))
        if (k > 0) and not processed[i, j, k - 1]:
            processed[i, j, k - 1] = True
            hq.heappush(bd, (pc[i, j, k - 1], i * stride0 + j * zlim + k - 1))
        if (k + 1 < zlim) and not processed[i, j, k + 1]:
            processed[i, j, k + 1] = True
            hq.heappush(bd, (pc[i, j, k + 1], i * stride0 + j * zlim + k + 1))
    else:
        for x in range(max(0, i - 1), min(i + 2, xlim)):
            for y in range(max(0, j - 1), min(j + 2, ylim)):
                for z in range(max(0, k - 1), min(k + 2, zlim)):
                    if not processed[x, y, z]:
                        processed[x, y, z] = True
                        nind = x * stride0 + y * zlim + z
                        hq.heappush(bd, (pc[x, y, z], nind))

@njit
def _where(arr):
    inds = np.where(arr)
    result = np.vstack(inds)
    return result


def ibip(
    im: npt.NDArray,
    inlets: npt.NDArray = None,
    outlets: npt.NDArray = None,
    dt: npt.NDArray = None,
    maxiter: int = 10000,
    return_sizes: bool = True,
    conn: str = 'min',
    min_size: int = 0,
):
    r"""
    Simulates non-wetting fluid injection on an image using the IBIP algorithm [3]_

    Parameters
    ----------
    im : ND-array
        Boolean array with ``True`` values indicating void voxels
    inlets : ND-array
        Boolean array with ``True`` values indicating where the invading fluid
        is injected from.  If ``None``, all faces will be used.
    dt : ND-array (optional)
        The distance transform of ``im``.  If not provided it will be
        calculated, so supplying it saves time.
    maxiter : scalar
        The number of steps to apply before stopping.  The default is to run
        for 10,000 steps which is almost certain to reach completion if the
        image is smaller than about 250-cubed.
    return_sizes : bool
        If ``True`` then an array containing the size of the sphere which first
        overlapped each voxel is returned. This array is not computed by default
        as it increases computation time.

    Returns
    -------
    results : dataclass-like
        A dataclass-like object with the following arrays as attributes:

        ============= ================================================================
        Attribute     Description
        ============= ================================================================
        im_seq        A numpy array with each voxel value containing the step at
                      which it was invaded.  Uninvaded voxels are set to -1.
        im_snwp       A numpy array with each voxel value indicating the saturation
                      present in the domain it was invaded. Solids are given 0, and
                      uninvaded regions are given -1.
        im_size       If ``return_sizes`` was set to ``True``, then a numpy array with
                      each voxel containing the radius of the sphere, in voxels,
                      that first overlapped it.
        ============= ================================================================

    See Also
    --------
    porosimetry
    drainage

    References
    ----------
    .. [3] Gostick JT, Misaghian N, Yang J, Boek ES. Simulating volume-controlled
       invasion of a non-wetting fluid in volumetric images using basic image
       processing tools. Computers & Geosciences. 158(1), 104978 (2022).
       `Link. <https://doi.org/10.1016/j.cageo.2021.104978>`__

    Notes
    -----
    This function is slower and is less capable than ``qbip``, which returns
    identical results, so it is recommended to use that instead.

    """
    # Process the boundary image
    if inlets is None:
        inlets = np.zeros_like(im)
        inlets[0, ...] = True
    inlets = inlets*im
    if maxiter is None:
        maxiter = im.sum()
    bd = np.copy(inlets > 0)
    if dt is None:  # Find dt if not given
        dt = edt(im)
    # Initialize inv image with -1 in the solid, and 0's in the void
    seq = -1*(~im)
    sizes = -1.0*(~im)
    desc = inspect.currentframe().f_code.co_name  # Get current func name
    for step in tqdm(range(1, maxiter), desc=desc, **settings.tqdm):
        # Find insertion points
        edge = bd*(dt > 0)
        if ~edge.any():
            break
        # Find the maximum value of the dt underlaying the new edge
        r_max = (dt*edge).max()
        # Find all values of the dt with that size
        dt_thresh = dt >= r_max
        # Extract the actual coordinates of the insertion sites
        pt = _where(edge*dt_thresh)
        seq = _insert_disk_at_points(
            im=seq,
            coords=pt,
            r=int(r_max),
            v=step,
            smooth=True,
        )
        if return_sizes:
            sizes = _insert_disk_at_points(
                im=sizes,
                coords=pt,
                r=int(r_max),
                v=r_max,
                smooth=True,
            )
        dt, bd = _update_dt_and_bd(dt, bd, pt)
        # Add neighbors of current points to bd image
        bd = _insert_disk_at_points(
            im=bd,
            coords=pt,
            r=1 if conn == 'min' else 2,
            v=1,
            smooth=False if conn == 'min' else True,
        )
    # Convert inv image so that uninvaded voxels are set to -1 and solid to 0
    temp = seq == 0  # Uninvaded voxels are set to -1 after _ibip
    seq[~im] = 0
    seq[temp] = -1
    seq = make_contiguous(im=seq, mode='symmetric')
    # Deal with invasion sizes similarly
    temp = sizes == 0
    sizes[~im] = 0
    sizes[temp] = -1

    # Deal with trapping if outlets were specified
    if outlets is not None:
        logger.info('Computing trapping and adjusting outputs')
        trapped = find_trapped_clusters(
            im=im,
            seq=seq,
            outlets=outlets,
            conn=conn,
            method='queue',
        )
        if min_size > 0:
            temp = find_small_clusters(
                im=im,
                trapped=trapped,
                min_size=min_size,
                conn=conn,
            )
            trapped = temp.im_trapped
        seq[trapped] = -1
        seq = make_contiguous(im=seq, mode='symmetric')
        sizes[trapped] = -1

    results = Results()
    results.im_seq = np.copy(seq)
    results.im_snwp = seq_to_satn(seq=seq, im=im)
    if return_sizes:
        results.im_size = np.copy(sizes)
    return results


@njit()
def _update_dt_and_bd(dt, bd, pt):
    if dt.ndim == 2:
        for i in range(pt.shape[1]):
            bd[pt[0, i], pt[1, i]] = True
            dt[pt[0, i], pt[1, i]] = 0
    else:
        for i in range(pt.shape[1]):
            bd[pt[0, i], pt[1, i], pt[2, i]] = True
            dt[pt[0, i], pt[1, i], pt[2, i]] = 0
    return dt, bd


def injection(
    im,
    pc=None,
    dt=None,
    inlets=None,
    outlets=None,
    maxiter=None,
    return_sizes=False,
    return_pressures=True,
    conn='min',
    min_size=0,
    method='qbip',
):
    r"""
    Performs injection of non-wetting fluid including the effect of gravity and
    trapping of wetting phase.

    Parameters
    ----------
    im : ndarray
        A boolean image of the porous media with ``True`` values indicating
        the void space
    pc : ndarray, optional
        Precomputed capillary pressure transform which is used to determine
        the invadability of each voxel. If not provided then the ``2/dt`` is used,
        which is equivalent to a surface tension and voxel size of unity, and a
        contact angle of 180 degrees.
    dt : ndarray (optional)
        The distance transform of ``im``.  If not provided it will be
        calculated, so supplying it saves time.
    inlets : ndarray, optional
        A boolean image with ``True`` values indicating the inlet locations.
        If not provided then the beginning of the x-axis is assumed.
    outlets : ndarray, optional
        A boolean image with ``True`` values indicating the outlet locations.
        If this is provided then trapped voxels of wetting phase are found and
        all the output images are adjusted accordingly. Note that trapping can
        be assessed during postprocessing as well.
    return_sizes : bool, default = `False`
        If ``True`` then an array containing the size of the sphere which first
        overlapped each pixel is returned. This array is not computed by default
        to save computation time.
    return_pressures : bool, default = ``True``
        If ``True`` then an array containing the capillary pressure at which
        each pixels was first invaded is returned.
    maxiter : int
        The maximum number of iteration to perform.  The default is equal to the
        number of void pixels in ``im``.
    min_size : int
        Any clusters of trapped voxels smaller than this size will be set to not
        trapped. This argument is only used if ``outlets`` is given. This is useful
        to prevent small voxels along edges of the void space from being set to
        trapped. These can appear to be trapped due to the jagged nature of the
        digital image. The default is 0, meaning this adjustment is not applied,
        but a value of 3 or 4 is recommended to activate this adjustment.
    conn : str
        Controls the shape of the structuring element used to find neighboring
        voxels.  Options are:

        ========= ==================================================================
        Option    Description
        ========= ==================================================================
        'min'     This corresponds to a cross with 4 neighbors in 2D and 6 neighbors
                  in 3D.
        'max'     This corresponds to a square or cube with 8 neighbors in 2D and
                  26 neighbors in 3D.
        ========= ==================================================================

    method : str
        Controls the method used perform the simulation.  Options are:

        ========= ==================================================================
        Option    Description
        ========= ==================================================================
        'qbip'    Uses 'queue-based invasion percolation' [1]_. This is the default.
                  It is much faster.
        'ibip'    Uses 'image-based invasion percolation' [2]_. This is only
                  provided for completeness since it is the original algorithm.
        ========= ==================================================================

    Returns
    -------
    results : Results object
        A dataclass-like object with the following attributes:

        ========== =================================================================
        Attribute  Description
        ========== =================================================================
        im_seq     A numpy array with each voxel value containing the step at
                   which it was invaded.  Uninvaded voxels are set to -1.
        im_snwp    A numpy array with each voxel value indicating the saturation
                   present in the domain it was invaded. Solids are given 0, and
                   uninvaded regions are given -1.
        im_pc      If ``return_pressures`` was set to ``True``, then a numpy array
                   with each voxel value indicating the capillary pressure at which
                   it was invaded. Uninvaded voxels have value of ``np.inf``.
        im_size    If ``return_sizes`` was set to ``True``, then a numpy array with
                   each voxel containing the radius of the sphere, in voxels, that
                   first overlapped it.
        ========== =================================================================

    References
    ----------
    .. [1] Gostick JT, Misaghian N, A Irannezhad, B Zhao. *A computationally
       efficient queue-based algorithm for simulating volume-controlled drainage
       under the influence of gravity on volumetric images*. `Advances in Water
       Resources <https://doi.org/10.1016/j.advwatres.2024.104799>`__. 193(11),
       104799 (2024)

    .. [2] Gostick JT, Misaghian N, Yang J, Boek ES. *Simulating volume-controlled
       invasion of a non-wetting fluid in volumetric images using basic image
       processing tools*. `Computers and the Geosciences
       <https://doi.org/10.1016/j.cageo.2021.104978>`__. 158(1), 104978 (2022)

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/injection.html>`__
    to view an online example.

    """
    if method == 'qbip':
        results = qbip(
            im=im,
            pc=pc,
            dt=dt,
            inlets=inlets,
            outlets=outlets,
            maxiter=maxiter,
            return_sizes=return_sizes,
            return_pressures=return_pressures,
            conn=conn,
            min_size=min_size,
        )
    elif method == 'ibip':
        results = ibip(
            im=im,
            dt=dt,
            inlets=inlets,
            outlets=outlets,
            maxiter=maxiter,
            return_sizes=return_sizes,
            conn=conn,
            min_size=min_size,
        )
    return results


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    import porespy as ps
    from porespy.simulations import drainage

    ps.settings.tqdm['disable'] = False
    ps.settings.tqdm['leave'] = True

    # %%
    im = ~ps.generators.random_spheres([60, 60], r=5, seed=0, clearance=4)

    inlets = np.zeros_like(im)
    inlets[0, ...] = True
    inlets = inlets*im
    pc = ps.filters.capillary_transform(im, voxel_size=1e-5)

    drn = drainage(im, pc=pc, inlets=inlets)
    inv1 = injection(im, pc=pc, inlets=inlets, return_sizes=True, method='qbip', conn='min', min_size=1)
    inv2 = injection(im, inlets=inlets, return_sizes=True, method='ibip', conn='min', min_size=1)

    # %%
    drn_data = ps.metrics.pc_map_to_pc_curve(im=im, pc=drn.im_pc, fix_ends=False, mode='drainage')
    im_pc1 = ps.filters.capillary_transform(im=im, dt=inv1.im_size, sigma=1.0, voxel_size=1e-5)
    inj_data = ps.metrics.pc_map_to_pc_curve(im=im, pc=inv1.im_pc, seq=inv1.im_seq, fix_ends=False, mode='drainage')
    im_pc2 = ps.filters.capillary_transform(im=im, dt=inv2.im_size, sigma=1.0, voxel_size=1e-5)
    ibip_data = ps.metrics.pc_map_to_pc_curve(im=im, pc=im_pc2, seq=inv2.im_seq, fix_ends=False, mode='drainage')

    fig, ax = plt.subplots()
    ax.step(np.log10(drn_data.pc), drn_data.snwp, where='post', linewidth=.5)
    ax.step(np.log10(inj_data.pc), inj_data.snwp, where='post', linewidth=1)
    ax.step(np.log10(ibip_data.pc), ibip_data.snwp, where='post', linewidth=.5)
