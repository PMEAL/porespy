import numpy as np
from numba import get_num_threads, get_thread_id, njit, prange
from porespy.tools import _get_axial_extent, _make_axial_extent_lookup


def _get_flat_indices(mask):
    dtype = np.int32 if mask.size <= np.iinfo(np.int32).max else np.int64
    return np.flatnonzero(mask).astype(dtype, copy=False)


@njit
def _remove_contained_disks(indices, eligible, dt):
    """Remove centers dominated by an eligible axial neighbor in place."""
    count = 0
    if eligible.ndim == 2:
        xlim, ylim = eligible.shape
        for q in range(len(indices)):
            ind = indices[q]
            i = ind // ylim
            j = ind - i * ylim
            required_radius = int(dt[i, j]) + 1
            contained = (
                (i > 0 and eligible[i - 1, j]
                 and int(dt[i - 1, j]) >= required_radius)
                or (i + 1 < xlim and eligible[i + 1, j]
                    and int(dt[i + 1, j]) >= required_radius)
                or (j > 0 and eligible[i, j - 1]
                    and int(dt[i, j - 1]) >= required_radius)
                or (j + 1 < ylim and eligible[i, j + 1]
                    and int(dt[i, j + 1]) >= required_radius)
            )
            if not contained:
                indices[count] = ind
                count += 1
    elif eligible.ndim == 3:
        xlim, ylim, zlim = eligible.shape
        stride0 = ylim * zlim
        for q in range(len(indices)):
            ind = indices[q]
            i = ind // stride0
            rem = ind - i * stride0
            j = rem // zlim
            k = rem - j * zlim
            required_radius = int(dt[i, j, k]) + 1
            contained = (
                (i > 0 and eligible[i - 1, j, k]
                 and int(dt[i - 1, j, k]) >= required_radius)
                or (i + 1 < xlim and eligible[i + 1, j, k]
                    and int(dt[i + 1, j, k]) >= required_radius)
                or (j > 0 and eligible[i, j - 1, k]
                    and int(dt[i, j - 1, k]) >= required_radius)
                or (j + 1 < ylim and eligible[i, j + 1, k]
                    and int(dt[i, j + 1, k]) >= required_radius)
                or (k > 0 and eligible[i, j, k - 1]
                    and int(dt[i, j, k - 1]) >= required_radius)
                or (k + 1 < zlim and eligible[i, j, k + 1]
                    and int(dt[i, j, k + 1]) >= required_radius)
            )
            if not contained:
                indices[count] = ind
                count += 1
    return indices[:count]


@njit(parallel=True)
def _find_interface(mask, interface):  # pragma: no cover
    """Find foreground voxels touching in-bounds background axially."""
    if mask.ndim == 2:
        xlim, ylim = mask.shape
        for i in prange(xlim):
            for j in range(ylim):
                interface[i, j] = mask[i, j] and (
                    (i > 0 and not mask[i - 1, j])
                    or (i + 1 < xlim and not mask[i + 1, j])
                    or (j > 0 and not mask[i, j - 1])
                    or (j + 1 < ylim and not mask[i, j + 1])
                )
    elif mask.ndim == 3:
        xlim, ylim, zlim = mask.shape
        for i in prange(xlim):
            for j in range(ylim):
                for k in range(zlim):
                    interface[i, j, k] = mask[i, j, k] and (
                        (i > 0 and not mask[i - 1, j, k])
                        or (i + 1 < xlim and not mask[i + 1, j, k])
                        or (j > 0 and not mask[i, j - 1, k])
                        or (j + 1 < ylim and not mask[i, j + 1, k])
                        or (k > 0 and not mask[i, j, k - 1])
                        or (k + 1 < zlim and not mask[i, j, k + 1])
                    )
    return interface


def _insert_disks_at_indices_parallel(
    im,
    indices,
    dt,
    ceil_distance,
    smooth=True,
    overwrite=False,
):  # pragma: no cover
    if overwrite and _use_merged_intervals(im, indices, dt):
        return _insert_disks_at_indices_parallel_merged(
            im=im,
            indices=indices,
            dt=dt,
            ceil_distance=ceil_distance,
            smooth=smooth,
        )
    return _insert_disks_at_indices_parallel_direct(
        im=im,
        indices=indices,
        dt=dt,
        ceil_distance=ceil_distance,
        smooth=smooth,
        overwrite=overwrite,
    )


@njit
def _use_merged_intervals(im, indices, dt):
    """Sample sphere sizes to choose between direct and merged scan-line writes."""
    if len(indices) == 0:
        return False
    nsamples = min(len(indices), 256)
    estimated_intervals = 0
    if im.ndim == 2:
        ylim = im.shape[1]
        for q in range(nsamples):
            ind = indices[q * len(indices) // nsamples]
            i = ind // ylim
            j = ind - i * ylim
            estimated_intervals += 2 * int(dt[i, j]) + 1
        nrows = im.shape[0]
    else:
        ylim, zlim = im.shape[1:]
        stride0 = ylim * zlim
        for q in range(nsamples):
            ind = indices[q * len(indices) // nsamples]
            i = ind // stride0
            rem = ind - i * stride0
            j = rem // zlim
            k = rem - j * zlim
            diameter = 2 * int(dt[i, j, k]) + 1
            estimated_intervals += diameter**2
        nrows = im.shape[0] * im.shape[1]
    # Row buffers and their initialization cost more than direct writes for
    # sparse disks. Benchmarks place the crossover near 256 generated intervals
    # per output row, while strongly overlapping spheres exceed this by orders
    # of magnitude.
    return estimated_intervals * len(indices) >= 256 * nrows * nsamples


@njit(parallel=True)
def _insert_disks_at_indices_parallel_direct(
    im,
    indices,
    dt,
    ceil_distance,
    smooth=True,
    overwrite=False,
):  # pragma: no cover
    npts = len(indices)
    if im.ndim == 2:
        xlim, ylim = im.shape
        for q in prange(npts):
            ind = indices[q]
            i = ind // ylim
            j = ind - i * ylim
            r = int(dt[i, j])
            radius_squared = r**2
            for x in range(max(0, i - r), min(i + r + 1, xlim)):
                dx = x - i
                y_extent = _get_axial_extent(
                    radius_squared - dx**2,
                    ceil_distance,
                    smooth,
                )
                y_start = max(0, j - y_extent)
                y_stop = min(j + y_extent + 1, ylim)
                if overwrite:
                    im[x, y_start:y_stop] = True
                else:
                    for y in range(y_start, y_stop):
                        if not im[x, y]:
                            im[x, y] = True
    elif im.ndim == 3:
        xlim, ylim, zlim = im.shape
        stride0 = ylim * zlim
        for q in prange(npts):
            ind = indices[q]
            i = ind // stride0
            rem = ind - i * stride0
            j = rem // zlim
            k = rem - j * zlim
            r = int(dt[i, j, k])
            radius_squared = r**2
            for x in range(max(0, i - r), min(i + r + 1, xlim)):
                dx = x - i
                yz_extent = _get_axial_extent(
                    radius_squared - dx**2,
                    ceil_distance,
                    smooth,
                )
                for y in range(
                    max(0, j - yz_extent),
                    min(j + yz_extent + 1, ylim),
                ):
                    dy = y - j
                    z_extent = _get_axial_extent(
                        radius_squared - dx**2 - dy**2,
                        ceil_distance,
                        smooth,
                    )
                    z_start = max(0, k - z_extent)
                    z_stop = min(k + z_extent + 1, zlim)
                    if overwrite:
                        im[x, y, z_start:z_stop] = True
                    else:
                        for z in range(z_start, z_stop):
                            if not im[x, y, z]:
                                im[x, y, z] = True
    return im


@njit(parallel=True)
def _insert_disks_at_indices_parallel_merged(
    im,
    indices,
    dt,
    ceil_distance,
    smooth=True,
):  # pragma: no cover
    """Insert disks by merging consecutive overlapping scan-line intervals."""
    nthreads = get_num_threads()
    if im.ndim == 2:
        xlim, ylim = im.shape
        starts = np.full((nthreads, xlim), ylim, dtype=np.int32)
        stops = np.zeros((nthreads, xlim), dtype=np.int32)
        for q in prange(len(indices)):
            thread = get_thread_id()
            ind = indices[q]
            i = ind // ylim
            j = ind - i * ylim
            r = int(dt[i, j])
            radius_squared = r**2
            for x in range(max(0, i - r), min(i + r + 1, xlim)):
                dx = x - i
                extent = _get_axial_extent(
                    radius_squared - dx**2,
                    ceil_distance,
                    smooth,
                )
                start = max(0, j - extent)
                stop = min(j + extent + 1, ylim)
                if start >= stop:
                    continue
                old_start = starts[thread, x]
                old_stop = stops[thread, x]
                if old_start == ylim:
                    starts[thread, x] = start
                    stops[thread, x] = stop
                elif (start <= old_stop) and (stop >= old_start):
                    starts[thread, x] = min(start, old_start)
                    stops[thread, x] = max(stop, old_stop)
                else:
                    im[x, old_start:old_stop] = True
                    starts[thread, x] = start
                    stops[thread, x] = stop
        for q in prange(nthreads * xlim):
            thread = q // xlim
            x = q - thread * xlim
            start = starts[thread, x]
            if start < ylim:
                im[x, start:stops[thread, x]] = True
    elif im.ndim == 3:
        xlim, ylim, zlim = im.shape
        stride0 = ylim * zlim
        nrows = xlim * ylim
        starts = np.full((nthreads, nrows), zlim, dtype=np.int32)
        stops = np.zeros((nthreads, nrows), dtype=np.int32)
        for q in prange(len(indices)):
            thread = get_thread_id()
            ind = indices[q]
            i = ind // stride0
            rem = ind - i * stride0
            j = rem // zlim
            k = rem - j * zlim
            r = int(dt[i, j, k])
            radius_squared = r**2
            for x in range(max(0, i - r), min(i + r + 1, xlim)):
                dx = x - i
                yz_extent = _get_axial_extent(
                    radius_squared - dx**2,
                    ceil_distance,
                    smooth,
                )
                for y in range(
                    max(0, j - yz_extent),
                    min(j + yz_extent + 1, ylim),
                ):
                    dy = y - j
                    z_extent = _get_axial_extent(
                        radius_squared - dx**2 - dy**2,
                        ceil_distance,
                        smooth,
                    )
                    start = max(0, k - z_extent)
                    stop = min(k + z_extent + 1, zlim)
                    if start >= stop:
                        continue
                    row = x * ylim + y
                    old_start = starts[thread, row]
                    old_stop = stops[thread, row]
                    if old_start == zlim:
                        starts[thread, row] = start
                        stops[thread, row] = stop
                    elif (start <= old_stop) and (stop >= old_start):
                        starts[thread, row] = min(start, old_start)
                        stops[thread, row] = max(stop, old_stop)
                    else:
                        im[x, y, old_start:old_stop] = True
                        starts[thread, row] = start
                        stops[thread, row] = stop
        for q in prange(nthreads * nrows):
            thread = q // nrows
            row = q - thread * nrows
            start = starts[thread, row]
            if start < zlim:
                x = row // ylim
                y = row - x * ylim
                im[x, y, start:stops[thread, row]] = True
    return im
