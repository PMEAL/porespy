import numpy as np
from numba import njit, prange


def _make_axial_extent_lookup(max_radius):
    squared_distance = np.arange(int(max_radius)**2 + 1)
    return np.ceil(np.sqrt(squared_distance)).astype(np.int32)


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


@njit(inline="always")
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


@njit(parallel=True)
def _insert_disks_at_indices_parallel(
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
                for y in range(
                    max(0, j - y_extent),
                    min(j + y_extent + 1, ylim),
                ):
                    if overwrite or not im[x, y]:
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
                    for z in range(
                        max(0, k - z_extent),
                        min(k + z_extent + 1, zlim),
                    ):
                        if overwrite or not im[x, y, z]:
                            im[x, y, z] = True
    return im
