import numpy as np
from numba import njit, prange
from porespy.tools import (
    _insert_disks_at_indices_parallel as _insert_disks_at_indices_parallel,
    _insert_disks_at_indices_parallel_direct as _insert_disks_at_indices_parallel_direct,
    _insert_disks_at_indices_parallel_merged as _insert_disks_at_indices_parallel_merged,
    _use_merged_intervals as _use_merged_intervals,
)


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
