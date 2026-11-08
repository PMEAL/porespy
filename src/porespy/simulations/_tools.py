from numba import njit
from porespy.tools import (
    _get_sphere_indices as _get_flat_indices,  # noqa: F401 - re-exported helper
    _find_sphere_interface as _find_interface,  # noqa: F401 - re-exported helper
    _insert_disks_at_indices_parallel as _insert_disks_at_indices_parallel,
    _insert_disks_at_indices_parallel_direct as _insert_disks_at_indices_parallel_direct,
    _insert_disks_at_indices_parallel_merged as _insert_disks_at_indices_parallel_merged,
    _use_merged_intervals as _use_merged_intervals,
)


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
