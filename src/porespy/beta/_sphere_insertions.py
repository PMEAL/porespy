import numpy as np

from porespy.tools import (
    _insert_disk_at_points,
    _insert_disk_at_points_parallel,
    get_edt,
)

__all__ = [
    'optimized_sphere_insertion_serial',
    'optimized_sphere_insertion_parallel',
]


edt = get_edt()


def optimized_sphere_insertion_serial(im, r, dt=None):
    if dt is None:
        dt = edt(im)
    dt = dt.astype(int)
    mask = dt == r
    crds = np.vstack(np.where(mask))
    sph = np.zeros_like(im, dtype=bool)
    sph = _insert_disk_at_points(
        im=sph,
        coords=crds,
        r=r,
        v=True,
        overwrite=False,
    )
    sph += dt >= r
    return sph


def optimized_sphere_insertion_parallel(im, r, dt=None):
    if dt is None:
        dt = edt(im)
    dt = dt.astype(int)
    mask = dt == r
    crds = np.vstack(np.where(mask))
    sph = np.zeros_like(im, dtype=bool)
    sph = _insert_disk_at_points_parallel(
        im=sph,
        coords=crds,
        r=r,
        v=True,
        overwrite=False,
    )
    sph += dt >= r
    return sph
