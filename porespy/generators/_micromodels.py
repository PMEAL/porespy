import numpy as np
import matplotlib.pyplot as plt
from porespy.generators import lattice_spheres
from porespy.tools import _insert_disks_at_points_parallel
from porespy.tools import extend_slice, extract_subsection
import scipy.ndimage as spim
from typing import List


__all__ = [
    'rectangular_pillars',
]


def rectangular_pillars(
    shape: List,
    spacing: List,
    dist=None,
    Rmin: int = 5,
    Rmax: int = None,
    lattice: str = 'sc',
    truncate: bool = True,
):
    shape = np.array(shape)
    new_shape = (np.ones_like(shape)*shape.max()*2).astype(int)
    if lattice.startswith('s'):
        pts = ~lattice_spheres(new_shape, r=1, spacing=spacing, offset=0)
    elif lattice.startswith('t'):
        pts = ~lattice_spheres(shape=new_shape, r=1, spacing=spacing, offset=0)
    if Rmax is None:
        Rmax = int(spacing/2)
    labels = spim.label(pts)[0]
    tmp = np.zeros_like(pts)
    slices = spim.find_objects(labels)
    for s in slices:
        sx = extend_slice(
            slices=s,
            shape=pts.shape,
            pad=[np.random.randint(Rmin, Rmax), spacing],
        )
        tmp[sx] = True
        sx = extend_slice(
            slices=s,
            shape=pts.shape,
            pad=[spacing, np.random.randint(Rmin, Rmax)],
        )
        tmp[sx] = True
    tmp, pts = _extract([tmp, pts], shape, spacing, truncate, lattice)
    return tmp, pts


def _extract(ims, shape, spacing, truncate, lattice):
    if lattice.startswith('s'):
        if truncate:
            end = shape
        else:
            end = (np.ceil(shape/spacing)*spacing).astype(int) + 1
        for i in range(len(ims)):
            ims[i] = ims[i][:end[0], :end[1]]
    if lattice.startswith('t'):
        new_shape = np.array(ims[0].shape)
        for i in range(len(ims)):
            ims[i] = spim.rotate(ims[i], -45, order=0, reshape=False)
        a, b = (new_shape/2).astype(int)
        s = (slice(a, a+1, None), slice(b, b+1, None))
        if truncate:
            a, b = (shape/2).astype(int)
        else:
            diag = np.around((spacing**2 + spacing**2)**0.5).astype(int)
            a, b = (np.ceil(shape/diag)*diag/2).astype(int)
        sx = extend_slice(slices=s, shape=ims[0].shape, pad=[a, b])
        for i in range(len(ims)):
            ims[i] = ims[i][sx]
    return ims


def cylindrical_pillars(shape, spacing, Rmin=5, Rmax=None, lattice='sc', truncate=True):
    if Rmax is None:
        Rmax = int(spacing/2)
    shape = np.array(shape)
    new_shape = (np.ones_like(shape)*shape.max()*2).astype(int)
    if lattice.startswith('s'):
        pts = ~lattice_spheres(new_shape, r=1, spacing=spacing, offset=0)
    elif lattice.startswith('t'):
        pts = ~lattice_spheres(new_shape, r=1, spacing=spacing, offset=0)
    coords = np.vstack(np.where(pts))
    radii = np.random.randint(Rmin, Rmax, pts.sum())
    tmp = np.zeros_like(pts, dtype=int)
    tmp = _insert_disks_at_points_parallel(
        im=tmp, coords=coords, radii=radii, v=1, smooth=True)
    if lattice.startswith('s'):
        if truncate:
            end = shape
        else:
            end = (np.ceil(shape/spacing)*spacing).astype(int) + 1
        tmp = tmp[:end[0], :end[1]]
        pts = pts[:end[0], :end[1]]
    tmp, pts = _extract([tmp, pts], shape, spacing, truncate, lattice)
    return tmp, pts


if __name__ == "__main__":
    rect_demo = False
    cyl_demo = True
    if rect_demo:
        fig, ax = plt.subplots(2, 2)
        np.random.seed(0)
        im1, pts1 = rectangular_pillars(
            shape=[400, 600], Rmin=2, Rmax=15, spacing=40, lattice='simple', truncate=True)
        im2, pts2 = rectangular_pillars(
            shape=[400, 600], Rmin=2, Rmax=15, spacing=40, lattice='tri', truncate=True)

        ax[0][0].imshow(im1 + 2.0*pts1, origin='lower', interpolation='none')
        ax[0][1].imshow(im2 + 2.0*pts2, origin='lower', interpolation='none')

        np.random.seed(0)
        im1, pts1 = rectangular_pillars(
            shape=[400, 600], Rmin=2, Rmax=15, spacing=40, lattice='simple', truncate=False)
        im2, pts2 = rectangular_pillars(
            shape=[400, 600], Rmin=2, Rmax=15, spacing=40, lattice='tri', truncate=False)

        ax[1][0].imshow(im1 + 2.0*pts1, origin='lower', interpolation='none')
        ax[1][1].imshow(im2 + 2.0*pts2, origin='lower', interpolation='none')
    if cyl_demo:
        fig, ax = plt.subplots(1, 2)
        np.random.seed(0)
        im1, pts1 = cylindrical_pillars(
            shape=[400, 600], Rmin=5, Rmax=15, spacing=40, lattice='simple', truncate=False)
        im2, pts2 = cylindrical_pillars(
            shape=[400, 600], Rmin=5, Rmax=15, spacing=40, lattice='tri', truncate=False)
        ax[0].imshow(im1 + 2.0*pts1, origin='lower', interpolation='none')
        ax[1].imshow(im2 + 2.0*pts2, origin='lower', interpolation='none')
