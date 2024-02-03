import porespy as ps
import numpy as np
import matplotlib.pyplot as plt
from porespy import beta
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
        pts = ~ps.generators.lattice_spheres(new_shape, r=1, spacing=spacing, offset=0)
    elif lattice.startswith('t'):
        pts = ~ps.generators.lattice_spheres(shape=new_shape, r=1, spacing=spacing, offset=0)
    if Rmax is None:
        Rmax = int(spacing/2)
    labels = spim.label(pts)[0]
    tmp = np.zeros_like(pts)
    slices = spim.find_objects(labels)
    for s in slices:
        sx = ps.tools.extend_slice(
            slices=s,
            shape=pts.shape,
            pad=[np.random.randint(Rmin, Rmax), spacing],
        )
        tmp[sx] = True
        sx = ps.tools.extend_slice(
            slices=s,
            shape=pts.shape,
            pad=[spacing, np.random.randint(Rmin, Rmax)],
        )
        tmp[sx] = True
    if lattice.startswith('s'):
        if truncate:
            end = shape
        else:
            end = (np.ceil(shape/spacing)*spacing).astype(int) + 1
        tmp = tmp[:end[0], :end[1]]
        pts = pts[:end[0], :end[1]]
    if lattice.startswith('t'):
        tmp = spim.rotate(tmp, -45, order=0, reshape=False)
        pts = spim.rotate(pts*1.0, -45, order=5, reshape=False) > 0.4
        a, b = (new_shape/2).astype(int)
        s = (slice(a, a+1, None), slice(b, b+1, None))
        if truncate:
            a, b = (shape/2).astype(int)
        else:
            diag = np.around((spacing**2 + spacing**2)**0.5).astype(int)
            a, b = (np.ceil(shape/diag)*diag/2).astype(int)
        sx = ps.tools.extend_slice(slices=s, shape=pts.shape, pad=[a, b])
        tmp = tmp[sx]
        pts = pts[sx]
    return tmp, pts


if __name__ == "__main__":

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
