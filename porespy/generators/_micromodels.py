import numpy as np
import matplotlib.pyplot as plt
from nanomesh import Mesher2D
from porespy.generators import lattice_spheres, borders, spheres_from_coords
from porespy.tools import _insert_disks_at_points_parallel
from porespy.tools import extend_slice, extract_subsection
import scipy.ndimage as spim
import scipy.stats as spst
from typing import List


__all__ = [
    'rectangular_pillar_array',
    'cylindrical_pillar_array',
    'random_cylindrical_pillars',
]


def _extract(im, shape, spacing, truncate, lattice):
    if lattice.startswith('s'):
        if truncate:
            end = shape
        else:
            end = (np.ceil(shape/spacing)*spacing).astype(int) + 1
        im = im[:end[0], :end[1]]
    if lattice.startswith('t'):
        new_shape = np.array(im.shape)
        im = spim.rotate(im, -45, order=0, reshape=False)
        a, b = (new_shape/2).astype(int)
        s = (slice(a, a+1, None), slice(b, b+1, None))
        if truncate:
            a, b = (shape/2).astype(int)
        else:
            diag = np.around((spacing**2 + spacing**2)**0.5).astype(int)
            a, b = (np.ceil(shape/diag)*diag/2).astype(int)
        sx = extend_slice(slices=s, shape=im.shape, pad=[a, b])
        im = im[sx]
    return im


def rectangular_pillar_array(
    shape: List,
    spacing: int = 40,
    dist: str = 'uniform',
    dist_kwargs: dict = {'loc': 5, 'scale': 10},
    lattice: str = 'sc',
    truncate: bool = True,
    seed: int = None,
):
    if seed is not None:
        np.random.seed(seed)
    if isinstance(dist, str):
        f = getattr(spst, dist)(**dist_kwargs)
    shape = np.array(shape)
    new_shape = (np.ones_like(shape)*shape.max()*2).astype(int)
    if lattice.startswith('s'):
        pts = ~lattice_spheres(new_shape, r=1, spacing=spacing, offset=0)
    elif lattice.startswith('t'):
        pts = ~lattice_spheres(shape=new_shape, r=1, spacing=spacing, offset=0)
    labels = spim.label(pts)[0]
    tmp = np.zeros_like(pts)
    slices = spim.find_objects(labels)
    for s in slices:
        sx = extend_slice(
            slices=s,
            shape=pts.shape,
            pad=[np.around(f.rvs()).astype(int), spacing],
        )
        tmp[sx] = True
        sx = extend_slice(
            slices=s,
            shape=pts.shape,
            pad=[spacing, np.around(f.rvs()).astype(int)],
        )
        tmp[sx] = True
    tmp = _extract(tmp, shape, spacing, truncate, lattice)
    pts = _extract(pts, shape, spacing, truncate, lattice)
    return tmp


def cylindrical_pillar_array(
    shape: List,
    spacing: int = 40,
    dist: str = 'uniform',
    dist_kwargs: dict = {'loc': 5, 'scale': 10},
    lattice: str = 'sc',
    truncate: bool = True,
    seed: int = None,
):
    if seed is not None:
        np.random.seed(seed)
    if isinstance(dist, str):
        f = getattr(spst, dist)(**dist_kwargs)
    shape = np.array(shape)
    new_shape = (np.ones_like(shape)*shape.max()*2).astype(int)
    if lattice.startswith('s'):
        pts = ~lattice_spheres(new_shape, r=1, spacing=spacing, offset=0)
    elif lattice.startswith('t'):
        pts = ~lattice_spheres(new_shape, r=1, spacing=spacing, offset=0)
    coords = np.vstack(np.where(pts))
    radii = f.rvs(pts.sum())
    tmp = np.ones_like(pts, dtype=int)
    tmp = _insert_disks_at_points_parallel(
        im=tmp, coords=coords, radii=radii, v=0, smooth=True, overwrite=True)
    if lattice.startswith('s'):
        if truncate:
            end = shape
        else:
            end = (np.ceil(shape/spacing)*spacing).astype(int) + 1
        tmp = tmp[:end[0], :end[1]]
        pts = pts[:end[0], :end[1]]
    tmp = _extract(tmp, shape, spacing, truncate, lattice)
    pts = _extract(pts, shape, spacing, truncate, lattice)
    return tmp


def random_cylindrical_pillars(
    shape: list,
    f: float = 0.75,
    a: int = 1000,
    n: int = 30,
    truncate : bool = True,
    seed: int = None,
):
    r"""
    A 2D micromodel with randomly located cylindrical pillars of random radius

    Parameter
    ---------
    shape : array_like
        The X, Y size of the desired image in pixels
    f : scalar
        A factor to control the relative size of the pillars. `f = 1` results in
        pillars that just touch each other, while `f < 1` will add more space
        between the pillars
    a : scalar
        Controls the number of pillars in the image, with a small value giving
        more pillars. The default is 1500.  Technically this parameter sets the
        minimum area for each triangle in the mesh.
    n : scalar
        The maximum distance between cylinders on the edges of the image. This
        controls the number and size of the cylinders along the edges.  The default
        is 50, but this must be adjusted when
    """
    if seed is not None:
        np.random.seed(seed)
    if len(shape) != 2:
        raise Exception("Shape must be 2D")
    im = np.ones(shape, dtype=float)
    bd = borders(im.shape, mode='faces')
    im[bd] = 0.0

    mesher = Mesher2D(im)
    mesher.generate_contour(max_edge_dist=n/f)

    mesh = mesher.triangulate(opts=f'q0a{a}ne')
    # mesh.plot_pyvista(jupyter_backend='static', show_edges=True)
    tri = mesh.triangle_dict

    r_max = np.inf*np.ones([tri['vertices'].shape[0], ])
    for e in tri['edges']:
        L = np.sqrt(np.sum(np.diff(tri['vertices'][e], axis=0)**2))
        if L/2 > 1:
            r_max[e[0]] = min(r_max[e[0]], L/2)
            r_max[e[1]] = min(r_max[e[1]], L/2)

    mask = np.ravel(tri['vertex_markers'] >= 0)
    r = f*(r_max[mask])

    coords = tri['vertices'][mask]
    coords = np.pad(
        array=coords,
        pad_width=((0, 0), (0, 1)),
        mode='constant',
        constant_values=0,
    )
    coords = np.vstack((coords.T, r)).T
    if truncate:
        im_w_spheres = spheres_from_coords(coords, smooth=True, mode='extended')
    else:
        im_w_spheres = spheres_from_coords(coords, smooth=True, mode='contained')
    return im_w_spheres


if __name__ == "__main__":

    rect_demo = False
    cyl_demo = False
    rand_cyl = True

    if rect_demo:
        fig, ax = plt.subplots(2, 2)
        np.random.seed(0)
        im1 = rectangular_pillar_array(
            shape=[400, 600], spacing=40, lattice='simple', truncate=True)
        im2 = rectangular_pillar_array(
            shape=[400, 600], spacing=40, lattice='tri', truncate=True)

        ax[0][0].imshow(im1, origin='lower', interpolation='none')
        ax[0][1].imshow(im2, origin='lower', interpolation='none')

        np.random.seed(0)
        im1 = rectangular_pillar_array(
            shape=[400, 600], spacing=40, lattice='simple', truncate=False)
        im2 = rectangular_pillar_array(
            shape=[400, 600], spacing=40, lattice='tri', truncate=False)

        ax[1][0].imshow(im1, origin='lower', interpolation='none')
        ax[1][1].imshow(im2, origin='lower', interpolation='none')

    if cyl_demo:
        fig, ax = plt.subplots(2, 2)
        np.random.seed(0)
        im1 = cylindrical_pillar_array(
            shape=[400, 600], spacing=40, lattice='simple', truncate=True)
        im2 = cylindrical_pillar_array(
            shape=[400, 600], spacing=40, lattice='tri', truncate=True)
        ax[0][0].imshow(im1, origin='lower', interpolation='none')
        ax[0][1].imshow(im2, origin='lower', interpolation='none')

        np.random.seed(0)
        im1 = cylindrical_pillar_array(
            shape=[400, 600], spacing=40, lattice='simple', truncate=False)
        im2 = cylindrical_pillar_array(
            shape=[400, 600], spacing=40, lattice='tri', truncate=False)
        ax[1][0].imshow(im1, origin='lower', interpolation='none')
        ax[1][1].imshow(im2, origin='lower', interpolation='none')

    if rand_cyl:
        im = random_cylindrical_pillars(
            shape=[2500, 1500],
            f=.7,
            a=1000,
            n=50,
        )
        plt.imshow(im)
