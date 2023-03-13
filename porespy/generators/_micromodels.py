# import porespy as ps
import numpy as np
import scipy.ndimage as spim
import scipy.spatial as sptl
from porespy.tools import ps_rect, ps_round, extend_slice, get_tqdm, Results
from porespy.tools import _insert_disks_at_points
from porespy.generators import lattice_spheres, line_segment
from porespy import settings
import scipy.stats as spst


__all__ = [
    'rectangular_pillars',
]


tqdm = get_tqdm()


def cross(r, t=0):
    cr = np.zeros([2*r+1, 2*r+1], dtype=bool)
    cr[r-t:r+t+1, :] = True
    cr[:, r-t:r+t+1] = True
    return cr


def ex(r, t=0):
    x = np.eye(2*r + 1).astype(bool)
    x += np.fliplr(x)
    x = spim.binary_dilation(x, structure=ps_rect(w=2*t+1, ndim=2))
    return x


def rectangular_pillars(
    shape=[5, 5],
    spacing=30,
    dist=None,
    Rmin=5,
    Rmax=None,
    lattice='sc',
    return_edges=False,
    return_centers=False
):
    r"""
    A 2D micromodel with rectangular pillars arranged on a regular lattice

    Parameters
    ----------
    shape : list
        The number of pillars in the x and y directions.  The size of the of the
        image will be dictated by the ``spacing`` argument.
    spacing : int
        The distance between neighboring pores centers in pixels. Note that if a
        triangular lattice is used the distance is diagonal, meaning the number
        of pixels will be less than the length by $\sqrt{2}$.
    dist : scipy.stats object
        A "frozen" stats object which can be called to produce random variates. For
        instance, ``dist = sp.stats.norm(loc=50, scale=10))`` would return values
        from a distribution with a mean  of 50 pixels and a standard deviation of
        10 pixels. These values are obtained by calling ``dist.rvs()``. To validate
        the distribution use ``plt.hist(dist.rvs(10000))``, which plots a histogram
        of 10,000 values sampled from ``dist``. If ``dist`` is not provided then a
        uniform distribution between ``Rmin`` and ``Rmax`` is used.
    Rmin : int
        The minimum size of the openings between pillars in pixels. This is used as
        a lower limit on the sizes provided by the chosen distribution, ``f``.  The
        default is 5.
    Rmax : int
        The maximum size of the openings between pillars in pixels.  This is used
        as an upper limit on the sizes provided by the chosen distribution.  If
        not provided then ``spacing/2`` is used to ensure no pillars are
        over-written.
    lattice : str
        The type of lattice to use. Options are:

        ======== ===================================================================
        lattice  description
        ======== ===================================================================
        'sc'     A simple cubic lattice where the pillars are aligned vertically and
                 horizontally with the standard grid. In this case the meaning of
                 ``spacing``, ``Rmin`` and ``Rmax`` directly refers to the number of
                 pixels.
        'tri'    A triangular matrix, which is esentially a cubic matrix rotated 45
                 degrees. In this case the mean of ``spacing``, ``Rmin`` and ``Rmax``
                 refer to the length of a pixel.
        ======== ===================================================================

    return_edges : boolean, optional, default is ``False``
        If ``True`` an image of the edges between each pore center is also returned
        along with the micromodel
    return_centers : boolean, optional, default is ``False``
        If ``True`` an image with markers located at each pore center is also
        returned along with the micromodel

    Returns
    -------
    im or ims : ndarray or dataclass
        If ``return_centers`` and ``return_edges`` are both ``False``, then only
        an ndarray of the micromodel is returned.  If either or both are ``True``
        then a ``dataclass-like`` object is return with multiple images attached
        as attributes:

        ========== =================================================================
        attribute  description
        ========== =================================================================
        im         A 2D image whose size is dictated by the number of pillars
                   (given by ``shape``) and the ``spacing`` between them.
        centers    An image the same size as ``im`` with ``True`` values marking
                   the center of each pore body.
        edges      An image the same size as ``im`` with ``True`` values marking
                   the edges connecting the pore centers. Note that the ``centers``
                   have been removed from this image.
        ========== =================================================================

    Examples
    --------
        `Click here
        <https://porespy.org/examples/generators/reference/rectangular_pillars.html>`_
        to view online example.
    """
    # Parse the various input arguments
    if lattice.startswith('s'):
        # This strel is used to dilate edges of lattice
        strel = cross
        if Rmax is None:
            Rmax = spacing/2
        Rmax = Rmax + 1
        lattice = 'sc'  # In case user specified s, sq or square, etc.
    elif lattice.startswith('t'):
        # This strel is used to dilate edges of lattice
        strel = ex
        shape = np.array(shape) - 1
        if Rmax is None:
            Rmax = spacing/2 - 1
        Rmin = int(Rmin*np.sin(np.deg2rad(45)))
        Rmax = int((Rmax-2)*np.sin(np.deg2rad(45)))
        # Spacing for lattice_spheres function used below is based on horiztonal
        # distance between lattice cells, which is the hypotenuse of the diagonal
        # distance between pillars in the final micromodel, so adjust accordingly
        spacing = int(np.sqrt(spacing**2 + spacing**2))
        lattice = 'tri'  # In case user specified t, or triangle, etc.
    else:
        raise Exception(f"Unrecognized lattice type {lattice}")
    # Assert Rmin of 1 pixel
    Rmin = max(1, Rmin)
    # Generate base points which define pore centers
    centers = ~lattice_spheres(
        shape=[shape[0]*spacing+1, shape[1]*spacing+1],
        spacing=spacing,
        r=1,
        offset=0,
        lattice=lattice)
    # Retrieve indices of center points
    crds = np.where(centers)
    # Perform tessellation of center points
    tri = sptl.Delaunay(np.vstack(crds).T)
    # Add edges to image connecting each center point to make the requested lattice
    edges = np.zeros_like(centers, dtype=bool)
    msg = 'Adding edges of triangulation to image'
    for s in tqdm(tri.simplices, msg, **settings.tqdm):
        s2 = s.tolist()
        s2.append(s[0])
        for i in range(len(s)):
            P1, P2 = tri.points[s2[i]], tri.points[s2[i+1]]
            L = np.sqrt(np.sum(np.square(np.subtract(P1, P2))))
            if ((lattice == 'tri') and (L < spacing)) \
                    or ((lattice == 'sc') and (L <= spacing)):
                crds = line_segment(P1, P2)
                edges[tuple(crds)] = True
    # Remove intersections from lattice so edges are isolated clusters
    temp = spim.binary_dilation(centers, structure=ps_rect(w=1, ndim=2))
    edges = edges*~temp
    # Label each edge so they can be processed individually
    if lattice == 'sc':
        labels, N = spim.label(edges, structure=ps_round(r=1, ndim=2, smooth=False))
    else:
        labels, N = spim.label(edges, structure=ps_rect(w=3, ndim=2))
    # Obtain "slice" objects for each edge
    slices = spim.find_objects(labels)
    # Dilate each edge by some random amount, chosen from given distribution
    throats = np.zeros_like(edges, dtype=int)
    msg = 'Dilating edges to random widths'
    if dist is None:  # If user did not provide a distribution, use a uniform one
        dist = spst.uniform(loc=Rmin, scale=Rmax)
    for i, s in enumerate(tqdm(slices, msg, **settings.tqdm)):
        # Choose a random size, repeating until it is between Rmin and Rmax
        r = np.inf
        while (r > Rmax) or (r <= Rmin):
            r = np.around(dist.ppf(q=np.random.rand()), decimals=0).astype(int)
        if lattice == 'tri':  # Convert spacing to number of pixels
            r = int(r*np.sin(np.deg2rad(45)))
        # Isolate edge in s small subregion of image
        s2 = extend_slice(s, throats.shape, pad=2*r+1)
        mask = labels[s2] == (i + 1)
        # Apply dilation to subimage
        t = spim.binary_dilation(mask, structure=strel(r=r, t=1))
        # Insert subimage into main image
        throats[s2] += t
    # Generate requested images and return
    micromodel = throats > 0
    if (not return_edges) and (not return_centers):
        return micromodel
    else:
        # If multiple images are requested, attach them to a Results object
        ims = Results()
        ims.im = micromodel
        if return_edges:
            ims.edges = edges
        if return_centers:
            ims.centers = centers
        return ims


def points_to_spheres(im):
    from scipy.spatial import distance_matrix
    if im.ndim == 3:
        x, y, z = np.where(im > 0)
        coords = np.vstack((x, y, z)).T
    else:
        x, y = np.where(im > 0)
        coords = np.vstack((x, y))
    if im.dtype == bool:
        dmap = distance_matrix(coords.T, coords.T)
        mask = dmap < 1
        dmap[mask] = np.inf
        r = np.around(dmap.min(axis=0)/2, decimals=0).astype(int)
    else:
        r = im[x, y].flatten()
    im_spheres = np.zeros_like(im, dtype=bool)
    im_spheres = _insert_disks_at_points(
        im_spheres,
        coords=coords,
        radii=r,
        v=True,
        smooth=False,
    )
    return im_spheres


def random_cylindrical_pillars(
    shape=[1500, 1500],
    f=0.45,
    a=1500,
):
    from nanomesh import Mesher2D
    from porespy.generators import borders, spheres_from_coords

    if len(shape) != 2:
        raise Exception("Shape must be 2D")
    im = np.ones(shape, dtype=float)
    bd = borders(im.shape, mode='faces')
    im[bd] = 0.0

    mesher = Mesher2D(im)
    mesher.generate_contour(max_edge_dist=50, level=0.999)

    mesh = mesher.triangulate(opts=f'q0a{a}ne')
    # mesh.plot_pyvista(jupyter_backend='static', show_edges=True)
    tri = mesh.triangle_dict

    r_max = np.inf*np.ones([tri['vertices'].shape[0], ])
    for e in tri['edges']:
        L = np.sqrt(np.sum(np.diff(tri['vertices'][e], axis=0)**2))
        if tri['vertex_markers'][e[0]] == 0:
            r_max[e[0]] = min(r_max[e[0]], L/2)
        if tri['vertex_markers'][e[1]] == 0:
            r_max[e[1]] = min(r_max[e[1]], L/2)

    mask = np.ravel(tri['vertex_markers'] == 0)
    r = f*(2*r_max[mask])

    coords = tri['vertices'][mask]
    coords = np.pad(
        array=coords,
        pad_width=((0, 0), (0, 1)),
        mode='constant',
        constant_values=0)
    coords = np.vstack((coords.T, r)).T
    im_w_spheres = spheres_from_coords(coords, smooth=True, mode='contained')
    return im_w_spheres


if __name__ == '__main__':
    import porespy as ps
    import matplotlib.pyplot as plt

    # im = ~ps.generators.lattice_spheres([1501, 1501], r=1, offset=0, spacing=100)
    # im = im.astype(int)
    # inds = np.where(im)
    # im[inds] = np.random.randint(2, 50, len(inds[0]))
    # im = points_to_spheres(im)
    # plt.imshow(im)

    f = spst.norm(loc=47, scale=16.8)
    # f = spst.lognorm(loc=np.log10(47.0), s=np.log10(16.8))
    # Inspect the distribution
    if 0:
        plt.hist(f.rvs(10000))

    im, edges, centers = \
        ps.generators.rectangular_pillars(
            shape=[15, 30],
            spacing=137,
            dist=f,
            Rmin=1,
            Rmax=None,
            lattice='tri',
            return_edges=True,
            return_centers=True,
        )
    fig, ax = plt.subplots()
    # ax.imshow(im + edges*1.0 + centers*2.0, interpolation='none')
    ax.imshow(im, interpolation='none')
    ax.axis(False);
