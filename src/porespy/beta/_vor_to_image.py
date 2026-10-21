import numpy as np
import scipy.ndimage as spim
from skimage.segmentation import watershed
from porespy.generators import line_segment
from porespy.tools import get_edt
from porespy.tools._label import _label_components


__all__ = [
    'vor_to_im',
    'pts_to_voronoi',
]


edt = get_edt()


def vor_to_im(vor, im, centroids=True):
    r"""
    Render a 2-D Voronoi tessellation onto a boolean image.

    Traces each finite Voronoi ridge as a line of ``True`` voxels. Ridges
    that extend to infinity (vertex index ``-1``) are skipped. Optionally
    the original seed points are also marked.

    Parameters
    ----------
    vor : scipy.spatial.Voronoi
        A Voronoi tessellation object, typically created with
        ``scipy.spatial.Voronoi``.
    im : ndarray
        A 2-D array used only to determine the output shape and dtype. Its
        values are ignored; the returned image is zero-initialised.
    centroids : bool, optional
        If ``True`` (default), the seed points (``vor.points``) are marked
        as ``True`` in the output image in addition to the ridge lines.

    Returns
    -------
    im : ndarray of bool
        A 2-D boolean array with the same shape as the input ``im``. Pixels
        on Voronoi ridges (and seed points, if ``centroids=True``) are
        ``True``; all other pixels are ``False``.

    Notes
    -----
    Infinite ridges (those with a vertex index of ``-1`` in
    ``vor.ridge_vertices``) are silently skipped. Ridge vertices that fall
    outside the image bounds are clipped at the boundary during line
    rasterisation.

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.spatial as sptl
    >>> pts = np.random.rand(50, 2) * 200
    >>> vor = sptl.Voronoi(pts)
    >>> im = np.zeros([200, 200], dtype=bool)
    >>> result = vor_to_im(vor, im)

    """
    im = np.zeros_like(im)
    Nx, Ny = im.shape
    for v in vor.ridge_vertices:
        if -1 not in v:  # Need a way to deal with instead of ignore -1's
            X0, X1 = vor.vertices[v]
            inds = line_segment(X0, X1)
            masks = [(inds[i] >= 0)*(inds[i] < im.shape[i]) for i in range(im.ndim)]
            mask = np.all(np.vstack(masks), axis=0)
            inds = [inds[i][mask] for i in range(im.ndim)]
            if len(inds[0]) > 0:
                im[tuple(inds)] = True
    if centroids:
        pts = vor.points
        pts = np.vstack(pts)
        im[tuple((pts).astype(int).T)] = True
    return im


def pts_to_voronoi(im, r=0, centroids=True, borders=True):
    r"""
    Compute a Voronoi tessellation from seed points via watershed.

    Each ``True`` pixel in ``im`` is treated as a seed. An EDT-based watershed
    grows regions from those seeds to fill the domain. Optionally the seeds
    are iteratively relaxed to region centroids (Lloyd's algorithm) before the
    final tessellation is returned.

    Parameters
    ----------
    im : ndarray of bool
        A 2-D boolean image where ``True`` pixels mark the Voronoi seed
        points.
    r : int, optional
        Number of Lloyd relaxation iterations. At each step the seeds are
        replaced by the centroids of the current Voronoi regions and the
        tessellation is recomputed. ``r=0`` (default) uses the original seeds
        without relaxation.
    centroids : bool, optional
        If ``True`` (default), the centroid of each final Voronoi region is
        marked with the value ``-1`` in the returned label image.
    borders : bool, optional
        If ``True`` (default), watershed lines (the Voronoi boundaries) are
        included in the output as pixels with value ``0``. If ``False``,
        every pixel is assigned to its nearest region and no boundary pixels
        exist.

    Returns
    -------
    ws : ndarray of int
        A 2-D integer label array the same shape as ``im``. Each Voronoi
        region carries a unique positive integer label. Boundary pixels are
        ``0`` (when ``borders=True``), and centroid pixels are ``-1`` (when
        ``centroids=True``).

    Notes
    -----
    The watershed is seeded from the labelled connected components of ``im``
    and grows into the distance-transform of ``~im``. Each relaxation
    iteration replaces the current seeds with the centre-of-mass of every
    labelled region, then calls ``pts_to_voronoi`` recursively with ``r=0``.

    Examples
    --------
    >>> import numpy as np
    >>> im = np.zeros([200, 200], dtype=bool)
    >>> pts = (np.random.rand(50, 2) * 200).astype(int)
    >>> im[pts[:, 0], pts[:, 1]] = True
    >>> ws = pts_to_voronoi(im, r=2)

    """
    dt = edt(~im)
    markers, N = _label_components(im, conn="min")
    ws = watershed(dt, markers, watershed_line=borders)
    for _ in range(r):
        pts = spim.center_of_mass(input=ws, labels=ws, index=range(1, N))
        pts = np.vstack(pts)
        im = np.zeros_like(im, dtype=bool)
        im[tuple((pts).astype(int).T)] = 1
        ws = pts_to_voronoi(im, r=0, centroids=False, borders=borders)
    if centroids:
        pts = spim.center_of_mass(input=ws, labels=ws, index=range(1, N))
        pts = np.vstack(pts)
        ws[tuple((pts).astype(int).T)] = -1
    return ws


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import porespy as ps
    import scipy.spatial as sptl

    fig, ax = plt.subplots(1, 2)

    np.random.seed(0)
    Np = 500
    Nx = 500
    Ny = 500
    r = 2

    im = np.zeros([Nx, Ny], dtype=bool)
    pts = np.random.rand(Np, 2)*im.shape

    vor = sptl.Voronoi(pts)
    im1 = vor_to_im(vor, im)
    im1 = _label_components(~im1, conn="min")[0]
    im1 = ps.tools.randomize_colors(im1)
    ax[0].imshow(im1, origin='lower', interpolation='none', cmap=plt.cm.plasma)

    im = np.zeros([Nx, Ny], dtype=bool)
    im[tuple((pts).astype(int).T)] = 1
    im2 = pts_to_voronoi(im, r=r, borders=True)
    im2 = ps.tools.randomize_colors(im2)
    ax[1].imshow(im2, origin='lower', interpolation='none', cmap=plt.cm.plasma)
