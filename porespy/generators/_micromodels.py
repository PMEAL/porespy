# import porespy as ps
import numpy as np
import scipy.ndimage as spim
import scipy.spatial as sptl
from porespy.tools import ps_rect, ps_round, extend_slice, get_tqdm
from porespy.generators import lattice_spheres, line_segment
from porespy import settings


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


def rectangular_pillars(shape=[5, 5], spacing=30, Rmin=2, Rmax=20, lattice='sc'):
    r"""
    A 2D micromodel with rectangular pillars arranged on a regular lattice

    Parameters
    ----------
    shape : list
        The number of pillars in the x and y directions.  The size of the of the
        image will be dictated by the ``spacing`` argument.
    spacing : int
        The number of pixels between neighboring pores centers.
    Rmin : int
        The minimum size of the openings between pillars in pixels
    Rmax : int
        The maximum size of the openings between pillars in pixels
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

    Returns
    -------
    ims : dataclass
        Several images are generated in internally, so they are all returned as
        attributes of a dataclass-like object.  The attributes are as follows:

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
    if lattice == 'sc':
        strel = cross
        Rmax = Rmax + 1
    else:
        strel = ex
        shape = np.array(shape) - 1
        Rmin = int(Rmin*np.sin(np.deg2rad(45)))
        Rmax = int((Rmax-2)*np.sin(np.deg2rad(45)))
    centers = ~lattice_spheres(
        shape=[shape[0]*spacing+1, shape[1]*spacing+1],
        spacing=spacing,
        r=1,
        offset=0,
        lattice=lattice)
    Rmin = max(1, Rmin)
    crds = np.where(centers)
    tri = sptl.Delaunay(np.vstack(crds).T)
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
    temp = spim.binary_dilation(centers, structure=ps_rect(w=1, ndim=2))
    edges = edges*~temp
    if lattice == 'sc':
        labels, N = spim.label(edges, structure=ps_round(r=1, ndim=2, smooth=False))
    else:
        labels, N = spim.label(edges, structure=ps_rect(w=3, ndim=2))
    slices = spim.find_objects(labels)
    throats = np.zeros_like(edges, dtype=int)
    msg = 'Dilating edges to random widths'
    for i, s in enumerate(tqdm(slices, msg, **settings.tqdm)):
        r = np.random.randint(Rmin, Rmax)
        s2 = extend_slice(s, throats.shape, pad=2*r+1)
        mask = labels[s2] == (i + 1)
        t = spim.binary_dilation(mask, structure=strel(r=r, t=1))
        throats[s2] += t
    micromodel = throats > 0
    return micromodel, edges, centers
