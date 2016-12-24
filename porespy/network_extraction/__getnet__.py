import scipy as sp
import scipy.ndimage as spim
import scipy.sparse as sprs
from skimage.segmentation import find_boundaries

def extract_pore_network(im):
    r"""
    Analyzes an image that has been partitioned into pore regions and extracts
    the pore geometry and network connectivity.

    Returns
    -------
    A dictionary containing all the pore and throat sizes, as well as the
    network topological information.  The dictionary names use the OpenPNM
    convention (i.e. 'pore.coords', 'throat.conns') so it may be converted
    directly to an OpenPNM network object using the ``update`` command.
    """
    from skimage.morphology import disk, square, ball, cube
    dt = spim.distance_transform_edt(im > 0)

    if im.ndim == 2:
        ball = disk
        cube = square
    edges = find_boundaries(im)
    mn = spim.minimum_filter(im + (im == 0)*2*sp.amax(im), footprint=ball(1))
    mx = spim.maximum_filter(im, footprint=ball(1))
    hits = sp.where((mn != mx) * (mx > 0))
    conns = sp.vstack((mn[hits], mx[hits])).T - 1
    data = sp.ones(sp.shape(conns)[0])
    Np = sp.amax(conns) + 1
    adjmat = sprs.coo_matrix((data, (conns[:,0], conns[:,1])), shape=[Np, Np])
    adjmat = adjmat.tocsr().tocoo()
    adjmat.data = sp.ones_like(adjmat.data)
    Nt = adjmat.data.size
    conns = sp.vstack([adjmat.row, adjmat.col]).T

    coords = sp.zeros((Np, im.ndim), dtype=int)
    volume = sp.zeros((Np, ), dtype=int)
    diameter = sp.zeros((Np, ), dtype=int)
    for i in sp.unique(im)[1:]:
        vxls = sp.where(im == i)
        pore = i - 1
        coords[pore, :] = sp.mean(vxls, axis=1)
        volume[pore] = vxls[0].size
        diameter[pore] = sp.amax(dt[vxls])
    if im.ndim == 2:
        coords = sp.vstack((coords.T, sp.zeros((Np,)))).T

    net = {}
    net['pore.all'] = sp.ones((Np, ), dtype=bool)
    net['throat.all'] = sp.ones((Nt, ), dtype=bool)
    net['pore.coords'] = coords
    net['throat.conns'] = conns
    net['pore.volume'] = volume
    net['pore.diameter'] = diameter

    return net