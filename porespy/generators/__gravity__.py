import numpy as np
import numba
from edt import edt
from skimage.morphology import disk, ball, square, cube
from porespy import settings
from porespy.tools import get_tqdm
from porespy.filters import trim_disconnected_blobs, fftmorphology
import scipy.ndimage as spim
tqdm = get_tqdm()


@numba.jit(nopython=True, parallel=False)
def insert_disks_at_points(im, coords, radii, v, smooth=True):
    r"""
    Insert spheres (or disks) of specified radii into an ND-image at given locations.

    This function uses numba to accelerate the process.

    Parameters
    ----------
    im : ND-array
        The image into which the spheres/disks should be inserted. This is an
        'in-place' operation.
    coords : ND-array
        The center point of each sphere/disk in an array of shape
        ``ndim by npts``
    radii : array_like
        The radii of the spheres/disks to add.
    v : scalar
        The value to insert
    smooth : boolean
        If ``True`` (default) then the spheres/disks will not have the litte
        nibs on the surfaces.
    """
    npts = len(coords[0])
    if im.ndim == 2:
        xlim, ylim = im.shape
        for i in range(npts):
            r = radii[i]
            s = _make_disk(r, smooth)
            pt = coords[:, i]
            for a, x in enumerate(range(pt[0]-r, pt[0]+r+1)):
                if (x >= 0) and (x < xlim):
                    for b, y in enumerate(range(pt[1]-r, pt[1]+r+1)):
                        if (y >= 0) and (y < ylim):
                            if (s[a, b] == 1):  # and (im[x, y] == 0):
                                im[x, y] = v
    elif im.ndim == 3:
        xlim, ylim, zlim = im.shape
        for i in range(npts):
            r = radii[i]
            s = _make_ball(r, smooth)
            pt = coords[:, i]
            for a, x in enumerate(range(pt[0]-r, pt[0]+r+1)):
                if (x >= 0) and (x < xlim):
                    for b, y in enumerate(range(pt[1]-r, pt[1]+r+1)):
                        if (y >= 0) and (y < ylim):
                            for c, z in enumerate(range(pt[2]-r, pt[2]+r+1)):
                                if (z >= 0) and (z < zlim):
                                    if (s[a, b, c] == 1):  # and (im[x, y, z] == 0):
                                        im[x, y, z] = v
    return im


@numba.jit(nopython=True, parallel=False)
def _make_disk(r, smooth=True):
    r"""
    Generate a strel suitable for use in numba nojit function.

    Numba won't allow calls to skimage strel generators so this function
    makes one, also using njit.
    """
    s = np.zeros((2*r+1, 2*r+1), dtype=type(r))
    if smooth:
        thresh = r - 0.001
    else:
        thresh = r
    for i in range(2*r+1):
        for j in range(2*r+1):
            if ((i - r)**2 + (j - r)**2)**0.5 <= thresh:
                s[i, j] = 1
    return s


@numba.jit(nopython=True, parallel=False)
def _make_ball(r, smooth=True):
    r"""
    Generate a strel suitable for use in numba nojit function.

    Numba won't allow calls to skimage strel generators so this function
    makes one, also using njit.
    """
    s = np.zeros((2*r+1, 2*r+1, 2*r+1), dtype=type(r))
    if smooth:
        thresh = r - 0.001
    else:
        thresh = r
    for i in range(2*r+1):
        for j in range(2*r+1):
            for k in range(2*r+1):
                if ((i - r)**2 + (j - r)**2 + (k - r)**2)**0.5 <= thresh:
                    s[i, j, k] = 1
    return s


def pseudo_gravity_packing(im, r, clearance=0, max_iter=1000):
    r"""
    Iteratively inserts spheres at the lowest accessible point in an image,
    mimicking a gravity packing.

    Parameters
    ----------
    im : ND-array
        The image into which the spheres should be inserted, with ``True``
        values indicating valid locations
    r : int
        The radius of the spheres to add
    clearance : int (default is 0)
        Adds the given abount space between each sphere.  Number can be
        negative for overlapping but should not be less than ``r``.
    max_iter : int (default is 1000)
        The maximum number of spheres to add

    Returns
    -------
    im : ND-array
        The input image ``im`` with the spheres added.

    Notes
    -----
    The direction of "gravity" along the x-axis, towards x=0.

    """
    print('-' * 60)
    print(f'Adding monodisperse spheres of radius {r}.')
    r = r - 1
    strel = disk if im.ndim == 2 else ball
    sites = fftmorphology(im == 1, strel=strel(r), mode='erosion')
    inlets = np.zeros_like(im)
    inlets[-(r+1), ...] = True
    sites = trim_disconnected_blobs(im=sites, inlets=inlets)
    x_min = np.where(sites)[0].min()
    n = None
    for n in tqdm(range(max_iter), **settings.tqdm):
        if im.ndim == 2:
            x, y = np.where(sites[x_min:x_min+2*r, ...])
        else:
            x, y, z = np.where(sites[x_min:x_min+2*r, ...])
        if len(x) == 0:
            break
        options = np.where(x == x.min())[0]
        choice = np.random.randint(len(options))
        if im.ndim == 2:
            cen = np.vstack([x[options[choice]] + x_min,
                             y[options[choice]]])
        else:
            cen = np.vstack([x[options[choice]] + x_min,
                             y[options[choice]],
                             z[options[choice]]])
        im = insert_disks_at_points(im, coords=cen,
                                    radii=np.array([r - clearance]), v=0)
        sites = insert_disks_at_points(sites, coords=cen,
                                       radii=np.array([2*r]), v=0)
        x_min += x.min()
    print(f'A total of {n} spheres were added.')
    im = spim.minimum_filter(input=im, footprint=strel(1))
    return im
