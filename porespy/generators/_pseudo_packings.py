import numba
import numpy as np
import scipy.ndimage as spim
from edt import edt
from skimage.morphology import disk, ball
from porespy import settings
from porespy.tools import get_tqdm, ps_round
from porespy.filters import trim_disconnected_blobs, fftmorphology
from loguru import logger
tqdm = get_tqdm()


@numba.jit(nopython=True, parallel=False)
def insert_disks_at_points(im, coords, radii, v, smooth=True):
    r"""
    Insert spheres of specified radii into an ndarray at given locations.

    Parameters
    ----------
    im : ndarray
        The image into which the spheres/disks should be inserted. This is an
        'in-place' operation.
    coords : ndarray
        The center point of each sphere/disk in an array of shape
        ``ndim by npts``
    radii : array_like
        The radii of the spheres/disks to add.
    v : scalar
        The value to insert
    smooth : boolean
        If ``True`` (default) then the spheres/disks will not have the litte
        nibs on the surfaces.

    Notes
    -----
    This function uses numba to accelerate the process.

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


def pseudo_gravity_packing(im, r, clearance=0, axis=0, maxiter=1000):
    r"""
    Iteratively inserts spheres at the lowest accessible point in an image,
    mimicking a gravity packing.

    Parameters
    ----------
    im : ndarray
        Image with ``True`` values indicating the phase where spheres should be
        inserted. A common option would be a cylindrical plug which would
        result in a tube filled with beads.
    r : int
        The radius of the spheres to be added
    clearance : int (default is 0)
        The amount space to add between neighboring spheres. The value can be
        negative for overlapping spheres, but ``abs(clearance) > r``.
    axis : int (default is 0)
        The axis along which gravity acts.
    maxiter : int (default is 1000)
        The maximum number of spheres to add

    Returns
    -------
    spheres : ndarray
        An image the same size as ``im`` with spheres indicated by ``True``.
        The spheres are only inserted at locations that are accessible
        from the top of the image.

    """
    logger.debug(f'Adding spheres of radius {r}')
    im = np.swapaxes(im, 0, axis)
    im_temp = np.zeros_like(im, dtype=bool)
    r = r - 1
    strel = disk if im.ndim == 2 else ball
    sites = fftmorphology(im == 1, strel=strel(r), mode='erosion')
    inlets = np.zeros_like(im)
    inlets[-(r+1), ...] = True
    sites = trim_disconnected_blobs(im=sites, inlets=inlets)
    x_min = np.where(sites)[0].min()
    n = None
    for n in tqdm(range(maxiter), **settings.tqdm):
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
        im_temp = insert_disks_at_points(im_temp, coords=cen,
                                         radii=np.array([r - clearance]), v=True)
        sites = insert_disks_at_points(sites, coords=cen,
                                       radii=np.array([2*r]), v=0)
        x_min += x.min()
    logger.debug(f'A total of {n} spheres were added')
    im_temp = np.swapaxes(im_temp, 0, axis)
    return im_temp


def pseudo_electrostatic_packing(im, r, sites=None,
                                 clearance=0,
                                 protrusion=0,
                                 maxiter=1000):
    r"""
    Iterativley inserts spheres as close to the given sites as possible.

    Parameters
    ----------
    im : ndarray
        Image with ``True`` values indicating the phase where spheres should be
        inserted.
    r : int
        Radius of spheres to insert.
    sites : ndarray (optional)
        An image with ``True`` values indicating the electrostatic attraction points.
        If this is not given then the peaks in the distance transform are used.
    clearance : int (optional, default=0)
        The amount of space to put between each sphere. Negative values are
        acceptable to create overlaps, but ``abs(clearance) < r``.
    protrusion : int (optional, default=0)
        The amount that spheres are allowed to protrude beyond the active phase.
    maxiter : int (optional, default=1000)
        The maximum number of spheres to insert.

    Returns
    -------
    im : ndarray
        An image with inserted spheres indicated by ``True``

    """
    im_temp = np.zeros_like(im, dtype=bool)
    dt_im = edt(im)
    if sites is None:
        dt2 = spim.gaussian_filter(dt_im, sigma=0.5)
        strel = ps_round(r, ndim=im.ndim, smooth=True)
        sites = (spim.maximum_filter(dt2, footprint=strel) == dt2)*im
    dt = edt(sites == 0).astype(int)
    sites = (sites == 0)*(dt_im >= (r - protrusion))
    dtmax = int(dt_im.max()*2)
    dt[~sites] = dtmax
    r = r + clearance
    # Get initial options
    options = np.where(dt == 1)
    for _ in tqdm(range(maxiter), **settings.tqdm):
        hits = dt[options] < dtmax
        if hits.sum() == 0:
            if dt.min() == dtmax:
                break
            options = np.where(dt == dt.min())
            hits = dt[options] < dtmax
        if hits.size == 0:
            break
        choice = np.where(hits)[0][0]
        cen = np.vstack([options[i][choice] for i in range(im.ndim)])
        im_temp = insert_disks_at_points(im_temp, coords=cen,
                                         radii=np.array([r-clearance]), v=True)
        dt = insert_disks_at_points(dt, coords=cen,
                                    radii=np.array([2*r-clearance]), v=int(dtmax))
    return im_temp
