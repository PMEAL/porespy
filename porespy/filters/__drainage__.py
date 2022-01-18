import numpy as np
from edt import edt
import numba
import matplotlib.pyplot as plt
from porespy.filters import trim_disconnected_blobs
from porespy import settings
from porespy.tools import get_tqdm
from porespy.tools import Results
from scipy.ndimage import label
tqdm = get_tqdm()


@numba.jit(nopython=True, parallel=False)
def insert_disks_at_points(im, coords, radii, v, smooth=True):
    r"""
    Insert spheres (or disks) of specified radii into an ND-image at given locations.

    This function uses numba to accelerate the process, and does not overwrite
    any existing values (i.e. only writes to locations containing zeros).

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
                            if (s[a, b] == 1) and (im[x, y] == 0):
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
                                    if (s[a, b, c] == 1) and (im[x, y, z] == 0):
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


def drainage(pc, im, inlets=None, bins=25, rho=1000, g=9.81, voxel_size=1):
    r"""
    Simulate drainage using image-based sphere insertion, optionally including
    gravity

    Parameters
    ----------
    pc : ndarray
        An array containing precomputed capillary pressure values in each
        voxel. These values are typically obtained using the Washburn
        equation with user-specified surface tension and contact angle,
        along with the distance transform values for the radius.
    im : ndarray, optional
        The image of the porous media with ``True`` values indicating the
        void space.
    inlets : ndarray (default = x0)
        A boolean image the same shape as ``im``, with ``True`` values
        indicating the inlet locations. See Notes.  If not specified it is
        assumed that the invading phase enters from the bottom (x=0)
    bins : int or array_like (default = 25)
        The range of pressures to apply. If an integer is given
        then bins will be created between the lowest and highest pressures
        in the invasion pressure map, given by
        :math:`P = \frac{-2 \sigma}{r} + \rho g h` where :math:`r` is the
        local radius determined from a distance transform on ``im`` and
        :math:`h` is the x-dimension of the image. If a list is given, each
        value in the list is used directly in order.
    rho : float (default = 997)
        The density difference between the invading and defending phases.
        e.g. If air is displacing water this value should be -997 (1-998).
    g : float (default = 9.81)
        The gravitational constant prevailing for the simulation. The default
        is 9.81. If the domain is on an angle, such as a tilted micromodel,
        this value should be scaled appropriately by the user
        (i.e. g = 9.81 sin(alpha) where alpha is the angle relative to the
        horizonal).  Setting this value to removes any gravity effects.
    voxel_size : float
        The resolution of the image in meters per voxel edge

    Returns
    -------
    results : Results object
        A dataclass-like object with the following attributes:

        ========== ========= ==================================================
        Attribute  Datatype  Description
        ========== ========= ==================================================
        pc_inv     ndarray   A numpy array with each voxel value indicating
                             the capillary pressure at which it was invaded
        ========== ========= ==================================================

    Notes
    -----
    - The direction of gravity is always towards the x=0 axis
    - This algorithm has only been tested for gravity stabilized
      configurations, meaning the more dense fluid is on the bottom.
      Be sure that ``inlets`` are specified accordingly.

    """
    im = np.array(im, dtype=bool)
    pc[~im] = 0
    dt = edt(im)

    # Generate image for correcting entry pressure by gravitational effects
    h = np.ones_like(im, dtype=bool)
    h[0, ...] = False
    h = (edt(h) + 1)*voxel_size   # This could be done quicker using clever logic
    rgh = rho*g*h
    fn = pc + rgh

    if inlets is None:
        inlets = np.zeros_like(im)
        inlets[0, ...] = True

    if isinstance(bins, int):  # Use values fn for invasion steps
        Ps = np.linspace(fn[~np.isinf(fn)].min(), fn[~np.isinf(fn)].max(), bins)
    else:
        Ps = bins

    # Initialize empty arrays to accumulate results of each loop
    inv = np.zeros_like(im, dtype=float)
    seeds = np.zeros_like(im, dtype=bool)
    for p in tqdm(Ps, **settings.tqdm):
        # Find all locations in image invadable at current pressure
        temp = fn <= p
        # Trim locations not connected to the inlets
        new_seeds = trim_disconnected_blobs(temp, inlets=inlets)
        # Isolate only newly found locations to speed up inserting
        temp = new_seeds*(~seeds)
        # Find i,j,k coordinates of new locations
        coords = np.where(temp)
        # Add new locations to list of invaded locations
        seeds += new_seeds
        # Extract the local size of sphere to insert at each new location
        radii = dt[coords].astype(int)
        # Insert spheres are new locations of given radii
        inv = insert_disks_at_points(inv, np.vstack(coords), radii, p, smooth=True)
    return inv


def gravity_mio(im, inlets=None, bins=25,
                sigma=0.072, rho=1000, g=9.81,
                voxel_size=1, spacing=None):
    r"""
    Performs image-based porosimetry including gravity effects

    Parameters
    ----------
    im : ndarray
        The image into which the non-wetting fluid invades with ``True``
        indicating the void phase.  If studying a micromodel it may be
        more computationally efficient toto use a 2D image along with
        specifying the ``spacing`` argument.
    inlets : ndarray (default = x0)
        A boolean image the same shape as ``im``, with ``True`` values
        indicating the inlet locations. See Notes.  If not specified it is
        assumed that the invading phase enters from the bottom (x=0)
    bins : int or array_like (default = 25)
        The range of pressures to apply. If an integer is given
        then bins will be created between the lowest and highest pressures
        in the invasion pressure map, given by
        :math:`P = \frac{-2 \sigma}{r} + \rho g h` where :math:`r` is the
        local radius determined from a distance transform on ``im`` and
        :math:`h` is the x-dimension of the image. If a list is given, each
        value in the list is used directly in order.
    sigma : float (default = 0.072)
        The surface tension between the invading and defending phases
    rho : float (default = 997)
        The density difference between the invading and defending phases.
        e.g. If air is displacing water this value should be -997 (1-998).
    g : float (default = 9.81)
        The gravitational constant prevailing for the simulation. The default
        is 9.81. If the domain is on an angle, such as a tilted micromodel,
        this value should be scaled appropriately by the user
        (i.e. g = 9.81 sin(alpha) where alpha is the angle relative to the
        horizonal).
    voxel_size : float
        The resolution of the image in meters per voxel edge
    spacing : float
        If simulating a micromodel using a 2D image (or a 3D image of pillars),
        the spacing between the top and bottom plates indicates the
        perpendicular radii of curvature. The computation of the local capillary
        entry pressure can be made more accurate by specifing this additional
        value. If given, this argument is used to compute the capillary
        pressure as:

        .. math::

            P_c = -1 \sigma (\frac{1}{ \cdot dt} + \frac{1}{spacing}) \frac{1}{vx}

        where :math:`vx` = ``voxel_size``

    Notes
    -----
    - The direction of gravity is always towards the x=0 axis
    - This algorithm has only been tested for gravity stabilized
      configurations, meaning the more density fluid is on the bottom.
      Be sure that ``inlets`` are specified accordingly.

    """
    vx_res = voxel_size
    dt = edt(im)

    # Generate image for correcting entry pressure by gravitational effects
    h = np.ones_like(im, dtype=bool)
    h[0, ...] = False
    h = (edt(h) + 1)*vx_res   # This could be done quicker using clever logic
    rgh = rho*g*h
    if spacing is None:
        if im.ndim == 2:
            pc = 1*sigma/(dt*vx_res)
        else:
            pc = 2*sigma/(dt*vx_res)
    else:
        pc = sigma*(1/dt + 1/spacing)*(1/vx_res)
    fn = pc + rgh

    if inlets is None:
        inlets = np.zeros_like(im)
        inlets[0, ...] = True
    # Use radii values in mio image for invasion steps
    if isinstance(bins, int):
        Ps = np.linspace(fn[~np.isinf(fn)].min(), fn[~np.isinf(fn)].max(), bins)
    else:
        Ps = bins
    # Initialize empty arrays to accumulate results of each loop
    inv = np.zeros_like(im, dtype=float)
    seeds = np.zeros_like(im, dtype=bool)
    for p in tqdm(Ps, **settings.tqdm):
        # Find all locations in image invadable at current pressure
        temp = fn <= p
        # Trim locations not connected to the inlets
        new_seeds = trim_disconnected_blobs(temp, inlets=inlets)
        # Isolate only newly found locations to speed up inserting
        temp = new_seeds*(~seeds)
        # Find i,j,k coordinates of new locations
        coords = np.where(temp)
        # Add new locations to list of invaded locations
        seeds += new_seeds
        # Extract the local size of sphere to insert at each new location
        radii = dt[coords].astype(int)
        # Insert spheres are new locations of given radii
        inv = insert_disks_at_points(inv, np.vstack(coords), radii, p, smooth=True)
    return inv


# %%

# def find_trapped_wp(pc, outlets):
#     tqdm = get_tqdm()
#     trapped = np.zeros_like(outlets)
#     bins = np.unique(pc)
#     bins = bins[bins != 0]
#     for i in tqdm(bins, **settings.tqdm):
#         temp = pc > i
#         labels = label(temp)[0]
#         keep = np.unique(labels[outlets])
#         keep = keep[keep != 0]
#         trapped += temp*np.isin(labels, keep, invert=True)
#     return trapped

def reload_images(f):
    a = np.load(f)
    d = {}
    for i, angle in tqdm(enumerate(a['arr_1'])):
        d[angle] = a['arr_0'][i]
    return d


# def pc_to_satn(pc, im=None):
#     temp = ps.tools.make_contiguous(pc.astype(int))
#     satn = ps.filters.seq_to_satn(seq=temp, im=im)
#     return satn
