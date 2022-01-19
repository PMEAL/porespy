import numpy as np
from edt import edt
import numba
from skimage.morphology import skeletonize_3d
from porespy.filters import trim_disconnected_blobs, find_trapped_regions
from porespy.filters import pc_to_satn
from porespy import settings
from porespy.tools import get_tqdm
from porespy.tools import Results
from porespy.metrics import pc_curve
tqdm = get_tqdm()


@numba.jit(nopython=True, parallel=False)
def insert_disks_at_points(im, coords, radii, v, smooth=True):  # pragma: no cover
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
def _make_disk(r, smooth=True):  # pragma: no cover
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
def _make_ball(r, smooth=True):  # pragma: no cover
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


def drainage(im, voxel_size, pc=None, inlets=None, outlets=None, bins=25,
             delta_rho=1000, g=9.81, sigma=0.072, theta=180):
    r"""
    Simulate drainage using image-based sphere insertion, optionally including
    gravity

    Parameters
    ----------
    im : ndarray
        The image of the porous media with ``True`` values indicating the
        void space.
    voxel_size : float
        The resolution of the image in meters per voxel edge.
    pc : ndarray, optional
        An array containing precomputed capillary pressure values in each
        voxel. If not provided then the Washburn equation is used with the
        provided values of ``sigma`` and ``theta``. If the image is 2D only
        1 principle radii of curvature is included.
    inlets : ndarray (default = x0)
        A boolean image the same shape as ``im``, with ``True`` values
        indicating the inlet locations. See Notes. If not specified it is
        assumed that the invading phase enters from the bottom (x=0).
    outlets : ndarray, optional
        Similar to ``inlets`` except defining the outlets. This image is used
        to assess trapping.  \If not provided then trapping is ignored,
        otherwise a mask indicating which voxels were trapped is included
        amoung the returned data.
    bins : int or array_like (default = 25)
        The range of pressures to apply. If an integer is given
        then bins will be created between the lowest and highest pressures
        in the ``pc``.  If a list is given, each value in the list is used
        directly in order.
    delta_rho : float (default = 997)
        The density difference between the invading and defending phases.
        Note that if air is displacing water this value should be -997 (1-998).
    g : float (default = 9.81)
        The gravitational constant prevailing for the simulation. The default
        is 9.81. If the domain is on an angle, such as a tilted micromodel,
        this value should be scaled appropriately by the user
        (i.e. g = 9.81 sin(alpha) where alpha is the angle relative to the
        horizonal).  Setting this value to zeor removes any gravity effects.
    sigma : float (default = 0.072)
        The surface tension of the fluid pair. If ``pc`` is provided this is
        ignored.
    theta : float (defaut = 180)
        The contact angle of the sytem in degrees.  If ``pc`` is provded this
        is ignored.

    Returns
    -------
    results : Results object
        A dataclass-like object with the following attributes:

    ========== ================================================================
    Attribute  Description
    ========== ================================================================
    im_pc      A numpy array with each voxel value indicating the
               capillary pressure at which it was invaded
    im_satn    A numpy array with each voxel value indicating the global
               saturation value at the point it was invaded
    pc         1D array of capillary pressure values that were applied
    swnp       1D array of non-wetting phase saturations for each applied
               value of capillary pressure (``pc``).
    ========== ===============================================================

    Notes
    -----
    - The direction of gravity is always towards the x=0 axis
    - This algorithm has only been tested for gravity stabilized
      configurations, meaning the more dense fluid is on the bottom.
      Be sure that ``inlets`` are specified accordingly.

    """
    im = np.array(im, dtype=bool)
    dt = edt(im)
    if pc is None:
        pc = -(im.ndim-1)*sigma*np.cos(np.deg2rad(theta))/(dt*voxel_size)
    pc[~im] = 0  # Remove any infs or nans from pc computation

    # Generate image for correcting entry pressure by gravitational effects
    h = np.ones_like(im, dtype=bool)
    h[0, ...] = False
    h = (edt(h) + 1)*voxel_size   # This could be done quicker using clever logic
    rgh = delta_rho*g*h
    fn = pc + rgh

    if inlets is None:
        inlets = np.zeros_like(im)
        inlets[0, ...] = True

    if isinstance(bins, int):  # Use values fn for invasion steps
        sk = skeletonize_3d(im) > 0  # This is need to mask meaningful values
        temp = fn[sk]
        Ps = np.logspace(np.log10(temp.min()), np.log10(temp.max()), bins)
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

    _pccurve = pc_curve(im=im, pressures=inv)

    results = Results()
    results.im_pc = inv
    results.im_satn = pc_to_satn(pc=inv, im=im)
    results.pc = _pccurve.pc
    results.snwp = _pccurve.snwp

    if outlets is not None:
        seq = np.digitize(results.im_satn, bins=np.linspace(0, 1, len(Ps)))
        seq = (seq.astype(int) - 1)*im
        trapped = find_trapped_regions(seq=seq, outlets=outlets)
        results.trapped = trapped

    return results
