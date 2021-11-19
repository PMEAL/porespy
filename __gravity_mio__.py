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


def gravity_mio(im, inlets=None, bins=25,
                sigma=0.072, rho=1000, g=9.81,
                voxel_size=1, spacing=None):
    r"""
    Performs image-based porosimetry including gravity effects

    Parameters
    ----------
    im : ndarray
        The image into which the non-wetting fluid invades with ``True``
        indicating the void phase.  If studying a micromodel it's preferrable
        to use a 2D image along with specifying the ``spacing`` argument.
    inlets : ndarray (default = x0)
        A boolean image the same shape as ``im``, with ``True`` values
        indicating the inlet locations. Note that this algorithm only applies
        to gravity stabilized conditions, so use appropriate entry conditions
        (i.e. if water is displacing air, the ``inlets`` should be at or near
        the bottom).  If not specified it is assumed that the invading phase
        enters from the bottom (x=0)
    bins : int or array_like (default = 25)
        The range and number of pressures to apply. If an integer is given
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
        The gravitational constant prevailing for the simulation.  The default
        is 9.81. If the domain is on an angle, such as a tilted micromodel,
        this value should be scaled appropriately by the user
        (i.e. g = 9.81 sin(alpha) where alpha is the angle relative to the
        horizonal).
    voxel_size : float
        The resolution of the image in meters per voxel edge
    spacing : float
        If simulating a micromodel using a 2D image (or a 3D image of pillars),
        the spacing between the top and bottom plates indicates the
        perpendicular radii of curvature. The compuation of the local capillary
        entry pressure can be made more accurate by specifing this additional
        value. If given, this argument is used to compute the capillary
        pressure as:

        .. math::

            P_c = -2 \sigma (\frac{1}{ \cdot dt} + \frac{1}{spacing}) \frac{1}{vx}

        where :math:`vx` = ``voxel_size``

    Notes
    -----
    - The direction of gravity is always towards the x=0 axis
    - This algorithm only works for gravity stabilized configurations,
      meaning air entering from the top while water leaves out the bottom,
      or vice-versa. Be sure that ``inlets`` are specified accordingly.

    """
    vx_res = voxel_size
    dt = edt(im)

    # Generate image for correcting entry pressure by gravitational effect
    h = np.ones_like(im, dtype=bool)
    h[0, ...] = False
    h = edt(h)*vx_res  # This could be done quicker using clever logic
    rgh = rho*g*h
    if spacing is None:
        H = 2 if im.ndim == 3 else 1
        pc = H*sigma/(dt*vx_res)
    else:
        pc = 2*sigma*(1/dt + 1/spacing)*(1/vx_res)
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
def pc_curve(pc, im):
    tqdm = get_tqdm()
    pressures = np.unique(pc[im])
    pressures = pressures[pressures != 0]
    y = []
    Vp = im.sum()
    temp = pc[im]
    for p in tqdm(pressures, **settings.tqdm):
        y.append(((temp <= p)*(temp != 0)).sum()/Vp)
    pc_curve = Results()
    pc_curve.pc = pressures
    pc_curve.snwp = y
    return pc_curve


def find_trapped_wp(pc, outlets):
    tqdm = get_tqdm()
    trapped = np.zeros_like(outlets)
    bins = np.unique(pc)
    bins = bins[bins != 0]
    for i in tqdm(bins, **settings.tqdm):
        temp = pc > i
        labels = label(temp)[0]
        keep = np.unique(labels[outlets])
        keep = keep[keep != 0]
        trapped += temp*np.isin(labels, keep, invert=True)
    return trapped


def reload_images(f):
    a = np.load(f)
    d = {}
    for i, angle in tqdm(enumerate(a['arr_1'])):
        d[angle] = a['arr_0'][i]
    return d


def pc_to_satn(pc, im=None):
    temp = ps.tools.make_contiguous(pc.astype(int))
    satn = ps.filters.seq_to_satn(seq=temp, im=im)
    return satn


@numba.njit()
def satn_profile(satn, s, axis=0, span=10, mode='tile'):
    r"""
    Computes a saturation profile from an invasion image

    Parameters
    ----------
    satn : ndarray
        An image with each voxel indicating the saturation upon its
        invasion.  0's are treated a solid and -1's are treated as uninvaded
        void space.
    s : scalar
        The saturation value to use when thresholding the ``satn`` image
    axis : int
        The axis along which to profile should be measured
    span : int
        The number of layers to include in the moving average saturation
        calculation.
    mode : str
        How the moving average should be applied. Options are:

        * 'tile'
            The average is computed for discrete non-overlapping tiles of a
            size given by ``span``.
        * 'slide'
            The average is computed in a moving window starting at ``span/2``
            and sliding by a single voxel. This method provides more data
            points but is slower.

    Returns
    -------
    position : ndarray
        The position along the given axis at which saturation values are
        computed.  The units are in voxels.
    saturation : ndarray
        The computed saturation value at each position.
    """
    span = max(1, span)
    satn = np.swapaxes(satn, 0, axis)
    if mode == 'tile':
        y = np.zeros(int(satn.shape[0]/span))
        z = np.zeros_like(y)
        for i in range(int(satn.shape[0]/span)):
            void = satn[i*span:(i+1)*span, ...] != 0
            nwp = (satn[i*span:(i+1)*span, ...] < s)*(satn[i*span:(i+1)*span, ...] > 0)
            y[i] = nwp.sum()/void.sum()
            z[i] = i*span + (span-1)/2
    if mode == 'slide':
        y = np.zeros(int(satn.shape[0]-span))
        z = np.zeros_like(y)
        for i in range(int(satn.shape[0]-span)):
            void = satn[i:i+span, ...] != 0
            nwp = (satn[i:i+span, ...] < s)*(satn[i:i+span, ...] > 0)
            y[i] = nwp.sum()/void.sum()
            z[i] = i + (span-1)/2
    return (z, y)


def find_H(satn_profile, smin=0.1, smax=0.9):
    zmax = np.inf
    zmin = np.inf
    satn = satn_profile
    L = len(satn)
    for i, s in enumerate(satn):
        if (zmax == np.inf) and (satn[i] < smax):
            zmax = i
            smax = satn[i]
    satn = satn_profile[-1::-1]
    for i, s in enumerate(satn):
        if (zmin == np.inf) and (satn[i] > smin):
            zmin = i
            smin = satn[i]
    return (L-zmin, zmax, smin, smax)


# %%
if __name__ == '__main__':
    import porespy as ps

    vx = 0.0001
    L = int(0.1/vx)
    W = int(0.05/vx)
    t = int(0.05/vx)
    D = int(0.001/vx)
    print(L, W, t, D)

    # Enter parameters
    sigma = 0.071
    g = 9.81

    im = ps.generators.overlapping_spheres(shape=[L, W, t], r=D/2, porosity=0.65)
    # im[:, :, 0] = False
    # im[:, :, -1] = False
    # im[:, 0, :] = False
    # im[:, -1, :] = False
    inlets = np.zeros_like(im, dtype=bool)
    inlets[0, ...] = True
    outlets = np.zeros_like(im, dtype=bool)
    outlets[-1, ...] = True
    # mip = ps.filters.porosimetry(im=im, inlets=inlets)
    # a1 = mip[outlets].max()*vx
    # lt = ps.filters.local_thickness(im)
    # a = np.median(lt[lt > 0])*vx*2
    dt = edt(im)
    a = np.median(dt[dt > 0])*vx*2

    sim1 = {}
    sim2 = {}
    sim3 = {}
    sim4 = {}
    inv_Bo = [1000, 100, 10, 3.3, 1, 0.33, 0.1, 0.033, 0.01, 0.001, 0.0001]
    for i, dr in enumerate(inv_Bo):
        Bo = 1/inv_Bo[i]
        delta_rho = Bo*sigma/(g*a**2)

        temp = gravity_mio(im=im, inlets=inlets, sigma=sigma,
                           rho=delta_rho, g=g, voxel_size=vx,
                           bins=25).astype(int)
        sim1[i] = temp
        sim2[i] = ps.filters.seq_to_satn(temp, im)
        mask = ps.filters.find_trapped_regions(temp, outlets, return_mask=True)
        temp[mask] = -1
        sim3[i] = ps.filters.seq_to_satn(temp, im)
        temp[mask] = 0
        sim4[i] = ps.filters.seq_to_satn(temp, im)


# %%
    # h = 0
    # fig, ax = plt.subplots(1, 2)
    # ax[0].imshow(sim2[h]/im, origin='lower')
    # ax[1].imshow(sim3[h]/im, origin='lower')

# %%
    c = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
         'tab:brown', 'tab:pink', 'tab:olive', 'tab:cyan']
    plot = 0
    for h in range(len(inv_Bo)):
        if plot:
            plt.figure(h, figsize=[6, 6])
        for i, s in enumerate(np.arange(0.2, 1.0, 0.1)):
            s_actual = np.sum((sim2[h] < s)*im)/im.sum()
            pos, satn = satn_profile(satn=sim2[h], s=s, span=1, mode='slide')
            if plot:
                plt.plot(pos/im.shape[0], satn, '-', c=str(s_actual/1.5))
                plt.title("Bo = " + str(1/inv_Bo[h]))
            smin, smax = 0.01, 0.95
            zmin, zmax, smin, smax = find_H(satn, smin=smin, smax=smax)
            print(inv_Bo[h], s, zmin, zmax, smax-smin)
    # plt.plot([0, im.shape[0]], [smin, smin], 'k-')
    # plt.plot([0, im.shape[0]], [smax, smax], 'k-')








