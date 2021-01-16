import porespy as ps
import numpy as np
from edt import edt
import numba
from tqdm import tqdm
import matplotlib.pyplot as plt


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


def gravity_mio(im, inlets, sigma=0.072, rho=1000, g=9.81, voxel_size=1):
    vx_res = voxel_size
    # Generate image for correcting entry pressure by gravitational effect
    h = np.arange(0, im.shape[0])*vx_res
    h = np.broadcast_to(np.atleast_2d(h).T, im.shape)
    rgh = rho*g*h
    dt = edt(im)
    pc = 2*sigma/(dt*vx_res)
    fn = pc + rgh

    # Perform standard mio on image for comparison
    mio = ps.filters.porosimetry(im=im, inlets=inlets)
    # Use radii values in mio image for invasion steps
    Rs = np.unique(mio)[-1:0:-1]
    # Convert radii to capillary pressure, assuming perfectly wetting/non-wetting
    Ps = 2*sigma/(Rs*vx_res)
    Ps = np.concatenate([Ps, Ps*10])
    # Use the range of pressures in fn
    Ps = np.linspace(1, fn[~np.isinf(fn)].max(), 25)
    # Initialize empty arrays to accumulate results of each loop
    inv = np.zeros_like(im, dtype=float)
    seeds = np.zeros_like(im, dtype=bool)
    with tqdm(Ps) as pbar:
        for p in Ps:
            # Find all locations in image invadable at current pressure
            temp = fn <= p
            # Trim locations not connected to the inlets
            new_seeds = ps.filters.trim_disconnected_blobs(temp, inlets=inlets)
            # Isolate only newly found locations to speed up inserting
            temp = new_seeds*(~seeds)
            # Find i,j,k coordinates of new locations
            coords = np.where(temp)
            # Add new locations to list of invaded locations
            seeds += new_seeds
            # Extract the local size of sphere to insert at each new location
            radii = dt[coords].astype(int)
            # Convert pressure to corresponding radii for comparison to mio
            R = int(2*sigma/p/vx_res)
            # Insert spheres are new locations of given radii
            inv = insert_disks_at_points(inv, np.vstack(coords), radii, R, smooth=True)
            pbar.update()  # Increment progress bar
    return inv
