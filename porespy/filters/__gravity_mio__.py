import numpy as np
from edt import edt
import numba
import matplotlib.pyplot as plt
from porespy.filters import trim_disconnected_blobs
from porespy import settings
from porespy.tools import get_tqdm
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


def gravity_mio(im, inlets, bins=25, sigma=0.072, rho=1000, g=9.81, alpha=90, voxel_size=1):
    vx_res = voxel_size
    # Generate image for correcting entry pressure by gravitational effect
    h = np.ones_like(im, dtype=bool)
    h[0, ...] = False
    h = edt(h)*vx_res*np.sin(np.deg2rad(alpha))
    rgh = rho*g*h
    dt = edt(im)
    pc = 2*sigma/(dt*vx_res)
    fn = pc + rgh

    # Use radii values in mio image for invasion steps
    if isinstance(bins, int):
        Rs = np.unique(np.linspace(1, dt.max(), bins).astype(int))[-1::-1]
    else:
        Rs = bins
    # Convert radii to capillary pressure, assuming perfectly wetting/non-wetting
    Ps = 2*sigma/(Rs*vx_res)
    Ps = np.concatenate([Ps, Ps*10])
    # Use the range of pressures in fn
    Ps = np.linspace(1, fn[~np.isinf(fn)].max(), 25)
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
        # Convert pressure to corresponding radii for comparison to mio
        R = int(2*sigma/p/vx_res)
        # Insert spheres are new locations of given radii
        inv = insert_disks_at_points(inv, np.vstack(coords), radii, p, smooth=True)
    return inv


# %%
if __name__ == '__main__':
    import porespy as ps

    if True:
        im = ~ps.generators.RSA([2900, 2000, 20], r=10, clearance=1)
        im = np.pad(im, pad_width=[[0, 0], [1, 1], [1, 1]], mode='constant',
                    constant_values=False)
    else:
        im = np.load('packing.npz')['arr_0']
        im = np.swapaxes(im, 0, 1)
    inlets = np.zeros_like(im)
    inlets[-1, :, :] = True
    # mip = ps.filters.porosimetry(im=im, inlets=inlets)

    sim1 = []
    sim2 = []
    for i, alpha in enumerate([0, 15, 30, 45, 60]):
        # Enter parameters
        voxel_size = 0.5e-4
        delta_rho = -1205
        sigma = 0.064
        a = 0.001  # Average pore size, seems to be plate spacing?
        g = 9.81*np.sin(np.deg2rad(alpha))
        h = im.shape[0]*voxel_size
        rgh = delta_rho*g*h*np.sin(np.deg2rad(alpha))
        Bo = delta_rho*g*(a**2)/sigma
        print(Bo)

        temp = gravity_mio(im=im, inlets=inlets, sigma=sigma,
                           rho=delta_rho, g=g, alpha=alpha,
                           voxel_size=voxel_size)
        sim1.append(temp)
        outlets = np.zeros_like(im)
        outlets[0, :, :] = True
        trapped = ps.filters.find_trapped_regions(seq=sim1[i], outlets=outlets)
        sim2.append(sim1[i] * ~trapped)

# %%
    c = ['tab:blue', 'tab:red', 'tab:orange', 'tab:purple', 'tab:green']
    for i in range(5):
        print(i)
        pc = np.unique(sim1[i])[1:]
        snwp_w_trapping = []
        snwp = []
        for p in pc:
            snwp.append((sim1[i] == p).sum()/im.sum())
            snwp_w_trapping.append((sim2[i] == p).sum()/im.sum())
        snwp_w_trapping = np.cumsum(snwp_w_trapping)
        snwp = np.cumsum(snwp)
        plt.plot(snwp_w_trapping, pc, '-o', color=c[i])
        plt.ylim([-1500, 1500])
        # plt.plot(pc, snwp, 'b-o')
