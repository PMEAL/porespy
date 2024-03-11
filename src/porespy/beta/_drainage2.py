import numpy as np
from edt import edt
from numba import njit, prange
from porespy.filters import trim_disconnected_blobs, pc_to_satn
from porespy import settings
from porespy.tools import get_tqdm, Results
tqdm = get_tqdm()


__all__ = [
    'drainage',
    'elevation_map',
]


def elevation_map(im_or_shape, voxel_size=1, axis=0):
    r"""
    Generate a image of distances from given axis

    Parameters
    ----------
    im_or_shape : ndarray or list
        This dictates the shape of the output image. If an image is supplied, then
        it's shape is used. Otherwise, the shape should be supplied as a N-D long
        list of the shape for each axis (i.e. `[200, 200]` or `[300, 300, 300]`).
    voxel_size : scalar, optional, default is 1
        The size of the voxels in physical units (i.e. `100e-6` would be 100 um per
        voxel side). If not given that 1 is used, so the returned image is in units
        of voxels.
    axis : int, optional, default is 0
        The direction along which the height is calculated.  The default is 0, which
        is the 'x-axis'.

    Returns
    -------
    elevation : ndarray
        A numpy array of the specified shape with the values in each voxel indicating
        the height of that voxel from the beginning of the specified axis.

    """
    if len(im_or_shape) <= 3:
        im = np.zeros(*im_or_shape, dtype=bool)
    else:
        im = im_or_shape
    im = np.swapaxes(im, 0, axis)
    a = np.arange(0, im.shape[0])
    b = np.reshape(a, [im.shape[0], 1, 1])
    c = np.tile(b, (1, *im.shape[1:]))
    c = c*voxel_size
    h = np.swapaxes(c, 0, axis)
    return h


def drainage(im, pc, inlets=None, bins=25):
    r"""
    Simulate drainage using image-based sphere insertion, optionally including
    gravity

    Parameters
    ----------
    im : ndarray
        The image of the porous media with ``True`` values indicating the
        void space.
    pc : ndarray
        An array containing capillary pressure map.
    inlets : ndarray (default = x0)
        A boolean image the same shape as ``im``, with ``True`` values
        indicating the inlet locations. See Notes. If not specified it is
        assumed that the invading phase enters from the bottom (x=0).
    bins : int or array_like (default =  `None`)
        The range of pressures to apply. If an integer is given
        then bins will be created between the lowest and highest pressures
        in the ``pc``.  If a list is given, each value in the list is used
        directly in order.

    Returns
    -------
    results : Results object
        A dataclass-like object with the following attributes:

        ========== =================================================================
        Attribute  Description
        ========== =================================================================
        im_pc      A numpy array with each voxel value indicating the
                   capillary pressure at which it was invaded
        im_snwp    A numpy array with each voxel value indicating the global
                   non-wetting phase saturation value at the point it was invaded
        ========== =================================================================

    """
    im = np.array(im, dtype=bool)
    dt = np.around(edt(im), decimals=0).astype(int)
    pc[~im] = np.inf

    if inlets is None:
        inlets = np.zeros_like(im)
        inlets[0, ...] = True

    if isinstance(bins, int):
        vals = np.unique(pc)
        vals = vals[~np.isinf(vals)]
        bins = np.logspace(np.log10(vals.min()), np.log10(vals.max()), bins)
    # Digitize pc
    pc_dig = np.digitize(pc, bins=bins)
    pc_dig[~im] = 0
    Ps = np.unique(pc_dig[im])

    # Initialize empty arrays to accumulate results of each loop
    seeds = np.zeros_like(im, dtype=bool)
    inv_pc = np.zeros_like(im, dtype=float)

    count = 0
    for p in tqdm(Ps, **settings.tqdm):
        # Find all locations in image invadable at current pressure
        temp = (pc_dig <= p)*im
        # Trim locations not connected to the inlets
        new_seeds = trim_disconnected_blobs(temp, inlets=inlets)
        # Isolate only newly found locations to speed up inserting
        temp = new_seeds*(~seeds)
        # Find i,j,k coordinates of new locations
        coords = np.where(temp)
        # Add new locations to list of invaded locations
        seeds += new_seeds
        # Extract the local size of sphere to insert at each new location
        radii = dt[coords]
        # Insert spheres at new locations of given radii
        inv_pc = _insert_disks_npoints_nradii_1value_parallel(
            im=inv_pc, coords=coords, radii=radii, v=bins[count])
        count += 1

    # Set uninvaded voxels to inf
    inv_pc[(inv_pc == 0)*im] = np.inf

    # Initialize results object and attached arrays
    results = Results()
    satn = pc_to_satn(pc=inv_pc, im=im)
    results.im_snwp = satn
    results.im_pc = inv_pc
    return results


@njit(parallel=True)
def _insert_disks_npoints_nradii_1value_parallel(
    im,
    coords,
    radii,
    v,
    overwrite=False,
    smooth=False,
):  # pragma: no cover
    if im.ndim == 2:
        xlim, ylim = im.shape
        for row in prange(len(coords[0])):
            i, j = coords[0][row], coords[1][row]
            r = radii[row]
            for a, x in enumerate(range(i-r, i+r+1)):
                if (x >= 0) and (x < xlim):
                    for b, y in enumerate(range(j-r, j+r+1)):
                        if (y >= 0) and (y < ylim):
                            R = ((a - r)**2 + (b - r)**2)**0.5
                            if (R <= r)*(~smooth) or (R < r)*(smooth):
                                if overwrite or (im[x, y] == 0):
                                    im[x, y] = v
    else:
        xlim, ylim, zlim = im.shape
        for row in prange(len(coords[0])):
            i, j, k = coords[0][row], coords[1][row], coords[2][row]
            r = radii[row]
            for a, x in enumerate(range(i-r, i+r+1)):
                if (x >= 0) and (x < xlim):
                    for b, y in enumerate(range(j-r, j+r+1)):
                        if (y >= 0) and (y < ylim):
                            for c, z in enumerate(range(k-r, k+r+1)):
                                if (z >= 0) and (z < zlim):
                                    R = ((a - r)**2 + (b - r)**2 + (c - r)**2)**0.5
                                    if (R <= r)*(~smooth) or (R < r)*(smooth):
                                        if overwrite or (im[x, y, z] == 0):
                                            im[x, y, z] = v
    return im


if __name__ == "__main__":
    import numpy as np
    import porespy as ps
    import matplotlib.pyplot as plt
    from edt import edt

    # %%
    im = ps.generators.blobs(shape=[200, 200, 200], porosity=0.7, blobiness=1.5, seed=0)
    inlets = np.zeros_like(im)
    inlets[0, ...] = True
    dt = edt(im)
    voxel_size = 1e-4
    sigma = 0.072
    theta = 180
    delta_rho = 1000
    g = 9.81

    pc = -2*sigma*np.cos(np.radians(theta))/(dt*voxel_size)
    drn1 = drainage(im=im, pc=pc, inlets=inlets, bins=50)
    pc_curve1 = ps.metrics.pc_map_to_pc_curve(drn1.im_pc, im=im)

    h = elevation_map(im, voxel_size=voxel_size)
    pc = pc + delta_rho*g*h
    drn2 = drainage(im=im, pc=pc, inlets=inlets, bins=50)
    pc_curve2 = ps.metrics.pc_map_to_pc_curve(drn2.im_pc, im=im)

    fix, ax = plt.subplots()
    ax.step(np.log10(pc_curve1.pc), pc_curve1.snwp, where='post')
    ax.step(np.log10(pc_curve2.pc), pc_curve2.snwp, where='post')
    ax.set_xlabel('log(Capillary Pressure [Pa])')
    ax.set_ylabel('Non-wetting Phase Saturation')
