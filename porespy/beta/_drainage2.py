import numpy as np
from edt import edt
import numba
from porespy.filters import trim_disconnected_blobs, find_trapped_regions
from porespy.filters import find_disconnected_voxels
from porespy.filters import pc_to_satn, satn_to_seq, seq_to_satn
from porespy import settings
from porespy.tools import _insert_disks_at_points
from porespy.tools import get_tqdm
from porespy.tools import Results
tqdm = get_tqdm()


__all__ = [
    'drainage',
]


def drainage(im, pc, inlets=None, bins=25, return_size=False, return_sequence=False):
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
    dt = edt(im)
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
    inv_pc = np.zeros_like(im, dtype=float)
    inv_size = np.zeros_like(im, dtype=float)
    inv_seq = np.zeros_like(im, dtype=int)
    seeds = np.zeros_like(im, dtype=bool)

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
        radii = dt[coords].astype(int)
        # Insert spheres at new locations of given radii
        inv_pc = _insert_disks_at_points(im=inv_pc, coords=np.vstack(coords),
                                         radii=radii, v=bins[count], smooth=True)
        if return_size:
            inv_size = _insert_disks_at_points(im=inv_size, coords=np.vstack(coords),
                                               radii=radii, v=radii, smooth=True)
        if return_sequence:
            inv_seq = _insert_disks_at_points(im=inv_seq, coords=np.vstack(coords),
                                              radii=radii, v=count+1, smooth=True)
        count += 1

    # Set uninvaded voxels to inf
    inv_pc[(inv_pc == 0)*im] = np.inf
    inv_size[(inv_pc == 0)*im] = -1
    inv_seq[(inv_pc == 0)*im] = -1

    # Initialize results object
    results = Results()
    satn = pc_to_satn(pc=inv_pc, im=im)
    results.im_snwp = satn
    results.im_pc = inv_pc
    if return_size:
        results.im_size = inv_size
    if return_sequence:
        results.im_sequence = inv_seq
    return results


if __name__ == "__main__":
    import numpy as np
    import porespy as ps
    import matplotlib.pyplot as plt
    from copy import copy
    from edt import edt

    # %%
    np.random.seed(6)
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
    drn = drainage(im=im, pc=pc, inlets=inlets, bins=50)
    pc_curve = ps.metrics.pc_map_to_pc_curve(drn.im_pc, im=im)
    plt.step(pc_curve.pc, pc_curve.snwp, where='post')

    a = np.arange(0, im.shape[0])
    b = np.reshape(a, [im.shape[0], 1, 1])
    c = np.tile(b, (1, im.shape[1], im.shape[1]))
    pc = pc + delta_rho*g*(c*voxel_size)
    drn = drainage(im=im, pc=pc, inlets=inlets, bins=50)
    pc_curve = ps.metrics.pc_map_to_pc_curve(drn.im_pc, im=im)
    plt.step(pc_curve.pc, pc_curve.snwp, where='post')














