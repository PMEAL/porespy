import numpy as np
from skimage.morphology import ball, disk
from porespy.filters import (
    local_thickness,
    find_trapped_regions,
    size_to_satn,
    size_to_seq,
    seq_to_satn,
    pc_to_satn,
    pc_to_seq,
    trim_disconnected_blobs,
    find_disconnected_voxels,
)
from porespy.tools import (
    Results,
    get_tqdm,
    make_contiguous,
    _insert_disks_at_points,
    _insert_disks_at_points_parallel,
)
from porespy import settings
from edt import edt
from numba import njit, prange


__all__ = [
    'imbibition',
    'imbibition_dt',
]


tqdm = get_tqdm()


def imbibition_dt(im, inlets=None, residual=None):
    r"""
    This is a reference implementation of imbibition using distance transforms
    """
    im = np.array(im, dtype=bool)
    dt = np.around(edt(im), decimals=0).astype(int)
    bins = np.linspace(1, dt.max() + 1, dt.max() + 1, dtype=int)
    im_seq = -np.ones_like(im, dtype=int)
    im_size = np.zeros_like(im, dtype=float)
    for i, r in enumerate(tqdm(bins, **settings.tqdm)):
        seeds = dt >= r
        wp = im*~(edt(~seeds, parallel=settings.ncores) < r)
        if inlets is not None:
            wp = trim_disconnected_blobs(wp, inlets=inlets)
        if residual is not None:
            blobs = trim_disconnected_blobs(residual, inlets=wp)
            seeds = dt >= r
            seeds = trim_disconnected_blobs(seeds, inlets=blobs + inlets)
            wp = im*~(edt(~seeds, parallel=settings.ncores) < r)
        mask = wp*(im_seq == -1)
        im_size[mask] = r
        im_seq[mask] = i+1
    if residual is not None:
        im_seq[im_seq > 0] += 1
        im_seq[residual] = 1
        im_size[residual] = np.inf
    results = Results()
    results.im_seq = im_seq
    results.im_size = im_size
    return results


def imbibition(
    im,
    pc,
    dt=None,
    inlets=None,
    residual=None,
    bins=25,
    return_snwp=False,
):
    r"""
    Performs an imbibition simulation using image-based sphere insertion

    Parameters
    ----------
    im : ndarray
        The image of the porous materials with void indicated by ``True``
    pc : ndarray
        An array containing precomputed capillary pressure values in each
        voxel. This can include gravity effects or not. This can be generated
        by ``capillary_transform``.
    inlets : ndarray
        An image the same shape as ``im`` with ``True`` values indicating the
        wetting fluid inlet(s).  If ``None`` then the wetting film is able to
        appear anywhere within the domain.
    residual : ndarray, optional
        A boolean mask the same shape as ``im`` with ``True`` values
        indicating to locations of residual wetting phase.
    bins : int or array_like (default = 25)
        The range of pressures to apply. If an integer is given
        then bins will be created between the lowest and highest pressures
        in ``pc``. If a list is given, each value in the list is used
        directly in order.

    Notes
    -----
    The simulation proceeds as though the non-wetting phase pressure is very
    high and is slowly lowered. Then imbibition occurs into the smallest
    accessible regions at each step. Blind or inaccessible pores are
    assumed to be filled with wetting phase.

    Examples
    --------

    """
    if dt is None:
        dt = edt(im)

    if inlets is None:
        inlets = np.zeros_like(im)
        inlets[0, ...] = True

    pc[~im] = 0  # Remove any infs or nans from pc computation

    if isinstance(bins, int):
        vmax = pc[pc < np.inf].max()
        vmin = pc[im][pc[im] > -np.inf].min()
        Ps = np.logspace(np.log10(vmax), np.log10(vmin), bins)
    else:
        Ps = np.unique(bins)[::-1]  # To ensure they are in descending order

    # Initialize empty arrays to accumulate results of each loop
    im_pc = np.zeros_like(im, dtype=float)
    strel = ball(1) if im.ndim == 3 else disk(1)
    for i in tqdm(range(len(Ps)), **settings.tqdm):
        # This can be made faster if I find a way to get only seeds on edge, so
        # less spheres need to be drawn
        invadable = (pc < Ps[i])*im
        inv_mask = np.zeros_like(im, dtype=bool)
        if np.any(invadable):
            coords = np.where(invadable)
            radii = dt[coords].astype(int)
            inv_mask = _insert_disks_at_points_parallel(
                im=inv_mask,
                coords=np.vstack(coords),
                radii=radii,
                v=True,
                smooth=True,
                overwrite=True,
            )
        if inlets is not None:
            mask = trim_disconnected_blobs((~inv_mask)*im, inlets=inlets, strel=strel)
            im_pc += mask*Ps[i]*(im_pc == 0)

    # Collect data in a Results object
    result = Results()
    im_pc[im_pc == 0] = -np.inf
    im_pc[~im] = 0
    result.im_pc = im_pc
    result.im_seq = pc_to_seq(pc=im_pc, im=im, mode='imbibition')
    return result


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


# %%

if __name__ == '__main__':
    import porespy as ps
    import matplotlib.pyplot as plt
    import numpy as np
    from edt import edt
    from copy import copy
    ps.visualization.set_mpl_style()

    cm = copy(plt.cm.turbo)
    cm.set_under('grey')
    cm.set_over('k')

    i = np.random.randint(1, 100000)  # bad: 38364, good: 65270, 71698
    im = ps.generators.blobs([500, 500], porosity=0.55, blobiness=1.5, seed=i)
    im = ps.filters.fill_blind_pores(im, surface=True)

    inlets = np.zeros_like(im)
    inlets[0, ...] = True
    outlets = np.zeros_like(im)
    outlets[-1, ...] = True
    pc = ps.simulations.capillary_transform(im=im, voxel_size=1e-4)

    imb = imbibition(im=im, pc=pc, inlets=inlets)
    pc_curve1 = ps.metrics.pc_map_to_pc_curve(
        pc=imb.im_pc,
        im=im,
        seq=imb.im_seq,
        mode='imbibition',
    )

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(imb.im_seq, origin='lower', cmap=cm)

    mask = ps.filters.find_trapped_regions(seq=imb.im_seq, outlets=outlets)
    imb.im_pc[mask] = np.inf
    imb.im_seq[mask] = -1
    ax[1].imshow(imb.im_pc, origin='lower', cmap=cm)

    pc_curve2 = ps.metrics.pc_map_to_pc_curve(
        pc=imb.im_pc,
        im=im,
        seq=imb.im_seq,
        mode='imbibition',
    )
    fig, ax = plt.subplots()
    ax.semilogx(pc_curve1.pc, pc_curve1.snwp, 'b->', label='imbibition')
    ax.semilogx(pc_curve2.pc, pc_curve2.snwp, 'r-<', label='imbibition with trapping')
    ax.legend(loc='lower right')












