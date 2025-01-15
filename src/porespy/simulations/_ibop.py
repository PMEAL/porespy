import numpy as np
import numpy.typing as npt
from typing import Literal
from skimage.morphology import ball, disk, cube, square
from porespy import settings
from porespy.metrics import pc_curve
from porespy.tools import (
    _insert_disks_at_points,
    _insert_disks_at_points_parallel,
    get_tqdm,
    Results,
)
from porespy.filters import (
    trim_disconnected_blobs,
    find_trapped_regions,
    pc_to_satn,
    pc_to_seq,
)
from porespy.generators import (
    borders,
)
try:
    from pyedt import edt
except ModuleNotFoundError:
    from edt import edt


__all__ = [
    'drainage',
    'ibop',
]


tqdm = get_tqdm()


def ibop(
    im: npt.NDArray,
    pc: npt.NDArray = None,
    dt: npt.NDArray = None,
    inlets: npt.NDArray = None,
    outlets: npt.NDArray = None,
    residual: npt.NDArray = None,
    bins: int = None,
    return_sizes: bool = False,
    conn: Literal['min', 'max'] = 'min',
):
    r"""
    Simulate drainage using image-based sphere insertion, optionally including
    gravity

    Parameters
    ----------
    im : ndarray
        The image of the porous media with ``True`` values indicating the
        void space.
    pc : ndarray, optional
        Precomputed capillary pressure transform which is used to determine
        the invadability of each voxel. If not provided then twice the inverse of
        the distance transform of `im` is used.
    dt : ndarray (optional)
        The distance transform of ``im``.  If not provided it will be
        calculated, so supplying it saves time.
    inlets : ndarray, optional
        A boolean image the same shape as ``im``, with ``True`` values
        indicating the inlet locations. If not specified then access limitations
        are not applied so the result is essentially a local thickness filter.
    outlets : ndarray, optional
        A boolean image with ``True`` values indicating the outlet locations.
        If this is provided then trapped voxels of wetting phase are found and
        all the output images are adjusted accordingly. Note that trapping can
        be assessed during postprocessing as well.
    residual : ndarray, optional
        A boolean array indicating the locations of any residual invading
        phase. This is added to the intermediate image prior to trimming
        disconnected clusters, so will create connections to some clusters
        that would otherwise be removed. The residual phase is indicated
        in the final image by ``-np.inf`` values, since these are invaded at
        all applied capillary pressures.
    bins : int or array_like (default = None)
        The range of pressures to apply. If an integer is given then the given
        number of bins will be created between the lowest and highest values in
        ``pc``. If a list is given, each value in the list is used in ascending
        order. If `None` is given (default) then all the possible values in `pc`
        are used (or `dt` if `pc` is not given).
    return_sizes : bool, default = `False`
        If `True` then an array containing the size of the sphere which first
        overlapped each pixel is returned. This array is not computed by default
        as computing it increases computation time.
    conn : str
        Controls the shape of the structuring element used to find neighboring
        voxels when looking connectivity of invading blobs.  Options are:

        ========= ==================================================================
        Option    Description
        ========= ==================================================================
        'min'     This corresponds to a cross with 4 neighbors in 2D and 6 neighbors
                  in 3D.
        'max'     This corresponds to a square or cube with 8 neighbors in 2D and
                  26 neighbors in 3D.
        ========= ==================================================================

    Returns
    -------
    results : Results object
        A dataclass-like object with the following attributes:

        ========== ============================================================
        Attribute  Description
        ========== ============================================================
        im_seq     An ndarray with each voxel indicating the step number at
                   which it was first invaded by non-wetting phase
        im_satn    A numpy array with each voxel value indicating the global
                   saturation value at the point it was invaded
        im_size    If `return_sizes` was set to `True`, then a numpy array with
                   each voxel containing the radius of the sphere, in voxels, that
                   first overlapped it.
        im_pc      A numpy array with each voxel value indicating the
                   capillary pressure at which it was invaded.
        im_trapped A numpy array with ``True`` values indicating trapped voxels
        pc         1D array of capillary pressure values that were applied
        swnp       1D array of non-wetting phase saturations for each applied
                   value of capillary pressure (``pc``).
        ========== ============================================================

    See Also
    --------
    drainage

    Notes
    -----
    This algorithm only provides sensible results for gravity stabilized
    configurations, meaning the more dense fluid is on the bottom. Be sure that
    ``inlets`` are specified accordingly.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/simulations/reference/drainage.html>`_
    to view online example.

    """
    im = np.array(im, dtype=bool)

    if dt is None:
        dt = edt(im)

    if inlets is None:
        inlets = borders(shape=im.shape, mode='faces') * im

    if outlets is not None:
        outlets = outlets*im
        if np.sum(inlets * outlets):
            raise Exception('Specified inlets and outlets overlap')

    if pc is None:
        pc = 2.0/dt
    pc[~im] = 0  # Remove any infs or nans from pc computation

    if isinstance(bins, int):  # Use values in pc for invasion steps
        vmax = pc[pc < np.inf].max()
        vmin = pc[im][pc[im] > -np.inf].min()
        Ps = np.linspace(vmin, vmax*1.1, bins)
    elif bins is None:
        Ps = np.unique(pc[im])
    else:
        Ps = np.unique(bins)  # To ensure they are in ascending order

    # Initialize empty arrays to accumulate results of each loop
    pc_inv = np.zeros_like(im, dtype=float)
    pc_size = np.zeros_like(im, dtype=float)
    seeds = np.zeros_like(im, dtype=bool)

    # Begin IBOP algorithm
    if conn == 'min':
        strel = ball(1) if im.ndim == 3 else disk(1)
    elif conn == 'max':
        strel = cube(3) if im.ndim == 3 else square(3)
    else:
        raise Exception(f"Unrecognized value for conn ({conn})")
    for p in tqdm(Ps, **settings.tqdm):
        # Find all locations in image invadable at current pressure
        invadable = (pc <= p)*im
        # Trim locations not connected to the inlets
        if inlets is not None:
            invadable = trim_disconnected_blobs(
                im=invadable,
                inlets=inlets,
                strel=strel,
            )
        # Isolate only newly found locations to speed up inserting
        temp = invadable*(~seeds)
        # Find (i, j, k) coordinates of new locations
        coords = np.where(temp)
        # Add new locations to list of invaded locations
        seeds += invadable
        # Extract the local size of sphere to insert at each new location
        radii = dt[coords]
        # Insert spheres of given radii at new locations
        pc_inv = _insert_disks_at_points_parallel(
            im=pc_inv,
            coords=np.vstack(coords),
            radii=radii.astype(int),
            v=p,
            smooth=True,
            overwrite=False,
        )
        if return_sizes and (np.size(radii) > 0):
            pc_size = _insert_disks_at_points_parallel(
                im=pc_size,
                coords=np.vstack(coords),
                radii=radii.astype(int),
                v=np.amin(radii),
                smooth=True,
                overwrite=False,
            )
        # Deal with impact of residual, if present
        if residual is not None:
            # Find residual connected to current invasion front
            inv_temp = (pc_inv > 0)
            if np.any(inv_temp):
                # Find invadable pixels connected to surviving residual
                temp = trim_disconnected_blobs(residual, inv_temp, strel=strel)*~inv_temp
                if np.any(temp):
                    # Trim invadable pixels not connected to residual
                    new_seeds = trim_disconnected_blobs(invadable, temp, strel=strel)
                    # Find (i, j, k) coordinates of new locations
                    coords = np.where(new_seeds)
                    # Add new locations to list of invaded locations
                    seeds += new_seeds
                    # Extract the local size of sphere to insert at each new location
                    radii = dt[coords].astype(int)
                    # Insert spheres of given radii at new locations
                    pc_inv = _insert_disks_at_points_parallel(
                        im=pc_inv,
                        coords=np.vstack(coords),
                        radii=radii.astype(int),
                        v=p,
                        smooth=True,
                        overwrite=False,
                    )
                    if return_sizes and (np.size(radii) > 0):
                        pc_size = _insert_disks_at_points_parallel(
                            im=pc_size,
                            coords=np.vstack(coords),
                            radii=radii.astype(int),
                            v=np.amin(radii),
                            smooth=True,
                            overwrite=False,
                        )

    # Set uninvaded voxels to inf
    pc_inv[(pc_inv == 0)*im] = np.inf

    # Add residual if given
    if residual is not None:
        pc_inv[residual] = -np.inf

    # Analyze trapping and adjust computed images accordingly
    trapped = None
    if outlets is not None:
        seq = pc_to_seq(pc_inv, im=im, mode='drainage')
        trapped = find_trapped_regions(
            im=im,
            seq=seq,
            outlets=outlets,
            method='cluster',
        )
        trapped[seq == -1] = True
        pc_inv[trapped] = np.inf
        if residual is not None:  # Re-add residual to inv
            pc_inv[residual] = -np.inf

    # Initialize results object
    results = Results()
    results.im_satn = pc_to_satn(pc=pc_inv, im=im, mode='drainage')
    results.im_seq = pc_to_seq(pc=pc_inv, im=im, mode='drainage')
    results.im_pc = pc_inv
    if trapped is not None:
        results.im_seq[trapped] = -1
        results.im_satn[trapped] = -1
        results.im_pc[trapped] = np.inf
    if return_sizes:
        pc_size[pc_inv == np.inf] = np.inf
        pc_size[pc_inv == -np.inf] = -np.inf
        results.im_size = pc_size
    results.pc, results.snwp = pc_curve(im=im, pc=results.im_pc)
    return results


def drainage(
    im: npt.NDArray,
    pc: npt.NDArray = None,
    dt: npt.NDArray = None,
    inlets: npt.NDArray = None,
    outlets: npt.NDArray = None,
    residual: npt.NDArray = None,
    bins: int = 25,
    return_sizes: bool = False,
):
    results = ibop(
        im=im,
        pc=pc,
        dt=dt,
        inlets=inlets,
        outlets=outlets,
        residual=residual,
        bins=bins,
        return_sizes=return_sizes,
    )
    return results


drainage.__doc__ = ibop.__doc__


if __name__ == "__main__":
    import numpy as np
    import porespy as ps
    import matplotlib.pyplot as plt
    from copy import copy
    from edt import edt

    # %% Run this cell to regenerate the variables in drainage
    bg = 'white'
    plots = True
    im = ps.generators.blobs(
        shape=[500, 500],
        porosity=0.7,
        blobiness=1.5,
    )
    im = ps.filters.fill_blind_pores(im, surface=True)
    inlets = np.zeros_like(im)
    inlets[0, :] = True
    outlets = np.zeros_like(im)
    outlets[-1, :] = True

    lt = ps.filters.local_thickness(im)
    dt = edt(im)
    residual = lt > 25
    bins = 25
    pc = ps.filters.capillary_transform(
        im=im,
        dt=dt,
        sigma=0.072,
        theta=180,
        rho_nwp=1000,
        rho_wp=0,
        g=0,
        voxel_size=1e-4,
    )

    # %% Run different drainage simulations
    drn1 = ps.simulations.drainage(
        im=im,
        pc=pc,
        inlets=inlets,
    )
    drn2 = ps.simulations.drainage(
        im=im,
        pc=pc,
        inlets=inlets,
        outlets=outlets,
    )
    drn3 = ps.simulations.drainage(
        im=im,
        pc=pc,
        inlets=inlets,
        residual=residual,
    )
    drn4 = ps.simulations.drainage(
        im=im,
        pc=pc,
        inlets=inlets,
        outlets=outlets,
        residual=residual,
    )
    drn5 = ps.simulations.drainage(
        im=im,
        pc=pc,
    )

    # %% Visualize the invasion configurations for each scenario
    if plots:
        cmap = copy(plt.cm.plasma)
        cmap.set_under(color='black')
        cmap.set_over(color='grey')
        # cmap.set_bad(color='white')
        vmax = pc.max()*2
        fig, ax = plt.subplots(2, 3, facecolor=bg)

        tmp = np.copy(drn1.im_pc)
        tmp[~im] = -1
        tmp[tmp == np.inf] = vmax*2
        tmp[tmp == np.inf] = -1
        ax[0][0].imshow(tmp, cmap=cmap, vmin=0, vmax=vmax)
        ax[0][0].set_title("No trapping, no residual")

        tmp = np.copy(drn2.im_pc)
        tmp[~im] = -1
        tmp[tmp == np.inf] = vmax*2
        tmp[tmp == np.inf] = -1
        ax[0][1].imshow(tmp, cmap=cmap, vmin=0, vmax=vmax)
        ax[0][1].set_title("With trapping, no residual")

        tmp = np.copy(drn3.im_pc)
        tmp[~im] = -1
        tmp[tmp == np.inf] = vmax*2
        tmp[tmp == np.inf] = -1
        ax[1][0].imshow(tmp, cmap=cmap, vmin=0, vmax=vmax)
        ax[1][0].set_title("No trapping, with residual")

        tmp = np.copy(drn4.im_pc)
        tmp[~im] = -1
        tmp[tmp == np.inf] = vmax*2
        tmp[tmp == np.inf] = -1
        ax[1][1].imshow(tmp, cmap=cmap, vmin=0, vmax=vmax)
        ax[1][1].set_title("With trapping, with residual")

        tmp = np.copy(drn5.im_pc)
        tmp[~im] = -1
        tmp[tmp == np.inf] = vmax*2
        tmp[tmp == np.inf] = -1
        ax[0][2].imshow(tmp, cmap=cmap, vmin=0, vmax=vmax)
        ax[0][2].set_title("No access limitations")

        ax[1][2].step(drn1.pc, drn1.snwp, 'b-o', where='post',
                      label="No trapping, no residual")
        ax[1][2].step(drn2.pc, drn2.snwp, 'r--o', where='post',
                      label="With trapping, no residual")
        ax[1][2].step(drn3.pc, drn3.snwp, 'g--o', where='post',
                      label="No trapping, with residual")
        ax[1][2].step(drn4.pc, drn4.snwp, 'm--o', where='post',
                      label="With trapping, with residual")
        ax[1][2].legend()
