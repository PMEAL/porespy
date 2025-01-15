import numpy as np
import numpy.typing as npt
from porespy.filters import seq_to_satn
from numba import njit
from porespy import settings
from porespy.tools import (
    get_tqdm,
    get_border,
    make_contiguous,
    _insert_disk_at_points,
    Results,
)
try:
    from pyedt import edt
except ModuleNotFoundError:
    from edt import edt


tqdm = get_tqdm()


__all__ = [
    "ibip",
]


def ibip(
    im: npt.NDArray,
    inlets: npt.NDArray = None,
    dt: npt.NDArray = None,
    maxiter: int = 10000,
    return_sizes: bool = True,
):
    r"""
    Performs invasion percolation on given image using the IBIP algorithm [1]_

    Parameters
    ----------
    im : ND-array
        Boolean array with ``True`` values indicating void voxels
    inlets : ND-array
        Boolean array with ``True`` values indicating where the invading fluid
        is injected from.  If ``None``, all faces will be used.
    dt : ND-array (optional)
        The distance transform of ``im``.  If not provided it will be
        calculated, so supplying it saves time.
    maxiter : scalar
        The number of steps to apply before stopping.  The default is to run
        for 10,000 steps which is almost certain to reach completion if the
        image is smaller than about 250-cubed.
    return_sizes : bool
        If `True` then an array containing the size of the sphere which first
        overlapped each voxel is returned. This array is not computed by default
        as it increases computation time.

    Returns
    -------
    results : dataclass-like
        A dataclass-like object with the following arrays as attributes:

        ============= ===============================================================
        Attribute     Description
        ============= ===============================================================
        im_seq        A numpy array with each voxel value containing the step at
                      which it was invaded.  Uninvaded voxels are set to -1.
        im_satn       A numpy array with each voxel value indicating the saturation
                      present in the domain it was invaded. Solids are given 0, and
                      uninvaded regions are given -1.
        im_size       If `return_sizes` was set to `True`, then a numpy array with
                      each voxel containing the radius of the sphere, in voxels,
                      that first overlapped it.
        ============= ===============================================================

    See Also
    --------
    porosimetry
    drainage
    qbip
    ibop

    References
    ----------
    .. [1] Gostick JT, Misaghian N, Yang J, Boek ES. Simulating volume-controlled
       invasion of a non-wetting fluid in volumetric images using basic image
       processing tools. Computers & Geosciences. 158(1), 104978 (2022). `Link.
       <https://doi.org/10.1016/j.cageo.2021.104978>`_

    Notes
    -----
    This function is slower and is less capable than `qbip`, which returns identical
    results, so it is recommended to use that instead.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/ibip.html>`_
    to view an online example.

    """
    # Process the boundary image
    if inlets is None:
        inlets = get_border(shape=im.shape, mode='faces')
    inlets = inlets*im
    bd = np.copy(inlets > 0)
    if dt is None:  # Find dt if not given
        dt = edt(im)
    # Initialize inv image with -1 in the solid, and 0's in the void
    seq = -1*(~im)
    sizes = -1.0*(~im)
    scratch = np.copy(bd)
    for step in tqdm(range(1, maxiter), **settings.tqdm):
        # Find insertion points
        edge = scratch*(dt > 0)
        if ~edge.any():
            break
        # Find the maximum value of the dt underlaying the new edge
        r_max = (dt*edge).max()
        # Find all values of the dt with that size
        dt_thresh = dt >= r_max
        # Extract the actual coordinates of the insertion sites
        pt = _where(edge*dt_thresh)
        seq = _insert_disk_at_points(
            im=seq,
            coords=pt,
            r=int(r_max),
            v=step,
            smooth=True,
        )
        if return_sizes:
            sizes = _insert_disk_at_points(
                im=sizes,
                coords=pt,
                r=int(r_max),
                v=r_max,
                smooth=True,
            )
        dt, bd = _update_dt_and_bd(dt, bd, pt)
        scratch = _insert_disk_at_points(
            im=scratch,
            coords=pt,
            r=1,
            v=1,
            smooth=False,
        )
    # Convert inv image so that uninvaded voxels are set to -1 and solid to 0
    temp = seq == 0  # Uninvaded voxels are set to -1 after _ibip
    seq[~im] = 0
    seq[temp] = -1
    seq = make_contiguous(im=seq, mode='symmetric')
    # Deal with invasion sizes similarly
    temp = sizes == 0
    sizes[~im] = 0
    sizes[temp] = -1
    results = Results()
    results.im_size = np.copy(sizes)
    results.im_seq = np.copy(seq)
    results.im_satn = seq_to_satn(seq)
    return results


@njit(parallel=False)
def _where(arr):
    inds = np.where(arr)
    result = np.vstack(inds)
    return result


@njit()
def _update_dt_and_bd(dt, bd, pt):
    if dt.ndim == 2:
        for i in range(pt.shape[1]):
            bd[pt[0, i], pt[1, i]] = True
            dt[pt[0, i], pt[1, i]] = 0
    else:
        for i in range(pt.shape[1]):
            bd[pt[0, i], pt[1, i], pt[2, i]] = True
            dt[pt[0, i], pt[1, i], pt[2, i]] = 0
    return dt, bd


if __name__ == "__main__":
    import numpy as np
    import porespy as ps
    import matplotlib.pyplot as plt
    from copy import copy

    # %% Run this cell to regenerate the variables
    np.random.seed(6)
    bg = 'white'
    plots = True
    im = ps.generators.blobs(shape=[300, 300], porosity=0.7, blobiness=2)
    inlets = np.zeros_like(im)
    inlets[0, :] = True
    ip = ps.simulations.ibip(im=im, inlets=inlets)

    # %% Generate some plots
    if plots:
        cmap = copy(plt.cm.plasma)
        cmap.set_under(color='black')
        cmap.set_over(color='grey')
        cmap.set_bad('grey')
        fig, ax = plt.subplots(1, 1)
        kw = ps.visualization.prep_for_imshow(ip.im_seq/im, im)
        kw['vmin'] = 0
        ax.imshow(**kw, cmap=cmap)
