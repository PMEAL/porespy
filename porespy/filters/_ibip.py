import numpy as np
from edt import edt
from porespy.tools import get_tqdm
import scipy.ndimage as spim
from porespy.tools import get_border, make_contiguous
from porespy.tools import _insert_disk_at_points
from porespy.tools import Results
import numba
from porespy import settings
tqdm = get_tqdm()


def ibip(im, inlets=None, dt=None, maxiter=10000):
    r"""
    Performs invasion percolation on given image using iterative image dilation

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

    Returns
    -------
    results : Results object
        A custom object with the following two arrays as attributes:

        'inv_sequence'
            An ndarray the same shape as ``im`` with each voxel labelled by
            the sequence at which it was invaded.

        'inv_size'
            An ndarray the same shape as ``im`` with each voxel labelled by
            the ``inv_size`` at which was filled.

    See Also
    --------
    porosimetry

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/ibip.html>`_
    to view online example.

    """
    # Process the boundary image
    if inlets is None:
        inlets = get_border(shape=im.shape, mode='faces')
    bd = np.copy(inlets > 0)
    if dt is None:  # Find dt if not given
        dt = edt(im)
    dt = dt.astype(int)  # Conert the dt to nearest integer
    # Initialize inv image with -1 in the solid, and 0's in the void
    inv = -1*(~im)
    sizes = -1*(~im)
    scratch = np.copy(bd)
    for step in tqdm(range(1, maxiter), **settings.tqdm):
        pt = _where(bd)
        scratch = np.copy(bd)
        temp = _insert_disk_at_points(im=scratch, coords=pt,
                                       r=1, v=1, smooth=False)
        # Reduce to only the 'new' boundary
        edge = temp*(dt > 0)
        if ~edge.any():
            break
        # Find the maximum value of the dt underlaying the new edge
        r_max = (dt*edge).max()
        # Find all values of the dt with that size
        dt_thresh = dt >= r_max
        # Extract the actual coordinates of the insertion sites
        pt = _where(edge*dt_thresh)
        inv = _insert_disk_at_points(im=inv, coords=pt,
                                      r=r_max, v=step, smooth=True)
        sizes = _insert_disk_at_points(im=sizes, coords=pt,
                                        r=r_max, v=r_max, smooth=True)
        dt, bd = _update_dt_and_bd(dt, bd, pt)
    # Convert inv image so that uninvaded voxels are set to -1 and solid to 0
    temp = inv == 0  # Uninvaded voxels are set to -1 after _ibip
    inv[~im] = 0
    inv[temp] = -1
    inv = make_contiguous(im=inv, mode='symmetric')
    # Deal with invasion sizes similarly
    temp = sizes == 0
    sizes[~im] = 0
    sizes[temp] = -1
    results = Results()
    results.inv_sequence = inv
    results.inv_sizes = sizes
    return results


@numba.jit(nopython=True, parallel=False)
def _where(arr):
    inds = np.where(arr)
    result = np.vstack(inds)
    return result


@numba.jit(nopython=True)
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


def find_trapped_regions(seq, outlets=None, bins=25, return_mask=True):
    r"""
    Find the trapped regions given an invasion sequence image

    Parameters
    ----------
    seq : ndarray
        An image with invasion sequence values in each voxel.  Regions
        labelled -1 are considered uninvaded, and regions labelled 0 are
        considered solid.
    outlets : ndarray, optional
        An image the same size as ``seq`` with ``True`` indicating outlets
        and ``False`` elsewhere.  If not given then all image boundaries
        are considered outlets.
    bins : int
        The resolution to use when thresholding the ``seq`` image.  By default
        the invasion sequence will be broken into 25 discrete steps and
        trapping will be identified at each step. A higher value of ``bins``
        will provide a more accurate trapping analysis, but is more time
        consuming. If ``None`` is specified, then *all* the steps will
        analyzed, providing the highest accuracy.
    return_mask : bool
        If ``True`` (default) then the returned image is a boolean mask
        indicating which voxels are trapped.  If ``False``, then a copy of
        ``seq`` is returned with the trapped voxels set to uninvaded and
        the invasion sequence values adjusted accordingly.

    Returns
    -------
    trapped : ND-image
        An image, the same size as ``seq``.  If ``return_mask`` is ``True``,
        then the image has ``True`` values indicating the trapped voxels.  If
        ``return_mask`` is ``False``, then a copy of ``seq`` is returned with
        trapped voxels set to 0.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/find_trapped_regions.html>`_
    to view online example.

    """
    seq = np.copy(seq)
    if outlets is None:
        outlets = get_border(seq.shape, mode='faces')
    trapped = np.zeros_like(outlets)
    if bins is None:
        bins = np.unique(seq)[-1::-1]
        bins = bins[bins > 0]
    elif isinstance(bins, int):
        bins = np.linspace(seq.max(), 1, bins)
    for i in tqdm(bins, **settings.tqdm):
        temp = seq >= i
        labels = spim.label(temp)[0]
        keep = np.unique(labels[outlets])[1:]
        trapped += temp*np.isin(labels, keep, invert=True)
    if return_mask:
        return trapped
    else:
        seq[trapped] = -1
        seq = make_contiguous(seq, mode='symmetric')
        return seq


if __name__ == "__main__":
    import numpy as np
    import porespy as ps
    import matplotlib.pyplot as plt
    from copy import copy

    # %% Run this cell to regenerate the variables in drainage
    np.random.seed(6)
    bg = 'white'
    plots = True
    im = ps.generators.blobs(shape=[300, 300], porosity=0.7, blobiness=2)
    inlets = np.zeros_like(im)
    inlets[0, :] = True
    ip = ps.filters.ibip(im=im, inlets=inlets)

    # %% Generate some plots
    if plots:
        cmap = copy(plt.cm.plasma)
        cmap.set_under(color='black')
        cmap.set_over(color='grey')
        fig, ax = plt.subplots(1, 1)
        kw = ps.visualization.prep_for_imshow(ip.inv_sequence, im)
        kw['vmin'] = 0
        ax.imshow(**kw, cmap=cmap)
