import numpy as np
from loguru import logger
from edt import edt
from porespy.tools import get_tqdm, get_border
tqdm = get_tqdm()


def ibip_gpu(im, dt=None, inlets=None, max_iters=10000):
    """
    Performs invasion percolation on given image using iterative image dilation
    on GPU.

    Parameters
    ----------
    im : array_like
        Boolean array with ``True`` values indicating void voxels.  If a standard
        numpy array is passed, it is converted to a cupy array.
    dt : array_like, optional
        The distance transform of ``im``.  If a standard numpy array is passed,
        it is converted to a cupy array.
    inlets : array_like, optional
        Boolean array with ``True`` values indicating where the invading fluid
        is injected from.  If ``None``, all faces will be used.  If a standard
        numpy array is passed, it is converted to a cupy array.
    max_iters : scalar, optional
        The number of steps to apply before stopping.  The default is to run
        for 10,000 steps which is almost certain to reach completion if the
        image is smaller than about 250-cubed.

    Returns
    -------
    inv_sequence : ndarray
        An array the same shape as ``im`` with each voxel labelled by the
        sequence at which it was invaded.  The returned array will be a cupy
    inv_size : ndarray
        An array the same shape as ``im`` with each voxel labelled by the
        ``inv_size`` at which was filled.

    """
    import cupy as cp
    from cupyx.scipy import ndimage as cndi
    if dt is None:
        if isinstance(im, cp.ndarray):
            im = cp.asnumpy(im)
        dt = edt(im)
    im_g = cp.array(im)
    dt_g = cp.array(dt)
    if inlets is None:
        inlets = get_border(shape=im.shape)
    inlets_g = cp.array(inlets)

    bd_g = cp.copy(inlets_g > 0)
    dt_g = dt_g.astype(int)
    # alternative to _ibip
    inv_g = -1*((~im_g).astype(int))
    sizes_g = -1*((~im_g).astype(int))
    if im_g.ndim == 3:
        strel_g = ball_gpu
    else:
        strel_g = disk_gpu
    for step in tqdm(range(1, max_iters)):
        temp_g = cndi.binary_dilation(input=bd_g, structure=strel_g(1, smooth=False))
        edge_g = temp_g*(dt_g > 0)
        if ~cp.any(edge_g):
            logger.info('No more accessible invasion sites found')
            break
        # Find the maximum value of the dt underlaying the new edge
        r_max_g = dt_g[edge_g].max()
        # Find all values of the dt with that size
        dt_thresh_g = dt_g >= r_max_g
        # insert the disk/sphere
        pt_g = cp.where(edge_g*dt_thresh_g)  # will be used later in updating bd
        # ------------------------------------------------
        # update inv image
        bi_dial_g = cndi.binary_dilation(input=edge_g*dt_thresh_g,
                                         structure=strel_g(r_max_g.item()))
        bi_dial_step_g = bi_dial_g*step
        inv_prev_g = cp.copy(inv_g)
        mask_inv_prev_g = ~(inv_prev_g > 0)
        dial_single_g = mask_inv_prev_g*bi_dial_step_g
        inv_g = inv_prev_g+dial_single_g
        # update size image
        bi_dial_size_g = bi_dial_g*r_max_g
        sizes_prev_g = cp.copy(sizes_g)
        mask_sizes_prev_g = ~(sizes_prev_g > 0)
        dial_single_size_g = mask_sizes_prev_g*bi_dial_size_g
        sizes_g = sizes_prev_g+dial_single_size_g
        # ------------------------------------------------
        # Update boundary image with newly invaded points
        bd_g[pt_g] = True
        dt_g[pt_g] = 0
        if step == (max_iters - 1):  # If max_iters reached, end loop
            logger.info('Maximum number of iterations reached')
            break
    temp_g = inv_g == 0
    inv_g[~im_g] = 0
    inv_g[temp_g] = -1
    inv_seq_g = make_contiguous_gpu(im=inv_g)
    temp_g = sizes_g == 0
    sizes_g[~im_g] = 0
    sizes_g[temp_g] = -1
    inv_sequence = cp.asnumpy(inv_seq_g)
    inv_size = cp.asnumpy(sizes_g)
    return inv_sequence, inv_size


def rankdata_gpu(im_arr):
    """
    GPU alternative to scipy's rankdata using 'dense' method.
    Assign ranks to data, dealing with ties appropriately.

    Parameters
    ----------
    im_arr : cupy array_like
        DESCRIPTION.

    Returns
    -------
    dense : cupy ND-array
        An array of length equal to the size of im_arr, containing rank scores.

    """
    import cupy as cp
    arr = cp.ravel(im_arr)
    sorter = cp.argsort(arr)
    inv = cp.empty(sorter.size, dtype=cp.intp)
    inv[sorter] = cp.arange(sorter.size, dtype=cp.intp)
    arr = arr[sorter]
    obs = cp.r_[True, arr[1:] != arr[:-1]]
    dense = obs.cumsum()[inv]
    return dense


def make_contiguous_gpu(im):
    """
    Take an image with arbitrary greyscale values and adjust them to ensure
    all values fall in a contiguous range starting at 0.

    Parameters
    ----------
    im : cupy ND-array
        An ND array containing greyscale values

    Returns
    -------
    im_new : cupy ND-array
        An ND-array the same size as ``im`` but with all values in contiguous
        order.

    """
    shape = im.shape
    im_flat = im.flatten()
    mask_neg = im_flat < 0
    im_neg = -rankdata_gpu(-im_flat[mask_neg])
    mask_pos = im_flat > 0
    im_pos = rankdata_gpu(im_flat[mask_pos])
    im_flat[mask_pos] = im_pos
    im_flat[mask_neg] = im_neg
    im_new = np.reshape(im_flat, shape)
    return im_new


def ball_gpu(radius, smooth=True):
    """
    Generates a ball-shaped structuring element.

    Parameters
    ----------
    radius : int
        The radius of the ball-shaped structuring element.
    smooth : bool, optional
        Indicates whether the balls should include the nibs (``False``) on
        the surface or not (``True``).  The default is ``True``.

    Returns
    -------
    cupy ND-array
        The structuring element where elements of the neighborhood
        are 1 and 0 otherwise.

    """
    import cupy as cp
    n = 2 * radius + 1
    Z, Y, X = cp.mgrid[-radius:radius:n * 1j,
                       -radius:radius:n * 1j,
                       -radius:radius:n * 1j]
    s = X ** 2 + Y ** 2 + Z ** 2
    if smooth:
        radius = radius - 0.001
    return s <= radius * radius


def disk_gpu(radius, smooth=True):
    """
    Generates a flat, disk-shaped structuring element.

    Parameters
    ----------
    radius : int
        The radius of the disk-shaped structuring element.
    smooth : bool, optional
        Indicates whether the disks should include the nibs (``False``) on
        the surface or not (``True``).  The default is ``True``.

    Returns
    -------
    cupy ND-array
        The structuring element where elements of the neighborhood
        are 1 and 0 otherwise.

    """
    import cupy as cp
    L = cp.arange(-radius, radius + 1)
    X, Y = cp.meshgrid(L, L)
    if smooth:
        radius = radius - 0.001
    return (X ** 2 + Y ** 2) <= radius ** 2


if __name__ == '__main__':
    im = ps.generators.blobs(shape=[200, 200])
    a = ps.filters.ibip_gpu(im=im)
    