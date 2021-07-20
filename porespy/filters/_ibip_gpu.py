import numpy as np
from loguru import logger
from edt import edt
from porespy.tools import get_tqdm, get_border
from porespy.tools import Results
tqdm = get_tqdm()


def ibip_gpu(im, dt=None, inlets=None, maxiter=10000):  # pragma: no cover
    """
    Performs invasion percolation on given image using iterative image
    dilation on GPU.

    Parameters
    ----------
    im : array_like
        Boolean array with ``True`` values indicating void voxels. If a
        standard numpy array is passed, it is converted to a cupy array.
    dt : array_like, optional
        The distance transform of ``im``. If a standard numpy array is
        passed, it is converted to a cupy array.
    inlets : array_like, optional
        Boolean array with ``True`` values indicating where the invading
        fluid is injected from.  If ``None``, all faces will be used.
        If a standard numpy array is passed, it is converted to a cupy
        array.
    maxiter : int, optional
        The number of steps to apply before stopping.  The default is to
        run for 10,000 steps which is almost certain to reach completion
        if the image is smaller than about 250-cubed.

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

    """
    import cupy as cp
    from cupyx.scipy import ndimage as cndi

    im_gpu = cp.array(im)
    dt = edt(cp.asnumpy(im)) if dt is None else dt
    dt_gpu = cp.array(dt)
    inlets = get_border(shape=im.shape) if inlets is None else inlets
    inlets_gpu = cp.array(inlets)
    bd_gpu = cp.copy(inlets_gpu > 0)
    dt_gpu = dt_gpu.astype(int)

    # Alternative to _ibip
    inv_gpu = -1*((~im_gpu).astype(int))
    sizes_gpu = -1*((~im_gpu).astype(int))
    strel_gpu = ball_gpu if im_gpu.ndim == 3 else disk_gpu

    for step in tqdm(range(1, maxiter)):
        temp_gpu = cndi.binary_dilation(input=bd_gpu,
                                        structure=strel_gpu(1, smooth=False))
        edge_gpu = temp_gpu * (dt_gpu > 0)
        if ~cp.any(edge_gpu):
            logger.info('No more accessible invasion sites found')
            break
        # Find the maximum value of the dt underlaying the new edge
        r_max_gpu = dt_gpu[edge_gpu].max()
        # Find all values of the dt with that size
        dt_thresh_gpu = dt_gpu >= r_max_gpu
        # Insert the disk/sphere
        pt_gpu = cp.where(edge_gpu * dt_thresh_gpu)  # will be used later in updating bd
        # Update inv image
        bi_dial_gpu = cndi.binary_dilation(input=edge_gpu*dt_thresh_gpu,
                                           structure=strel_gpu(r_max_gpu.item()))
        bi_dial_step_gpu = bi_dial_gpu * step
        inv_prev_gpu = cp.copy(inv_gpu)
        mask_inv_prev_gpu = ~(inv_prev_gpu > 0)
        dial_single_gpu = mask_inv_prev_gpu * bi_dial_step_gpu
        inv_gpu = inv_prev_gpu + dial_single_gpu
        # Update size image
        bi_dial_size_gpu = bi_dial_gpu * r_max_gpu
        sizes_prev_gpu = cp.copy(sizes_gpu)
        mask_sizes_prev_gpu = ~(sizes_prev_gpu > 0)
        dial_single_size_gpu = mask_sizes_prev_gpu * bi_dial_size_gpu
        sizes_gpu = sizes_prev_gpu + dial_single_size_gpu
        # Update boundary image with newly invaded points
        bd_gpu[pt_gpu] = True
        dt_gpu[pt_gpu] = 0
        if step == (maxiter - 1):  # If max_iters reached, end loop
            logger.info('Maximum number of iterations reached')
            break

    temp_gpu = inv_gpu == 0
    inv_gpu[~im_gpu] = 0
    inv_gpu[temp_gpu] = -1
    inv_seq_gpu = make_contiguous_gpu(im=inv_gpu)
    temp_gpu = sizes_gpu == 0
    sizes_gpu[~im_gpu] = 0
    sizes_gpu[temp_gpu] = -1
    inv_sequence = cp.asnumpy(inv_seq_gpu)
    inv_size = cp.asnumpy(sizes_gpu)
    results = Results()
    results.inv_sequence = inv_sequence
    results.inv_size = inv_size
    return results


def rankdata_gpu(im_arr):  # pragma: no cover
    """
    GPU alternative to scipy's rankdata using 'dense' method.
    Assign ranks to data, dealing with ties appropriately.

    Parameters
    ----------
    im_arr : cupy ndarray
        Input image.

    Returns
    -------
    dense : cupy ndarray
        An array of length equal to the size of im_arr, containing rank
        scores.

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


def make_contiguous_gpu(im):  # pragma: no cover
    """
    Take an image with arbitrary greyscale values and adjust them to
    ensure all values fall in a contiguous range starting at 0.

    Parameters
    ----------
    im : cupy ndarray
        Input array containing greyscale values

    Returns
    -------
    im_new : cupy ndarray
        Array the same size as ``im`` but with all values in contiguous
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


def ball_gpu(radius, smooth=True):  # pragma: no cover
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
    cupy ndarray
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


def disk_gpu(radius, smooth=True):  # pragma: no cover
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
    cupy ndarray
        The structuring element where elements of the neighborhood are
        1 and 0 otherwise.

    """
    import cupy as cp
    L = cp.arange(-radius, radius + 1)
    X, Y = cp.meshgrid(L, L)
    if smooth:
        radius = radius - 0.001
    return (X ** 2 + Y ** 2) <= radius ** 2


if __name__ == '__main__':
    import porespy as ps
    im = ps.generators.blobs(shape=[200, 200])
    out = ps.filters.ibip_gpu(im=im)
