import numpy as np
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.exposure import rescale_intensity, match_histograms
import dask
dask.config.set(scheduler="threads")
# from dask.diagnostics import ProgressBar


def nl_means_layered(im, cores=None, axis=0, patch_size=5, patch_distance=15, h=4):
    r"""
    Apply the non-local means filter to each 2D layer of a stack in parallel.

    This applies ``skimage.restoration.denoise_nl_means`` to each slice, so
    refer to the documentation of that function for further information.

    Parameters
    ----------
    im : ndarray
        The greyscale image with noise to be removed
    cores : int (optional)
        The number of cores to use for the processing. By default all
        available cores are used.
    axis : int
        The axis along which slices should be taken. This should
        correspond to the axis of rotation of the tomography stage, so if
        the sample was rotated about the z-axis, then use ``axis=2``.
    patch_size : int
        Size of patches used for denoising
    patch_distance : int
        Maximal distance in pixels where to search patches used for
        denoising.
    h : float
        Cut-off distance (in gray levels). The higher ``h``, the more
        permissive one is in accepting patches. A higher h results in a
        smoother image, at the expense of blurring features. For a
        Gaussian noise of standard deviation sigma, a rule of thumb is to
        choose the value of ``h`` to be sigma of slightly less.

    Notes
    -----
    The quality of the result depends on ``patch_size``,
    ``patch_distance``, ``h``, and ``sigma``.  It is recommended to
    experiment with a single slice first until a suitable set of
    parameters is found.

    Each slice in the stack is adjusted to have the same histogram and
    intensity.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/nl_means_layered.html>`_
    to view online example.

    """

    @dask.delayed
    def apply_func(func, **kwargs):
        return func(**kwargs)

    temp = np.copy(im)
    for i in range(im.shape[2]):
        temp[:, :, i] = match_histograms(temp[:, :, i], temp[:, :, 0],
                                         multichannel=False)
    p2, p98 = np.percentile(temp, (2, 98))
    temp = rescale_intensity(temp, in_range=(p2, p98))
    temp = temp / temp.max()
    sigma_est = np.mean(estimate_sigma(temp[:, :, 0], multichannel=False))

    kw = {'image': temp,
          'patch_size': patch_size,
          'patch_distance': patch_distance,
          'h': h * sigma_est,
          'multichannel': False,
          'fast_mode': True}

    temp = np.swapaxes(temp, 0, axis)
    results = []
    for i in range(im.shape[2]):
        layer = temp[i, ...]
        kw["image"] = layer
        t = apply_func(func=denoise_nl_means, **kw)
        results.append(t)
    # with ProgressBar():
    #     ims = dask.compute(results, num_workers=cores)[0]
    ims = dask.compute(results, num_workers=cores)[0]
    result = np.array(ims)
    result = np.swapaxes(result, 0, axis)
    return result
