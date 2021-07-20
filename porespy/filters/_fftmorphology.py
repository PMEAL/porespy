import numpy as np
from scipy.signal import fftconvolve


def fftmorphology(im, strel, mode='opening'):
    r"""
    Perform morphological operations on binary images using fft approach for
    improved performance

    Parameters
    ----------
    im : ndarray
        The binary image on which to perform the morphological operation
    strel : ndarray
        The structuring element to use.  Must have the same dims as ``im``.
    mode : string
        The type of operation to perform.  Options are 'dilation', 'erosion',
        'opening' and 'closing'.

    Returns
    -------
    image : ndarray
        A copy of the image with the specified moropholgical operation applied
        using the fft-based methods available in ``scipy.signal.fftconvolve``.

    Notes
    -----
    This function uses ``scipy.signal.fftconvolve`` which *can* be more than
    10x faster than the standard binary morphology operation in
    ``scipy.ndimage``.  This speed up may not always be realized, depending
    on the scipy distribution used.

    Examples
    --------
    >>> import porespy as ps
    >>> from numpy import array_equal
    >>> import scipy.ndimage as spim
    >>> from skimage.morphology import disk
    >>> im = ps.generators.blobs(shape=[100, 100], porosity=0.8)

    Check that erosion, dilation, opening, and closing are all the same as
    the ``scipy.ndimage`` functions:

    >>> result = ps.filters.fftmorphology(im, strel=disk(5), mode='erosion')
    >>> temp = spim.binary_erosion(im, structure=disk(5))
    >>> array_equal(result, temp)
    True

    >>> result = ps.filters.fftmorphology(im, strel=disk(5), mode='dilation')
    >>> temp = spim.binary_dilation(im, structure=disk(5))
    >>> array_equal(result, temp)
    True

    >>> result = ps.filters.fftmorphology(im, strel=disk(5), mode='opening')
    >>> temp = spim.binary_opening(im, structure=disk(5))
    >>> array_equal(result, temp)
    True

    >>> result = ps.filters.fftmorphology(im, strel=disk(5), mode='closing')
    >>> temp = spim.binary_closing(im, structure=disk(5))
    >>> array_equal(result, temp)
    True

    """

    def erode(im, strel):
        t = fftconvolve(im, strel, mode='same') > (strel.sum() - 0.1)
        return t

    def dilate(im, strel):
        t = fftconvolve(im, strel, mode='same') > 0.1
        return t

    im = np.squeeze(im)

    # Perform erosion and dilation
    # The array must be padded with 0's so it works correctly at edges
    temp = np.pad(array=im, pad_width=1, mode='constant', constant_values=0)
    if mode.startswith('ero'):
        temp = erode(temp, strel)
    if mode.startswith('dila'):
        temp = dilate(temp, strel)

    # Remove padding from resulting image
    if im.ndim == 2:
        result = temp[1:-1, 1:-1]
    elif im.ndim == 3:
        result = temp[1:-1, 1:-1, 1:-1]

    # Perform opening and closing
    if mode.startswith('open'):
        temp = fftmorphology(im=im, strel=strel, mode='erosion')
        result = fftmorphology(im=temp, strel=strel, mode='dilation')
    if mode.startswith('clos'):
        temp = fftmorphology(im=im, strel=strel, mode='dilation')
        result = fftmorphology(im=temp, strel=strel, mode='erosion')

    return result
