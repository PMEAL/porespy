from scipy.signal import convolve

from porespy.tools import ps_rect

__all__ = [
    "local_diff",
]


def local_diff(vals, im, strel=None):
    r"""
    Computes the difference pixel and the average of it's neighbors.

    Parameters
    ----------
    vals : ndarray
        The array containing the values of interest
    im : ndarray
        A boolean image of the domain
    strel : ndarray, optional
        The struturing element to use when doing the convolution to find the
        neighbor values.  This defines the size and shape of the area searched.
        If not provided then a 3**ndim cube is used.

    Returns
    -------
    diff : ndarray
        An array containing the difference between each pixel and the average
        of it's neighbors.  The result is not normalized or squared so may
        contain negative values which might be of interest if the direction of the
        difference is relevant.
    """
    if strel is None:
        strel = ps_rect(w=3, ndim=im.ndim)
    numer = convolve(vals*im, strel, mode='same')
    denom = convolve(im*1.0, strel, mode='same')
    ave = numer/denom
    diff = ave - vals
    diff[~im] = 0
    return diff
