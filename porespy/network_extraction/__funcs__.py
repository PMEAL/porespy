import scipy as sp
import scipy.ndimage as spim
import scipy.spatial as sptl
from skimage.morphology import reconstruction, watershed
from skimage.segmentation import find_boundaries
from skimage.morphology import disk, square, ball, cube


def partition_pore_space(dt, peaks):
    r"""
    Applies a watershed segmentation to partition the distance transform into
    discrete pores.

    Parameters
    ----------
    dt : ND-array
        The distance transform of the pore space with maximum values located
        at the pore centers.  The function inverts this array to convert peaks
        to valleys for the sake of the watersheding.

    peaks : ND-array
        An array the same size as ``dt`` indicating where the pore centers are.
        If boolean, the peaks are labeled; if numeric then it's assumed that
        peaks have already been given desired labels.

    Returns
    -------
    A ND-array the same size as ``dt`` with regions belonging to each peak
    labelled.

    Notes
    -----
    Find the appropriate ``peaks`` array is the tricky part.  In principle,
    each local maxima in the distance transform is a pore center, but due to
    loss of fidelity in the voxel representation this leads to many extraneous
    peaks.  Several functions are available to help select peaks which can then
    be passed to this function.

    See Also
    --------
    SNOW_peaks

    Examples
    --------
    >>> import porespy as ps
    """
    if peaks.dtype == bool:
        peaks = spim.label(input=peaks)[0]
    regions = watershed(-dt, markers=peaks)
    return regions


def trim_extrema(im, h, mode='maxima'):
    r"""
    This trims local extrema by a specified amount, essentially decapitating
    peaks or flooding valleys, or both.

    Parameters
    ----------
    im : ND-array
        The image whose extrema are to be removed

    h : scalar
        The height to remove from each peak or fill in each valley

    mode : string {'maxima' | 'minima' | 'extrema'}
        Specifies whether to remove maxima or minima or both

    Returns
    -------
    A copy of the input image with all the peaks and/or valleys removed.

    Notes
    -----
    This function is referred to as **imhmax** or **imhmin** in Mablab.
    """
    if mode == 'maxima':
        result = reconstruction(seed=im - h, mask=im, method='dilation')
    elif mode == 'minima':
        result = reconstruction(seed=im + h, mask=im, method='erosion')
    elif mode == 'extrema':
        result = reconstruction(seed=im - h, mask=im, method='dilation')
        result = reconstruction(seed=result + h, mask=result, method='erosion')
    return result
