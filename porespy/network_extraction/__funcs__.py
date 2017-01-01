import scipy as sp
import scipy.ndimage as spim
import scipy.spatial as sptl
from skimage.morphology import reconstruction, watershed
from skimage.segmentation import find_boundaries
from skimage.morphology import disk, square, ball, cube


def maxima_filt(im, strel):
    def f(vals):
        vals[24] *= 0.9999
        return sp.amax(vals)
    a = spim.filters.generic_filter(input=im, function=f, footprint=disk(3))
    return a


def all_peaks(dt, r=3):
    r"""
    Returns all local maxima in the distance transform.

    Parameters
    ----------
    dt : ND-array
        The distance transform of the pore space.  This may be calculated and
        filtered using any means desired.

    r : scalar
        The radius of the structuring element used in the maximum filter.  This
        controls the localness of any maxima. The default is 3 voxels, which
        means that a voxel is considered maximal if no other voxel within
        a radius of 3 voxels has a higher number.

    Returns
    -------
    An ND-array of booleans with True values at the location of any local
    maxima.
    """
    dt = dt.squeeze()
    im = dt > 0
    if im.ndim == 2:
        ball = disk
        cube = square
    elif im.ndim == 3:
        ball = ball
        cube = cube
    else:
        raise Exception("only 2-d and 3-d images are supported")
    mx = spim.maximum_filter(dt + 2*(~im), footprint=ball(r))
    peaks = (dt == mx)*im
    return peaks


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


def bounding_box_indices(roi):
    r"""
    Given the ND-coordinates of voxels defining the region of interest, the
    indices of a bounding box are returned.
    """
    maxs = sp.amax(roi, axis=1)
    mins = sp.amin(roi, axis=1)
    x = []
    x.append(slice(mins[0], maxs[0]))
    x.append(slice(mins[1], maxs[1]))
    if sp.shape(roi)[0] == 3:  # If image if 3D, add third coordinate
        x.append(slice(mins[2], maxs[2]))
    inds = tuple(x)
    return inds
