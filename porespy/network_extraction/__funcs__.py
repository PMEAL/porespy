import scipy as sp
import scipy.ndimage as spim
from skimage.morphology import reconstruction, watershed
from porespy.tools import randomize_colors


def align_images_with_openpnm(im):
    if im.ndim == 2:
        pass
    elif im.ndim == 3:
        im = (sp.swapaxes(im, 2, 0))
        im = im[:, -1::-1, :]
    return im


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
    from skimage.morphology import disk, square, ball, cube
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


def partition_pore_space(im, peaks):
    r"""
    Applies a watershed segmentation to partition the distance transform into
    discrete pores.

    Parameters
    ----------
    im : ND-array
        Either the Boolean array of the pore space or a distance tranform.
        Passing in a pre-exisint distance transform saves time, since one is
        is by the function if a Boolean array is passed.

    peaks : ND-array
        An array the same size as ``dt`` indicating where the pore centers are.
        If boolean, the peaks are labeled; if numeric then it's assumed that
        peaks have already been given desired labels.

    Returns
    -------
    A ND-array the same size as ``dt`` with regions belonging to each peak
    labelled.  The region number is randomized so that neighboring regions
    are contrasting colors for easier visualization.

    Notes
    -----
    Find the appropriate ``peaks`` array is the tricky part.  In principle,
    each local maxima in the distance transform is a pore center, but due to
    loss of fidelity in the voxel representation this leads to many extraneous
    peaks.  Several functions are available to help select peaks which can then
    be passed to this function.

    See Also
    --------
    snow

    """
    print('_'*60)
    print('Partitioning Pore Space using Marker Based Watershed')
    if im.dtype == bool:
        im = spim.distance_transform_edt(im)
    if peaks.dtype == bool:
        peaks = spim.label(input=peaks)[0]
    regions = watershed(-im, markers=peaks)
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
