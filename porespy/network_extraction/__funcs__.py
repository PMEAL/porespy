import scipy as sp
import scipy.ndimage as spim
import scipy.spatial as sptl
from skimage.morphology import reconstruction, watershed
from skimage.segmentation import find_boundaries
from porespy.tools import randomize_colors


def get_indices_from_slices(slices, shape, pad=1):
    r"""
    Given a list slices (one slice for each dimension), this function returns
    an set of coordinate matrices suitable for direct indexing into a Numpy
    array.

    Parameters
    ----------
    slices : a list or tuple contain slice objects
        There should be one slice object for each dimension.  This function can
        handle arbitrary dimensionality

    shape : array_like
        The shape of the image into which the slices point.  This is used
        to ensure that the ```pad``` does not extend outside the domain.

    pad : scalar
        Specify a large region than indicated by the slices (See Notes)

    Returns
    -------
    A set of coordinate matrices that can be used to extract a subset of a
    Numpy array directly.

    Note
    -----
    This function is redundant if ```pad``` is not needed since the slices
    can just be used directly to extract a subset of the image.  The main
    motivation of this function is allow extraction of an extended subset
    which includes bits of the neighboring regions.
    """
    pad = [pad]*len(shape)
    inds = []
    for s in slices:
        if s.start == 0:
            pad[0] = 0
        if s.stop == shape[dim]:
            pad[dim] = 0
        inds.append(range(s.start-pad, s.stop+pad))
    inds = sp.meshgrid(*inds)
    return inds


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
    if peaks.dtype == bool:
        peaks = spim.label(input=peaks)[0]
    regions = watershed(-dt, markers=peaks)
    regions = randomize_colors(regions)
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
