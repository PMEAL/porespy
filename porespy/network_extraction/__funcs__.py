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


def find_peaks2(dt, r, minpeak=0):
    if dt.ndim == 2:
        from skimage.morphology import disk as ball
        from skimage.morphology import square as cube
    else:
        from skimage.morphology import ball, cube
    dt_temp = dt + (dt == 0)*minpeak
    mx = spim.maximum_filter(input=dt_temp, size=2*r+1)
    peaks = (dt == mx)*(dt > 0)
    return peaks


def find_peaks(dt, r=3):
    r"""
    Returns all local maxima in the distance transform

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
    An ND-array of booleans with ``True`` values at the location of any local
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
        Passing in a pre-existing distance transform saves time, since one is
        calculated by the function if a Boolean array is passed.

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
        print('Boolean image received, applying distance transform')
        im = spim.distance_transform_edt(im)
    if peaks.dtype == bool:
        print('Boolean peaks received, applying labeling')
        peaks = spim.label(input=peaks)[0]
    regions = watershed(-im, markers=peaks)
    return regions
