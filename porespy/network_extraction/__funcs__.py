import scipy as sp
import scipy.ndimage as spim
from skimage.morphology import watershed


def align_image_with_openpnm(im):
    r"""
    Rotates an image to agree with the coordinates used in OpenPNM.  It is
    unclear why they are not in agreement to start with.  This is necessary
    for overlaying the image and the network in Paraview.

    Parameters
    ----------
    im : ND-array
        The image to be rotated.  Can be the Boolean image of the pore space or
        any other image of interest.

    Returns
    -------
    Returns the image rotated accordingly.
    """
    if im.ndim == 2:
        im = (sp.swapaxes(im, 1, 0))
        im = im[-1::-1, :]
    elif im.ndim == 3:
        im = (sp.swapaxes(im, 2, 0))
        im = im[:, -1::-1, :]
    return im


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
