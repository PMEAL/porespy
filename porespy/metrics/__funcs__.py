import scipy as sp
from skimage.segmentation import clear_border
from scipy.ndimage import label


def porosity(im):
    r'''
    '''
    e = sp.sum(im > 0)/im.size
    return e


def apply_chords(im, spacing=0, axis=0, trim_edges=True):
    r"""
    Adds chords to the void space in the specified direction.  The chords are
    separated by 1 voxels plus the provided spacing.

    Parameters
    ----------
    im : ND-array
        A 2D image of the porous material with void marked as True.

    spacing : int (default = 0)
        Chords are automatically separed by 1 voxel and this argument increases
        the separation.

    axis : int (default = 0)
        The axis along which the chords are drawn.

    trim_edges : bool (default = True)
        Whether or not to remove chords that touch the edges of the image.
        These chords are artifically shortened, so skew the chord length
        distribution

    Returns
    -------
    An ND-array of the same size as ```im``` with True values indicating the
    chords.

    See Also
    --------
    apply_chords_3D

    """
    if spacing < 0:
        raise Exception('Spacing cannot be less than 0')
    dims1 = sp.arange(0, im.ndim)
    dims2 = sp.copy(dims1)
    dims2[axis] = 0
    dims2[0] = axis
    im = sp.moveaxis(a=im, source=dims1, destination=dims2)
    im = sp.atleast_3d(im)
    ch = sp.zeros_like(im, dtype=bool)
    ch[:, ::4+2*spacing, ::4+2*spacing] = 1
    chords = im*ch
    chords = sp.squeeze(chords)
    if trim_edges:
        temp = clear_border(label(chords == 1)[0]) > 0
        chords = temp*chords
    chords = sp.moveaxis(a=chords, source=dims1, destination=dims2)
    return chords


def apply_chords_3D(im, spacing=0, trim_edges=True):
    r"""
    Adds chords to the void space in all three principle directions.  The
    chords are seprated by 1 voxel plus the provided spacing.  Chords in the X,
    Y and Z directions are labelled 1, 2 and 3 resepctively.

    Parameters
    ----------
    im : ND-array
        A 3D image of the porous material with void space marked as True.

    spacing : int (default = 0)
        Chords are automatically separed by 1 voxel on all sides, and this
        argument increases the separation.

    trim_edges : bool (default = True)
        Whether or not to remove chords that touch the edges of the image.
        These chords are artifically shortened, so skew the chord length
        distribution

    Returns
    -------
    An ND-array of the same size as ```im``` with values of 1 indicating
    x-direction chords, 2 indicating y-direction chords, and 3 indicating
    z-direction chords.

    """
    if im.ndim < 3:
        raise Exception('Must be a 3D image to use this function')
    if spacing < 0:
        raise Exception('Spacing cannot be less than 0')
    ch = sp.zeros_like(im, dtype=int)
    ch[:, ::4+2*spacing, ::4+2*spacing] = 1  # X-direction
    ch[::4+2*spacing, :, 2::4+2*spacing] = 2  # Y-direction
    ch[2::4+2*spacing, 2::4+2*spacing, :] = 3  # Z-direction
    chords = ch*im
    if trim_edges:
        temp = clear_border(label(chords > 0)[0]) > 0
        chords = temp*chords
    return chords


def size_distribution(im, bins=None, return_im=False):
    r"""
    Given an image containing the size of the feature to which each voxel
    belongs (as produced by ```simulations.feature_size```), this determines
    the total volume of each feature and returns a tuple containing *radii* and
    *counts* suitable for plotting.

    Parameters
    ----------
    im : array_like
        An array containing the local feature size

    Returns
    -------
    Tuple containing radii, counts
        Two arrays containing the radii of the largest spheres, and the number
        of voxels that are encompassed by spheres of each radii.

    Notes
    -----
    The term *foreground* is used since this function can be applied to both
    pore space or the solid, whichever is set to True.

    """
    inds = sp.where(im > 0)
    bins = sp.unique(im)[1:]
    hist = sp.histogram(a=im[inds], bins=bins)
    radii = hist[1][0:-1]
    counts = hist[0]
    return radii, counts
