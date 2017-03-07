import scipy as sp


def porosity(im):
    r'''
    '''
    sv = sp.sum(im == 0)
    pv = sp.sum(im > 0)
    e = pv/(pv+sv)
    return e


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
