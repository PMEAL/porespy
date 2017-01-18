import scipy as sp
import scipy.ndimage as spim


def size_distribution(im, bins=None):
    r"""
    For each voxel, this functions calculates the radius of the largest sphere
    that both engulfs the voxel and fits entirely within the foreground. This
    is not the same as a simple distance transform, which finds the largest
    sphere that could be *centered* on each voxel.

    Parameters
    ----------
    im : array_like
        A binary image with the phase of interest set to True

    bins : scalar or array_like
        The number of bins to use when creating the histogram, or the
        specific values to use.  The default is to use 1 bin for each
        unique value found in the size distribution.

    Returns
    -------
    radii, counts
        Two arrays containing the radii of the largest spheres, and the number
        of voxels that are encompassed by spheres of each radii.

    Notes
    -----
    The term *foreground* is used since this function can be applied to both
    pore space or the solid, whichever is set to True.

    """
    from skimage.morphology import cube
    if im.ndim == 2:
        from  skimage.morphology import square as cube
    dt = spim.distance_transform_edt(im)
    sizes = sp.unique(sp.around(dt))
    im_new = sp.zeros_like(im, dtype=int)
    print("Performing Image Opening")
    print('0%|'+'-'*len(sizes)+'|100%')
    print('  |', end='')
    for r in sizes:
        print('|', end='')
        im_temp = dt >= r
        im_temp = spim.distance_transform_edt(~im_temp) <= r
        im_new[im_temp] = r
    print('|')
    # Trim outer edge of features to remove noise
    im_new = spim.binary_erosion(input=im, structure=cube(1))*im_new
    inds = sp.where(im_new > 0)
    if bins is None:
        bins = 10
    else:
        bins = sp.unique(im_new)[1:]
    hist = sp.histogram(a=im_new[inds], bins=bins)
    radii = hist[1][0:-1]
    counts = hist[0]
    return radii, counts
