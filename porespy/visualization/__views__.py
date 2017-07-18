import scipy as sp
import scipy.ndimage as spim


def sem(im, direction='X'):
    r"""
    Simulates an SEM photograph looking into the porous material in the
    specified direction.  Features are colored according to their depth into
    the image, so darker features are further away.

    Parameters
    ----------
    im : array_like
        ND-image of the porous material with the solid phase marked as 1 or
        True

    direction : string
        Specify the axis along which the camera will point.  Options are
        'X', 'Y', and 'Z'.

    Returns
    -------
    A 2D greyscale image suitable for use in matplotlib\'s ```imshow```
    function.
    """
    im = sp.array(~im, dtype=int)
    if direction in ['Y', 'y']:
        im = sp.transpose(im, axes=[1, 0, 2])
    if direction in ['Z', 'z']:
        im = sp.transpose(im, axes=[2, 1, 0])
    t = im.shape[0]
    depth = sp.reshape(sp.arange(0, t), [t, 1, 1])
    im = im*depth
    im = sp.amax(im, axis=0)
    return im


def xray(im, direction='X'):
    r"""
    Simulates an X-ray radiograph looking through the porouls material in the
    specfied direction.  The resulting image is colored according to the amount
    of attenuation an X-ray would experience, so regions with more solid will
    appear darker.

    Parameters
    ----------
    im : array_like
        ND-image of the porous material with the solid phase marked as 1 or
        True

    direction : string
        Specify the axis along which the camera will point.  Options are
        'X', 'Y', and 'Z'.

    Returns
    -------
    A 2D greyscale image suitable for use in matplotlib\'s ```imshow```
    function.
    """
    im = sp.array(~im, dtype=int)
    if direction in ['Y', 'y']:
        im = sp.transpose(im, axes=[1, 0, 2])
    if direction in ['Z', 'z']:
        im = sp.transpose(im, axes=[2, 1, 0])
    im = sp.sum(im, axis=0)
    return im


def local_thickness(im):
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

    return_im : boolean
        If true, the image with the radius in each voxel will be return as
        well.

    Returns
    -------
    An image the same shape as ```im``` with color corresponding to the largest
    sphere which overlaps a given pixel.

    Notes
    -----
    The term *foreground* is used since this function can be applied to both
    pore space or the solid, whichever is set to True.

    """
    from skimage.morphology import cube
    if im.ndim == 2:
        from skimage.morphology import square as cube
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
    return im_new
