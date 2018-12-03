import scipy as sp
import matplotlib.pyplot as plt


def show_planes(planes, overlaid=False):
    r"""
    """
    plt.subplot(1, 3, 1)
    plt.imshow(planes[0])
    plt.subplot(1, 3, 2)
    plt.imshow(planes[1])
    plt.subplot(1, 3, 3)
    plt.imshow(planes[2])


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
