import numpy as np
import scipy.ndimage as spim
from auto_all import start_all, end_all
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection


start_all()


def show_3D(im):  # pragma: no cover
    r"""
    Rotates a 3D image and creates an angled view for rough 2D visualization.

    Because it rotates the image it can be slow for large images, so is
    mostly meant for rough checking of small prototype images.

    Parameters
    ----------
    im : ndarray
        The 3D array to be viewed from an angle

    Returns
    -------
    image : ndarray
        A 2D veiw of the given 3D image

    Notes
    -----
    Although this is meant to be *quick* it can still take on the order of
    minutes to render very large images.  It uses ``scipy.ndimage.rotate``
    with no interpolation to view the 3D image from an angle, then casts the
    result into a 2D projection.

    """
    im = ~np.copy(im)
    if im.ndim < 3:
        raise Exception('show_3D only applies to 3D images')
    im = spim.rotate(input=im, angle=22.5, axes=[0, 1], order=0)
    im = spim.rotate(input=im, angle=45, axes=[2, 1], order=0)
    im = spim.rotate(input=im, angle=-17, axes=[0, 1], order=0, reshape=False)
    mask = im != 0
    view = np.where(mask.any(axis=2), mask.argmax(axis=2), 0)
    view = view.max() - view
    f = view.max()/5
    view[view == view.max()] = -f
    view = (view + f)**2
    return view


def show_planes(im, spacing=10):  # pragma: no cover
    r"""
    Create a quick montage showing a 3D image in all three directions

    Parameters
    ----------
    im : ndarray
        A 3D image of the porous material
    spacing : int (optional, default=10)
        Controls the amount of space to put between each panel

    Returns
    -------
    image : ndarray
        A 2D array containing the views.  This single image can be viewed using
        ``matplotlib.pyplot.imshow``.

    """
    s = spacing
    if np.squeeze(im.ndim) < 3:
        raise Exception('This view is only necessary for 3D images')
    x, y, z = (np.array(im.shape)/2).astype(int)
    im_xy = im[:, :, z]
    im_xz = im[:, y, :]
    im_yz = np.rot90(im[x, :, :])

    new_x = im_xy.shape[0] + im_yz.shape[0] + s

    new_y = im_xy.shape[1] + im_xz.shape[1] + s

    new_im = np.zeros([new_x + 2*s, new_y + 2*s], dtype=im.dtype)

    # Add xy image to upper left corner
    new_im[s:im_xy.shape[0] + s,
           s:im_xy.shape[1] + s] = im_xy
    # Add xz image to lower left coner
    x_off = im_xy.shape[0] + 2*s
    y_off = im_xy.shape[1] + 2*s
    new_im[s:s + im_xz.shape[0],
           y_off:y_off + im_xz.shape[1]] = im_xz
    new_im[x_off:x_off + im_yz.shape[0],
           s:s + im_yz.shape[1]] = im_yz

    return new_im


def sem(im, axis=0):  # pragma: no cover
    r"""
    Simulates an SEM photograph looking into the porous material.

    Features are colored according to their depth into the image, so
    darker features are further away.

    Parameters
    ----------
    im : array_like
        ndarray of the porous material with the solid phase marked as 1 or
        True
    axis : int
        Specifes the axis along which the camera will point.

    Returns
    -------
    image : ndarray
        A 2D greyscale image suitable for use in matplotlib's ``imshow``
        function.

    """
    im = np.array(~im, dtype=int)
    if axis == 1:
        im = np.transpose(im, axes=[1, 0, 2])
    if axis == 2:
        im = np.transpose(im, axes=[2, 1, 0])
    t = im.shape[0]
    depth = np.reshape(np.arange(0, t), [t, 1, 1])
    im = im*depth
    im = np.amax(im, axis=0)
    return im


def xray(im, axis=0):  # pragma: no cover
    r"""
    Simulates an X-ray radiograph looking through the porous material.

    The resulting image is colored according to the amount of attenuation an
    X-ray would experience, so regions with more solid will appear darker.

    Parameters
    ----------
    im : array_like
        ndarray of the porous material with the solid phase marked as 1 or
        True

    axis : int
        Specifes the axis along which the camera will point.

    Returns
    -------
    image : ndarray
        A 2D greyscale image suitable for use in matplotlib\'s ```imshow```
        function.
    """
    im = np.array(~im, dtype=int)
    if axis == 1:
        im = np.transpose(im, axes=[1, 0, 2])
    if axis == 2:
        im = np.transpose(im, axes=[2, 1, 0])
    im = np.sum(im, axis=0)
    return im


end_all()
