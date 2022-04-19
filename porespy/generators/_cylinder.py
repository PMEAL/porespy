import numpy as np
from edt import edt


def cylindrical_plug(shape, r=None, axis=2):
    r"""
    Generates a cylindrical plug suitable for use as a mask on a tomogram

    Parameters
    ----------
    shape : array_like
        The shape of the image to create.  Can be 3D, or 2D in which case
        a circle is returned.
    r : int (optional)
        The diameter of the cylinder to create. If not provided then the
        largest possible radius is used to fit within the image.
    axis : int
        The direction along with the cylinder's axis of rotation should be
        oriented.  The default is 2, which is the z-direction.

    Returns
    -------
    cylinder : ndarray
        A boolean image of the size given by ``shape`` with ``True``'s
        inside the cylinder and ``False``'s outside.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/generators/reference/cylindrical_plug.html>`_
    to view online example.

    """
    shape = np.array(shape, dtype=int)
    axes = np.array(list(set([0, 1, 2]).difference(set([axis]))), dtype=int)
    if len(shape) == 3:
        im2d = np.ones(shape=shape[axes])
        im2d[int(shape[axes[0]]/2), int(shape[axes[1]]/2)] = 0
        dt = edt(im2d)
        if r is None:
            r = int(min(shape[axes])/2)
        circ = dt < r
        tile_ax = [1, 1, 1]
        tile_ax[axis] = shape[axis]
        circ = np.expand_dims(circ, axis)
        cyl = np.tile(circ, tile_ax)
    if len(shape) == 2:
        im2d = np.ones(shape=shape)
        im2d[int(shape[0]/2), int(shape[1]/2)] = 0
        dt = edt(im2d)
        if r is None:
            r = int(min(shape[axes])/2)
        cyl = dt < r
    return cyl
