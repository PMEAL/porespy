import numpy as np


__all__ = ['faces', 'borders']


def faces(shape, inlet=None, outlet=None):
    r"""
    Generate an image with ``True`` values on the specified ``inlet`` and
    ``outlet`` faces

    Parameters
    ----------
    shape : list
        The ``[x, y, z (optional)]`` shape to generate. This will likely
        be obtained from ``im.shape`` where ``im`` is the image for which
        an array of faces is required.
    inlet : int
        The axis where the faces should be added (e.g. ``inlet=0`` will
        put ``True`` values on the ``x=0`` face). A value of ``None``
        (default) bypasses the addition of inlets.
    outlet : int
        Same as ``inlet`` except for the outlet face. This is optional. It
        can be be applied at the same time as ``inlet``, instead of
        ``inlet`` (if ``inlet`` is set to ``None``), or ignored
        (if ``outlet = None``).

    Returns
    -------
    faces : ndarray
        A boolean image of the given ``shape`` with ``True`` values on the
        specified ``inlet`` and/or ``outlet`` face(s).

    Examples
    --------
    `Click here
    <https://porespy.org/examples/generators/reference/faces.html>`_
    to view online example.

    """
    im = np.zeros(shape, dtype=bool)
    # Parse inlet and outlet
    if inlet is not None:
        im = np.swapaxes(im, 0, inlet)
        im[0, ...] = True
        im = np.swapaxes(im, 0, inlet)
    if outlet is not None:
        im = np.swapaxes(im, 0, outlet)
        im[-1, ...] = True
        im = np.swapaxes(im, 0, outlet)
    if (inlet is None) and (outlet is None):
        raise Exception('Both inlet and outlet were given as None')
    return im


def borders(shape, thickness=1, mode='edges'):
    r"""
    Creates an array of specified size with corners, edges or faces
    labelled as ``True``.

    This can be used as mask to manipulate values laying on the perimeter
    of an image.

    Parameters
    ----------
    shape : array_like
        The shape of the array to return.  Can be either 2D or 3D.
    thickness : scalar (default is 1)
        The number of pixels/voxels layers to place along perimeter.
    mode : string
        The type of border to create.  Options are 'faces', 'edges'
        (default) and 'corners'.  In 2D 'corners' and 'edges' give the
        same result.

    Returns
    -------
    image : ndarray
        An ndarray of specified shape with ``True`` values at the
        perimeter and ``False`` elsewhere

    Examples
    --------
    `Click here
    <https://porespy.org/examples/generators/reference/borders.html>`_
    to view online example.

    """
    ndims = len(shape)
    t = thickness
    border = np.ones(shape, dtype=bool)
    if mode == 'faces':
        if ndims == 2:
            border[t:-t, t:-t] = False
        if ndims == 3:
            border[t:-t, t:-t, t:-t] = False
    elif mode == 'edges':
        if ndims == 2:
            border[t:-t, 0::] = False
            border[0::, t:-t] = False
        if ndims == 3:
            border[0::, t:-t, t:-t] = False
            border[t:-t, 0::, t:-t] = False
            border[t:-t, t:-t, 0::] = False
    elif mode == 'corners':
        if ndims == 2:
            border[t:-t, 0::] = False
            border[0::, t:-t] = False
        if ndims == 3:
            border[t:-t, 0::, 0::] = False
            border[0::, t:-t, 0::] = False
            border[0::, 0::, t:-t] = False
    return border
