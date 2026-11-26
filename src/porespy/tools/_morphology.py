import numpy as np
from skimage.morphology import footprint_rectangle

from ._utils import get_edt

__all__ = [
    'get_strel',
    'ball',
    'disk',
    'cube',
    'square',
    'ps_disk',
    'ps_ball',
    'ps_round',
    'ps_rect',
]


edt = get_edt()


def ball(r):
    se = np.ones([r*2+1]*3, dtype=bool)
    se[r, r, r] = False
    se = edt(se) <= r
    return se


def disk(r):
    se = np.ones([r*2+1]*2, dtype=bool)
    se[r, r] = False
    se = edt(se) <= r
    return se


def cube(w):
    se = np.ones([w, w, w], dtype=bool)
    return se


def square(w):
    se = np.ones([w, w], dtype=bool)
    return se


def get_strel():
    se = {2: {'min': disk(1),
              'max': square(3)},
          3: {'min': ball(1),
              'max': cube(3)}}
    return se


def ps_disk(r, smooth=True):
    r"""
    Creates circular disk structuring element for morphological operations

    Parameters
    ----------
    r : float or int
        The desired radius of the structuring element
    smooth : boolean
        Indicates whether the faces of the sphere should have the little
        nibs (``True``) or not (``False``, default)

    Returns
    -------
    disk : ndarray
        A 2D numpy bool array of the structring element

    Examples
    --------
    `Click here
    <https://porespy.org/examples/tools/reference/ps_disk.html>`__
    to view online example.

    """
    disk = ps_round(r=r, ndim=2, smooth=smooth)
    return disk


def ps_ball(r, smooth=True):
    r"""
    Creates spherical ball structuring element for morphological operations

    Parameters
    ----------
    r : scalar
        The desired radius of the structuring element
    smooth : boolean
        Indicates whether the faces of the sphere should have the little
        nibs (``True``) or not (``False``, default)

    Returns
    -------
    ball : ndarray
        A 3D numpy array of the structuring element

    Examples
    --------
    `Click here
    <https://porespy.org/examples/tools/reference/ps_ball.html>`__
    to view online example.

    """
    ball = ps_round(r=r, ndim=3, smooth=smooth)
    return ball


def ps_round(r, ndim, smooth=True):
    r"""
    Creates round structuring element with the given radius and dimensionality

    Parameters
    ----------
    r : scalar
        The desired radius of the structuring element
    ndim : int
        The dimensionality of the element, either 2 or 3.
    smooth : boolean
        Indicates whether the faces of the sphere should have the little
        nibs (``True``) or not (``False``, default)

    Returns
    -------
    strel : ndarray
        A 3D numpy array of the structuring element

    Examples
    --------
    `Click here
    <https://porespy.org/examples/tools/reference/ps_round.html>`__
    to view online example.

    """
    rad = int(np.ceil(r))
    other = np.ones([2*rad + 1 for i in range(ndim)], dtype=bool)
    other[tuple(rad for i in range(ndim))] = False
    if smooth:
        ball = edt(other) < r
    else:
        ball = edt(other) <= r
    return ball


def ps_rect(w, ndim):
    r"""
    Creates rectilinear structuring element with the given size and
    dimensionality

    Parameters
    ----------
    w : scalar
        The desired width of the structuring element
    ndim : int
        The dimensionality of the element, either 2 or 3.

    Returns
    -------
    strel : ndarray
        A numpy array of the structuring element

    Examples
    --------
    `Click here
    <https://porespy.org/examples/tools/reference/ps_rect.html>`__
    to view online example.

    """
    return footprint_rectangle((w,) * ndim)
