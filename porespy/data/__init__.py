# coding: utf-8

"""Standard test images.
"""

import os as _os

import numpy as np

from .. import data_dir


def _load_npz(f):
    """Load an image file located in the data directory.

    Parameters
    ----------
    f : string
        File name.

    Returns
    -------
    img : ndarray
        Image loaded from ``porespy.data_dir``.
    """

    return np.load(_os.path.join(data_dir, f)).items()[0][1]


def tau():
    """
    Returns
    -------
    img : ndarray
    Greek symbol tau

    """
    return _load_npz("tau.npz")


def epsilon():
    """
    Returns
    -------
    img : ndarray
    Greek symbol epsilon

    """

    return _load_npz("epsilon.npz")
