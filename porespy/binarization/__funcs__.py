from skimage import filters
import porespy as ps
import matplotlib.pyplot as plt
import scipy.ndimage as spim
import scipy as sp
from skimage.morphology import ball, disk, square, cube


def goto(im):
    r"""
    Uses Otsu's method to find a threshold, then uses binary opening and
    closing to remove noise.

    Parameters
    ----------
    im : ND-image
        The greyscale image of the porous medium to be binarized.

    Returns
    -------
    An ND-image the same size as ``im`` but with True and False values
    indicating the binarized or segmented phases.

    """
    if im.ndim == 2:
        ball = disk
        cube = square
    val = filters.threshold_otsu(im)
    im = im >= val
    # Remove speckled noise from void and solid phases
    im = spim.binary_closing(input=im, structure=ball(1))
    im = spim.binary_opening(input=im, structure=ball(1))
    # Clean up edges
    im = spim.binary_closing(input=im, structure=cube(3))
    im = spim.binary_opening(input=im, structure=cube(3))
    return im
