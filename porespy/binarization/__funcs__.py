from skimage import filters
import porespy as ps
import matplotlib.pyplot as plt
import scipy.ndimage as spim
import scipy as sp
from skimage.morphology import ball, cube


def goto(im):
    r"""
    """
    val = filters.threshold_otsu(im)
    im = im >= val
    im = spim.binary_closing(input=im, structure=ball(1))
    im = spim.binary_opening(input=im, structure=ball(1))
    return im
