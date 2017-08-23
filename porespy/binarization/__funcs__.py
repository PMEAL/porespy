from skimage import filters
import porespy as ps
import matplotlib.pyplot as plt
import scipy.ndimage as spim
import scipy as sp
from skimage.morphology import ball, disk, square, cube
from skimage.segmentation import clear_border


def simple_otsu(im, trim_solid=True):
    r"""
    Uses Otsu's method to find a threshold, then uses binary opening and
    closing to remove noise.

    Parameters
    ----------
    im : ND-image
        The greyscale image of the porous medium to be binarized.

    trim_solid : Boolean
        If True (default) then all solid voxels not connected to an image
        boundary are trimmed.

    Returns
    -------
    An ND-image the same size as ``im`` but with True and False values
    indicating the binarized or segmented phases.

    Examples
    --------

    >>> im = ps.generators.blobs([300, 300], porosity=0.5)
    >>> im = ps.generators.add_noise(im)
    >>> im = spim.gaussian_filter(im, sigma=1)
    >>> im = simple_otsu(im)

    """
    if im.ndim == 2:
        ball = disk
        cube = square
    im = sp.pad(array=im, pad_width=1, mode='constant', constant_values=1)
    val = filters.threshold_otsu(im)
    im = im >= val
    # Remove speckled noise from void and solid phases
    im = spim.binary_closing(input=im, structure=ball(1))
    im = spim.binary_opening(input=im, structure=ball(1))
    # Clean up edges
    im = spim.binary_closing(input=im, structure=cube(3))
    im = spim.binary_opening(input=im, structure=cube(3))
    im = im[[slice(1, im.shape[d]-1) for d in range(im.ndim)]]
    temp = clear_border(~im)
    im = im + temp
    return im
