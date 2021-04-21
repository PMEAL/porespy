import numpy as np
import scipy.ndimage as spim
import scipy.signal as spsig
from tqdm import tqdm
import skimage
from skimage.morphology import square
import imageio
from porespy.filters import fftmorphology
import matplotlib.pyplot as plt
import math
from porespy import settings


def boxcount(im, bins=10, d_min=1, d_max=None):
    r"""
    Calculates fractal dimension of the image.

    This function scans the image using a measuring element
    and counts the number of pixels or voxels (for a 2-D or 3-D image,
    respectively) that lay over the edge of a pore [1]_.

    Parameters
    ----------
    im : ND-array
        The image of the porous material.
    bins : int or ND-array
        The number of times to iteratively scan the image. The default is 10.
        If an array is entered, this is directly taken as the measuring
        element sizes.
    d_min : int
        The size of the smallest measuring element. Measuring elements are
        taken as a square or a cube.
    d_max : int
        The size of the largest measuring element. Measuring elements are
        taken as a square or a cube.

    Returns
    -------
    Ds : ND-array
        The measuring element sizes. This array has 'bins'-number of elements.
    N : ND-array
        The number of pixels or voxels that lay over the edge of a pore,
        corresponding to each measuring element size.
    slope: ND-array
        The gradient of N. This array has the same number of elements as Ds and N.


    References
    ----------
    [1] See Boxcounting on `Wikipedia <https://en.wikipedia.org/wiki/Box_counting>`_

    """
    from collections import namedtuple

    im = np.array(im, dtype=bool)

    if (len(im.shape) != 2 and len(im.shape) != 3):
        raise Exception('Image must be 2-dimensional or 3-dimensional')

    if isinstance(bins, int):
        Ds = np.unique(np.logspace(1, np.log10(min(im.shape)), bins).astype(int))
    else:
        Ds = np.array(bins).astype(int)

    N = []
    for d in tqdm(Ds, **settings.tqdm):
        result = 0
        for i in range(0, im.shape[0], d):
            for j in range(0, im.shape[1], d):
                if len(im.shape) == 2:
                    temp = im[i:i+d, j:j+d]
                    result += np.any(temp)
                    result -= np.all(temp)
                else:
                    for k in range(0, im.shape[2], d):
                        temp = im[i:i+d, j:j+d, k:k+d]
                        result += np.any(temp)
                        result -= np.all(temp)
        N.append(result)
    slope = -1*np.gradient(np.log(np.array(N)), np.log(Ds))
    data = namedtuple("data", ("size", "count", "slope"))
    data.size = Ds
    data.count = N
    data.slope = slope
    return data
