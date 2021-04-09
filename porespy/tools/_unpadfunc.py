import porespy as ps
import numpy as np
import matplotlib.pyplot as plt
from porespy.tools import get_border, extend_slice, extract_subsection


def unpad(im, pad_width):
    r"""
    removes padding from an image then extracts the result corresponding to the original image shape.
    Parameters
    ----------
    im : ND-image
        The image to which ``func`` should be applied
    pad_width : int or list of ints
        The amount of padding on each axis.  Refer to ``numpy.pad``
        documentation for more details.
    Notes
    -----
    A use case for this is when using ``skimage.morphology.skeletonize_3d``
    to ensure that the skeleton extends beyond the edges of the image.
    """
    if type(pad_width) == int:
        new_pad_width = []
        for r in range(0, len(im.shape)):
            new_pad_width.append(pad_width)
        pad_width = new_pad_width

    if type(pad_width) == list and np.ndim(pad_width) == 1:
        pad_width = np.asarray(pad_width)
        shape = im.shape - pad_width[0] - pad_width[1]

        if shape[0] < 1:
            shape = np.array(im.shape) * shape
        s_im = []
        for dim in range(im.ndim):
            lower_im = pad_width[0]
            upper_im = shape[dim] + pad_width[0]
            s_im.append(slice(int(lower_im), int(upper_im)))

    if type(pad_width) == list and np.ndim(pad_width) == 2:
        pad_width = np.asarray(pad_width)
        shape = np.asarray(im.shape)
        s_im = []
        for dim in range(im.ndim):
            shape[dim] = im.shape[dim] - pad_width[dim][0] - pad_width[dim][1]
            lower_im = pad_width[dim][0]
            upper_im = shape[dim] + pad_width[dim][0]
            s_im.append(slice(int(lower_im), int(upper_im)))

    return im[tuple(s_im)]
