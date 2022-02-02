import numpy as np


def unpad(im, pad_width):
    r"""
    Remove padding from a previously padded image given original pad widths

    Parameters
    ----------
    im : ndarray
        The padded image from which padding should be removed
    pad_width : int or list of ints
        The amount of padding previously added to each axis. This should be
        the same as the values used to add original padding. Refer to
        ``numpy.pad`` documentation for more details.

    Notes
    -----
    A use case for this is when using ``skimage.morphology.skeletonize_3d``
    to ensure that the skeleton extends beyond the edges of the image, but the
    padding should be subsequently removed.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/tools/reference/unpad.html>`_
    to view online example.

    """
    pad_width = np.asarray(pad_width).squeeze()

    if pad_width.ndim == 0:
        new_pad_width = []
        for r in range(0, len(im.shape)):
            new_pad_width.append(pad_width)
        pad_width = np.array(new_pad_width)

    if pad_width.ndim == 1:
        shape = im.shape - pad_width[0] - pad_width[1]
        if shape[0] < 1:
            shape = np.array(im.shape) * shape
        s_im = []
        for dim in range(im.ndim):
            lower_im = pad_width[0]
            upper_im = shape[dim] + pad_width[0]
            s_im.append(slice(int(lower_im), int(upper_im)))

    if pad_width.ndim == 2:
        shape = np.asarray(im.shape)
        s_im = []
        for dim in range(im.ndim):
            shape[dim] = im.shape[dim] - pad_width[dim][0] - pad_width[dim][1]
            lower_im = pad_width[dim][0]
            upper_im = shape[dim] + pad_width[dim][0]
            s_im.append(slice(int(lower_im), int(upper_im)))

    return im[tuple(s_im)]
