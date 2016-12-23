import sys
import scipy as sp
import scipy.ndimage as spim
from collections import namedtuple
import matplotlib.pyplot as plt


def porosimetry(im, npts=25, sizes=None, inlets=None):
    r"""
    Simulates a porosimetry experiment on a binary image.  This function is
    equivalent to morphological image opening or the full morphology approach.

    Parameters
    ----------
    im : ND-array
        The binary image of the pore space.

    npts : scalar
        The number of invasion points to simulate.  Points will be generated
        spanning the range of sizes in the distance transform.  The default is
        25 points

    sizes : array_like
        The sizes to invade.  Use this argument instead of ``npts`` for more
        control of the range and spacing of points.

    inlets : ND-array, boolean
        A boolean mask with True values indicating where the invasion enters
        the image.  By default all faces are considered inlets, akin to a
        mercury porosimetry experiment.

    Returns
    -------
    A single ND-array with numerical values in each element indicating at
    which size it was invaded at.  The obtain the invading fluid configuration
    for invasion of all accessible locations greater than R, use boolean logic
    such as ``invasion_pattern = returned_array >= R``.

    Notes
    -----
    Although this method is equivalent to morphological image opening, it is
    done using distance transforms instead of convolution filters.  This
    approach is much faster for than using dilations and erosions when the
    structuring element is large.

    """
    dt = spim.distance_transform_edt(im)
    if inlets is None:
        inlets = sp.ones_like(im, dtype=bool)
        if im.ndim == 2:
            inlets[1:-2, 1:-2] = False
        elif im.ndim == 3:
            inlets[1:-2, 1:-2,1:-2] = False
    if npts is not None:
        sizes = sp.logspace(sp.log10(sp.amax(dt)), 0.1, npts)
    imresults = sp.zeros(sp.shape(im))
    for r in sizes:
        imtemp = dt >= r
        labels = spim.label(imtemp + inlets)[0]
        inlet_labels = sp.unique(labels[inlets])
        imtemp = sp.in1d(labels.ravel(), inlet_labels)
        imtemp = sp.reshape(imtemp, im.shape)
        im = spim.distance_transform_edt(~(imtemp^inlets)) <= r
        imresults[(imresults == 0)*im] = r
    return imresults
