import sys
import scipy as sp
import scipy.ndimage as spim
from collections import namedtuple
import matplotlib.pyplot as plt


def feature_size(im, bins=None):
    r"""
    For each voxel, this functions calculates the radius of the largest sphere
    that both engulfs the voxel and fits entirely within the foreground. This
    is not the same as a simple distance transform, which finds the largest
    sphere that could be *centered* on each voxel.

    Parameters
    ----------
    im : array_like
        A binary image with the phase of interest set to True

    bins : scalar or array_like
        The number of bins to use when creating the histogram, or the
        specific values to use.  The default is to use 1 bin for each
        unique value found in the size distribution.

    Returns
    -------
    An image with the pore size values in each voxel

    Notes
    -----
    The term *foreground* is used since this function can be applied to both
    pore space or the solid, whichever is set to True.

    """
    from skimage.morphology import cube
    if im.ndim == 2:
        from  skimage.morphology import square as cube
    dt = spim.distance_transform_edt(im)
    sizes = sp.unique(sp.around(dt))
    im_new = sp.zeros_like(im, dtype=int)
    print("Performing Image Opening")
    print('0%|'+'-'*len(sizes)+'|100%')
    print('  |', end='')
    for r in sizes:
        print('|', end='')
        im_temp = dt >= r
        im_temp = spim.distance_transform_edt(~im_temp) <= r
        im_new[im_temp] = r
    print('|')
    # Trim outer edge of features to remove noise
    im_new = spim.binary_erosion(input=im, structure=cube(1))*im_new
    return im_new


def porosimetry(im, npts=25, sizes=None, inlets=None):
    r"""
    Simulates a porosimetry experiment on a binary image.  This function is
    equivalent to the morphological image opening and/or the full morphology
    approaches.

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
    Although this function is equivalent to morphological image opening, it is
    done using distance transforms instead of convolution filters.  This
    approach is much faster than using dilations and erosions when the
    structuring element is large.

    """
    dt = spim.distance_transform_edt(im)
    if inlets is None:
        inlets = sp.ones_like(im, dtype=bool)
        if im.ndim == 2:
            inlets[1:-2, 1:-2] = False
        elif im.ndim == 3:
            inlets[1:-2, 1:-2,1:-2] = False
    if sizes is None:
        sizes = sp.logspace(sp.log10(sp.amax(dt)), 0.1, npts)

    imresults = sp.zeros(sp.shape(im))
    print('Porosimetry Running')
    print('0%|'+'-'*len(sizes)+'|100%')
    print('  |', end='')
    for r in sizes:
        print('|', end='')
        sys.stdout.flush()
        imtemp = dt >= r
        labels = spim.label(imtemp + inlets)[0]
        inlet_labels = sp.unique(labels[inlets])
        imtemp = sp.in1d(labels.ravel(), inlet_labels)
        imtemp = sp.reshape(imtemp, im.shape)
        im = spim.distance_transform_edt(~(imtemp^inlets)) <= r
        imresults[(imresults == 0)*im] = r
    print('|')
    return imresults
