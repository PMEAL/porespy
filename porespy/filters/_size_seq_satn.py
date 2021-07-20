import numpy as np
from porespy.tools import make_contiguous
from scipy.stats import rankdata


def size_to_seq(size, im=None, bins=None):
    r"""
    Converts an image of invasion size values into sequence values

    This is meant to accept the output of the ``porosimetry`` function.

    Parameters
    ----------
    size : ndarray
        The image containing invasion size values in each voxel.
    im : ndarray, optional
        A binary image of the porous media, with ``True`` indicating the
        void space and ``False`` indicating the solid phase. If not given
        then it is assumed that the solid is identified as ``size == 0``.
    bins : array_like or int (optional)
        The bins to use when converting sizes to sequence.  The default is
        to create 1 bin for each unique value in ``size``.  If an **int**
        is supplied it is interpreted as the number of bins between 0 and the
        maximum value in ``size``.  If an array is supplied it is used as
        the bins directly.

    Returns
    -------
    seq : ndarray
        An ndarray the same shape as ``size`` with invasion size values
        replaced by the invasion sequence.  This assumes that the invasion
        process occurs via increasing pressure steps, such as produced by
        the ``porosimetry`` function.

    """
    solid = size == 0
    if bins is None:
        bins = np.unique(size)
    elif isinstance(bins, int):
        bins = np.linspace(0, size.max(), bins)
    vals = np.digitize(size, bins=bins, right=True)
    # Invert the vals so smallest size has largest sequence
    vals = -(vals - vals.max() - 1) * ~solid
    # In case too many bins are given, remove empty ones
    vals = make_contiguous(vals, mode='keep_zeros')
    return vals


def size_to_satn(size, im=None, bins=None):
    r"""
    Converts an image of invasion size values into saturations.

    This is meant to accept the output of the ``porosimetry`` function.

    Parameters
    ----------
    size : ndarray
        The image containing invasion size values in each voxel.
    im : ndarray, optional
        A binary image of the porous media, with ``True`` indicating the
        void space and ``False`` indicating the solid phase. If not given
        then it is assumed that the solid is identified as ``size == 0``.
    bins : array_like or int (optional)
        The bins to use when converting sizes to saturation.  The default is
        to create 1 bin for each unique value in ``size``.  If an **int**
        is supplied it is interpreted as the number of bins between 0 and the
        maximum value in ``size``.  If an array is supplied it is used as
        the bins directly.

    Returns
    -------
    satn : ndarray
        An ndarray the same size as ``seq`` but with sequence values replaced
        by the fraction of void space invaded at or below the sequence number.
        Solid voxels and uninvaded voxels are represented by 0 and -1,
        respectively.
    """
    if bins is None:
        bins = np.unique(size)
    elif isinstance(bins, int):
        bins = np.linspace(0, size.max(), bins)
    if im is None:
        im = (size != 0)
    void_vol = im.sum()
    satn = -np.ones_like(size, dtype=float)
    for r in bins[-1::-1]:
        hits = (size >= r) * (size > 0)
        temp = hits.sum()/void_vol
        satn[hits * (satn == -1)] = temp
    satn *= (im > 0)
    return satn


def seq_to_satn(seq, im=None):
    r"""
    Converts an image of invasion sequence values to saturation values.

    This is meant to accept the output of the ``ibip`` function.

    Parameters
    ----------
    seq : ndarray
        The image containing invasion sequence values in each voxel.  Solid
        should be indicated as 0's and uninvaded voxels as -1.
    im : ndarray, optional
        A binary image of the porous media, with ``True`` indicating the
        void space and ``False`` indicating the solid phase. If not given
        then it is assumed that the solid is identified as ``seq == 0``.

    Returns
    -------
    satn : ndarray
        An ndarray the same size as ``seq`` but with sequence values replaced
        by the fraction of void space invaded at or below the sequence number.
        Solid voxels and uninvaded voxels are represented by 0 and -1,
        respectively.

    """
    seq = np.copy(seq).astype(int)
    if im is None:
        solid_mask = seq == 0
    else:
        solid_mask = im == 0
    uninvaded_mask = seq == -1
    seq[seq <= 0] = 0
    seq = rankdata(seq, method='dense') - 1
    b = np.bincount(seq)
    if (solid_mask.sum() > 0) or (uninvaded_mask.sum() > 0):
        b[0] = 0
    c = np.cumsum(b)
    seq = np.reshape(seq, solid_mask.shape)
    satn = c[seq]/(seq.size - solid_mask.sum())
    satn[solid_mask] = 0
    satn[uninvaded_mask] = -1
    return satn
