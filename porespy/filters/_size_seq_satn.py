import numpy as np
from porespy.tools import make_contiguous
from scipy.stats import rankdata


__all__ = [
    'size_to_seq',
    'size_to_satn',
    'seq_to_satn',
    'pc_to_satn',
    'satn_to_seq',
]


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
        process occurs via largest regions first, such as produced by
        the ``porosimetry`` function.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/size_to_seq.html>`_
    to view online example.

    """
    if im is None:
        solid = size == 0
    else:
        solid = im == 0
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

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/size_to_satn.html>`_
    to view online example.
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

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/seq_to_satn.html>`_
    to view online example.
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


def pc_to_satn(pc, im):
    r"""
    Converts an image of capillary entry pressures to saturation values

    Parameters
    ----------
    pc : ndarray
        A Numpy array with the value in each voxel indicating the capillary
        pressure at which it was invaded. In order to accomodateh the
        possibility of positive and negative capillary pressure values,
        uninvaded voxels should be indicated by ``+inf`` and residual phase
        by ``-inf``. Solid vs void phase is defined by ``im`` which is
        mandatory.
    im : ndarray
        A Numpy array with ``True`` values indicating the void space

    Returns
    -------
    satn : ndarray
        A Numpy array with each voxel value indicating the global saturation
        at which it was invaded.

    Notes
    -----
    If any ``-inf`` values are present the minimum saturation will start at
    a value greater than 0 since residual was present. If any ``+inf`` values
    are present the maximum saturation will be less than 1.0 since not all
    wetting phase was displaced.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/pc_to_satn.html>`_
    to view online example.

    """
    a = np.digitize(pc, bins=np.unique(pc))
    a[~im] = 0
    a[np.where(pc == np.inf)] = -1
    satn = seq_to_satn(seq=a, im=im)
    return satn


def satn_to_seq(satn, im=None):
    r"""
    Converts an image of nonwetting phase saturations to invasion sequence
    values

    Parameters
    ----------
    satn : ndarray
        A Numpy array with the value in each voxel indicating the global
        saturation at the point it was invaded. -1 indicates a voxel that
        not invaded.
    im : ndarray
        A Numpy array with ``True`` values indicating the void space.

    Returns
    -------
    satn : ndarray
        A Numpy array with each voxel value indicating the global saturation
        at which it was invaded. Solid voxels are indicated by 0 and
        uninvaded by -1.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/satn_to_seq.html>`_
    to view online example.

    """
    if im is None:
        im = satn > 0
    values = np.unique(satn)
    seq = np.digitize(satn, bins=values)
    # Set uninvaded by to -1
    seq[satn == -1] = -1
    # Set solids back to 0
    seq[~im] = 0
    # Ensure values are contiguous while keeping -1 and 0
    seq = make_contiguous(im=seq, mode='symmetric')
    return seq
