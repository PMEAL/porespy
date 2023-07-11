import numpy as np
from porespy.tools import make_contiguous
from scipy.stats import rankdata


__all__ = [
    'size_to_seq',
    'size_to_satn',
    'seq_to_satn',
    'pc_to_satn',
    'pc_to_seq',
    'satn_to_seq',
]


def size_to_seq(size, im=None, bins=None, mode='drainage'):
    r"""
    Converts an image of invasion size values into invasion sequence values

    Parameters
    ----------
    size : ndarray
        The image containing invasion size in each voxel. Values of 0 are
        assumed to be solid (if ``im`` is not given) and values of -1 are
        assumed to be uninvaded.
    im : ndarray, optional
        A binary image of the porous media, with ``True`` indicating the
        void space and ``False`` indicating the solid phase. If not given
        then it is assumed that the solid is identified as ``size == 0``.
    bins : array_like or int (optional)
        The bins to use when converting sizes to sequence.  The default is
        to create 1 bin for each unique value in ``size`` (except for -1 and 0).
        If an **int** is supplied, it is interpreted as the number of bins between 1
        and the maximum value in ``size``.  If an array is supplied it is used as
        the bins directly.
    mode : str
        Controls how the sizes are converted to a sequence. The options are:

        ============= ==============================================================
        `mode`        Description
        ============= ==============================================================
        'drainage'    The sizes are assumed to have been filled from largest to
                      smallest, ignoring 0's and -1's
        'imbibition'  The sizes are assumed to have been filled from smallest to
                      largest, ignoring 0's and -1's
        ============= ==============================================================

    Returns
    -------
    seq : ndarray
        An ndarray the same shape as ``size`` with invasion size values
        replaced by the invasion sequence, according to the specified `mode`.
        Any uninvaded voxels, indicated by -1 in ``size`` will be indicated by
        -1 in ``seq``.

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
    uninvaded = size == -1
    if bins is None:
        bins = np.unique(size)
    elif isinstance(bins, int):
        bins = np.linspace(0, size.max(), bins)
    vals = np.digitize(size, bins=bins, right=True)
    if mode.startswith('im'):
        vals[solid] = 0
        vals[uninvaded] = -1
        vals = make_contiguous(vals, mode='symmetric')
    if mode.startswith('dr'):
        vals = make_contiguous(vals, mode='symmetric')
        vals = vals.max() + 1 - vals
        vals[solid] = 0
        vals[uninvaded] = -1
    return vals


def size_to_satn(size, im=None, bins=None, mode='drainage'):
    r"""
    Converts an image of invasion size values into non-wetting phase saturations.

    Parameters
    ----------
    size : ndarray
        The image containing invasion size values in each voxel. Solid
        should be indicated as 0's and uninvaded voxels as -1.
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
    mode : str
        Controls how the sizes are converted to saturations. The options are:

        ============= ==============================================================
        `mode`        Description
        ============= ==============================================================
        'drainage'    The sizes are assumed to have been filled from largest to
                      smallest, ignoring 0's and -1's
        'imbibition'  The sizes are assumed to have been filled from smallest to
                      largest, ignoring 0's and -1's
        ============= ==============================================================

    Returns
    -------
    satn : ndarray
        An ndarray the same shape as ``size`` but with size values replaced
        by the fraction of void space invaded at each size, according to the
        specified `mode`. Solid voxels and uninvaded voxels are represented by 0
        and -1, respectively.

    Notes
    -----
    If any ``-1`` values are present in `size` the maximum saturation will be less
    than 1.0 since this means that not all wetting phase was displaced.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/size_to_satn.html>`_
    to view online example.
    """
    if bins is None:
        bins = np.unique(size[size > 0])
    elif isinstance(bins, int):
        bins = np.linspace(0, size.max(), bins)
    if im is None:
        im = ~(size == 0)
    void_vol = im.sum()
    satn = -np.ones_like(size, dtype=float)
    if mode.startswith('im'):
        for r in bins:
            hits = (size <= r) * (size > 0)
            temp = hits.sum()/void_vol
            satn[hits * (satn == -1)] = temp
    elif mode.startswith('dr'):
        for r in bins[-1::-1]:
            hits = (size >= r) * (size > 0)
            temp = hits.sum()/void_vol
            satn[hits * (satn == -1)] = temp
    satn *= (im > 0)
    return satn


def seq_to_satn(seq, im=None, mode='drainage'):
    r"""
    Converts an image of invasion sequence values to non-wetting phase saturation
    values.

    Parameters
    ----------
    seq : ndarray
        The image containing invasion sequence values in each voxel. Solid
        should be indicated as 0's and uninvaded voxels as -1.
    im : ndarray, optional
        A binary image of the porous media, with ``True`` indicating the
        void space and ``False`` indicating the solid phase. If not given
        then it is assumed that the solid is identified as ``seq == 0``.
    mode : str
        Controls how the sequences are converted to saturations. The options are:

        ============= ==============================================================
        `mode`        Description
        ============= ==============================================================
        'drainage'    The saturation is assumed to increase with increasing sequence
        'imbibition'  The saturation is assumed to decrease with increasing sequence
        ============= ==============================================================

    Returns
    -------
    satn : ndarray
        An ndarray the same shape as ``seq`` but with sequence values replaced
        by the fraction of void space invaded at the sequence number, accounting
        for the specified `mode`. Solid voxels and uninvaded voxels are represented
        by 0 and -1, respectively.

    Notes
    -----
    If any ``-1`` values are present in `seq` the maximum saturation will be less
    than 1.0 since this means that not all wetting phase was displaced.

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
    uninvaded_mask = seq == -1  # Store uninvaded locations
    seq[seq <= 0] = 0  # Set uninvaded to solid for next steps
    if mode.startswith('im'):
        seq = seq.max() - seq + 1
        seq[solid_mask] = 0
        seq[uninvaded_mask] = 0
    seq = rankdata(seq, method='dense') - 1
    b = np.bincount(seq)
    if (solid_mask.sum(dtype=np.int64) > 0) or \
            (uninvaded_mask.sum(dtype=np.int64) > 0):
        b[0] = 0
    c = np.cumsum(b)
    seq = np.reshape(seq, solid_mask.shape)
    satn = c[seq]/(seq.size - solid_mask.sum(dtype=np.int64))
    satn[solid_mask] = 0
    satn[uninvaded_mask] = -1
    return satn


def pc_to_seq(pc, im, mode='drainage'):
    r"""
    Converts an image of capillary entry pressures to invasion sequence values

    Parameters
    ----------
    pc : ndarray
        A Numpy array with the value in each voxel indicating the capillary
        pressure at which it was invaded. In order to accommodate the
        possibility of both positive and negative capillary pressure values,
        uninvaded voxels should be indicated by ``+inf`` and residual phase
        by ``-inf``. Solid vs void phase is defined by ``im`` which is
        mandatory.
    im : ndarray
        A Numpy array with ``True`` values indicating the void space
    mode : str
        Controls how the pressures are converted to sequence. The options are:

        ============= ==============================================================
        `mode`        Description
        ============= ==============================================================
        'drainage'    The pressures are assumed to have been filled from smallest to
                      largest. Voxels with -np.inf are treated as though they are
                      invaded by non-wetting fluid at the start of the process, and
                      voxels with +np.inf are treated as though they are never
                      invaded.
        'imbibition'  The pressures are assumed to have been filled from largest to
                      smallest. Voxels with -np.inf are treated as though they are
                      already occupied by non-wetting fluid at the start of the
                      process, and voxels with +np.inf are treated as though they
                      are filled with wetting phase.
        ============= ==============================================================

    Returns
    -------
    seq : ndarray
        A Numpy array the same shape as `pc`, with each voxel value indicating
        the sequence at which it was invaded, according to the specified `mode`.
        Uninvaded voxels are set to -1.

    Notes
    -----
    Voxels with `+inf` are treated as though they were never invaded so are given a
    sequence value of -1. Voxels with  `-inf` are treated as though they were
    invaded by non-wetting phase at the start of the simulation so are given a
    sequence number of 1 for both mode `drainage` and `imbibition`.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/pc_to_seq.html>`_
    to view online example.
    """
    inf = pc == np.inf  # save for later
    if mode == 'drainage':
        bins = np.unique(pc)
    elif mode == 'imbibition':
        pc[pc == -np.inf] = np.inf
        bins = np.unique(pc)[-1::-1]
    a = np.digitize(pc, bins=bins)
    a[~im] = 0
    a[np.where(inf)] = -1
    a = make_contiguous(a, mode='symmetric')
    return a


def pc_to_satn(pc, im, mode='drainage'):
    r"""
    Converts an image of capillary entry pressures to saturation values

    Parameters
    ----------
    pc : ndarray
        A Numpy array with the value in each voxel indicating the capillary
        pressure at which it was invaded. In order to accommodate the
        possibility of both positive and negative capillary pressure values,
        uninvaded voxels should be indicated by ``+inf`` and residual phase
        by ``-inf``. Solid vs void phase is defined by ``im`` which is
        mandatory.
    im : ndarray
        A Numpy array with ``True`` values indicating the void space
    mode : str
        Controls how the pressures are converted to sequence. The options are:

        ============= ==============================================================
        `mode`        Description
        ============= ==============================================================
        'drainage'    The pressures are assumed to have been filled from smallest to
                      largest.
        'imbibition'  The pressures are assumed to have been filled from largest to
                      smallest
        ============= ==============================================================

    Returns
    -------
    satn : ndarray
        A Numpy array the same shape as `pc`, with each voxel value indicating
        the global saturation at which it was invaded, according to the specified
        `mode`. Voxels with  `-inf` are treated as though they were invaded
        at the start of the simulation so are given a sequence number of 1 for both
        mode `drainage` and `imbibition`.

    Notes
    -----
    If any ``+inf`` values are present the maximum saturation will be less than
    1.0 since not all wetting phase was displaced.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/pc_to_satn.html>`_
    to view online example.

    """
    a = np.digitize(pc, bins=np.unique(pc))
    a[~im] = 0
    a[np.where(pc == np.inf)] = -1
    satn = seq_to_satn(seq=a, im=im, mode=mode)
    return satn


def satn_to_seq(satn, im=None, mode='drainage'):
    r"""
    Converts an image of nonwetting phase saturations to invasion sequence
    values

    Parameters
    ----------
    satn : ndarray
        A Numpy array with the value in each voxel indicating the global
        saturation at the point it was invaded. -1 indicates a voxel that
        not invaded, and 0 indicates solid phase.
    im : ndarray
        A Numpy array with ``True`` values indicating the void space.
    mode : str
        Controls how the saturations are converted to sequence. The options are:

        ============= ==============================================================
        `mode`        Description
        ============= ==============================================================
        'drainage'    The pressures are assumed to have been filled from smallest to
                      largest, ignoring 0's and -1's
        'imbibition'  The sizes are assumed to have been filled from largest to
                      smallest, ignoring 0's and -1's
        ============= ==============================================================

    Returns
    -------
    seq : ndarray
        A Numpy array the same shape as `satn` with each voxel value indicating
        the sequence in which it was invaded, according to the specified `mode`.
        Solid voxels are indicated by 0 and uninvaded by -1.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/satn_to_seq.html>`_
    to view online example.

    """
    if im is None:
        im = satn > 0
    uninvaded = satn == -1
    values = np.unique(satn)
    seq = np.digitize(satn, bins=values)
    # Set uninvaded by to -1
    seq[satn == -1] = -1
    # Set solids back to 0
    seq[~im] = 0
    # Ensure values are contiguous while keeping -1 and 0
    seq = make_contiguous(im=seq, mode='symmetric')
    if mode.startswith('im'):
        seq = (seq.max() + 1) - seq
        seq[~im] = 0
    seq[uninvaded] = -1
    return seq
