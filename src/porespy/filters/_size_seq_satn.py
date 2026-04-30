import numpy as np

from porespy.tools import make_contiguous

__all__ = [
    'size_to_seq',
    'size_to_satn',
    'seq_to_satn',
    'pc_to_satn',
    'pc_to_seq',
    'satn_to_seq',
    'satn_to_time',
    'size_to_pc',
]


def size_to_seq(size, im=None, bins=None, mode='drainage'):
    r"""
    Converts a size map to a sequence map

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
    <https://porespy.org/examples/filters/reference/size_to_seq.html>`__
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
    Converts size map to a saturation map

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
    <https://porespy.org/examples/filters/reference/size_to_satn.html>`__
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


def seq_to_satn(seq, im, mode='drainage'):
    r"""
    Converts a sequence map to a saturation map

    Parameters
    ----------
    seq : ndarray
        The image containing invasion sequence values in each voxel. Residual phase
        should be indicated as 0's and uninvaded (i.e. trapped) voxels as -1.
        `im` is used to mask to remove solid voxels from computation
    im : ndarray
        A binary image of the porous media, with ``True`` indicating the
        void space and ``False`` indicating the solid phase.
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
        for the specified `mode`. Residual fluid and uninvaded voxels are represented
        by 0 and -1, respectively.  Solie phase is also represented as 0, so `im`
        must be used as a mask.

    Notes
    -----
    If any `-1` values are present in `seq` the maximum saturation will be less
    than 1.0 since this means that not all defending phase was displaced. If any
    `0` are present in `seq` then the minimum saturation will be greater than 0,
    since this means some invading phase was already present.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/seq_to_satn.html>`__
    to view online example.
    """
    seq = np.copy(seq).astype(int)
    trapped_mask = seq == -1  # Store uninvaded locations
    residual_mask = (seq == 0)*im
    seq[trapped_mask] = seq.max() + 1  # Assign trapped voxels with highest seq
    seq += 1  # Increment by 1 so residual voxels are 1 instead of 0
    seq[~im] = 0  # Set solid voxels to 0
    seq[residual_mask] = 1  # Set residual to 1
    b = np.bincount(seq.flatten())  # Count number of voxels with each sequence
    if mode.startswith('imb'):
        c = 1 - np.cumsum(b[1:])/im.sum(dtype=np.int64)  # Convert to saturation
    elif mode.startswith('drain'):
        c = np.cumsum(b[1:])/im.sum(dtype=np.int64)  # Convert to saturation
    else:
        raise Exception('Unrecognized mode')
    c = np.hstack(([0], c))  # Add 0 for solid
    satn = c[seq]
    satn[~im] = 0.0  # Ensure solids are 0
    satn[trapped_mask] = -1.0  # Update trapped voxels to have -1 still
    return satn


def pc_to_seq(pc, im, mode='drainage'):
    r"""
    Converts a capillary pressure map to a sequence map

    Parameters
    ----------
    pc : ndarray
        A Numpy array with the value in each voxel indicating the capillary
        pressure at which it was invaded. In order to accommodate the
        possibility of both positive and negative capillary pressure values,
        trapped or residual wetting phase should be indicated by ``+inf`` and
        residual or trapped non-wetting phase by ``-inf``.
    im : ndarray
        A Numpy array with ``True`` values indicating the void space. This is
        used to mask the ``pc`` image so it does not matter what values are placed
        in the solid voxels of ``pc``.
    mode : str
        Controls how the pressures are converted to sequence. The options are:

        ============= ==============================================================
        `mode`        Description
        ============= ==============================================================
        'drainage'    The pressures are assumed to have been filled from smallest to
                      largest. Voxels with $-inf$ are treated as though they are
                      invaded by non-wetting fluid at the start of the process,
                      and voxels with $+inf$ are treated as though they are never
                      invaded.
        'imbibition'  The pressures are assumed to have been filled from largest to
                      smallest. Voxels with $+inf$ are treated as though they are
                      already occupied by wetting fluid at the start of the
                      process, and voxels with $*inf$ are treated as though they
                      are never filled with wetting phase.
        ============= ==============================================================

    Returns
    -------
    seq : ndarray
        A Numpy array the same shape as `pc`, with each voxel value indicating
        the sequence at which it was invaded, according to the specified `mode`.
        Uninvaded voxels (either trapped wp during drainage or trapped nwp during
        imbibion) are set to -1. Preinvaded voxels (either residual nwp during
        drainage or residual wp during imbition) are set to 0.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/pc_to_seq.html>`__
    to view online example.
    """
    pc = np.copy(pc)
    if mode == 'drainage':
        bins = np.unique(pc[im])
        a = np.digitize(pc, bins=bins, right=False) + 1  # To ensure no voxels are 0
        a[pc == -np.inf] = 0  # Set residual nwp to lowest sequence (i.e. 0)
        a[pc == np.inf] = -1  # Set trapped wp to -1 (only option)
    elif mode == 'imbibition':
        bins = np.flip(np.unique(pc[im]))
        a = np.digitize(pc, bins=bins, right=True)
        a[pc == np.inf] = 0
        a[pc == -np.inf] = -1
    a[~im] = 0
    a = make_contiguous(a, mode='symmetric')
    return a


def pc_to_satn(pc, im, mode='drainage'):
    r"""
    Converts a capillary pressure map to a saturation map

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
    <https://porespy.org/examples/filters/reference/pc_to_satn.html>`__
    to view online example.

    """
    a = np.digitize(pc, bins=np.unique(pc))
    a[~im] = 0
    a[np.where(pc == np.inf)] = -1
    satn = seq_to_satn(seq=a, im=im, mode=mode)
    return satn


def satn_to_seq(satn, im, residual=None, mode='drainage'):
    r"""
    Converts a saturaiton map to a sequence map

    Parameters
    ----------
    satn : ndarray
        A Numpy array with the value in each voxel indicating the global
        saturation at the point it was invaded. -1 indicates a voxel that
        not invaded, and 0 indicates solid phase.
    im : ndarray
        A Numpy array with ``True`` values indicating the void space.
    residual : ndarray
        An `ndarray` with `True` values indicating the locations of any residual
        phase.  If this is not provided, it will be assumed that no residual was
        present, which will result in an incorrect conversion to sequence if this
        assumption is not correct.
    mode : str
        Controls how the saturations are converted to sequence. The options are:

        ============= ==============================================================
        `mode`        Description
        ============= ==============================================================
        'drainage'    Non-wetting fluid displaces wetting fluid
        'imbibition'  Wetting fluid dispalces the non-wetting fluid
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
    <https://porespy.org/examples/filters/reference/satn_to_seq.html>`__
    to view online example.

    """
    uninvaded = satn == -1
    values = np.unique(satn)
    seq = np.digitize(satn, bins=values)
    # Set uninvaded to 0
    seq[uninvaded] = -1  # Not sure this is needed?
    # Set solids back to 0
    seq[~im] = 0
    # Ensure values are contiguous while keeping -1 and 0
    seq = make_contiguous(im=seq, mode='symmetric')
    if mode.startswith('im'):
        seq = (seq.max() + 1) - seq
        seq[~im] = 0
    seq[uninvaded] = -1
    if residual is not None:
        seq[residual] = 0
        seq = make_contiguous(im=seq, mode='symmetric')
    return seq


def size_to_pc(im, size, f=None, **kwargs):
    r"""
    Converts size map into capillary pressure map

    Parameters
    ----------
    im : ndarray
        A Numpy array with ``True`` values indicating the void space.
    size : ndarray
        The image containing invasion size values in each voxel. Solid
        should be indicated as 0's and uninvaded voxels as -1.
    f : function handle, optional
        A function to compute the capillary pressure which receives `size` as
        the first argument, followed by any additional `**kwargs`. If not
        provided then the Washburn equation is used, which requires `theta` and
        `sigma` to be specified as additional keyword arguments.
    **kwargs : Key word arguments
        All additional keyword arguments are passed on to `f`.

    Returns
    -------
    pc : ndarray
        An image with each voxel containing the capillary pressure at which it was
        invaded. Any uninvaded voxels in `size` are set to `np.inf` which is meant
        to indicate that these voxels are never invaded.

    Notes
    -----
    The function `f` should be of the form:

    .. code-block::

        def func(r, a, b):
            pc = ...  # Some equation for capillary pressure using r, a and b
            return pc

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/size_to_pc.html>`__
    to view online example.

    """
    if f is None:
        def f(r, sigma, theta, voxel_size):
            pc = -2*sigma*np.cos(np.deg2rad(theta))/(r*voxel_size)
            return pc
    pc = f(size, **kwargs)
    pc[~im] = 0.0
    pc[im == -1] = np.inf
    return pc


def satn_to_time(satn, im, flow_rate, voxel_size=1.0, mode='drainage'):
    r"""
    Converts a saturation map to a time-of-invasion map for a constant flow-rate
    injection.

    Parameters
    ----------
    satn : ndarray
        The saturation map produced by an invasion simulation. Each void voxel
        holds the global saturation at the moment it was invaded (in the
        convention used by `seq_to_satn` and `pc_to_satn`: invading-phase
        saturation for `drainage`, defending-phase saturation for `imbibition`).
        Solid voxels should be `0` and trapped (never-invaded) voxels `-1`.
    im : ndarray
        A boolean image of the porous medium, with `True` indicating the
        void space and `False` indicating the solid phase.
    flow_rate : scalar
        The volumetric injection rate of the displacing fluid, in units of
        `[length]^ndim / [time]` consistent with `voxel_size`.
    voxel_size : scalar
        The physical size of a voxel side, in `[length]` units consistent with
        `flow_rate`. Default is `1.0`, in which case the returned times are in
        units of voxels^ndim / `flow_rate`.
    mode : str
        Controls how the saturation is interpreted. Options are:

        ============= ==============================================================
        `mode`        Description
        ============= ==============================================================
        'drainage'    `satn` is the invading-phase saturation; it increases over
                      time. `t = satn * V_void / flow_rate`.
        'imbibition'  `satn` is the defending-phase saturation; it decreases over
                      time. `t = (1 - satn) * V_void / flow_rate`.
        ============= ==============================================================

    Returns
    -------
    time : ndarray
        An ndarray the same shape as `satn`. Each void voxel holds the time at
        which it was invaded. Solid voxels are `0` and trapped voxels are `-1`.

    Notes
    -----
    `V_void = im.sum() * voxel_size**im.ndim` is the total void volume. The
    underlying assumption is constant volumetric flow rate, so injected volume
    grows linearly with time and saturation maps directly to time. Residual
    voxels (already-invaded at $t=0$) end up at a small positive time equal to
    the time it would take to inject the residual volume.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/satn_to_time.html>`__
    to view online example.

    """
    if not (mode.startswith('dr') or mode.startswith('im')):
        raise Exception(f"Unrecognized mode {mode!r}")
    V_void = im.sum() * voxel_size ** im.ndim
    if mode.startswith('dr'):
        time = satn * V_void / flow_rate
    else:
        time = (1.0 - satn) * V_void / flow_rate
    time[~im] = 0.0
    time[satn == -1] = -1.0
    return time


if __name__ == "__main__":
    import numpy as np

    import porespy as ps

    im = np.ones([20, 20], dtype=bool)
    im[-1, :] = False
    im[0, :] = False
    pc = ps.filters.capillary_transform(im)
    inlets = ps.generators.faces(im.shape, inlet=1)
    inv = ps.simulations.injection(im, pc, inlets=inlets)

    # Check drainage with trapping
    tmp = np.copy(inv.im_seq)
    tmp[tmp >= 12] = -1  # Set some to uninvaded/trapped
    satn = ps.filters.seq_to_satn(seq=tmp, im=im, mode='drainage')
    assert ((satn >= 0)*im).sum()/im.sum() == satn.max()
    # Now convert it back!
    seq = satn_to_seq(satn, im=im, mode='drainage')
    assert np.all(seq == tmp)

    # Check drainage wtih trapping and residual
    tmp = np.copy(inv.im_seq)
    tmp[tmp >= 12] = -1  # Set some to uninvaded/trapped
    tmp[tmp > 0] -= 1
    residual = (tmp == 0)*im
    satn = ps.filters.seq_to_satn(seq=tmp, im=im, mode='drainage')
    assert ((satn >= 0)*im).sum()/im.sum() == satn.max()
    # Now convert it back!
    seq = satn_to_seq(satn, im=im, residual=residual, mode='drainage')
    assert np.all(seq == tmp)

    # Now check imbibition with trapping
    tmp = np.copy(inv.im_seq)
    tmp[tmp == 1] = -1  # Set some to uninvaded/trapped
    tmp[tmp > 0] = tmp.max() - tmp[tmp > 0] + 1  # Invert sequence numbers
    satn = ps.filters.seq_to_satn(seq=tmp, im=im, mode='imbibition')
    assert (1 - (satn > 0).sum()/im.sum()) == satn[satn > 0].min()
    # Now convert it back!
    seq = satn_to_seq(satn, im=im, mode='imbibition')
    assert np.all(seq == tmp)

    # Check drainage wtih trapping and residual
    tmp = np.copy(inv.im_seq)
    tmp[tmp == 1] = -1  # Set some to uninvaded/trapped
    tmp[tmp > 0] = tmp.max() - tmp[tmp > 0] + 1  # Invert sequence numbers
    tmp[(tmp > 0)*(tmp < 6)] = 0
    tmp = ps.tools.make_contiguous(seq, mode='symmetric')
    residual = (tmp == 0)*im
    satn = ps.filters.seq_to_satn(seq=tmp, im=im, mode='drainage')
    assert ((satn >= 0)*im).sum()/im.sum() == satn.max()
    # Now convert it back!
    seq = satn_to_seq(satn, im=im, residual=residual, mode='drainage')
    assert np.all(seq == tmp)
