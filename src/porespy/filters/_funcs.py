import logging
import operator
import warnings
from typing import Literal

import dask
import numpy as np
import numpy.typing as npt
import scipy.ndimage as spim
from skimage.morphology import reconstruction
from skimage.segmentation import clear_border

from porespy.tools import (
    _check_for_singleton_axes,
    get_edt,
    get_slices_grid,
    get_strel,
    get_tqdm,
    recombine,
    settings,
    unpad,
)
from porespy.tools._label import _label_components

__all__ = [
    "apply_chords",
    "apply_chords_3D",
    "apply_padded",
    "chunked_func",
    "distance_transform_lin",
    "find_dt_artifacts",
    "flood",
    "flood_func",
    "hold_peaks",
    "nphase_border",
    "prune_branches",
    "region_size",
    "trim_extrema",
]


edt = get_edt()
tqdm = get_tqdm()
strel = get_strel()
logger = logging.getLogger(__name__)


def apply_padded(
    im: npt.NDArray,
    pad_width,
    func,
    pad_val: int = 1,
    **kwargs,
):
    r"""
    Applies padding to an image before sending to `func`, then extracts
    the result corresponding to the original image shape.

    Parameters
    ----------
    im : ndarray
        The image to which `func` should be applied
    pad_width : int or list of ints
        The amount of padding to apply to each axis. Refer to `numpy.pad`
        documentation for more details.
    pad_val : scalar
        The value to place into the padded voxels.  The default is 1 (or
        `True`) which extends the pore space.
    func : function handle
        The function to apply to the padded image.
    kwargs
        Additional keyword arguments are collected and passed to `func`.

    Notes
    -----
    A use case for this is when using `skimage.morphology.skeletonize`
    to ensure that the skeleton extends beyond the edges of the image.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/apply_padded.html>`__
    to view online example.

    """
    padded = np.pad(im, pad_width=pad_width, mode="constant", constant_values=pad_val)
    temp = func(padded, **kwargs)
    result = unpad(im=temp, pad_width=pad_width)
    return result


def hold_peaks(
    im: npt.NDArray,
    axis: int = -1,
    ascending: bool = True,
):
    r"""
    Replaces each voxel with the highest value along the given axis.

    Parameters
    ----------
    im : ndarray
        A greyscale image whose peaks are to be found.
    axis : int
        The axis along which the operation is to be applied.
    ascending : bool
        If `True` (default) the given `axis` is scanned from 0 to end.
        If `False`, it is scanned in reverse order from end to 0.

    Returns
    -------
    result : ndarray
        A copy of `im` with each voxel is replaced with the highest value along
        the given axis.

    Notes
    -----
    "im" must be a greyscale image. In case a Boolean image is fed into this
    method, it will be converted to float values [0.0,1.0] before proceeding.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/hold_peaks.html>`__
    to view online example.

    """
    A = im.astype(float)
    B = np.swapaxes(A, axis, -1)
    if ascending is False:  # Flip the axis of interest (-1)
        B = np.flip(B, axis=-1)
    updown = np.empty((*B.shape[:-1], B.shape[-1] + 1), B.dtype)
    updown[..., 0], updown[..., -1] = -1, -1
    np.subtract(B[..., 1:], B[..., :-1], out=updown[..., 1:-1])
    chnidx = np.where(updown)
    chng = updown[chnidx]
    (pkidx,) = np.where((chng[:-1] > 0) & (chng[1:] < 0) | (chnidx[-1][:-1] == 0))
    pkidx = (*map(operator.itemgetter(pkidx), chnidx),)
    out = np.zeros_like(A)
    aux = out.swapaxes(axis, -1)
    aux[(*map(operator.itemgetter(slice(1, None)), pkidx),)] = np.diff(B[pkidx])
    aux[..., 0] = B[..., 0]
    result = out.cumsum(axis=axis)
    if ascending is False:  # Flip it back
        result = np.flip(result, axis=-1)
    return result


def distance_transform_lin(
    im: npt.NDArray,
    axis: int = 0,
    mode: Literal["forward", "backward", "both"] = "both",
):
    r"""
    Replaces each void voxel with the linear distance to the nearest solid
    voxel along the specified axis.

    Parameters
    ----------
    im : ndarray
        The image of the porous material with `True` values indicating
        the void phase (or phase of interest).
    axis : int
        The direction along which the distance should be measured, the
        default is 0 (i.e. along the x-direction).
    mode : str
        Controls how the distance is measured. Options are:

        ========== =================================================================
        Mode       Description
        ========== =================================================================
        'forward'  Distances are measured in the increasing direction along the
                   specified axis
        'reverse'  Distances are measured in the reverse direction. 'backward' is
                   also accepted.
        'both'     Distances are calculated in both directions (by recursively
                   calling itself), then reporting the minimum value of the two
                   results.
        ========== =================================================================

    Returns
    -------
    image : ndarray
        A copy of `im` with each foreground voxel containing the
        distance to the nearest background along the specified axis.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/distance_transform_lin.html>`__
    to view online example.

    """
    _check_for_singleton_axes(im)

    if mode in ["backward", "reverse"]:
        im = np.flip(im, axis)
        im = distance_transform_lin(im=im, axis=axis, mode="forward")
        im = np.flip(im, axis)
        return im
    elif mode in ["both"]:
        im_f = distance_transform_lin(im=im, axis=axis, mode="forward")
        im_b = distance_transform_lin(im=im, axis=axis, mode="backward")
        return np.minimum(im_f, im_b)
    b = np.cumsum(im > 0, axis=axis)
    c = np.diff(b * (im == 0), axis=axis)
    d = np.minimum.accumulate(c, axis=axis)
    if im.ndim == 1:
        e = np.pad(d, pad_width=[1, 0], mode="constant", constant_values=0)
    elif im.ndim == 2:
        ax = [[[1, 0], [0, 0]], [[0, 0], [1, 0]]]
        e = np.pad(d, pad_width=ax[axis], mode="constant", constant_values=0)
    elif im.ndim == 3:
        ax = [
            [[1, 0], [0, 0], [0, 0]],
            [[0, 0], [1, 0], [0, 0]],
            [[0, 0], [0, 0], [1, 0]],
        ]
        e = np.pad(d, pad_width=ax[axis], mode="constant", constant_values=0)
    f = im * (b + e)
    return f


def trim_extrema(
    im: npt.NDArray,
    h: float,
    mode="maxima",
):
    r"""
    Trims local extrema in greyscale values by a specified amount.

    This essentially decapitates peaks and/or floods valleys.

    Parameters
    ----------
    im : ndarray
        The image whose extrema are to be removed
    h : float
        The height to remove from each peak or fill in each valley
    mode : string {'maxima' | 'minima' | 'extrema'}
        Specifies whether to remove maxima or minima or both

    Returns
    -------
    image : ndarray
        A copy of the input image with all the peaks and/or valleys
        removed.

    Notes
    -----
    (1) This function is referred to as **imhmax** or **imhmin** in Matlab.

    (2) If the provided `h` is larger than ALL peaks in the array, then the
    baseline values of the array are changed as well.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/trim_extrema.html>`__
    to view online example.

    """
    mask = np.copy(im)
    im = np.copy(im)
    if mode == "maxima":
        result = reconstruction(seed=im - h, mask=mask, method="dilation")
    elif mode == "minima":
        result = reconstruction(seed=im + h, mask=mask, method="erosion")
    elif mode == "extrema":
        result = reconstruction(seed=im - h, mask=mask, method="dilation")
        result = reconstruction(seed=result + h, mask=result, method="erosion")
    return result


def flood(
    im: npt.NDArray,
    labels: npt.NDArray,
    mode: Literal[
        "maximum", "minimum", "median", "mean", "size", "standard_deviations", "variance"
    ] = "max",
):
    r"""
    Floods/fills each region in an image with a single value based on the
    specific values in that region.

    This function calls the various functions in `scipy.ndimage.measurements`
    but instead of returning a list of values, it fills each region with its
    value.  This is useful for visualization and statistics.

    Parameters
    ----------
    im : array_like
        An image with the numerical values of interest in each voxel,
        and 0's elsewhere.
    labels : array_like
        An array containing labels identifying each individual region to be
        flooded. If not provided then connected-component labeling is applied
        to `im > 0`.
    mode : string
        Specifies how to determine the value to flood each region. Options
        taken from the `scipy.ndimage.measurements` function include:

        ===================== ======================================================
        Option                Description
        ===================== ======================================================
        maximum               Floods each region with the local max in that region.
                              The keyword `max` is also accepted.
        minimum               Floods each region the local minimum in that region.
                              The keyword `min` is also accepted.
        median                Floods each region the local median in that region
        mean                  Floods each region the local mean in that region
        size                  Floods each region with the size of that region.  This
                              is actually accomplished with `scipy.ndimage.sum` by
                              converting `im` to a boolean image (`im = im > 0`).
        standard_deviation    Floods each region with the value of the standard
                              deviation of the voxels in `im`.
        variance              Floods each region with the value of the variance of
                              the voxels in `im`.
        ===================== ======================================================

    Returns
    -------
    flooded : ndarray
        A copy of `im` with new values placed in each forground voxel
        based on the `mode`.

    See Also
    --------
    prop_to_image
    flood_func
    region_size

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/flood.html>`__
    to view online example.

    """
    mask = labels > 0
    N = labels.max()
    mode = "sum" if mode == "size" else mode
    mode = "maximum" if mode == "max" else mode
    mode = "minimum" if mode == "min" else mode
    f = getattr(spim, mode)
    vals = f(input=im * mask, labels=labels, index=range(0, N + 1))
    flooded = vals[labels]
    flooded = flooded * mask
    return flooded


def flood_func(
    im: npt.NDArray,
    func,
    labels: npt.NDArray = None,
):
    r"""
    Flood each isolated region in an image with a constant value calculated by
    the given function.

    Parameters
    ----------
    im : ndarray
        An image with the numerical values of interest in each voxel,
        and 0's elsewhere.
    func : Numpy function handle
        The function to be applied to each region in the image.  Any Numpy
        function that returns a scalar value can be passed, such as `amin`,
        `amax`, `sum`, `mean`, `median`, etc.
    labels : ndarray
        An array containing labels identifying each individual region to be
        flooded. If not provided then connected-component labeling is applied to
        `im > 0`.

    Returns
    -------
    flooded : ndarray
        An image the same size as `im` with each isolated region flooded
        with a constant value based on the given `func` and the values
        in `im`.

    See Also
    --------
    flood, region_size

    Notes
    -----
    Many of the functions in `scipy.ndimage` can be applied to
    individual regions using the `index` argument.  This function extends
    that behavior to all numpy function, in the event you wanted to compute
    the cosine of the values in each region for some reason. This function
    also floods the original image instead of returning a list of values for
    each region.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/flood_func.html>`__
    to view online example.

    """
    if labels is None:
        labels = _label_components(im=im > 0, conn="min")[0]
    slices = spim.find_objects(labels)
    flooded = np.zeros_like(im, dtype=float)
    for i, s in enumerate(slices):
        sub_im = labels[s] == (i + 1)
        val = func(im[s][sub_im])
        flooded[s] += sub_im * val
    return flooded


def find_dt_artifacts(dt: npt.NDArray):
    r"""
    Label points in a distance transform that are closer to image boundary
    than solid

    Parameters
    ----------
    dt : ndarray
        The distance transform of the phase of interest.

    Returns
    -------
    image : ndarray
        An ndarray the same shape as `dt` with numerical values
        indicating the maximum amount of error in each volxel, which is
        found by subtracting the distance to nearest edge of image from
        the distance transform value. In other words, this is the error
        that would be found if there were a solid voxel lurking just
        beyond the nearest edge of the image.  Obviously, voxels with a
        value of zero have no error.

    Notes
    -----
    These points could *potentially* be erroneously high since their
    distance values do not reflect the possibility that solid may have
    been present beyond the border of the image but was lost by trimming.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/find_dt_artifacts.html>`__
    to view online example.

    """
    temp = np.ones(shape=dt.shape) * np.inf
    for ax in range(dt.ndim):
        dt_lin = distance_transform_lin(
            np.ones_like(temp, dtype=bool), axis=ax, mode="both"
        )
        temp = np.minimum(temp, dt_lin)
    result = np.clip(dt - temp, a_min=0, a_max=np.inf)
    return result


def region_size(
    im: npt.NDArray,
    conn: Literal["max", "min"] = "min",
):
    r"""
    Replace each voxel with the size of the region to which it belongs

    Parameters
    ----------
    im : ndarray
        Either a boolean image wtih `True` indicating the features of
        interest, in which case connected-component labeling will be applied to
        find regions, or a greyscale image with integer values indicating
        regions.
    conn : str
        Can be either `'min'` or `'max'` and controls the shape of the structuring
        element used to determine voxel connectivity.  The default if `'min'` which
        imposes the strictest criteria, so that voxels must share a face to be
        considered connected.

    Returns
    -------
    image : ndarray
        A copy of `im` with each voxel value indicating the size of the
        region to which it belongs.  This is particularly useful for
        finding chord sizes on the image produced by `apply_chords`.

    See Also
    --------
    flood

    Notes
    -----
    This function provides the same result as `flood` with `mode='size'`,
    although does the computation in a different way.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/region_size.html>`__
    to view online example.

    """
    if im.dtype == bool:
        im = _label_components(im=im, conn=conn)[0]
    counts = np.bincount(im.flatten())
    counts[0] = 0
    return counts[im]


def apply_chords(
    im: npt.NDArray,
    spacing: int = 1,
    axis: int = 0,
    trim_edges: bool = True,
    label: bool = False,
):
    r"""
    Adds chords to the void space in the specified direction.

    Parameters
    ----------
    im : ndarray
        An image of the porous material with void marked as `True`.
    spacing : int
        Separation between chords.  The default is 1 voxel.  This can be
        decreased to 0, meaning that the chords all touch each other,
        which automatically sets the `label` argument to `True`. The option
        ``spacing=0`` is deprecated and will be removed in a future release.
    axis : int (default = 0)
        The axis along which the chords are drawn.
    trim_edges : bool (default = `True`)
        Whether or not to remove chords that touch the edges of the image.
        These chords are artifically shortened, so skew the chord length
        distribution.
    label : bool (default is `False`)
        If `True` the chords in the returned image are each given a
        unique label, such that all voxels lying on the same chord have
        the same value.  This is automatically set to `True` if spacing
        is 0, but is `False` otherwise.

    Returns
    -------
    image : ndarray
        A copy of `im` with non-zero values indicating the chords.

    See Also
    --------
    apply_chords_3D

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/apply_chords.html>`__
    to view online example.

    """
    _check_for_singleton_axes(im)
    if spacing < 0:
        raise Exception("Spacing cannot be less than 0")
    if spacing == 0:
        warnings.warn(
            "spacing=0 is deprecated and will be removed in a future release; "
            "use spacing>=1 instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        label = True
    result = np.zeros(im.shape, dtype=int)  # Will receive chords at end
    slxyz = [slice(None, None, spacing * (axis != i) + 1) for i in [0, 1, 2]]
    slices = tuple(slxyz[: im.ndim])
    if spacing == 0:
        s = [[0, 1, 0], [0, 1, 0], [0, 1, 0]]
        if im.ndim == 3:
            s = np.pad(
                np.atleast_3d(s),
                pad_width=((0, 0), (0, 0), (1, 1)),
                mode="constant",
                constant_values=0,
            )
        s = np.swapaxes(s, 0, axis)
        chords = spim.label(im, structure=s)[0]
        if trim_edges:
            chords = clear_border(chords)
        result[slices] = chords
    else:
        # A one-voxel gap lets full connectivity distinguish neighboring chords.
        result[slices] = im[slices]
        chords = _label_components(result, conn="max")[0]
        if trim_edges:
            # Preserve the historical treatment of the sampled image boundaries.
            chords = clear_border(chords[slices])
            result.fill(0)
            result[slices] = chords
        else:
            result = chords
    if label is False:  # Remove label if not requested
        result = result > 0
    return result


def apply_chords_3D(
    im: npt.NDArray,
    spacing: int = 0,
    trim_edges: bool = True,
):
    r"""
    Adds chords to the void space in all three principle directions.

    Chords in the X, Y and Z directions are labelled 1, 2 and 3 respectively.

    Parameters
    ----------
    im : ndarray
        A 3D image of the porous material with void space marked as `True`.
    spacing : int (default = 0)
        Chords are automatically separated by 1 voxel on all sides, and this
        argument increases the separation.
    trim_edges : bool (default is `True`)
        Whether or not to remove chords that touch the edges of the image.
        These chords are artificially shortened, so skew the chord length
        distribution

    Returns
    -------
    image : ndarray
        A copy of `im` with values of 1 indicating x-direction chords,
        2 indicating y-direction chords, and 3 indicating z-direction
        chords.

    Notes
    -----
    The chords are separated by a spacing of at least 1 voxel so that
    connected-component labeling can detect individual chords.

    See Also
    --------
    apply_chords

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/apply_chords_3D.html>`__
    to view online example.

    """
    _check_for_singleton_axes(im)
    if im.ndim < 3:
        raise Exception("Must be a 3D image to use this function")
    if spacing < 0:
        raise Exception("Spacing cannot be less than 0")
    ch = np.zeros_like(im, dtype=int)
    ch[:, :: 4 + 2 * spacing, :: 4 + 2 * spacing] = 1  # X-direction
    ch[:: 4 + 2 * spacing, :, 2 :: 4 + 2 * spacing] = 2  # Y-direction
    ch[2 :: 4 + 2 * spacing, 2 :: 4 + 2 * spacing, :] = 3  # Z-direction
    chords = ch * im
    if trim_edges:
        labels = _label_components(im=chords > 0, conn="max")[0]
        temp = clear_border(labels) > 0
        chords = temp * chords
    return chords


def _get_axial_shifts(ndim=2, conn="min"):
    r"""
    Helper function to generate the axial shifts that will be performed on
    the image to identify bordering pixels/voxels
    """
    neighbors = strel[ndim][conn]
    if ndim == 2:
        neighbors[1, 1] = 0
        x, y = np.where(neighbors)
        x -= 1
        y -= 1
        return np.vstack((x, y)).T
    else:
        neighbors[1, 1, 1] = 0
        x, y, z = np.where(neighbors)
        x -= 1
        y -= 1
        z -= 1
        return np.vstack((x, y, z)).T


def _make_stack(im, conn="min"):
    r"""
    Creates a stack of images with one extra dimension to the input image
    with length equal to the number of borders to search + 1.

    Image is rolled along the axial shifts so that the border pixel is
    overlapping the original pixel. First image in stack is the original.
    Stacking makes direct vectorized array comparisons possible.

    """
    ndim = len(np.shape(im))
    axial_shift = _get_axial_shifts(ndim, conn)
    if ndim == 2:
        stack = np.zeros([np.shape(im)[0], np.shape(im)[1], len(axial_shift) + 1])
        stack[:, :, 0] = im
        for i in range(len(axial_shift)):
            ax0, ax1 = axial_shift[i]
            temp = np.roll(np.roll(im, ax0, 0), ax1, 1)
            stack[:, :, i + 1] = temp
        return stack
    elif ndim == 3:
        stack = np.zeros(
            [np.shape(im)[0], np.shape(im)[1], np.shape(im)[2], len(axial_shift) + 1]
        )
        stack[:, :, :, 0] = im
        for i in range(len(axial_shift)):
            ax0, ax1, ax2 = axial_shift[i]
            temp = np.roll(np.roll(np.roll(im, ax0, 0), ax1, 1), ax2, 2)
            stack[:, :, :, i + 1] = temp
        return stack


def nphase_border(
    im: npt.NDArray,
    conn: Literal["min", "max"] = "min",
):
    r"""
    Identifies the voxels in regions that border *N* other regions.

    Useful for finding triple-phase boundaries.

    Parameters
    ----------
    im : ndarray
        An ND image of the porous material containing discrete values in
        the pore space identifying different regions. e.g. the result of a
        snow-partition
    conn : str
        Can be either `'min'` or `'max'` and controls the shape of the structuring
        element used to determine voxel connectivity.  The default if `'min'` which
        imposes the strictest criteria, so that voxels must share a face to be
        considered connected.

    Returns
    -------
    image : ndarray
        A copy of `im` with voxel values equal to the number of uniquely
        different bordering values

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/nphase_border.html>`__
    to view online example.

    """
    _check_for_singleton_axes(im)
    # Get dimension of image
    ndim = len(np.shape(im))
    if ndim not in [2, 3]:
        raise NotImplementedError("Function only works for 2d and 3d images")
    # Pad image to handle edges
    im = np.pad(im, pad_width=1, mode="edge")
    # Stack rolled images for each neighbor to be inspected
    stack = _make_stack(im, conn)
    # Sort the stack along the last axis
    stack.sort()
    out = np.ones_like(im)
    # Run through stack recording when neighbor id changes
    # Number of changes is number of unique bordering regions
    for k in range(np.shape(stack)[ndim])[1:]:
        if ndim == 2:
            mask = stack[:, :, k] != stack[:, :, k - 1]
        elif ndim == 3:
            mask = stack[:, :, :, k] != stack[:, :, :, k - 1]
        out += mask
    # Un-pad
    if ndim == 2:
        return out[1:-1, 1:-1].copy()
    else:
        return out[1:-1, 1:-1, 1:-1].copy()


def prune_branches(
    skel: npt.NDArray,
    branch_points=None,
    iterations: int = 1,
):
    r"""
    Remove all dangling ends or tails of a skeleton

    Parameters
    ----------
    skel : ndarray
        A image of a full or partial skeleton from which the tails should
        be trimmed.
    branch_points : ndarray, optional
        An image the same size `skel` with `True` values indicating the
        branch points of the skeleton.  If this is not provided it is
        calculated automatically.
    iterations : int
        The number of times to recursively repeat the process.  The default is
        1.

    Returns
    -------
    array
        An ndarray containing the skeleton with tails removed.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/prune_branches.html>`__
    to view online example.

    """
    skel = skel > 0
    cube = strel[skel.ndim]["max"]
    # Create empty image to house results
    im_result = np.zeros_like(skel)
    # If branch points are not supplied, attempt to find them
    if branch_points is None:
        branch_points = spim.convolve(skel * 1.0, weights=cube) > 3
        branch_points = branch_points * skel
    # Store original branch points before dilating
    pts_orig = branch_points
    # Find arcs of skeleton by deleting branch points
    arcs = skel * (~branch_points)
    # Label arcs
    arc_labels = _label_components(im=arcs, conn="max")[0]
    # Dilate branch points so they overlap with the arcs
    branch_points = spim.binary_dilation(branch_points, structure=cube)
    pts_labels = _label_components(im=branch_points, conn="max")[0]
    # Now scan through each arc to see if it's connected to two branch points
    slices = spim.find_objects(arc_labels)
    label_num = 0
    for s in slices:
        label_num += 1
        # Find branch point labels the overlap current arc
        hits = pts_labels[s] * (arc_labels[s] == label_num)
        # If image contains 2 branch points, then it's not a tail.
        if len(np.unique(hits)) == 3:
            im_result[s] += arc_labels[s] == label_num
    # Add missing branch points back to arc image to make complete skeleton
    im_result += skel * pts_orig
    if iterations > 1:
        iterations -= 1
        im_temp = np.copy(im_result)
        im_result = prune_branches(
            skel=im_result, branch_points=None, iterations=iterations
        )
        if np.all(im_temp == im_result):
            iterations = 0
    return im_result


def chunked_func(
    func,
    parallel_kw={"divs": 2, "overlap": None, "cores": None},
    im_arg=["input", "image", "im"],
    strel_arg=["strel", "structure", "footprint"],
    **kwargs,
):
    r"""
    Performs the specified operation "chunk-wise" in parallel using `dask`.

    This can be used to save memory by doing one chunk at a time
    (`cores=1`) or to increase computation speed by spreading the work
    across multiple cores (e.g. `cores=8`)

    This function can be used with any operation that applies a
    structuring element of some sort, since this implies that the
    operation is local and can be chunked.

    Parameters
    ----------
    func : function handle
        The function which should be applied to each chunk, such as
        `spipy.ndimage.binary_dilation`.

    parallel_kw : dict
        Dictionary containing the settings for parallelization by chunking. The
        optional settings include divs (scalar or list of scalars,
        default = [2, 2, 2]), overlap (scalar or list of scalars, optional),
        and cores (scalar, default is all available cores).

        Divs is the number of times to divide the image for parallel
        processing. If `1` then parallel processing does not occur. `2` is
        equivalent to `[2, 2, 2]` for a 3D image.

        Overlap is the amount of overlap to include when dividing up the image.
        This value will almost always be the size (i.e. radius) of the
        structuring element. If not specified then the amount of overlap
        is inferred from the size of the structuring element, in which
        case the `strel_arg` must be specified.

        Cores is the number of cores that will be used to parallel process all
        domains. If ``None`` then all cores will be used but user can specify
        any integer values to control the memory usage. Setting value to 1 will
        effectively process the chunks in serial to minimize memory usage.

    im_arg : str
        The keyword used by `func` for the image to be operated on. By
        default this function will look for `image`, `input`, and
        `im` which are commonly used by *scipy.ndimage* and *skimage*.
    strel_arg : str
        The keyword used by `func` for the structuring element to apply.
        This is only needed if `overlap` is not specified. By default
        this function will look for `strel`, `structure`, and
        `footprint` which are commonly used by *scipy.ndimage* and
        *skimage*.
    kwargs
        All other arguments are passed to `func` as keyword arguments.
        Note that PoreSpy will fetch the image from this list of keywords
        using the value provided to `im_arg`.

    Returns
    -------
    result : ndarray
        An image the same size as the input image, with the specified
        filter applied as though done on a single large image. There
        should be *no* difference.

    Notes
    -----
    This function divides the image into the specified number of chunks,
    but also applies a padding to each chunk to create an overlap with
    neighboring chunks. This way the operation does not have any edge
    artifacts. The amount of padding is usually equal to the radius of the
    structuring element but some functions do not use one, such as the
    distance transform and Gaussian blur.  In these cases the user can
    specify `overlap`.

    See Also
    --------
    scikit-image.util.apply_parallel

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/chunked_func.html>`__
    to view online example.

    """
    # parse out divs, cores, overlap from parallel_kw
    # take default from settings if not on parallel_kw dict
    divs = parallel_kw.get("divs", settings.divs)
    cores = parallel_kw.get("cores", settings.ncores)
    overlap = parallel_kw.get("overlap", settings.overlap)

    @dask.delayed
    def apply_func(func, **kwargs):
        # Apply function on sub-slice of overall image
        return func(**kwargs)

    # Determine the value for im_arg
    if isinstance(im_arg, str):
        im_arg = [im_arg]
    for item in im_arg:
        if item in kwargs.keys():
            im = kwargs[item]
            im_arg = item
            break
    # Fetch image from the kwargs dict
    im = kwargs[im_arg]
    # Determine the number of divisions to create
    divs = np.ones((im.ndim,), dtype=int) * np.array(divs)
    if cores is None:
        cores = settings.ncores
    # If overlap given then use it, otherwise search for strel in kwargs
    if overlap is not None:
        overlap = overlap * (divs > 1)
    else:
        if isinstance(strel_arg, str):
            strel_arg = [strel_arg]
        for item in strel_arg:
            if item in kwargs.keys():
                strel = kwargs[item]
                break
        overlap = np.array(strel.shape) * (divs > 1)
    slices = get_slices_grid(im=im, divs=divs, overlap=overlap)
    # Apply func to each subsection of the image
    res = []
    for s in slices:
        # Extract subsection from image and input into kwargs
        kwargs[im_arg] = dask.delayed(np.ascontiguousarray(im[tuple(s)]))
        res.append(apply_func(func=func, **kwargs))
    # Have dask actually compute the function on each subsection in parallel
    # with ProgressBar():
    # ims = dask.compute(res, num_workers=cores)[0]
    ims = dask.compute(res, num_workers=cores)[0]
    # Finally, put the pieces back together into a single master image, im2
    im2 = recombine(ims=ims, slices=slices, overlap=overlap)
    return im2
