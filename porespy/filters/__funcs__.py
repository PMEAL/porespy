import operator as op

import dask
import numpy as np
import scipy.ndimage as spim
from edt import edt
from loguru import logger
from skimage.morphology import (ball, cube, diamond, disk, octahedron,
                                reconstruction, square)
from skimage.segmentation import clear_border

from porespy import settings
from porespy.tools import (_check_for_singleton_axes, extract_subsection,
                           fftmorphology, get_border, get_tqdm, ps_ball,
                           ps_disk, subdivide, unpad)

tqdm = get_tqdm()


def apply_padded(im, pad_width, func, pad_val=1, **kwargs):
    r"""
    Applies padding to an image before sending to ``func``, then extracts
    the result corresponding to the original image shape.

    Parameters
    ----------
    im : ND-array
        The image to which ``func`` should be applied
    pad_width : int or list of ints
        The amount of padding to apply to each axis. Refer to
        ``numpy.pad`` documentation for more details.
    pad_val : scalar
        The value to place into the padded voxels.  The default is 1 (or
        ``True``) which extends the pore space.
    func : function handle
        The function to apply to the padded image.
    kwargs
        Additional keyword arguments are collected and passed to ``func``.

    Notes
    -----
    A use case for this is when using ``skimage.morphology.skeletonize_3d``
    to ensure that the skeleton extends beyond the edges of the image.

    """
    padded = np.pad(im, pad_width=pad_width,
                    mode='constant', constant_values=pad_val)
    temp = func(padded, **kwargs)
    result = unpad(im=temp, pad_width=pad_width)
    return result


def trim_small_clusters(im, size=1):
    r"""
    Remove isolated voxels or clusters of a given size or smaller

    Parameters
    ----------
    im : ND-array
        The binary image from which voxels are to be removed.
    size : scalar
        The threshold size of clusters to trim.  As clusters with this
        many voxels or fewer will be trimmed.  The default is 1 so only
        single voxels are removed.

    Returns
    -------
    im : ND-array
        A copy of ``im`` with clusters of voxels smaller than the given
        ``size`` removed.

    """
    if im.ndim == 2:
        strel = disk(1)
    elif im.ndim == 3:
        strel = ball(1)
    else:
        raise Exception("Only 2D or 3D images are accepted")
    filtered_array = np.copy(im)
    labels, N = spim.label(filtered_array, structure=strel)
    id_sizes = np.array(spim.sum(im, labels, range(N + 1)))
    area_mask = id_sizes <= size
    filtered_array[area_mask[labels]] = 0
    return filtered_array


def hold_peaks(im, axis=-1):
    r"""
    Replaces each voxel with the highest value along the given axis.

    Parameters
    ----------
    im : ND-array
        A greyscale image whose peaks are to be found.
    axis : int
        The axis along which the operation is to be applied.

    Returns
    -------
    result : ND-array
        A copy of ``im`` with each voxel is replaced with the highest value along
        the given axis.

    Notes
    -----
    "im" must be a greyscale image. In case a Boolean image is fed into this
    method, it will be first converted to float values [0.0,1.0] before proceeding.

    """

    A = im.astype(float)
    B = np.swapaxes(A, axis, -1)
    updown = np.empty((*B.shape[:-1], B.shape[-1] + 1), B.dtype)
    updown[..., 0], updown[..., -1] = -1, -1
    np.subtract(B[..., 1:], B[..., :-1], out=updown[..., 1:-1])
    chnidx = np.where(updown)
    chng = updown[chnidx]
    (pkidx,) = np.where((chng[:-1] > 0) & (chng[1:] < 0) | (chnidx[-1][:-1] == 0))
    pkidx = (*map(op.itemgetter(pkidx), chnidx),)
    out = np.zeros_like(A)
    aux = out.swapaxes(axis, -1)
    aux[(*map(op.itemgetter(slice(1, None)), pkidx),)] = np.diff(B[pkidx])
    aux[..., 0] = B[..., 0]
    result = out.cumsum(axis=axis)
    return result


def distance_transform_lin(im, axis=0, mode="both"):
    r"""
    Replaces each void voxel with the linear distance to the nearest solid
    voxel along the specified axis.

    Parameters
    ----------
    im : ND-array
        The image of the porous material with ``True`` values indicating
        the void phase (or phase of interest).
    axis : int
        The direction along which the distance should be measured, the
        default is 0 (i.e. along the x-direction).
    mode : str
        Controls how the distance is measured. Options are:

            'forward'
                Distances are measured in the increasing direction
                along the specified axis
            'reverse'
                Distances are measured in the reverse direction.
                'backward' is also accepted.
            'both'
                Distances are calculated in both directions (by
                recursively calling itself), then reporting the minimum value
                of the two results.

    Returns
    -------
    image : ND-array
        A copy of ``im`` with each foreground voxel containing the
        distance to the nearest background along the specified axis.

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


def find_disconnected_voxels(im, conn=None):
    r"""
    This identifies all pore (or solid) voxels that are not connected to
    the edge of the image.  This can be used to find blind pores, or
    remove artifacts such as solid phase voxels that are floating in space.

    Parameters
    ----------
    im : ND-array
        A Boolean image, with True values indicating the phase for which
        disconnected voxels are sought.
    conn : int
        For 2D the options are 4 and 8 for square and diagonal neighbors,
        while for the 3D the options are 6 and 26, similarily for square
        and diagonal neighbors. The default is the maximum option.

    Returns
    -------
    image : ND-array
        An ND-image the same size as ``im``, with True values indicating
        voxels of the phase of interest (i.e. True values in the original
        image) that are not connected to the outer edges.

    Notes
    -----
    image : ND-array
        The returned array (e.g. ``holes``) be used to trim blind pores
        from ``im`` using: ``im[holes] = False``

    """
    _check_for_singleton_axes(im)

    if im.ndim == 2:
        if conn == 4:
            strel = disk(1)
        elif conn in [None, 8]:
            strel = square(3)
        else:
            raise Exception("Received conn is not valid")
    elif im.ndim == 3:
        if conn == 6:
            strel = ball(1)
        elif conn in [None, 26]:
            strel = cube(3)
        else:
            raise Exception("Received conn is not valid")
    labels, N = spim.label(input=im, structure=strel)
    holes = clear_border(labels=labels) > 0
    return holes


def fill_blind_pores(im, conn=None):
    r"""
    Fills all pores that are not connected to the edges of the image.

    Parameters
    ----------
    im : ND-array
        The image of the porous material

    Returns
    -------
    image : ND-array
        A version of ``im`` but with all the disconnected pores removed.
    conn : int
        For 2D the options are 4 and 8 for square and diagonal neighbors,
        while for the 3D the options are 6 and 26, similarily for square
        and diagonal neighbors. The default is the maximum option.

    See Also
    --------
    find_disconnected_voxels

    """
    im = np.copy(im)
    holes = find_disconnected_voxels(im, conn=conn)
    im[holes] = False
    return im


def trim_floating_solid(im, conn=None):
    r"""
    Removes all solid that that is not attached to the edges of the image.

    Parameters
    ----------
    im : ND-array
        The image of the porous material
    conn : int
        For 2D the options are 4 and 8 for square and diagonal neighbors,
        while for the 3D the options are 6 and 26, similarily for square
        and diagonal neighbors. The default is the maximum option.

    Returns
    -------
    image : ND-array
        A version of ``im`` but with all the disconnected solid removed.

    See Also
    --------
    find_disconnected_voxels

    """
    im = np.copy(im)
    holes = find_disconnected_voxels(~im, conn=conn)
    im[holes] = True
    return im


def trim_nonpercolating_paths(im, inlets, outlets):
    r"""
    Remove all nonpercolating paths between specified edges or faces

    Parameters
    ----------
    im : ND-array
        The image of the porous material with ```True`` values indicating the
        phase of interest
    inlets : ND-image
        A boolean mask indicating locations of inlets, such as produced by
        ``porespy.generators.faces``.
    outlets : ND-image
        A boolean mask indicating locations of outlets, such as produced by
        ``porespy.generators.faces``.

    Returns
    -------
    image : ND-array
        A copy of ``im`` with all the nonpercolating paths removed

    Notes
    -----
    This function is essential when performing transport simulations on an
    image since image regions that do not span between the desired inlet and
    outlet do not contribute to the transport.

    See Also
    --------
    find_disconnected_voxels
    trim_floating_solid
    trim_blind_pores

    """
    labels = spim.label(im)[0]
    IN = np.unique(labels * inlets)
    OUT = np.unique(labels * outlets)
    hits = np.array(list(set(IN).intersection(set(OUT))))
    new_im = np.isin(labels, hits[hits > 0])
    return new_im


def trim_extrema(im, h, mode="maxima"):
    r"""
    Trims local extrema in greyscale values by a specified amount.

    This essentially decapitates peaks and/or floods valleys.

    Parameters
    ----------
    im : ND-array
        The image whose extrema are to be removed
    h : float
        The height to remove from each peak or fill in each valley
    mode : string {'maxima' | 'minima' | 'extrema'}
        Specifies whether to remove maxima or minima or both

    Returns
    -------
    image : ND-array
        A copy of the input image with all the peaks and/or valleys
        removed.

    Notes
    -----
    (1) This function is referred to as **imhmax** or **imhmin** in Matlab.

    (2) If the provided ``h`` is larger than ALL peaks in the array, then the
    baseline values of the array are changed as well.

    """
    mask = np.copy(im)
    im = np.copy(im)
    if mode == 'maxima':
        result = reconstruction(seed=im - h, mask=mask, method='dilation')
    elif mode == 'minima':
        result = reconstruction(seed=im + h, mask=mask, method='erosion')
    elif mode == 'extrema':
        result = reconstruction(seed=im - h, mask=mask, method='dilation')
        result = reconstruction(seed=result + h, mask=result, method='erosion')
    return result


def flood(im, regions=None, mode="max"):
    r"""
    Floods/fills each region in an image with a single value based on the
    specific values in that region.

    The ``mode`` argument is used to determine how the value is calculated.

    Parameters
    ----------
    im : array_like
        An image with isolated regions with numerical values in each voxel,
        and 0's elsewhere.
    regions : array_like
        An array the same shape as ``im`` with each region labeled.  If
        ``None`` is supplied (default) then ``scipy.ndimage.label`` is
        used with its default arguments.
    mode : string
        Specifies how to determine which value should be used to flood
        each region. Options are:

            'maximum'
                Floods each region with the local max in that region
            'minimum'
                Floods each region the local minimum in that region
            'median'
                Floods each region the local median in that region
            'mean'
                Floods each region the local mean in that region
            'size'
                Floods each region with the size of that region

    Returns
    -------
    image : ND-array
        A copy of ``im`` with new values placed in each forground voxel
        based on the ``mode``.

    See Also
    --------
    prop_to_image

    """
    mask = im > 0
    if regions is None:
        labels, N = spim.label(mask)
    else:
        labels = np.copy(regions)
        N = labels.max()
    mode = "sum" if mode == "size" else mode
    mode = "maximum" if mode == "max" else mode
    mode = "minimum" if mode == "min" else mode
    if mode in ["mean", "median", "maximum", "minimum", "sum"]:
        f = getattr(spim, mode)
        vals = f(input=im, labels=labels, index=range(0, N + 1))
        im_flooded = vals[labels]
        im_flooded = im_flooded * mask
    else:
        raise Exception(mode + " is not a recognized mode")
    return im_flooded


def flood_func(im, func, labels=None):
    r"""
    Flood each isolated region in an image with a constant value calculated by
    the given function.

    Parameters
    ----------
    im : ND-array
        The image containing distinct regions which should each be flooded
    func : Numpy function handle
        The function to be applied to each region in the image.  Any Numpy
        function that returns a scalar value can be passed, such as ``amin``,
        ``amax``, ``sum``, ``mean``, ``median``, etc.
    labels : ND-image, optional
        An array containing labels identify each individual region to be
        flooded. If not provided then ``scipy.ndimage.label`` is applied to
        ``im > 0``.

    Returns
    -------
    flooded : ND-array
        An image the same size as ``im`` with each isolated region flooded
        with a constant value based on the given ``func`` and the values
        in ``im``.

    """
    if labels is None:
        labels, N = spim.label(im > 0)
    slices = spim.find_objects(labels)
    new_im = np.zeros_like(im, dtype=float)
    for i, s in enumerate(slices):
        sub_im = labels[s] == (i + 1)
        val = func(im[s][sub_im])
        new_im[s] += sub_im*val
    return new_im


def find_dt_artifacts(dt):
    r"""
    lavel points in a distance transform that are closer to image bounadry than
    solid.

    These points could *potentially* be erroneously high since their
    distance values do not reflect the possibility that solid may have
    been present beyond the border of the image but lost by trimming.

    Parameters
    ----------
    dt : ND-array
        The distance transform of the phase of interest.

    Returns
    -------
    image : ND-array
        An ND-array the same shape as ``dt`` with numerical values
        indicating the maximum amount of error in each volxel, which is
        found by subtracting the distance to nearest edge of image from
        the distance transform value. In other words, this is the error
        that would be found if there were a solid voxel lurking just
        beyond the nearest edge of the image.  Obviously, voxels with a
        value of zero have no error.

    """
    temp = np.ones(shape=dt.shape) * np.inf
    for ax in range(dt.ndim):
        dt_lin = distance_transform_lin(np.ones_like(temp, dtype=bool),
                                        axis=ax, mode="both")
        temp = np.minimum(temp, dt_lin)
    result = np.clip(dt - temp, a_min=0, a_max=np.inf)
    return result


def region_size(im):
    r"""
    Replace each voxel with size of region to which it belongs

    Parameters
    ----------
    im : ND-array
        Either a boolean image wtih ``True`` indicating the features of
        interest, in which case ``scipy.ndimage.label`` will be applied to
        find regions, or a greyscale image with integer values indicating
        regions.

    Returns
    -------
    image : ND-array
        A copy of ``im`` with each voxel value indicating the size of the
        region to which it belongs.  This is particularly useful for
        finding chord sizes on the image produced by ``apply_chords``.

    See Also
    --------
    flood

    """
    if im.dtype == bool:
        im = spim.label(im)[0]
    counts = np.bincount(im.flatten())
    counts[0] = 0
    chords = counts[im]
    return chords


def apply_chords(im, spacing=1, axis=0, trim_edges=True, label=False):
    r"""
    Adds chords to the void space in the specified direction. The chords
    are separated by 1 voxel plus the provided spacing.

    Parameters
    ----------
    im : ND-array
        An image of the porous material with void marked as ``True``.
    spacing : int
        Separation between chords.  The default is 1 voxel.  This can be
        decreased to 0, meaning that the chords all touch each other,
        which automatically sets to the ``label`` argument to ``True``.
    axis : int (default = 0)
        The axis along which the chords are drawn.
    trim_edges : bool (default = ``True``)
        Whether or not to remove chords that touch the edges of the image.
        These chords are artifically shortened, so skew the chord length
        distribution.
    label : bool (default is ``False``)
        If ``True`` the chords in the returned image are each given a
        unique label, such that all voxels lying on the same chord have
        the same value.  This is automatically set to ``True`` if spacing
        is 0, but is ``False`` otherwise.

    Returns
    -------
    image : ND-array
        A copy of ``im`` with non-zero values indicating the chords.

    See Also
    --------
    apply_chords_3D

    """
    _check_for_singleton_axes(im)
    if spacing < 0:
        raise Exception("Spacing cannot be less than 0")
    if spacing == 0:
        label = True
    result = np.zeros(im.shape, dtype=int)  # Will receive chords at end
    slxyz = [slice(None, None, spacing * (axis != i) + 1) for i in [0, 1, 2]]
    slices = tuple(slxyz[: im.ndim])
    s = [[0, 1, 0], [0, 1, 0], [0, 1, 0]]  # Straight-line structuring element
    if im.ndim == 3:  # Make structuring element 3D if necessary
        s = np.pad(np.atleast_3d(s), pad_width=((0, 0), (0, 0), (1, 1)),
                   mode="constant", constant_values=0)
    im = im[slices]
    s = np.swapaxes(s, 0, axis)
    chords = spim.label(im, structure=s)[0]
    if trim_edges:  # Label on border chords will be set to 0
        chords = clear_border(chords)
    result[slices] = chords  # Place chords into empty image created at top
    if label is False:  # Remove label if not requested
        result = result > 0
    return result


def apply_chords_3D(im, spacing=0, trim_edges=True):
    r"""
    Adds chords to the void space in all three principle directions. The
    chords are seprated by 1 voxel plus the provided spacing. Chords in
    the X, Y and Z directions are labelled 1, 2 and 3 resepctively.

    Parameters
    ----------
    im : ND-array
        A 3D image of the porous material with void space marked as True.
    spacing : int (default = 0)
        Chords are automatically separed by 1 voxel on all sides, and this
        argument increases the separation.
    trim_edges : bool (default is ``True``)
        Whether or not to remove chords that touch the edges of the image.
        These chords are artifically shortened, so skew the chord length
        distribution

    Returns
    -------
    image : ND-array
        A copy of ``im`` with values of 1 indicating x-direction chords,
        2 indicating y-direction chords, and 3 indicating z-direction
        chords.

    Notes
    -----
    The chords are separated by a spacing of at least 1 voxel so that
    tools that search for connected components, such as
    ``scipy.ndimage.label`` can detect individual chords.

    See Also
    --------
    apply_chords

    """
    _check_for_singleton_axes(im)
    if im.ndim < 3:
        raise Exception("Must be a 3D image to use this function")
    if spacing < 0:
        raise Exception("Spacing cannot be less than 0")
    ch = np.zeros_like(im, dtype=int)
    ch[:, :: 4 + 2 * spacing, :: 4 + 2 * spacing] = 1   # X-direction
    ch[:: 4 + 2 * spacing, :, 2::4 + 2 * spacing] = 2   # Y-direction
    ch[2::4 + 2 * spacing, 2::4 + 2 * spacing, :] = 3   # Z-direction
    chords = ch * im
    if trim_edges:
        temp = clear_border(spim.label(chords > 0)[0]) > 0
        chords = temp * chords
    return chords


def local_thickness(im, sizes=25, mode="hybrid", **kwargs):
    r"""
    For each voxel, this function calculates the radius of the largest
    sphere that both engulfs the voxel and fits entirely within the
    foreground.

    This is not the same as a simple distance transform, which finds the
    largest sphere that could be *centered* on each voxel.

    Parameters
    ----------
    im : ND-array
        A binary image with the phase of interest set to True
    sizes : array_like or scalar
        The sizes to invade.  If a list of values of provided they are
        used directly. If a scalar is provided then that number of points
        spanning the min and max of the distance transform are used.
    mode : str
        Controls with method is used to compute the result. Options are:

            'hybrid'
                (default) Performs a distance tranform of the void
                space, thresholds to find voxels larger than ``sizes[i]``, trims
                the resulting mask if ``access_limitations`` is ``True``, then
                dilates it using the efficient fft-method to obtain the
                non-wetting fluid configuration.
            'dt'
                Same as 'hybrid', except uses a second distance transform,
                relative to the thresholded mask, to find the invading fluid
                configuration. The choice of 'dt' or 'hybrid' depends on speed,
                which is system and installation specific.
            'mio'
                Using a single morphological image opening step to obtain
                the invading fluid confirguration directly, *then* trims if
                ``access_limitations`` is ``True``. This method is not ideal and
                is included for comparison purposes.

    Returns
    -------
    image : ND-array
        A copy of ``im`` with the pore size values in each voxel.

    See Also
    --------
    porosimetry

    Notes
    -----
    The term *foreground* is used since this function can be applied to
    both pore space or the solid, whichever is set to ``True``.

    This function is identical to ``porosimetry`` with ``access_limited``
    set to ``False``.

    The way local thickness is found in PoreSpy differs from the
    traditional method (i.e. used in ImageJ
    `<https://imagej.net/Local_Thickness>`_). Our approach is probably
    slower, but it allows for the same code to be used for
    ``local_thickness`` and ``porosimetry``, since we can 'trim' invaded
    regions that are not connected to the inlets in the ``porosimetry``
    function. This is not needed in ``local_thickness`` however.

    """
    im_new = porosimetry(im=im, sizes=sizes, access_limited=False, mode=mode, **kwargs)
    return im_new


def porosimetry(im, sizes=25, inlets=None, access_limited=True, mode='hybrid',
                fft=True, **kwargs):
    r"""
    Performs a porosimetry simulution on an image.

    Parameters
    ----------
    im : ND-array
        An ND image of the porous material containing ``True`` values in the
        pore space.
    sizes : array_like or scalar
        The sizes to invade.  If a list of values of provided they are
        used directly.  If a scalar is provided then that number of points
        spanning the min and max of the distance transform are used.
    inlets : ND-array, boolean
        A boolean mask with ``True`` values indicating where the invasion
        enters the image.  By default all faces are considered inlets,
        akin to a mercury porosimetry experiment.  Users can also apply
        solid boundaries to their image externally before passing it in,
        allowing for complex inlets like circular openings, etc.
        This argument is only used if ``access_limited`` is ``True``.
    access_limited : bool
        This flag indicates if the intrusion should only occur from the
        surfaces (``access_limited`` is ``True``, which is the default),
        or if the invading phase should be allowed to appear in the core
        of the image.  The former simulates experimental tools like
        mercury intrusion porosimetry, while the latter is useful for
        comparison to gauge the extent of shielding effects in the sample.
    mode : str
        Controls with method is used to compute the result. Options are:

            'hybrid'
                (default) Performs a distance tranform of the void
                space, thresholds to find voxels larger than ``sizes[i]``,
                trims the resulting mask if ``access_limitations`` is ``True``,
                then dilates it using the efficient fft-method to obtain the
                non-wetting fluid configuration.
            'dt'
                Same as 'hybrid', except uses a second distance
                transform, relative to the thresholded mask, to find the
                invading fluid configuration. The choice of 'dt' or 'hybrid'
                depends on speed, which is system and installation specific.
            'mio'
                Using a single morphological image opening step to
                obtain the invading fluid confirguration directly, *then* trims
                if ``access_limitations`` is ``True``.  This method is not
                ideal and is included for comparison purposes. The
                morphological operations are done using fft-based method
                implementations.

    fft : boolean (default is ``True``)
        Indicates whether to use the ``fftmorphology`` function in
        ``porespy.filters`` or to use the standard morphology functions in
        ``scipy.ndimage``.  Always use ``fft=True`` unless you have a good
        reason not to.

    Returns
    -------
    image : ND-array
        A copy of ``im`` with voxel values indicating the sphere radius at
        which it becomes accessible from the inlets.  This image can be
        used to find invading fluid configurations as a function of
        applied capillary pressure by applying a boolean comparison:
        ``inv_phase = im > r`` where ``r`` is the radius (in voxels) of
        the invading sphere.  Of course, ``r`` can be converted to
        capillary pressure using a preferred model.

    Notes
    -----
    There are many ways to perform this filter, and PoreSpy offers 3,
    which users can choose between via the ``mode`` argument. These
    methods all work in a similar way by finding which foreground voxels
    can accomodate a sphere of a given radius, then repeating for smaller
    radii.

    See Also
    --------
    local_thickness

    """
    _check_for_singleton_axes(im)
    dt = edt(im > 0)

    if inlets is None:
        inlets = get_border(im.shape, mode="faces")

    if isinstance(sizes, int):
        sizes = np.logspace(start=np.log10(np.amax(dt)), stop=0, num=sizes)
    else:
        sizes = np.unique(sizes)[-1::-1]

    if im.ndim == 2:
        strel = ps_disk
        strel_2 = disk
    else:
        strel = ps_ball
        strel_2 = ball
    # Parse kwargs for any parallelization arguments
    parallel = kwargs.pop('parallel', False)
    cores = kwargs.pop('cores', None)
    divs = kwargs.pop('divs', 2)

    if mode == "mio":
        pw = int(np.floor(dt.max()))
        impad = np.pad(im, mode="symmetric", pad_width=pw)
        inlets = np.pad(inlets, mode="symmetric", pad_width=pw)
        # sizes = np.unique(np.around(sizes, decimals=0).astype(int))[-1::-1]
        imresults = np.zeros(np.shape(impad))
        for r in tqdm(sizes, **settings.tqdm):
            if parallel:
                imtemp = chunked_func(func=spim.binary_erosion,
                                      input=impad, structure=strel(r),
                                      overlap=int(2*r) + 1,
                                      cores=cores, divs=divs)
            elif fft:
                imtemp = fftmorphology(impad, strel(r), mode="erosion")
            else:
                imtemp = spim.binary_erosion(input=impad,
                                             structure=strel(r))
            if access_limited:
                imtemp = trim_disconnected_blobs(imtemp, inlets,
                                                 strel=strel_2(1))
            if parallel:
                imtemp = chunked_func(func=spim.binary_dilation,
                                      input=imtemp, structure=strel(r),
                                      overlap=int(2*r) + 1,
                                      cores=cores, divs=divs)
            elif fft:
                imtemp = fftmorphology(imtemp, strel(r), mode="dilation")
            else:
                imtemp = spim.binary_dilation(input=imtemp,
                                              structure=strel(r))
            if np.any(imtemp):
                imresults[(imresults == 0) * imtemp] = r
        imresults = extract_subsection(imresults, shape=im.shape)
    elif mode == "dt":
        imresults = np.zeros(np.shape(im))
        for r in tqdm(sizes, **settings.tqdm):
            imtemp = dt >= r
            if access_limited:
                imtemp = trim_disconnected_blobs(imtemp, inlets,
                                                 strel=strel_2(1))
            if np.any(imtemp):
                imtemp = edt(~imtemp) < r
                imresults[(imresults == 0) * imtemp] = r
    elif mode == "hybrid":
        imresults = np.zeros(np.shape(im))
        for r in tqdm(sizes, **settings.tqdm):
            imtemp = dt >= r
            if access_limited:
                imtemp = trim_disconnected_blobs(imtemp, inlets,
                                                 strel=strel_2(1))
            if np.any(imtemp):
                if parallel:
                    imtemp = chunked_func(func=spim.binary_dilation,
                                          input=imtemp, structure=strel(r),
                                          overlap=int(2*r) + 1,
                                          cores=cores, divs=divs)
                elif fft:
                    imtemp = fftmorphology(imtemp, strel(r),
                                           mode="dilation")
                else:
                    imtemp = spim.binary_dilation(input=imtemp,
                                                  structure=strel(r))
                imresults[(imresults == 0) * imtemp] = r
    else:
        raise Exception("Unrecognized mode " + mode)
    return imresults


def trim_disconnected_blobs(im, inlets, strel=None):
    r"""
    Removes foreground voxels not connected to specified inlets.

    Parameters
    ----------
    im : ND-array
        The image containing the blobs to be trimmed
    inlets : ND-array or tuple of indices
        The locations of the inlets.  Can either be a boolean mask the
        same shape as ``im``, or a tuple of indices such as that returned
        by the ``where`` function.  Any voxels *not* connected directly to
        the inlets will be trimmed.
    strel : array-like
        The neighborhood over which connectivity should be checked. It
        must be symmetric and the same dimensionality as the image. It is
        passed directly to the ``scipy.ndimage.label`` function as the
        ``structure`` argument so refer to that docstring for additional
        info.

    Returns
    -------
    image : ND-array
        An array of the same shape as ``im``, but with all foreground
        voxels not connected to the ``inlets`` removed.

    """
    if type(inlets) == tuple:
        temp = np.copy(inlets)
        inlets = np.zeros_like(im, dtype=bool)
        inlets[temp] = True
    elif (inlets.shape == im.shape) and (inlets.max() == 1):
        inlets = inlets.astype(bool)
    else:
        raise Exception("inlets not valid, refer to docstring for info")
    if strel is None:
        if im.ndim == 3:
            strel = cube(3)
        else:
            strel = square(3)
    labels = spim.label(inlets + (im > 0), structure=strel)[0]
    keep = np.unique(labels[inlets])
    keep = keep[keep > 0]
    im2 = np.isin(labels, keep)
    im2 = im2 * im
    return im2


def _get_axial_shifts(ndim=2, include_diagonals=False):
    r"""
    Helper function to generate the axial shifts that will be performed on
    the image to identify bordering pixels/voxels
    """
    if ndim == 2:
        if include_diagonals:
            neighbors = square(3)
        else:
            neighbors = diamond(1)
        neighbors[1, 1] = 0
        x, y = np.where(neighbors)
        x -= 1
        y -= 1
        return np.vstack((x, y)).T
    else:
        if include_diagonals:
            neighbors = cube(3)
        else:
            neighbors = octahedron(1)
        neighbors[1, 1, 1] = 0
        x, y, z = np.where(neighbors)
        x -= 1
        y -= 1
        z -= 1
        return np.vstack((x, y, z)).T


def _make_stack(im, include_diagonals=False):
    r"""
    Creates a stack of images with one extra dimension to the input image
    with length equal to the number of borders to search + 1.

    Image is rolled along the axial shifts so that the border pixel is
    overlapping the original pixel. First image in stack is the original.
    Stacking makes direct vectorized array comparisons possible.

    """
    ndim = len(np.shape(im))
    axial_shift = _get_axial_shifts(ndim, include_diagonals)
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


def nphase_border(im, include_diagonals=False):
    r"""
    Identifies the voxels in regions that border *N* other regions.

    Useful for finding triple-phase boundaries.

    Parameters
    ----------
    im : ND-array
        An ND image of the porous material containing discrete values in
        the pore space identifying different regions. e.g. the result of a
        snow-partition

    include_diagonals : bool
        When identifying bordering pixels (2D) and voxels (3D) include
        those shifted along more than one axis

    Returns
    -------
    image : ND-array
        A copy of ``im`` with voxel values equal to the number of uniquely
        different bordering values

    """
    _check_for_singleton_axes(im)
    # Get dimension of image
    ndim = len(np.shape(im))
    if ndim not in [2, 3]:
        raise NotImplementedError("Function only works for 2d and 3d images")
    # Pad image to handle edges
    im = np.pad(im, pad_width=1, mode="edge")
    # Stack rolled images for each neighbor to be inspected
    stack = _make_stack(im, include_diagonals)
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


def prune_branches(skel, branch_points=None, iterations=1, **kwargs):
    r"""
    Removes all dangling ends or tails of a skeleton.

    Parameters
    ----------
    skel : ND-array
        A image of a full or partial skeleton from which the tails should
        be trimmed.

    branch_points : ND-array, optional
        An image the same size ``skel`` with True values indicating the
        branch points of the skeleton.  If this is not provided it is
        calculated automatically.

    Returns
    -------
    array
        An ND-array containing the skeleton with tails removed.

    """
    skel = skel > 0
    if skel.ndim == 2:
        from skimage.morphology import square as cube
    else:
        from skimage.morphology import cube
    parallel = kwargs.pop('parallel', False)
    divs = kwargs.pop('divs', 2)
    cores = kwargs.pop('cores', None)
    # Create empty image to house results
    im_result = np.zeros_like(skel)
    # If branch points are not supplied, attempt to find them
    if branch_points is None:
        branch_points = spim.convolve(skel * 1.0, weights=cube(3)) > 3
        branch_points = branch_points * skel
    # Store original branch points before dilating
    pts_orig = branch_points
    # Find arcs of skeleton by deleting branch points
    arcs = skel * (~branch_points)
    # Label arcs
    arc_labels = spim.label(arcs, structure=cube(3))[0]
    # Dilate branch points so they overlap with the arcs
    if parallel:
        branch_points = chunked_func(func=spim.binary_dilation,
                                     input=branch_points, structure=cube(3),
                                     overlap=3, divs=divs, cores=cores)
    else:
        branch_points = spim.binary_dilation(branch_points, structure=cube(3))
    pts_labels = spim.label(branch_points, structure=cube(3))[0]
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
        im_result = prune_branches(skel=im_result,
                                   branch_points=None,
                                   iterations=iterations,
                                   parallel=parallel,
                                   divs=divs, cores=cores)
        if np.all(im_temp == im_result):
            iterations = 0
    return im_result


def chunked_func(func,
                 overlap=None,
                 divs=2,
                 cores=None,
                 im_arg=["input", "image", "im"],
                 strel_arg=["strel", "structure", "footprint"],
                 **kwargs):
    r"""
    Performs the specfied operation "chunk-wise" in parallel using dask.

    This can be used to save memory by doing one chunk at a time
    (``cores=1``) or to increase computation speed by spreading the work
    across multiple cores (e.g. ``cores=8``)

    This function can be used with any operation that applies a
    structuring element of some sort, since this implies that the
    operation is local and can be chunked.

    Parameters
    ----------
    func : function handle
        The function which should be applied to each chunk, such as
        ``spipy.ndimage.binary_dilation``.
    overlap : scalar or list of scalars, optional
        The amount of overlap to include when dividing up the image. This
        value will almost always be the size (i.e. diameter) of the
        structuring element. If not specified then the amount of overlap
        is inferred from the size of the structuring element, in which
        case the ``strel_arg`` must be specified.
    divs : scalar or list of scalars (default = [2, 2, 2])
        The number of chunks to divide the image into in each direction.
        The default is 2 chunks in each direction, resulting in a
        quartering of the image and 8 total chunks (in 3D).  A scalar is
        interpreted as applying to all directions, while a list of scalars
        is interpreted as applying to each individual direction.
    cores : scalar
        The number of cores which should be used.  By default, all cores
        will be used, or as many are needed for the given number of
        chunks, which ever is smaller.
    im_arg : str
        The keyword used by ``func`` for the image to be operated on. By
        default this function will look for ``image``, ``input``, and
        ``im`` which are commonly used by *scipy.ndimage* and *skimage*.
    strel_arg : str
        The keyword used by ``func`` for the structuring element to apply.
        This is only needed if ``overlap`` is not specified. By default
        this function will look for ``strel``, ``structure``, and
        ``footprint`` which are commonly used by *scipy.ndimage* and
        *skimage*.
    kwargs
        All other arguments are passed to ``func`` as keyword arguments.
        Note that PoreSpy will fetch the image from this list of keywords
        using the value provided to ``im_arg``.

    Returns
    -------
    result : ND-array
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
    specify ``overlap``.

    See Also
    --------
    skikit-image.util.apply_parallel

    Examples
    --------
    >>> import scipy.ndimage as spim
    >>> import porespy as ps
    >>> from skimage.morphology import ball
    >>> im = ps.generators.blobs(shape=[100, 100, 100])
    >>> f = spim.binary_dilation
    >>> im2 = ps.filters.chunked_func(func=f, overlap=7, im_arg='input',
    ...                               input=im, structure=ball(3), cores=1)
    >>> im3 = spim.binary_dilation(input=im, structure=ball(3))
    >>> np.all(im2 == im3)
    True

    """

    @dask.delayed
    def apply_func(func, **kwargs):
        # Apply function on sub-slice of overall image
        return func(**kwargs)

    # Determine the value for im_arg
    if type(im_arg) == str:
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
    # If overlap given then use it, otherwise search for strel in kwargs
    if overlap is not None:
        halo = overlap * (divs > 1)
    else:
        if type(strel_arg) == str:
            strel_arg = [strel_arg]
        for item in strel_arg:
            if item in kwargs.keys():
                strel = kwargs[item]
                break
        halo = np.array(strel.shape) * (divs > 1)
    slices = subdivide(im=im, divs=divs, overlap=halo, flatten=True)
    # Apply func to each subsection of the image
    res = []
    for s in slices:
        # Extract subsection from image and input into kwargs
        kwargs[im_arg] = im[tuple(s)]
        res.append(apply_func(func=func, **kwargs))
    # Have dask actually compute the function on each subsection in parallel
    # with ProgressBar():
        # ims = dask.compute(res, num_workers=cores)[0]
    ims = dask.compute(res, num_workers=cores)[0]
    # Finally, put the pieces back together into a single master image, im2
    im2 = np.zeros_like(im, dtype=im.dtype)
    for i, s in enumerate(slices):
        # Prepare new slice objects into main and sub-sliced image
        a = []  # Slices into main image
        b = []  # Slices into chunked image
        for dim in range(im.ndim):
            if s[dim].start == 0:
                ax = bx = 0
            else:
                ax = s[dim].start + halo[dim]
                bx = halo[dim]
            if s[dim].stop == im.shape[dim]:
                ay = by = im.shape[dim]
            else:
                ay = s[dim].stop - halo[dim]
                by = s[dim].stop - s[dim].start - halo[dim]
            a.append(slice(ax, ay, None))
            b.append(slice(bx, by, None))
        # Convert lists of slices to tuples
        a = tuple(a)
        b = tuple(b)
        # Insert image chunk into main image
        try:
            im2[a] = ims[i][b]
        except ValueError:
            raise IndexError('The applied filter seems to have returned a '
                             + 'larger image that it was sent.')
    return im2
