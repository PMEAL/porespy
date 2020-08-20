import sys
import dask
import dask.array as da
from dask.diagnostics import ProgressBar
import warnings
import numpy as np
from numba import njit, prange
from edt import edt
import operator as op
from tqdm import tqdm
import scipy.ndimage as spim
import scipy.spatial as sptl
from collections import namedtuple
from scipy.signal import fftconvolve
from skimage.segmentation import clear_border, watershed
from skimage.morphology import ball, disk, square, cube, diamond, octahedron
from skimage.morphology import reconstruction
from porespy.tools import randomize_colors, fftmorphology
from porespy.tools import get_border, extend_slice, extract_subsection
from porespy.tools import _create_alias_map
from porespy.tools import ps_disk, ps_ball


def apply_padded(im, pad_width, func, pad_val=1, **kwargs):
    r"""
    Applies padding to an image before sending to ``func``, then extracts the
    result corresponding to the original image shape.

    Parameters
    ----------
    im : ND-image
        The image to which ``func`` should be applied
    pad_width : int or list of ints
        The amount of padding to apply to each axis.  Refer to ``numpy.pad``
        documentation for more details.
    pad_val : scalar
        The value to place into the padded voxels.  The default is 1 (or
        ``True``) which extends the pore space.
    func : function handle
        The function to apply to the padded image
    kwargs : additional keyword arguments
        All additional keyword arguments are collected and passed to ``func``.

    Notes
    -----
    A use case for this is when using ``skimage.morphology.skeletonize_3d``
    to ensure that the skeleton extends beyond the edges of the image.
    """
    padded = np.pad(im, pad_width=pad_width,
                    mode='constant', constant_values=pad_val)
    temp = func(padded, **kwargs)
    result = extract_subsection(im=temp, shape=im.shape)
    return result


def trim_small_clusters(im, size=1):
    r"""
    Remove isolated voxels or clusters of a given size or smaller

    Parameters
    ----------
    im : ND-array
        The binary image from which voxels are to be removed
    size : scalar
        The threshold size of clusters to trim.  As clusters with this many
        voxels or fewer will be trimmed.  The default is 1 so only single
        voxels are removed.

    Returns
    -------
    im : ND-image
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
    Replaces each voxel with the highest value along the given axis

    Parameters
    ----------
    im : ND-image
        A greyscale image whose peaks are to
    """
    A = im
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
        The image of the porous material with ``True`` values indicating the
        void phase (or phase of interest)

    axis : int
        The direction along which the distance should be measured, the default
        is 0 (i.e. along the x-direction)

    mode : string
        Controls how the distance is measured.  Options are:

        'forward' - Distances are measured in the increasing direction along
        the specified axis

        'reverse' - Distances are measured in the reverse direction.
        *'backward'* is also accepted.

        'both' - Distances are calculated in both directions (by recursively
        calling itself), then reporting the minimum value of the two results.

    Returns
    -------
    image : ND-array
        A copy of ``im`` with each foreground voxel containing the distance to
        the nearest background along the specified axis.
    """
    if im.ndim != im.squeeze().ndim:
        warnings.warn(
            "Input image conains a singleton axis:"
            + str(im.shape)
            + " Reduce dimensionality with np.squeeze(im) to avoid"
            + " unexpected behavior."
        )
    if mode in ["backward", "reverse"]:
        im = np.flip(im, axis)
        im = distance_transform_lin(im=im, axis=axis, mode="forward")
        im = np.flip(im, axis)
        return im
    elif mode in ["both"]:
        im_f = distance_transform_lin(im=im, axis=axis, mode="forward")
        im_b = distance_transform_lin(im=im, axis=axis, mode="backward")
        return np.minimum(im_f, im_b)
    else:
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


def snow_partitioning(im, dt=None, r_max=4, sigma=0.4, return_all=False,
                      mask=True, randomize=True):
    r"""
    Partitions the void space into pore regions using a marker-based watershed
    algorithm, with specially filtered peaks as markers.

    The SNOW network extraction algorithm (Sub-Network of an Over-segmented
    Watershed) was designed to handle to perculiarities of high porosity
    materials, but it applies well to other materials as well.

    Parameters
    ----------
    im : array_like
        A boolean image of the domain, with ``True`` indicating the pore space
        and ``False`` elsewhere.
    dt : array_like, optional
        The distance transform of the pore space.  This is done automatically
        if not provided, but if the distance transform has already been
        computed then supplying it can save some time.
    r_max : int
        The radius of the spherical structuring element to use in the Maximum
        filter stage that is used to find peaks.  The default is 4
    sigma : float
        The standard deviation of the Gaussian filter used in step 1.  The
        default is 0.4.  If 0 is given then the filter is not applied, which is
        useful if a distance transform is supplied as the ``im`` argument that
        has already been processed.
    return_all : boolean
        If set to ``True`` a named tuple is returned containing the original
        image, the distance transform, the filtered peaks, and the final
        pore regions.  The default is ``False``
    mask : boolean
        Apply a mask to the regions where the solid phase is.  Default is
        ``True``
    randomize : boolean
        If ``True`` (default), then the region colors will be randomized before
        returning.  This is helpful for visualizing otherwise neighboring
        regions have simlar coloring are are hard to distinguish.

    Returns
    -------
    image : ND-array
        An image the same shape as ``im`` with the void space partitioned into
        pores using a marker based watershed with the peaks found by the
        SNOW algorithm [1].

    Notes
    -----
    If ``return_all`` is ``True`` then a **named tuple** is returned containing
    all of the images used during the process.  They can be access as
    attriutes with the following names:

        * ``im``: The binary image of the void space
        * ``dt``: The distance transform of the image
        * ``peaks``: The peaks of the distance transform after applying the
        steps of the SNOW algorithm
        * ``regions``: The void space partitioned into pores using a marker
        based watershed with the peaks found by the SNOW algorithm

    References
    ----------
    [1] Gostick, J. "A versatile and efficient network extraction algorithm
    using marker-based watershed segmenation".  Physical Review E. (2017)

    """
    tup = namedtuple("results", field_names=["im", "dt", "peaks", "regions"])
    print("-" * 60)
    print("Beginning SNOW Algorithm")
    im_shape = np.array(im.shape)
    if im.dtype is not bool:
        print("Converting supplied image (im) to boolean")
        im = im > 0
    if dt is None:
        print("Peforming Distance Transform")
        if np.any(im_shape == 1):
            ax = np.where(im_shape == 1)[0][0]
            dt = edt(im.squeeze())
            dt = np.expand_dims(dt, ax)
        else:
            dt = edt(im)

    tup.im = im
    tup.dt = dt

    if sigma > 0:
        print("Applying Gaussian blur with sigma =", str(sigma))
        dt = spim.gaussian_filter(input=dt, sigma=sigma)

    peaks = find_peaks(dt=dt, r_max=r_max)
    print("Initial number of peaks: ", spim.label(peaks)[1])
    peaks = trim_saddle_points(peaks=peaks, dt=dt, max_iters=500)
    print("Peaks after trimming saddle points: ", spim.label(peaks)[1])
    peaks = trim_nearby_peaks(peaks=peaks, dt=dt)
    peaks, N = spim.label(peaks)
    print("Peaks after trimming nearby peaks: ", N)
    tup.peaks = peaks
    if mask:
        mask_solid = im > 0
    else:
        mask_solid = None
    regions = watershed(image=-dt, markers=peaks, mask=mask_solid)
    if randomize:
        regions = randomize_colors(regions)
    if return_all:
        tup.regions = regions
        return tup
    else:
        return regions


def snow_partitioning_n(im, r_max=4, sigma=0.4, return_all=True,
                        mask=True, randomize=False, alias=None):
    r"""
    This function partitions an imaging oontain an arbitrary number of phases
    into regions using a marker-based watershed segmentation. Its an extension
    of snow_partitioning function with all phases partitioned together.

    Parameters
    ----------
    im : ND-array
        Image of porous material where each phase is represented by unique
        integer starting from 1 (0's are ignored).
    r_max : scalar
        The radius of the spherical structuring element to use in the Maximum
        filter stage that is used to find peaks.  The default is 4.
    sigma : scalar
        The standard deviation of the Gaussian filter used.  The default is
        0.4. If 0 is given then the filter is not applied, which is useful if a
        distance transform is supplied as the ``im`` argument that has already
        been processed.
    return_all : boolean (default is False)
        If set to ``True`` a named tuple is returned containing the original
        image, the combined distance transform, list of each phase max label,
        and the final combined regions of all phases.
    mask : boolean (default is True)
        Apply a mask to the regions which are not under concern.
    randomize : boolean
        If ``True`` (default), then the region colors will be randomized before
        returning.  This is helpful for visualizing otherwise neighboring
        regions have similar coloring and are hard to distinguish.
    alias : dict (Optional)
        A dictionary that assigns unique image label to specific phases. For
        example {1: 'Solid'} will show all structural properties associated
        with label 1 as Solid phase properties. If ``None`` then default
        labelling will be used i.e {1: 'Phase1',..}.

    Returns
    -------
    An image the same shape as ``im`` with the all phases partitioned into
    regions using a marker based watershed with the peaks found by the
    SNOW algorithm [1].  If ``return_all`` is ``True`` then a **named tuple**
    is returned with the following attribute:

        * ``im`` : The actual image of the porous material
        * ``dt`` : The combined distance transform of the image
        * ``phase_max_label`` : The list of max label of each phase in order to
        distinguish between each other
        * ``regions`` : The partitioned regions of n phases using a marker
        based watershed with the peaks found by the SNOW algorithm

    References
    ----------
    [1] Gostick, J. "A versatile and efficient network extraction algorithm
    using marker-based watershed segmentation".  Physical Review E. (2017)

    [2] Khan, ZA et al. "Dual network extraction algorithm to investigate
    multiple transport processes in porous materials: Image-based modeling
    of pore and grain-scale processes".  Computers in Chemical Engineering.
    (2019)

    See Also
    ----------
    snow_partitioning

    Notes
    -----
    In principle it is possible to perform a distance transform on each
    phase separately, merge these into a single image, then apply the
    watershed only once. This, however, has been found to create edge artifacts
    between regions arising from the way watershed handles plateaus in the
    distance transform. To overcome this, this function applies the watershed
    to each of the distance transforms separately, then merges the segmented
    regions back into a single image.

    """
    # Get alias if provided by user
    al = _create_alias_map(im=im, alias=alias)
    # Perform snow on each phase and merge all segmentation and dt together
    phases_num = np.unique(im * 1)
    phases_num = np.trim_zeros(phases_num)
    combined_dt = 0
    combined_region = 0
    num = [0]
    for i, j in enumerate(phases_num):
        print("_" * 60)
        if alias is None:
            print("Processing Phase {}".format(j))
        else:
            print("Processing Phase {}".format(al[j]))
        phase_snow = snow_partitioning(
            im == j,
            dt=None,
            r_max=r_max,
            sigma=sigma,
            return_all=return_all,
            mask=mask,
            randomize=randomize,
        )
        combined_dt += phase_snow.dt
        phase_snow.regions *= phase_snow.im
        phase_snow.regions += num[i]
        phase_ws = phase_snow.regions * phase_snow.im
        phase_ws[phase_ws == num[i]] = 0
        combined_region += phase_ws
        num.append(np.amax(combined_region))
    if return_all:
        tup = namedtuple(
            "results", field_names=["im", "dt", "phase_max_label", "regions"]
        )
        tup.im = im
        tup.dt = combined_dt
        tup.phase_max_label = num[1:]
        tup.regions = combined_region
        return tup
    else:
        return combined_region


def find_peaks(dt, r_max=4, footprint=None, **kwargs):
    r"""
    Finds local maxima in the distance transform

    Parameters
    ----------
    dt : ND-array
        The distance transform of the pore space.  This may be calculated and
        filtered using any means desired.
    r_max : scalar
        The size of the structuring element used in the maximum filter.  This
        controls the localness of any maxima. The default is 4 voxels.
    footprint : ND-array
        Specifies the shape of the structuring element used to define the
        neighborhood when looking for peaks.  If ``None`` (the default) is
        specified then a spherical shape is used (or circular in 2D).

    Returns
    -------
    image : ND-array
        An array of booleans with ``True`` values at the location of any
        local maxima.

    Notes
    -----
    It is also possible ot the ``peak_local_max`` function from the
    ``skimage.feature`` module as follows:

    ``peaks = peak_local_max(image=dt, min_distance=r, exclude_border=0,
    indices=False)``

    The *skimage* function automatically uses a square structuring element
    which is significantly faster than using a circular or spherical element.
    """
    im = dt > 0
    if im.ndim != im.squeeze().ndim:
        warnings.warn("Input image conains a singleton axis:"
                      + str(im.shape)
                      + " Reduce dimensionality with np.squeeze(im) to avoid"
                      + " unexpected behavior.")
    if footprint is None:
        if im.ndim == 2:
            footprint = disk
        elif im.ndim == 3:
            footprint = ball
        else:
            raise Exception("only 2-d and 3-d images are supported")
    parallel = kwargs.pop('parallel', False)
    cores = kwargs.pop('cores', None)
    divs = kwargs.pop('cores', 2)
    if parallel:
        overlap = max(footprint(r_max).shape)
        peaks = chunked_func(func=find_peaks, overlap=overlap,
                             im_arg='dt', dt=dt, footprint=footprint,
                             cores=cores, divs=divs)
    else:
        mx = spim.maximum_filter(dt + 2 * (~im), footprint=footprint(r_max))
        peaks = (dt == mx) * im
    return peaks


def reduce_peaks(peaks):
    r"""
    Any peaks that are broad or elongated are replaced with a single voxel
    that is located at the center of mass of the original voxels.

    Parameters
    ----------
    peaks : ND-image
        An image containing ``True`` values indicating peaks in the distance
        transform

    Returns
    -------
    image : ND-array
        An array with the same number of isolated peaks as the original image,
        but fewer total ``True`` voxels.

    Notes
    -----
    The center of mass of a group of voxels is used as the new single voxel, so
    if the group has an odd shape (like a horse shoe), the new voxel may *not*
    lie on top of the original set.
    """
    if peaks.ndim == 2:
        strel = square
    else:
        strel = cube
    markers, N = spim.label(input=peaks, structure=strel(3))
    inds = spim.measurements.center_of_mass(
        input=peaks, labels=markers, index=np.arange(1, N + 1)
    )
    inds = np.floor(inds).astype(int)
    # Centroid may not be on old pixel, so create a new peaks image
    peaks_new = np.zeros_like(peaks, dtype=bool)
    peaks_new[tuple(inds.T)] = True
    return peaks_new


def trim_saddle_points(peaks, dt, max_iters=10, verbose=1):
    r"""
    Removes peaks that were mistakenly identified because they lied on a
    saddle or ridge in the distance transform that was not actually a true
    local peak.

    Parameters
    ----------
    peaks : ND-array
        A boolean image containing True values to mark peaks in the distance
        transform (``dt``)

    dt : ND-array
        The distance transform of the pore space for which the true peaks are
        sought.

    max_iters : int
        The maximum number of iterations to run while eroding the saddle
        points.  The default is 10, which is usually not reached; however,
        a warning is issued if the loop ends prior to removing all saddle
        points.

    Returns
    -------
    image : ND-array
        An image with fewer peaks than the input image

    References
    ----------
    [1] Gostick, J. "A versatile and efficient network extraction algorithm
    using marker-based watershed segmenation".  Physical Review E. (2017)

    """
    peaks = np.copy(peaks)
    if dt.ndim == 2:
        from skimage.morphology import square as cube
    else:
        from skimage.morphology import cube
    labels, N = spim.label(peaks)
    slices = spim.find_objects(labels)
    for i in range(N):
        s = extend_slice(s=slices[i], shape=peaks.shape, pad=10)
        peaks_i = labels[s] == i + 1
        dt_i = dt[s]
        im_i = dt_i > 0
        iters = 0
        peaks_dil = np.copy(peaks_i)
        while iters < max_iters:
            iters += 1
            peaks_dil = spim.binary_dilation(input=peaks_dil, structure=cube(3))
            peaks_max = peaks_dil * np.amax(dt_i * peaks_dil)
            peaks_extended = (peaks_max == dt_i) * im_i
            if np.all(peaks_extended == peaks_i):
                break  # Found a true peak
            elif np.sum(peaks_extended * peaks_i) == 0:
                peaks_i = False
                break  # Found a saddle point
        peaks[s] = peaks_i
        if iters >= max_iters and verbose:
            print(
                "Maximum number of iterations reached, consider "
                + "running again with a larger value of max_iters"
            )
    return peaks


def trim_nearby_peaks(peaks, dt):
    r"""
    Finds pairs of peaks that are nearer to each other than to the solid phase,
    and removes the peak that is closer to the solid.

    Parameters
    ----------
    peaks : ND-array
        A boolean image containing True values to mark peaks in the distance
        transform (``dt``)

    dt : ND-array
        The distance transform of the pore space for which the true peaks are
        sought.

    Returns
    -------
    image : ND-array
        An array the same size as ``peaks`` containing a subset of the peaks
        in the original image.

    Notes
    -----
    Each pair of peaks is considered simultaneously, so for a triplet of peaks
    each pair is considered.  This ensures that only the single peak that is
    furthest from the solid is kept.  No iteration is required.

    References
    ----------
    [1] Gostick, J. "A versatile and efficient network extraction algorithm
    using marker-based watershed segmenation".  Physical Review E. (2017)
    """
    peaks = np.copy(peaks)
    if dt.ndim == 2:
        from skimage.morphology import square as cube
    else:
        from skimage.morphology import cube
    peaks, N = spim.label(peaks, structure=cube(3))
    crds = spim.measurements.center_of_mass(peaks, labels=peaks,
                                            index=np.arange(1, N + 1))
    crds = np.vstack(crds).astype(int)  # Convert to numpy array of ints
    # Get distance between each peak as a distance map
    tree = sptl.cKDTree(data=crds)
    temp = tree.query(x=crds, k=2)
    nearest_neighbor = temp[1][:, 1]
    dist_to_neighbor = temp[0][:, 1]
    del temp, tree  # Free-up memory
    dist_to_solid = dt[tuple(crds.T)]  # Get distance to solid for each peak
    hits = np.where(dist_to_neighbor < dist_to_solid)[0]
    # Drop peak that is closer to the solid than it's neighbor
    drop_peaks = []
    for peak in hits:
        if dist_to_solid[peak] < dist_to_solid[nearest_neighbor[peak]]:
            drop_peaks.append(peak)
        else:
            drop_peaks.append(nearest_neighbor[peak])
    drop_peaks = np.unique(drop_peaks)
    # Remove peaks from image
    slices = spim.find_objects(input=peaks)
    for s in drop_peaks:
        peaks[slices[s]] = 0
    return peaks > 0


def find_disconnected_voxels(im, conn=None):
    r"""
    This identifies all pore (or solid) voxels that are not connected to the
    edge of the image.  This can be used to find blind pores, or remove
    artifacts such as solid phase voxels that are floating in space.

    Parameters
    ----------
    im : ND-image
        A Boolean image, with True values indicating the phase for which
        disconnected voxels are sought.
    conn : int
        For 2D the options are 4 and 8 for square and diagonal neighbors, while
        for the 3D the options are 6 and 26, similarily for square and diagonal
        neighbors.  The default is the maximum option.

    Returns
    -------
    image : ND-array
        An ND-image the same size as ``im``, with True values indicating
        voxels of the phase of interest (i.e. True values in the original
        image) that are not connected to the outer edges.

    Notes
    -----
    image : ND-array
        The returned array (e.g. ``holes``) be used to trim blind pores from
        ``im`` using: ``im[holes] = False``

    """
    if im.ndim != im.squeeze().ndim:
        warnings.warn(
            "Input image conains a singleton axis:"
            + str(im.shape)
            + " Reduce dimensionality with np.squeeze(im) to avoid"
            + " unexpected behavior."
        )
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
        For 2D the options are 4 and 8 for square and diagonal neighbors, while
        for the 3D the options are 6 and 26, similarily for square and diagonal
        neighbors.  The default is the maximum option.

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
        For 2D the options are 4 and 8 for square and diagonal neighbors, while
        for the 3D the options are 6 and 26, similarily for square and diagonal
        neighbors.  The default is the maximum option.

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


def trim_nonpercolating_paths(im, inlet_axis=0, outlet_axis=0,
                              inlets=None, outlets=None):
    r"""
    Removes all nonpercolating paths between specified edges

    This function is essential when performing transport simulations on an
    image, since image regions that do not span between the desired inlet and
    outlet do not contribute to the transport.

    Parameters
    ----------
    im : ND-array
        The image of the porous material with ```True`` values indicating the
        phase of interest
    inlet_axis : int
        Inlet axis of boundary condition. For three dimensional image the
        number ranges from 0 to 2. For two dimensional image the range is
        between 0 to 1. If ``inlets`` is given then this argument is ignored.
    outlet_axis : int
        Outlet axis of boundary condition. For three dimensional image the
        number ranges from 0 to 2. For two dimensional image the range is
        between 0 to 1. If ``outlets`` is given then this argument is ignored.
    inlets : ND-image (optional)
        A boolean mask indicating locations of inlets.  If this argument is
        supplied then ``inlet_axis`` is ignored.
    outlets : ND-image (optional)
        A boolean mask indicating locations of outlets. If this argument is
        supplied then ``outlet_axis`` is ignored.

    Returns
    -------
    image : ND-array
        A copy of ``im`` with all the nonpercolating paths removed

    See Also
    --------
    find_disconnected_voxels
    trim_floating_solid
    trim_blind_pores

    """
    if im.ndim != im.squeeze().ndim:
        warnings.warn(
            "Input image conains a singleton axis:"
            + str(im.shape)
            + " Reduce dimensionality with np.squeeze(im) to avoid"
            + " unexpected behavior."
        )
    im = trim_floating_solid(~im)
    labels = spim.label(~im)[0]
    if inlets is None:
        inlets = np.zeros_like(im, dtype=bool)
        if im.ndim == 3:
            if inlet_axis == 0:
                inlets[0, :, :] = True
            elif inlet_axis == 1:
                inlets[:, 0, :] = True
            elif inlet_axis == 2:
                inlets[:, :, 0] = True
        if im.ndim == 2:
            if inlet_axis == 0:
                inlets[0, :] = True
            elif inlet_axis == 1:
                inlets[:, 0] = True
    if outlets is None:
        outlets = np.zeros_like(im, dtype=bool)
        if im.ndim == 3:
            if outlet_axis == 0:
                outlets[-1, :, :] = True
            elif outlet_axis == 1:
                outlets[:, -1, :] = True
            elif outlet_axis == 2:
                outlets[:, :, -1] = True
        if im.ndim == 2:
            if outlet_axis == 0:
                outlets[-1, :] = True
            elif outlet_axis == 1:
                outlets[:, -1] = True
    IN = np.unique(labels * inlets)
    OUT = np.unique(labels * outlets)
    new_im = np.isin(labels, list(set(IN) ^ set(OUT)), invert=True)
    im[new_im == 0] = True
    return ~im


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
        A copy of the input image with all the peaks and/or valleys removed.

    Notes
    -----
    This function is referred to as **imhmax** or **imhmin** in Matlab.

    """
    result = im
    if mode in ["maxima", "extrema"]:
        result = reconstruction(seed=im - h, mask=im, method="dilation")
    elif mode in ["minima", "extrema"]:
        result = reconstruction(seed=im + h, mask=im, method="erosion")
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
        ``None`` is supplied (default) then ``scipy.ndimage.label`` is used
        with its default arguments.
    mode : string
        Specifies how to determine which value should be used to flood each
        region.  Options are:

        'maximum' - Floods each region with the local maximum in that region

        'minimum' - Floods each region the local minimum in that region

        'median' - Floods each region the local median in that region

        'mean' - Floods each region the local mean in that region

        'size' - Floods each region with the size of that region

    Returns
    -------
    image : ND-array
        A copy of ``im`` with new values placed in each forground voxel based
        on the ``mode``.

    See Also
    --------
    props_to_image

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


def find_dt_artifacts(dt):
    r"""
    Finds points in a distance transform that are closer to wall than solid.

    These points could *potentially* be erroneously high since their distance
    values do not reflect the possibility that solid may have been present
    beyond the border of the image but lost by trimming.

    Parameters
    ----------
    dt : ND-array
        The distance transform of the phase of interest

    Returns
    -------
    image : ND-array
        An ND-array the same shape as ``dt`` with numerical values indicating
        the maximum amount of error in each volxel, which is found by
        subtracting the distance to nearest edge of image from the distance
        transform value. In other words, this is the error that would be found
        if there were a solid voxel lurking just beyond the nearest edge of
        the image.  Obviously, voxels with a value of zero have no error.

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
        region to which it belongs.  This is particularly useful for finding
        chord sizes on the image produced by ``apply_chords``.

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
    Adds chords to the void space in the specified direction.  The chords are
    separated by 1 voxel plus the provided spacing.

    Parameters
    ----------
    im : ND-array
        An image of the porous material with void marked as ``True``.

    spacing : int
        Separation between chords.  The default is 1 voxel.  This can be
        decreased to 0, meaning that the chords all touch each other, which
        automatically sets to the ``label`` argument to ``True``.

    axis : int (default = 0)
        The axis along which the chords are drawn.

    trim_edges : bool (default = ``True``)
        Whether or not to remove chords that touch the edges of the image.
        These chords are artifically shortened, so skew the chord length
        distribution.

    label : bool (default is ``False``)
        If ``True`` the chords in the returned image are each given a unique
        label, such that all voxels lying on the same chord have the same
        value.  This is automatically set to ``True`` if spacing is 0, but is
        ``False`` otherwise.

    Returns
    -------
    image : ND-array
        A copy of ``im`` with non-zero values indicating the chords.

    See Also
    --------
    apply_chords_3D

    """
    if im.ndim != im.squeeze().ndim:
        warnings.warn(
            "Input image conains a singleton axis:"
            + str(im.shape)
            + " Reduce dimensionality with np.squeeze(im) to avoid"
            + " unexpected behavior."
        )
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
    Adds chords to the void space in all three principle directions.  The
    chords are seprated by 1 voxel plus the provided spacing.  Chords in the X,
    Y and Z directions are labelled 1, 2 and 3 resepctively.

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
        2 indicating y-direction chords, and 3 indicating z-direction chords.

    Notes
    -----
    The chords are separated by a spacing of at least 1 voxel so that tools
    that search for connected components, such as ``scipy.ndimage.label`` can
    detect individual chords.

    See Also
    --------
    apply_chords

    """
    if im.ndim != im.squeeze().ndim:
        warnings.warn(
            "Input image conains a singleton axis:"
            + str(im.shape)
            + " Reduce dimensionality with np.squeeze(im) to avoid"
            + " unexpected behavior."
        )
    if im.ndim < 3:
        raise Exception("Must be a 3D image to use this function")
    if spacing < 0:
        raise Exception("Spacing cannot be less than 0")
    ch = np.zeros_like(im, dtype=int)
    ch[:, :: 4 + 2 * spacing, :: 4 + 2 * spacing] = 1       # X-direction
    ch[:: 4 + 2 * spacing, :, 2::4 + 2 * spacing] = 2     # Y-direction
    ch[2::4 + 2 * spacing, 2::4 + 2 * spacing, :] = 3   # Z-direction
    chords = ch * im
    if trim_edges:
        temp = clear_border(spim.label(chords > 0)[0]) > 0
        chords = temp * chords
    return chords


def local_thickness(im, sizes=25, mode="hybrid", **kwargs):
    r"""
    For each voxel, this function calculates the radius of the largest sphere
    that both engulfs the voxel and fits entirely within the foreground.

    This is not the same as a simple distance transform, which finds the
    largest sphere that could be *centered* on each voxel.

    Parameters
    ----------
    im : array_like
        A binary image with the phase of interest set to True

    sizes : array_like or scalar
        The sizes to invade.  If a list of values of provided they are used
        directly.  If a scalar is provided then that number of points spanning
        the min and max of the distance transform are used.

    mode : string
        Controls with method is used to compute the result.  Options are:

        'hybrid' - (default) Performs a distance tranform of the void space,
        thresholds to find voxels larger than ``sizes[i]``, trims the resulting
        mask if ``access_limitations`` is ``True``, then dilates it using the
        efficient fft-method to obtain the non-wetting fluid configuration.

        'dt' - Same as 'hybrid', except uses a second distance transform,
        relative to the thresholded mask, to find the invading fluid
        configuration.  The choice of 'dt' or 'hybrid' depends on speed, which
        is system and installation specific.

        'mio' - Using a single morphological image opening step to obtain the
        invading fluid confirguration directly, *then* trims if
        ``access_limitations`` is ``True``.  This method is not ideal and is
        included mostly for comparison purposes.

    Returns
    -------
    image : ND-array
        A copy of ``im`` with the pore size values in each voxel

    See Also
    --------
    porosimetry

    Notes
    -----
    The term *foreground* is used since this function can be applied to both
    pore space or the solid, whichever is set to ``True``.

    This function is identical to ``porosimetry`` with ``access_limited`` set
    to ``False``.

    The way local thickness is found in PoreSpy differs from the traditional
    method (i.e. `used in ImageJ <https://imagej.net/Local_Thickness>`_).
    Our approach is probably slower, but it allows for the same code to be
    used for ``local_thickness`` and ``porosimetry``, since we can 'trim'
    invaded regions that are not connected to the inlets in the ``porosimetry``
    function.  This is not needed in ``local_thickness`` however.

    """
    im_new = porosimetry(im=im, sizes=sizes, access_limited=False, mode=mode,
                         **kwargs)
    return im_new


def porosimetry(im, sizes=25, inlets=None, access_limited=True, mode='hybrid',
                fft=True, **kwargs):
    r"""
    Performs a porosimetry simulution on an image

    Parameters
    ----------
    im : ND-array
        An ND image of the porous material containing ``True`` values in the
        pore space.

    sizes : array_like or scalar
        The sizes to invade.  If a list of values of provided they are used
        directly.  If a scalar is provided then that number of points spanning
        the min and max of the distance transform are used.

    inlets : ND-array, boolean
        A boolean mask with ``True`` values indicating where the invasion
        enters the image.  By default all faces are considered inlets,
        akin to a mercury porosimetry experiment.  Users can also apply
        solid boundaries to their image externally before passing it in,
        allowing for complex inlets like circular openings, etc.  This argument
        is only used if ``access_limited`` is ``True``.

    access_limited : Boolean
        This flag indicates if the intrusion should only occur from the
        surfaces (``access_limited`` is ``True``, which is the default), or
        if the invading phase should be allowed to appear in the core of
        the image.  The former simulates experimental tools like mercury
        intrusion porosimetry, while the latter is useful for comparison
        to gauge the extent of shielding effects in the sample.

    mode : string
        Controls with method is used to compute the result.  Options are:

        'hybrid' - (default) Performs a distance tranform of the void space,
        thresholds to find voxels larger than ``sizes[i]``, trims the resulting
        mask if ``access_limitations`` is ``True``, then dilates it using the
        efficient fft-method to obtain the non-wetting fluid configuration.

        'dt' - Same as 'hybrid', except uses a second distance transform,
        relative to the thresholded mask, to find the invading fluid
        configuration.  The choice of 'dt' or 'hybrid' depends on speed, which
        is system and installation specific.

        'mio' - Using a single morphological image opening step to obtain the
        invading fluid confirguration directly, *then* trims if
        ``access_limitations`` is ``True``.  This method is not ideal and is
        included mostly for comparison purposes.  The morphological operations
        are done using fft-based method implementations.

    fft : boolean (default is ``True``)
        Indicates whether to use the ``fftmorphology`` function in
        ``porespy.filters`` or to use the standard morphology functions in
        ``scipy.ndimage``.  Always use ``fft=True`` unless you have a good
        reason not to.

    Returns
    -------
    image : ND-array
        A copy of ``im`` with voxel values indicating the sphere radius at
        which it becomes accessible from the inlets.  This image can be used
        to find invading fluid configurations as a function of applied
        capillary pressure by applying a boolean comparison:
        ``inv_phase = im > r`` where ``r`` is the radius (in voxels) of the
        invading sphere.  Of course, ``r`` can be converted to capillary
        pressure using a preferred model.

    Notes
    -----
    There are many ways to perform this filter, and PoreSpy offers 3, which
    users can choose between via the ``mode`` argument.  These methods all
    work in a similar way by finding which foreground voxels can accomodate
    a sphere of a given radius, then repeating for smaller radii.

    See Also
    --------
    local_thickness

    """
    if im.ndim != im.squeeze().ndim:
        warnings.warn("Input image contains a singleton axis:"
                      + str(im.shape)
                      + " Reduce dimensionality with np.squeeze(im) to avoid"
                      + " unexpected behavior.")

    dt = edt(im > 0)

    if inlets is None:
        inlets = get_border(im.shape, mode="faces")

    if isinstance(sizes, int):
        sizes = np.logspace(start=np.log10(np.amax(dt)), stop=0, num=sizes)
    else:
        sizes = np.unique(sizes)[-1::-1]

    if im.ndim == 2:
        strel = ps_disk
    else:
        strel = ps_ball

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
        with tqdm(sizes) as pbar:
            for r in sizes:
                pbar.update()
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
                    imtemp = trim_disconnected_blobs(imtemp, inlets)
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
        with tqdm(sizes) as pbar:
            for r in sizes:
                pbar.update()
                imtemp = dt >= r
                if access_limited:
                    imtemp = trim_disconnected_blobs(imtemp, inlets)
                if np.any(imtemp):
                    imtemp = edt(~imtemp) < r
                    imresults[(imresults == 0) * imtemp] = r
    elif mode == "hybrid":
        imresults = np.zeros(np.shape(im))
        with tqdm(sizes) as pbar:
            for r in sizes:
                pbar.update()
                imtemp = dt >= r
                if access_limited:
                    imtemp = trim_disconnected_blobs(imtemp, inlets)
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
    Removes foreground voxels not connected to specified inlets

    Parameters
    ----------
    im : ND-array
        The image containing the blobs to be trimmed
    inlets : ND-array or tuple of indices
        The locations of the inlets.  Can either be a boolean mask the same
        shape as ``im``, or a tuple of indices such as that returned by the
        ``where`` function.  Any voxels *not* connected directly to
        the inlets will be trimmed.
    strel : ND-array
        The neighborhood over which connectivity should be checked. It must
        be symmetric and the same dimensionality as the image.  It is passed
        directly to the ``scipy.ndimage.label`` function as the ``structure``
        argument so refer to that docstring for additional info.

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
    if im.ndim == 3:
        strel = cube
    else:
        strel = square
    labels = spim.label(inlets + (im > 0), structure=strel(3))[0]
    keep = np.unique(labels[inlets])
    keep = keep[keep > 0]
    if len(keep) > 0:
        im2 = np.reshape(np.in1d(labels, keep), newshape=im.shape)
    else:
        im2 = np.zeros_like(im)
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
        An ND image of the porous material containing discrete values in the
        pore space identifying different regions. e.g. the result of a
        snow-partition

    include_diagonals : boolean
        When identifying bordering pixels (2D) and voxels (3D) include those
        shifted along more than one axis

    Returns
    -------
    image : ND-array
        A copy of ``im`` with voxel values equal to the number of uniquely
        different bordering values
    """
    if im.ndim != im.squeeze().ndim:
        warnings.warn(
            "Input image conains a singleton axis:"
            + str(im.shape)
            + " Reduce dimensionality with np.squeeze(im) to avoid"
            + " unexpected behavior."
        )
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
    skel : ND-image
        A image of a full or partial skeleton from which the tails should be
        trimmed.

    branch_points : ND-image, optional
        An image the same size ``skel`` with True values indicating the branch
        points of the skeleton.  If this is not provided it is calculated
        automatically.

    Returns
    -------
    An ND-image containing the skeleton with tails removed.

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
    Performs the specfied operation "chunk-wise" in parallel using dask

    This can be used to save memory by doing one chunk at a time (``cores=1``)
    or to increase computation speed by spreading the work across multiple
    cores (e.g. ``cores=8``)

    This function can be used with any operation that applies a structuring
    element of some sort, since this implies that the operation is local
    and can be chunked.

    Parameters
    ----------
    func : function handle
        The function which should be applied to each chunk, such as
        ``spipy.ndimage.binary_dilation``.
    overlap : scalar or list of scalars, optional
        The amount of overlap to include when dividing up the image.  This
        value will almost always be the size (i.e. diameter) of the
        structuring element. If not specified then the amount of overlap is
        inferred from the size of the structuring element, in which case the
        ``strel_arg`` must be specified.
    divs : scalar or list of scalars (default = [2, 2, 2])
        The number of chunks to divide the image into in each direction.  The
        default is 2 chunks in each direction, resulting in a quartering of
        the image and 8 total chunks (in 3D).  A scalar is interpreted as
        applying to all directions, while a list of scalars is interpreted
        as applying to each individual direction.
    cores : scalar
        The number of cores which should be used.  By default, all cores will
        be used, or as many are needed for the given number of chunks, which
        ever is smaller.
    im_arg : string
        The keyword used by ``func`` for the image to be operated on.  By
        default this function will look for ``image``, ``input``, and ``im``
        which are commonly used by *scipy.ndimage* and *skimage*.
    strel_arg : string
        The keyword used by ``func`` for the structuring element to apply.
        This is only needed if ``overlap`` is not specified. By default this
        function will look for ``strel``, ``structure``, and ``footprint``
        which are commonly used by *scipy.ndimage* and *skimage*.
    kwargs : additional keyword arguments
        All other arguments are passed to ``func`` as keyword arguments. Note
        that PoreSpy will fetch the image from this list of keywords using the
        value provided to ``im_arg``.

    Returns
    -------
    result : ND-image
        An image the same size as the input image, with the specified filter
        applied as though done on a single large image.  There should be *no*
        difference.

    Notes
    -----
    This function divides the image into the specified number of chunks, but
    also applies a padding to each chunk to create an overlap with neighboring
    chunks.  This way the operation does not have any edge artifacts. The
    amount of padding is usually equal to the radius of the structuring
    element but some functions do not use one, such as the distance transform
    and Gaussian blur.  In these cases the user can specify ``overlap``.

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

    # Import the array_split methods
    from array_split import shape_split, ARRAY_BOUNDS

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
    slices = np.ravel(
        shape_split(
            im.shape, axis=divs, halo=halo.tolist(), tile_bounds_policy=ARRAY_BOUNDS
        )
    )
    # Apply func to each subsection of the image
    res = []
    # print('Image will be broken into the following chunks:')
    for s in slices:
        # Extract subsection from image and input into kwargs
        kwargs[im_arg] = im[tuple(s)]
        # print(kwargs[im_arg].shape)
        res.append(apply_func(func=func, **kwargs))
    # Have dask actually compute the function on each subsection in parallel
    # with ProgressBar():
    #    ims = dask.compute(res, num_workers=cores)[0]
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
        im2[a] = ims[i][b]
    return im2


def snow_partitioning_parallel(im,
                               overlap='dt',
                               divs=2,
                               mode='parallel',
                               num_workers=None,
                               crop=True,
                               zoom_factor=0.5,
                               r_max=5,
                               sigma=0.4,
                               return_all=False):
    r"""
    Perform SNOW algorithm in parallel and serial mode to reduce time and
    memory usage repectively by geomertirc domain decomposition of large size
    image.

    Parameters
    ----------
    im: ND_array
        A binary image of porous media with 'True' values indicating phase of
        interest

    overlap: float or int or str
        Overlapping thickness between two subdomains that is used to merge
        watershed segmented regions at intersection of two or more subdomains.
        If 'dt' the overlap will be calculated based on maximum
        distance transform in the whole image.
        If 'ws' the overlap will be calculated by finding the maximum dimension
        of the bounding box of largest segmented region. The image is scale down
        by 'zoom_factor' provided by user.
        If any real number of overlap is provided then this value will be
        considered as overlapping thickness.

    divs: list or int
        Number of domains each axis will be divided.
        If a scalar is provided then it will be assigned to all axis.
        If list is provided then each respective axis will be divided by its
        corresponding number in the list. For example [2, 3, 4] will divide
        z, y and x axis to 2, 3, and 4 respectively.

    mode: str
        if 'parallel' then all subdomains will be processed in number of cores
        provided as num_workers
        if 'serial' then all subdomains will be processed one by one in one core
        of CPU.

    num_workers: int or None
        Number of cores that will be used to parallel process all domains.
        If None then all cores will be used but user can specify any integer
        values to control the memory usage.

    crop: bool
        If True the image shape is cropped to fit specified division.

    zoom_factor: float or int
        The amount of zoom appiled to image to find overlap thickness using "ws"
        overlap mode.

    return_all : boolean
        If set to ``True`` a named tuple is returned containing the original
        image, the distance transform, and the final
        pore regions.  The default is ``False``

    Returns
    ----------
    regions: ND_array
        Partitioned image of segmentated regions with unique labels. Each
        region correspond to pore body while intersection with other region
        correspond throat area.
    """
    # --------------------------------------------------------------------------
    # Adjust image shape according to specified dimension
    tup = namedtuple("results", field_names=["im", "dt", "regions"])
    if isinstance(divs, int):
        divs = [divs for i in range(im.ndim)]
    shape = []
    for i in range(im.ndim):
        shape.append(divs[i] * (im.shape[i] // divs[i]))

    if shape != list(im.shape):
        if crop:
            for i in range(im.ndim):
                im = im.swapaxes(0, i)
                im = im[:shape[i], ...]
                im = im.swapaxes(i, 0)
            print(f'Image is cropped to shape {shape}.')
        else:
            print('-' * 80)
            print(f"Possible image shape for specified divisions is {shape}.")
            print("To crop the image please set crop argument to 'True'.")
            return
    # --------------------------------------------------------------------------
    # Get overlap thickness from distance transform
    chunk_shape = (np.array(shape) / np.array(divs)).astype(int)
    print('# Beginning parallel SNOW algorithm...')
    print('=' * 80)
    print('Calculating overlap thickness')
    if overlap == 'dt':
        dt = edt((im > 0), parallel=0)
        overlap = dt.max()
    elif overlap == 'ws':
        rev = spim.interpolation.zoom(im, zoom=zoom_factor, order=0)
        rev = rev > 0
        dt = edt(rev, parallel=0)
        rev_snow = snow_partitioning(rev, dt=dt, r_max=r_max, sigma=sigma)
        labels, counts = np.unique(rev_snow, return_counts=True)
        node = np.where(counts == counts[1:].max())[0][0]
        slices = spim.find_objects(rev_snow)
        overlap = max(rev_snow[slices[node - 1]].shape) / (zoom_factor * 2.0)
        dt = edt((im > 0), parallel=0)
    else:
        overlap = overlap / 2.0
        dt = edt((im > 0), parallel=0)
    print('Overlap Thickness: ' + str(int(2.0 * overlap)) + ' voxels')
    # --------------------------------------------------------------------------
    # Get overlap and trim depth of all image dimension
    depth = {}
    trim_depth = {}
    for i in range(im.ndim):
        depth[i] = int(2.0 * overlap)
        trim_depth[i] = int(2.0 * overlap) - 1

    tup.im = im
    tup.dt = dt
    # --------------------------------------------------------------------------
    # Applying snow to image chunks
    im = da.from_array(dt, chunks=chunk_shape)
    im = da.overlap.overlap(im, depth=depth, boundary='none')
    im = im.map_blocks(chunked_snow, r_max=r_max, sigma=sigma)
    im = da.overlap.trim_internal(im, trim_depth, boundary='none')
    if mode == 'serial':
        num_workers = 1
    elif mode == 'parallel':
        num_workers = num_workers
    else:
        raise Exception('Mode of operation can either be parallel or serial')
    with ProgressBar():
        # print('-' * 80)
        print('Applying snow to image chunks')
        regions = im.compute(num_workers=num_workers)
    # --------------------------------------------------------------------------
    # Relabelling watershed chunks
    # print('-' * 80)
    print('Relabelling watershed chunks')
    regions = relabel_chunks(im=regions, chunk_shape=chunk_shape)
    # --------------------------------------------------------------------------
    # Stitching watershed chunks
    # print('-' * 80)
    print('Stitching watershed chunks')
    regions = watershed_stitching(im=regions, chunk_shape=chunk_shape)
    print('=' * 80)
    if return_all:
        tup.regions = regions
        return tup
    else:
        return regions

    return regions


def chunked_snow(im, r_max=5, sigma=0.4):
    r"""
    Partitions the void space into pore regions using a marker-based watershed
    algorithm, with specially filtered peaks as markers.

    The SNOW network extraction algorithm (Sub-Network of an Over-segmented
    Watershed) was designed to handle to perculiarities of high porosity
    materials, but it applies well to other materials as well.

    Parameters
    ----------
    im : array_like
        Distance transform of phase of interest in a binary image
    r_max : int
        The radius of the spherical structuring element to use in the Maximum
        filter stage that is used to find peaks.  The default is 5
    sigma : float
        The standard deviation of the Gaussian filter used in step 1.  The
        default is 0.4.  If 0 is given then the filter is not applied, which is
        useful if a distance transform is supplied as the ``im`` argument that
        has already been processed.

    Returns
    -------
    image : ND-array
        An image the same shape as ``im`` with the void space partitioned into
        pores using a marker based watershed with the peaks found by the
        SNOW algorithm [1].

    References
    ----------
    [1] Gostick, J. "A versatile and efficient network extraction algorithm
    using marker-based watershed segmenation".  Physical Review E. (2017)
    """

    dt = spim.gaussian_filter(input=im, sigma=sigma)
    peaks = find_peaks(dt=dt, r_max=r_max)
    peaks = trim_saddle_points(peaks=peaks, dt=dt, max_iters=99, verbose=0)
    peaks = trim_nearby_peaks(peaks=peaks, dt=dt)
    peaks, N = spim.label(peaks)
    regions = watershed(image=-dt, markers=peaks, mask=im > 0)

    return regions * (im > 0)


def pad(im, pad_width=1, constant_value=0):
    r"""
    Pad the image with a constant values and width.

    Parameters:
    ----------
    im : ND-array
        The image that requires padding
    pad_width : int
        The number of values that will be padded from the edges. Default values
        is 1.
    contant_value : int
        Pads with the specified constant value

    return:
    -------
    output: ND-array
        Padded image with same dimnesions as provided image
    """
    shape = np.array(im.shape)
    pad_shape = shape + (2 * pad_width)
    temp = np.zeros(pad_shape, dtype=np.uint32)
    if constant_value != 0:
        temp = temp + constant_value
    if im.ndim == 3:
        temp[pad_width: -pad_width,
             pad_width: -pad_width,
             pad_width: -pad_width] = im
    elif im.ndim == 2:
        temp[pad_width: -pad_width,
             pad_width: -pad_width] = im
    else:
        temp[pad_width: -pad_width] = im

    return temp


def relabel_chunks(im, chunk_shape):  # pragma: no cover
    r"""
    Assign new labels to each chunk or sub-domain of actual image. This
    prevents from two or more regions to have same label.

    Parameters:
    -----------

    im: ND-array
        Actual image that contains repeating labels in chunks or sub-domains

    chunk_shape: tuple
        The shape of chunk that will be relabeled in actual image. Note the
        chunk shape should be a multiple of actual image shape otherwise some
        labels will not be relabeled.

    return:
    -------
    output : ND-array
        Relabeled image with unique label assigned to each region.
    """
    im = pad(im, pad_width=1)
    im_shape = np.array(im.shape, dtype=np.uint32)
    max_num = 0
    c = np.array(chunk_shape, dtype=np.uint32) + 2
    num = (im_shape / c).astype(int)

    if im.ndim == 3:
        for z in range(num[0]):
            for y in range(num[1]):
                for x in range(num[2]):
                    chunk = im[z * c[0]: (z + 1) * c[0],
                               y * c[1]: (y + 1) * c[1],
                               x * c[2]: (x + 1) * c[2]]
                    chunk += max_num
                    chunk[chunk == max_num] = 0
                    max_num = chunk.max()
                    im[z * c[0]: (z + 1) * c[0],
                       y * c[1]: (y + 1) * c[1],
                       x * c[2]: (x + 1) * c[2]] = chunk
    else:
        for y in range(num[0]):
            for x in range(num[1]):
                chunk = im[y * c[0]: (y + 1) * c[0],
                           x * c[1]: (x + 1) * c[1]]
                chunk += max_num
                chunk[chunk == max_num] = 0
                max_num = chunk.max()
                im[y * c[0]: (y + 1) * c[0],
                   x * c[1]: (x + 1) * c[1]] = chunk

    return im


def trim_internal_slice(im, chunk_shape):  # pragma: no cover
    r"""
    Delete extra slices from image that were used to stitch two or more chunks
    together.

    Parameters:
    -----------

    im :  ND-array
        image that contains extra slices in x, y, z direction.

    chunk_shape : tuple
        The shape of the chunk from which image is subdivided.

    Return:
    -------
    output :  ND-array
        Image without extra internal slices. The shape of the image will be
        same as input image provided for  waterhsed segmentation.
    """
    im_shape = np.array(im.shape, dtype=np.uint32)
    c1 = np.array(chunk_shape, dtype=np.uint32) + 2
    c2 = np.array(chunk_shape, dtype=np.uint32)
    num = (im_shape / c1).astype(int)
    out_shape = num * c2
    out = np.empty((out_shape), dtype=np.uint32)

    if im.ndim == 3:
        for z in range(num[0]):
            for y in range(num[1]):
                for x in range(num[2]):
                    chunk = im[z * c1[0]: (z + 1) * c1[0],
                               y * c1[1]: (y + 1) * c1[1],
                               x * c1[2]: (x + 1) * c1[2]]

                    out[z * c2[0]: (z + 1) * c2[0],
                        y * c2[1]: (y + 1) * c2[1],
                        x * c2[2]: (x + 1) * c2[2]] = chunk[1:-1, 1:-1, 1:-1]
    else:
        for y in range(num[0]):
            for x in range(num[1]):
                chunk = im[y * c1[0]: (y + 1) * c1[0],
                           x * c1[1]: (x + 1) * c1[1]]

                out[y * c2[0]: (y + 1) * c2[0],
                    x * c2[1]: (x + 1) * c2[1]] = chunk[1:-1, 1:-1]

    return out


def watershed_stitching(im, chunk_shape):
    r"""
    Stitch individual sub-domains of watershed segmentation into one big
    segmentation with all boundary labels of each sub-domain relabeled to merge
    boundary regions.

    Parameters:
    -----------
    im : ND-array
        A worked image with watershed segmentation performed on all sub-domains
        individually.

    chunk_shape: tuple
        The shape of the sub-domain in which image segmentation is performed.

    return:
    -------
    output : ND-array
        Stitched watershed segmentation with all sub-domains merged to form a
        single watershed segmentation.
    """

    c_shape = np.array(chunk_shape)
    cuts_num = (np.array(im.shape) / c_shape).astype(np.uint32)

    for axis, num in enumerate(cuts_num):
        keys = []
        values = []
        if num > 1:
            im = im.swapaxes(0, axis)
            for i in range(1, num):
                sl = i * (chunk_shape[axis] + 3) - (i - 1)
                sl1 = im[sl - 3, ...]
                sl1_mask = sl1 > 0
                sl2 = im[sl - 1, ...] * sl1_mask
                sl1_labels = sl1.flatten()[sl1.flatten() > 0]
                sl2_labels = sl2.flatten()[sl2.flatten() > 0]
                if sl1_labels.size != sl2_labels.size:
                    raise Exception('The selected overlapping thickness is not '
                                    'suitable for input image. Change '
                                    'overlapping criteria '
                                    'or manually input value.')
                keys.append(sl1_labels)
                values.append(sl2_labels)
            im = replace_labels(array=im, keys=keys, values=values)
            im = im.swapaxes(axis, 0)
    im = trim_internal_slice(im=im, chunk_shape=chunk_shape)
    im = resequence_labels(array=im)

    return im


@njit(parallel=True)
def copy(im, output):  # pragma: no cover
    r"""
    The function copy the input array and make output array that is allocated
    in different memory space. This a numba version of copy function of numpy.
    Because each element is copied using parallel approach this implementation
    is facter than numpy version of copy.

    parameter:
    ----------
    array: ND-array
        Array that needs to be copied

    Return:
    -------
    output: ND-array
        Copied array
    """

    if im.ndim == 3:
        for i in prange(im.shape[0]):
            for j in prange(im.shape[1]):
                for k in prange(im.shape[2]):
                    output[i, j, k] = im[i, j, k]
    elif im.ndim == 2:
        for i in prange(im.shape[0]):
            for j in prange(im.shape[1]):
                output[i, j] = im[i, j]
    else:
        for i in prange(im.shape[0]):
            output[i] = im[i]

    return output


@njit(parallel=True)
def _replace(array, keys, values, ind_sort):  # pragma: no cover
    r"""
    This function replace keys elements in input array with new value elements.
    This function is used as internal function of replace_relabels.

    Parameter:
    ----------
    array : ND-array
        Array which requires replacing labels
    keys :  1D-array
        The unique labels that need to be replaced
    values : 1D-array
        The unique values that will be assigned to labels

    return:
    -------
    array : ND-array
        Array with replaced labels.
    """
    # ind_sort = np.argsort(keys)
    keys_sorted = keys[ind_sort]
    values_sorted = values[ind_sort]
    s_keys = set(keys)

    for i in prange(array.shape[0]):
        if array[i] in s_keys:
            ind = np.searchsorted(keys_sorted, array[i])
            array[i] = values_sorted[ind]


def replace_labels(array, keys, values):
    r"""
    Replace labels in array provided as keys to values.

    Parameter:
    ----------
    array : ND-array
        Array which requires replacing labels
    keys :  1D-array
        The unique labels that need to be replaced
    values : 1D-array
        The unique values that will be assigned to labels

    return:
    -------
    array : ND-array
        Array with replaced labels.
    """
    a_shape = array.shape
    array = array.flatten()
    keys = np.concatenate(keys, axis=0)
    values = np.concatenate(values, axis=0)
    ind_sort = np.argsort(keys)
    _replace(array, keys, values, ind_sort)

    return array.reshape(a_shape)


@njit()
def _sequence(array, count):  # pragma: no cover
    r"""
    Internal function of resequnce_labels method. This function resquence array
    elements in an ascending order using numba technique which is many folds
    faster than make contigious funcition.

    parameter:
    ----------
    array: 1d-array
        1d-array that needs resquencing
    count: 1d-array
        1d-array of zeros having same size as array

    return:
    -------
    array: 1d-array
        The input array with elements resequenced in ascending order
    Note: The output of this function is not same as make_contigous or
    relabel_sequential function of scikit-image. This function resequence and
    randomize the regions while other methods only do resequencing and output
    sorted array.
    """
    a = 1
    i = 0
    while i < (len(array)):
        data = array[i]
        if data != 0:
            if count[data] == 0:
                count[data] = a
                a += 1
        array[i] = count[data]
        i += 1


@njit(parallel=True)
def amax(array):  # pragma: no cover
    r"""
    Find largest element in an array using fast parallel numba technique

    Parameter:
    ----------
    array: ND-array
        array in which largest elements needs to be calcuted

    return:
    scalar: float or int
        The largest element value in the input array
    """

    return np.max(array)


def resequence_labels(array):
    r"""
    Resequence the lablels to make them contigious.

    Parameter:
    ----------
    array: ND-array
        Array that requires resequencing

    return:
    -------
    array : ND-array
        Resequenced array with same shape as input array
    """
    a_shape = array.shape
    array = array.ravel()
    max_num = amax(array) + 1
    count = np.zeros(max_num, dtype=np.uint32)
    _sequence(array, count)

    return array.reshape(a_shape)
