from collections import namedtuple

import dask.array as da
import numpy as np
import scipy.ndimage as spim
import scipy.spatial as sptl
from edt import edt
from loguru import logger
from numba import njit, prange
from skimage.morphology import ball, cube, disk, square
from skimage.segmentation import watershed

from porespy.filters import chunked_func
from porespy.tools import _check_for_singleton_axes, extend_slice


def snow_partitioning(im, dt=None, r_max=4, sigma=0.4):
    r"""
    Partition the void space into pore regions using a marker-based
    watershed algorithm, with specially filtered peaks as markers.

    Parameters
    ----------
    im : array_like
        A boolean image of the domain, with ``True`` indicating the pore
        space and ``False`` elsewhere.
    dt : array_like, optional
        The distance transform of the pore space.  This is done
        automatically if not provided, but if the distance transform has
        already been computed then supplying it can save some time.
    r_max : int
        The radius of the spherical structuring element to use in the
        Maximum filter stage that is used to find peaks. The default is 4.
    sigma : float
        The standard deviation of the Gaussian filter used in step 1. The
        default is 0.4.  If 0 is given then the filter is not applied,
        which is useful if a distance transform is supplied as the ``im``
        argument that has already been processed.

    Returns
    -------
    A **named tuple** containing all of the images used during the
    process.  They can be accessed as attriutes with the following names:

        ``im``
            The binary image of the void space
        ``dt``
            The distance transform of the image
        ``peaks``
            The peaks of the distance transform after applying the steps of the
            SNOW algorithm
        ``regions``
            The void space partitioned into pores using a marker
            based watershed with the peaks found by the SNOW algorithm

    Notes
    -----
    The SNOW network extraction algorithm (Sub-Network of an
    Over-segmented Watershed) was designed to handle to perculiarities of
    high porosity materials, but it applies well to other materials as
    well.

    References
    ----------
    [1] Gostick, J. "A versatile and efficient network extraction algorithm
    using marker-based watershed segmenation".  Physical Review E. (2017)

    """
    logger.trace("Beginning SNOW algorithm")
    im_shape = np.array(im.shape)
    if im.dtype is not bool:
        logger.trace("Converting supplied image to boolean")
        im = im > 0
    if dt is None:
        logger.trace("Peforming distance transform")
        if np.any(im_shape == 1):
            dt = edt(im.squeeze())
            dt = dt.reshape(im.shape)
        else:
            dt = edt(im)

    if sigma > 0:
        logger.trace(f"Applying Gaussian blur with sigma = {sigma}")
        dt = spim.gaussian_filter(input=dt, sigma=sigma)

    peaks = find_peaks(dt=dt, r_max=r_max)
    logger.debug(f"Initial number of peaks: {spim.label(peaks)[1]}")
    peaks = trim_saddle_points(peaks=peaks, dt=dt, max_iters=500)
    logger.debug(f"Peaks after trimming saddle points: {spim.label(peaks)[1]}")
    peaks = trim_nearby_peaks(peaks=peaks, dt=dt)
    peaks, N = spim.label(peaks)
    logger.debug(f"Peaks after trimming nearby peaks: {N}")
    regions = watershed(image=-dt, markers=peaks, mask=im > 0)
    # Catch any isolated regions that were missed
    # TODO: I'm not sure if this approach is universal so I'm going to comment it
    # out for now, and mark it as a todo
    # labels = spim.label((regions == 0)*(im > 0))[0]
    # regions += (labels + regions.max())*(labels > 0)
    tup = namedtuple("results", field_names=["im", "dt", "peaks", "regions"])
    tup.im = im
    tup.dt = dt
    tup.peaks = peaks
    tup.regions = regions
    return tup


def snow_partitioning_n(im, r_max=4, sigma=0.4):
    r"""
    This function partitions an imaging oontain an arbitrary number of
    phases into regions using a marker-based watershed segmentation. Its
    an extension of snow_partitioning function with all phases partitioned
    together.

    Parameters
    ----------
    im : ND-array
        Image of porous material where each phase is represented by unique
        integer starting from 1 (0's are ignored).
    r_max : scalar
        The radius of the spherical structuring element to use in the
        Maximum filter stage that is used to find peaks. The default is 4.
    sigma : scalar
        The standard deviation of the Gaussian filter used. The default is
        0.4. If 0 is given then the filter is not applied.

    Returns
    -------
    A **named tuple** with the following attribute:

        ``im``
            The actual image of the porous material
        ``dt``
            The combined distance transform of the image
        ``phase_max_label``
            The list of max label of each phase in order to
            distinguish between each other
        ``regions``
            The partitioned regions of n phases using a marker
            based watershed with the peaks found by the SNOW algorithm

    References
    ----------
    [1] Gostick, J. "A versatile and efficient network extraction
    algorithm using marker-based watershed segmentation".  Physical Review
    E. (2017)

    [2] Khan, ZA et al. "Dual network extraction algorithm to investigate
    multiple transport processes in porous materials: Image-based modeling
    of pore and grain-scale processes". Computers in Chemical Engineering.
    (2019)

    See Also
    --------
    snow_partitioning

    Notes
    -----
    In principle it is possible to perform a distance transform on each
    phase separately, merge these into a single image, then apply the
    watershed only once. This, however, has been found to create edge
    artifacts between regions arising from the way watershed handles
    plateaus in the distance transform. To overcome this, this function
    applies the watershed to each of the distance transforms separately,
    then merges the segmented regions back into a single image.

    """
    # Perform snow on each phase and merge all segmentation and dt together
    phases_num = np.unique(im * 1)
    phases_num = np.trim_zeros(phases_num)
    combined_dt = 0
    combined_region = 0
    num = [0]
    for i, j in enumerate(phases_num):
        logger.trace(f"Processing Phase {j}")
        phase_snow = snow_partitioning(im == j, dt=None, r_max=r_max, sigma=sigma)
        combined_dt += phase_snow.dt
        phase_snow.regions *= phase_snow.im
        phase_snow.regions += num[i]
        phase_ws = phase_snow.regions * phase_snow.im
        phase_ws[phase_ws == num[i]] = 0
        combined_region += phase_ws
        num.append(np.amax(combined_region))

    tup = namedtuple("results",
                     field_names=["im", "dt", "phase_max_label", "regions"])
    tup.im = im
    tup.dt = combined_dt
    tup.phase_max_label = num[1:]
    tup.regions = combined_region
    return tup


def find_peaks(dt, r_max=4, strel=None, **kwargs):
    r"""
    Finds local maxima in the distance transform

    Parameters
    ----------
    dt : ND-array
        The distance transform of the pore space.  This may be calculated
        and filtered using any means desired.
    r_max : scalar
        The size of the structuring element used in the maximum filter.
        This controls the localness of any maxima. The default is 4 voxels.
    strel : ND-array
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
    which is significantly faster than using a circular or spherical
    element.

    """
    im = dt > 0
    _check_for_singleton_axes(im)

    if strel is None:
        if im.ndim == 2:
            strel = disk
        elif im.ndim == 3:
            strel = ball
        else:  # pragma: no cover
            raise Exception("Only 2d and 3d images are supported")
    parallel = kwargs.pop('parallel', False)
    cores = kwargs.pop('cores', None)
    divs = kwargs.pop('cores', 2)
    if parallel:
        overlap = max(strel(r_max).shape)
        peaks = chunked_func(func=find_peaks, overlap=overlap,
                             im_arg='dt', dt=dt, footprint=strel,
                             cores=cores, divs=divs)
    else:
        mx = spim.maximum_filter(dt + 2 * (~im), footprint=strel(r_max))
        peaks = (dt == mx) * im
    return peaks


def reduce_peaks(peaks):
    r"""
    Any peaks that are broad or elongated are replaced with a single voxel
    that is located at the center of mass of the original voxels.

    Parameters
    ----------
    peaks : ND-image
        An image containing ``True`` values indicating peaks in the
        distance transform

    Returns
    -------
    image : ND-array
        An array with the same number of isolated peaks as the original
        image, but fewer total ``True`` voxels.

    Notes
    -----
    The center of mass of a group of voxels is used as the new single
    voxel, so if the group has an odd shape (like a horse shoe), the new
    voxel may *not* lie on top of the original set.

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


def trim_saddle_points(peaks, dt, max_iters=10):
    r"""
    Removes peaks that were mistakenly identified because they lied on a
    saddle or ridge in the distance transform that was not actually a true
    local peak.

    Parameters
    ----------
    peaks : ND-array
        A boolean image containing True values to mark peaks in the
        distance transform (``dt``)
    dt : ND-array
        The distance transform of the pore space for which the true peaks
        are sought.
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
        s = extend_slice(slices[i], shape=peaks.shape, pad=10)
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
        if iters >= max_iters:  # pragma: no cover
            logger.warning("Maximum number of iterations reached, consider"
                           " running again with a larger value of max_iters")
    return peaks


def trim_nearby_peaks(peaks, dt, f=1.0):
    r"""
    Finds pairs of peaks that are nearer to each other than to the solid
    phase, and removes the peak that is closer to the solid.

    Parameters
    ----------
    peaks : ND-array
        A boolean image containing True values to mark peaks in the
        distance transform (``dt``)
    dt : ND-array
        The distance transform of the pore space for which the true peaks
        are sought.
    f : scalar
        Controls how close peaks must be before they are considered near to each
        other.  Sets of peaks are tagged as near if ``d_neighbor < f * d_solid``.

    Returns
    -------
    image : ND-array
        An array the same size as ``peaks`` containing a subset of the
        peaks in the original image.

    Notes
    -----
    Each pair of peaks is considered simultaneously, so for a triplet of
    peaks each pair is considered.  This ensures that only the single peak
    that is furthest from the solid is kept.  No iteration is required.

    References
    ----------
    [1] Gostick, J. "A versatile and efficient network extraction
    algorithm using marker-based watershed segmenation".  Physical Review
    E. (2017)

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
    hits = np.where(dist_to_neighbor < f * dist_to_solid)[0]
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


def _estimate_overlap(im, mode='dt', zoom=0.25):
    logger.trace('Calculating overlap thickness')
    if mode == 'watershed':
        rev = spim.interpolation.zoom(im, zoom=zoom, order=0)
        rev = rev > 0
        dt = edt(rev, parallel=0)
        rev_snow = snow_partitioning(rev, dt=dt)
        labels, counts = np.unique(rev_snow, return_counts=True)
        node = np.where(counts == counts[1:].max())[0][0]
        slices = spim.find_objects(rev_snow)
        overlap = max(rev_snow[slices[node - 1]].shape) / (zoom * 2.0)
    if mode == 'dt':
        dt = edt((im > 0), parallel=0)
        overlap = dt.max()
    return overlap

def snow_partitioning_parallel(im,
                               r_max=4,
                               sigma=0.4,
                               divs=2,
                               overlap=None,
                               num_workers=None):
    r"""
    Performs SNOW algorithm in parallel (or serial) to reduce time
    (or memory usage) by geomertirc domain decomposition of large images.

    Parameters
    ----------
    im : ND-array
        A binary image of porous media with 'True' values indicating
        phase of interest.
    overlap : float (optional)
        The amount of overlap to apply between chunks.  If not provided it will
        be estiamted using ``porespy.tools.estimate_overlap`` with ``mode='dt'``.
    divs : list or int
        Number of domains each axis will be divided. Options are:
          - scalar: it will be assigned to all axis.
          - list: each respective axis will be divided by its corresponding
            number in the list. For example [2, 3, 4] will divide z, y and
            x axis to 2, 3, and 4 respectively.
    num_workers : int or None
        Number of cores that will be used to parallel process all domains.
        If ``None`` then all cores will be used but user can specify any
        integer values to control the memory usage.  Setting value to 1 will
        effectively process the chunks in serial to minimize memory usage.

    Returns
    -------
    regions : ND-array
        Partitioned image of segmentated regions with unique labels. Each
        region correspond to pore body while intersection with other
        region correspond throat area.

    """
    # Adjust image shape according to specified dimension
    if isinstance(divs, int):
        divs = [divs for i in range(im.ndim)]
    shape = []
    for i in range(im.ndim):
        shape.append(divs[i] * (im.shape[i] // divs[i]))

    if tuple(shape) != im.shape:
        for i in range(im.ndim):
            im = im.swapaxes(0, i)
            im = im[:shape[i], ...]
            im = im.swapaxes(i, 0)
        logger.debug(f'Image was cropped to shape {shape}')

    # Get overlap thickness from distance transform
    chunk_shape = (np.array(shape) / np.array(divs)).astype(int)
    logger.trace('Beginning parallel SNOW algorithm...')

    if overlap is None:
        overlap = _estimate_overlap(im, mode='dt')
    overlap = overlap / 2.0
    logger.debug(f'Overlap thickness: {int(2 * overlap)} voxels')

    dt = edt((im > 0), parallel=0)

    # Get overlap and trim depth of all image dimension
    depth = {}
    trim_depth = {}
    for i in range(im.ndim):
        depth[i] = int(2.0 * overlap)
        trim_depth[i] = int(2.0 * overlap) - 1

    # Applying SNOW to image chunks
    regions = da.from_array(dt, chunks=chunk_shape)
    regions = da.overlap.overlap(regions, depth=depth, boundary='none')
    regions = regions.map_blocks(_snow_chunked, r_max=r_max, sigma=sigma)
    regions = da.overlap.trim_internal(regions, trim_depth, boundary='none')
    # TODO: use dask ProgressBar once compatible w/ logging.
    logger.trace('Applying snow to image chunks')
    regions = regions.compute(num_workers=num_workers)

    # Relabelling watershed chunks
    logger.trace('Relabelling watershed chunks')
    regions = relabel_chunks(im=regions, chunk_shape=chunk_shape)

    # Stitching watershed chunks
    logger.trace('Stitching watershed chunks')
    regions = _watershed_stitching(im=regions, chunk_shape=chunk_shape)
    tup = namedtuple("results", field_names=["im", "dt", "regions"])
    tup.im = im
    tup.dt = dt
    tup.regions = regions

    return tup

def _pad(im, pad_width=1, constant_value=0):
    r"""
    Pad the image with a constant values and width.

    Parameters
    ----------
    im : ND-array
        The image that requires padding
    pad_width : int
        The number of values that will be padded from the edges. Default
        values is 1.
    contant_value : int
        Pads with the specified constant value

    Returns
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


def relabel_chunks(im, chunk_shape):
    r"""
    Assign new labels to each chunk or sub-domain of actual image. This
    prevents from two or more regions to have same label.

    Parameters
    ----------
    im: ND-array
        Actual image that contains repeating labels in chunks/sub-domains.

    chunk_shape: tuple
        The shape of chunk that will be relabeled in actual image. Note
        the chunk shape should be a multiple of actual image shape
        otherwise some labels will not be relabeled.

    Returns
    -------
    output : ND-array
        Relabeled image with unique label assigned to each region.

    """
    im = _pad(im, pad_width=1)
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


def _trim_internal_slice(im, chunk_shape):
    r"""
    Delete extra slices from image that were used to stitch two or more
    chunks together.

    Parameters:
    -----------
    im :  ND-array
        image that contains extra slices in x, y, z direction.

    chunk_shape : tuple
        The shape of the chunk from which image is subdivided.

    Return:
    -------
    output : ND-array
        Image without extra internal slices. The shape of the image will
        be same as input image provided for waterhsed segmentation.

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


def _watershed_stitching(im, chunk_shape):
    r"""
    Stitch individual sub-domains of watershed segmentation into one big
    segmentation with all boundary labels of each sub-domain relabeled to
    merge boundary regions.

    Parameters:
    -----------
    im : ND-array
        A worked image with watershed segmentation performed on all
        sub-domains individually.

    chunk_shape: tuple
        The shape of the sub-domain in which segmentation is performed.

    Returns
    -------
    output : ND-array
        Stitched watershed segmentation with all sub-domains merged to
        form a single watershed segmentation.

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
            im = _replace_labels(array=im, keys=keys, values=values)
            im = im.swapaxes(axis, 0)
    im = _trim_internal_slice(im=im, chunk_shape=chunk_shape)
    im = _resequence_labels(array=im)

    return im


@njit(parallel=True)
def _copy(im, output):
    r"""
    The function copy the input array and make output array that is
    allocated in different memory space. This a numba version of copy
    function of numpy. Because each element is copied using parallel
    approach this implementation is faster than numpy version of copy.

    Parameters
    ----------
    array: ND-array
        Array that needs to be copied.

    Returns
    -------
    output: ND-array
        Copied array.

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
def _replace(array, keys, values, ind_sort):
    r"""
    This function replace keys elements in input array with new value
    elements. This function is used as internal function of
    replace_relabels.

    Parameters
    ----------
    array : ND-array
        Array which requires replacing labels.
    keys :  array_like
        1d array containing unique labels that need to be replaced.
    values : array_like
        1d array containing unique values that will be assigned to labels.

    Returns
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


def _replace_labels(array, keys, values):
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
def _sequence(array, count):
    r"""
    Internal function of resequnce_labels method. This function resquence
    array elements in an ascending order using numba technique which is
    many folds faster than make contigious funcition.

    Parameters
    ----------
    array: ND-array
        1d array that needs resquencing.
    count: array_like
        1d array of zeros having same size as array.

    Returns
    -------
    array: 1d-array
        The input array with elements resequenced in ascending order

    Notes
    -----
    The output of this function is not same as make_contigous or
    relabel_sequential function of scikit-image. This function resequence
    and randomize the regions while other methods only do resequencing and
    output sorted array.

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
def _amax(array):
    r"""
    Find largest element in an array using fast parallel numba technique.

    Parameter:
    ----------
    array: ND-array
        Array in which largest elements needs to be calcuted.

    Returns
    -------
    scalar: float or int
        The largest element value in the input array.

    """
    return np.max(array)


def _resequence_labels(array):
    r"""
    Resequence the lablels to make them contigious.

    Parameters
    ----------
    array: ND-array
        Array that requires resequencing.

    Returns
    -------
    array : ND-array
        Resequenced array with same shape as input array.

    """
    a_shape = array.shape
    array = array.ravel()
    max_num = _amax(array) + 1
    count = np.zeros(max_num, dtype=np.uint32)
    _sequence(array, count)

    return array.reshape(a_shape)

def _snow_chunked(dt, r_max=5, sigma=0.4):
    r"""
    This private version of snow is called during snow_parallel.  Dask does not
    all the calls to the logger between each step apparently.
    """
    dt2 = spim.gaussian_filter(input=dt, sigma=sigma)
    peaks = find_peaks(dt=dt2, r_max=r_max)
    peaks = trim_saddle_points(peaks=peaks, dt=dt2, max_iters=99)
    peaks = trim_nearby_peaks(peaks=peaks, dt=dt2)
    peaks, N = spim.label(peaks)
    regions = watershed(image=-dt2, markers=peaks)
    return regions * (dt > 0)
