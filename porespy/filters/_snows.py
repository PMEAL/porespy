import dask.array as da
import inspect as insp
import numpy as np
from numba import njit, prange
from edt import edt
import scipy.ndimage as spim
import scipy.spatial as sptl
from skimage.segmentation import watershed
from skimage.morphology import ball, disk, square, cube
from porespy.tools import _check_for_singleton_axes
from porespy.tools import extend_slice, ps_rect, ps_round
from porespy.tools import Results
from porespy.tools import get_tqdm
from porespy.filters import chunked_func
from porespy import settings
from loguru import logger


tqdm = get_tqdm()


def snow_partitioning(im, dt=None, r_max=4, sigma=0.4, peaks=None):
    r"""
    Partition the void space into pore regions using a marker-based
    watershed algorithm

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
    peaks : ndarray, optional
        Optionally, it is possible to supply an array containing peaks, which
        are used as markers in the watershed segmentation. If a boolean array
        is received (``True`` indicating peaks), then ``scipy.ndimage.label``
        with cubic connectivity is used to label them. If an integer array is
        received then it is assumed the peaks have already been labelled.
        This allows for comparison of peak finding algorithms for instance.
        If this argument is provided, then ``r_max`` and ``sigma`` are ignored
        since these are specfically used in the peak finding process.

    Returns
    -------
    results : Results object
        A custom object with the following data as attributes:

        ============ ==========================================================
        Item         Description
        ============ ==========================================================
        ``im``       The binary image of the void space
        ``dt``       The distance transform of the image
        ``peaks``    The peaks of the distance transform after applying the
                     steps of the SNOW algorithm
        ``regions``  The void space partitioned into pores using a marker
                     based watershed with the peaks found by the SNOW algorithm
        ============ ==========================================================

    Notes
    -----
    The SNOW network extraction algorithm (Sub-Network of an
    Over-segmented Watershed) was designed to handle to perculiarities of
    high porosity materials, but it applies well to other materials as
    well.

    References
    ----------
    [1] Gostick, J. "A versatile and efficient network extraction
    algorithm using marker-based watershed segmenation".  Physical Review
    E. (2017)

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/snow_partitioning.html>`_
    to view online example.

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
            dt = dt.reshape(im_shape)
        else:
            dt = edt(im)

    if peaks is None:
        if sigma > 0:
            logger.trace(f"Applying Gaussian blur with sigma = {sigma}")
            dt_blurred = spim.gaussian_filter(input=dt, sigma=sigma)*im
        else:
            dt_blurred = np.copy(dt)
        peaks = find_peaks(dt=dt_blurred, r_max=r_max)

        logger.debug(f"Initial number of peaks: {spim.label(peaks)[1]}")
        peaks = trim_saddle_points(peaks=peaks, dt=dt)
        logger.debug(f"Peaks after trimming saddle points: {spim.label(peaks)[1]}")
        peaks = trim_nearby_peaks(peaks=peaks, dt=dt)
        logger.debug(f"Peaks after trimming nearby points: {spim.label(peaks)[1]}")
    peaks, N = spim.label(peaks > 0, structure=ps_rect(3, im.ndim))
    regions = watershed(image=-dt, markers=peaks)
    tup = Results()
    tup.im = im
    tup.dt = dt
    tup.peaks = peaks
    tup.regions = regions * (im > 0)
    return tup


def snow_partitioning_n(im, r_max=4, sigma=0.4, peaks=None):
    r"""
    This function partitions an imaging oontain an arbitrary number of
    phases into regions using a marker-based watershed segmentation.

    Parameters
    ----------
    im : ndarray
        Image of porous material where each phase is represented by unique
        integer starting from 1 (0's are ignored).
    r_max : scalar
        The radius of the spherical structuring element to use in the
        Maximum filter stage that is used to find peaks. The default is 4.
    sigma : scalar
        The standard deviation of the Gaussian filter used. The default is
        0.4. If 0 is given then the filter is not applied.
    peaks : ndarray, optional
        Optionally, it is possible to supply an array containing peaks, which
        are used as markers in the watershed segmentation. Must be a boolean
        array with ``True`` indicating peaks; ``scipy.ndimage.label``
        with cubic connectivity is used to label them. If this argument is
        provided then ``r_max`` and ``sigma`` are ignored, since these are
        specfically used in the peak finding process.

    Returns
    -------
    results : Results object
        A custom object with the following data as attributes:

        - 'im'
            The original image of the porous material

        - 'dt'
            The combined distance transform in alll phases of the image

        - 'phase_max_label'
            The list of max label of each phase in order to
            distinguish between each other

        - 'regions'
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

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/snow_partitioning_n.html>`_
    to view online example.

    """
    # Perform snow on each phase and merge all segmentation and dt together
    phases_num = np.unique(im).astype(int)
    phases_num = phases_num[phases_num > 0]
    combined_dt = 0
    combined_region = 0
    _peaks = np.zeros_like(im, dtype=int)
    num = [0]
    for i, j in enumerate(phases_num):
        logger.trace(f"Processing Phase {j}")
        # Isolate active phase from image
        phase = im == j
        # Limit peaks to active phase only
        temp = peaks*phase if peaks is not None else None
        phase_snow = snow_partitioning(phase, dt=None, r_max=r_max,
                                       sigma=sigma, peaks=temp)
        combined_dt += phase_snow.dt
        phase_snow.regions *= phase_snow.im
        phase_snow.regions += num[i]
        phase_ws = phase_snow.regions * phase_snow.im
        phase_ws[phase_ws == num[i]] = 0
        combined_region += phase_ws
        _peaks = _peaks + phase_snow.peaks + (phase_snow.peaks > 0)*num[i]
        num.append(np.amax(combined_region))

    tup = Results()
    tup.im = im
    tup.dt = combined_dt
    tup.phase_max_label = num[1:]
    tup.regions = combined_region
    tup.peaks = _peaks
    return tup


def find_peaks(dt, r_max=4, strel=None, sigma=None, divs=1):
    r"""
    Finds local maxima in the distance transform

    Parameters
    ----------
    dt : ndarray
        The distance transform of the pore space.  This may be calculated
        and filtered using any means desired.
    r_max : scalar
        The radius of the spherical element used in the maximum filter.
        This controls the localness of any maxima. The default is 4 voxels.
    strel : ndarray
        Instead of supplying ``r_max``, this argument allows a custom
        structuring element allowing control over both size and shape.
    sigma : float or list of floats
        If given, then a gaussian filter is applied to the distance transform
        using this value for the kernel
        (i.e. ``scipy.ndimage.gaussian_filter(dt, sigma)``)
    divs : int or array_like
        The number of times to divide the image for parallel processing.
        If ``1`` then parallel processing does not occur.  ``2`` is
        equivalent to ``[2, 2, 2]`` for a 3D image. The number of cores
        used is specified in ``porespy.settings.ncores`` and defaults to
        all cores.

    Returns
    -------
    image : ndarray
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

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/find_peaks.html>`_
    to view online example.

    """
    im = dt > 0
    _check_for_singleton_axes(im)
    if strel is None:
        strel = ps_round(r=r_max, ndim=im.ndim)
    if sigma is not None:
        dt = spim.gaussian_filter(dt, sigma=sigma)
    parallel = False
    if isinstance(divs, int):
        divs = [divs]*len(im.shape)
    if np.any(np.array(divs) > 1):
        parallel = True
        logger.info(f'Performing {insp.currentframe().f_code.co_name} in parallel')
    if parallel:
        overlap = max(strel.shape)
        mx = chunked_func(func=spim.maximum_filter, overlap=overlap,
                          im_arg='input', input=dt + 2.0 * (~im),
                          footprint=strel,
                          cores=settings.ncores, divs=divs)
    else:
        # The "2 * (~im)" sets solid voxels to 2 so peaks are not found
        # at the void/solid interface
        mx = spim.maximum_filter(dt + 2.0 * (~im), footprint=strel)
    peaks = (dt == mx) * im
    return peaks


def reduce_peaks(peaks):
    r"""
    Any peaks that are broad or elongated are replaced with a single voxel
    that is located at the center of mass of the original voxels.

    Parameters
    ----------
    peaks : ndarray
        An image containing ``True`` values indicating peaks in the
        distance transform

    Returns
    -------
    image : ndarray
        An array with the same number of isolated peaks as the original
        image, but fewer total ``True`` voxels.

    Notes
    -----
    The center of mass of a group of voxels is used as the new single
    voxel, so if the group has an odd shape (like a horse shoe), the new
    voxel may *not* lie on top of the original set.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/reduce_peaks.html>`_
    to view online example.

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


def trim_saddle_points(peaks, dt, maxiter=20):
    r"""
    Removes peaks that were mistakenly identified because they lied on a
    saddle or ridge in the distance transform that was not actually a true
    local peak.

    Parameters
    ----------
    peaks : ndarray
        A boolean image containing ``True`` values to mark peaks in the
        distance transform (``dt``)
    dt : ndarray
        The distance transform of the pore space for which the peaks
        are sought.
    maxiter : int
        The number of iteration to use when finding saddle points.
        The default value is 20.

    Returns
    -------
    image : ndarray
        An image with fewer peaks than the input image

    References
    ----------
    [1] Gostick, J. "A versatile and efficient network extraction algorithm
    using marker-based watershed segmentation".  Physical Review E. (2017)

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/trim_saddle_points.html>`_
    to view online example.

    """
    new_peaks = np.zeros_like(peaks, dtype=bool)
    if dt.ndim == 2:
        from skimage.morphology import square as cube
    else:
        from skimage.morphology import cube
    labels, N = spim.label(peaks > 0)
    slices = spim.find_objects(labels)
    for i, s in tqdm(enumerate(slices), **settings.tqdm):
        sx = extend_slice(s, shape=peaks.shape, pad=maxiter)
        peaks_i = labels[sx] == i + 1
        dt_i = dt[sx]
        im_i = dt_i > 0
        iters = 0
        while iters < maxiter:
            iters += 1
            peaks_dil = spim.binary_dilation(input=peaks_i, structure=cube(3))
            peaks_max = peaks_dil * np.amax(dt_i * peaks_dil)
            peaks_extended = (peaks_max == dt_i) * im_i
            if np.all(peaks_extended == peaks_i):
                new_peaks[sx] += peaks_i
                break  # Found a true peak
            elif np.sum(peaks_extended * peaks_i) == 0:
                break  # Found a saddle point
            peaks_i = peaks_extended
        if iters >= maxiter:
            logger.debug(
                "Maximum number of iterations reached, consider "
                + "running again with a larger value of max_iters"
            )
    return new_peaks*peaks


def trim_saddle_points_legacy(peaks, dt, maxiter=10):
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
    maxiter : int
        The maximum number of iterations to run while eroding the saddle
        points.  The default is 10, which is usually not reached; however,
        a warning is issued if the loop ends prior to removing all saddle
        points.

    Returns
    -------
    image : ND-array
        An image with fewer peaks than the input image

    Notes
    -----
    This version of the function was included in versions of PoreSpy < 2. It
    is too aggressive in trimming peaks, so was rewritten for PoreSpy >= 2.
    This is included here for legacy reasons

    References
    ----------
    [1] Gostick, J. "A versatile and efficient network extraction algorithm
    using marker-based watershed segmenation".  Physical Review E. (2017)

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/trim_saddle_points_legacy.html>`_
    to view online example.
    """
    new_peaks = np.zeros_like(peaks, dtype=bool)
    if dt.ndim == 2:
        from skimage.morphology import square as cube
    else:
        from skimage.morphology import cube
    labels, N = spim.label(peaks > 0)
    slices = spim.find_objects(labels)
    for i, s in tqdm(enumerate(slices), **settings.tqdm):
        sx = extend_slice(s, shape=peaks.shape, pad=10)
        peaks_i = labels[sx] == i + 1
        dt_i = dt[sx]
        im_i = dt_i > 0
        iters = 0
        while iters < maxiter:
            iters += 1
            peaks_dil = spim.binary_dilation(input=peaks_i, structure=cube(3))
            peaks_max = peaks_dil * np.amax(dt_i * peaks_dil)
            peaks_extended = (peaks_max == dt_i) * im_i
            if np.all(peaks_extended == peaks_i):
                break  # Found a true peak
            elif np.sum(peaks_extended * peaks_i) == 0:
                peaks_i = False
                break  # Found a saddle point
            # The following line was also missing from the original.  It has
            # no effect on the result but without this the "maxiters" is
            # reached very often.
            peaks_i = peaks_extended
        # The following line is essentially a bug.  It should be:
        # peaks[s] += peaks_i. Without the += the peaks_i image overwrites
        # the entire slice s, which may include other peaks that are within
        # 10 voxels due to the padding of s with extend_slice.
        new_peaks[sx] = peaks_i
        if iters >= maxiter:
            logger.debug(
                "Maximum number of iterations reached, consider "
                + "running again with a larger value of max_iters"
            )
    return new_peaks*peaks


def trim_nearby_peaks(peaks, dt, f=1):
    r"""
    Removes peaks that are nearer to another peak than to solid

    Parameters
    ----------
    peaks : ndarray
        A image containing nonzeros values indicating peaks in the distance
        transform (``dt``).  If ``peaks`` is boolean, a boolean is returned;
        if ``peaks`` have already been labelled, then the original labels
        are returned, missing the trimmed peaks.
    dt : ndarray
        The distance transform of the pore space
    f : scalar
        Controls how close peaks must be before they are considered near
        to each other. Sets of peaks are tagged as too near if
        ``d_neighbor < f * d_solid``.

    Returns
    -------
    image : ndarray
        An array the same size and type as ``peaks`` containing a subset of
        the peaks in the original image.

    Notes
    -----
    Each pair of peaks is considered simultaneously, so for a triplet of nearby
    peaks, each pair is considered.  This ensures that only the single peak
    that is furthest from the solid is kept.  No iteration is required.

    References
    ----------
    [1] Gostick, J. "A versatile and efficient network extraction
    algorithm using marker-based watershed segmenation". Physical Review
    E. (2017)

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/trim_nearby_peaks.html>`_
    to view online example.

    """
    if dt.ndim == 2:
        from skimage.morphology import square as cube
    else:
        from skimage.morphology import cube

    labels, N = spim.label(peaks > 0, structure=cube(3))
    crds = spim.measurements.center_of_mass(peaks > 0, labels=labels,
                                            index=np.arange(1, N + 1))
    crds = np.vstack(crds).astype(int)  # Convert to numpy array of ints
    L = dt[tuple(crds.T)]  # Get distance to solid for each peak
    # Add tiny amount to joggle points to avoid equal distances to solid
    # arange was added instead of random values so the results are repeatable
    L = L + np.arange(len(L))*1e-6

    tree = sptl.KDTree(data=crds)
    # Find list of nearest peak to each peak
    temp = tree.query(x=crds, k=2)
    nearest_neighbor = temp[1][:, 1]
    dist_to_neighbor = temp[0][:, 1]
    del temp, tree  # Free-up memory
    hits = np.where(dist_to_neighbor <= f * L)[0]
    # Drop peak that is closer to the solid than it's neighbor
    drop_peaks = []
    for i in hits:
        if L[i] < L[nearest_neighbor[i]]:
            drop_peaks.append(i)
        else:
            drop_peaks.append(nearest_neighbor[i])
    drop_peaks = np.unique(drop_peaks)

    new_peaks = ~np.isin(labels, drop_peaks+1)*peaks
    return new_peaks


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
                               cores=None,
                               ):
    r"""
    Performs SNOW algorithm in parallel (or serial) to reduce time
    (or memory usage) by geomertirc domain decomposition of large images.

    Parameters
    ----------
    im : ndarray
        A binary image of porous media with 'True' values indicating
        phase of interest.
    overlap : float (optional)
        The amount of overlap to apply between chunks.  If not provided it
        will be estiamted using ``porespy.tools.estimate_overlap`` with
        ``mode='dt'``.
    divs : list or int
        Number of domains each axis will be divided. Options are:
          - scalar: it will be assigned to all axis.
          - list: each respective axis will be divided by its
            corresponding number in the list. For example [2, 3, 4] will
            divide z, y and x axis to 2, 3, and 4 respectively.
    cores : int or None
        Number of cores that will be used to parallel process all domains.
        If ``None`` then all cores will be used but user can specify any
        integer values to control the memory usage.  Setting value to 1
        will effectively process the chunks in serial to minimize memory
        usage.

    Returns
    -------
    regions : ndarray
        Partitioned image of segmentated regions with unique labels. Each
        region correspond to pore body while intersection with other
        region correspond throat area.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/snow_partitioning_parallel.html>`_
    to view online example.

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
    regions = regions.map_blocks(_snow_chunked, r_max=r_max,
                                 sigma=sigma, dtype=dt.dtype)
    regions = da.overlap.trim_internal(regions, trim_depth, boundary='none')
    # TODO: use dask ProgressBar once compatible w/ logging.
    logger.trace('Applying snow to image chunks')
    regions = regions.compute(num_workers=cores)

    # Relabelling watershed chunks
    logger.trace('Relabelling watershed chunks')
    regions = relabel_chunks(im=regions, chunk_shape=chunk_shape)

    # Stitching watershed chunks
    logger.trace('Stitching watershed chunks')
    regions = _watershed_stitching(im=regions, chunk_shape=chunk_shape)
    tup = Results()
    tup.im = im
    tup.dt = dt
    tup.regions = regions
    return tup


def _pad(im, pad_width=1, constant_value=0):
    r"""
    Pad the image with a constant values and width.

    Parameters
    ----------
    im : ndarray
        The image that requires padding
    pad_width : int
        The number of values that will be padded from the edges. Default
        values is 1.
    contant_value : int
        Pads with the specified constant value

    Returns
    -------
    output: ndarray
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
    im: ndarray
        Actual image that contains repeating labels in chunks/sub-domains.
    chunk_shape: tuple
        The shape of chunk that will be relabeled in actual image. Note
        the chunk shape should be a multiple of actual image shape
        otherwise some labels will not be relabeled.

    Returns
    -------
    output : ndarray
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
    im :  ndarray
        image that contains extra slices in x, y, z direction.
    chunk_shape : tuple
        The shape of the chunk from which image is subdivided.

    Return:
    -------
    output : ndarray
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
    im : ndarray
        A worked image with watershed segmentation performed on all
        sub-domains individually.
    chunk_shape: tuple
        The shape of the sub-domain in which segmentation is performed.

    Returns
    -------
    output : ndarray
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
    array: ndarray
        Array that needs to be copied.

    Returns
    -------
    output: ndarray
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
    array : ndarray
        Array which requires replacing labels.
    keys :  array_like
        1d array containing unique labels that need to be replaced.
    values : array_like
        1d array containing unique values that will be assigned to labels.

    Returns
    -------
    array : ndarray
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
    array : ndarray
        Array which requires replacing labels
    keys :  1D-array
        The unique labels that need to be replaced
    values : 1D-array
        The unique values that will be assigned to labels

    return:
    -------
    array : ndarray
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
    array: ndarray
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
    array: ndarray
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
    array: ndarray
        Array that requires resequencing.

    Returns
    -------
    array : ndarray
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
    This private version of snow is called during snow_parallel.
    """
    dt2 = spim.gaussian_filter(input=dt, sigma=sigma)
    peaks = find_peaks(dt=dt2, r_max=r_max)
    peaks = trim_saddle_points(peaks=peaks, dt=dt)
    peaks = trim_nearby_peaks(peaks=peaks, dt=dt)
    peaks, N = spim.label(peaks > 0)
    regions = watershed(image=-dt, markers=peaks)
    return regions * (dt > 0)
