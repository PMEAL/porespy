from collections import namedtuple
import scipy as sp
import scipy.ndimage as spim
import scipy.spatial as sptl
from scipy.signal import fftconvolve
from tqdm import tqdm
from numba import jit
from skimage.segmentation import clear_border
from skimage.morphology import ball, disk, square, cube
from skimage.morphology import reconstruction, watershed
from porespy.tools import randomize_colors
from porespy.tools import get_border, extend_slice
from porespy.tools import fftmorphology


def snow_partitioning(im, r_max=4, sigma=0.4, return_all=False):
    r"""
    This function partitions the void space into pore regions using a
    marker-based watershed algorithm.  The key to this function is that true
    local maximum of the distance transform are found by trimming various
    types of extraneous peaks.

    The SNOW network extraction algorithm (Sub-Network of an Over-segmented
    Watershed) was designed to handle to perculiarities of high porosity
    materials, but it applies well to other materials as well.

    Parameters
    ----------
    im : array_like
        Can be either (a) a boolean image of the domain, with ``True``
        indicating the pore space and ``False`` elsewhere, or (b) a distance
        transform of the domain calculated externally by the user.  Option (b)
        is faster if a distance transform is already available.

    r_max : scalar
        The radius of there spherical structuring element to use in the Maximum
        filter stage that is used to find peaks.  The default is 4

    sigma : scalar
        The standard deviation of the Gaussian filter used in step 1.  The
        default is 0.4.  If 0 is given then the filter is not applied, which is
        useful if a distance transform is supplied as the ``im`` argument that
        has already been processed.

    return_all : boolean (default is False)
        If set to ``True`` a named tuple is returned containing the original
        image, the distance transform, the filtered peaks, and the final
        pore regions.

    Returns
    -------
    An image the same shape as ``im`` with the void space partitioned into
    pores using a marker based watershed with the peaks found by the
    SNOW algorithm [1].  If ``return_all`` is ``True`` then a **named tuple**
    is returned with the following attribute:

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
    tup = namedtuple('results', field_names=['im', 'dt', 'peaks', 'regions'])
    im = im.squeeze()
    print('_'*60)
    print("Beginning SNOW Algorithm")

    if im.dtype == 'bool':
        print('Peforming Distance Transform')
        dt = spim.distance_transform_edt(input=im)
    else:
        dt = im
        im = dt > 0

    tup.im = im
    tup.dt = dt

    if sigma > 0:
        print('Applying Gaussian blur with sigma =', str(sigma))
        dt = spim.gaussian_filter(input=dt, sigma=sigma)

    peaks = find_peaks(dt=dt)
    print('Initial number of peaks: ', spim.label(peaks)[1])
    peaks = trim_saddle_points(peaks=peaks, dt=dt, max_iters=500)
    print('Peaks after trimming saddle points: ', spim.label(peaks)[1])
    peaks = trim_nearby_peaks(peaks=peaks, dt=dt)
    peaks, N = spim.label(peaks)
    print('Peaks after trimming nearby peaks: ', N)
    tup.peaks = peaks
    regions = watershed(image=-dt, markers=peaks, mask=dt > 0)
    regions = randomize_colors(regions)
    if return_all:
        tup.regions = regions
        return tup
    else:
        return regions


def find_peaks(dt, r=4, footprint=None):
    r"""
    Returns all local maxima in the distance transform

    Parameters
    ----------
    dt : ND-array
        The distance transform of the pore space.  This may be calculated and
        filtered using any means desired.

    r : scalar
        The size of the structuring element used in the maximum filter.  This
        controls the localness of any maxima. The default is 4 voxels.

    footprint : ND-array
        Specifies the shape of the structuring element used to define the
        neighborhood when looking for peaks.  If none is specified then a
        spherical shape is used (or circular in 2D).

    Returns
    -------
    An ND-array of booleans with ``True`` values at the location of any local
    maxima.

    Notes
    -----
    It is also possible ot the ``peak_local_max`` function from the
    ``skimage.feature`` module as follows:

    ``peaks = peak_local_max(image=dt, min_distance=r, exclude_border=0,
    indices=False)``

    This automatically uses a square structuring element which is significantly
    faster than using a circular or spherical element.
    """
    dt = dt.squeeze()
    im = dt > 0
    if footprint is None:
        if im.ndim == 2:
            footprint = disk
        elif im.ndim == 3:
            footprint = ball
        else:
            raise Exception("only 2-d and 3-d images are supported")
    mx = spim.maximum_filter(dt + 2*(~im), footprint=footprint(r))
    peaks = (dt == mx)*im
    return peaks


def reduce_peaks(peaks):
    r"""
    Any peaks that are broad or elongated are replaced with a single voxel
    that is located at the center of mass of the original voxels.

    Parameters
    ----------
    peaks : ND-image
        An image containing True values indicating peaks in the distance
        transform

    Returns
    -------
    An array with the same number of isolated peaks as the original image, but
    fewer total voxels.

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
    inds = spim.measurements.center_of_mass(input=peaks,
                                            labels=markers,
                                            index=sp.arange(1, N))
    inds = sp.floor(inds).astype(int)
    # Centroid may not be on old pixel, so create a new peaks image
    peaks = sp.zeros_like(peaks, dtype=bool)
    peaks[tuple(inds.T)] = True
    return peaks


def trim_saddle_points(peaks, dt, max_iters=10):
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
    An image with fewer peaks than was received.
    """
    if dt.ndim == 2:
        from skimage.morphology import square as cube
    else:
        from skimage.morphology import cube
    labels, N = spim.label(peaks)
    slices = spim.find_objects(labels)
    for i in range(N):
        s = extend_slice(s=slices[i], shape=peaks.shape, pad=10)
        peaks_i = labels[s] == i+1
        dt_i = dt[s]
        im_i = dt_i > 0
        iters = 0
        peaks_dil = sp.copy(peaks_i)
        while iters < max_iters:
            iters += 1
            peaks_dil = spim.binary_dilation(input=peaks_dil,
                                             structure=cube(3))
            peaks_max = peaks_dil*sp.amax(dt_i*peaks_dil)
            peaks_extended = (peaks_max == dt_i)*im_i
            if sp.all(peaks_extended == peaks_i):
                break  # Found a true peak
            elif sp.sum(peaks_extended*peaks_i) == 0:
                peaks_i = False
                break  # Found a saddle point
        peaks[s] = peaks_i
        if iters >= max_iters:
            print('Maximum number of iterations reached, consider' +
                  'running again with a larger value of max_iters')
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
    An array the same size as ``peaks`` containing a subset of the peaks in
    the original image.

    Notes
    -----
    Each pair of peaks is considered simultaneously, so for a triplet of peaks
    each pair is considered.  This ensures that only the single peak that is
    furthest from the solid is kept.  No iteration is required.
    """
    if dt.ndim == 2:
        from skimage.morphology import square as cube
    else:
        from skimage.morphology import cube
    peaks, N = spim.label(peaks, structure=cube(3))
    crds = spim.measurements.center_of_mass(peaks, labels=peaks,
                                            index=sp.arange(1, N+1))
    crds = sp.vstack(crds).astype(int)  # Convert to numpy array of ints
    # Get distance between each peak as a distance map
    tree = sptl.cKDTree(data=crds)
    temp = tree.query(x=crds, k=2)
    nearest_neighbor = temp[1][:, 1]
    dist_to_neighbor = temp[0][:, 1]
    del temp, tree  # Free-up memory
    dist_to_solid = dt[list(crds.T)]  # Get distance to solid for each peak
    hits = sp.where(dist_to_neighbor < dist_to_solid)[0]
    # Drop peak that is closer to the solid than it's neighbor
    drop_peaks = []
    for peak in hits:
        if dist_to_solid[peak] < dist_to_solid[nearest_neighbor[peak]]:
            drop_peaks.append(peak)
        else:
            drop_peaks.append(nearest_neighbor[peak])
    drop_peaks = sp.unique(drop_peaks)
    # Remove peaks from image
    slices = spim.find_objects(input=peaks)
    for s in drop_peaks:
        peaks[slices[s]] = 0
    return (peaks > 0)


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
        neighbors.  The default is max

    Returns
    -------
    An ND-image the same size as ``im``, with True values indicating voxels of
    the phase of interest (i.e. True values in the original image) that are
    not connected to the outer edges.

    Notes
    -----
    The returned array (e.g. ``holes``) be used to trim blind pores from
    ``im`` using: ``im[holes] = False``

    """
    if im.ndim == 2:
        if conn == 4:
            strel = disk(1)
        elif conn in [None, 8]:
            strel = square(3)
    elif im.ndim == 3:
        if conn == 6:
            strel = ball(1)
        elif conn in [None, 26]:
            strel = cube(3)
    labels, N = spim.label(input=im, structure=strel)
    holes = clear_border(labels=labels) > 0
    return holes


def fill_blind_pores(im):
    r"""
    Fills all pores that are not connected to the edges of the image.

    Parameters
    ----------
    im : ND-array
        The image of the porous material

    Returns
    -------
        A version of ``im`` but with all the disconnected pores removed.

    See Also
    --------
    find_disconnected_voxels

    """
    holes = find_disconnected_voxels(im)
    im[holes] = False
    return im


def trim_floating_solid(im):
    r"""
    Removes all solid that that is not attached to the edges of the image.

    Parameters
    ----------
    im : ND-array
        The image of the porous material

    Returns
    -------
        A version of ``im`` but with all the disconnected solid removed.

    See Also
    --------
    find_disconnected_voxels

    """
    holes = find_disconnected_voxels(~im)
    im[holes] = True
    return im


def trim_nonpercolating_paths(im, inlet_axis=0, outlet_axis=0):
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
        between 0 to 1.

    outlet_axis : int
        Outlet axis of boundary condition. For three dimensional image the
        number ranges from 0 to 2. For two dimensional image the range is
        between 0 to 1.

    Returns
    -------
    A copy of ``im`` but with all the nonpercolating paths removed

    See Also
    --------
    find_disconnected_voxels
    trim_floating_solid
    trim_blind_pores

    """
    im = trim_floating_solid(~im)
    labels = spim.label(~im)[0]
    inlet = sp.zeros_like(im, dtype=int)
    outlet = sp.zeros_like(im, dtype=int)
    if im.ndim == 3:
        if inlet_axis == 0:
            inlet[0, :, :] = 1
        elif inlet_axis == 1:
            inlet[:, 0, :] = 1
        elif inlet_axis == 2:
            inlet[:, :, 0] = 1

        if outlet_axis == 0:
            outlet[-1, :, :] = 1
        elif outlet_axis == 1:
            outlet[:, -1, :] = 1
        elif outlet_axis == 2:
            outlet[:, :, -1] = 1

    if im.ndim == 2:
        if inlet_axis == 0:
            inlet[0, :] = 1
        elif inlet_axis == 1:
            inlet[:, 0] = 1

        if outlet_axis == 0:
            outlet[-1, :] = 1
        elif outlet_axis == 1:
            outlet[:, -1] = 1
    IN = sp.unique(labels*inlet)
    OUT = sp.unique(labels*outlet)
    new_im = sp.isin(labels, list(set(IN) ^ set(OUT)), invert=True)
    im[new_im == 0] = True
    return ~im


def trim_extrema(im, h, mode='maxima'):
    r"""
    This trims local extrema in greyscale values by a specified amount,
    essentially decapitating peaks or flooding valleys, or both.

    Parameters
    ----------
    im : ND-array
        The image whose extrema are to be removed

    h : scalar
        The height to remove from each peak or fill in each valley

    mode : string {'maxima' | 'minima' | 'extrema'}
        Specifies whether to remove maxima or minima or both

    Returns
    -------
    A copy of the input image with all the peaks and/or valleys removed.

    Notes
    -----
    This function is referred to as **imhmax** or **imhmin** in Matlab.

    """
    result = im
    if mode in ['maxima', 'extrema']:
        result = reconstruction(seed=im - h, mask=im, method='dilation')
    elif mode in ['minima', 'extrema']:
        result = reconstruction(seed=im + h, mask=im, method='erosion')
    return result


@jit(forceobj=True)
def flood(im, regions=None, mode='max'):
    r"""
    Floods/fills each region in an image with a single value based on the
    specific values in that region.  The ``mode`` argument is used to
    determine how the value is calculated.

    Parameters
    ----------
    im : array_like
        An ND image with isolated regions containing 0's elsewhere.

    regions : array_like
        An array the same shape as ``im`` with each region labeled.  If None is
        supplied (default) then ``scipy.ndimage.label`` is used with its
        default arguments.

    mode : string
        Specifies how to determine which value should be used to flood each
        region.  Options are:

    *'max'* : Floods each region with the local maximum in that region

    *'min'* : Floods each region the local minimum in that region

    *'size'* : Floods each region with the size of that region

    Returns
    -------
    An ND-array the same size as ``im`` with new values placed in each
    forground voxel based on the ``mode``.

    See Also
    --------
    props_to_image

    """
    mask = im > 0
    if regions is None:
        labels, N = spim.label(mask)
    else:
        labels = sp.copy(regions)
        N = labels.max()
    I = im.flatten()
    L = labels.flatten()
    if mode.startswith('max'):
        V = sp.zeros(shape=N+1, dtype=float)
        for i in range(len(L)):
            if V[L[i]] < I[i]:
                V[L[i]] = I[i]
    elif mode.startswith('min'):
        V = sp.ones(shape=N+1, dtype=float)*sp.inf
        for i in range(len(L)):
            if V[L[i]] > I[i]:
                V[L[i]] = I[i]
    elif mode.startswith('size'):
        V = sp.zeros(shape=N+1, dtype=int)
        for i in range(len(L)):
            V[L[i]] += 1
    im_flooded = sp.reshape(V[labels], newshape=im.shape)
    im_flooded = im_flooded*mask
    return im_flooded


def apply_chords(im, spacing=0, axis=0, trim_edges=True):
    r"""
    Adds chords to the void space in the specified direction.  The chords are
    separated by 1 voxel plus the provided spacing.

    Parameters
    ----------
    im : ND-array
        An image of the porous material with void marked as True.

    spacing : int (default = 0)
        Chords are automatically separated by 1 voxel and this argument
        increases the separation.

    axis : int (default = 0)
        The axis along which the chords are drawn.

    trim_edges : bool (default = True)
        Whether or not to remove chords that touch the edges of the image.
        These chords are artifically shortened, so skew the chord length
        distribution

    Returns
    -------
    An ND-array of the same size as ```im``` with True values indicating the
    chords.

    See Also
    --------
    apply_chords_3D

    """
    if spacing < 0:
        raise Exception('Spacing cannot be less than 0')
    dims1 = sp.arange(0, im.ndim)
    dims2 = sp.copy(dims1)
    dims2[axis] = 0
    dims2[0] = axis
    im = sp.moveaxis(a=im, source=dims1, destination=dims2)
    im = sp.atleast_3d(im)
    ch = sp.zeros_like(im, dtype=bool)
    if im.ndim == 2:
        ch[:, ::2+spacing, ::2+spacing] = 1
    if im.ndim == 3:
        ch[:, ::4+2*spacing, ::4+2*spacing] = 1
    chords = im*ch
    chords = sp.squeeze(chords)
    if trim_edges:
        temp = clear_border(spim.label(chords == 1)[0]) > 0
        chords = temp*chords
    chords = sp.moveaxis(a=chords, source=dims1, destination=dims2)
    return chords


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

    trim_edges : bool (default = True)
        Whether or not to remove chords that touch the edges of the image.
        These chords are artifically shortened, so skew the chord length
        distribution

    Returns
    -------
    An ND-array of the same size as ```im``` with values of 1 indicating
    x-direction chords, 2 indicating y-direction chords, and 3 indicating
    z-direction chords.

    Notes
    -----
    The chords are separated by a spacing of at least 1 voxel so that tools
    that search for connected components, such as ``scipy.ndimage.label`` can
    detect individual chords.

    See Also
    --------
    apply_chords

    """
    if im.ndim < 3:
        raise Exception('Must be a 3D image to use this function')
    if spacing < 0:
        raise Exception('Spacing cannot be less than 0')
    ch = sp.zeros_like(im, dtype=int)
    ch[:, ::4+2*spacing, ::4+2*spacing] = 1  # X-direction
    ch[::4+2*spacing, :, 2::4+2*spacing] = 2  # Y-direction
    ch[2::4+2*spacing, 2::4+2*spacing, :] = 3  # Z-direction
    chords = ch*im
    if trim_edges:
        temp = clear_border(spim.label(chords > 0)[0]) > 0
        chords = temp*chords
    return chords


def local_thickness(im, sizes=25):
    r"""
    For each voxel, this functions calculates the radius of the largest sphere
    that both engulfs the voxel and fits entirely within the foreground. This
    is not the same as a simple distance transform, which finds the largest
    sphere that could be *centered* on each voxel.

    Parameters
    ----------
    im : array_like
        A binary image with the phase of interest set to True

    sizes : array_like or scalar
        The sizes to invade.  If a list of values of provided they are used
        directly.  If a scalar is provided then that number of points spanning
        the min and max of the distance transform are used.

    Returns
    -------
    An image with the pore size values in each voxel

    Notes
    -----
    The term *foreground* is used since this function can be applied to both
    pore space or the solid, whichever is set to True.

    This function is identical to porosimetry with ``access_limited`` set to
    False.

    """
    im_new = porosimetry(im=im, sizes=sizes, access_limited=False)
    return im_new


def porosimetry(im, sizes=25, inlets=None, access_limited=True,
                mode='fft'):
    r"""
    Performs a porosimetry simulution on the image

    Parameters
    ----------
    im : ND-array
        An ND image of the porous material containing True values in the
        pore space.

    sizes : array_like or scalar
        The sizes to invade.  If a list of values of provided they are used
        directly.  If a scalar is provided then that number of points spanning
        the min and max of the distance transform are used.

    inlets : ND-array, boolean
        A boolean mask with True values indicating where the invasion
        enters the image.  By default all faces are considered inlets,
        akin to a mercury porosimetry experiment.  Users can also apply
        solid boundaries to their image externally before passing it in,
        allowing for complex inlets like circular openings, etc.  This argument
        is only used if ``access_limited`` is ``True``.

    access_limited : Boolean
        This flag indicates if the intrusion should only occur from the
        surfaces (``access_limited`` is True, which is the default), or
        if the invading phase should be allowed to appear in the core of
        the image.  The former simulates experimental tools like mercury
        intrusion porosimetry, while the latter is useful for comparison
        to gauge the extent of shielding effects in the sample.

    mode : string
        Controls with method is used to compute the result.  Options are:

        *'fft'* - (default) Performs a distance tranform of the void space,
        thresholds to find voxels larger than ``sizes[i]``, trims the resulting
        mask if ``access_limitations`` is ``True``, then dilates it using the
        efficient fft-method to obtain the non-wetting fluid configuration.

        *'dt'* - Same as 'fft', except uses a second distance transform,
        relative to the thresholded mask, to find the invading fluid
        configuration.  The choice of 'dt' or 'fft' depends on speed, which
        is system and installation specific.

        *'mio'* - Using a single morphological image opening step to obtain the
        invading fluid confirguration directly, *then* trims if
        ``access_limitations`` is ``True``.  This method is not ideal and is
        included mostly for comparison purposes.

    Returns
    -------
    An ND-image with voxel values indicating the sphere radius at which it
    becomes accessible from the inlets.  This image can be used to find
    invading fluid configurations as a function of applied capillary pressure
    by applying a boolean comparison: ``inv_phase = im > r`` where ``r`` is
    the radius (in voxels) of the invading sphere.  Of course, ``r`` can be
    converted to capillary pressure using your favorite model.

    See Also
    --------
    fftmorphology

    """
    def trim_blobs(im, inlets):
        temp = sp.zeros_like(im)
        temp[inlets] = True
        labels, N = spim.label(im + temp)
        im = im ^ (clear_border(labels=labels) > 0)
        return im

    dt = spim.distance_transform_edt(im > 0)

    if inlets is None:
        inlets = get_border(im.shape, mode='faces')
    inlets = sp.where(inlets)

    if isinstance(sizes, int):
        sizes = sp.logspace(start=sp.log10(sp.amax(dt)), stop=0, num=sizes)
    else:
        sizes = sp.sort(a=sizes)[-1::-1]

    if im.ndim == 2:
        strel = disk
    else:
        strel = ball

    imresults = sp.zeros(sp.shape(im))
    if mode == 'mio':
        for r in tqdm(sizes):
            imtemp = fftmorphology(im, strel(r), mode='opening')
            if access_limited:
                imtemp = trim_blobs(imtemp, inlets)
            if sp.any(imtemp):
                imresults[(imresults == 0)*imtemp] = r
    if mode == 'dt':
        for r in tqdm(sizes):
            imtemp = dt >= r
            if access_limited:
                imtemp = trim_blobs(imtemp, inlets)
            if sp.any(imtemp):
                imtemp = spim.distance_transform_edt(~imtemp) < r
                imresults[(imresults == 0)*imtemp] = r
    if mode == 'fft':
        for r in tqdm(sizes):
            imtemp = dt >= r
            if access_limited:
                imtemp = trim_blobs(imtemp, inlets)
            if sp.any(imtemp):
                imtemp = fftconvolve(imtemp, strel(r), mode='same') > 0.1
                imresults[(imresults == 0)*imtemp] = r
    return imresults
