import scipy as sp
import scipy.ndimage as spim
import scipy.spatial as sptl
from skimage.segmentation import find_boundaries
from skimage.morphology import watershed, reconstruction
from porespy.tools import get_border, flood


def snow(im, r_max=4, sigma=0):
    r"""
    This function extracts the true local maximum of the distance transform of
    a pore space image.  These local maxima can then be used as markers in a
    marker based watershed segmentation such as that included in Scikit-Image
    or through the MorphoJ plugin in ImageJ.

    The SNOW network extraction algorithm (Sub-Network of an Over-segmented
    Watershed) was designed to handle to perculiarities of high porosity
    materials, but it applies equally well to other materials as well.

    Parameters
    ----------
    im : array_like
        Can be either (a) a boolean image of the domain, with ``True``
        indicating the pore space and ``False`` elsewhere, of (b) a distance
        transform of the domain calculated externally by the user.

    r_max : scalar
        The radius of there spherical structuring element to use in the Maximum
        filter stage that is used to find peaks.  The default is 4

    sigma : scalar
        The standard deviation of the Gaussian filter used in step 1.  The
        default is 0.4.

    Returns
    -------
    An array the same shape as the input image, with non-zero values indicating
    the subset of peaks found by the algorithm.  The peaks are returned as a
    label array that can be directly used as markers in a watershed
    segmentation.

    """
    im = im.squeeze()
    print('_'*60)
    print("Beginning SNOW Algorithm to remove spurious peaks")

    if im.dtype == 'bool':
        dt = spim.distance_transform_edt(input=im)
    else:
        dt = im
        im = dt > 0

    if sigma > 0:
        dt = spim.gaussian_filter(input=dt, sigma=sigma)

    peaks = peak_local_max(image=-dt, min_distance=r_max/2, exclude_border=0,
                           indices=False)
    print('Initial number of peaks: ', spim.label(peaks)[1])
    peaks = trim_saddle_points(peaks=peaks, dt=dt)
    peaks = trim_nearby_peaks(peaks=peaks, dt=dt)
    peaks, N = spim.label(peaks)
    print('Final number of peaks: ', N)
    return peaks


def reduce_peaks_to_points():
    print("Thin broad or elongated peaks to single pixel")
    start_peaks = sp.sum(peaks)
    markers, N = spim.label(input=peaks, structure=cube(3))
    inds = spim.measurements.center_of_mass(input=peaks,
                                            labels=markers,
                                            index=range(1, N))
    inds = sp.floor(inds).astype(int)
    # Centroid may not be on old pixel, so create a new peaks image
    peaks = sp.zeros_like(peaks, dtype=bool)
    peaks[tuple(inds.T)] = True
    end_peaks = sp.sum(peaks)
    print("--> Number of peaks removed:", str(start_peaks - end_peaks))


def trim_saddle_points(peaks, dt, max_iters=10):
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
    return peaks


def trim_saddle_points_orig():
    print("Remove peaks on saddles in the distance transform")
    start_peaks = spim.label(peaks, structure=cube(3))[1]
    # R is +1 higher than R in maximum_filter step
    peaks_dil = spim.binary_dilation(input=peaks,
                                     structure=ball(r_max+1))
    peaks_max = flood(im=dt*peaks_dil, mode='max')
    peaks = (peaks_max == dt)*peaks
    end_peaks = spim.label(peaks, structure=cube(3))[1]
    print("--> Number of peaks removed:", str(start_peaks - end_peaks))


def trim_nearby_peaks(peaks, dt):
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


def trim_nearby_peaks_orig(peaks, dt):
    print("Remove peaks nearer to another peak than to solid")
    start_peaks = spim.label(peaks, structure=cube(3))[1]
    if min_spacing is None:
        min_spacing = dt.max()*0.8
    iters = 0
    while iters < 10:
        iters += 1
        crds = sp.where(peaks)  # Find locations of all peaks
        dist_to_solid = dt[crds]  # Get distance to solid for each peak
        dist_to_solid += sp.rand(dist_to_solid.size)*1e-5  # Perturb distances
        crds = sp.vstack(crds).T  # Convert peak locations to ND-array
        dist = sptl.distance.cdist(XA=crds, XB=crds)  # Get distance between peaks
        sp.fill_diagonal(a=dist, val=sp.inf)  # Remove 0's in diagonal
        dist[dist > min_spacing] = sp.inf  # Keep peaks that are far apart
        dist_to_nearest_neighbor = sp.amin(dist, axis=0)
        nearby_neighbors = sp.where(dist_to_nearest_neighbor < dist_to_solid)[0]
        for peak in nearby_neighbors:
            nearest_neighbor = sp.amin(sp.where(dist[peak, :] == sp.amin(dist[peak, :]))[0])
            if dist_to_solid[peak] < dist_to_solid[nearest_neighbor]:
                peaks[tuple(crds[peak])] = 0
            else:
                peaks[tuple(crds[nearest_neighbor])] = 0
        if len(nearby_neighbors) == 0:
            break
    end_peaks = spim.label(peaks, structure=cube(3))[1]
    print("--> Number of peaks removed:", str(start_peaks - end_peaks))
