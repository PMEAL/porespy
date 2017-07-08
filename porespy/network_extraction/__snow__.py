import scipy as sp
import scipy.ndimage as spim
import scipy.spatial as sptl
from skimage.segmentation import find_boundaries
from skimage.morphology import watershed, reconstruction
from porespy.tools import get_border, flood


def SNOW_peaks(dt, min_spacing=None, r_max=3, r_min=3, do_steps=[1, 2, 3, 4]):
    r"""
    The SNOW network extraction algorithm (Sub-Network of an Over-segmented
    Watershed) is specifically designed for high porosity materials.

    Parameters
    ----------
    dt : 2D or 3D array (numeric)
        The distance transform of the pore space.  This argument is used,
        instead of the binary image of the pore space, so provide control
        over how the distance transform is obtained, and what sort of filtering
        or pretreatment is applied.

    min_spacing : scalar
        When two peaks are nearer to each other than to solid, the one that is
        furthest from the solid is kept.  However, if the spacing between the
        peaks is greater than ``min_spacing``, they are both kept.  This
        prevents some large pores from being merged.  If not specified, a value
        of 0.8 times the maximum value in the distance transform is used.

    r_max : scalar
        The radius of there spherical structuring element to use in the maximum
        filter stage.  A low value results in many spurious peaks, while a high
        value missing legitimate peaks.  The default is 4.

    Returns
    -------
    An array the same shape as the input image, with non-zero values indicating
    the subset of peaks found by the algorithm.  The peaks are returned as a
    label array that can be directly used as markers in a watershed
    segmentation.

    Notes
    -----
    The SNOW algorithm works with a basic Euclidean distance transform with
    no pretreatment, but certain unforseen applications may work better with
    city-block distance transform for instance.  Users may also wish to apply
    some image smoothing, such as a median or gausssian filter.

    """
    from skimage.morphology import disk, square, ball, cube
    dt = dt.squeeze()
    im = dt > 0
    if im.ndim == 2:
        ball = disk
        cube = square
    elif im.ndim == 3:
        ball = ball
        cube = cube
    else:
        raise Exception("only 2-d and 3-d images are supported")

    print("Beginning SNOW Algorithm to remove spurious peaks")
    # Step 0: Find ALL local maxima peaks in watershed
    mx = spim.maximum_filter(dt, footprint=ball(r_max))
    peaks = (dt == mx)*im

    N_peaks_initial = spim.label(peaks, structure=cube(3))[1]
    for step in do_steps:

        if step == 1:
            print("Step 1: Remove peaks in regions that are too small")
            # Apply watershed segmentation to initial peaks
            markers, N = spim.label(peaks, structure=cube(3))
            regions = watershed(-dt, markers=markers, mask=dt > 0)
            # Keep peaks that are in small pores by eroding the solid first
            temp_peaks = sp.copy(peaks)
            start_peaks = spim.label(peaks, structure=cube(3))[1]
            im_temp = spim.binary_erosion(~im, structure=ball(2), iterations=1)
            edges = find_boundaries(regions)
            regions2 = spim.binary_opening((1-edges)*(~im_temp),
                                           structure=ball(3))
            peaks = peaks*((regions > 0)*regions2)
            borders = get_border(im.shape, mode='edges')
            peaks += temp_peaks*borders
            end_peaks = spim.label(peaks, structure=cube(3))[1]
            print("--> Number of peaks removed:", str(start_peaks - end_peaks))

        if step == 2:
            print("Step 2: Remove peaks on saddles in the distance transform")
            start_peaks = spim.label(peaks, structure=cube(3))[1]
            iters = 0
            while iters < 10:
                iters += 1
                mx = spim.maximum_filter(dt*(1 - peaks) + 2*(~im),
                                         footprint=cube(3))
                bad_peaks = (dt == (mx*peaks))*im
                peaks = peaks ^ bad_peaks
                if sp.sum(bad_peaks) == 0:
                    break
            end_peaks = spim.label(peaks, structure=cube(3))[1]
            print("--> Number of peaks removed:", str(start_peaks - end_peaks))

        if step == 22:
            print("Step 22: Remove peaks on saddles in the distance transform")
            start_peaks = spim.label(peaks, structure=cube(3))[1]
            # R is +1 higher than R in maximum_filter step
            peaks_dil = spim.binary_dilation(input=peaks,
                                             structure=ball(r_max+1))
            peaks_max = flood(im=dt*peaks_dil, mode='max')
            peaks = (peaks_max == dt)*peaks
            end_peaks = spim.label(peaks, structure=cube(3))[1]
            print("--> Number of peaks removed:", str(start_peaks - end_peaks))

        if step == 3:
            print("Step 3: Thin broad or elongated peaks to single pixel")
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

        if step == 4:
            print("Step 4: Remove peaks nearer to another peak than to solid")
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

    # Step 5: Label the peaks and return
    markers, N_peaks_final = spim.label(peaks, structure=cube(3))
    print("SNOW algorithm complete:", str(N_peaks_initial-N_peaks_final),
          "peaks have been removed")
    return markers
