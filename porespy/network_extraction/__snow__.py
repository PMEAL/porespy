import scipy as sp
import scipy.ndimage as spim
import scipy.spatial as sptl
from skimage.segmentation import find_boundaries
from skimage.morphology import disk, square, ball, cube, watershed

class SNOW(object):
    r"""
    The SNOW network extraction algorithm (Sub-Network of an Over-segmented
    Watershed) is specifically designed for high porosity materials.
    """

    def __init__(self, im=None, dt=None):
        if im is not None:
            dt = spim.distance_transform_edt(im)
        else:
            im = dt == 0
        self.im = im.squeeze()
        self.dt = dt.squeeze()
        if im.ndim == 2:
            self.ball = disk
            self.cube = square
        elif im.ndim == 3:
            self.ball = ball
            self.cube = cube
        else:
            raise Exception("only 2d and 3d images are supported")

    def run(self):
        peaks = self.get_initial_peaks()
        peaks = self.remove_peaks_in_small_regions(peaks)
        peaks = self.remove_peaks_on_plateaus(peaks)
        peaks = self.thin_broad_peaks(peaks)
        peaks = self.remove_nearby_peaks(peaks)
        regions = self.apply_watershed(peaks)
        edges = find_boundaries(regions)
        return (regions, peaks, edges)

    def get_initial_peaks(self):
        mx = spim.maximum_filter(self.dt + 2*(~self.im), footprint=self.ball(3))
        peaks = (self.dt == mx)*self.im
        return peaks

    def apply_watershed(self, peaks):
        # Apply watershed segmentation
        markers, N = spim.label(peaks, structure=self.cube(3))
        regions = watershed(-self.dt, markers=markers)
        return regions*self.im

    def remove_peaks_in_small_regions(self, peaks):
        regions = self.apply_watershed(peaks)
        edges = find_boundaries(regions)
        # Keep peaks that are in small pores by eroding the solid before opening
        im_temp = spim.binary_erosion(~self.im, structure=self.ball(2), iterations=1)
        regions2 = spim.binary_opening((1-edges)*(~im_temp), structure=self.ball(3))
        # Find peaks that are within their regions after opening and keep only them
        peaks = peaks*((regions > 0)*regions2)
        return peaks

    def remove_peaks_on_plateaus(self, peaks, max_iter=10):
        iters = 0
        while iters < max_iter:
            iters += 1
            mx = spim.maximum_filter(self.dt*(1 - peaks) + 2*(~self.im),
                                     footprint=self.cube(3))
            bad_peaks = (self.dt == (mx*peaks))*self.im
            peaks = peaks^bad_peaks
            if sp.sum(bad_peaks) == 0:
                break
        return peaks

    def thin_broad_peaks(self, peaks):
        markers, N = spim.label(input=peaks, structure=self.cube(3))
        inds = spim.measurements.center_of_mass(input=peaks, labels=markers,
                                                index=sp.unique(markers)[1:])
        inds = sp.floor(inds).astype(int)
        # Centroid may not be on old pixel, so just create a new peaks image
        peaks = sp.zeros_like(peaks, dtype=bool)
        peaks[tuple(inds.T)] = True
        return peaks

    def remove_nearby_peaks(self, peaks, max_iter=10, min_spacing=None):
        min_spacing = self.dt.max()*0.8
        iters = 0
        while iters < max_iter:
            iters += 1
            crds = sp.where(peaks)  # Find locations of all peaks
            dist_to_solid = self.dt[crds]  # Get distance to solid for each peak
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
        return peaks
