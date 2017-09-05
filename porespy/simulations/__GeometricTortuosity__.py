import porespy as ps
import scipy as sp
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize_3d
from skimage.morphology import disk, square, ball, cube
import scipy.ndimage as spim
import scipy.spatial as sptl
import skfmm
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing


class GeometricTortuosity():
    def __init__(self, im):
        self.im = im
        self._dt = spim.distance_transform_edt(self.im)
        self._make_skeleton()
        self._find_peaks_on_skel()

    def run(self, mode='group'):
        r"""
        """
        if mode == 'group':
            pass
        elif mode == 'individual':
            pass
        else:
            raise Exception('Unsupported mode:'+mode)

    @property
    def skeleton(self):
        if not hasattr(self, '_skel'):
            self._make_skeleton()
        return self._skel

    @property
    def peaks(self):
        if not hasattr(self, '_peaks'):
            self._find_peaks_on_skel()
        return self._peaks

    @property
    def dt(self):
        if not hasattr(self, '_dt'):
            print('Calculating distance transform...')
            self._dt = spim.distance_transform_edt(self.im)
        return self._dt

    def _make_skeleton(self):
        print('Obtaining skeleton...')
        from skimage.morphology import disk, square, ball, cube
        if self.im.ndim == 2:
            cube = square
            ball = disk
        skel = skeletonize_3d(self.im)
        skel = spim.binary_dilation(input=skel, structure=cube(3))
        skel = spim.binary_erosion(input=skel, structure=ball(1))
        self._skel = skel

    def _find_peaks_on_skel(self):
        print('Finding peaks...')
        from skimage.morphology import disk, square, ball, cube
        if self.im.ndim == 2:
            cube = square
            ball = disk
        dt = spim.gaussian_filter(input=self.dt, sigma=0.4)
        peaks = ps.network_extraction.find_peaks(dt)
        peaks = ps.network_extraction.trim_saddle_points(peaks, dt)
        # Clean up peaks to ensure they overlap the skeleton
        print('Total number of peaks in image: ', spim.label(peaks)[1])
        ok_peaks = self.skeleton*peaks
        print('Peaks originally on skeleton: ', spim.label(ok_peaks)[1])
        dil_peaks = spim.binary_dilation(input=peaks*(~ok_peaks),
                                         structure=cube(5))
        new_peaks = self.skeleton*dil_peaks
        peaks = ps.network_extraction.reduce_peaks_to_points(new_peaks +
                                                             ok_peaks)
        peaks = self.skeleton*peaks  # Ensure all peaks overlap skeleton
        print('Peaks on skeleton after adjustments: ', spim.label(peaks)[1])
        self._peaks = peaks

    def _get_start_points(self):
        if hasattr(self, '_start_points'):
            points = self._start_points
        else:
            points = self.peaks
        return points

    def _set_start_points(self, points):
        self._start_points = points

    start_points = property(fget=_get_start_points, fset=_set_start_points)

    def _get_end_points(self):
        if hasattr(self, '_end_points'):
            points = self._end_points
        else:
            points = self.start_points
        return points

    def _set_end_points(self, points):
        self._end_points = points

    end_points = property(fget=_get_end_points, fset=_set_end_points)

    def set_start_points(self, mode='random', npoints=100, axis=0):
        if mode == 'random':
            labels, N = spim.label(self.peaks)
            labels = ps.tools.randomize_colors(labels)
            points = (labels <= npoints)*self.peaks
            self._start_points = sp.where(points)
        if mode == 'maxima':
            d = sp.sort(self.dt[self.peaks > 0])
            if d.size > npoints:
                dmin = d[-npoints]
            else:
                dmin = d[0]
            points = (self.dt > dmin)*self.peaks
            self._start_points = sp.where(points)
        if mode == 'face':
            coords = sp.vstack(sp.where(self.peaks)).T
            ind = sp.argsort(coords[:, axis])
            coords = coords[ind[:npoints]]
            points = sp.zeros_like(self.peaks)
            points[tuple(coords.T)] = True
            self._start_points = sp.where(points)

    def set_end_points(self, mode='face', npoints=100, axis=0):
        if mode == 'face':
            coords = sp.vstack(sp.where(self.peaks)).T
            ind = sp.argsort(coords[:, axis])
            coords = coords[ind[-npoints:]]
            points = sp.zeros_like(self.peaks)
            points[tuple(coords.T)] = True
            self._end_points = sp.where(points)

    @property
    def euclidean_distance_matrix(self):
        if not hasattr(self, '_dmap'):
            coords_in = sp.vstack(self.start_points).T
            coords_out = sp.vstack(self.end_points).T
            dmap = sptl.distance_matrix(coords_in, coords_out)
            N = dmap.shape[0]
            dmap[range(N), range(N)] = sp.inf  # Distance to self = infinity
            self._dmap = dmap
        return self._dmap

    @property
    def geometric_distance_matrix(self):
        if not hasattr(self, '_gmap'):
            gmap = self._fmm_single_start()
            self._gmap = gmap
        return self._gmap

    @property
    def tortuosity_matrix(self):
        dmap = self.euclidean_distance_matrix
        gmap = self.geometric_distance_matrix
        tau = sp.linalg.triu(gmap/dmap, k=1)
        self.tau = tau
        return tau

    def _fmm_group_start(self):
        coords_in = self.start_points
        coords_out = self.end_points
        speed = self.skeleton*1.0
        phi = self.skeleton*1
        phi[coords_in] = 0  # Set start point of FMM walk
        ma = sp.ma.MaskedArray(phi, self.skeleton == 0)
        td = skfmm.travel_time(ma, speed)
        td = sp.array(td)
        self._gmap = td[coords_out]

    def _fmm_single_start(self):
        coords_in = self.start_points
        coords_out = self.end_points
        npoints = sp.size(coords_in[0])
        peaks = sp.zeros_like(self.peaks)
        peaks[coords_in] = True
        slices = spim.find_objects(input=spim.label(peaks)[0])
        tau = sp.zeros(shape=[npoints, npoints])
        speed = self.skeleton*1.0
        phi = self.skeleton*1
        print('Performing fast marching from starting points')
        for i in tqdm(sp.arange(npoints)):
            s = slices[i]
            phi[s] = 0  # Set start point of FMM walk
            ma = sp.ma.MaskedArray(phi, self.skeleton == 0)
            td = skfmm.travel_time(ma, speed)
            td = sp.array(td)
            phi[s] = 1  # Remove start point for next iteration
            tau[i, :] = td[coords_out]
        self._gmap = tau

    def _fmm_on_void_space(self):
        coords_in = self.start_points
        coords_out = self.end_points
        npoints = sp.size(coords_in[0])
        peaks = sp.zeros_like(self.peaks)
        peaks[coords_in] = True
        slices = spim.find_objects(input=spim.label(peaks)[0])
        tau = sp.zeros(shape=[npoints, npoints])
        speed = self.im*1.0
        phi = self.im*1
        print('Performing fast marching from starting points')
        for i in tqdm(sp.arange(npoints)):
            s = slices[i]
            phi[s] = 0  # Set start point of FMM walk
            ma = sp.ma.MaskedArray(phi, self.im == 0)
            td = skfmm.travel_time(ma, speed)
            td = sp.array(td)
            phi[s] = 1  # Remove start point for next iteration
            tau[i, :] = td[coords_out]
        self._gmap = tau

    def plot_tau_distribution(self, fig=None, bins=26, normed=True, **kwargs):
        if fig is None:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
        else:
            ax = fig.get_axes()[0]
        ax.hist(tau[tau > 0], bins=bins, normed=normed, **kwargs)
        return fig

    def plot_geometric_v_euclidean(self, fig=None, **kwargs):
        if fig is None:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
        else:
            ax = fig.get_axes()[0]
        x = self.euclidean_distance_matrix
        y = self.geometric_distance_matrix
        x = sp.linalg.triu(x, k=1)
        y = sp.linalg.triu(y, k=1)
        x = x[y > 0]
        y = y[y > 0]
        ax.plot(x, y, **kwargs)
        ax.plot([0, y.max()], [0, y.max()], 'k--')
        return fig


if __name__ == '__main__':
#    im = ps.generators.cylinders([150, 300, 300], radius=5, nfibers=300, phi_max=15)
    im = ps.generators.overlapping_spheres(shape=[400, 400], radius=8, porosity=0.7)
    gt = GeometricTortuosity(im)

    gt.set_start_points(mode='face', axis=0, npoints=20)
    gt.set_end_points(mode='face', axis=0, npoints=20)
    gt._fmm_group_start()
    print(gt._gmap.mean())

    gt.set_start_points(mode='face', axis=1, npoints=20)
    gt.set_end_points(mode='face', axis=1, npoints=20)
    gt._fmm_group_start()
    print(gt._gmap.mean())

#    gt.set_start_points(mode='face', axis=2, npoints=20)
#    gt.set_end_points(mode='face', axis=2, npoints=20)
#    gt._fmm_group_start()
#    print(gt._gmap.mean())
