import porespy as ps
import pytest
import numpy as np
import scipy.ndimage as spim
from skimage.morphology import disk, ball


class FilterTest():
    def setup_class(self):
        np.random.seed(0)
        self.im = ps.generators.blobs(shape=[100, 100, 100], blobiness=2)
        # Ensure that im was generated as expeccted
        assert ps.metrics.porosity(self.im) == 0.499829
        self.im_dt = spim.distance_transform_edt(self.im)

    def test_im_in_not_im_out(self):
        im = self.im[:, :, 50]
        for item in ps.filters.__dir__():
            if ~item.startswith('__'):
                temp = getattr(ps.filters, item)
                assert temp is not im

    def test_porosimetry_compare_modes_2D(self):
        im = self.im[:, :, 50]
        sizes = np.arange(25, 1, -1)
        fft = ps.filters.porosimetry(im, mode='hybrid', sizes=sizes)
        mio = ps.filters.porosimetry(im, mode='mio', sizes=sizes)
        dt = ps.filters.porosimetry(im, mode='dt', sizes=sizes)
        assert np.all(fft == dt)
        assert np.all(fft == mio)

    def test_porosimetry_npts_10(self):
        mip = ps.filters.porosimetry(im=self.im, sizes=10)
        steps = np.unique(mip)
        ans = np.array([0.00000000, 1.00000000, 1.37871571, 1.61887041,
                        1.90085700, 2.23196205, 2.62074139, 3.07724114,
                        3.61325732, 4.24264069])
        assert np.allclose(steps, ans)

    def test_porosimetry_compare_modes_3D(self):
        im = self.im
        sizes = np.arange(25, 1, -1)
        fft = ps.filters.porosimetry(im, sizes=sizes, mode='hybrid')
        mio = ps.filters.porosimetry(im, sizes=sizes, mode='mio')
        dt = ps.filters.porosimetry(im, sizes=sizes, mode='dt')
        assert np.all(fft == dt)
        assert np.all(fft == mio)

    def test_porosimetry_with_sizes(self):
        s = np.logspace(0.01, 0.6, 5)
        mip = ps.filters.porosimetry(im=self.im, sizes=s)
        assert np.allclose(np.unique(mip)[1:], s)

    def test_apply_chords_axis0(self):
        c = ps.filters.apply_chords(im=self.im, spacing=3, axis=0)
        assert c.sum() == 23722
        c = ps.filters.apply_chords(im=self.im, axis=0)
        assert c.sum() == 102724

    def test_apply_chords_axis1(self):
        c = ps.filters.apply_chords(im=self.im, spacing=3, axis=1)
        assert c.sum() == 23422
        c = ps.filters.apply_chords(im=self.im, axis=1)
        assert c.sum() == 102205

    def test_apply_chords_axis2(self):
        c = ps.filters.apply_chords(im=self.im, spacing=3, axis=2)
        assert c.sum() == 23752
        c = ps.filters.apply_chords(im=self.im, axis=2)
        assert c.sum() == 103347

    def test_apply_chords_with_negative_spacing(self):
        with pytest.raises(Exception):
            ps.filters.apply_chords(im=self.im, spacing=-1)

    def test_apply_chords_without_trimming(self):
        c = ps.filters.apply_chords(im=self.im, trim_edges=False)
        assert c.sum() == 125043
        c = ps.filters.apply_chords(im=self.im, spacing=3, trim_edges=False)
        assert c.sum() == 31215

    def test_apply_chords_3D(self):
        ps.filters.apply_chords_3D(self.im)

    def test_flood(self):
        im = ~ps.generators.lattice_spheres(shape=[100, 100], offset=3,
                                            radius=10)
        sz = ps.filters.flood(im*2.0, mode='max')
        assert np.all(np.unique(sz) == [0, 2])
        sz = ps.filters.flood(im, mode='min')
        assert np.all(np.unique(sz) == [0, 1])
        sz = ps.filters.flood(im, mode='size')
        assert np.all(np.unique(sz) == [0, 305])

    def test_find_disconnected_voxels_2d(self):
        h = ps.filters.find_disconnected_voxels(self.im[:, :, 0])
        assert np.sum(h) == 477

    def test_find_disconnected_voxels_2d_conn4(self):
        h = ps.filters.find_disconnected_voxels(self.im[:, :, 0], conn=4)
        assert np.sum(h) == 652

    def test_find_disconnected_voxels_3d(self):
        h = ps.filters.find_disconnected_voxels(self.im)
        assert np.sum(h) == 55

    def test_find_disconnected_voxels_3d_conn6(self):
        h = ps.filters.find_disconnected_voxels(self.im, conn=6)
        assert np.sum(h) == 202

    def test_trim_nonpercolating_paths_2d_axis0(self):
        h = ps.filters.trim_nonpercolating_paths(self.im[:, :, 0],
                                                 inlet_axis=0, outlet_axis=0)
        assert np.sum(h) == 3178

    def test_trim_nonpercolating_paths_2d_axis1(self):
        h = ps.filters.trim_nonpercolating_paths(self.im[:, :, 0],
                                                 inlet_axis=1, outlet_axis=1)
        assert np.sum(h) == 1067

    def test_trim_nonpercolating_paths_3d_axis0(self):
        h = ps.filters.trim_nonpercolating_paths(self.im,
                                                 inlet_axis=0, outlet_axis=0)
        assert np.sum(h) == 499733

    def test_trim_nonpercolating_paths_3d_axis1(self):
        h = ps.filters.trim_nonpercolating_paths(self.im,
                                                 inlet_axis=1, outlet_axis=1)
        assert np.sum(h) == 499693

    def test_trim_nonpercolating_paths_3d_axis2(self):
        h = ps.filters.trim_nonpercolating_paths(self.im,
                                                 inlet_axis=2, outlet_axis=2)
        assert np.sum(h) == 499611

    def test_fill_blind_pores(self):
        h = ps.filters.find_disconnected_voxels(self.im)
        b = ps.filters.fill_blind_pores(h)
        h = ps.filters.find_disconnected_voxels(b)
        assert np.sum(h) == 0

    def test_trim_floating_solid(self):
        f = ps.filters.trim_floating_solid(~self.im)
        assert np.sum(f) > np.sum(~self.im)

    def test_trim_extrema_min(self):
        dt = self.im_dt[:, :, 45:55]
        min1 = np.min(dt[self.im[:, :, 45:55]])
        min_im = ps.filters.trim_extrema(dt, h=2, mode='minima')
        min2 = np.min(min_im[self.im[:, :, 45:55]])
        assert min2 > min1

    def test_trim_extrema_max(self):
        dt = self.im_dt[:, :, 45:55]
        max1 = np.max(dt[self.im[:, :, 45:55]])
        max_im = ps.filters.trim_extrema(dt, h=2, mode='maxima')
        max2 = np.max(max_im[self.im[:, :, 45:55]])
        assert max1 > max2

    def test_local_thickness(self):
        lt = ps.filters.local_thickness(self.im, mode='dt')
        assert lt.max() == self.im_dt.max()
        lt = ps.filters.local_thickness(self.im, mode='mio')
        assert lt.max() == self.im_dt.max()
        lt = ps.filters.local_thickness(self.im, mode='hybrid')
        assert lt.max() == self.im_dt.max()

    def test_local_thickness_known_sizes(self):
        im = np.zeros(shape=[300, 300])
        im = ps.generators.RSA(im=im, radius=20)
        im = ps.generators.RSA(im=im, radius=10)
        im = im > 0
        lt = ps.filters.local_thickness(im, sizes=[20, 10])
        assert np.all(np.unique(lt) == [0, 10, 20])

    def test_porosimetry(self):
        im2d = self.im[:, :, 50]
        lt = ps.filters.local_thickness(im2d)
        sizes = np.unique(lt)
        mip = ps.filters.porosimetry(im2d,
                                     sizes=len(sizes),
                                     access_limited=False)
        assert mip.max() <= sizes.max()

    def test_morphology_fft_dilate_2D(self):
        im = self.im[:, :, 50]
        truth = spim.binary_dilation(im, structure=disk(3))
        test = ps.tools.fftmorphology(im, strel=disk(3), mode='dilation')
        assert np.all(truth == test)

    def test_morphology_fft_erode_2D(self):
        im = self.im[:, :, 50]
        truth = spim.binary_erosion(im, structure=disk(3))
        test = ps.tools.fftmorphology(im, strel=disk(3), mode='erosion')
        assert np.all(truth == test)

    def test_morphology_fft_opening_2D(self):
        im = self.im[:, :, 50]
        truth = spim.binary_opening(im, structure=disk(3))
        test = ps.tools.fftmorphology(im, strel=disk(3), mode='opening')
        assert np.all(truth == test)

    def test_morphology_fft_closing_2D(self):
        im = self.im[:, :, 50]
        truth = spim.binary_closing(im, structure=disk(3))
        test = ps.tools.fftmorphology(im, strel=disk(3), mode='closing')
        assert np.all(truth == test)

    def test_morphology_fft_dilate_3D(self):
        im = self.im
        truth = spim.binary_dilation(im, structure=ball(3))
        test = ps.tools.fftmorphology(im, strel=ball(3), mode='dilation')
        assert np.all(truth == test)

    def test_morphology_fft_erode_3D(self):
        im = self.im
        truth = spim.binary_erosion(im, structure=ball(3))
        test = ps.tools.fftmorphology(im, strel=ball(3), mode='erosion')
        assert np.all(truth == test)

    def test_morphology_fft_opening_3D(self):
        im = self.im
        truth = spim.binary_opening(im, structure=ball(3))
        test = ps.tools.fftmorphology(im, strel=ball(3), mode='opening')
        assert np.all(truth == test)

    def test_morphology_fft_closing_3D(self):
        im = self.im
        truth = spim.binary_closing(im, structure=ball(3))
        test = ps.tools.fftmorphology(im, strel=ball(3), mode='closing')
        assert np.all(truth == test)

    def test_reduce_peaks(self):
        im = ~ps.generators.lattice_spheres(shape=[50, 50], radius=5, offset=3)
        peaks = ps.filters.reduce_peaks(im)
        assert spim.label(im)[1] == spim.label(peaks)[1]
        im = ~ps.generators.lattice_spheres(shape=[50, 50, 50], radius=5,
                                            offset=3)
        peaks = ps.filters.reduce_peaks(im)
        assert spim.label(im)[1] == spim.label(peaks)[1]

    def test_nphase_border_2d_no_diagonals(self):
        im = np.zeros([110, 110])
        for i in range(6):
            im[int(10*2*i):int(10*(2*i+1)), :] += 2
            im[:, int(10*2*i):int(10*(2*i+1))] += 4
        borders = ps.filters.nphase_border(im, include_diagonals=False)
        nb, counts = np.unique(borders, return_counts=True)
        assert nb.tolist() == [1.0, 2.0, 3.0]
        assert counts.tolist() == [8100, 3600, 400]

    def test_nphase_border_2d_diagonals(self):
        im = np.zeros([110, 110])
        for i in range(6):
            im[int(10*2*i):int(10*(2*i+1)), :] += 2
            im[:, int(10*2*i):int(10*(2*i+1))] += 4
        borders = ps.filters.nphase_border(im, include_diagonals=True)
        nb, counts = np.unique(borders, return_counts=True)
        assert nb.tolist() == [1.0, 2.0, 4.0]
        assert counts.tolist() == [8100, 3600, 400]

    def test_nphase_border_3d_no_diagonals(self):
        im3d = np.zeros([110, 110, 110])
        for i in range(6):
            im3d[int(10*2*i):int(10*(2*i+1)), :, :] += 2
            im3d[:, int(10*2*i):int(10*(2*i+1)), :] += 4
            im3d[:, :, int(10*2*i):int(10*(2*i+1))] += 8
        borders = ps.filters.nphase_border(im3d, include_diagonals=False)
        nb, counts = np.unique(borders, return_counts=True)
        assert nb.tolist() == [1.0, 2.0, 3.0, 4.0]
        assert counts.tolist() == [729000, 486000, 108000, 8000]

    def test_nphase_border_3d_diagonals(self):
        im3d = np.zeros([110, 110, 110])
        for i in range(6):
            im3d[int(10*2*i):int(10*(2*i+1)), :, :] += 2
            im3d[:, int(10*2*i):int(10*(2*i+1)), :] += 4
            im3d[:, :, int(10*2*i):int(10*(2*i+1))] += 8
        borders = ps.filters.nphase_border(im3d, include_diagonals=True)
        nb, counts = np.unique(borders, return_counts=True)
        assert nb.tolist() == [1.0, 2.0, 4.0, 8.0]
        assert counts.tolist() == [729000, 486000, 108000, 8000]

    def test_find_dt_artifacts(self):
        im = ps.generators.lattice_spheres(shape=[50, 50], radius=4, offset=5)
        dt = spim.distance_transform_edt(im)
        ar = ps.filters.find_dt_artifacts(dt)
        inds = np.where(ar == ar.max())
        assert np.all(dt[inds] - ar[inds] == 1)

    def test_snow_partitioning_n(self):
        im = self.im
        snow = ps.filters.snow_partitioning_n(im + 1, r_max=4, sigma=0.4,
                                              return_all=True, mask=True,
                                              randomize=False, alias=None)
        assert np.amax(snow.regions) == 44
        assert not np.any(np.isnan(snow.regions))
        assert not np.any(np.isnan(snow.dt))
        assert not np.any(np.isnan(snow.im))

if __name__ == '__main__':
    t = FilterTest()
    self = t
    t.setup_class()
    for item in t.__dir__():
        if item.startswith('test'):
            print('running test: '+item)
            t.__getattribute__(item)()
