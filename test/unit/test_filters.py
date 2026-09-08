import numpy as np
import pytest
import scipy.ndimage as spim
from skimage.morphology import ball, disk, skeletonize
from skimage.util import random_noise

import porespy as ps

edt = ps.tools.get_edt()
ps.settings.tqdm['disable'] = True


class FilterTest():
    def setup_class(self):
        self.im = ps.generators.blobs(shape=[100, 100, 100],
                                      blobiness=2,
                                      seed=0,
                                      porosity=0.499829,
                                      periodic=False,)
        # Ensure that im was generated as expected
        assert self.im.sum()/self.im.size == 0.499829
        self.im_dt = edt(self.im)

    def test_im_in_not_im_out(self):
        im = self.im[:, :, 50]
        for item in ps.filters.__dir__():
            if not item.startswith('__'):
                temp = getattr(ps.filters, item)
                assert temp is not im

    def test_porosimetry_compare_modes_2d(self):
        im = self.im[:, :, 50]
        sizes = np.arange(25, 1, -1)
        fft = ps.filters.porosimetry(im, method='conv', sizes=sizes)
        dsi = ps.filters.porosimetry(im, method='dsi', sizes=sizes)
        dt = ps.filters.porosimetry(im, method='dt', sizes=sizes)
        assert np.all(fft == dt)
        assert np.all(fft == dsi)

    def test_porosimetry_num_points(self):
        mip = ps.filters.porosimetry(im=self.im, sizes=None)
        steps = np.unique(mip)
        ans = np.array([0., 1., 2., 3., 4,])
        assert np.allclose(steps, ans)

    def test_porosimetry_compare_modes_3d(self):
        im = self.im
        sizes = np.arange(25, 1, -1)
        fft = ps.filters.porosimetry(im, sizes=sizes, method='conv')
        dsi = ps.filters.porosimetry(im, sizes=sizes, method='dsi')
        dt = ps.filters.porosimetry(im, sizes=sizes, method='dt')
        assert np.all(fft == dt)
        assert np.all(fft == dsi)

    def test_porosimetry_with_sizes(self):
        s = np.logspace(0.01, 0.6, 5)
        mip = ps.filters.porosimetry(im=self.im, sizes=s)
        assert np.all(np.isin(np.unique(mip)[1:], s))

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
        im = ~ps.generators.lattice_spheres(shape=[100, 100], spacing=26,
                                            r=10)
        labels = spim.label(im)[0]
        sz = ps.filters.flood(im*2.0, labels=labels, mode='max')
        assert np.all(np.unique(sz) == [0, 2])
        sz = ps.filters.flood(im, labels=labels, mode='min')
        assert np.all(np.unique(sz) == [0, 1])
        sz = ps.filters.flood(im, labels=labels, mode='size')
        assert np.all(np.unique(sz) == [0, 305])

    def test_flood_func(self):
        im = ~ps.generators.lattice_spheres(shape=[100, 100], spacing=26,
                                            r=10)
        labels = spim.label(im)[0]
        sz = ps.filters.flood_func(im*2.0, labels=labels, func=np.amax)
        assert np.all(np.unique(sz) == [0, 2])

    def test_find_disconnected_voxels_2d(self):
        h = ps.filters.find_disconnected_voxels(self.im[:, :, 0], conn='max')
        assert np.sum(h) == 477

    def test_find_disconnected_voxels_2d_conn4(self):
        h = ps.filters.find_disconnected_voxels(self.im[:, :, 0], conn='min')
        assert np.sum(h) == 652

    def test_find_disconnected_voxels_3d(self):
        h = ps.filters.find_disconnected_voxels(self.im, conn='max')
        assert np.sum(h) == 55

    def test_find_disconnected_voxels_3d_conn6(self):
        h = ps.filters.find_disconnected_voxels(self.im, conn='min')
        assert np.sum(h) == 202

    @pytest.mark.parametrize(
        "shape, conn",
        [
            ((31, 29), "min"),
            ((31, 29), "max"),
            ((17, 19, 21), "min"),
            ((17, 19, 21), "max"),
        ],
    )
    def test_find_disconnected_voxels_matches_scipy(self, shape, conn):
        rng = np.random.default_rng(0)
        im = rng.random(shape) > 0.6
        inlets = np.zeros(shape, dtype=bool)
        inlets[0, ...] = True
        structure = ps.tools.get_strel()[len(shape)][conn]
        labels, _ = spim.label(im, structure=structure)
        keep = np.unique(labels[inlets])
        expected = np.isin(labels, keep[keep > 0], invert=True) * im

        actual = ps.filters.find_disconnected_voxels(
            im=im,
            inlets=inlets,
            conn=conn,
        )
        assert np.array_equal(actual, expected)

        actual_from_indices = ps.filters.find_disconnected_voxels(
            im=im,
            inlets=np.where(inlets),
            conn=conn,
        )
        assert np.array_equal(actual_from_indices, expected)

    def test_trim_nonpercolating_paths_2d_axis0(self):
        np.random.seed(0)
        im = ps.generators.blobs(
            shape=[200, 200], porosity=0.55875, blobiness=2, periodic=False,)
        assert im.sum()/im.size == 0.55875
        inlets = np.zeros_like(im)
        inlets[0, :] = 1
        outlets = np.zeros_like(im)
        outlets[-1, :] = 1
        assert spim.label(im)[1] > 1
        h = ps.filters.trim_nonpercolating_paths(
            im=im, inlets=inlets, outlets=outlets)
        assert spim.label(h)[1] == 1
        h2 = ps.filters.trim_nonpercolating_paths(
            im=im, axis=0)
        assert np.all(h == h2)

    def test_trim_nonpercolating_paths_2d_axis1(self):
        np.random.seed(0)
        im = ps.generators.blobs(
            shape=[200, 200], porosity=0.55875, blobiness=2, periodic=False,)
        assert im.sum()/im.size == 0.55875
        inlets = np.zeros_like(im)
        inlets[:, 0] = 1
        outlets = np.zeros_like(im)
        outlets[:, -1] = 1
        assert spim.label(im)[1] > 1
        h = ps.filters.trim_nonpercolating_paths(
            im=im, inlets=inlets, outlets=outlets)
        assert spim.label(h)[1] == 1
        h2 = ps.filters.trim_nonpercolating_paths(
            im=im, axis=1)
        assert np.all(h == h2)

    def test_trim_nonpercolating_paths_no_paths(self):
        np.random.seed(0)
        im = ps.generators.blobs(
            shape=[200, 200], porosity=0.2535, blobiness=2, periodic=False,)
        assert im.sum()/im.size == 0.2535
        inlets = np.zeros_like(im)
        inlets[:, 0] = 1
        outlets = np.zeros_like(im)
        outlets[:, -1] = 1
        assert spim.label(im)[1] > 1
        h = ps.filters.trim_nonpercolating_paths(im=im,
                                                 inlets=inlets,
                                                 outlets=outlets)
        assert h.sum() == 0

    def test_trim_nonpercolating_paths_3d_axis2(self):
        np.random.seed(0)
        im = ps.generators.blobs(
            shape=[100, 100, 100], porosity=0.550191, blobiness=2, periodic=False,)
        assert im.sum()/im.size == 0.550191
        inlets = np.zeros_like(im)
        inlets[..., 0] = 1
        outlets = np.zeros_like(im)
        outlets[..., -1] = 1
        assert spim.label(im)[1] > 1
        h = ps.filters.trim_nonpercolating_paths(im=im,
                                                 inlets=inlets,
                                                 outlets=outlets)
        assert spim.label(h)[1] == 1
        h2 = ps.filters.trim_nonpercolating_paths(
            im=im, axis=2)
        assert np.all(h == h2)

    def test_trim_nonpercolating_paths_3d_axis1(self):
        np.random.seed(0)
        im = ps.generators.blobs(
            shape=[100, 100, 100], porosity=0.550191, blobiness=2, periodic=False,)
        assert im.sum()/im.size == 0.550191
        inlets = np.zeros_like(im)
        inlets[:, 0, :] = 1
        outlets = np.zeros_like(im)
        outlets[:, -1, :] = 1
        assert spim.label(im)[1] > 1
        h = ps.filters.trim_nonpercolating_paths(im=im,
                                                 inlets=inlets,
                                                 outlets=outlets)
        assert spim.label(h)[1] == 1
        h2 = ps.filters.trim_nonpercolating_paths(
            im=im, axis=1)
        assert np.all(h == h2)

    def test_trim_nonpercolating_paths_3d_axis0(self):
        np.random.seed(0)
        im = ps.generators.blobs(
            shape=[100, 100, 100], porosity=0.550191, blobiness=2, periodic=False,)
        assert im.sum()/im.size == 0.550191
        inlets = np.zeros_like(im)
        inlets[0, ...] = 1
        outlets = np.zeros_like(im)
        outlets[-1, ...] = 1
        assert spim.label(im)[1] > 1
        h = ps.filters.trim_nonpercolating_paths(im=im,
                                                 inlets=inlets,
                                                 outlets=outlets)
        assert spim.label(h)[1] == 1
        h2 = ps.filters.trim_nonpercolating_paths(
            im=im, axis=0)
        assert np.all(h == h2)

    def test_trim_disconnected_voxels(self):
        np.random.seed(0)
        im = ps.generators.blobs(
            shape=[200, 200], porosity=0.55875, blobiness=2, periodic=False,)
        assert im.sum()/im.size == 0.55875
        inlets = np.zeros_like(im)
        inlets[0, ...] = 1
        n1 = spim.label(im)[1]
        h = ps.filters.trim_disconnected_voxels(im=im, inlets=inlets, conn='min')
        n2 = spim.label(h)[1]
        assert n1 > n2
        assert spim.label(h + inlets)[1] == 1

    def test_fill_closed_pores(self):
        h = ps.filters.find_disconnected_voxels(self.im)
        b = ps.filters.fill_closed_pores(h)
        h = ps.filters.find_disconnected_voxels(b)
        assert np.sum(h) == 0

    def test_fill_closed_pores_surface_blobs_2D(self):
        im = ps.generators.blobs(
            shape=[100, 100], porosity=0.6021, seed=0, periodic=False,)
        assert im.sum()/im.size == 0.6021
        im2 = ps.filters.fill_closed_pores(im)
        assert im.sum() == 6021
        assert im2.sum() < im.sum()

    def test_fill_invalid_pores_surface_blobs_3D(self):
        im = ps.generators.blobs(
            shape=[100, 100, 100], porosity=0.497569, seed=0, periodic=False,)
        assert im.sum()/im.size == 0.497569
        im2 = ps.filters.fill_invalid_pores(im)
        labels, N = spim.label(im2, ps.tools.ps_rect(3, ndim=3))
        assert N == 1

    def test_trim_floating_solid(self):
        f = ps.filters.trim_floating_solid(~self.im)
        assert np.sum(f) > np.sum(~self.im)

    def test_trim_floating_solid_w_surface(self):
        im = ps.generators.lattice_spheres(shape=[101, 101], r=5,
                                           offset=0, spacing=20)
        im2 = ps.filters.trim_floating_solid(im, incl_surface=False)
        assert im2.sum() < im.size
        im3 = ps.filters.trim_floating_solid(im, incl_surface=True)
        assert im3.sum() == im.size

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
        lt = ps.filters.local_thickness(self.im, method='dt')
        np.testing.assert_almost_equal(lt.max(), self.im_dt.max(), decimal=6)
        lt = ps.filters.local_thickness(self.im, method='imj')
        np.testing.assert_almost_equal(lt.max(), self.im_dt.max(), decimal=6)
        lt = ps.filters.local_thickness(self.im, method='conv')
        np.testing.assert_almost_equal(lt.max(), self.im_dt.max(), decimal=6)

    def test_local_thickness_imj_2d(self):
        im = self.im[:, :, 50]
        lt = ps.filters.local_thickness_imj(im)
        assert lt.shape == im.shape
        assert lt.max() > 0

    def test_local_thickness_known_sizes(self):
        im = np.zeros(shape=[300, 300])
        im = ps.generators.random_spheres(im=im, r=20)
        im = ps.generators.random_spheres(im=im, r=10)
        lt = ps.filters.local_thickness(im, sizes=[20, 10])
        assert np.all(np.unique(lt) == [0, 10, 20])

    def test_morphology_fft_dilate_2d(self):
        im = self.im[:, :, 50]
        truth = spim.binary_dilation(im, structure=disk(3))
        test = ps.filters.fftmorphology(im, strel=disk(3), mode='dilation')
        assert np.all(truth == test)

    def test_morphology_fft_erode_2d(self):
        im = self.im[:, :, 50]
        truth = spim.binary_erosion(im, structure=disk(3))
        test = ps.filters.fftmorphology(im, strel=disk(3), mode='erosion')
        assert np.all(truth == test)

    def test_morphology_fft_opening_2d(self):
        im = self.im[:, :, 50]
        truth = spim.binary_opening(im, structure=disk(3))
        test = ps.filters.fftmorphology(im, strel=disk(3), mode='opening')
        assert np.all(truth == test)

    def test_morphology_fft_closing_2d(self):
        im = self.im[:, :, 50]
        truth = spim.binary_closing(im, structure=disk(3))
        test = ps.filters.fftmorphology(im, strel=disk(3), mode='closing')
        assert np.all(truth == test)

    def test_morphology_fft_dilate_3d(self):
        im = self.im
        truth = spim.binary_dilation(im, structure=ball(3))
        test = ps.filters.fftmorphology(im, strel=ball(3), mode='dilation')
        assert np.all(truth == test)

    def test_morphology_fft_erode_3d(self):
        im = self.im
        truth = spim.binary_erosion(im, structure=ball(3))
        test = ps.filters.fftmorphology(im, strel=ball(3), mode='erosion')
        assert np.all(truth == test)

    def test_morphology_fft_opening_3d(self):
        im = self.im
        truth = spim.binary_opening(im, structure=ball(3))
        test = ps.filters.fftmorphology(im, strel=ball(3), mode='opening')
        assert np.all(truth == test)

    def test_morphology_fft_closing_3d(self):
        im = self.im
        truth = spim.binary_closing(im, structure=ball(3))
        test = ps.filters.fftmorphology(im, strel=ball(3), mode='closing')
        assert np.all(truth == test)

    def test_erode_2D_smooth(self):
        im = ps.generators.blobs([100, 100], porosity=0.5, seed=0)
        r = 5
        smooth = True
        se = ps.tools.ps_round(r=r, ndim=im.ndim, smooth=smooth)
        truth = spim.binary_erosion(im, structure=se, border_value=1)
        test = ps.filters.erode(im=im, r=r, method='conv', smooth=smooth)
        assert np.all(truth == test)

    def test_erode_2D_not_smooth(self):
        im = ps.generators.blobs([100, 100], porosity=0.5, seed=0)
        r = 5
        smooth = False
        se = ps.tools.ps_round(r=r, ndim=im.ndim, smooth=smooth)
        truth = spim.binary_erosion(im, structure=se, border_value=1)
        test = ps.filters.erode(im=im, r=r, method='conv', smooth=smooth)
        assert np.all(truth == test)

    def test_erode_2D_smooth_dt_vs_conv(self):
        im = ps.generators.blobs([100, 100], porosity=0.5, seed=0)
        r = 5
        smooth = True
        test1 = ps.filters.erode(im=im, r=r, method='conv', smooth=smooth)
        test2 = ps.filters.erode(im=im, r=r, method='dt', smooth=smooth)
        assert np.all(test1 == test2)

    def test_erode_2D_not_smooth_dt_vs_conv(self):
        im = ps.generators.blobs([100, 100], porosity=0.5, seed=0)
        r = 5
        smooth = False
        test1 = ps.filters.erode(im=im, r=r, method='conv', smooth=smooth)
        test2 = ps.filters.erode(im=im, r=r, method='dt', smooth=smooth)
        assert np.all(test1 == test2)

    def test_dilate_2D_smooth(self):
        im = ps.generators.blobs([100, 100], porosity=0.5, seed=0)
        r = 5
        smooth = True
        se = ps.tools.ps_round(r=r, ndim=im.ndim, smooth=smooth)
        truth = spim.binary_dilation(im, structure=se)
        test = ps.filters.dilate(im=im, r=r, method='conv', smooth=smooth)
        assert np.all(truth == test)

    def test_dilate_2D_not_smooth(self):
        im = ps.generators.blobs([100, 100], porosity=0.5, seed=0)
        r = 5
        smooth = False
        se = ps.tools.ps_round(r=r, ndim=im.ndim, smooth=smooth)
        truth = spim.binary_dilation(im, structure=se)
        test = ps.filters.dilate(im=im, r=r, method='conv', smooth=smooth)
        assert np.all(truth == test)

    def test_dilate_2D_smooth_dt_vs_conv(self):
        im = ps.generators.blobs([100, 100], porosity=0.5, seed=0)
        r = 5
        smooth = True
        test1 = ps.filters.dilate(im=im, r=r, method='conv', smooth=smooth)
        test2 = ps.filters.dilate(im=im, r=r, method='dt', smooth=smooth)
        assert np.all(test1 == test2)

    def test_dilate_2D_not_smooth_dt_vs_conv(self):
        im = ps.generators.blobs([100, 100], porosity=0.5, seed=0)
        r = 5
        smooth = False
        test1 = ps.filters.dilate(im=im, r=r, method='conv', smooth=smooth)
        test2 = ps.filters.dilate(im=im, r=r, method='dt', smooth=smooth)
        assert np.all(test1 == test2)

    def test_reduce_peaks(self):
        im = ~ps.generators.lattice_spheres(shape=[50, 50], r=5, offset=3)
        peaks = ps.filters.reduce_peaks(im)
        assert spim.label(im)[1] == spim.label(peaks)[1]
        im = ~ps.generators.lattice_spheres(shape=[50, 50, 50], r=5,
                                            offset=3)
        peaks = ps.filters.reduce_peaks(im)
        assert spim.label(im)[1] == spim.label(peaks)[1]

    def test_nphase_border_2d_no_diagonals(self):
        im = np.zeros([110, 110])
        for i in range(6):
            im[int(10*2*i):int(10*(2*i+1)), :] += 2
            im[:, int(10*2*i):int(10*(2*i+1))] += 4
        borders = ps.filters.nphase_border(im, conn="min")
        nb, counts = np.unique(borders, return_counts=True)
        assert nb.tolist() == [1.0, 2.0, 3.0]
        assert counts.tolist() == [8100, 3600, 400]

    def test_nphase_border_2d_diagonals(self):
        im = np.zeros([110, 110])
        for i in range(6):
            im[int(10*2*i):int(10*(2*i+1)), :] += 2
            im[:, int(10*2*i):int(10*(2*i+1))] += 4
        borders = ps.filters.nphase_border(im, conn='max')
        nb, counts = np.unique(borders, return_counts=True)
        assert nb.tolist() == [1.0, 2.0, 4.0]
        assert counts.tolist() == [8100, 3600, 400]

    def test_nphase_border_3d_no_diagonals(self):
        im3d = np.zeros([110, 110, 110])
        for i in range(6):
            im3d[int(10*2*i):int(10*(2*i+1)), :, :] += 2
            im3d[:, int(10*2*i):int(10*(2*i+1)), :] += 4
            im3d[:, :, int(10*2*i):int(10*(2*i+1))] += 8
        borders = ps.filters.nphase_border(im3d, conn="min")
        nb, counts = np.unique(borders, return_counts=True)
        assert nb.tolist() == [1.0, 2.0, 3.0, 4.0]
        assert counts.tolist() == [729000, 486000, 108000, 8000]

    def test_nphase_border_3d_diagonals(self):
        im3d = np.zeros([110, 110, 110])
        for i in range(6):
            im3d[int(10*2*i):int(10*(2*i+1)), :, :] += 2
            im3d[:, int(10*2*i):int(10*(2*i+1)), :] += 4
            im3d[:, :, int(10*2*i):int(10*(2*i+1))] += 8
        borders = ps.filters.nphase_border(im3d, conn='max')
        nb, counts = np.unique(borders, return_counts=True)
        assert nb.tolist() == [1.0, 2.0, 4.0, 8.0]
        assert counts.tolist() == [729000, 486000, 108000, 8000]

    def test_find_dt_artifacts(self):
        im = ps.generators.lattice_spheres(shape=[50, 50], r=4, offset=5)
        dt = edt(im)
        ar = ps.filters.find_dt_artifacts(dt)
        inds = np.where(ar == ar.max())
        assert np.all(dt[inds] - ar[inds] == 1)

    def test_snow_partitioning_n_2D(self):
        im = ps.generators.blobs(
            shape=[500, 500], porosity=0.494604, blobiness=1, seed=0, periodic=False)
        assert im.sum()/im.size == 0.494604
        snow = ps.filters.snow_partitioning_n(im + 1, r_max=4, sigma=0.4)
        assert np.amax(snow.regions) == 136
        assert not np.any(np.isnan(snow.regions))
        assert not np.any(np.isnan(snow.dt))
        assert not np.any(np.isnan(snow.im))

    def test_snow_partitioning_n_3D(self):
        im = ps.generators.blobs(shape=[100, 100, 100],
                                 porosity=0.495157,
                                 blobiness=0.75,
                                 seed=0,
                                 periodic=False,)
        assert im.sum()/im.size == 0.495157
        snow = ps.filters.snow_partitioning_n(im + 1, r_max=4, sigma=0.4)
        assert np.amax(snow.regions) == 620
        assert not np.any(np.isnan(snow.regions))
        assert not np.any(np.isnan(snow.dt))
        assert not np.any(np.isnan(snow.im))

    def test_snow_partitioning_parallel(self):
        np.random.seed(1)
        im = ps.generators.overlapping_spheres(shape=[1000, 1000],
                                               r=10,
                                               porosity=0.5)
        parallel_kw = {"divs": [2, 2], "cores": None, "overlap": None}
        snow = ps.filters.snow_partitioning_parallel(im,
                                                     parallel_kw=parallel_kw,
                                                     r_max=5,
                                                     sigma=0.4)
        # assert np.amax(snow.regions) == 919
        assert not np.any(np.isnan(snow.regions))
        assert not np.any(np.isnan(snow.dt))
        assert not np.any(np.isnan(snow.im))

    def test_chunked_func_2d(self):
        from skimage.morphology import disk
        im = disk(50)
        f = ps.filters.fftmorphology
        s = disk(1)
        parallel_kw = {"divs": 2, "overlap": 3, "cores": None}
        a = ps.filters.chunked_func(func=f, im=im, parallel_kw=parallel_kw,
                                    im_arg='im', strel=s, mode='erosion')
        b = ps.filters.fftmorphology(im, strel=s, mode='erosion')
        assert np.all(a == b)

    def test_chunked_func_3d(self):
        from skimage.morphology import ball
        im = ball(50)
        f = ps.filters.fftmorphology
        s = ball(1)
        parallel_kw = {"divs": 2, "overlap": 3, "cores": None}
        a = ps.filters.chunked_func(func=f, im=im, im_arg='im',
                                    parallel_kw=parallel_kw, strel=s,
                                    mode='erosion')
        b = ps.filters.fftmorphology(im, strel=s, mode='erosion')
        assert np.all(a == b)

    def test_chunked_func_3d_w_strel(self):
        from skimage.morphology import ball
        im = ball(50)
        f = ps.filters.fftmorphology
        s = ball(1)
        a = ps.filters.chunked_func(func=f, im=im, im_arg='im',
                                    strel_arg='strel', strel=s, mode='erosion')
        b = ps.filters.fftmorphology(im, strel=s, mode='erosion')
        assert np.all(a == b)

    def test_chunked_func_w_ill_defined_filter(self):
        import scipy.signal as spsg
        im = ps.generators.blobs(
            shape=[100, 100, 100], porosity=0.497569, seed=0, periodic=False,)
        assert im.sum()/im.size == 0.497569
        with pytest.raises(IndexError):
            parallel_kw = {"divs": 2, "overlap": 5, "cores": None}
            ps.filters.chunked_func(func=spsg.convolve,
                                    in1=im*1.0,
                                    in2=ps.tools.ps_ball(5),
                                    im_arg='in1', strel_arg='in2',
                                    parallel_kw=parallel_kw)

    def test_prune_branches(self):
        im = ps.generators.random_spheres([100, 100, 100], r=4, seed=0)
        skel1 = skeletonize(im)
        skel2 = ps.filters.prune_branches(skel1)
        # TODO: This is failing on github
        # assert skel1.sum() > skel2.sum()

    def test_prune_branches_n2(self):
        im = ps.generators.random_spheres(shape=[100, 100, 100], r=4, seed=0)
        skel1 = skeletonize(im)
        skel2 = ps.filters.prune_branches(skel1, iterations=1)
        skel3 = ps.filters.prune_branches(skel1, iterations=2)
        assert skel1.sum() > skel2.sum()
        assert skel2.sum() > skel3.sum()
        skel4 = ps.filters.prune_branches(skel1, iterations=3)
        assert skel3.sum() > skel4.sum()

    def test_apply_padded(self):
        im = ps.generators.blobs(
            shape=[100, 100], periodic=False,)
        skel1 = skeletonize(im)
        skel2 = ps.filters.apply_padded(
            im=im,
            pad_width=20,
            pad_val=1,
            func=skeletonize,
        )
        assert (skel1.astype(bool)).sum() != (skel2.astype(bool)).sum()

    def test_trim_small_clusters(self):
        im = ps.generators.blobs(shape=[100, 100],
                                 blobiness=2,
                                 porosity=0.4028,
                                 seed=0,
                                 periodic=False,)
        assert im.sum()/im.size == 0.4028
        im5 = ps.filters.trim_small_clusters(im=im, min_size=5)
        im10 = ps.filters.trim_small_clusters(im=im, min_size=10)
        assert im5.sum() > im10.sum()
        label, N = spim.label(im10)
        for i in range(N):
            assert np.sum(label == i) > 10
        label, N = spim.label(im*~im10)
        for i in range(1, N):
            assert np.sum(label == i) <= 10

    def test_hold_peaks_input(self):
        im = self.im[:50, :50, :50]
        result_bool = ps.filters.hold_peaks(im, axis=0)
        result_float = ps.filters.hold_peaks(im.astype(float), axis=0)
        assert np.all(result_bool == result_float)

    def test_hold_peaks_algorithm(self):
        im = self.im[:, :, 5]
        dt = edt(im)
        dt_hold_peaks = ps.filters.hold_peaks(dt, axis=0)
        diff = abs(np.max(dt_hold_peaks, axis=0) - np.max(dt, axis=0))
        assert np.all(diff <= 1e-15)

    def test_nl_means_layered(self):
        im = ps.generators.blobs(shape=[50, 50, 50],
                                 porosity=0.492664,
                                 blobiness=0.5,
                                 seed=0,
                                 periodic=False,)
        assert im.sum()/im.size == 0.492664
        np.random.seed(0)
        im2 = random_noise(im)
        filt = ps.filters.nl_means_layered(im=im2)
        p1 = (filt[0, ...] > 0.5).sum()
        p2 = (im[0, ...]).sum()
        np.testing.assert_approx_equal(np.around(p1 / p2, decimals=1), 1)

    def test_trim_nearby_peaks(self):
        im = ps.generators.blobs(shape=[400, 400],
                                 blobiness=[2, 1],
                                 porosity=0.5916375,
                                 seed=0,
                                 periodic=False,)
        assert im.sum()/im.size == 0.5916375
        im_dt = edt(im)
        dt = spim.gaussian_filter(input=im_dt, sigma=0.4)
        peaks = ps.filters.find_peaks(dt=dt, r_max=4)
        labels, N = spim.label(peaks, structure=ps.tools.ps_rect(3, 2))
        assert N == 148
        peaks1 = ps.filters.trim_saddle_points(peaks=peaks, dt=im_dt)
        labels, N = spim.label(peaks1, structure=ps.tools.ps_rect(3, 2))
        assert N == 135
        peaks2 = ps.filters.trim_nearby_peaks(peaks=peaks1, dt=im_dt, f=1)
        labels, N = spim.label(peaks2, structure=ps.tools.ps_rect(3, 2))
        assert N == 113

    def test_trim_nearby_peaks_threshold(self):
        im = ps.generators.blobs(shape=[400, 400],
                                 blobiness=[2, 1],
                                 porosity=0.5916375,
                                 seed=0,
                                 periodic=False,)
        assert im.sum()/im.size == 0.5916375
        im_dt = edt(im)
        dt = spim.gaussian_filter(input=im_dt, sigma=0.4)
        peaks = ps.filters.find_peaks(dt=dt)
        peaks_far = ps.filters.trim_nearby_peaks(peaks=peaks, dt=dt)
        peaks_close = ps.filters.trim_nearby_peaks(peaks=peaks, dt=dt, f=0.3)
        num_peaks_after_far_trim = spim.label(peaks_far)[1]
        num_peaks_after_close_trim = spim.label(peaks_close)[1]
        assert num_peaks_after_far_trim <= num_peaks_after_close_trim

    def test_regions_size(self):
        im = ps.generators.blobs(
            shape=[50, 50], porosity=0.1164, seed=0, periodic=False,)
        assert im.sum()/im.size == 0.1164
        s = ps.filters.region_size(im)
        hits = [1, 2, 3, 4, 5, 6, 8, 9, 18, 23, 24, 26, 28, 31]
        assert np.all(hits == np.unique(s)[1:])
        np.random.seed(0)
        im = ps.generators.blobs(
            shape=[20, 20, 20], porosity=0.121, seed=0, periodic=False,)
        assert im.sum()/im.size == 0.121
        s = ps.filters.region_size(im)
        hits = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 16, 17, 19, 31, 32, 37]
        assert np.all(hits == np.unique(s)[1:])

    def test_find_trapped_clusters_side_outlet(self):
        im = ps.generators.blobs(
            shape=[100, 100], porosity=0.6, seed=7, periodic=False,)
        inlets = np.zeros_like(im)
        inlets[0, :] = True
        outlets = np.zeros_like(im)
        outlets[:, -1] = True
        inv = ps.simulations.drainage(im, inlets=inlets)
        trp1 = ps.filters.find_trapped_clusters(
            im=im,
            seq=inv.im_seq,
            outlets=outlets,
            method='labels',
            )
        inv = ps.simulations.drainage(im, inlets=inlets)
        trp2 = ps.filters.find_trapped_clusters(
            im=im,
            seq=inv.im_seq,
            outlets=outlets,
            method='queue',
        )
        assert np.all(trp1 == trp2)

    def test_find_trapped_clusters_return_mask_top_outlet(self):
        im = ps.generators.blobs(
            shape=[100, 100], porosity=0.6, seed=7, periodic=False,)
        inlets = np.zeros_like(im)
        inlets[0, :] = True
        outlets = np.zeros_like(im)
        outlets[-1, :] = True
        inv = ps.simulations.drainage(im, inlets=inlets)
        trp1 = ps.filters.find_trapped_clusters(
            im=im,
            seq=inv.im_seq,
            outlets=outlets,
            method='labels',
            )
        inv = ps.simulations.drainage(im, inlets=inlets)
        trp2 = ps.filters.find_trapped_clusters(
            im=im,
            seq=inv.im_seq,
            outlets=outlets,
            method='queue',
        )
        assert np.all(trp1 == trp2)

    def test_find_trapped_regions_top_outlet(self):
        im = ps.generators.blobs(
            shape=[100, 100], porosity=0.6, seed=7, periodic=False,)
        inlets = np.zeros_like(im)
        inlets[0, :] = True
        outlets = np.zeros_like(im)
        outlets[-1, :] = True
        inv = ps.simulations.drainage(im, inlets=inlets)
        trp1 = ps.filters.find_trapped_clusters(
            im=im,
            seq=inv.im_seq,
            outlets=outlets,
            method='labels',
        )
        inv = ps.simulations.drainage(im, inlets=inlets)
        trp2 = ps.filters.find_trapped_clusters(
            im=im,
            seq=inv.im_seq,
            outlets=outlets,
            method='queue',
        )

        assert np.all(trp1 == trp2)

    def test_find_trapped_clusters_with_imbibition(self):
        # This image has some surface pores which should not become trapped
        # because outlets are all surfaces
        im = ps.generators.blobs([100, 100], porosity=0.6, seed=1)
        faces = ps.generators.borders(im.shape, mode='faces')
        pc = ps.filters.capillary_transform(
            im=im,
            sigma=0.465,
            theta=140,
            voxel_size=1e-5,
        )
        imb = ps.simulations.imbibition(im=im, pc=pc, steps=50)
        mask = ps.filters.find_trapped_clusters(
            im=im, seq=imb.im_seq, outlets=faces, method='labels')
        assert np.sum(mask[faces]) == 0
        mask = ps.filters.find_trapped_clusters(
            im=im, seq=imb.im_seq, outlets=faces, method='queue')
        assert np.sum(mask[faces]) == 0

    def test_find_trapped_clusters_warns_when_invasion_misses_outlets(
        self, caplog
    ):
        import logging
        im = ps.generators.blobs(
            shape=[100, 100], porosity=0.6, seed=3, periodic=False)
        im = ps.filters.fill_closed_pores(im)
        inlets = ps.generators.faces(im.shape, inlet=0)
        outlets = ps.generators.faces(im.shape, outlet=0)
        # Use few steps so invasion stops before reaching the far face
        pc = ps.filters.capillary_transform(
            im=im, sigma=0.072, theta=180, voxel_size=1e-5)
        steps = np.arange(pc[im].min(), pc[im].max()/5, 25)
        inv = ps.simulations.drainage(im=im, pc=pc, inlets=inlets, steps=steps)
        msg = "Invasion did not reach outlets"
        for method in ("labels", "queue"):
            caplog.clear()
            with caplog.at_level(logging.WARNING, logger="porespy.filters"):
                _ = ps.filters.find_trapped_clusters(
                    im=im, seq=inv.im_seq, outlets=outlets, method=method)
            assert any(msg in rec.message for rec in caplog.records)
        # Sanity check: no warning when invasion does reach outlets
        steps_full = np.arange(pc[im].min(), pc[im].max()*2, 25)
        inv = ps.simulations.drainage(
            im=im, pc=pc, inlets=inlets, steps=steps_full)
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="porespy.filters"):
            _ = ps.filters.find_trapped_clusters(
                im=im, seq=inv.im_seq, outlets=outlets, method="labels")
        assert not any(msg in rec.message for rec in caplog.records)


if __name__ == '__main__':
    t = FilterTest()
    self = t
    t.setup_class()
    for item in t.__dir__():
        if item.startswith('test'):
            print(f'Running test: {item}')
            t.__getattribute__(item)()
