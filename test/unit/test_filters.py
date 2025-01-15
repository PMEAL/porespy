import pytest
import numpy as np
from edt import edt
import porespy as ps
import scipy.ndimage as spim
from skimage.morphology import disk, ball
from skimage.util import random_noise
from scipy.stats import norm
ps.settings.tqdm['disable'] = True


class FilterTest():
    def setup_class(self):
        np.random.seed(0)
        self.im = ps.generators.blobs(shape=[100, 100, 100], blobiness=2)
        # Ensure that im was generated as expeccted
        assert ps.metrics.porosity(self.im) == 0.499829
        self.im_dt = edt(self.im)

    def test_im_in_not_im_out(self):
        im = self.im[:, :, 50]
        for item in ps.filters.__dir__():
            if ~item.startswith('__'):
                temp = getattr(ps.filters, item)
                assert temp is not im

    def test_porosimetry_compare_modes_2d(self):
        im = self.im[:, :, 50]
        sizes = np.arange(25, 1, -1)
        fft = ps.filters.porosimetry(im, mode='hybrid', sizes=sizes)
        mio = ps.filters.porosimetry(im, mode='mio', sizes=sizes)
        dt = ps.filters.porosimetry(im, mode='dt', sizes=sizes)
        assert np.all(fft == dt)
        assert np.all(fft == mio)

    def test_porosimetry_num_points(self):
        mip = ps.filters.porosimetry(im=self.im, sizes=10)
        steps = np.unique(mip)
        ans = np.array([0.00000000, 1.00000000, 1.37871571, 1.61887041,
                        1.90085700, 2.23196205, 2.62074139, 3.07724114,
                        3.61325732])
        assert np.allclose(steps, ans)

    def test_porosimetry_compare_modes_3d(self):
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
        np.random.seed(0)
        im = ps.generators.blobs([200, 200], porosity=0.55, blobiness=2)
        inlets = np.zeros_like(im)
        inlets[0, :] = 1
        outlets = np.zeros_like(im)
        outlets[-1, :] = 1
        assert spim.label(im)[1] > 1
        h = ps.filters.trim_nonpercolating_paths(im=im,
                                                 inlets=inlets,
                                                 outlets=outlets)
        assert spim.label(h)[1] == 1

    def test_trim_nonpercolating_paths_2d_axis1(self):
        np.random.seed(0)
        im = ps.generators.blobs([200, 200], porosity=0.55, blobiness=2)
        inlets = np.zeros_like(im)
        inlets[:, 0] = 1
        outlets = np.zeros_like(im)
        outlets[:, -1] = 1
        assert spim.label(im)[1] > 1
        h = ps.filters.trim_nonpercolating_paths(im=im,
                                                 inlets=inlets,
                                                 outlets=outlets)
        assert spim.label(h)[1] == 1

    def test_trim_nonpercolating_paths_no_paths(self):
        np.random.seed(0)
        im = ps.generators.blobs([200, 200], porosity=0.25, blobiness=2)
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
        im = ps.generators.blobs([100, 100, 100], porosity=0.55, blobiness=2)
        inlets = np.zeros_like(im)
        inlets[..., 0] = 1
        outlets = np.zeros_like(im)
        outlets[..., -1] = 1
        assert spim.label(im)[1] > 1
        h = ps.filters.trim_nonpercolating_paths(im=im,
                                                 inlets=inlets,
                                                 outlets=outlets)
        assert spim.label(h)[1] == 1

    def test_trim_nonpercolating_paths_3d_axis1(self):
        np.random.seed(0)
        im = ps.generators.blobs([100, 100, 100], porosity=0.55, blobiness=2)
        inlets = np.zeros_like(im)
        inlets[:, 0, :] = 1
        outlets = np.zeros_like(im)
        outlets[:, -1, :] = 1
        assert spim.label(im)[1] > 1
        h = ps.filters.trim_nonpercolating_paths(im=im,
                                                 inlets=inlets,
                                                 outlets=outlets)
        assert spim.label(h)[1] == 1

    def test_trim_nonpercolating_paths_3d_axis0(self):
        np.random.seed(0)
        im = ps.generators.blobs([100, 100, 100], porosity=0.55, blobiness=2)
        inlets = np.zeros_like(im)
        inlets[0, ...] = 1
        outlets = np.zeros_like(im)
        outlets[-1, ...] = 1
        assert spim.label(im)[1] > 1
        h = ps.filters.trim_nonpercolating_paths(im=im,
                                                 inlets=inlets,
                                                 outlets=outlets)
        assert spim.label(h)[1] == 1

    def test_trim_disconnected_blobs(self):
        np.random.seed(0)
        im = ps.generators.blobs([200, 200], porosity=0.55, blobiness=2)
        inlets = np.zeros_like(im)
        inlets[0, ...] = 1
        n1 = spim.label(im)[1]
        h = ps.filters.trim_disconnected_blobs(im=im, inlets=inlets)
        n2 = spim.label(h)[1]
        assert n1 > n2
        assert spim.label(h + inlets)[1] == 1

    def test_fill_blind_pores(self):
        h = ps.filters.find_disconnected_voxels(self.im)
        b = ps.filters.fill_blind_pores(h)
        h = ps.filters.find_disconnected_voxels(b)
        assert np.sum(h) == 0

    def test_fill_blind_pores_w_surface(self):
        im = ~ps.generators.lattice_spheres(shape=[101, 101], r=5,
                                            offset=0, spacing=20)
        im2 = ps.filters.fill_blind_pores(im, surface=False)
        assert im2.sum() > 0
        im3 = ps.filters.fill_blind_pores(im, surface=True)
        assert im3.sum() == 0

    def test_fill_blind_pores_surface_blobs_2D(self):
        im = ps.generators.blobs([100, 100], porosity=0.6, seed=0)
        im2 = ps.filters.fill_blind_pores(im)
        assert im.sum() == 6021
        assert im2.sum() == 5981
        im3 = ps.filters.fill_blind_pores(im, surface=True)
        assert im3.sum() == 5699

    def test_fill_blind_pores_surface_blobs_3D(self):
        im = ps.generators.blobs([100, 100, 100], porosity=0.5)
        im2 = ps.filters.fill_blind_pores(im, surface=True)
        labels, N = spim.label(im2, ps.tools.ps_rect(3, ndim=3))
        assert N == 1

    def test_trim_floating_solid(self):
        f = ps.filters.trim_floating_solid(~self.im)
        assert np.sum(f) > np.sum(~self.im)

    def test_trim_floating_solid_w_surface(self):
        im = ps.generators.lattice_spheres(shape=[101, 101], r=5,
                                           offset=0, spacing=20)
        im2 = ps.filters.trim_floating_solid(im, surface=False)
        assert im2.sum() < im.size
        im3 = ps.filters.trim_floating_solid(im, surface=True)
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
        lt = ps.filters.local_thickness(self.im, mode='dt')
        np.testing.assert_almost_equal(lt.max(), self.im_dt.max(), decimal=6)
        lt = ps.filters.local_thickness(self.im, mode='mio')
        np.testing.assert_almost_equal(lt.max(), self.im_dt.max(), decimal=6)
        lt = ps.filters.local_thickness(self.im, mode='hybrid')
        np.testing.assert_almost_equal(lt.max(), self.im_dt.max(), decimal=6)

    def test_local_thickness_known_sizes(self):
        im = np.zeros(shape=[300, 300])
        im = ps.generators.RSA(im, r=20)
        im = ps.generators.RSA(im, r=10)
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
        im = ps.generators.lattice_spheres(shape=[50, 50], r=4, offset=5)
        dt = spim.distance_transform_edt(im)
        ar = ps.filters.find_dt_artifacts(dt)
        inds = np.where(ar == ar.max())
        assert np.all(dt[inds] - ar[inds] == 1)

    def test_snow_partitioning_n_2D(self):
        np.random.seed(0)
        im = ps.generators.blobs([500, 500], blobiness=1)
        snow = ps.filters.snow_partitioning_n(im + 1, r_max=4, sigma=0.4)
        assert np.amax(snow.regions) == 136
        assert not np.any(np.isnan(snow.regions))
        assert not np.any(np.isnan(snow.dt))
        assert not np.any(np.isnan(snow.im))

    def test_snow_partitioning_n_3D(self):
        np.random.seed(0)
        im = ps.generators.blobs([100, 100, 100], blobiness=0.75)
        snow = ps.filters.snow_partitioning_n(im + 1, r_max=4, sigma=0.4)
        assert np.amax(snow.regions) == 620
        assert not np.any(np.isnan(snow.regions))
        assert not np.any(np.isnan(snow.dt))
        assert not np.any(np.isnan(snow.im))

    def test_snow_partitioning_parallel(self):
        np.random.seed(1)
        im = ps.generators.overlapping_spheres(shape=[1000, 1000],
                                                r=10, porosity=0.5)
        snow = ps.filters.snow_partitioning_parallel(im,
                                                      divs=[2, 2],
                                                      cores=None,
                                                      r_max=5, sigma=0.4)
        # assert np.amax(snow.regions) == 919
        assert not np.any(np.isnan(snow.regions))
        assert not np.any(np.isnan(snow.dt))
        assert not np.any(np.isnan(snow.im))

    def test_chunked_func_2d(self):
        from skimage.morphology import disk
        im = disk(50)
        f = ps.filters.fftmorphology
        s = disk(1)
        a = ps.filters.chunked_func(func=f, im=im, overlap=3, im_arg='im',
                                    strel=s, mode='erosion')
        b = ps.filters.fftmorphology(im, strel=s, mode='erosion')
        assert np.all(a == b)

    def test_chunked_func_3d(self):
        from skimage.morphology import ball
        im = ball(50)
        f = ps.filters.fftmorphology
        s = ball(1)
        a = ps.filters.chunked_func(func=f, im=im, im_arg='im', overlap=3,
                                    strel=s, mode='erosion')
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
        im = ps.generators.blobs(shape=[100, 100, 100])
        with pytest.raises(IndexError):
            ps.filters.chunked_func(func=spsg.convolve,
                                    in1=im*1.0,
                                    in2=ps.tools.ps_ball(5),
                                    im_arg='in1', strel_arg='in2',
                                    overlap=5)

    def test_prune_branches(self):
        from skimage.morphology import skeletonize
        im = ps.generators.random_spheres([100, 100, 100], r=4, seed=0)
        skel1 = skeletonize(im)
        skel2 = ps.filters.prune_branches(skel1)
        assert skel1.sum() > skel2.sum()

    def test_prune_branches_n2(self):
        from skimage.morphology import skeletonize
        im = ps.generators.random_spheres([100, 100, 100], r=4, seed=0)
        skel1 = skeletonize(im)
        skel2 = ps.filters.prune_branches(skel1, iterations=1)
        skel3 = ps.filters.prune_branches(skel1, iterations=2)
        assert skel1.sum() > skel2.sum()
        assert skel2.sum() > skel3.sum()
        skel4 = ps.filters.prune_branches(skel1, iterations=3)
        assert skel3.sum() > skel4.sum()

    def test_apply_padded(self):
        from skimage.morphology import skeletonize
        im = ps.generators.blobs(shape=[100, 100])
        skel1 = skeletonize(im)
        skel2 = ps.filters.apply_padded(
            im=im,
            pad_width=20,
            pad_val=1,
            func=skeletonize,
        )
        assert (skel1.astype(bool)).sum() != (skel2.astype(bool)).sum()

    def test_trim_small_clusters(self):
        np.random.seed(0)
        im = ps.generators.blobs(shape=[100, 100], blobiness=2, porosity=0.4)
        im5 = ps.filters.trim_small_clusters(im=im, size=5)
        im10 = ps.filters.trim_small_clusters(im=im, size=10)
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
        dt = spim.distance_transform_edt(input=im)
        dt_hold_peaks = ps.filters.hold_peaks(dt, axis=0)
        diff = abs(np.max(dt_hold_peaks, axis=0) - np.max(dt, axis=0))
        assert np.all(diff <= 1e-15)

    def test_nl_means_layered(self):
        im = ps.generators.blobs(shape=[50, 50, 50], blobiness=.5)
        np.random.seed(0)
        im2 = random_noise(im)
        filt = ps.filters.nl_means_layered(im=im2)
        p1 = (filt[0, ...] > 0.5).sum()
        p2 = (im[0, ...]).sum()
        np.testing.assert_approx_equal(np.around(p1 / p2, decimals=1), 1)

    def test_trim_nearby_peaks(self):
        np.random.seed(0)
        im = ps.generators.blobs(shape=[400, 400],
                                 blobiness=[2, 1],
                                 porosity=0.6)
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
        np.random.seed(0)
        im = ps.generators.blobs(shape=[400, 400],
                                 blobiness=[2, 1],
                                 porosity=0.6)
        im_dt = edt(im)
        dt = spim.gaussian_filter(input=im_dt, sigma=0.4)
        peaks = ps.filters.find_peaks(dt=dt)
        peaks_far = ps.filters.trim_nearby_peaks(peaks=peaks, dt=dt)
        peaks_close = ps.filters.trim_nearby_peaks(peaks=peaks, dt=dt, f=0.3)
        num_peaks_after_far_trim = spim.label(peaks_far)[1]
        num_peaks_after_close_trim = spim.label(peaks_close)[1]
        assert num_peaks_after_far_trim <= num_peaks_after_close_trim

    def test_regions_size(self):
        np.random.seed(0)
        im = ps.generators.blobs([50, 50], porosity=0.1)
        s = ps.filters.region_size(im)
        hits = [1, 2, 3, 4, 5, 6, 8, 9, 18, 23, 24, 26, 28, 31]
        assert np.all(hits == np.unique(s)[1:])
        np.random.seed(0)
        im = ps.generators.blobs([20, 20, 20], porosity=0.1)
        s = ps.filters.region_size(im)
        hits = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 16, 17, 19, 31, 32, 37]
        assert np.all(hits == np.unique(s)[1:])


if __name__ == '__main__':
    t = FilterTest()
    self = t
    t.setup_class()
    for item in t.__dir__():
        if item.startswith('test'):
            print(f'Running test: {item}')
            t.__getattribute__(item)()
