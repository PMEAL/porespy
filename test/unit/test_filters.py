import porespy as ps
import pytest
import scipy as sp
import numpy as np
import scipy.ndimage as spim


class FilterTest():
    def setup_class(self):
        sp.random.seed(0)
        self.im = ps.generators.blobs(shape=[100, 100, 100], blobiness=2)
        self.im_dt = spim.distance_transform_edt(self.im)
        self.flood_im = np.zeros([21, 21])
        self.flood_im[1:20, 1:20] = 1
        self.flood_im[:, 4] = 0
        self.flood_im[8, :] = 0
        self.flood_im_dt = spim.distance_transform_edt(self.flood_im)

    def test_porosimetry_npts_10(self):
        mip = ps.filters.porosimetry(im=self.im, sizes=10)
        ans = sp.array([0.00000000, 1.00000000, 1.37871571, 1.61887041,
                        1.90085700, 2.23196205, 2.62074139, 3.07724114,
                        3.61325732])
        assert sp.allclose(sp.unique(mip), ans)

    def test_porosimetry_with_sizes(self):
        s = sp.logspace(0.01, 0.6, 5)
        mip = ps.filters.porosimetry(im=self.im, sizes=s)
        assert sp.allclose(sp.unique(mip)[1:], s)

    def test_apply_chords_axis0(self):
        c = ps.filters.apply_chords(im=self.im, spacing=0, axis=0)
        assert c.sum() == 25879

    def test_apply_chords_axis1(self):
        c = ps.filters.apply_chords(im=self.im, spacing=0, axis=1)
        assert c.sum() == 25569

    def test_apply_chords_axis2(self):
        c = ps.filters.apply_chords(im=self.im, spacing=0, axis=2)
        assert c.sum() == 25840

    def test_apply_chords_with_spacing(self):
        with pytest.raises(Exception):
            c = ps.filters.apply_chords(im=self.im, spacing=-1)
        c = ps.filters.apply_chords(im=self.im, spacing=1)
        assert c.sum() == 11367

    def test_apply_chords_without_trimming(self):
        c = ps.filters.apply_chords(im=self.im, trim_edges=False)
        assert c.sum() == 31215

    def test_flood_size(self):
        m = ps.filters.flood(im=self.flood_im, mode='size')
        s = sp.unique(m)
        assert len(s) == 5
        assert max(s) == 165

    def test_flood_max(self):
        m = ps.filters.flood(im=self.flood_im_dt, mode='max')
        s = sp.unique(m)
        assert len(s) == 4
        assert max(s) == 6.0

    def test_flood_min(self):
        m = ps.filters.flood(im=self.flood_im_dt, mode='min')
        s = sp.unique(m)
        assert len(s) == 2
        assert max(s) == 1.0

    def test_find_disconnected_voxels_2d(self):
        h = ps.filters.find_disconnected_voxels(self.im[:, :, 0])
        assert sp.sum(h) == 477
#        print(sp.sum(h))

    def test_find_disconnected_voxels_2d_conn4(self):
        h = ps.filters.find_disconnected_voxels(self.im[:, :, 0], conn=4)
        assert sp.sum(h) == 652

    def test_find_disconnected_voxels_3d(self):
        h = ps.filters.find_disconnected_voxels(self.im)
        assert sp.sum(h) == 55

    def test_find_disconnected_voxels_3d_conn6(self):
        h = ps.filters.find_disconnected_voxels(self.im, conn=6)
        assert sp.sum(h) == 202

    def test_trim_nonpercolating_paths_2d_axis0(self):
        h = ps.filters.trim_nonpercolating_paths(self.im[:, :, 0],
                                                 inlet_axis=0, outlet_axis=0)
        assert sp.sum(h) == 3178

    def test_trim_nonpercolating_paths_2d_axis1(self):
        h = ps.filters.trim_nonpercolating_paths(self.im[:, :, 0],
                                                 inlet_axis=1, outlet_axis=1)
        assert sp.sum(h) == 1067

    def test_trim_nonpercolating_paths_3d_axis0(self):
        h = ps.filters.trim_nonpercolating_paths(self.im,
                                                 inlet_axis=0, outlet_axis=0)
        assert sp.sum(h) == 499733

    def test_trim_nonpercolating_paths_3d_axis1(self):
        h = ps.filters.trim_nonpercolating_paths(self.im,
                                                 inlet_axis=1, outlet_axis=1)
        assert sp.sum(h) == 499693

    def test_trim_nonpercolating_paths_3d_axis2(self):
        h = ps.filters.trim_nonpercolating_paths(self.im,
                                                 inlet_axis=2, outlet_axis=2)
        assert sp.sum(h) == 499611

    def test_fill_blind_pores(self):
        h = ps.filters.find_disconnected_voxels(self.im)
        b = ps.filters.fill_blind_pores(h)
        h = ps.filters.find_disconnected_voxels(b)
        assert sp.sum(h) == 0

    def test_trim_floating_solid(self):
        f = ps.filters.trim_floating_solid(~self.im)
        assert sp.sum(f) > sp.sum(~self.im)

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
        lt = ps.filters.local_thickness(self.im)
        assert lt.max() == self.im_dt.max()

    def test_porosimetry(self):
        im2d = self.im[:, :, 50]
        lt = ps.filters.local_thickness(im2d)
        sizes = np.unique(lt)
        mip = ps.filters.porosimetry(im2d,
                                     sizes=len(sizes),
                                     access_limited=False)
        assert mip.max() <= sizes.max()


if __name__ == '__main__':
    t = FilterTest()
    self = t
    t.setup_class()
    for item in t.__dir__():
        if item.startswith('test'):
            print('running test: '+item)
            t.__getattribute__(item)()
