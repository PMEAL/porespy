import porespy as ps
import pytest
import scipy as sp
import scipy.ndimage as spim


class FilterTest():
    def setup_class(self):
        sp.random.seed(0)
        self.im = ps.generators.blobs(shape=[100, 100, 100], blobiness=2)
        self.im_dt = spim.distance_transform_edt(self.im)

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

    def test_apply_chords3D(self):
        c = ps.filters.apply_chords_3D(im=self.im)
        assert c.sum() == 162885
        assert sp.all(sp.unique(c) == [0, 1, 2, 3])

    def test_apply_chords3D_with_spacing(self):
        with pytest.raises(Exception):
            c = ps.filters.apply_chords(im=self.im, spacing=-1)
        c = ps.filters.apply_chords_3D(im=self.im, spacing=1)
        assert c.sum() == 74250

    def test_apply_chords3D_without_trimming(self):
        c = ps.filters.apply_chords_3D(im=self.im, trim_edges=False)
        assert c.sum() == 187576

    def test_flood_size(self):
        m = ps.filters.flood(im=self.im[:, :, 50:53], mode='size')
        s = sp.unique(m)
        assert len(s) == 19
        assert max(s) == 13316

    def test_flood_max(self):
        dt = spim.distance_transform_edt(self.im[:, :, 50:53])
        m = ps.filters.flood(im=dt, mode='max')
        s = sp.unique(m)


if __name__ == '__main__':
    t = FilterTest()
    t.setup_class()
    t.test_apply_chords_axis0()
    t.test_apply_chords_axis1()
    t.test_apply_chords_axis2()
    t.test_apply_chords_with_spacing()
    t.test_apply_chords_without_trimming()
    t.test_apply_chords3D()
    t.test_apply_chords3D_with_spacing()
    t.test_flood_size()
    self = t
