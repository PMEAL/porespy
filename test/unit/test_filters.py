import porespy as ps
from porespy import filters
import scipy as sp
import scipy.ndimage as spim


class FilterTest():
    def setup_class(self):
        sp.random.seed(0)
        self.im = ps.generators.blobs(shape=[100, 100, 100], blobiness=2)

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
        c = ps.filters.apply_chords(im=self.im, spacing=1)
        assert c.sum() == 11367

    def test_apply_chords_without_trimming(self):
        c = ps.filters.apply_chords(im=self.im, trim_edges=False)
        assert c.sum() == 31215


if __name__ == '__main__':
    t = FilterTest()
    t.setup_class()
    t.test_apply_chords_axis0()
    t.test_apply_chords_axis1()
    t.test_apply_chords_axis2()
    t.test_apply_chords_with_spacing()
    t.test_apply_chords_without_trimming()
