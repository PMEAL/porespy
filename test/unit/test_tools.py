import porespy as ps
import scipy as sp
import scipy.ndimage as spim
import matplotlib.pyplot as plt
import pytest
from skimage.morphology import disk, ball
plt.close('all')


class ToolsTest():
    def setup_class(self):
        self.im = sp.random.randint(0, 10, 20)
        sp.random.seed(0)
        self.blobs = ps.generators.blobs(shape=[101, 101])
        self.labels, N = spim.label(input=self.blobs)

    def test_randomize_colors(self):
        randomized_im = ps.tools.randomize_colors(im=self.im)
        assert sp.unique(self.im).size == sp.unique(randomized_im).size
        assert sp.all(sp.unique(self.im) == sp.unique(randomized_im))

    def test_make_contiguous_size(self):
        cont_im = ps.tools.make_contiguous(self.im)
        assert sp.unique(self.im).size == sp.unique(cont_im).size

    def test_make_contiguous_contiguity(self):
        cont_im = ps.tools.make_contiguous(self.im)
        assert sp.all(sp.arange(sp.unique(self.im).size) == sp.unique(cont_im))

    def test_get_slice(self):
        one_lab = self.labels == 10
        my_slice = ps.tools.get_slice(one_lab, center=[75, 9], size=10)
        assert sp.sum(one_lab) == sp.sum(one_lab[my_slice])

    def test_find_outer_region(self):
        one_lab = self.labels == 10
        my_slice = ps.tools.get_slice(one_lab, center=[75, 9], size=10)
        small_slice = one_lab[my_slice]
        outer = ps.tools.find_outer_region(small_slice)
        assert sp.sum(outer) == sp.sum(small_slice[:, 0])

    def test_extract_subsection(self):
        sec = ps.tools.extract_subsection(self.blobs, [0.5])
        assert sp.all(sp.array(sp.shape(sec)) == 50)

    def test_extract_cylinder(self):
        cyl = ps.tools.extract_cylinder(self.im3D)

    def test_bbox_to_slices(self):
        s = ps.tools.bbox_to_slices([0, 0, 0, 10, 10, 10])
        assert sp.all(self.im3D[s].shape == (10, 10, 10))

    def test_get_slices(self):
        x, y, z = ps.tools.get_planes(self.im3D)
        assert sp.all(x.shape == (51, 51))
        assert sp.all(y.shape == (51, 51))
        assert sp.all(z.shape == (51, 51))
        with pytest.raises(ValueError):
            ps.tools.get_planes(self.im2D)


if __name__ == '__main__':
    t = ToolsTest()
    self = t
    t.setup_class()
    for item in t.__dir__():
        if item.startswith('test'):
            print('running test: '+item)
            t.__getattribute__(item)()
