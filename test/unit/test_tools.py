import porespy as ps
import scipy as sp
import scipy.ndimage as spim
import matplotlib.pyplot as plt
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

    def test_fft_dilate_2d(self):
        sp.random.seed(1)
        im = ps.generators.blobs(shape=[100, 100])
        r = 2
        strel = disk(r)
        im_d = ps.tools.fft_dilate(im, strel)
        diff = im_d*~im
        dt = spim.distance_transform_edt(diff.astype(int))**2
        assert sp.amax(dt) <= r**2

    def test_fft_dilate_3d(self):
        sp.random.seed(1)
        im = ps.generators.blobs(shape=[100, 100, 100])
        r = 2
        strel = ball(r)
        im_d = ps.tools.fft_dilate(im, strel)
        diff = im_d*~im
        dt = spim.distance_transform_edt(diff.astype(int))**2
        assert sp.amax(dt) <= r**2


if __name__ == '__main__':
    t = ToolsTest()
    t.setup_class()
    t.test_randomize_colors()
    t.test_make_contiguous_size()
    t.test_make_contiguous_contiguity()
    t.test_get_slice()
    t.test_find_outer_region()
    t.test_extract_subsection()
    t.test_fft_dilate_2d()
    t.test_fft_dilate_3d()
