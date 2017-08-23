import porespy as ps
import scipy as sp
import scipy.ndimage as spim


class BinarizationTest():
    def setup_class(self):
        pass

    def test_simple_otsu(self):
        im = ps.generators.blobs(shape=[300, 300])
        im = ps.generators.add_noise(im)
        im = spim.gaussian_filter(input=im, sigma=1)
        im = ps.binarization.simple_otsu(im=im)
        assert im.dtype == bool
