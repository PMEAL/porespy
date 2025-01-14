import numpy as np
from edt import edt
import porespy as ps
import scipy.ndimage as spim
from skimage.morphology import skeletonize
ps.settings.loglevel = "CRITICAL"
ps.settings.tqdm['disable'] = True


class ParallelTest():
    def setup_class(self):
        np.random.seed(0)
        self.im = ps.generators.blobs(shape=[100, 100, 100], blobiness=2)
        # Ensure that im was generated as expeccted
        assert ps.metrics.porosity(self.im) == 0.499829
        self.im_dt = edt(self.im)

    def test_find_peaks_2D(self):
        im = ps.generators.blobs(shape=[200, 200], blobiness=2)
        dt = edt(im)
        mx_serial = ps.filters.find_peaks(dt=dt)
        mx_parallel_1 = ps.filters.find_peaks(dt=dt, divs=2)
        assert np.all(mx_serial == mx_parallel_1)

    def test_find_peaks_3D(self):
        im = ps.generators.blobs(shape=[100, 100, 100], blobiness=2)
        dt = edt(im)
        mx_serial = ps.filters.find_peaks(dt=dt)
        mx_parallel_1 = ps.filters.find_peaks(dt=dt, divs=2)
        assert np.all(mx_serial == mx_parallel_1)

    def test_porosimetry(self):
        im = ps.generators.blobs(shape=[100, 100, 100], blobiness=2)
        mio_serial = ps.filters.porosimetry(im, mode='mio', divs=1)
        mio_parallel = ps.filters.porosimetry(im, mode='mio', divs=2)
        assert np.all(mio_serial == mio_parallel)

    def test_local_thickness(self):
        im = ps.generators.blobs(shape=[100, 100, 100], blobiness=2)
        lt_serial = ps.filters.local_thickness(im, mode='mio', divs=1)
        lt_parallel = ps.filters.local_thickness(im, mode='mio', divs=2)
        assert np.all(lt_serial == lt_parallel)

    def test_blobs_3D(self):
        np.random.seed(0)
        im1 = ps.generators.blobs(shape=[101, 101, 101], divs=1)
        np.random.seed(0)
        im2 = ps.generators.blobs(shape=[101, 101, 101], divs=2)
        assert np.all(im1 == im2)

    def test_blobs_2D(self):
        np.random.seed(0)
        s = 100
        im1 = ps.generators.blobs(shape=[s, s], divs=1, porosity=.5)
        np.random.seed(0)
        im2 = ps.generators.blobs(shape=[s, s], divs=2, porosity=.5)
        assert np.sum(im1 != im2) < 5


if __name__ == '__main__':
    t = ParallelTest()
    self = t
    t.setup_class()
    for item in t.__dir__():
        if item.startswith('test'):
            print('running test: '+item)
            t.__getattribute__(item)()
