import pytest
import numpy as np
from edt import edt
import porespy as ps
import scipy.ndimage as spim
from skimage.morphology import disk, ball, skeletonize_3d


class FilterTest():
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
        mx_parallel_1 = ps.filters.find_peaks(dt=dt, parallel=True)
        assert np.all(mx_serial == mx_parallel_1)

    def test_find_peaks_2D_with_args(self):
        im = ps.generators.blobs(shape=[200, 200], blobiness=2)
        dt = edt(im)
        mx_serial = ps.filters.find_peaks(dt=dt)
        mx_parallel_1 = ps.filters.find_peaks(dt=dt, parallel=True,
                                              cores=4, divs=2)
        assert np.all(mx_serial == mx_parallel_1)

    def test_porosimetry(self):
        im = ps.generators.blobs(shape=[100, 100, 100], blobiness=2)
        mio_serial = ps.filters.porosimetry(im, mode='mio', parallel=False)
        mio_parallel = ps.filters.porosimetry(im, mode='mio', parallel=True)
        assert np.all(mio_serial == mio_parallel)


if __name__ == '__main__':
    t = FilterTest()
    self = t
    t.setup_class()
    for item in t.__dir__():
        if item.startswith('test'):
            print('running test: '+item)
            t.__getattribute__(item)()
