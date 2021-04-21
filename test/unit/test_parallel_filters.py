import numpy as np
import pytest
import scipy.ndimage as spim
from edt import edt
from skimage.morphology import ball, disk, skeletonize_3d

import porespy as ps


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

    def test_local_thickness(self):
        im = ps.generators.blobs(shape=[100, 100, 100], blobiness=2)
        lt_serial = ps.filters.local_thickness(im, mode='mio', parallel=False)
        lt_parallel = ps.filters.local_thickness(im, mode='mio', parallel=True)
        assert np.all(lt_serial == lt_parallel)

    def test_prune_branches(self):
        im = ps.generators.lattice_spheres(shape=[100, 100, 100], r=4)
        skel1 = skeletonize_3d(im)
        skel2 = ps.filters.prune_branches(skel1, parallel=False)
        skel3 = ps.filters.prune_branches(skel1, parallel=True)
        assert np.all(skel2 == skel3)

    def test_prune_branches_w_iterations(self):
        im = ps.generators.lattice_spheres(shape=[100, 100, 100], r=4)
        skel1 = skeletonize_3d(im)
        skel2 = ps.filters.prune_branches(skel1, parallel=False, ierations=5)
        skel3 = ps.filters.prune_branches(skel1, parallel=True, iterations=5)
        assert np.all(skel2 == skel3)

    def test_blobs(self):
        np.random.seed(0)
        im1 = ps.generators.blobs(shape=[100, 100, 100])
        np.random.seed(0)
        im2 = ps.generators.blobs(shape=[100, 100, 100], parallel=True)
        assert np.all(im1 == im2)


if __name__ == '__main__':
    t = ParallelTest()
    self = t
    t.setup_class()
    for item in t.__dir__():
        if item.startswith('test'):
            print('running test: '+item)
            t.__getattribute__(item)()
