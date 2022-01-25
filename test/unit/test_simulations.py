# import pytest
import numpy as np
from edt import edt
import porespy as ps
import scipy.ndimage as spim
from skimage.morphology import disk, ball, skeletonize_3d
from skimage.util import random_noise
from scipy.stats import norm
ps.settings.tqdm['disable'] = True


class SimulationsTest():
    def setup_class(self):
        np.random.seed(0)
        self.im = ps.generators.blobs(shape=[100, 100, 100], blobiness=2)
        # Ensure that im was generated as expeccted
        assert ps.metrics.porosity(self.im) == 0.499829
        self.im_dt = edt(self.im)

    def test_drainage_with_gravity(self):
        np.random.seed(2)
        im = ps.generators.blobs(shape=[100, 100], porosity=0.7)
        dt = edt(im)
        pc = -2*0.072*np.cos(np.deg2rad(180))/dt
        np.testing.assert_approx_equal(pc[im].max(), 0.144)
        drn = ps.simulations.drainage(pc=pc, im=im, voxel_size=1e-5, g=9.81)
        np.testing.assert_approx_equal(drn.im_pc.max(), np.inf)
        drn2 = ps.simulations.drainage(pc=pc, im=im, voxel_size=1e-4, g=0)
        np.testing.assert_approx_equal(drn2.im_pc[im].max(), np.inf)
        im = ps.filters.fill_blind_pores(im)
        drn = ps.simulations.drainage(pc=pc, im=im, voxel_size=1e-5, g=9.81)
        np.testing.assert_approx_equal(drn.im_pc.max(), 10.04657972914)
        drn2 = ps.simulations.drainage(pc=pc, im=im, voxel_size=1e-4, g=0)
        np.testing.assert_approx_equal(drn2.im_pc[im].max(), 0.14622522289864)


if __name__ == '__main__':
    t = SimulationsTest()
    self = t
    t.setup_class()
    for item in t.__dir__():
        if item.startswith('test'):
            print(f'Running test: {item}')
            t.__getattribute__(item)()
