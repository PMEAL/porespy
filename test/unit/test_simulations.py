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

    def test_gdd(self):
        np.random.seed(1)
        im = ps.generators.blobs(shape=[100, 100, 100], porosity=0.7)
        res = ps.simulations.tortuosity_gdd(im=im, scale_factor=3)

        np.testing.assert_approx_equal(res[0], 1.3939444950116722, significant=5)
        np.testing.assert_approx_equal(res[1], 1.420361317605694, significant=5)
        np.testing.assert_approx_equal(res[2], 1.3962838936596436, significant=5)

    def test_gdd_dataframe(self):
        np.random.seed(2)
        im = ps.generators.blobs(shape=[100, 100, 100], porosity=0.7)
        df = ps.simulations.chunks_to_dataframe(im=im, scale_factor=3)
        assert len(df.iloc[:, 0]) == 54
        assert df.columns[0] == 'Throat Number'
        assert df.columns[1] == 'Tortuosity'
        assert df.columns[2] == 'Diffusive Conductance'
        assert df.columns[3] == 'Porosity'

        np.testing.assert_array_almost_equal(np.array(df.iloc[:, 1]),
                                             np.array([1.326868, 1.283062, 1.371618,
                                                       1.334747, 1.46832,  1.415894,
                                                       1.756516, 1.512354, 1.369171,
                                                       1.394996, 1.576798, 1.386702,
                                                       1.390045, 1.331828, 1.364359,
                                                       1.406702, 1.428381, 1.497239,
                                                       1.209865, 1.248376, 1.333118,
                                                       1.395648, 1.447592, 1.260381,
                                                       1.571421, 1.348176, 1.362535,
                                                       1.292804, 1.468329, 1.40084,
                                                       1.409297, 1.268648, 1.552551,
                                                       1.435069, 1.330031, 1.460921,
                                                       1.473522, 1.34229,  1.258255,
                                                       1.266575, 1.488935, 1.260175,
                                                       1.471782, 1.295077, 1.463962,
                                                       1.494004, 1.551485, 1.363379,
                                                       1.474238, 1.311737, 1.483244,
                                                       1.287134, 1.735833, 1.38633]),
                                                       decimal=4)


if __name__ == '__main__':
    t = SimulationsTest()
    self = t
    t.setup_class()
    for item in t.__dir__():
        if item.startswith('test'):
            print(f'Running test: {item}')
            t.__getattribute__(item)()
