import pytest
import numpy as np
from edt import edt
import porespy as ps
import scipy.ndimage as spim
from skimage.morphology import disk, ball, skeletonize
from skimage.util import random_noise
from scipy.stats import norm

ps.settings.tqdm['disable'] = True


@pytest.mark.skip(reason="Sometimes fails, probably due to randomness")
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
        from porespy import beta
        np.random.seed(1)
        im = ps.generators.blobs(shape=[100, 100, 100], porosity=0.7)
        res = beta.tortuosity_gdd(im=im, scale_factor=3)

        np.testing.assert_approx_equal(res.tau[0], 1.3940746215566113, significant=5)
        np.testing.assert_approx_equal(res.tau[1], 1.4540191053977147, significant=5)
        np.testing.assert_approx_equal(res.tau[2], 1.4319705063316652, significant=5)

    def test_gdd_dataframe(self):
        from porespy import beta
        np.random.seed(2)
        im = ps.generators.blobs(shape=[100, 100, 100], porosity=0.7)
        df = beta.chunks_to_dataframe(im=im, scale_factor=3)
        assert len(df.iloc[:, 0]) == 54
        assert df.columns[0] == 'Throat Number'
        assert df.columns[1] == 'Tortuosity'
        assert df.columns[2] == 'Diffusive Conductance'
        assert df.columns[3] == 'Porosity'

        np.testing.assert_array_almost_equal(np.array(df.iloc[:, 1]),
                                             np.array([1.329061, 1.288042, 1.411449,
                                                       1.273172, 1.46565,  1.294553,
                                                       1.553851, 1.299077, 1.417645,
                                                       1.332902, 1.365739, 1.37725,
                                                       1.408786, 1.279847, 1.365632,
                                                       1.31547,  1.425769, 1.417447,
                                                       1.399028, 1.262936, 1.311554,
                                                       1.447341, 1.504881, 1.196132,
                                                       1.508335, 1.273323, 1.361239,
                                                       1.334868, 1.443466, 1.328017,
                                                       1.564574, 1.264049, 1.504227,
                                                       1.471079, 1.366275, 1.349767,
                                                       1.473522, 1.34229,  1.258255,
                                                       1.266575, 1.488935, 1.260175,
                                                       1.471782, 1.295077, 1.463962,
                                                       1.494004, 1.551485, 1.363379,
                                                       1.474238, 1.311737, 1.483244,
                                                       1.287134, 1.735833, 1.38633],),
                                                       decimal=4)


if __name__ == '__main__':
    t = SimulationsTest()
    self = t
    t.setup_class()
    for item in t.__dir__():
        if item.startswith('test'):
            print(f'Running test: {item}')
            t.__getattribute__(item)()
