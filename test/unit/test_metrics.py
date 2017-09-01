import porespy as ps
import scipy as sp


class MetricsTest():

    def setup_class(self):
        sp.random.seed(0)
        im2D = ps.generators.circle_pack(shape=[100, 100], radius=5, offset=2)
        self.im2D = im2D

    def test_porosity(self):
        phi = ps.metrics.porosity(im=self.im2D)
        assert phi == 0.6619
