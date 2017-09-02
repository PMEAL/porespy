import porespy as ps
import scipy as sp


class MetricsTest():

    def setup_class(self):
        sp.random.seed(0)
        im2D = ps.generators.circle_pack(shape=[100, 100], radius=5, offset=2)
        im2D_big = ps.generators.circle_pack(shape=[500, 500],
                                             radius=10,
                                             offset=10,
                                             packing='square')
        self.im2D = im2D
        self.im2D_big = im2D_big

    def test_porosity(self):
        phi = ps.metrics.porosity(im=self.im2D)
        assert phi == 0.6619

    def test_tpcf_fft(self):
        tpcf_fft_1 = ps.metrics.two_point_correlation_fft(self.im2D)
        tpcf_fft_2 = ps.metrics.two_point_correlation_fft(self.im2D_big)
        # autocorrelation fn should level off at around the porosity
        t = 0.2
        phi1 = ps.metrics.porosity(im=self.im2D)
        assert sp.sqrt((sp.mean(tpcf_fft_1.probability[-5:]) - phi1)**2) < t
        phi2 = ps.metrics.porosity(im=self.im2D_big)
        assert sp.sqrt((sp.mean(tpcf_fft_2.probability[-5:]) - phi2)**2) < t
