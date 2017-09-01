import porespy as ps
import scipy as sp
import time


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

    def test_tpcf(self):
        # Brute Force
        t1 = time.time()
        tpcf_bf = ps.metrics.two_point_correlation_bf(self.im2D)
        t2 = time.time()
        # Fast Fourier Transform
        tpcf_fft = ps.metrics.two_point_correlation_fft(self.im2D)
        t3 = time.time()
        tpcf_bf2 = ps.metrics.two_point_correlation_bf(self.im2D_big)
        t4 = time.time()
        tpcf_fft2 = ps.metrics.two_point_correlation_fft(self.im2D_big)
        t5 = time.time()
        # BF is faster for small images
        # Although any improvements could fail this test
        assert (t2 - t1) < (t3 - t2)
        # FFT faster than BF for larger images
        assert (t5 - t4) < (t4 - t3)
        # autocorrelation fn should level off at around the porosity
        t = 0.2
        phi1 = ps.metrics.porosity(im=self.im2D)
        assert sp.sqrt((sp.mean(tpcf_bf.probability[-5:]) - phi1)**2) < t
        assert sp.sqrt((sp.mean(tpcf_fft.probability[-5:]) - phi1)**2) < t
        phi2 = ps.metrics.porosity(im=self.im2D_big)
        assert sp.sqrt((sp.mean(tpcf_bf2.probability[-5:]) - phi2)**2) < t
        assert sp.sqrt((sp.mean(tpcf_fft2.probability[-5:]) - phi2)**2) < t
