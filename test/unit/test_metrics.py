import porespy as ps
import scipy as sp
import numpy as np


class MetricsTest():

    def setup_class(self):
        sp.random.seed(0)
        im2D = ps.generators.lattice_spheres(shape=[100, 100],
                                             radius=5, offset=2,
                                             lattice='square')
        im2D_big = ps.generators.lattice_spheres(shape=[500, 500],
                                                 radius=10, offset=10,
                                                 lattice='square')
        self.im2D = im2D
        self.im2D_big = im2D_big
        self.im3D = ps.generators.lattice_spheres(shape=[51, 51, 51],
                                                  radius=4, offset=2,
                                                  lattice='cubic')
        self.blobs = ps.generators.blobs(shape=[101, 101, 101], porosity=0.5,
                                         blobiness=[1, 2, 3])

    def test_porosity(self):
        phi = ps.metrics.porosity(im=self.im2D)
        assert phi == 0.6619

    def test_tpcf_fft_2d(self):
        tpcf_fft_1 = ps.metrics.two_point_correlation_fft(self.im2D)
        tpcf_fft_2 = ps.metrics.two_point_correlation_fft(self.im2D_big)
        # autocorrelation fn should level off at around the porosity
        t = 0.2
        phi1 = ps.metrics.porosity(im=self.im2D)
        assert sp.sqrt((sp.mean(tpcf_fft_1.probability[-5:]) - phi1)**2) < t
        phi2 = ps.metrics.porosity(im=self.im2D_big)
        assert sp.sqrt((sp.mean(tpcf_fft_2.probability[-5:]) - phi2)**2) < t

    def test_tpcf_fft_3d(self):
        tpcf_fft = ps.metrics.two_point_correlation_fft(self.im3D)
        t = 0.2
        phi1 = ps.metrics.porosity(im=self.im3D)
        assert sp.sqrt((sp.mean(tpcf_fft.probability[-5:]) - phi1)**2) < t

    def test_pore_size_distribution(self):
        mip = ps.filters.porosimetry(self.im3D)
        psd = ps.metrics.pore_size_distribution(mip)
        assert sp.sum(psd.satn) == 1.0

    def test_two_point_correlation_bf(self):
        tpcf_bf = ps.metrics.two_point_correlation_bf(self.im2D)
        # autocorrelation fn should level off at around the porosity
        t = 0.2
        phi1 = ps.metrics.porosity(im=self.im2D)
        assert sp.sqrt((sp.mean(tpcf_bf.probability[-5:]) - phi1)**2) < t

    def test_rev(self):
        rev = ps.metrics.representative_elementary_volume(self.blobs)
        assert (sp.mean(rev.porosity) - 0.5)**2 < 0.05

    def test_radial_density(self):
        den = ps.metrics.radial_density(self.blobs)
        assert den.F.max() == 1


if __name__ == '__main__':
    t = MetricsTest()
    self = t
    t.setup_class()
    for item in t.__dir__():
        if item.startswith('test'):
            print('running test: '+item)
            t.__getattribute__(item)()
