import porespy as ps
import scipy as sp
import scipy.ndimage as spim
import imageio
import pytest


class MetricsTest():

    def setup_class(self):
        sp.random.seed(0)
        self.im2D = ps.generators.lattice_spheres(shape=[100, 100],
                                                  radius=5, offset=2,
                                                  lattice='square')
        self.im2D_big = ps.generators.lattice_spheres(shape=[500, 500],
                                                      radius=10, offset=10,
                                                      lattice='square')
        self.im3D = ps.generators.lattice_spheres(shape=[51, 51, 51],
                                                  radius=4, offset=2,
                                                  lattice='cubic')
        self.blobs = ps.generators.blobs(shape=[101, 101, 101], porosity=0.5,
                                         blobiness=[1, 2, 3])
        self.regions = imageio.mimread('../fixtures/partitioned_regions.tif')

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
        assert den.cdf.max() == 1

    def test_props_to_DataFrame(self):
        label = spim.label(self.im2D)[0]
        rp = ps.metrics.regionprops_3D(label)
        ps.metrics.props_to_DataFrame(rp)

    def test_props_to_image(self):
        label = spim.label(self.im2D)[0]
        rp = ps.metrics.regionprops_3D(label)
        ps.metrics.props_to_image(rp, self.im2D.shape, 'solidity')

    def test_porosity_profile(self):
        ps.metrics.porosity_profile(self.im2D, axis=0)
        ps.metrics.porosity_profile(self.im2D, axis=1)
        with pytest.raises(Exception):
            ps.metrics.porosity_profile(self.im2D, axis=2)

    def test_linear_density(self):
        im = ps.filters.distance_transform_lin(self.im2D, axis=0, mode='both')
        ps.metrics.linear_density(im)

    def test_chord_length_distribution_2D(self):
        chords = ps.filters.apply_chords(self.im2D)
        ps.metrics.chord_length_distribution(chords, normalization='length')

    def test_chord_length_distribution_3D(self):
        chords = ps.filters.apply_chords(self.im3D)
        ps.metrics.chord_length_distribution(chords, normalization='length')

    def test_mesh_surface_area(self):
        region = self.regions == 1
        mesh = ps.tools.mesh_region(region)
        a = ps.metrics.mesh_surface_area(mesh)
        assert sp.around(a, decimals=2) == 968.06
        b = ps.metrics.mesh_surface_area(verts=mesh.verts, faces=mesh.faces)
        assert sp.around(b, decimals=2) == sp.around(a, decimals=2)

    def test_region_surface_areas(self):
        regions = self.regions
        areas = ps.metrics.region_surface_areas(regions)
        assert not sp.any(sp.isnan(areas))

    def test_region_interface_areas(self):
        regions = self.regions
        areas = ps.metrics.region_surface_areas(regions)
        ia = ps.metrics.region_interface_areas(regions, areas)
        assert sp.all(ia.conns[0] == [0, 17])
        assert sp.around(ia.area[0], decimals=2) == 175.27


if __name__ == '__main__':
    t = MetricsTest()
    self = t
    t.setup_class()
    for item in t.__dir__():
        if item.startswith('test'):
            print('running test: '+item)
            t.__getattribute__(item)()
