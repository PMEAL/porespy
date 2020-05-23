import porespy as ps
import numpy as np
import scipy.ndimage as spim
from skimage import io
import pytest
from pathlib import Path
import os
from numpy.testing import assert_allclose


class MetricsTest():

    def setup_class(self):
        np.random.seed(0)
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
        path = Path(os.path.realpath(__file__),
                    '../../../test/fixtures/partitioned_regions.tif')
        self.regions = np.array(io.imread(path))

    def test_porosity(self):
        phi = ps.metrics.porosity(im=self.im2D)
        assert phi == 0.6619

    def test_tpcf_fft_2d(self):
        tpcf_fft_1 = ps.metrics.two_point_correlation_fft(self.im2D)
        tpcf_fft_2 = ps.metrics.two_point_correlation_fft(self.im2D_big)
        # autocorrelation fn should level off at around the porosity
        t = 0.2
        phi1 = ps.metrics.porosity(im=self.im2D)
        assert np.sqrt((np.mean(tpcf_fft_1.probability[-5:]) - phi1)**2) < t
        phi2 = ps.metrics.porosity(im=self.im2D_big)
        assert np.sqrt((np.mean(tpcf_fft_2.probability[-5:]) - phi2)**2) < t

    def test_tpcf_fft_3d(self):
        tpcf_fft = ps.metrics.two_point_correlation_fft(self.im3D)
        t = 0.2
        phi1 = ps.metrics.porosity(im=self.im3D)
        assert np.sqrt((np.mean(tpcf_fft.probability[-5:]) - phi1)**2) < t

    def test_pore_size_distribution(self):
        mip = ps.filters.porosimetry(self.im3D)
        psd = ps.metrics.pore_size_distribution(mip)
        assert np.sum(psd.satn) == 1.0

    def test_two_point_correlation_bf(self):
        tpcf_bf = ps.metrics.two_point_correlation_bf(self.im2D)
        # autocorrelation fn should level off at around the porosity
        t = 0.2
        phi1 = ps.metrics.porosity(im=self.im2D)
        assert np.sqrt((np.mean(tpcf_bf.probability[-5:]) - phi1)**2) < t

    def test_rev(self):
        rev = ps.metrics.representative_elementary_volume(self.blobs)
        assert (np.mean(rev.porosity) - 0.5)**2 < 0.05

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
        im = ps.generators.lattice_spheres(shape=[999, 999],
                                           radius=15, offset=4)
        p = ps.metrics.porosity_profile(im, axis=0)
        assert p.max() == 100
        assert_allclose(p.min(), 24.524524524524523)

    def test_porosity_profile_ndim_check(self):
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

    def test_chord_counts(self):
        im = np.ones([100, 50])
        crds = ps.filters.apply_chords(im, spacing=1, trim_edges=False)
        c = ps.metrics.chord_counts(crds)
        assert np.all(c == 100)
        crds = ps.filters.apply_chords(im, spacing=1, trim_edges=False, axis=1)
        c = ps.metrics.chord_counts(crds)
        assert np.all(c == 50)

    def test_mesh_surface_area(self):
        region = self.regions == self.regions.max()
        mesh = ps.tools.mesh_region(region)
        a = ps.metrics.mesh_surface_area(mesh)
        assert np.around(a, decimals=2) == 258.3
        b = ps.metrics.mesh_surface_area(verts=mesh.verts, faces=mesh.faces)
        assert np.around(b, decimals=2) == np.around(a, decimals=2)

    def test_region_surface_areas(self):
        regions = self.regions
        areas = ps.metrics.region_surface_areas(regions)
        assert not np.any(np.isnan(areas))

    def test_region_interface_areas(self):
        regions = self.regions
        areas = ps.metrics.region_surface_areas(regions)
        ia = ps.metrics.region_interface_areas(regions, areas)
        assert np.all(ia.conns[0] == [2, 19])
        assert np.around(ia.area[0], decimals=2) == 3.59

    def test_phase_fraction(self):
        im = np.reshape(np.random.randint(0, 10, 1000), [10, 10, 10])
        labels = np.unique(im, return_counts=True)[1]
        counts = ps.metrics.phase_fraction(im, normed=False)
        assert np.all(labels == counts)
        fractions = ps.metrics.phase_fraction(im, normed=True)
        assert fractions.sum() == 1
        assert np.allclose(fractions, counts/counts.sum())
        with pytest.raises(Exception):
            ps.metrics.phase_fraction(numpy.random.rand(10, 10, 10), normed=True)

    def test_representative_elementary_volume(self):
        im = ps.generators.lattice_spheres(shape=[999, 999],
                                           radius=15, offset=4)
        rev = ps.metrics.representative_elementary_volume(im)
        assert_allclose(np.average(rev.porosity), im.sum()/im.size, rtol=1e-1)

        im = ps.generators.lattice_spheres(shape=[151, 151, 151],
                                           radius=9, offset=4)
        rev = ps.metrics.representative_elementary_volume(im)
        assert_allclose(np.average(rev.porosity), im.sum()/im.size, rtol=1e-1)


if __name__ == '__main__':
    t = MetricsTest()
    self = t
    t.setup_class()
    for item in t.__dir__():
        if item.startswith('test'):
            print('running test: '+item)
            t.__getattribute__(item)()
