import porespy as ps
import scipy as sp
import pytest
import scipy.ndimage as spim
import matplotlib.pyplot as plt
plt.close('all')


class GeneratorTest():

    def setup_class(self):
        sp.random.seed(10)

    def test_cylinders(self):
        X = 100
        Y = 100
        # Fibers don't work in 2D
        with pytest.raises(Exception):
            im = ps.generators.cylinders(shape=[X, Y], radius=4, nfibers=20)
        # But this works
        im = ps.generators.cylinders(shape=[1, X, Y], radius=1, nfibers=20)
        assert im.dtype == bool
        assert sp.shape(im.squeeze()) == (X, Y)
        im = ps.generators.cylinders(shape=[50, 50, 50], radius=1, nfibers=20)
        assert sp.shape(im.squeeze()) == (50, 50, 50)

    def test_insert_shape(self):
        im = sp.zeros([11, 11])
        shape = sp.ones([3, 3])
        im = ps.generators.insert_shape(im, [5, 5], shape)
        assert sp.sum(im) == 9

    def test_bundle_of_tubes(self):
        im = ps.generators.bundle_of_tubes(shape=[101, 101, 1], spacing=10)
        labels, N = spim.label(input=im)
        assert N == 100

    def test_overlapping_spheres_2d(self):
        phis = sp.arange(0.1, 0.9, 0.2)
        for phi in phis:
            im = ps.generators.overlapping_spheres(shape=[101, 101], radius=5,
                                                   porosity=phi)
            phi_actual = im.sum() / sp.size(im)
            assert abs(phi_actual - phi) < 0.02

    def test_overlapping_spheres_3d(self):
        phis = sp.arange(0.1, 0.9, 0.2)
        for phi in phis:
            im = ps.generators.overlapping_spheres(shape=[100, 100, 50],
                                                   radius=8, porosity=phi)
            phi_actual = im.sum() / sp.size(im)
            assert abs(phi_actual - phi) < 0.02

    def test_polydisperse_spheres(self):
        phis = sp.arange(0.1, 0.9, 0.2)
        dist = sp.stats.norm(loc=7, scale=2)
        for phi in phis:
            im = ps.generators.polydisperse_spheres(shape=[100, 100, 50],
                                                    porosity=phi, dist=dist,
                                                    nbins=10)
            phi_actual = im.sum() / sp.size(im)
            assert abs(phi_actual - phi) < 0.05

    def test_voronoi_edges(self):
        sp.random.seed(0)
        im = ps.generators.voronoi_edges(shape=[50, 50, 50],
                                         radius=2,
                                         ncells=25,
                                         flat_faces=True)
        top_slice = im[:, :, 0]
        assert sp.sum(top_slice) == 1409

    def test_lattice_spheres_square(self):
        im = ps.generators.lattice_spheres(shape=[101, 101], radius=5,
                                           offset=0, lattice='sc')
        labels, N = spim.label(input=~im)
        assert N == 100

    def test_lattice_spheres_triangular(self):
        im = ps.generators.lattice_spheres(shape=[101, 101], radius=5,
                                           lattice='triangular')
        labels, N = spim.label(input=~im)
        assert N == 85

    def test_lattice_spheres_sc(self):
        im = ps.generators.lattice_spheres(shape=[101, 101, 101],
                                           radius=4, offset=1,
                                           lattice='sc')
        labels, N = spim.label(input=~im)
        assert N == 1000

    def test_lattice_spheres_fcc(self):
        im = ps.generators.lattice_spheres(shape=[101, 101, 101],
                                           radius=4, offset=2,
                                           lattice='fcc')
        labels, N = spim.label(input=~im)
        assert N == 392

    def test_lattice_spheres_bcc(self):
        im = ps.generators.lattice_spheres(shape=[101, 101, 101],
                                           radius=4, offset=2,
                                           lattice='bcc')
        labels, N = spim.label(input=~im)
        assert N == 1024

    def test_noise_simplex(self):
        pass

    def test_noise_perlin(self):
        pass

    def test_blobs_1d_shape(self):
        im = ps.generators.blobs(shape=[101])
        assert len(list(im.shape)) == 3

    def test_RSA_2d_single(self):
        sp.random.seed(0)
        im = sp.zeros([100, 100], dtype=int)
        im = ps.generators.RSA(im, radius=10, volume_fraction=0.5)
        assert sp.sum(im > 0) == 5095
        assert sp.sum(im > 1) == 20

    def test_RSA_2d_multi(self):
        sp.random.seed(0)
        im = sp.zeros([100, 100], dtype=int)
        im = ps.generators.RSA(im, radius=10, volume_fraction=0.5)
        im = ps.generators.RSA(im, radius=5, volume_fraction=0.75)
        assert sp.sum(im > 0) == 6520
        assert sp.sum(im > 1) == 44

    def test_RSA_3d_single(self):
        sp.random.seed(0)
        im = sp.zeros([50, 50, 50], dtype=int)
        im = ps.generators.RSA(im, radius=5, volume_fraction=0.5)
        assert sp.sum(im > 0) == 45602
        assert sp.sum(im > 1) == 121

    def test_RSA_mask_edge_2d(self):
        im = sp.zeros([100, 100], dtype=int)
        im = ps.generators.RSA(im, radius=10, volume_fraction=0.5,
                               mode='contained')
        coords = sp.argwhere(im == 2)
        assert ~sp.any(coords < 10)
        assert ~sp.any(coords > 90)

    def test_RSA_mask_edge_3d(self):
        im = sp.zeros([50, 50, 50], dtype=int)
        im = ps.generators.RSA(im, radius=5, volume_fraction=0.5,
                               mode='contained')
        coords = sp.argwhere(im == 2)
        assert ~sp.any(coords < 5)
        assert ~sp.any(coords > 45)


if __name__ == '__main__':
    t = GeneratorTest()
    self = t
    t.setup_class()
    for item in t.__dir__():
        if item.startswith('test'):
            print('running test: '+item)
            t.__getattribute__(item)()
