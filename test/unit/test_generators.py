import porespy as ps
import numpy as np
import scipy as sp
import pytest
import scipy.ndimage as spim
import matplotlib.pyplot as plt
plt.close('all')


class GeneratorTest():

    def setup_class(self):
        np.random.seed(10)

    def test_cylinders(self):
        X = 100
        Y = 100
        # Fibers don't work in 2D
        with pytest.raises(Exception):
            im = ps.generators.cylinders(shape=[X, Y], radius=4, ncylinders=20)
        # But this works
        im = ps.generators.cylinders(shape=[1, X, Y], radius=1, ncylinders=20)
        assert im.dtype == bool
        assert np.shape(im.squeeze()) == (X, Y)
        im = ps.generators.cylinders(shape=[50, 50, 50], radius=1,
                                     ncylinders=20)
        assert np.shape(im.squeeze()) == (50, 50, 50)

    def test_insert_shape_center_defaults(self):
        im = np.zeros([11, 11])
        shape = np.ones([3, 3])
        im = ps.generators.insert_shape(im, element=shape, center=[5, 5])
        assert np.sum(im) == np.prod(shape.shape)

        im = np.zeros([11, 11])
        shape = np.ones([4, 4])
        with pytest.raises(Exception):
            im = ps.generators.insert_shape(im, element=shape, center=[5, 5])

    def test_insert_shape_center_overlay(self):
        im = np.ones([10, 10])
        shape = np.ones([3, 3])
        im = ps.generators.insert_shape(im, element=shape, center=[5, 5],
                                        value=1.0, mode='overlay')
        assert np.sum(im) == (np.prod(im.shape) + np.prod(shape.shape))

    def test_insert_shape_corner_overwrite(self):
        im = np.ones([10, 10])
        shape = np.ones([3, 3])
        im = ps.generators.insert_shape(im, element=shape, corner=[5, 5],
                                        value=1.0, mode='overlay')
        assert np.sum(im) == (np.prod(im.shape) + np.prod(shape.shape))
        assert im[5, 5] == 2
        assert im[4, 5] == 1 and im[5, 4] == 1

    def test_insert_shape_center_outside_im(self):
        im = np.zeros([11, 11])
        shape = np.ones([3, 3])
        im = ps.generators.insert_shape(im, element=shape, center=[-1, -1])
        assert np.sum(im) == 1

        im = np.zeros([11, 11])
        shape = np.ones([3, 3])
        im = ps.generators.insert_shape(im, element=shape, center=[0, -1])
        assert np.sum(im) == 2

        im = np.zeros([11, 11])
        shape = np.ones([3, 3])
        im = ps.generators.insert_shape(im, element=shape, center=[10, 10])
        assert np.sum(im) == 4

        im = np.zeros([11, 11])
        shape = np.ones([3, 3])
        im = ps.generators.insert_shape(im, element=shape, center=[14, 14])
        assert np.sum(im) == 0

        im = np.zeros([11, 11])
        shape = np.ones([4, 4])
        with pytest.raises(Exception):
            im = ps.generators.insert_shape(im, element=shape, center=[10, 10])

        im = np.zeros([11, 11])
        shape = np.ones([4, 3])
        with pytest.raises(Exception):
            im = ps.generators.insert_shape(im, element=shape, center=[10, 10])

    def test_insert_shape_corner_outside_im(self):
        im = np.zeros([11, 11])
        shape = np.ones([3, 3])
        im = ps.generators.insert_shape(im, element=shape, corner=[-1, -1])
        assert np.sum(im) == 4

        im = np.zeros([11, 11])
        shape = np.ones([3, 3])
        im = ps.generators.insert_shape(im, element=shape, corner=[-1, 1])
        assert np.sum(im) == 6

        im = np.zeros([11, 11])
        shape = np.ones([3, 3])
        im = ps.generators.insert_shape(im, element=shape, corner=[-3, -3])
        assert np.sum(im) == 0

        im = np.zeros([11, 11])
        shape = np.ones([3, 3])
        im = ps.generators.insert_shape(im, element=shape, corner=[10, 9])
        assert np.sum(im) == 2

        im = np.zeros([11, 11])
        shape = np.ones([3, 3])
        im = ps.generators.insert_shape(im, element=shape, corner=[13, 13])
        assert np.sum(im) == 0

        im = np.zeros([11, 11])
        shape = np.ones([3, 4])
        im = ps.generators.insert_shape(im, element=shape, corner=[9, 9])
        assert np.sum(im) == 4

        im = np.zeros([11, 11])
        shape = np.ones([3, 4])
        im = ps.generators.insert_shape(im, element=shape, corner=[0, -1])
        assert np.sum(im) == 9

    def test_bundle_of_tubes(self):
        im = ps.generators.bundle_of_tubes(shape=[101, 101, 1], spacing=10)
        labels, N = spim.label(input=im)
        assert N == 100

    def test_overlapping_spheres_2d(self):
        phis = np.arange(0.1, 0.9, 0.2)
        for phi in phis:
            im = ps.generators.overlapping_spheres(shape=[101, 101],
                                                   radius=5,
                                                   porosity=phi)
            phi_actual = im.sum() / np.size(im)
            assert abs(phi_actual - phi) < 0.02

    def test_overlapping_spheres_3d(self):
        phis = np.arange(0.1, 0.9, 0.2)
        for phi in phis:
            im = ps.generators.overlapping_spheres(shape=[100, 100, 50],
                                                   radius=8, porosity=phi)
            phi_actual = im.sum() / np.size(im)
            assert abs(phi_actual - phi) < 0.02

    def test_polydisperse_spheres(self):
        phis = np.arange(0.1, 0.9, 0.2)
        dist = sp.stats.norm(loc=7, scale=2)
        for phi in phis:
            im = ps.generators.polydisperse_spheres(shape=[100, 100, 50],
                                                    porosity=phi, dist=dist,
                                                    nbins=10)
            phi_actual = im.sum() / np.size(im)
            assert abs(phi_actual - phi) < 0.1

    def test_voronoi_edges(self):
        np.random.seed(0)
        im = ps.generators.voronoi_edges(shape=[50, 50, 50],
                                         radius=2,
                                         ncells=25,
                                         flat_faces=True)
        top_slice = im[:, :, 0]
        assert np.sum(top_slice) == 1409

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
        np.random.seed(0)
        im = np.zeros([100, 100], dtype=int)
        im = ps.generators.RSA(im, radius=10, volume_fraction=0.5)
        assert np.sum(im > 0) == 5095
        assert np.sum(im > 1) == 20

    def test_RSA_2d_multi(self):
        np.random.seed(0)
        im = np.zeros([100, 100], dtype=int)
        im = ps.generators.RSA(im, radius=10, volume_fraction=0.5)
        im = ps.generators.RSA(im, radius=5, volume_fraction=0.75)
        assert np.sum(im > 0) == 6520
        assert np.sum(im > 1) == 44

    def test_RSA_3d_single(self):
        np.random.seed(0)
        im = np.zeros([50, 50, 50], dtype=int)
        im = ps.generators.RSA(im, radius=5, volume_fraction=0.5)
        assert np.sum(im > 0) == 45602
        assert np.sum(im > 1) == 121

    def test_RSA_mask_edge_2d(self):
        im = np.zeros([100, 100], dtype=int)
        im = ps.generators.RSA(im, radius=10, volume_fraction=0.5,
                               mode='contained')
        coords = np.argwhere(im == 2)
        assert ~np.any(coords < 10)
        assert ~np.any(coords > 90)

    def test_RSA_mask_edge_3d(self):
        im = np.zeros([50, 50, 50], dtype=int)
        im = ps.generators.RSA(im, radius=5, volume_fraction=0.5,
                               mode='contained')
        coords = np.argwhere(im == 2)
        assert ~np.any(coords < 5)
        assert ~np.any(coords > 45)

    def test_line_segment(self):
        X0 = [3, 4]
        X1 = [5, 9]
        L1, L2 = ps.generators.line_segment(X0, X1)
        assert np.all(L1 == [3, 3, 4, 4, 5, 5])
        assert np.all(L2 == [4, 5, 6, 7, 8, 9])

        X0 = [3, 4, 5]
        X1 = [5, 9, 13]
        L1, L2, L3 = ps.generators.line_segment(X0, X1)
        assert np.all(L1 == [3, 3, 4, 4, 4, 4, 4, 5, 5])
        assert np.all(L2 == [4, 5, 5, 6, 6, 7, 8, 8, 9])
        assert np.all(L3 == [5, 6, 7, 8, 9, 10, 11, 12, 13])


if __name__ == '__main__':
    t = GeneratorTest()
    self = t
    t.setup_class()
    for item in t.__dir__():
        if item.startswith('test'):
            print('running test: '+item)
            t.__getattribute__(item)()
