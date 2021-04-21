import numpy as np
import pytest
import scipy.ndimage as spim
import scipy.stats as spst
from numpy.testing import assert_allclose

import porespy as ps


class GeneratorTest():

    def setup_class(self):
        np.random.seed(10)

    def test_cylinders(self):
        # Testing cylinders with number of cylinders as input
        X = Y = 100
        # Fibers don't work in 2D
        with pytest.raises(Exception):
            im = ps.generators.cylinders(shape=[X, Y], r=4, ncylinders=20)
        # But this works
        im = ps.generators.cylinders(shape=[1, X, Y], r=1, ncylinders=20)
        assert im.dtype == bool
        assert np.shape(im.squeeze()) == (X, Y)
        im = ps.generators.cylinders(shape=[50, 50, 50], r=1, ncylinders=20)
        assert np.shape(im.squeeze()) == (50, 50, 50)
        # Now, testing cylinders with porosity as input
        # max_iter must at least be 3
        with pytest.raises(Exception):
            im = ps.generators.cylinders(
                shape=[50, 50, 50], r=4, porosity=0.5, max_iter=2)
        im = ps.generators.cylinders(
            shape=[50, 50, 50], r=3, porosity=0.5, max_iter=10)
        assert im.dtype == bool
        assert np.shape(im.squeeze()) == (50, 50, 50)
        porosity = im.sum() / im.size
        np.testing.assert_allclose(porosity, 0.5, rtol=1e-2)

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
            im = ps.generators.overlapping_spheres(
                shape=[101, 101], r=5, porosity=phi)
            phi_actual = im.sum() / np.size(im)
            assert abs(phi_actual - phi) < 0.02

    def test_overlapping_spheres_3d(self):
        phis = np.arange(0.1, 0.9, 0.2)
        for phi in phis:
            im = ps.generators.overlapping_spheres(
                shape=[100, 100, 50], r=8, porosity=phi)
            phi_actual = im.sum() / np.size(im)
            assert abs(phi_actual - phi) < 0.02

    def test_polydisperse_spheres(self):
        phis = np.arange(0.1, 0.5, 0.2)
        dist = spst.norm(loc=7, scale=2)
        for phi in phis:
            im = ps.generators.polydisperse_spheres(
                shape=[100, 100, 50], porosity=phi, dist=dist, nbins=10)
            phi_actual = im.sum() / np.size(im)
            assert abs(phi_actual - phi) < 0.1

    def test_voronoi_edges(self):
        np.random.seed(0)
        im = ps.generators.voronoi_edges(
            shape=[50, 50, 50], r=2, ncells=25, flat_faces=True)
        top_slice = im[:, :, 0]
        assert np.sum(top_slice) == 1409

    def test_lattice_spheres_square(self):
        im = ps.generators.lattice_spheres(
            shape=[101, 101], r=5, spacing=10, lattice='sc')
        labels, N = spim.label(input=~im)
        assert N == 100

    def test_lattice_spheres_triangular(self):
        im = ps.generators.lattice_spheres(
            shape=[101, 101], r=5, spacing=15, lattice='triangular')
        labels, N = spim.label(input=~im)
        assert N == 85

    def test_lattice_spheres_sc(self):
        im = ps.generators.lattice_spheres(
            shape=[101, 101, 101], r=4, spacing=10, lattice='sc')
        labels, N = spim.label(input=~im)
        assert N == 1000

    def test_lattice_spheres_fcc(self):
        im = ps.generators.lattice_spheres(
            shape=[101, 101, 101], r=4, spacing=12, lattice='fcc')
        labels, N = spim.label(input=~im)
        assert N == 2457

    def test_lattice_spheres_bcc(self):
        im = ps.generators.lattice_spheres(
            shape=[101, 101, 101], r=4, spacing=12, lattice='bcc')
        labels, N = spim.label(input=~im)
        assert N == 1241

    def test_blobs_1d_shape(self):
        im = ps.generators.blobs(shape=[101])
        assert len(list(im.shape)) == 3

    def test_RSA_2d_contained(self):
        im = np.zeros([100, 100], dtype=int)
        im = ps.generators.RSA(im, r=10, volume_fraction=0.5, mode='contained')
        im = np.pad(im, pad_width=1, mode='constant', constant_values=False)
        lt = ps.filters.local_thickness(im)
        assert len(np.unique(lt)) == 2

    def test_RSA_2d_extended(self):
        im = np.zeros([100, 100], dtype=int)
        im = ps.generators.RSA(im, r=10, volume_fraction=0.5, mode='extended')
        im = np.pad(im, pad_width=1, mode='constant', constant_values=False)
        lt = ps.filters.local_thickness(im)
        assert len(np.unique(lt)) > 2

    def test_RSA_3d_contained(self):
        im = np.zeros([100, 100, 100], dtype=int)
        im = ps.generators.RSA(im, r=10, volume_fraction=0.5, mode='contained')
        lt = ps.filters.local_thickness(im, sizes=[10, 9, 8, 7, 6, 5])
        assert len(np.unique(lt)) == 2

    def test_RSA_3d_extended(self):
        im = np.zeros([100, 100, 100], dtype=int)
        im = ps.generators.RSA(im, r=10, volume_fraction=0.5, mode='extended')
        im = np.pad(im, pad_width=1, mode='constant', constant_values=False)
        lt = ps.filters.local_thickness(im, sizes=[10, 9, 8, 7, 6, 5])
        assert len(np.unique(lt)) > 2

    def test_RSA_2d_seqential_additions(self):
        im = np.zeros([100, 100], dtype=int)
        im = ps.generators.RSA(im, r=10)
        phi1 = ps.metrics.porosity(im)
        im = ps.generators.RSA(im, r=5)
        phi2 = ps.metrics.porosity(im)
        assert phi2 > phi1

    def test_RSA_preexisting_structure(self):
        im = ps.generators.blobs(shape=[200, 200, 200])
        phi1 = im.sum()/im.size
        im = ps.generators.RSA(im, r=8, n_max=200, mode='contained')
        phi2 = im.sum()/im.size
        assert phi2 > phi1
        # Ensure that 3 passes through RSA fills up image
        im = ps.generators.RSA(im, r=8, n_max=200, mode='contained')
        im = ps.generators.RSA(im, r=8, n_max=200, mode='contained')
        im = ps.generators.RSA(im, r=8, n_max=200, mode='contained')
        phi1 = im.sum()/im.size
        im = ps.generators.RSA(im, r=8, n_max=200, mode='contained')
        phi2 = im.sum()/im.size
        assert phi2 == phi1

    def test_RSA_shape(self):
        rsa = ps.generators.RSA(im_or_shape=[200, 200], r=10)
        assert np.all(rsa.shape == (200, 200))

    def test_RSA_clearance(self):
        np.random.seed(0)
        rsa0 = ps.generators.RSA(im_or_shape=[200, 200], r=9, clearance=0)
        np.random.seed(0)
        rsa2p = ps.generators.RSA(im_or_shape=[200, 200], r=9, clearance=2)
        assert rsa0.sum() > rsa2p.sum()
        np.random.seed(0)
        rsa1n = ps.generators.RSA(im_or_shape=[200, 200], r=9, clearance=-1)
        assert rsa0.sum() < rsa1n.sum()

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

    def test_pseudo_gravity_packing_monodisperse(self):
        im = np.ones([400, 400], dtype=bool)
        np.random.seed(0)
        im = ps.generators.pseudo_gravity_packing(im=im, r=20, clearance=0)
        e1 = im.sum()/im.size
        im = np.ones([400, 400], dtype=bool)
        np.random.seed(0)
        im = ps.generators.pseudo_gravity_packing(im=im, r=20, clearance=5)
        e2 = im.sum()/im.size
        assert e2 < e1
        im = np.ones([400, 400], dtype=bool)
        np.random.seed(0)
        im = ps.generators.pseudo_gravity_packing(im=im, r=20, max_iter=10)
        e3 = im.sum()/im.size
        im = np.ones([400, 400], dtype=bool)
        np.random.seed(0)
        im = ps.generators.pseudo_gravity_packing(im=im, r=50, max_iter=10)
        e4 = im.sum()/im.size
        assert e4 > e3

    def test_pseudo_gravity_packing_values(self):
        np.random.seed(0)
        # 2d
        im = np.ones([50, 50], dtype=bool)
        im = ps.generators.pseudo_gravity_packing(im=im, r=5, clearance=0)
        assert_allclose(np.linalg.norm(im), 37.3497, rtol=1e-5)
        # 3d
        im = np.ones([50, 50, 50], dtype=bool)
        im = ps.generators.pseudo_gravity_packing(im=im, r=5, clearance=0)
        assert_allclose(np.linalg.norm(im), 218.3804, rtol=1e-5)

    def test_pseudo_electrostatic_packing(self):
        im1 = ps.generators.blobs(shape=[100, 100])
        im2 = ps.generators.pseudo_electrostatic_packing(
            im=im1, r=3, clearance=1, protrusion=1)
        assert (im1.sum() > im2.sum())
        assert im2.sum() > 0

    def test_pseudo_electrostatic_packing_values(self):
        np.random.seed(0)
        # 2d
        im = ps.generators.blobs(shape=[100, 100])
        im = ps.generators.pseudo_electrostatic_packing(
            im=im, r=3, clearance=1, protrusion=1)
        assert_allclose(np.linalg.norm(im), 46.2276, rtol=1e-5)
        # 3d
        im = ps.generators.blobs(shape=[50, 50, 50])
        im = ps.generators.pseudo_electrostatic_packing(
            im=im, r=3, clearance=1, protrusion=1)
        assert_allclose(np.linalg.norm(im), 135.3403, rtol=1e-5)

    def test_faces(self):
        im = ps.generators.faces(shape=[10, 10], inlet=0)
        assert im.sum() == 10
        im = ps.generators.faces(shape=[10, 10], outlet=0)
        assert im.sum() == 10
        im = ps.generators.faces(shape=[10, 10], inlet=0, outlet=0)
        assert im.sum() == 20
        im = ps.generators.faces(shape=[10, 10, 10], inlet=0)
        assert im.sum() == 100
        im = ps.generators.faces(shape=[10, 10, 10], outlet=0)
        assert im.sum() == 100
        im = ps.generators.faces(shape=[10, 10, 10], inlet=0, outlet=0)
        assert im.sum() == 200
        with pytest.raises(Exception):
            ps.generators.faces(shape=[10, 10, 10])

    def test_fractal_noise_2D(self):
        s = [100, 100]
        # Ensure identical images are returned if seed is same
        im1 = ps.generators.fractal_noise(shape=s, seed=0, cores=1)
        im2 = ps.generators.fractal_noise(shape=s, seed=0, cores=1)
        assert np.linalg.norm(im1) == np.linalg.norm(im2)
        # Ensure different images are returned even if seed is same
        im1 = ps.generators.fractal_noise(shape=s, mode='perlin',
                                          seed=0, octaves=2, cores=1)
        im2 = ps.generators.fractal_noise(shape=s, mode='perlin',
                                          seed=0, octaves=4, cores=1)
        assert np.linalg.norm(im1) != np.linalg.norm(im2)
        # Check uniformization
        im1 = ps.generators.fractal_noise(shape=s, mode='cubic',
                                          uniform=True, cores=1)
        assert im1.min() >= 0
        assert im1.max() <= 1
        im2 = ps.generators.fractal_noise(shape=s, mode='cubic',
                                          uniform=False, cores=1)
        assert im2.min() < 0

    def test_cantor_dust(self):
        np.random.seed(0)
        im2D = ps.generators.random_cantor_dust([100, 100], 6, f=0.8)
        assert im2D.shape == (128, 128)
        np.testing.assert_allclose(im2D.sum()/im2D.size, 0.29180908203125)
        np.random.seed(0)
        im3D = ps.generators.random_cantor_dust([100, 100, 100], 6, f=0.8)
        assert im3D.shape == (128, 128, 128)
        np.testing.assert_allclose(im3D.sum()/im3D.size, 0.31614160537720)

    def test_sierpinski_foam(self):
        im2D = ps.generators.sierpinski_foam(4, 4, 2)
        assert im2D.shape == (324, 324)
        im3D = ps.generators.sierpinski_foam(4, 4, 3)
        assert im3D.shape == (324, 324, 324)
        im3D = ps.generators.sierpinski_foam(4, 4, 3, max_size=1000)
        assert im3D.shape == (12, 12, 12)
        im2D = ps.generators.sierpinski_foam(4, 2, 2)
        np.testing.assert_allclose(im2D.sum()/im2D.size, 0.7901234567901234)
        im2D = ps.generators.sierpinski_foam(4, 3, 2)
        np.testing.assert_allclose(im2D.sum()/im2D.size, 0.7023319615912208)
        im2D = ps.generators.sierpinski_foam(4, 4, 2)
        np.testing.assert_allclose(im2D.sum()/im2D.size, 0.6242950769699741)
        # Ensure the exact same image is produced each time
        im2D = ps.generators.sierpinski_foam(4, 2, 2)
        np.testing.assert_allclose(im2D.sum()/im2D.size, 0.7901234567901234)


if __name__ == '__main__':
    t = GeneratorTest()
    self = t
    t.setup_class()
    for item in t.__dir__():
        if item.startswith('test'):
            print(f"Running test: {item}")
            t.__getattribute__(item)()
