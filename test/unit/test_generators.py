import porespy as ps
import scipy as sp
import pytest
import scipy.ndimage as spim
import matplotlib.pyplot as plt
plt.close('all')


class GeneratorTest():

    def setup_class(self):
        sp.random.seed(0)

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
        im = ps.generators.overlapping_spheres(shape=[101, 101], radius=5,
                                               porosity=0.5)
        poro = sp.sum(im)/sp.size(im)
        print(poro)
        plt.figure()
        plt.imshow(im[:, :])
        assert (poro-0.5)**2 < 0.1

    def test_overlapping_spheres_3d(self):
        target = 0.5
        im = ps.generators.overlapping_spheres(shape=[50, 50, 50], radius=5,
                                               porosity=target)
        poro = sp.sum(im)/sp.size(im)
        assert (poro-target)**2 < 0.15

    def test_polydisperse_spheres(self):
        target = 0.5
        dist = sp.stats.norm(loc=5, scale=1)
        im = ps.generators.polydisperse_spheres(shape=[50, 50, 50],
                                                porosity=target,
                                                dist=dist,
                                                nbins=10)
        poro = sp.sum(im)/sp.size(im)
        assert (poro-target)**2 < 0.15

    def test_voronoi_edges(self):
        sp.random.seed(0)
        im = ps.generators.voronoi_edges(shape=[50, 50, 50],
                                         radius=2,
                                         ncells=25,
                                         flat_faces=True)
        top_slice = im[:, :, 0]
        assert sp.sum(top_slice) == 1409

    def test_circle_pack_square(self):
        im = ps.generators.circle_pack(shape=[101, 101], radius=5)
        labels, N = spim.label(input=~im)
        assert N == 100

    def test_circle_pack_triangular(self):
        im = ps.generators.circle_pack(shape=[101, 101], radius=5,
                                       packing='triangular')
        labels, N = spim.label(input=~im)
        assert N == 85

    def test_sphere_pack_sc(self):
        im = ps.generators.sphere_pack(shape=[101, 101, 101],
                                       radius=4,
                                       offset=1)
        labels, N = spim.label(input=~im)
        assert N == 1000

    def test_sphere_pack_fcc(self):
        im = ps.generators.sphere_pack(shape=[101, 101, 101],
                                       radius=4,
                                       offset=2,
                                       packing='fcc')
        labels, N = spim.label(input=~im)
        assert N == 392

    def test_sphere_pack_bcc(self):
        im = ps.generators.sphere_pack(shape=[101, 101, 101],
                                       radius=4,
                                       offset=2,
                                       packing='bcc')
        labels, N = spim.label(input=~im)
        assert N == 1024

    def test_noise_simplex(self):
        pass

    def test_noise_perlin(self):
        pass

if __name__ == '__main__':
    t = GeneratorTest()
    t.setup_class()
    t.test_insert_shape()
    t.test_bundle_of_tubes()
    t.test_overlapping_spheres_3d()
    t.test_polydisperse_spheres()
    t.test_voronoi_edges()
    t.test_circle_pack_square()
    t.test_circle_pack_triangular()
    t.test_sphere_pack_sc()
    t.test_sphere_pack_fcc()
    t.test_sphere_pack_bcc()
