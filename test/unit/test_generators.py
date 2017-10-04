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
        print(N)
        assert N == 101

    def test_polydisperse_spheres(self):
        pass

    def test_voronoi_edges(self):
        pass

    def test_circle_pack_triangular(self):
        pass

    def test_sphere_pack_sc(self):
        pass

    def test_sphere_pack_fcc(self):
        pass

    def test_sphere_pack_bcc(self):
        pass

    def test_noise_simplex(self):
        pass

    def test_noise_perlin(self):
        pass

if __name__ == '__main__':
    t = GeneratorTest()
    t.setup_class()
    t.test_insert_shape()
    t.test_bundle_of_tubes()