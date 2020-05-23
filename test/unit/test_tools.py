import porespy as ps
import numpy as np
import scipy as sp
import scipy.ndimage as spim
import matplotlib.pyplot as plt
import pytest


class ToolsTest():
    def setup_class(self):
        plt.close('all')
        self.im = np.random.randint(0, 10, 20)
        np.random.seed(0)
        self.blobs = ps.generators.blobs(shape=[101, 101])
        self.im2D = ps.generators.blobs(shape=[51, 51])
        self.im3D = ps.generators.blobs(shape=[51, 51, 51])
        self.labels, N = spim.label(input=self.blobs)

    def test_randomize_colors(self):
        randomized_im = ps.tools.randomize_colors(im=self.im)
        assert np.unique(self.im).size == np.unique(randomized_im).size
        assert np.all(np.unique(self.im) == np.unique(randomized_im))

    def test_make_contiguous_size(self):
        cont_im = ps.tools.make_contiguous(self.im)
        assert np.unique(self.im).size == np.unique(cont_im).size

    def test_make_contiguous_contiguity(self):
        cont_im = ps.tools.make_contiguous(self.im)
        assert np.all(np.arange(np.unique(self.im).size) == np.unique(cont_im))

    def test_make_contiguous_negs(self):
        im = np.array([[0, 0, 1, 3], [-2, -4, 1, 3], [-4, 3, 5, 0]])
        a = ps.tools.make_contiguous(im, keep_zeros=True).max()
        b = ps.tools.make_contiguous(im, keep_zeros=False).max()
        assert a == b

    def test_extract_subsection(self):
        sec = ps.tools.extract_subsection(self.blobs, [0.5])
        assert np.all(np.array(np.shape(sec)) == 50)

    def test_extract_cylinder(self):
        im = np.ones([200, 300, 400], dtype=bool)
        cx = ps.tools.extract_cylinder(im)
        assert cx.sum() == 14132200
        cy = ps.tools.extract_cylinder(im, axis=1)
        assert cy.sum() == 9419100
        cz = ps.tools.extract_cylinder(im, axis=2)
        assert cz.sum() == 12558800
        cr = ps.tools.extract_cylinder(im, r=100)
        assert cr.sum() == 6279400

    def test_bbox_to_slices(self):
        s = ps.tools.bbox_to_slices([0, 0, 0, 10, 10, 10])
        assert np.all(self.im3D[s].shape == (10, 10, 10))

    def test_get_planes(self):
        x, y, z = ps.tools.get_planes(self.im3D)
        assert np.all(x.shape == (51, 51))
        assert np.all(y.shape == (51, 51))
        assert np.all(z.shape == (51, 51))
        with pytest.raises(ValueError):
            ps.tools.get_planes(self.im2D)

    def test_get_planes_not_squeezed(self):
        x, y, z = ps.tools.get_planes(self.im3D, squeeze=False)
        assert np.all(x.shape == (1, 51, 51))
        assert np.all(y.shape == (51, 1, 51))
        assert np.all(z.shape == (51, 51, 1))

    def test_get_border(self):
        c = ps.tools.get_border(self.im2D.shape, thickness=1, mode='corners')
        assert c.sum() == 4
        c = ps.tools.get_border(self.im3D.shape, thickness=1, mode='corners')
        assert c.sum() == 8
        c = ps.tools.get_border(self.im2D.shape, thickness=1, mode='edges')
        assert c.sum() == 200
        c = ps.tools.get_border(self.im3D.shape, thickness=1, mode='edges')
        assert c.sum() == 596
        c = ps.tools.get_border(self.im2D.shape, thickness=1, mode='faces')
        assert c.sum() == 200
        c = ps.tools.get_border(self.im3D.shape, thickness=1, mode='faces')
        assert c.sum() == 15002

    def test_align_image_w_openpnm(self):
        im = ps.tools.align_image_with_openpnm(np.ones([40, 50]))
        assert im.shape == (50, 40)
        im = ps.tools.align_image_with_openpnm(np.ones([40, 50, 60]))
        assert im.shape == (60, 50, 40)

    def test_inhull(self):
        X = np.random.rand(25, 2)
        hull = sp.spatial.ConvexHull(X)
        assert not ps.tools.in_hull([[0, 0]], hull)
        assert ps.tools.in_hull([np.mean(X, axis=0)], hull)
        X = np.random.rand(25, 3)
        hull = sp.spatial.ConvexHull(X)
        assert not ps.tools.in_hull([[0, 0, 0]], hull)
        assert ps.tools.in_hull([np.mean(X, axis=0)], hull)

    def test_insert_sphere_2D(self):
        im = np.zeros(shape=[200, 200], dtype=bool)
        im = ps.tools.insert_sphere(im, [100, 100], 50)
        im = ps.tools.insert_sphere(im, [10, 100], 50)
        im = ps.tools.insert_sphere(im, [180, 100], 50)

    def test_insert_sphere_3D(self):
        im = np.zeros(shape=[200, 200, 200], dtype=bool)
        im = ps.tools.insert_sphere(im, [100, 100, 100], 50)
        im = ps.tools.insert_sphere(im, [10, 100, 100], 50)
        im = ps.tools.insert_sphere(im, [180, 100, 100], 50)

    def test_subdivide_3D(self):
        im = np.ones([50, 100, 150])
        ims = ps.tools.subdivide(im, divs=1)
        assert ims.shape == (1, 1, 1)
        assert np.all(im[tuple(ims[0, 0, 0])] == im)
        ims = ps.tools.subdivide(im, divs=2)
        assert ims.shape == (2, 2, 2)
        assert im[tuple(ims[0, 0, 0])].sum() == np.prod(im.shape)/8

    def test_subdivide_2D(self):
        im = np.ones([50, 100])
        ims = ps.tools.subdivide(im, divs=2)
        assert ims.shape == (2, 2)
        assert im[tuple(ims[0, 0])].sum() == np.prod(im.shape)/4


if __name__ == '__main__':
    t = ToolsTest()
    self = t
    t.setup_class()
    for item in t.__dir__():
        if item.startswith('test'):
            print('running test: '+item)
            t.__getattribute__(item)()
