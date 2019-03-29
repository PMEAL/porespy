import porespy as ps
import scipy as sp
import scipy.ndimage as spim
import matplotlib.pyplot as plt
import pytest


class ToolsTest():
    def setup_class(self):
        plt.close('all')
        self.im = sp.random.randint(0, 10, 20)
        sp.random.seed(0)
        self.blobs = ps.generators.blobs(shape=[101, 101])
        self.im2D = ps.generators.blobs(shape=[51, 51])
        self.im3D = ps.generators.blobs(shape=[51, 51, 51])
        self.labels, N = spim.label(input=self.blobs)

    def test_randomize_colors(self):
        randomized_im = ps.tools.randomize_colors(im=self.im)
        assert sp.unique(self.im).size == sp.unique(randomized_im).size
        assert sp.all(sp.unique(self.im) == sp.unique(randomized_im))

    def test_make_contiguous_size(self):
        cont_im = ps.tools.make_contiguous(self.im)
        assert sp.unique(self.im).size == sp.unique(cont_im).size

    def test_make_contiguous_contiguity(self):
        cont_im = ps.tools.make_contiguous(self.im)
        assert sp.all(sp.arange(sp.unique(self.im).size) == sp.unique(cont_im))

    def test_make_contiguous_negs(self):
        im = sp.array([[0, 0, 1, 3], [-2, -4, 1, 3], [-4, 3, 5, 0]])
        a = ps.tools.make_contiguous(im, keep_zeros=True).max()
        b = ps.tools.make_contiguous(im, keep_zeros=False).max()
        assert a == b

    def test_get_slice(self):
        one_lab = self.labels == 10
        my_slice = ps.tools.get_slice(one_lab, center=[75, 9], size=10)
        assert sp.sum(one_lab) == sp.sum(one_lab[tuple(my_slice)])

    def test_find_outer_region(self):
        one_lab = self.labels == 10
        my_slice = ps.tools.get_slice(one_lab, center=[75, 9], size=10)
        small_slice = one_lab[tuple(my_slice)]
        outer = ps.tools.find_outer_region(small_slice)
        assert sp.sum(outer) == sp.sum(small_slice[:, 0])

    def test_extract_subsection(self):
        sec = ps.tools.extract_subsection(self.blobs, [0.5])
        assert sp.all(sp.array(sp.shape(sec)) == 50)

    # def test_extract_cylinder(self):
    #     cylinder = ps.tools.extract_cylinder(self.im3D)

    def test_bbox_to_slices(self):
        s = ps.tools.bbox_to_slices([0, 0, 0, 10, 10, 10])
        assert sp.all(self.im3D[s].shape == (10, 10, 10))

    def test_get_planes(self):
        x, y, z = ps.tools.get_planes(self.im3D)
        assert sp.all(x.shape == (51, 51))
        assert sp.all(y.shape == (51, 51))
        assert sp.all(z.shape == (51, 51))
        with pytest.raises(ValueError):
            ps.tools.get_planes(self.im2D)

    def test_get_planes_not_squeezed(self):
        x, y, z = ps.tools.get_planes(self.im3D, squeeze=False)
        assert sp.all(x.shape == (1, 51, 51))
        assert sp.all(y.shape == (51, 1, 51))
        assert sp.all(z.shape == (51, 51, 1))

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
        im = ps.tools.align_image_with_openpnm(sp.ones([40, 50]))
        assert im.shape == (50, 40)
        im = ps.tools.align_image_with_openpnm(sp.ones([40, 50, 60]))
        assert im.shape == (60, 50, 40)

    def test_inhull(self):
        X = sp.rand(25, 2)
        hull = sp.spatial.ConvexHull(X)
        assert not ps.tools.in_hull([[0, 0]], hull)
        assert ps.tools.in_hull([sp.mean(X, axis=0)], hull)
        X = sp.rand(25, 3)
        hull = sp.spatial.ConvexHull(X)
        assert not ps.tools.in_hull([[0, 0, 0]], hull)
        assert ps.tools.in_hull([sp.mean(X, axis=0)], hull)

    def test_insert_sphere_2D(self):
        im = sp.zeros(shape=[200, 200], dtype=bool)
        im = ps.tools.insert_sphere(im, [100, 100], 50)
        im = ps.tools.insert_sphere(im, [10, 100], 50)
        im = ps.tools.insert_sphere(im, [180, 100], 50)

    def test_insert_sphere_3D(self):
        im = sp.zeros(shape=[200, 200, 200], dtype=bool)
        im = ps.tools.insert_sphere(im, [100, 100, 100], 50)
        im = ps.tools.insert_sphere(im, [10, 100, 100], 50)
        im = ps.tools.insert_sphere(im, [180, 100, 100], 50)


if __name__ == '__main__':
    t = ToolsTest()
    self = t
    t.setup_class()
    for item in t.__dir__():
        if item.startswith('test'):
            print('running test: '+item)
            t.__getattribute__(item)()
