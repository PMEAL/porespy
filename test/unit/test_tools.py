import sys
import porespy as ps
import numpy as np
import scipy.spatial as sptl
import scipy.ndimage as spim
import matplotlib.pyplot as plt
import pytest
from edt import edt
ps.settings.tqdm['disable'] = True


class ToolsTest():

    def setup_class(self):
        plt.close('all')
        np.random.seed(0)
        self.im = np.random.randint(0, 10, 20)
        np.random.seed(0)
        self.blobs = ps.generators.blobs(shape=[101, 101])
        self.im2D = ps.generators.blobs(shape=[51, 51])
        self.im3D = ps.generators.blobs(shape=[51, 51, 51])
        self.labels, N = spim.label(input=self.blobs)

    def test_unpad(self):
        pad_width = [10, 20]
        im = ps.generators.blobs([200, 300], porosity=0.3)
        im1 = np.pad(im, pad_width, mode="constant", constant_values=1)
        im2 = ps.tools.unpad(im1, pad_width)
        assert np.all(im == im2)

    def test_unpad_int_padwidth(self):
        pad_width = 10
        im = ps.generators.blobs([200, 300], porosity=0.3)
        im1 = np.pad(im, pad_width, mode="constant", constant_values=1)
        im2 = ps.tools.unpad(im1, pad_width)
        assert np.all(im == im2)

    def test_unpad_different_padwidths_on_each_axis(self):
        pad_width = [[10, 20], [30, 40]]
        im = ps.generators.blobs([200, 300], porosity=0.3)
        im1 = np.pad(im, pad_width, mode="constant", constant_values=1)
        im2 = ps.tools.unpad(im1, pad_width)
        assert np.all(im == im2)

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

    def test_make_contiguous_w_negs_and_modes(self):
        im = np.array([[0, 0, 1, 3], [-2, -4, 1, 3], [-4, 3, 5, 0]])
        a = ps.tools.make_contiguous(im, mode='keep_zeros').flatten()
        assert np.all(a == [0, 0, 3, 4, 2, 1, 3, 4, 1, 4, 5, 0])
        b = ps.tools.make_contiguous(im, mode='clipped').flatten()
        assert np.all(b == [0, 0, 1, 2, 0, 0, 1, 2, 0, 2, 3, 0])
        c = ps.tools.make_contiguous(im, mode='symmetric').flatten()
        assert np.all(c == [0, 0, 1, 2, -1, -2, 1, 2, -2, 2, 3, 0])
        d = ps.tools.make_contiguous(im, mode='none').flatten()
        assert np.all(d == [3, 3, 4, 5, 2, 1, 4, 5, 1, 5, 6, 3])

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

    def test_align_image_w_openpnm(self):
        im = ps.tools.align_image_with_openpnm(np.ones([40, 50]))
        assert im.shape == (50, 40)
        im = ps.tools.align_image_with_openpnm(np.ones([40, 50, 60]))
        assert im.shape == (60, 50, 40)

    def test_inhull(self):
        X = np.random.rand(25, 2)
        hull = sptl.ConvexHull(X)
        assert not ps.tools.in_hull([[0, 0]], hull)
        assert ps.tools.in_hull([np.mean(X, axis=0)], hull)
        X = np.random.rand(25, 3)
        hull = sptl.ConvexHull(X)
        assert not ps.tools.in_hull([[0, 0, 0]], hull)
        assert ps.tools.in_hull([np.mean(X, axis=0)], hull)

    def test_insert_sphere_2D_no_overwrite(self):
        im = np.zeros(shape=[200, 200], dtype=bool)
        im = ps.tools.insert_sphere(im, c=[100, 100], r=50, v=1, overwrite=False)
        im = ps.tools.insert_sphere(im, c=[110, 100], r=50, v=2, overwrite=False)
        im = ps.tools.insert_sphere(im, c=[90, 100], r=50, v=3, overwrite=False)
        vals, counts = np.unique(im, return_counts=True)
        assert np.all(np.unique(im) == vals)
        assert counts[1] > counts[2]

    def test_insert_sphere_2D_w_overwrite(self):
        im = np.zeros(shape=[200, 200], dtype=bool)
        im = ps.tools.insert_sphere(im, c=[100, 100], r=50, v=1, overwrite=True)
        im = ps.tools.insert_sphere(im, c=[110, 100], r=50, v=2, overwrite=True)
        im = ps.tools.insert_sphere(im, c=[90, 100], r=50, v=3, overwrite=True)
        vals, counts = np.unique(im, return_counts=True)
        assert np.all(np.unique(im) == vals)
        assert counts[1] < counts[2]

    def test_insert_sphere_3D_no_overwrite(self):
        im = np.zeros(shape=[200, 200, 200], dtype=bool)
        im = ps.tools.insert_sphere(im, c=[100, 100, 100], r=50, v=1,
                                    overwrite=False)
        im = ps.tools.insert_sphere(im, c=[110, 100, 100], r=50, v=2,
                                    overwrite=False)
        im = ps.tools.insert_sphere(im, c=[90, 100, 100], r=50, v=3,
                                    overwrite=False)
        vals, counts = np.unique(im, return_counts=True)
        assert np.all(np.unique(im) == vals)
        assert counts[1] > counts[2]

    def test_insert_sphere_3D_w_overwrite(self):
        im = np.zeros(shape=[200, 200, 200], dtype=bool)
        im = ps.tools.insert_sphere(im, c=[100, 100, 100], r=50, v=1,
                                    overwrite=True)
        im = ps.tools.insert_sphere(im, c=[110, 100, 100], r=50, v=2,
                                    overwrite=True)
        im = ps.tools.insert_sphere(im, c=[90, 100, 100], r=50, v=3,
                                    overwrite=True)
        vals, counts = np.unique(im, return_counts=True)
        assert np.all(np.unique(im) == vals)
        assert counts[1] < counts[2]

    def test_insert_cylinder(self):
        im = np.zeros([100, 100, 100], dtype=bool)
        im = ps.tools.insert_cylinder(im, [20, 20, 20], [80, 80, 80], r=30)
        assert im.sum() == 356924

    def test_insert_cylinder_outside_image(self):
        im = np.zeros([50, 50, 50], dtype=bool)
        with pytest.raises(Exception):
            im = ps.tools.insert_cylinder(im, [20, 20, 20], [80, 80, 80], r=30)

    def test_subdivide_2D_with_vector_overlap(self):
        im = np.ones([150, 150])
        s = ps.tools.subdivide(im, divs=3, overlap=[10, 20])
        assert np.all(im[s[0]].shape == (60, 70))
        assert np.all(im[s[1]].shape == (60, 90))
        assert np.all(im[s[4]].shape == (70, 90))

    def test_subdivide_2D_with_scalar_overlap(self):
        im = np.ones([150, 150])
        s = ps.tools.subdivide(im, divs=3, overlap=10)
        assert np.all(im[s[0]].shape == (60, 60))
        assert np.all(im[s[1]].shape == (60, 70))
        assert np.all(im[s[4]].shape == (70, 70))

    def test_subdivide_2D_with_vector_overlap_flattened(self):
        im = np.ones([150, 150])
        s = ps.tools.subdivide(im, divs=3, overlap=[10, 20])
        assert np.all(im[s[0]].shape == (60, 70))
        assert np.all(im[s[1]].shape == (60, 90))
        assert np.all(im[s[4]].shape == (70, 90))

    def test_subdivide_3D_with_vector_overlap(self):
        im = np.ones([150, 150, 150])
        s = ps.tools.subdivide(im, divs=3, overlap=[10, 20, 30])
        assert np.all(im[s[0]].shape == (60, 70, 80))
        assert np.all(im[s[1]].shape == (60, 70, 110))
        assert np.all(im[s[13]].shape == (70, 90, 110))

    def test_subdivide_3D_with_scalar_overlap(self):
        im = np.ones([150, 150, 150])
        s = ps.tools.subdivide(im, divs=3, overlap=10)
        assert np.all(im[s[0]].shape == (60, 60, 60))
        assert np.all(im[s[1]].shape == (60, 60, 70))
        assert np.all(im[s[13]].shape == (70, 70, 70))

    def test_subdivided_shape(self):
        im = np.ones([150, 150, 150])
        s = ps.tools.subdivide(im, divs=3, overlap=[10, 20, 30])
        assert np.all(len(s) == 27)

    def test_recombine_2d_zero_overlap(self):
        im = np.random.rand(160, 160)
        s = ps.tools.subdivide(im, divs=2, overlap=[0, 0])
        ims = []
        for i, _ in enumerate(s):
            ims.append(im[s[i]])
        im2 = ps.tools.recombine(ims=ims, slices=s, overlap=[0, 0])
        assert np.all(im == im2)

    def test_recombine_2d_with_vector_overlap(self):
        im = np.random.rand(160, 160)
        s = ps.tools.subdivide(im, divs=2, overlap=[10, 10])
        ims = []
        for i, _ in enumerate(s):
            ims.append(im[s[i]])
        im2 = ps.tools.recombine(ims=ims, slices=s, overlap=10)
        assert np.all(im == im2)
        im2 = ps.tools.recombine(ims=ims, slices=s, overlap=[10, 10])
        assert np.all(im == im2)

    def test_recombine_2d_with_scalar_overlap(self):
        im = np.random.rand(160, 160)
        s = ps.tools.subdivide(im, divs=2, overlap=10)
        ims = []
        for i, _ in enumerate(s):
            ims.append(im[s[i]])
        im2 = ps.tools.recombine(ims=ims, slices=s, overlap=10)
        assert np.all(im == im2)

    def test_recombine_3d_zero_overlap(self):
        im = np.random.rand(160, 160, 160)
        s = ps.tools.subdivide(im, divs=2, overlap=[0, 0, 0])
        ims = []
        for i, _ in enumerate(s):
            ims.append(im[s[i]])
        im2 = ps.tools.recombine(ims=ims, slices=s, overlap=[0, 0, 0])
        assert np.all(im == im2)

    def test_recombine_3d_with_vector_overlap(self):
        im = np.random.rand(160, 160, 160)
        s = ps.tools.subdivide(im, divs=2, overlap=[10, 10, 10])
        ims = []
        for i, _ in enumerate(s):
            ims.append(im[s[i]])
        im2 = ps.tools.recombine(ims=ims, slices=s, overlap=10)
        assert np.all(im == im2)
        im2 = ps.tools.recombine(ims=ims, slices=s, overlap=[10, 10, 10])
        assert np.all(im == im2)

    def test_recombine_2d_odd_shape(self):
        im = np.random.rand(143, 152)
        s = ps.tools.subdivide(im, divs=2, overlap=10)
        ims = []
        for i, _ in enumerate(s):
            ims.append(im[s[i]])
        im2 = ps.tools.recombine(ims=ims, slices=s, overlap=10)
        assert np.all(im == im2)

    def test_recombine_2d_odd_shape_vector_overlap(self):
        im = np.random.rand(143, 177)
        s = ps.tools.subdivide(im, divs=2, overlap=[10, 20])
        ims = []
        for i, _ in enumerate(s):
            ims.append(im[s[i]])
        im2 = ps.tools.recombine(ims=ims, slices=s, overlap=[10, 20])
        assert np.all(im == im2)

    def test_recombine_3d_odd_shape_vector_overlap(self):
        im = np.random.rand(143, 177, 111)
        s = ps.tools.subdivide(im, divs=3, overlap=[10, 20, 25])
        ims = []
        for i, _ in enumerate(s):
            ims.append(im[s[i]])
        im2 = ps.tools.recombine(ims=ims, slices=s, overlap=[10, 20, 25])
        assert np.all(im == im2)

    def test_sanitize_filename(self):
        fname = "test.stl.stl"
        assert ps.tools.sanitize_filename(fname, "stl") == "test.stl.stl"
        assert ps.tools.sanitize_filename(fname, "vtk") == "test.stl.stl.vtk"
        assert ps.tools.sanitize_filename(fname, "stl", exclude_ext=True) == "test.stl"

    def test_extract_regions(self):
        im = spim.label(self.im2D)[0]
        im = im*ps.tools.extract_regions(im, labels=[2, 3], trim=False)
        assert np.all(np.unique(im) == [0, 2, 3])

    condition = sys.platform.startswith("win")  # and sys.version_info[:2] == (3, 8)

    @pytest.mark.skipif(condition, reason="scikit-fmm clashes with numpy")
    def test_marching_map(self):
        im = ps.generators.lattice_spheres(shape=[101, 101],
                                           r=5, spacing=25,
                                           offset=[5, 5], lattice='tri')
        bd = np.zeros_like(im)
        bd[:, 0] = True
        fmm = ps.tools.marching_map(path=im, start=bd)
        assert fmm.max() > 100

    def test_ps_strels(self):
        c = ps.tools.ps_disk(r=3)
        assert c.sum() == 25
        c = ps.tools.ps_disk(r=3, smooth=False)
        assert c.sum() == 29
        b = ps.tools.ps_ball(r=3)
        assert b.sum() == 93
        b = ps.tools.ps_ball(r=3, smooth=False)
        assert b.sum() == 123
        s = ps.tools.ps_rect(w=3, ndim=2)
        assert s.sum() == 9
        c = ps.tools.ps_rect(w=3, ndim=3)
        assert c.sum() == 27

    def test_find_outer_region(self):
        outer = ps.tools.find_outer_region(self.im3D)
        assert outer.sum() == 1989
        outer = ps.tools.find_outer_region(self.im2D)
        assert outer.sum() == 64

    def test_numba_insert_disk_2D(self):
        im = np.zeros([50, 50], dtype=int)
        c = np.vstack([[10, 10], [30, 40]]).T
        im = ps.tools._insert_disk_at_points(im=im, coords=c, r=10, v=2)
        assert im.max() == 2
        assert im.sum() == 1220
        im = ps.tools._insert_disk_at_points(im=im, coords=c, r=10, v=2.4,
                                             smooth=False)
        assert im.max() == 2
        assert im.sum() == 1266
        im = ps.tools._insert_disk_at_points(im=im, coords=c, r=10, v=3,
                                             overwrite=False)
        assert im.max() == 2
        im = ps.tools._insert_disk_at_points(im=im, coords=c, r=10, v=3,
                                             overwrite=True)
        assert im.max() == 3

    def test_numba_insert_disk_3D(self):
        im = np.zeros([50, 50, 50], dtype=int)
        c = np.vstack([[10, 10, 10], [30, 40, 20]]).T
        im = ps.tools._insert_disk_at_points(im=im, coords=c, r=10, v=2)
        assert im.max() == 2
        assert im.sum() == 16556
        im = ps.tools._insert_disk_at_points(im=im, coords=c, r=10, v=2.4,
                                             smooth=False)
        assert im.max() == 2
        assert im.sum() == 16674
        im = ps.tools._insert_disk_at_points(im=im, coords=c, r=10, v=3,
                                             overwrite=False)
        assert im.max() == 2
        im = ps.tools._insert_disk_at_points(im=im, coords=c, r=10, v=3,
                                             overwrite=True)
        assert im.max() == 3

    def test_numba_insert_disks_2D(self):
        im = np.zeros([50, 50], dtype=int)
        c = np.vstack([[10, 10], [30, 40]]).T
        r = np.array([10, 10], dtype=int)
        im = ps.tools._insert_disks_at_points(im=im, coords=c, radii=r, v=2)
        assert im.max() == 2
        assert im.sum() == 1220
        im = ps.tools._insert_disks_at_points(im=im, coords=c, radii=r, v=2.4,
                                              smooth=False)
        assert im.max() == 2
        assert im.sum() == 1266
        im = ps.tools._insert_disks_at_points(im=im, coords=c, radii=r, v=3,
                                              overwrite=False)
        assert im.max() == 2
        im = ps.tools._insert_disks_at_points(im=im, coords=c, radii=r, v=3,
                                              overwrite=True)
        assert im.max() == 3

    def test_numba_insert_disks_3D(self):
        im = np.zeros([50, 50, 50], dtype=int)
        c = np.vstack([[10, 10, 10], [30, 40, 20]]).T
        r = np.array([10, 10], dtype=int)
        im = ps.tools._insert_disks_at_points(im=im, coords=c, radii=r, v=2)
        assert im.max() == 2
        assert im.sum() == 16556
        im = ps.tools._insert_disks_at_points(im=im, coords=c, radii=r, v=2.4,
                                              smooth=False)
        assert im.max() == 2
        assert im.sum() == 16674
        im = ps.tools._insert_disks_at_points(im=im, coords=c, radii=r, v=3,
                                              overwrite=False)
        assert im.max() == 2
        im = ps.tools._insert_disks_at_points(im=im, coords=c, radii=r, v=3,
                                              overwrite=True)
        assert im.max() == 3

    def test_find_bbox_2D(self):
        temp = np.ones([50, 50], dtype=bool)
        temp[25, 25] = False
        im2D = edt(temp) < 10
        bbox = ps.tools.find_bbox(im2D)
        assert im2D[bbox].shape == (19, 19)
        im2D = edt(temp) <= 10
        bbox = ps.tools.find_bbox(im2D)
        assert im2D[bbox].shape == (21, 21)
        bbox = ps.tools.find_bbox(im2D, order_by='corners')
        assert bbox == [[15, 15], [36, 36]]

    def test_find_bbox_3D(self):
        temp = np.ones([50, 50, 50], dtype=bool)
        temp[25, 25, 25] = False
        im2D = edt(temp) < 10
        bbox = ps.tools.find_bbox(im2D)
        assert im2D[bbox].shape == (19, 19, 19)
        im2D = edt(temp) <= 10
        bbox = ps.tools.find_bbox(im2D)
        assert im2D[bbox].shape == (21, 21, 21)
        bbox = ps.tools.find_bbox(im2D, order_by='corners')
        assert bbox == [[15, 15, 15], [36, 36, 36]]

    def test_tic_toc(self):
        from porespy.tools import tic, toc
        from time import sleep
        tic()
        sleep(1)
        t = toc(quiet=True)
        assert t > 1


if __name__ == '__main__':
    t = ToolsTest()
    self = t
    t.setup_class()
    for item in t.__dir__():
        if item.startswith('test'):
            print(f"Running test: {item}")
            t.__getattribute__(item)()
