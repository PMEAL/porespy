import os
import sys
from pathlib import Path

import numpy as np
import pyvista as pv

import porespy as ps


class ExportTest():

    def setup_class(self):
        self.path = os.path.dirname(os.path.abspath(sys.argv[0]))

    def test_export_to_palabos(self):
        X = Y = Z = 20
        S = X * Y * Z
        im = ps.generators.blobs(
            shape=[X, Y, Z], porosity=0.7, blobiness=1, periodic=False,)
        tmp = os.path.join(self.path, 'palabos.dat')
        ps.io.to_palabos(im, tmp, solid=0)
        assert os.path.isfile(tmp)
        with open(tmp) as f:
            val = f.read().splitlines()
        val = np.asarray(val).astype(int)
        assert np.size(val) == S
        assert np.sum(val == 0) + np.sum(val == 1) + np.sum(val == 2) == S
        os.remove(tmp)

    def test_to_vtk_2d(self):
        im = ps.generators.blobs(shape=[20, 20], periodic=False,)
        ps.io.to_vtk(im, filename='vtk_func_test')
        assert os.stat('vtk_func_test.vti').st_size == 831
        os.remove('vtk_func_test.vti')

    def test_to_vtk_3d(self):
        im = ps.generators.blobs(shape=[20, 20, 20], periodic=False,)
        ps.io.to_vtk(im, filename='vtk_func_test')
        assert os.stat('vtk_func_test.vti').st_size == 8433
        os.remove('vtk_func_test.vti')

    def test_dict_to_vtk(self):
        im = ps.generators.blobs(shape=[20, 20, 20], periodic=False,)
        ps.io.dict_to_vtk({'im': im}, filename="dictvtk")
        a = os.stat('dictvtk.vti').st_size
        os.remove('dictvtk.vti')
        ps.io.dict_to_vtk({'im': im, 'im_neg': ~im}, filename="dictvtk")
        b = os.stat('dictvtk.vti').st_size
        assert a < b
        os.remove('dictvtk.vti')

    def test_to_stl_openstl_and_triangles_formats(self):
        im = np.zeros((8, 8, 8), dtype=bool)
        im[2:6, 2:6, 2:6] = True

        tris_openstl = ps.io.to_stl(im, method='direct', fmt='openstl')
        tris_triangles = ps.io.to_stl(im, method='direct', fmt='triangles')

        assert tris_openstl.shape == tris_triangles.shape
        assert tris_openstl.shape[1:] == (4, 3)
        assert np.issubdtype(tris_openstl.dtype, np.number)
        assert np.array_equal(tris_openstl, tris_triangles)

        # Normals should be unit length (allowing tiny numerical error)
        mags = np.linalg.norm(tris_openstl[:, 0, :], axis=1)
        assert np.allclose(mags, 1.0, atol=1e-6)

    def test_to_stl_methods_produce_valid_triangles(self):
        im = np.zeros((10, 10, 10), dtype=bool)
        im[2:8, 2:8, 2:8] = True

        for method in ['direct', 'marching-cubes']:
            tris = ps.io.to_stl(im, method=method, fmt='openstl', voxel_size=2)

            assert tris.ndim == 3
            assert tris.shape[1:] == (4, 3)
            assert tris.shape[0] > 0
            assert np.issubdtype(tris.dtype, np.number)
            assert np.all(tris[:, 1:, :] >= 0.0)

    def test_to_stl_vfn_and_skimage_aliases(self):
        im = np.zeros((8, 8, 8), dtype=bool)
        im[1:7, 1:7, 1:7] = True

        v1, f1, n1 = ps.io.to_stl(im, method='direct', fmt='vfn')
        v2, f2, n2 = ps.io.to_stl(im, method='direct', fmt='skimage')

        assert v1.shape[1] == 3
        assert f1.shape[1] == 3
        assert n1.shape[1] == 3
        assert np.array_equal(v1, v2)
        assert np.array_equal(f1, f2)
        assert np.array_equal(n1, n2)

    def test_to_stl_pyvista_format(self):
        im = np.zeros((8, 8, 8), dtype=bool)
        im[2:6, 2:6, 2:6] = True

        mesh = ps.io.to_stl(im, method='marching-cubes', fmt='pyvista')

        assert isinstance(mesh, pv.PolyData)
        assert mesh.n_points > 0
        assert mesh.n_cells > 0

    def test_to_stl_remove_duplicates_reduces_indexed_mesh_size(self):
        im = np.zeros((8, 8, 8), dtype=bool)
        im[2:6, 2:6, 2:6] = True

        v0, f0, _ = ps.io.to_stl(im, method='direct', fmt='vfn', remove_duplicates=False)
        v1, f1, _ = ps.io.to_stl(im, method='direct', fmt='vfn', remove_duplicates=True)
        v2, f2, _ = ps.io.to_stl(
            im,
            method='direct',
            fmt='vfn',
            remove_duplicates=True,
            tol=1.0,
        )

        assert v1.shape[0] <= v0.shape[0]
        assert f1.shape[0] <= f0.shape[0]
        assert v2.shape[0] <= v0.shape[0]
        assert f2.shape[0] <= f0.shape[0]

    def test_zip_to_stack_and_folder_to_stack(self):
        p = Path(os.path.realpath(__file__),
                 '../../../test/fixtures/blobs_layers.zip').resolve()
        im = ps.io.zip_to_stack(p)
        assert im.shape == (100, 100, 10)
        p = Path(os.path.realpath(__file__),
                 '../../../test/fixtures/blobs_layers').resolve()
        im = ps.io.folder_to_stack(p)
        assert im.shape == (100, 100, 10)


if __name__ == "__main__":
    t = ExportTest()
    self = t
    t.setup_class()
    for item in t.__dir__():
        if item.startswith("test"):
            print(f"Running test: {item}")
            t.__getattribute__(item)()
