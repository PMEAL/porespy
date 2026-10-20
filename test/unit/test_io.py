import io
import os
import stat
import sys
from pathlib import Path
from unittest.mock import patch
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import imageio.v3 as iio
import numpy as np
import pyvista as pv
import pytest

import porespy as ps


def _triangle_area(tris):
    a = tris[:, 2] - tris[:, 1]
    b = tris[:, 3] - tris[:, 1]
    return np.linalg.norm(np.cross(a, b), axis=1).sum() / 2


class STLBoundaryTest():

    @pytest.mark.parametrize('method', ['direct', 'marching-cubes'])
    def test_closed_default_and_contained_object(self, method):
        im = np.zeros((8, 8, 8), dtype=bool)
        im[2:6, 2:6, 2:6] = True
        default = ps.io.to_stl(im, method=method)
        closed = ps.io.to_stl(im, method=method, close_faces=True)
        opened = ps.io.to_stl(im, method=method, close_faces=False)
        assert np.array_equal(default, closed)
        assert np.array_equal(closed, opened)

    @pytest.mark.parametrize('method', ['direct', 'marching-cubes'])
    @pytest.mark.parametrize('axis', [0, 1, 2])
    @pytest.mark.parametrize('complement', [False, True])
    @pytest.mark.parametrize('voxel_size', [0.5, 1, 2])
    def test_open_half_block(self, method, axis, complement, voxel_size):
        im = np.zeros((8, 8, 8), dtype=bool)
        slices = [slice(None)] * 3
        slices[axis] = slice(0, 4)
        im[tuple(slices)] = True
        if complement:
            im = ~im
        tris = ps.io.to_stl(
            im, method=method, voxel_size=voxel_size, close_faces=False,
        )
        # Direct faces span entire voxels; marching cubes spans voxel samples.
        width = 8 if method == 'direct' else 7
        location = 4 if method == 'direct' else 4.5
        assert len(tris) == 2 * width**2
        assert np.isclose(_triangle_area(tris), width**2 * voxel_size**2)
        assert np.allclose(tris[:, 1:, axis], location * voxel_size)
        tangential = np.delete(tris[:, 1:, :], axis, axis=2)
        low = 0 if method == 'direct' else 1
        assert np.isclose(tangential.min(), low * voxel_size)
        assert np.isclose(tangential.max(), 8 * voxel_size)
        # Switching the flag preserves the position of the internal plane.
        closed = ps.io.to_stl(im, method=method, voxel_size=voxel_size)
        plane = np.all(np.isclose(closed[:, 1:, axis], location * voxel_size), axis=1)
        assert plane.any()
        if method == 'direct':
            assert np.isclose(_triangle_area(closed), 256 * voxel_size**2)
        other = ps.io.to_stl(
            ~im, method=method, voxel_size=voxel_size, close_faces=False,
        )
        assert np.isclose(_triangle_area(other), _triangle_area(tris))
        assert np.allclose(other[:, 0].mean(axis=0), -tris[:, 0].mean(axis=0))
        geometric_normals = np.cross(tris[:, 2] - tris[:, 1], tris[:, 3] - tris[:, 1])
        geometric_normals = geometric_normals.astype(float)
        geometric_normals /= np.linalg.norm(geometric_normals, axis=1, keepdims=True)
        assert np.allclose(tris[:, 0], geometric_normals)

    @pytest.mark.parametrize('voxel_size', [0.5, 1, 2])
    def test_direct_internal_face_count(self, voxel_size):
        im = np.random.default_rng(0).random((5, 6, 7)) > 0.5
        face_count = sum(np.count_nonzero(np.diff(im, axis=axis)) for axis in range(3))
        tris = ps.io.to_stl(im, voxel_size=voxel_size, close_faces=False)
        assert len(tris) == 2 * face_count
        assert np.isclose(_triangle_area(tris), face_count * voxel_size**2)
        for axis in range(3):
            for boundary in [0, im.shape[axis] * voxel_size]:
                assert not np.any(np.all(tris[:, 1:, axis] == boundary, axis=1))
        other = ps.io.to_stl(~im, voxel_size=voxel_size, close_faces=False)
        assert len(other) == len(tris)
        assert np.isclose(_triangle_area(other), _triangle_area(tris))

    @pytest.mark.parametrize('method', ['direct', 'marching-cubes'])
    @pytest.mark.parametrize('fmt', [
        'openstl', 'triangles', 'vfn', 'skimage', 'pyvista',
        'trimesh', 'open3d', 'numpy-stl', 'meshio',
    ])
    @pytest.mark.parametrize('remove_duplicates', [False, True])
    @pytest.mark.parametrize('value', [False, True])
    def test_empty_open_mesh(self, method, fmt, remove_duplicates, value):
        optional = {'trimesh': 'trimesh', 'open3d': 'open3d',
                    'numpy-stl': 'stl', 'meshio': 'meshio'}
        if fmt in optional:
            pytest.importorskip(optional[fmt])
        mesh = ps.io.to_stl(
            np.full((4, 5, 6), value, dtype=bool), method=method, fmt=fmt,
            close_faces=False, remove_duplicates=remove_duplicates, tol=0.5,
        )
        if fmt in ['openstl', 'triangles']:
            assert mesh.shape == (0, 4, 3)
        elif fmt in ['vfn', 'skimage']:
            v, f, n = mesh
            assert v.shape == f.shape == n.shape == (0, 3)
            assert np.issubdtype(f.dtype, np.integer)
        elif fmt == 'pyvista':
            assert mesh.n_points == mesh.n_cells == 0
        elif fmt in ['trimesh', 'open3d']:
            assert np.asarray(mesh.vertices).shape == (0, 3)
            faces = mesh.faces if fmt == 'trimesh' else mesh.triangles
            assert np.asarray(faces).shape == (0, 3)
        elif fmt == 'numpy-stl':
            assert mesh.vectors.shape == (0, 3, 3)
        else:
            assert mesh.points.shape == mesh.cells_dict['triangle'].shape == (0, 3)

    @pytest.mark.parametrize('axis', [0, 1, 2])
    def test_open_marching_cubes_without_sampling_cells(self, axis):
        shape = [4, 5, 6]
        shape[axis] = 1
        im = np.zeros(shape, dtype=bool)
        im[0] = True
        tris = ps.io.to_stl(im, method='marching-cubes', close_faces=False)
        assert tris.shape == (0, 4, 3)

    @pytest.mark.parametrize('method', ['direct', 'marching-cubes'])
    @pytest.mark.parametrize('remove_duplicates', [False, True])
    def test_open_indexed_formats(self, method, remove_duplicates):
        im = np.zeros((8, 8, 8), dtype=bool)
        im[:4] = True
        options = dict(method=method, close_faces=False, remove_duplicates=remove_duplicates)
        tris = ps.io.to_stl(im, fmt='triangles', **options)
        v, f, n = ps.io.to_stl(im, fmt='vfn', **options)
        assert np.array_equal(v[f], tris[:, 1:])
        assert np.array_equal(n, tris[:, 0])
        mesh = ps.io.to_stl(im, fmt='pyvista', **options)
        assert mesh.n_cells == len(tris)
        assert np.isclose(mesh.area, _triangle_area(tris))
        if remove_duplicates:
            assert mesh.n_open_edges > 0

    @pytest.mark.parametrize('method', ['direct', 'marching-cubes'])
    @pytest.mark.parametrize('remove_duplicates', [False, True])
    @pytest.mark.parametrize('fmt', ['trimesh', 'open3d', 'numpy-stl', 'meshio'])
    def test_open_optional_formats(self, method, remove_duplicates, fmt):
        modules = {'trimesh': 'trimesh', 'open3d': 'open3d',
                   'numpy-stl': 'stl', 'meshio': 'meshio'}
        pytest.importorskip(modules[fmt])
        im = np.zeros((8, 8, 8), dtype=bool)
        im[:4] = True
        options = dict(method=method, close_faces=False, remove_duplicates=remove_duplicates)
        expected = ps.io.to_stl(im, fmt='triangles', **options)[:, 1:]
        mesh = ps.io.to_stl(im, fmt=fmt, **options)
        if fmt == 'trimesh':
            triangles = mesh.vertices[mesh.faces]
        elif fmt == 'open3d':
            triangles = np.asarray(mesh.vertices)[np.asarray(mesh.triangles)]
        elif fmt == 'numpy-stl':
            triangles = mesh.vectors
        else:
            triangles = mesh.points[mesh.cells_dict['triangle']]
        assert np.array_equal(triangles, expected)


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
        zipped = ps.io.zip_to_stack(p)
        assert zipped.shape == (100, 100, 10)
        p = Path(os.path.realpath(__file__),
                 '../../../test/fixtures/blobs_layers').resolve()
        im = ps.io.folder_to_stack(p)
        assert im.shape == (100, 100, 10)
        assert im.dtype == zipped.dtype
        assert np.array_equal(im, zipped)


@pytest.fixture(params=["folder", "zip"])
def stack_source(request, tmp_path):
    def read(slices, **kwargs):
        folder = tmp_path / "slices"
        folder.mkdir()
        for name, value in slices.items():
            path = folder / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(value, bytes):
                path.write_bytes(value)
            else:
                iio.imwrite(path, value)
        if request.param == "folder":
            return ps.io.folder_to_stack(folder, **kwargs)
        archive = tmp_path / "slices.zip"
        with ZipFile(archive, "w", compression=ZIP_DEFLATED) as zf:
            for path in folder.rglob("*"):
                if path.is_file():
                    zf.write(path, path.relative_to(folder).as_posix())
        result = ps.io.zip_to_stack(archive, **kwargs)
        # Loading an archive must preserve a pre-existing same-named directory.
        assert all((folder / name).is_file() for name in slices)
        return result

    return read


class ImageStackTest:
    @pytest.mark.parametrize(
        "extension,dtype",
        [
            (".tif", np.uint16),
            (".tiff", np.float32),
            (".tif", bool),
            (".png", np.uint8),
            (".png", np.uint16),
            (".bmp", np.uint8),
        ],
    )
    def test_order_values_dtype_and_metadata(self, stack_source, extension, dtype):
        # Insertion order, lexical order and natural order are all different.
        slices = {
            f"slice10{extension}": np.full((2, 5), 10, dtype=dtype),
            f"Slice2{extension}": np.full((2, 5), 2, dtype=dtype),
            f"slice1{extension}": np.arange(10).reshape(2, 5).astype(dtype),
            ".DS_Store": b"metadata",
            f"._slice1{extension}": b"resource fork",
            "__MACOSX/ignored.tif": b"metadata",
            "README.txt": b"notes",
        }
        result = stack_source(slices)
        expected = np.stack(
            [
                slices[f"slice1{extension}"],
                slices[f"Slice2{extension}"],
                slices[f"slice10{extension}"],
            ],
            axis=2,
        )
        assert result.dtype == expected.dtype
        assert np.array_equal(result, expected)

    @pytest.mark.parametrize(
        "slices,message",
        [
            ({}, "No supported image"),
            ({"notes.txt": b"notes"}, "No supported image"),
            (
                {
                    "1.tif": np.ones((2, 5), dtype=np.uint8),
                    "2.tif": np.ones((1, 5), dtype=np.uint8),
                },
                "shape",
            ),
            (
                {
                    "1.tif": np.ones((2, 5), dtype=np.uint8),
                    "2.tif": np.ones((2, 5), dtype=np.uint16),
                },
                "dtype",
            ),
            ({"1.png": np.zeros((2, 5, 3), dtype=np.uint8)}, "2D grayscale"),
            ({"1.tif": np.zeros((2, 5, 3), dtype=np.uint8)}, "2D grayscale"),
            (
                {
                    "1.tif": np.ones((2, 5), dtype=np.uint8),
                    "01.tif": np.ones((2, 5), dtype=np.uint8),
                },
                "ambiguous",
            ),
            ({"broken.tif": b"invalid image data"}, "broken.tif"),
        ],
    )
    def test_invalid_slices_fail_before_pixel_decode(self, stack_source, slices, message):
        with patch("porespy.io._unzipper.iio.imread") as decode:
            with pytest.raises(ValueError, match=message):
                stack_source(slices)
        decode.assert_not_called()

    def test_limit_is_checked_before_allocating_or_decoding(self, stack_source):
        slices = {"1.png": np.ones((2, 5), dtype=np.uint16)}
        with (
            patch("porespy.io._unzipper.iio.imread") as decode,
            patch(
                "porespy.io._unzipper.np.empty", side_effect=AssertionError("allocated stack")
            ),
        ):
            with pytest.raises(ValueError, match="requires 20 bytes"):
                stack_source(slices, max_bytes=19)
        decode.assert_not_called()

    def test_single_slice_and_exact_limit(self, stack_source):
        layer = np.array([[0, 500], [65535, 42]], dtype=np.uint16)
        result = stack_source({"001.tif": layer}, max_bytes=layer.nbytes)
        assert result.shape == (2, 2, 1)
        assert np.array_equal(result[..., 0], layer)

    @pytest.mark.parametrize("limit", [-1, 1.5, True])
    def test_invalid_limit(self, stack_source, limit):
        with pytest.raises(ValueError, match="max_bytes must be"):
            stack_source({"1.png": np.zeros((2, 5), dtype=np.uint8)}, max_bytes=limit)

    @pytest.mark.parametrize("failure", ["dtype", "shape", "decode"])
    def test_decoded_slice_is_validated(self, stack_source, failure):
        layer = np.zeros((2, 5), dtype=np.uint8)
        if failure == "decode":
            decoder = patch(
                "porespy.io._unzipper.iio.imread", side_effect=OSError("bad pixels")
            )
            message = "Could not read slice '1.png'"
        else:
            decoded = layer.astype(np.uint16) if failure == "dtype" else layer[:1]
            decoder = patch("porespy.io._unzipper.iio.imread", return_value=decoded)
            message = "shape .* and dtype"
        with decoder, pytest.raises(ValueError, match=message):
            stack_source({"1.png": layer})

    def test_zip_wrapper_folder_and_binary_stream(self):
        buffer = io.BytesIO()
        with ZipFile(buffer, "w") as zf:
            zf.writestr("sample/", b"")
            for name, value in [("10.png", 10), ("2.png", 2)]:
                zf.writestr(
                    f"sample/{name}",
                    iio.imwrite(
                        "<bytes>",
                        np.full((2, 5), value, dtype=np.uint8),
                        extension=".png",
                    ),
                )
            zf.writestr("__MACOSX/sample/._2.png", b"metadata")
        buffer.seek(0)
        result = ps.io.zip_to_stack(buffer)
        assert not buffer.closed
        assert result.shape == (2, 5, 2)
        assert np.array_equal(result[0, 0], [2, 10])

    @pytest.mark.parametrize(
        "names,message",
        [
            (["a/1.png", "b/2.png"], "one folder"),
            (["../1.png"], "Unsafe"),
            (["/1.png"], "Unsafe"),
            ([r"a\1.png"], "Unsafe"),
            (["C:/1.png"], "Unsafe"),
            (["1.png", "1.png"], "duplicate"),
        ],
    )
    def test_zip_rejects_unsafe_or_ambiguous_members(self, tmp_path, names, message):
        archive = tmp_path / "slices.zip"
        with ZipFile(archive, "w") as zf:
            for index, name in enumerate(names):
                # Preserve malformed names verbatim: ZipInfo normally replaces
                # backslashes with slashes on Windows, even when writing a ZIP.
                member = ZipInfo(name)
                member.filename = name
                if index and name == names[index - 1]:
                    with pytest.warns(UserWarning, match="Duplicate name"):
                        zf.writestr(member, b"not decoded")
                else:
                    zf.writestr(member, b"not decoded")
        with pytest.raises(ValueError, match=message):
            ps.io.zip_to_stack(archive)
        assert sorted(p.name for p in tmp_path.iterdir()) == ["slices.zip"]

    def test_zip_checks_original_name_after_windows_normalization(self, tmp_path):
        archive = tmp_path / "slices.zip"
        member = ZipInfo(r"a\1.png")
        member.filename = member.orig_filename
        with ZipFile(archive, "w") as zf:
            zf.writestr(member, b"not decoded")
        with ZipFile(archive) as zf:
            # Simulate Windows on every platform. ZipInfo retains orig_filename
            # when normalizing filename as the ZIP is opened.
            member = zf.infolist()[0]
            assert member.orig_filename == r"a\1.png"
            member.filename = "a/1.png"
            with (
                patch("porespy.io._unzipper.ZipFile", return_value=zf),
                patch("porespy.io._unzipper.iio.improps") as inspect_header,
                pytest.raises(ValueError, match="Unsafe image member path"),
            ):
                ps.io.zip_to_stack(archive)
            inspect_header.assert_not_called()

    @pytest.mark.parametrize("valid", [False, True])
    def test_zip_preserves_existing_directory(self, tmp_path, monkeypatch, valid):
        monkeypatch.chdir(tmp_path)
        folder = tmp_path / "slices"
        folder.mkdir()
        sentinel = folder / "important.txt"
        sentinel.write_text("keep")
        archive = tmp_path / "slices.zip"
        with ZipFile(archive, "w") as zf:
            if valid:
                zf.writestr(
                    "1.png",
                    iio.imwrite(
                        "<bytes>",
                        np.zeros((2, 5), dtype=np.uint8),
                        extension=".png",
                    ),
                )
            else:
                zf.writestr("broken.tif", b"invalid image")
        if valid:
            assert ps.io.zip_to_stack(archive).shape == (2, 5, 1)
        else:
            with pytest.raises(ValueError, match="broken.tif"):
                ps.io.zip_to_stack(archive)
        assert sentinel.read_text() == "keep"
        assert sorted(p.name for p in folder.iterdir()) == ["important.txt"]

    def test_folder_image_symlinks_are_refused(self, tmp_path):
        folder = tmp_path / "slices"
        folder.mkdir()
        image = tmp_path / "actual.png"
        iio.imwrite(image, np.zeros((2, 5), dtype=np.uint8))
        try:
            (folder / "1.png").symlink_to(image)
        except (OSError, NotImplementedError):
            pytest.skip("Creating symbolic links is not permitted on this platform")
        with pytest.raises(ValueError, match="Symbolic"):
            ps.io.folder_to_stack(folder)

    def test_zip_image_symlinks_are_refused(self, tmp_path):
        member = ZipInfo("1.png")
        member.create_system = 3
        member.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive = tmp_path / "slices.zip"
        with ZipFile(archive, "w") as zf:
            zf.writestr(member, "actual.png")
        with pytest.raises(ValueError, match="Symbolic"):
            ps.io.zip_to_stack(archive)

    @pytest.mark.parametrize("series", [False, True])
    def test_multi_page_tiff_is_refused_before_decode(self, stack_source, series):
        import tifffile

        buffer = io.BytesIO()
        with tifffile.TiffWriter(buffer) as tif:
            if series:
                tif.write(np.ones((2, 5), dtype=np.uint8))
                tif.write(np.ones((3, 6), dtype=np.uint8))
            else:
                tif.write(np.ones((2, 4, 5), dtype=np.uint8), photometric="minisblack")
        with patch("porespy.io._unzipper.iio.imread") as decode:
            with pytest.raises(ValueError, match="exactly one"):
                stack_source({"1.tif": buffer.getvalue()})
        decode.assert_not_called()


if __name__ == "__main__":
    t = ExportTest()
    self = t
    t.setup_class()
    for item in t.__dir__():
        if item.startswith("test"):
            print(f"Running test: {item}")
            t.__getattribute__(item)()
