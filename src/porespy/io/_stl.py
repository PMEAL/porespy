import numpy as np
import pyvista as pv
import scipy.ndimage as spim
import skimage.measure as ms
from numba import njit


__all__ = [
    "from_stl",
    "to_stl",
]


def _shell(stl_path, density, fill_holes=False):
    r"""
    Converts an STL file to a Numpy boolean array using Open3D shell voxelization

    Parameters
    ----------
    stl_path : str or path object
        The location and name of the STL file
    density : int
        Controls the resolution of the final image. This is the number of
        voxels along the longest axis of the bounding box, with higher
        values leading to more voxels and hence better resolution.
    fill_holes : bool (default=False)
        If `True` then steps are taken to ensure the mesh has no holes.
        This process can be slow so is not performed unless requested.

    Returns
    -------
    im : ndarray
        A boolean array with `True` values indicating the solid phase.

    Notes
    -----
    This method uses Open3D's ``VoxelGrid.create_from_triangle_mesh`` to
    voxelize the surface shell, then fills the interior with
    ``scipy.ndimage.binary_fill_holes``.  It is well suited for converting
    packings of particles (e.g. from DEM) into voxel images but might not
    work well on more nebulous shapes.
    """
    try:
        import open3d as o3d
    except ImportError:
        msg = 'open3d is required to use this function, which requires Python<=3.12'
        raise ImportError(msg)

    # Read the mesh
    mesh = o3d.io.read_triangle_mesh(stl_path)

    # Ensure the mesh is watertight
    mesh.compute_vertex_normals()
    mesh.orient_triangles()

    # Fill holes if any
    if fill_holes:
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        mesh = mesh.fill_holes()
        mesh = mesh.to_legacy()

    # Compute voxel size based on resolution
    bbox = mesh.get_axis_aligned_bounding_box()
    extent = bbox.get_max_bound() - bbox.get_min_bound()
    voxel_size = np.max(extent) / density

    # Create voxel grid
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(
        mesh,
        voxel_size=voxel_size
    )

    # Get voxels and convert to numpy array
    voxels = voxel_grid.get_voxels()

    if not voxels:  # Check if voxels list is empty
        return np.array([], dtype=bool)

    # Get the dimensions of the voxel grid
    voxel_indices = np.array([voxel.grid_index for voxel in voxels])
    grid_dims = voxel_indices.max(axis=0) + 1

    # Create boolean array and fill it
    voxel_array = np.zeros(grid_dims, dtype=bool)
    voxel_array[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = True

    # Fill inside of objects
    voxel_array = spim.binary_fill_holes(voxel_array)

    return voxel_array


def _enclosed(stl_path, density):
    r"""
    Converts an STL file to a boolean array using PyVista's enclosed-points test

    Parameters
    ----------
    stl_path : str or path object
        The location and name of the STL file
    density : int
        The number of voxels along the longest axis of the bounding box.
        Higher values give higher resolution but use more memory.

    Returns
    -------
    im : ndarray
        A boolean array with `True` values indicating the solid phase.

    Notes
    -----
    This method creates a regular grid over the mesh bounding box, then uses
    PyVista's ``select_enclosed_points`` to identify which grid points lie
    inside the surface.  It can be slow and memory-intensive for large meshes.
    """
    mesh = pv.read(stl_path)
    x_min, x_max, y_min, y_max, z_min, z_max = mesh.bounds
    extent = max(x_max - x_min, y_max - y_min, z_max - z_min)
    voxel_size = extent / density
    x = np.arange(x_min, x_max, voxel_size)
    y = np.arange(y_min, y_max, voxel_size)
    z = np.arange(z_min, z_max, voxel_size)
    x, y, z = np.meshgrid(x, y, z)

    # Create unstructured grid from the structured grid
    grid = pv.StructuredGrid(x, y, z)
    ugrid = pv.UnstructuredGrid(grid)

    # Get part of the mesh within the mesh's bounding surface
    selection = ugrid.select_enclosed_points(
        mesh.extract_surface(),
        tolerance=0.0,
        check_surface=False,
    )
    mask = selection['SelectedPoints'].view(bool)
    mask = mask.reshape(x.shape, order='F')
    mask = np.array(mask)
    return mask


def _voxelize(stl_path, density):
    r"""
    Converts an STL file to a boolean array using PyVista's voxelize_volume

    Parameters
    ----------
    stl_path : str or path object
        The location and name of the STL file
    density : int
        The number of voxels along the longest axis of the bounding box.
        Higher values give higher resolution but use more memory.

    Returns
    -------
    im : ndarray
        A boolean array with `True` values indicating the solid phase.

    Notes
    -----
    This method uses PyVista's ``voxelize_volume`` which internally creates a
    volumetric representation of the mesh at the given density.
    """
    mesh = pv.read(stl_path)
    bounds = mesh.bounds
    extent = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])
    voxel_size = extent / density
    vox = pv.voxelize_volume(mesh, density=voxel_size, check_surface=False)
    im = np.reshape(
        vox['InsideMesh'],
        np.array(vox.meshgrid[0].shape)-1,
        order='F',
    )
    return im.astype(bool)


def from_stl(stl_path, method='shell', density=100, fill_holes=False):
    r"""
    Converts an STL file to a boolean array

    Parameters
    ----------
    stl_path : str or path object
        The location and name of the STL file
    method : str
        The voxelization method to use.  Options are:

        ============ ===========================================================
        method       Description
        ============ ===========================================================
        ``'shell'``  Uses Open3D to voxelize the surface shell, then fills the
                     interior. This is the default and generally the fastest.
        ``'voxelize'`` Uses PyVista's ``voxelize_volume`` to create a volumetric
                     representation of the mesh.
        ``'enclosed'`` Uses PyVista's ``select_enclosed_points`` to test every
                     grid point against the surface. Can be slow and
                     memory-intensive for large meshes.
        ============ ===========================================================

    density : int
        The number of voxels along the longest axis of the bounding box.
        Higher values give higher resolution but use more memory. The default
        is 100 for all methods.
    fill_holes : bool
        If `True` then steps are taken to ensure the mesh has no holes before
        voxelization. Only used when ``method='shell'``. The default is
        `False`.

    Returns
    -------
    im : ndarray
        A boolean array with `True` values indicating the solid phase.

    """
    if method == 'shell':
        im = _shell(stl_path=stl_path, density=density, fill_holes=fill_holes)
    elif method == 'voxelize':
        im = _voxelize(stl_path=stl_path, density=density)
    elif method == 'enclosed':
        im = _enclosed(stl_path=stl_path, density=density)
    else:
        raise ValueError(
            f"Unsupported method '{method}'. "
            "Options are 'shell', 'voxelize', or 'enclosed'."
        )
    return im


def to_stl(
    im,
    filename=None,
    voxel_size=1,
    method='direct',
    fmt='openstl',
    remove_duplicates=False,
    tol=None,
):
    r"""
    Converts an voxel image to an STL mesh in a variety of formats

    Parameters
    ----------
    im : 3D image
        The image of the porous material
    voxel_size : int
        The side length of the voxels (voxels  are cubic)
    method : str
        Can be one of the options listed below:

        ================ ============================================================
        method           Description
        ================ ============================================================
        'direct'         Converts each exposed face of each voxel into 2 triangles
                         to generate a mesh that maps directly to the voxel image.
        'marching-cubes' Uses the marching cubes method in scikit-image to generate
                         a mesh. The result is naturally smoother than the 'direct'
                         approach, but this is due to last information.
        ================ ============================================================

    fmt : str
        The format of the returned mesh. Options are:

        - 'openstl' or 'triangles' (default)
        - 'skimage' or 'vfn' (as in verts, faces, vertex normals)
        - 'pyvista' (installed with PoreSpy)
        - 'trimesh' (not automatically installed with PoreSpy)
        - 'open3d' (not automatically installed with PoreSpy)
        - 'numpy-stl' (not automatically installed with PoreSpy)
        - 'meshio' (not automatically installed with PoreSpy)

    Notes
    -----
    The `openstl` package has the fastest read/write performance. It
    uses a basic numpy array of shape `[N, 4, 3]`. `N` is the number of
    triangles, `4` refers to `norms, vert1, vert2, vert3`, where `norms` and
    `vert<i>` are each `3` components long. Nn 'stl' file saved using `openstl`
    can be opened by most other packages, which convert it to a usable mesh.
    For example `mesh = pyvista.read('openstl-formatted-file.stl')` will create
    `mesh` that can be plotted with `pyvista.plot(mesh, eye_dome_lighting=True)`.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/io/reference/to_stl.html>`_
    to view online example.

    """
    # filename = sanitize_filename(filename, ext="stl", exclude_ext=True)
    if len(im.shape) == 2:
        im = im[:, :, np.newaxis]
    if method == 'marching-cubes':
        tris = _to_stl_marching_cubes(im, voxel_size)
    elif method == 'direct':
        tris = _to_stl_porespy(im, voxel_size)
    else:
        raise ValueError(f"{method} is not a supported method")

    if fmt == 'openstl' and not remove_duplicates:
        return tris

    v, f, n = tris_to_vfn(tris)
    if remove_duplicates:
        v, f = remove_duplicate_vertices(v, f, tol=tol)
        v, f = remove_duplicate_faces(verts=v, faces=f)
        tris = vfn_to_tris(v, f)
        n = tris[:, 0, :]

    if fmt in ['openstl', 'triangles']:
        return tris
    elif fmt in ['skimage', 'vfn']:
        mesh = v, f, n
    elif fmt == 'pyvista':
        mesh = pv.PolyData.from_regular_faces(v, f)
    elif fmt == 'trimesh':
        try:
            import trimesh
        except ModuleNotFoundError:
            msg = "trimesh can be installed with pip install trimesh"
            raise ModuleNotFoundError(msg)
        mesh = trimesh.Trimesh(vertices=v, faces=f, face_normals=n, process=False)
    elif fmt == 'open3d':
        try:
            import open3d as o3d
        except ModuleNotFoundError:
            msg = "open3d can be installed with pip install open3d"
            raise ModuleNotFoundError(msg)
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(v)
        mesh.triangles = o3d.utility.Vector3iVector(f.astype(np.int32))
        mesh.compute_vertex_normals()
    elif fmt == 'numpy-stl':
        try:
            from stl import mesh as stl_mesh
        except ModuleNotFoundError:
            msg = "numpy-stl can be installed with pip install numpy-stl"
            raise ModuleNotFoundError(msg)
        mesh = stl_mesh.Mesh(np.zeros(f.shape[0], dtype=stl_mesh.Mesh.dtype))
        mesh.vectors = v[f].astype(np.float32, copy=False)
        mesh.normals = n.astype(np.float32, copy=False)
    elif fmt == 'meshio':
        try:
            import meshio
        except ModuleNotFoundError:
            msg = "meshio can be installed with pip install meshio"
            raise ModuleNotFoundError(msg)
        mesh = meshio.Mesh(
            points=v,
            cells=[("triangle", f)],
            cell_data={"Normals": [n]},
        )
    else:
        raise ValueError(f"{fmt} is not a supported format")
    return mesh


def tris_to_vfn(tris):
    """
    Convert openstl-style triangles to vertices/faces/face-normals.

    Parameters
    ----------
    tris : ndarray
        Shape (N, 4, 3), laid out as [normal, v0, v1, v2].

    Returns
    -------
    verts : ndarray
        Vertex coordinates, shape (M, 3)
    faces : ndarray
        Face indices, shape (N, 3)
    face_normals : ndarray
        Face normals, shape (N, 3)
    """
    tris = np.asarray(tris)
    if tris.ndim != 3 or tris.shape[1:] != (4, 3):
        raise ValueError("tris must have shape (N, 4, 3)")

    n = tris[:, 0, :].astype(np.float32, copy=False)
    v = tris[:, 1:, :].reshape(-1, 3).astype(np.float32, copy=False)
    f = np.arange(v.shape[0], dtype=np.int64).reshape(-1, 3)

    return v, f, n


def vfn_to_tris(verts, faces, vertex_normals=None, voxel_size=1):
    """
    Convert indexed format (vertices/faces/vertex-normals) output to
    openstl-style triangle format.

    Parameters
    ----------
    verts : ndarray
        Vertex coordinates of shape (V, 3).
    faces : ndarray
        Triangle indices of shape (F, 3).
    vertex_normals : ndarray, optional
        Vertex normals from skimage.measure.marching_cubes, shape (V, 3).
        Accepted for API compatibility, but face normals are recomputed from
        geometry because openstl stores one normal per face.
    voxel_size : float
        Scalar voxel size used to scale vertex coordinates.

    Returns
    -------
    tris : ndarray
        Array of shape (F, 4, 3) in openstl layout:
        [face_normal, v0, v1, v2]
    """
    verts = np.asarray(verts, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)

    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError("verts must have shape (V, 3)")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape (F, 3)")

    verts = verts * voxel_size

    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]

    # Recompute per-face normals for openstl
    face_normals = np.cross(v1 - v0, v2 - v0)
    mag = np.linalg.norm(face_normals, axis=1, keepdims=True)
    mag[mag == 0] = 1.0
    face_normals = face_normals / mag

    tris = np.empty((faces.shape[0], 4, 3), dtype=np.float32)
    tris[:, 0] = face_normals
    tris[:, 1] = v0
    tris[:, 2] = v1
    tris[:, 3] = v2
    return tris


def remove_duplicate_vertices(verts, faces, tol=None):
    """
    Deduplicate vertices in an indexed mesh, remapping faces to the new indices.

    Parameters
    ----------
    verts : (V, 3) array
        Vertex coordinates.
    faces : (F, 3) array
        Triangle vertex indices.
    tol : float or None
        If None, use exact equality. Otherwise quantize coordinates by `tol`
        before comparison. A good value is the `voxel_size`.

    Returns
    -------
    verts : (V2, 3) array
        Deduplicated vertex coordinates, V2 <= V.
    faces : (F, 3) array
        Face indices remapped to the new vertex array. Face count is unchanged.
    """
    verts = np.asarray(verts, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)

    if tol is None:
        _, unique_idx, inverse = np.unique(
            verts, axis=0, return_index=True, return_inverse=True
        )
    else:
        key = np.rint(verts / tol).astype(np.int64)
        _, unique_idx, inverse = np.unique(
            key, axis=0, return_index=True, return_inverse=True
        )

    order = np.argsort(unique_idx)
    verts = verts[unique_idx[order]]
    remap = np.empty(order.shape[0], dtype=np.int64)
    remap[order] = np.arange(order.shape[0], dtype=np.int64)
    faces = remap[inverse][faces]

    return verts, faces


def remove_duplicate_faces(faces=None, verts=None, tris=None, tol=None):
    """
    Remove duplicate faces from an indexed mesh or a triangle soup.

    For an indexed mesh, two faces are duplicates if they reference the same
    set of vertex indices (regardless of winding order). For a triangle soup,
    two triangles are duplicates if their vertex coordinates are the same set
    (optionally within `tol`). Normals are recomputed from geometry when
    `tris` is returned.

    Parameters
    ----------
    faces : (F, 3) int array, optional
        Face vertex indices. Provide together with `verts` for the indexed
        mesh path.
    verts : (V, 3) float array, optional
        Vertex coordinates. Returned unchanged; provided only so the caller
        does not have to unpack the pair.
    tris : (N, 4, 3) float array, optional
        Triangle soup in openstl layout ``[normal, v0, v1, v2]``. Provide
        instead of `faces`/`verts` for the triangle-soup path.
    tol : float or None
        Coordinate tolerance used only for the triangle-soup path. If None,
        exact float equality is used. A good value is the `voxel_size`.

    Returns
    -------
    If `tris` was provided:
        tris : (M, 4, 3) array  —  M <= N, with recomputed normals.
    If `faces` was provided:
        verts : (V, 3) array  —  unchanged.
        faces : (F2, 3) array  —  F2 <= F.
    """
    if tris is not None:
        tris = np.asarray(tris, dtype=np.float32)
        v_3x3 = tris[:, 1:, :].copy()  # (N, 3, 3)
        if tol is not None:
            v_3x3 = np.rint(v_3x3 / tol).astype(np.int64)
        # Sort the 3 vertices within each triangle for order-independent comparison
        for i in range(len(v_3x3)):
            v_3x3[i] = v_3x3[i][np.lexsort(v_3x3[i].T[::-1])]
        key = v_3x3.reshape(-1, 9)
        _, keep = np.unique(key, axis=0, return_index=True)
        keep = np.sort(keep)
        tris = tris[keep].copy()
        # Recompute normals for the kept triangles
        v0, v1, v2 = tris[:, 1], tris[:, 2], tris[:, 3]
        fn = np.cross(v1 - v0, v2 - v0)
        mag = np.linalg.norm(fn, axis=1, keepdims=True)
        mag[mag == 0] = 1.0
        tris[:, 0] = fn / mag
        return tris
    else:
        faces = np.asarray(faces, dtype=np.int64)
        faces_sorted = np.sort(faces, axis=1)
        _, keep = np.unique(faces_sorted, axis=0, return_index=True)
        faces = faces[np.sort(keep)]
        if verts is not None:
            return verts, faces
        return faces


def _to_stl_marching_cubes(im, voxel_size):
    r"""
    Helper method to convert an array to an STL file.

    Parameters
    ----------
    im : 3D image
        The image of the porous material
    voxel_size : int
        The side length of the voxels (voxels are cubic)

    """
    mask = np.pad(im, pad_width=1, mode="constant", constant_values=False)
    verts, faces, _, _ = ms.marching_cubes(mask)
    verts = verts * voxel_size

    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]

    fn = np.cross(v1 - v0, v2 - v0)
    mag = np.linalg.norm(fn, axis=1, keepdims=True)
    mag[mag == 0] = 1.0
    fn = fn / mag

    tris = np.empty((faces.shape[0], 4, 3), dtype=np.float32)
    tris[:, 0] = fn
    tris[:, 1] = v0
    tris[:, 2] = v1
    tris[:, 3] = v2
    return tris


def _to_stl_porespy(im, voxel_size=1):
    mask = np.pad(im, pad_width=1, mode='constant', constant_values=False)
    xm, xp, ym, yp, zm, zp = _exposed_face_masks(mask)
    xm = xm[1:-1, 1:-1, 1:-1]
    xp = xp[1:-1, 1:-1, 1:-1]
    ym = ym[1:-1, 1:-1, 1:-1]
    yp = yp[1:-1, 1:-1, 1:-1]
    zm = zm[1:-1, 1:-1, 1:-1]
    zp = zp[1:-1, 1:-1, 1:-1]
    tris = _build_triangles(im, xm, xp, ym, yp, zm, zp)
    tris[:, 1:, :] *= voxel_size
    return tris


@njit
def _exposed_face_masks(mask):
    nx, ny, nz = mask.shape
    xm = np.zeros(mask.shape, dtype=np.bool_)
    xp = np.zeros(mask.shape, dtype=np.bool_)
    ym = np.zeros(mask.shape, dtype=np.bool_)
    yp = np.zeros(mask.shape, dtype=np.bool_)
    zm = np.zeros(mask.shape, dtype=np.bool_)
    zp = np.zeros(mask.shape, dtype=np.bool_)

    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                if not mask[x, y, z]:
                    continue
                xm[x, y, z] = (x == 0) or (not mask[x - 1, y, z])
                xp[x, y, z] = (x == nx - 1) or (not mask[x + 1, y, z])
                ym[x, y, z] = (y == 0) or (not mask[x, y - 1, z])
                yp[x, y, z] = (y == ny - 1) or (not mask[x, y + 1, z])
                zm[x, y, z] = (z == 0) or (not mask[x, y, z - 1])
                zp[x, y, z] = (z == nz - 1) or (not mask[x, y, z + 1])
    return xm, xp, ym, yp, zm, zp


@njit
def _build_triangles(mask, xm, xp, ym, yp, zm, zp):
    n_faces = (
        xm.sum() + xp.sum() + ym.sum() +
        yp.sum() + zm.sum() + zp.sum()
    )
    triangles = np.zeros((2 * n_faces, 4, 3), dtype=np.int32)
    j = 0

    nx, ny, nz = mask.shape
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                if not mask[x, y, z]:
                    continue

                if zm[x, y, z]:
                    triangles[j] = np.array(
                        [[0, 0, -1], [x, y, z], [x, y+1, z], [x+1, y, z]],
                        dtype=np.int32,
                    )
                    triangles[j+1] = np.array(
                        [[0, 0, -1], [x, y+1, z], [x+1, y+1, z], [x+1, y, z]],
                        dtype=np.int32,
                    )
                    j += 2

                if zp[x, y, z]:
                    triangles[j] = np.array(
                        [[0, 0, 1], [x, y, z+1], [x+1, y, z+1], [x, y+1, z+1]],
                        dtype=np.int32,
                    )
                    triangles[j+1] = np.array(
                        [[0, 0, 1], [x, y+1, z+1], [x+1, y, z+1], [x+1, y+1, z+1]],
                        dtype=np.int32,
                    )
                    j += 2

                if ym[x, y, z]:
                    triangles[j] = np.array(
                        [[0, -1, 0], [x, y, z], [x+1, y, z], [x, y, z+1]],
                        dtype=np.int32,
                    )
                    triangles[j+1] = np.array(
                        [[0, -1, 0], [x, y, z+1], [x+1, y, z], [x+1, y, z+1]],
                        dtype=np.int32,
                    )
                    j += 2

                if yp[x, y, z]:
                    triangles[j] = np.array(
                        [[0, 1, 0], [x, y+1, z], [x, y+1, z+1], [x+1, y+1, z]],
                        dtype=np.int32,
                    )
                    triangles[j+1] = np.array(
                        [[0, 1, 0], [x, y+1, z+1], [x+1, y+1, z+1], [x+1, y+1, z]],
                        dtype=np.int32,
                    )
                    j += 2

                if xm[x, y, z]:
                    triangles[j] = np.array(
                        [[-1, 0, 0], [x, y, z], [x, y, z+1], [x, y+1, z]],
                        dtype=np.int32,
                    )
                    triangles[j+1] = np.array(
                        [[-1, 0, 0], [x, y, z+1], [x, y+1, z+1], [x, y+1, z]],
                        dtype=np.int32,
                    )
                    j += 2

                if xp[x, y, z]:
                    triangles[j] = np.array(
                        [[1, 0, 0], [x+1, y, z], [x+1, y+1, z], [x+1, y, z+1]],
                        dtype=np.int32,
                    )
                    triangles[j+1] = np.array(
                        [[1, 0, 0], [x+1, y, z+1], [x+1, y+1, z], [x+1, y+1, z+1]],
                        dtype=np.int32,
                    )
                    j += 2

    return triangles[:j]


if __name__ == "__main__":
    import porespy as ps
    import pyvista as pv

    im = ps.generators.random_spheres([150, 150, 150], r=10, clearance=5, edges='extended')

    mesh1 = to_stl(im=im, method='marching-cubes', fmt='pyvista')
    mesh2 = to_stl(im=im, method='marching-cubes', remove_duplicates=True, fmt='pyvista')
    mesh3 = to_stl(im=im, method='direct', fmt='pyvista')
    mesh4 = to_stl(im=im, method='direct', remove_duplicates=True, fmt='pyvista')

    plotter = pv.Plotter()
    plotter.add_mesh(mesh1, opacity=0.5)
    plotter.add_mesh(mesh3, color='r')
    plotter.show()
