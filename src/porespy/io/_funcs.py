import numpy as np
import scipy.ndimage as nd
import skimage.measure as ms

from porespy.tools import get_edt, sanitize_filename

edt = get_edt()


__all__ = [
    "to_vtk",
    "dict_to_vtk",
    "to_palabos",
    "to_stl",
]


def dict_to_vtk(data, filename, voxel_size=1, origin=(0, 0, 0)):
    r"""
    Accepts multiple images as a dictionary and compiles them into a vtk file

    Parameters
    ----------
    data : dict
        A dictionary of *key: value* pairs, where the *key* is the name of the
        scalar property stored in each voxel of the array stored in the
        corresponding *value*.
    path : string
        Path to output file
    voxel_size : int
        The side length of the voxels (voxels  are cubic)
    origin : float
        data origin (according to selected voxel size)

    Notes
    -----
    Outputs a vtk, vtp or vti file that can opened in ParaView

    Examples
    --------
    `Click here
    <https://porespy.org/examples/io/reference/dict_to_vtk.html>`__
    to view online example.

    """
    try:
        from pyevtk.hl import imageToVTK
    except ModuleNotFoundError:
        msg = "The pyevtk package can be installed with pip install pyevtk"
        raise ModuleNotFoundError(msg)
    vs = voxel_size
    for entry in data:
        if data[entry].dtype == bool:
            data[entry] = data[entry].astype(np.int8)
        if data[entry].flags["C_CONTIGUOUS"]:
            data[entry] = np.ascontiguousarray(data[entry])
    imageToVTK(filename, cellData=data, spacing=(vs, vs, vs), origin=origin)


def to_vtk(im, filename, divide=False, downsample=False, voxel_size=1, vox=False):
    r"""
    Converts an array to a vtk file.

    Parameters
    ----------
    im : 3D image
        The image of the porous material
    path : string
        Path to output file
    divide : bool
        vtk files can get very large, this option allows you for two output
        files, divided at z = half. This allows for large data sets to be
        imaged without loss of information
    downsample : bool
        Very large images can be downsampled to half the size in each
        dimension, this doubles the effective voxel size
    voxel_size : int
        The side length of the voxels (voxels  are cubic)
    vox : bool
        For an image that is binary (1's and 0's) this reduces the file size by
        using int8 format (can also be used to reduce file size when accuracy
        is not necessary ie: just visulization)

    Notes
    -----
    Outputs a vtk, vtp or vti file that can opened in paraview

    Examples
    --------
    `Click here
    <https://porespy.org/examples/io/reference/to_vtk.html>`__
    to view online example.

    """
    try:
        from pyevtk.hl import imageToVTK
    except ModuleNotFoundError:
        msg = "The pyevtk package can be installed with pip install pyevtk"
        raise ModuleNotFoundError(msg)
    if len(im.shape) == 2:
        im = im[:, :, np.newaxis]
    if im.dtype == bool:
        vox = True
    if vox:
        im = im.astype(np.int8)
    vs = voxel_size
    if divide:
        split = np.round(im.shape[2] / 2).astype(np.int)
        im1 = im[:, :, 0:split]
        im2 = im[:, :, split:]
        imageToVTK(
            f"{filename}_1",
            cellData={"im": np.ascontiguousarray(im1)},
            spacing=(vs, vs, vs),
        )
        imageToVTK(
            f"{filename}_2",
            origin=(0.0, 0.0, split * vs),
            cellData={"im": np.ascontiguousarray(im2)},
            spacing=(vs, vs, vs),
        )
    elif downsample:
        im = nd.interpolation.zoom(im, zoom=0.5, order=0, mode="reflect")
        imageToVTK(
            filename,
            cellData={"im": np.ascontiguousarray(im)},
            spacing=(2 * vs, 2 * vs, 2 * vs),
        )
    else:
        imageToVTK(
            filename, cellData={"im": np.ascontiguousarray(im)}, spacing=(vs, vs, vs)
        )


def to_palabos(im, filename, solid=0):
    r"""
    Converts an ndarray image to a text file that Palabos can read in as a
    geometry for Lattice Boltzmann simulations. Uses a Euclidean distance
    transform to identify solid voxels neighboring fluid voxels and labels
    them as the interface.

    Parameters
    ----------
    im : ndarray
        The image of the porous material
    filename : string
        Path to output file
    solid : int
        The value of the solid voxels in the image used to convert image to
        binary with all other voxels assumed to be fluid.

    Notes
    -----
    File produced contains 3 values: 2 = Solid, 1 = Interface, 0 = Pore
    Palabos will run the simulation applying the specified pressure drop from
    x = 0 to x = -1.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/io/reference/to_palabos.html>`__
    to view online example.

    """
    # Create binary image for fluid and solid phases
    bin_im = im == solid
    # Transform to integer for distance transform
    bin_im = bin_im.astype(int)
    # Distance Transform computes Euclidean distance in lattice units to
    # Nearest fluid for every solid voxel
    dt = edt(bin_im)
    dt[dt > 2] = 2
    dt[(dt > 0) * (dt <= 2)] = 1
    dt = np.sqrt(dt).astype(int)
    # Write out data
    with open(filename, "w") as f:
        out_data = dt.flatten().tolist()
        f.write("\n".join(map(repr, out_data)))


def to_stl(im, filename, divide=False, downsample=False, voxel_size=1, vox=False):
    r"""
    Converts an array to an STL file.

    Parameters
    ----------
    im : 3D image
        The image of the porous material
    path : string
        Path to output file
    divide : bool
        vtk files can get very large, this option allows you for two output
        files, divided at z = half. This allows for large data sets to be
        imaged without loss of information
    downsample : bool
        very large images acan be downsampled to half the size in each
        dimension, this doubles the effective voxel size
    voxel_size : int
        The side length of the voxels (voxels  are cubic)
    vox : bool
        For an image that is binary (1's and 0's) this reduces the file size by
        using int8 format (can also be used to reduce file size when accuracy
        is not necessary ie: just visulization)

    Notes
    -----
    Outputs an STL file that can opened in Paraview

    Examples
    --------
    `Click here
    <https://porespy.org/examples/io/reference/to_stl.html>`__
    to view online example.

    """
    filename = sanitize_filename(filename, ext="stl", exclude_ext=True)
    if len(im.shape) == 2:
        im = im[:, :, np.newaxis]
    if im.dtype == bool:
        vox = True
    if vox:
        im = im.astype(np.int8)
    vs = voxel_size
    if divide:
        split = np.round(im.shape[2] / 2).astype(np.int)
        im1 = im[:, :, 0:split]
        im2 = im[:, :, split:]
        _save_stl(im1, vs, f"{filename}_1")
        _save_stl(im2, vs, f"{filename}_2")
    elif downsample:
        im = nd.interpolation.zoom(im, zoom=0.5, order=0, mode="reflect")
        _save_stl(im, vs * 2, filename)
    else:
        _save_stl(im, vs, filename)


def _save_stl(im, vs, filename):
    r"""
    Helper method to convert an array to an STL file.

    Parameters
    ----------
    im : 3D image
        The image of the porous material
    voxel_size : int
        The side length of the voxels (voxels are cubic)
    filename : string
        Path to output file

    """
    try:
        from stl import mesh
    except ModuleNotFoundError:
        msg = "numpy-stl can be installed with pip install numpy-stl"
        raise ModuleNotFoundError(msg)
    im = np.pad(im, pad_width=10, mode="constant", constant_values=True)
    vertices, faces, norms, values = ms.marching_cubes(im)
    vertices *= vs
    # Export the STL file
    export = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            export.vectors[i][j] = vertices[f[j], :]
    export.save(f"{filename}.stl")
