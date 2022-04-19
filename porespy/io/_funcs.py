import os
import imageio
import subprocess
import numpy as np
from stl import mesh
import scipy.ndimage as nd
import skimage.measure as ms
from porespy.tools import sanitize_filename
from porespy.networks import generate_voxel_image
from porespy.filters import reduce_peaks
from pyevtk.hl import imageToVTK
from skimage.morphology import ball
from edt import edt


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
    <https://porespy.org/examples/io/reference/dict_to_vtk.html>`_
    to view online example.

    """
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
    Outputs a vtk, vtp or vti file that can opened in paraview

    Examples
    --------
    `Click here
    <https://porespy.org/examples/io/reference/to_vtk.html>`_
    to view online example.

    """
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
        imageToVTK(f"{filename}_1",
                   cellData={"im": np.ascontiguousarray(im1)},
                   spacing=(vs, vs, vs),)
        imageToVTK(f"{filename}_2",
                   origin=(0.0, 0.0, split * vs),
                   cellData={"im": np.ascontiguousarray(im2)},
                   spacing=(vs, vs, vs),)
    elif downsample:
        im = nd.interpolation.zoom(im, zoom=0.5, order=0, mode="reflect")
        imageToVTK(filename,
                   cellData={"im": np.ascontiguousarray(im)},
                   spacing=(2 * vs, 2 * vs, 2 * vs),)
    else:
        imageToVTK(filename,
                   cellData={"im": np.ascontiguousarray(im)},
                   spacing=(vs, vs, vs))


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
    <https://porespy.org/examples/io/reference/to_palabos.html>`_
    to view online example.

    """
    # Create binary image for fluid and solid phases
    bin_im = im == solid
    # Transform to integer for distance transform
    bin_im = bin_im.astype(int)
    # Distance Transform computes Euclidean distance in lattice units to
    # Nearest fluid for every solid voxel
    dt = nd.distance_transform_edt(bin_im)
    dt[dt > np.sqrt(2)] = 2
    dt[(dt > 0) * (dt <= np.sqrt(2))] = 1
    dt = dt.astype(int)
    # Write out data
    with open(filename, "w") as f:
        out_data = dt.flatten().tolist()
        f.write("\n".join(map(repr, out_data)))


def openpnm_to_im(
    network,
    pore_shape="sphere",
    throat_shape="cylinder",
    max_dim=None,
    rtol=0.1,
):
    r"""
    Generates voxel image from an OpenPNM network object.

    Parameters
    ----------
    network : OpenPNM GenericNetwork
        Network from which voxel image is to be generated
    pore_shape : str
        Shape of pores in the network, valid choices are "sphere", "cube"
    throat_shape : str
        Shape of throats in the network, valid choices are "cylinder", "cuboid"
    max_dim : int
        Number of voxels in the largest dimension of the network
    rtol : float
        Stopping criteria for finding the smallest voxel image such that
        further increasing the number of voxels in each dimension by 25% would
        improve the predicted porosity of the image by less that ``rtol``

    Returns
    -------
    im : ndarray
        Voxelated image corresponding to the given pore network model

    Notes
    -----
    (1) The generated voxelated image is labeled with 0s, 1s and 2s signifying
    solid phase, pores, and throats respectively.

    (2) If max_dim is not provided, the method calculates it such that the
    further increasing it doesn't change porosity by much.

    """
    return generate_voxel_image(
        network,
        pore_shape=pore_shape,
        throat_shape=throat_shape,
        max_dim=max_dim,
        rtol=rtol,
    )


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
    <https://porespy.org/examples/io/reference/to_stl.html>`_
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
    im = np.pad(im, pad_width=10, mode="constant", constant_values=True)
    vertices, faces, norms, values = ms.marching_cubes(im)
    vertices *= vs
    # Export the STL file
    export = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            export.vectors[i][j] = vertices[f[j], :]
    export.save(f"{filename}.stl")


def to_paraview(im, filename, phase=2):
    r"""
    Converts an array to a paraview state file.

    Parameters
    ----------
    im : ndarray
        The image of the porous material.
    filename : str
        Path to output file.
    phase : str
        The desired phase of output image where phase = 0 represent the
        pore phase, phase = 1 represents the solid phase, and phase= 2 is
        the whole domain. The default value is 2.

    Notes
    -----
    Outputs an pvsm file that can opened in Paraview.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/io/reference/to_paraview.html>`_
    to view online example.

    """
    try:
        import paraview.simple
    except ModuleNotFoundError:
        msg = ("The paraview python bindings must be installed using conda"
               " install -c conda-forge paraview, however this may require"
               " using a virtualenv since conflicts with other packages are"
               " common. This is why it is not explicitly included as a"
               " dependency in porespy.")
        raise ModuleNotFoundError(msg)
    data = im.astype("uint8")
    file = os.path.splitext(filename)[0]
    path = file + ".tiff"
    if len(im.shape) == 2:
        imageio.imwrite(path, np.array(data))
        view = "Slice"
        zshape = 0
        xshape = im.shape[1]
        yshape = im.shape[0]
    elif len(im.shape) == 3:
        imageio.volsave(path, np.array(data))
        view = "Volume"
        zshape = im.shape[0]
        xshape = im.shape[2]
        yshape = im.shape[1]
    maxshape = max(xshape, yshape)
    paraview.simple._DisableFirstRenderCameraReset()
    # Create a new 'TIFF Series Reader'
    dtiff = paraview.simple.TIFFSeriesReader(FileNames=[path])
    # Get active view
    renderView1 = paraview.simple.GetActiveViewOrCreate("RenderView")
    # Uncomment following to set a specific view size
    # renderView1.ViewSize = [1612, 552]
    # Get layout
    _ = paraview.simple.GetLayout()

    # Show data in view
    dtiffDisplay = paraview.simple.Show(dtiff, renderView1, "UniformGridRepresentation")

    # Get color transfer function/color map for 'TiffScalars'
    tiffScalarsLUT = paraview.simple.GetColorTransferFunction("TiffScalars")

    # Get opacity transfer function/opacity map for 'TiffScalars'
    tiffScalarsPWF = paraview.simple.GetOpacityTransferFunction("TiffScalars")

    # Trace defaults for the display properties.
    dtiffDisplay.Representation = view
    dtiffDisplay.ColorArrayName = ["POINTS", "Tiff Scalars"]
    dtiffDisplay.LookupTable = tiffScalarsLUT
    dtiffDisplay.OSPRayScaleArray = "Tiff Scalars"
    dtiffDisplay.OSPRayScaleFunction = "PiecewiseFunction"
    dtiffDisplay.SelectOrientationVectors = "None"
    dtiffDisplay.ScaleFactor = maxshape / 10 - 0.1
    dtiffDisplay.SelectScaleArray = "Tiff Scalars"
    dtiffDisplay.GlyphType = "Arrow"
    dtiffDisplay.GlyphTableIndexArray = "Tiff Scalars"
    dtiffDisplay.GaussianRadius = maxshape / 200 - 0.005
    dtiffDisplay.SetScaleArray = ["POINTS", "Tiff Scalars"]
    dtiffDisplay.ScaleTransferFunction = "PiecewiseFunction"
    dtiffDisplay.OpacityArray = ["POINTS", "Tiff Scalars"]
    dtiffDisplay.OpacityTransferFunction = "PiecewiseFunction"
    dtiffDisplay.DataAxesGrid = "GridAxesRepresentation"
    dtiffDisplay.PolarAxes = "PolarAxesRepresentation"
    dtiffDisplay.ScalarOpacityUnitDistance = 8.256564094912507
    dtiffDisplay.ScalarOpacityFunction = tiffScalarsPWF
    dtiffDisplay.IsosurfaceValues = [0.5]
    dtiffDisplay.SliceFunction = "Plane"

    shape = np.array([xshape, yshape, zshape])

    # Init the 'Plane' selected for 'SliceFunction'
    dtiffDisplay.SliceFunction.Origin = [xi / 2 - 0.5 for xi in shape]

    # Reset view to fit data
    renderView1.ResetCamera()

    # Changing interaction mode based on data extents
    # renderView1.InteractionMode = mode
    renderView1.CameraPosition = [
        xshape / 2 - 0.5,
        yshape / 2 - 0.5,
        4.6 * np.sqrt(np.sum(shape / 2 - 0.5)**2)
    ]
    renderView1.CameraFocalPoint = [xi / 2 - 0.5 for xi in shape]

    # Get the material library
    _ = paraview.simple.GetMaterialLibrary()

    # Show color bar/color legend
    dtiffDisplay.SetScalarBarVisibility(renderView1, True)

    # Update the view to ensure updated data information
    renderView1.Update()

    # Saving camera placements for all active views
    # Current camera placement for renderView1
    # renderView1.InteractionMode = mode
    renderView1.CameraPosition = [
        xshape / 2 - 0.5,
        yshape / 2 - 0.5,
        4.6 * np.sqrt(np.sum(shape / 2 - 0.5)**2)
    ]
    renderView1.CameraFocalPoint = [xi / 2 - 0.5 for xi in shape]
    renderView1.CameraParallelScale = np.sqrt(np.sum(shape / 2 - 0.5)**2)

    # Uncomment the following to render all views
    # RenderAllViews()
    # Alternatively, if you want to write images, you can use SaveScreenshot(...).
    threshold1 = paraview.simple.Threshold(Input=dtiff)
    threshold1.Scalars = ["POINTS", "Tiff Scalars"]
    if phase == 0:
        threshold_range = [0.5, 1]
    elif phase == 1:
        threshold_range = [0, 0.5]
    else:
        threshold_range = [0, 1]
    threshold1.ThresholdRange = threshold_range

    # Show data in view
    _ = paraview.simple.Show(
        threshold1, renderView1, "UnstructuredGridRepresentation"
    )

    # Hide data in view
    paraview.simple.Hide(dtiff, renderView1)

    paraview.simple.SaveState(file + ".pvsm")


def open_paraview(filename=None, im=None, **kwargs):
    r"""
    Open a paraview state file or image directly in paraview.

    Parameters
    ----------
    filename : str
        Path to input state file.
    im : ndarray
        An image to open directly.  If no filename given, then this image is
        sent to ``to_paraview`` and a state file is created with a random name.
        Any additional keyword arguments are sent to ``to_paraview``.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/io/reference/open_paraview.html>`_
    to view online example.

    """
    if filename is None:
        from datetime import datetime
        now = datetime.now()
        filename = now.strftime("%d-%m-%Y_%H-%M-%S")
        to_paraview(im=im, filename=filename, **kwargs)
    file = os.path.splitext(filename)[0]
    statefile = file + ".pvsm"
    # paraview_path = "paraview.exe"
    paraview_path = "paraview"
    subprocess.Popen([paraview_path, statefile])


def spheres_to_comsol(filename, im=None, centers=None, radii=None):
    r"""
    Exports a sphere pack into a Comsol geometry file.

    An image containing spheres can be specified.  Alternatively as list of
    ``centers`` and ``radii`` can be given if known.

    Parameters
    ----------
    filename : string or path object
        Location and namge to output file
    im : ndarray (optional)
        A voxel image containing spheres indicated by non-zeros values.
        Spheres can be generated using a variety of methods and can overlap.
        The sphere centers and radii are found as the peaks in the
        distance transform.  If ``im`` is not supplied, then ``centers`` and
        ``radii`` must be given.
    centers : array_like (optional)
        An array (Ns, 3) of the spheres centers where Ns is the number of
        spheres.  This must be specified if ``im`` is not suppplied.
    radii : array_like (optional)
        An Ns length array of the spheres's. This must be specified if ``im``
        is not suppplied.

    Notes
    -----
    If ``im`` is given then some image analysis is performed to find sphere
    centers so it may not perfectly represent the spheres in the original
    image. This is especially true for overlapping sphere and spheres extending
    beyond the edge of the image.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/io/reference/spheres_to_comsol.html>`_
    to view online example.

    """
    from ._comsol import _save_to_comsol
    if im is not None:
        if im.ndim != 3:
            raise Exception('Image must be 3D.')
        dt = edt(im > 0)
        dt2 = nd.gaussian_filter(dt, sigma=0.1)
        peaks = (im > 0)*(nd.maximum_filter(dt2, footprint=ball(3)) == dt)
        peaks = reduce_peaks(peaks)
        centers = np.vstack(np.where(peaks)).T
        radii = dt[tuple(centers.T)].astype(int)
    _save_to_comsol(filename, centers, radii)
