import numpy as np
import scipy.ndimage as spim
from numba import njit, prange
import pyvista as pv

__all__ = [
    "show_3D",
    "show_planes",
    "sem",
    "xray",
    "render_volume",
    "render_network",
]


def render_network(
    network,
    size_by,
    plotter=None,
    plotterkws=dict(
    ),
    linekws=dict(
        line_width=3,
        color='k',
        show_scalar_bar=False,
    ),
    glyphkws=dict(
        ambient=0.2,
        diffuse=0.5,
        specular=0.5,
        specular_power=90,
        smooth_shading=True,
        cmap='turbo',
        show_scalar_bar=False,
    ),
):
    # Read the VTP file
    mesh = pv.read(network)
    size_by = 'network | properties | pore | inscribed_diameter'
    glyphs = mesh.glyph(scale=size_by, factor=1.0, geom=pv.Sphere())
    if plotter is None:
        plotter = pv.Plotter(**plotterkws)
    plotter.add_mesh(mesh, **linekws)
    plotter.add_mesh(glyphs, **glyphkws)
    return plotter


def render_volume(
    im,
    phase=0,
    voxel_size=1.0,
    voxelkws=dict(
        opacity=1.0,
        show_edges=False,
        cmap='bone',
        clim=[0, 2],
        line_width=0.1,
        ambient=0.2,
        diffuse=0.5,
        specular=0.5,
        specular_power=90,
        show_scalar_bar=False,
    ),
    plotterkws=dict(
    ),
    plotter=None,
):
    r"""
    Renders voxels in an interactive 3D view

    Parameters
    ----------
    im : ndarray
        The image to be rendered
    phase : int
        The phase to show.  The default is 0 which is typically the solid.
    voxel_size : float (default = 1.0)
        The side length of the voxels.
    plotter : pyVista.Plotter object, optional
        If a plotter object already exists and you with to add more items,
        you can provide it.  If not provided a new plotter will be created.
    plotterkws : dict
        A dictionary of keyword arguments that is passed to the plotter
        when adding the voxel mesh.

    Returns
    -------
    plotter : pyVista.plotter object
        The plotter object can be visualized with `plotter.show()`. Note
        that once it has been 'shown' no further adjustments can be made
        to it.

    Notes
    -----
    This function uses pyVista, which seems to work pretty well. For more
    powerful visualization it is recommended to expore the image to a
    VTK file using `porespy.io.to_vtk` then using Paraview.

    """
    grid = pv.ImageData()
    grid.dimensions = np.array(im.shape) + 1
    vx = voxel_size
    grid.spacing = (vx, vx, vx)
    grid.cell_data["values"] = im.flatten(order="F")
    threshold = grid.threshold([phase-0.01, phase+0.01])
    if plotter is None:
        plotter = pv.Plotter(**plotterkws)
    plotter.add_mesh(threshold, **voxelkws)
    plotter.set_background('white')
    return plotter


def show_3D(im):  # pragma: no cover
    r"""
    Rotates a 3D image and creates an angled view for rough 2D visualization.

    Because it rotates the image it can be slow for large images, so is mostly
    meant for rough checking of small prototype images.

    Parameters
    ----------
    im : ndarray
        The 3D array to be viewed from an angle

    Returns
    -------
    image : ndarray
        A 2D view of the given 3D image

    Notes
    -----
    Although this is meant to be *quick* it can still take on the order of
    minutes to render very large images.  It uses `scipy.ndimage.rotate`
    with no interpolation to view the 3D image from an angle, then casts the
    result into a 2D projection.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/visualization/reference/show_3D.html>`__
    to view online example.

    """
    im = ~np.copy(im)
    if im.ndim < 3:
        raise Exception("show_3D only applies to 3D images")
    im = spim.rotate(input=im, angle=22.5, axes=[0, 1], order=0)
    im = spim.rotate(input=im, angle=45, axes=[2, 1], order=0)
    im = spim.rotate(input=im, angle=-17, axes=[0, 1], order=0, reshape=False)
    mask = im != 0
    view = np.where(mask.any(axis=2), mask.argmax(axis=2), 0)
    view = view.max() - view
    f = view.max() / 5
    view[view == view.max()] = -f
    view = (view + f) ** 2
    return view


def show_planes(im, spacing=10):  # pragma: no cover
    r"""
    Create a quick montage showing a 3D image in all three directions

    Parameters
    ----------
    im : ndarray
        A 3D image of the porous material
    spacing : int (optional, default=10)
        Controls the amount of space to put between each panel

    Returns
    -------
    image : ndarray
        A 2D array containing the views.  This single image can be viewed using
        `matplotlib.pyplot.imshow`.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/visualization/reference/show_planes.html>`__
    to view online example.

    """
    s = spacing
    if np.squeeze(im.ndim) < 3:
        raise Exception("This view is only necessary for 3D images")
    x, y, z = (np.array(im.shape) / 2).astype(int)
    im_xy = im[:, :, z]
    im_xz = im[:, y, :]
    im_yz = np.rot90(im[x, :, :])

    new_x = im_xy.shape[0] + im_yz.shape[0] + s
    new_y = im_xy.shape[1] + im_xz.shape[1] + s
    new_im = np.zeros([new_x + 2 * s, new_y + 2 * s], dtype=im.dtype)

    # Add xy image to upper left corner
    new_im[s: im_xy.shape[0] + s, s: im_xy.shape[1] + s] = im_xy
    # Add xz image to lower left coner
    x_off = im_xy.shape[0] + 2 * s
    y_off = im_xy.shape[1] + 2 * s
    new_im[s: s + im_xz.shape[0], y_off: y_off + im_xz.shape[1]] = im_xz
    new_im[x_off: x_off + im_yz.shape[0], s: s + im_yz.shape[1]] = im_yz

    return new_im


def sem(im, axis=0):  # pragma: no cover
    r"""
    Simulates an SEM image looking into the porous material.

    Features are colored according to their depth into the image, so
    darker features are further away.

    Parameters
    ----------
    im : array_like
        ndarray of the porous material with the solid phase marked as 1 or
        True
    axis : int
        Specifes the axis along which the camera will point.

    Returns
    -------
    image : ndarray
        A 2D greyscale image suitable for use in matplotlib's `imshow`
        function.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/visualization/reference/sem.html>`__
    to view online example.

    """
    im = np.array(~im, dtype=int)
    im = np.swapaxes(im, 0, axis)
    view = _sem_parallel(im)
    im = np.swapaxes(im, 0, axis)
    return view


@njit(parallel=True)
def _sem_parallel(im):  # pragma: no cover
    r"""
    This function is called by `sem` to compute the height of the first
    voxel in each x, y column. It uses numba for speed, and is parallelized.
    """
    shape = im.shape
    depth = np.zeros(shape[:2])
    for x in prange(shape[0]):
        for y in prange(shape[1]):
            for z in range(shape[2] - 1, 0, -1):
                if not im[x][y][z]:
                    depth[x][y] = z / shape[2]
                    break
    return depth


def xray(im, axis=0):  # pragma: no cover
    r"""
    Simulates an X-ray radiograph looking through the porous material.

    The resulting image is colored according to the amount of attenuation an
    X-ray would experience, so regions with more solid will appear darker.

    Parameters
    ----------
    im : array_like
        ndarray of the porous material with the solid phase marked as 1 or `True`

    axis : int
        Specifes the axis along which the camera will point.

    Returns
    -------
    image : ndarray
        A 2D greyscale image suitable for use in matplotlib's `imshow` function.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/visualization/reference/sem.html>`__
    to view online example.
    """
    im = np.array(~im, dtype=int)
    if axis == 1:
        im = np.transpose(im, axes=[1, 0, 2])
    if axis == 2:
        im = np.transpose(im, axes=[2, 1, 0])
    im = np.sum(im, axis=0, dtype=np.int64)
    im = 1 - im / np.max(im)
    return im
