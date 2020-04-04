import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def imshow(im, ind=None, axis=None):
    r"""
    Convenient wrapper for matplotlib's ``imshow``.

    This automatically:
        * slices a 3D image in the middle of the last axis
        * uses a masked array to make 0's white
        * sets the origin to 'xy' so bottom-left corner is [0, 0]

    Parameters
    ----------
    im : ND-array
        The image to show.  Can be 2D or 3D.  If 2D then all other arguments
        are ignored.
    ind : int
        The slice to show if ``im`` is 3D.  If not given then the middle of
        the image is used.
    axis : int
        The axis to show if ``im`` is 3D.  If not given, then the last
        axis of the image is used, so an 'xy' slice is shown.
    """
    if im.ndim == 3:
        if axis is None:
            axis = 2
        if ind is None:
            ind = int(im.shape[axis]/2)
        im = im.take(indices=ind, axis=axis)
    im = np.ma.array(im, mask=im == 0)
    fig = plt.imshow(im, origin='xy')
    return fig


def show_mesh(mesh):
    r"""
    Visualizes the mesh of a region as obtained by ``get_mesh`` function in
    the ``metrics`` submodule.

    Parameters
    ----------
    mesh : tuple
        A mesh returned by ``skimage.measure.marching_cubes``

    Returns
    -------
    fig : Matplotlib figure
        A handle to a matplotlib 3D axis
    """
    lim_max = np.amax(mesh.verts, axis=0)
    lim_min = np.amin(mesh.verts, axis=0)

    # Display resulting triangular mesh using Matplotlib.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(mesh.verts[mesh.faces])
    mesh.set_edgecolor('k')

    ax.add_collection3d(mesh)
    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    ax.set_zlabel("z-axis")
    ax.set_xlim(lim_min[0], lim_max[0])
    ax.set_ylim(lim_min[1], lim_max[1])
    ax.set_zlim(lim_min[2], lim_max[2])

    return fig
