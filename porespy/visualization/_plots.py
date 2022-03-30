import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


__all__ = [
    'bar',
    'imshow',
    'show_mesh',
]


def bar(results, h='pdf', **kwargs):  # pragma: no cover
    r"""
    Convenience wrapper for matplotlib's ``bar``.

    This automatically:

        * fetches the ``bin_centers``
        * fetches the bin heights from the specified ``h``
        * sets the bin widths
        * sets the edges to black

    Parameters
    ----------
    results : object
        The objects returned by various functions in the
        ``porespy.metrics`` submodule, such as ``chord_length_distribution``.
    h : str
        The value to use for bin heights.  The default is ``pdf``, but
        ``cdf`` is another option. Depending on the function the named-tuple
        may have different options.
    kwargs : keyword arguments
        All other keyword arguments are passed to ``bar``, including
        ``edgecolor`` if you wish to overwrite the default black.

    Returns
    -------
    fig: Matplotlib figure handle

    Examples
    --------
    `Click here
    <https://porespy.org/examples/visualization/reference/bar.html>`_
    to view online example.
    """
    if 'edgecolor' not in kwargs:
        kwargs['edgecolor'] = 'k'
    fig = plt.bar(x=results.bin_centers, height=getattr(results, h),
                  width=results.bin_widths, **kwargs)
    xlab = [attr for attr in results.__dir__() if not attr.startswith('_')][0]
    plt.xlabel(xlab)
    plt.ylabel(h)
    return fig


def imshow(*im, ind=None, axis=None, **kwargs):  # pragma: no cover
    r"""
    Convenience wrapper for matplotlib's ``imshow``.

    This automatically:

        * slices a 3D image in the middle of the last axis
        * uses a masked array to make 0's white
        * sets the origin to 'lower' so bottom-left corner is [0, 0]
        * disables interpolation

    Parameters
    ----------
    im : ndarray
        The 2D or 3D image (or images) to show.  If 2D then all other
        arguments are ignored.
    ind : int
        The slice to show if ``im`` is 3D.  If not given then the middle of
        the image is used.
    axis : int
        The axis to show if ``im`` is 3D.  If not given, then the last
        axis of the image is used, so an 'lower' slice is shown.

    **kwargs
        All other keyword arguments are passed to ``plt.imshow``

    Note
    ----
    ``im`` can also be a series of unnamed arguments, in which case all
    received images will be shown using ``subplot``.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/visualization/reference/imshow.html>`_
    to view online example.
    """
    if 'origin' not in kwargs.keys():
        kwargs['origin'] = 'lower'
    if 'interpolation' not in kwargs.keys():
        kwargs['interpolation'] = 'none'
    if not isinstance(im, tuple):
        im = tuple([im])
    for i, image in enumerate(im):
        if image.ndim == 3:
            if axis is None:
                axis = 2
            if ind is None:
                ind = int(image.shape[axis]/2)
            image = image.take(indices=ind, axis=axis)
        image = np.ma.array(image, mask=image == 0)
        fig = plt.subplot(1, len(im), i+1)
        plt.imshow(image, **kwargs)
    return fig


def show_mesh(mesh):  # pragma: no cover
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

    Examples
    --------
    `Click here
    <https://porespy.org/examples/visualization/reference/show_mesh.html>`_
    to view online example.
    """
    try:
        verts = mesh.vertices
    except AttributeError:
        verts = mesh.verts

    lim_max = np.amax(verts, axis=0)
    lim_min = np.amin(verts, axis=0)

    # Display resulting triangular mesh using Matplotlib.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[mesh.faces])
    mesh.set_edgecolor('k')

    ax.add_collection3d(mesh)
    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    ax.set_zlabel("z-axis")
    ax.set_xlim(lim_min[0], lim_max[0])
    ax.set_ylim(lim_min[1], lim_max[1])
    ax.set_zlim(lim_min[2], lim_max[2])

    return fig
