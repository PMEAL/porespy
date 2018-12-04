import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def show_mesh(mesh):
    r"""
    Visualizes the mesh of a region as obtained by ``get_mesh`` function in
    the ``metrics`` submodule.
    """
    lim_max = sp.amax(mesh.verts, axis=0)
    lim_min = sp.amin(mesh.verts, axis=0)

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
