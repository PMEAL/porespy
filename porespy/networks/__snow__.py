import numpy as np
from porespy.networks import regions_to_network, add_boundary_regions
from porespy.networks import _net_dict
from porespy.networks import label_boundary_cells
from porespy.tools import pad_faces
from porespy.filters import snow_partitioning
from porespy.tools import make_contiguous
from porespy.metrics import region_surface_areas, region_interface_areas


def snow(im, voxel_size=1,
         boundary_faces=['top', 'bottom', 'left', 'right', 'front', 'back'],
         marching_cubes_area=False):
    r"""
    Analyzes an image that has been partitioned into void and solid regions
    and extracts the void and solid phase geometry as well as network
    connectivity.

    Parameters
    ----------
    im : ND-array
        Binary image in the Boolean form with True’s as void phase and False’s
        as solid phase.
    voxel_size : scalar
        The resolution of the image, expressed as the length of one side of a
        voxel, so the volume of a voxel would be **voxel_size**-cubed.  The
        default is 1, which is useful when overlaying the PNM on the original
        image since the scale of the image is alway 1 unit lenth per voxel.
    boundary_faces : list of strings
        Boundary faces labels are provided to assign hypothetical boundary
        nodes having zero resistance to transport process. For cubical
        geometry, the user can choose ‘left’, ‘right’, ‘top’, ‘bottom’,
        ‘front’ and ‘back’ face labels to assign boundary nodes. If no label is
        assigned then all six faces will be selected as boundary nodes
        automatically which can be trimmed later on based on user requirements.
    marching_cubes_area : bool
        If ``True`` then the surface area and interfacial area between regions
        will be using the marching cube algorithm. This is a more accurate
        representation of area in extracted network, but is quite slow, so
        it is ``False`` by default.  The default method simply counts voxels
        so does not correctly account for the voxelated nature of the images.

    Returns
    -------
    A dictionary containing the void phase size data, as well as the network
    topological information.  The dictionary names use the OpenPNM
    convention (i.e. 'pore.coords', 'throat.conns') so it may be converted
    directly to an OpenPNM network object using the ``update`` command.
    * ``net``: A dictionary containing all the void and solid phase size data,
        as well as the network topological information.  The dictionary names
        use the OpenPNM convention (i.e. 'pore.coords', 'throat.conns') so it
        may be converted directly to an OpenPNM network object using the
        ``update`` command.
    * ``im``: The binary image of the void space
    * ``dt``: The combined distance transform of the image
    * ``regions``: The void and solid space partitioned into pores and solids
        phases using a marker based watershed with the peaks found by the
        SNOW Algorithm.
    """

    # -------------------------------------------------------------------------
    # SNOW void phase
    regions = snow_partitioning(im=im, return_all=True)
    im = regions.im
    dt = regions.dt
    regions = regions.regions
    b_num = np.amax(regions)
    # -------------------------------------------------------------------------
    # Boundary Conditions
    regions = add_boundary_regions(regions=regions, faces=boundary_faces)
    # -------------------------------------------------------------------------
    # Padding distance transform and image to extract geometrical properties
    dt = pad_faces(im=dt, faces=boundary_faces)
    im = pad_faces(im=im, faces=boundary_faces)
    regions = regions*im
    regions = make_contiguous(regions)
    # -------------------------------------------------------------------------
    # Extract void and throat information from image
    net = regions_to_network(im=regions, dt=dt, voxel_size=voxel_size)
    # -------------------------------------------------------------------------
    # Extract marching cube surface area and interfacial area of regions
    if marching_cubes_area:
        areas = region_surface_areas(regions=regions)
        interface_area = region_interface_areas(regions=regions, areas=areas,
                                                voxel_size=voxel_size)
        net['pore.surface_area'] = areas * voxel_size**2
        net['throat.area'] = interface_area.area
    # -------------------------------------------------------------------------
    # Find void to void connections of boundary and internal voids
    boundary_labels = net['pore.label'] > b_num
    loc1 = net['throat.conns'][:, 0] < b_num
    loc2 = net['throat.conns'][:, 1] >= b_num
    pore_labels = net['pore.label'] <= b_num
    loc3 = net['throat.conns'][:, 0] < b_num
    loc4 = net['throat.conns'][:, 1] < b_num
    net['pore.boundary'] = boundary_labels
    net['throat.boundary'] = loc1 * loc2
    net['pore.internal'] = pore_labels
    net['throat.internal'] = loc3 * loc4
    # -------------------------------------------------------------------------
    # label boundary cells
    net = label_boundary_cells(network=net, boundary_faces=boundary_faces)
    # -------------------------------------------------------------------------
    # assign out values to dummy dict

    temp = _net_dict(net)
    temp.im = im.copy()
    temp.dt = dt
    temp.regions = regions
    return temp
