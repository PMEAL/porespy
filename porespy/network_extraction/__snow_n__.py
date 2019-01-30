import scipy as sp
from porespy.network_extraction import regions_to_network
from porespy.network_extraction import add_boundary_regions, label_boundary_cells
from porespy.network_extraction import assign_alias, snow_partitioning_n
from porespy.network_extraction import pad_distance_transform, connect_network_phases
from porespy.tools import make_contiguous
from porespy.metrics import region_surface_areas, region_interface_areas
from collections import namedtuple


def snow_n(im,
           voxel_size=1,
           boundary_faces=['top', 'bottom', 'left', 'right', 'front', 'back'],
           marching_cubes_area=False,
           alias=None):

    r"""
    Analyzes an image that has been partitioned into N phase regions
    and extracts all N phases geometerical information alongwith
    network connectivity between any ith and jth phase.

    Parameters
    ----------
    im : ND-array
        Image of porous material where each phase is represented by unique
        integer. Phase integer should start from 1. Boolean image will extract
        only one network labeled with True's only.

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

    alias : dict (Optional)
        A dictionary that assigns unique image label to specific phase.
        For example {1: 'Solid'} will show all structural properties associated
        with label 1 as Solid phase properties.
        If ``None`` then default labelling will be used i.e {1: 'Phase1',..}.

    Returns
    -------
    A dictionary containing all N phases size data, as well as the
    network topological information.  The dictionary names use the OpenPNM
    convention (i.e. 'pore.coords', 'throat.conns') so it may be converted
    directly to an OpenPNM network object using the ``update`` command.
    """
    # -------------------------------------------------------------------------
    # Get alias if provided by user
    al = assign_alias(im, alias=alias)
    # -------------------------------------------------------------------------
    # Perform snow on each phase and merge all segmentation and dt together
    snow = snow_partitioning_n(im, r_max=4, sigma=0.4, return_all=True,
                               mask=True, randomize=False, alias=al)
    # -------------------------------------------------------------------------
    # Add boundary regions
    f = boundary_faces
    regions = add_boundary_regions(regions=snow.regions, faces=f)
    # -------------------------------------------------------------------------
    # Padding distance transform to extract geometrical properties
    dt = pad_distance_transform(dt=snow.dt, boundary_faces=f)
    # -------------------------------------------------------------------------
    # For only one phase extraction with boundary regions
    phases_num = sp.unique(im * 1)
    phases_num = sp.trim_zeros(phases_num)
    if len(phases_num) == 1 and phases_num == 1:
        if f is not None:
            snow.im = pad_distance_transform(dt=snow.im, boundary_faces=f)
        regions = regions*snow.im
        regions = make_contiguous(regions)
    # -------------------------------------------------------------------------
    # Extract N phases sites and bond information from image
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
    # Find interconnection and interfacial area between ith and jth phases
    net = connect_network_phases(net=net, snow_partitioning_n=snow, alias=al,
                                 marching_cubes_area=marching_cubes_area)
    # -------------------------------------------------------------------------
    # label boundary cells
    net = label_boundary_cells(network=net, boundary_faces=f)
    # -------------------------------------------------------------------------
    # assign out values to namedtuple
    tup = namedtuple('results', field_names=['net', 'im', 'dt', 'regions'])
    tup.net = net
    tup.im = im.copy()
    tup.dt = dt
    tup.regions = regions
    return tup