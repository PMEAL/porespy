import numpy as np
from porespy.networks import regions_to_network
from porespy.networks import label_boundary_cells
from porespy.networks import add_boundary_regions
from porespy.networks import add_phase_interconnections
from porespy.tools import _create_alias_map
from porespy.networks import _net_dict
from porespy.filters import snow_partitioning_n
from porespy.tools import make_contiguous, pad_faces
from porespy.metrics import region_surface_areas, region_interface_areas


def snow_n(im,
           voxel_size=1,
           boundary_faces=['top', 'bottom', 'left', 'right', 'front', 'back'],
           marching_cubes_area=False,
           alias=None):
    r"""
    Analyzes an image that has been segemented into N phases and extracts all
    a network for each of the N phases, including geometerical information as
    well as network connectivity between each phase.

    Parameters
    ----------
    im : ND-array
        Image of porous material where each phase is represented by unique
        integer. Phase integer should start from 1 (0 is ignored)

    voxel_size : scalar
        The resolution of the image, expressed as the length of one side of a
        voxel, so the volume of a voxel would be **voxel_size**-cubed.  The
        default is 1, which is useful when overlaying the PNM on the original
        image since the scale of the image is always 1 unit lenth per voxel.

    boundary_faces : list of strings
        Boundary faces labels are provided to assign hypothetical boundary
        nodes having zero resistance to transport process. For cubical
        geometry, the user can choose ‘left’, ‘right’, ‘top’, ‘bottom’,
        ‘front’ and ‘back’ face labels to assign boundary nodes. If no label is
        assigned then all six faces will be selected as boundary nodes
        automatically which can be trimmed later on based on user requirements.

    marching_cubes_area : bool
        If ``True`` then the surface area and interfacial area between regions
        will be calculated using the marching cube algorithm. This is a more
        accurate representation of area in extracted network, but is quite
        slow, so it is ``False`` by default.  The default method simply counts
        voxels so does not correctly account for the voxelated nature of the
        images.

    alias : dict (Optional)
        A dictionary that assigns unique image label to specific phases. For
        example {1: 'Solid'} will show all structural properties associated
        with label 1 as Solid phase properties. If ``None`` then default
        labelling will be used i.e {1: 'Phase1',..}.

    Returns
    -------
    A dictionary containing all N phases size data, as well as the
    network topological information.  The dictionary names use the OpenPNM
    convention (i.e. 'pore.coords', 'throat.conns') so it may be converted
    directly to an OpenPNM network object using the ``update`` command.
    """
    # -------------------------------------------------------------------------
    # Get alias if provided by user
    al = _create_alias_map(im, alias=alias)
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
    dt = pad_faces(im=snow.dt, faces=f)
    # -------------------------------------------------------------------------
    # For only one phase extraction with boundary regions
    phases_num = np.unique(im).astype(int)
    phases_num = np.trim_zeros(phases_num)
    if len(phases_num) == 1:
        if f is not None:
            snow.im = pad_faces(im=snow.im, faces=f)
        regions = regions * (snow.im.astype(bool))
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
        net['pore.surface_area'] = areas * voxel_size ** 2
        net['throat.area'] = interface_area.area
    # -------------------------------------------------------------------------
    # Find interconnection and interfacial area between ith and jth phases
    net = add_phase_interconnections(net=net, snow_partitioning_n=snow,
                                     marching_cubes_area=marching_cubes_area,
                                     alias=al)
    # -------------------------------------------------------------------------
    # label boundary cells
    net = label_boundary_cells(network=net, boundary_faces=f)
    # -------------------------------------------------------------------------

    temp = _net_dict(net)
    temp.im = im.copy()
    temp.dt = dt
    temp.regions = regions
    return temp
