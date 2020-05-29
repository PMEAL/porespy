import numpy as np
from porespy.networks import regions_to_network
from porespy.networks import add_boundary_regions
from porespy.networks import label_boundary_cells
from porespy.networks import _net_dict
from porespy.tools import pad_faces
from porespy.filters import snow_partitioning
from porespy.metrics import region_surface_areas, region_interface_areas


def snow_dual(im,
              voxel_size=1,
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
        as solid phase. It can process the inverted configuration of the
        boolean image as well, but output labelling of phases will be inverted
        and solid phase properties will be assigned to void phase properties
        labels which will cause confusion while performing the simulation.
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
    A dictionary containing all the void and solid phase size data, as well as
    the network topological information.  The dictionary names use the OpenPNM
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

    References
    ----------
    [1] Gostick, J. "A versatile and efficient network extraction algorithm
    using marker-based watershed segmenation".  Phys. Rev. E 96, 023307 (2017)

    [2] Khan, ZA et al.  "Dual network extraction algorithm to investigate
    multiple transport processes in porous materials: Image-based modeling
    of pore and grain-scale processes. Computers and Chemical Engineering.
    123(6), 64-77 (2019)

    """
    # -------------------------------------------------------------------------
    # SNOW void phase
    pore_regions = snow_partitioning(im, return_all=True)
    # SNOW solid phase
    solid_regions = snow_partitioning(~im, return_all=True)
    # -------------------------------------------------------------------------
    # Combined Distance transform of two phases.
    pore_dt = pore_regions.dt
    solid_dt = solid_regions.dt
    dt = pore_dt + solid_dt
    # Calculates combined void and solid regions for dual network extraction
    pore_regions = pore_regions.regions
    solid_regions = solid_regions.regions
    pore_region = pore_regions*im
    solid_region = solid_regions*~im
    solid_num = np.amax(pore_regions)
    solid_region = solid_region + solid_num
    solid_region = solid_region * ~im
    regions = pore_region + solid_region
    b_num = np.amax(regions)
    # -------------------------------------------------------------------------
    # Boundary Conditions
    regions = add_boundary_regions(regions=regions, faces=boundary_faces)
    # -------------------------------------------------------------------------
    # Padding distance transform to extract geometrical properties
    dt = pad_faces(im=dt, faces=boundary_faces)
    # -------------------------------------------------------------------------
    # Extract void,solid and throat information from image
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
    # Find void to void, void to solid and solid to solid throat conns
    loc1 = net['throat.conns'][:, 0] < solid_num
    loc2 = net['throat.conns'][:, 1] >= solid_num
    loc3 = net['throat.conns'][:, 1] < b_num
    pore_solid_labels = loc1 * loc2 * loc3

    loc4 = net['throat.conns'][:, 0] >= solid_num
    loc5 = net['throat.conns'][:, 0] < b_num
    solid_solid_labels = loc4 * loc2 * loc5 * loc3

    loc6 = net['throat.conns'][:, 1] < solid_num
    pore_pore_labels = loc1 * loc6

    loc7 = net['throat.conns'][:, 1] >= b_num
    boundary_throat_labels = loc5 * loc7

    solid_labels = ((net['pore.label'] > solid_num) * ~
                    (net['pore.label'] > b_num))
    boundary_labels = net['pore.label'] > b_num
    b_sa = np.zeros(len(boundary_labels[boundary_labels == 1.0]))
    # -------------------------------------------------------------------------
    # Calculates void interfacial area that connects with solid and vice versa
    p_conns = net['throat.conns'][:, 0][pore_solid_labels]
    ps = net['throat.area'][pore_solid_labels]
    p_sa = np.bincount(p_conns, ps)
    s_conns = net['throat.conns'][:, 1][pore_solid_labels]
    s_pa = np.bincount(s_conns, ps)
    s_pa = np.trim_zeros(s_pa)  # remove pore surface area labels
    p_solid_surf = np.concatenate((p_sa, s_pa, b_sa))
    # -------------------------------------------------------------------------
    # Calculates interfacial area using marching cube method
    if marching_cubes_area:
        ps_c = net['throat.area'][pore_solid_labels]
        p_sa_c = np.bincount(p_conns, ps_c)
        s_pa_c = np.bincount(s_conns, ps_c)
        s_pa_c = np.trim_zeros(s_pa_c)  # remove pore surface area labels
        p_solid_surf = np.concatenate((p_sa_c, s_pa_c, b_sa))
    # -------------------------------------------------------------------------
    # Adding additional information of dual network
    net['pore.solid_void_area'] = (p_solid_surf * voxel_size**2)
    net['throat.void'] = pore_pore_labels
    net['throat.interconnect'] = pore_solid_labels
    net['throat.solid'] = solid_solid_labels
    net['throat.boundary'] = boundary_throat_labels
    net['pore.void'] = net['pore.label'] <= solid_num
    net['pore.solid'] = solid_labels
    net['pore.boundary'] = boundary_labels
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
