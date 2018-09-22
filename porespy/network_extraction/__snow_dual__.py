import scipy as sp
from porespy.network_extraction import regions_to_network
from porespy.network_extraction import add_boundary_regions
from porespy.filters import snow_partitioning


def snow_dual(im, voxel_size=1, boundary_faces=['top', 'bottom', 'left',
                                                'right', 'front', 'back']):

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

    Returns
    -------
    A dictionary containing all the void and solid phase size data, as well as
    the network topological information.  The dictionary names use the OpenPNM
    convention (i.e. 'pore.coords', 'throat.conns') so it may be converted
    directly to an OpenPNM network object using the ``update`` command.
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
    solid_num = sp.amax(pore_regions)
    solid_region = solid_region + solid_num
    solid_region = solid_region * ~im
    regions = pore_region + solid_region
    b_num = sp.amax(regions)
    # -------------------------------------------------------------------------
    # Boundary Conditions
    regions = add_boundary_regions(regions=regions, faces=boundary_faces)
    # -------------------------------------------------------------------------
    # Padding distance transform to extract geometrical properties
    f = boundary_faces
    if f is not None:
        faces = [(int('top' in f)*3, int('bottom' in f)*3),
                 (int('left' in f)*3, int('right' in f)*3)]
        if im.ndim == 3:
            faces = [(int('front' in f)*3, int('back' in f)*3)] + faces
        dt = sp.pad(dt, pad_width=faces, mode='edge')
    else:
        dt = dt
    # -------------------------------------------------------------------------
    # Extract void,solid and throat information from image
    net = regions_to_network(im=regions, dt=dt, voxel_size=voxel_size)
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
    b_sa = sp.zeros(len(boundary_labels[boundary_labels == 1.0]))
    # -------------------------------------------------------------------------
    # Calculates void interfacial area that connects with solid and vice versa
    p_conns = net['throat.conns'][:, 0][pore_solid_labels]
    ps = net['throat.area'][pore_solid_labels]
    p_sa = sp.bincount(p_conns, ps)
    s_conns = net['throat.conns'][:, 1][pore_solid_labels]
    s_pa = sp.bincount(s_conns, ps)
    s_pa = sp.trim_zeros(s_pa)  # remove pore surface area labels
    p_solid_surf = sp.concatenate((p_sa, s_pa, b_sa))
    # -------------------------------------------------------------------------
    # Calculates interfacial area using marching cube method
    ps_c = net['throat.area_mc'][pore_solid_labels]
    p_sa_c = sp.bincount(p_conns, ps_c)
    s_pa_c = sp.bincount(s_conns, ps_c)
    s_pa_c = sp.trim_zeros(s_pa_c)  # remove pore surface area labels
    p_solid_surf_c = sp.concatenate((p_sa_c, s_pa_c, b_sa))
    # -------------------------------------------------------------------------
    # Adding additional information of dual network
    net['pore.solid_IFA'] = p_solid_surf * voxel_size**2
    net['pore.solid_IFA_mc'] = (p_solid_surf_c * voxel_size**2)
    net['throat.void'] = pore_pore_labels
    net['throat.interconnect'] = pore_solid_labels
    net['throat.solid'] = solid_solid_labels
    net['throat.boundary'] = boundary_throat_labels
    net['pore.void'] = net['pore.label'] <= solid_num
    net['pore.solid'] = solid_labels
    net['pore.boundary'] = boundary_labels
    return net
