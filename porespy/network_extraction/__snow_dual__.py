import scipy as sp
from porespy.network_extraction import snow, extract_pore_network
from porespy.tools import make_contiguous
from skimage.segmentation import find_boundaries
                                                                                                                                                                                                                         

def snow_dual_network(im, voxel_size=1 , 
                      boundary_faces = ['top','bottom','left','right',
                                        'front','back']):

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
        
    boundary_faces
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
    ##-------------------------------------------------------------------------
    # SNOW void phase
    pore_regions = snow(im)
    # SNOW solid phase
    solid_regions = snow(~im)
    ##-------------------------------------------------------------------------
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
    ##-------------------------------------------------------------------------    
    # Boundary Conditions
    regions , dt = define_boundary_nodes(regions=regions, dt=dt,
                        faces= boundary_faces)
    ##-------------------------------------------------------------------------      
    # Extract void,solid and throat information from image
    net = extract_pore_network(im=regions, dt=dt, voxel_size=voxel_size)
    ##-------------------------------------------------------------------------
    # Find void to void, void to solid and solid to solid throat conns
    loc1 = net['throat.conns'][:, 0] < solid_num
    loc2 = net['throat.conns'][:, 1] >= solid_num
    loc3 = net['throat.conns'][:, 1] < b_num
    pore_solid_labels = loc1 * loc2 * loc3
    
    loc4 = net['throat.conns'][:, 0] >= solid_num
    loc5 = net['throat.conns'][:, 0] < b_num
    loc6 = net['throat.conns'][:, 1] < b_num
    solid_solid_labels = loc4 * loc2 * loc5 * loc6
    
    loc7 = net['throat.conns'][:, 1] < solid_num
    pore_pore_labels = loc1 * loc7
    
    loc8 = net['throat.conns'][:, 0] < b_num
    loc9 = net['throat.conns'][:, 1] >= b_num
    boundary_throat_labels = loc8 * loc9 
    
    solid_labels = (net['pore.label'] > solid_num) * ~(net['pore.label'] > b_num)
    boundary_labels = net['pore.label'] > b_num
    t_sa = sp.zeros(len(boundary_labels[boundary_labels==True]))
    ##-------------------------------------------------------------------------
    # Calculates void surface area that connects with solid and vice versa
    p_conns = net['throat.conns'][:, 0][pore_solid_labels]
    ps = net['throat.area'][pore_solid_labels]
    p_sa = sp.bincount(p_conns, ps)
    s_conns = net['throat.conns'][:, 1][pore_solid_labels]
    s_sa = sp.bincount(s_conns, ps)
    s_sa = sp.trim_zeros(s_sa)
    p_solid_surf = sp.concatenate((p_sa, s_sa, t_sa))
    ##-------------------------------------------------------------------------
    # Adding additional information of dual network
    net['pore.solid_interfacial_area'] = p_solid_surf * voxel_size**2
    net['throat.void'] = pore_pore_labels
    net['throat.interconnect'] = pore_solid_labels
    net['throat.solid'] = solid_solid_labels
    net['throat.boundary'] = boundary_throat_labels
    net['pore.void'] = net['pore.label'] <= solid_num
    net['pore.solid'] = solid_labels
    net['pore.boundary'] = boundary_labels
    return net


def define_boundary_nodes(regions=None, dt=None, 
              faces=['front','back','left','right','top','bottom']):
    ##-------------------------------------------------------------------------
    # Edge pad segmentation and distance transform 
    regions = sp.pad(regions,1,'edge')
    dt = sp.pad(dt,1,'edge')
    ##-------------------------------------------------------------------------
    # Remove boundary nodes interconnection
    regions[:,:,0] = regions[:,:,0] + regions.max()
    regions[:,:,-1] = regions[:,:,-1] + regions.max()
    regions[0,:,:] = regions[0,:,:] + regions.max()
    regions[-1,:,:] = regions[-1,:,:] + regions.max()
    regions[:,0,:] = regions[:,0,:] + regions.max()
    regions[:,-1,:] = regions[:,-1,:] + regions.max()
    regions[:,:,0] = (~find_boundaries(regions[:,:,0],
           mode='outer'))*regions[:,:,0]
    regions[:,:,-1] = (~find_boundaries(regions[:,:,-1],
           mode='outer'))*regions[:,:,-1]
    regions[0,:,:] = (~find_boundaries(regions[0,:,:],
           mode='outer'))*regions[0,:,:]
    regions[-1,:,:] = (~find_boundaries(regions[-1,:,:],
           mode='outer'))*regions[-1,:,:]
    regions[:,0,:] = (~find_boundaries(regions[:,0,:],
           mode='outer'))*regions[:,0,:]
    regions[:,-1,:] = (~find_boundaries(regions[:,-1,:],
           mode='outer'))*regions[:,-1,:]
    ##-------------------------------------------------------------------------
    # Remove unselected faces
    if 'top' not in faces:
        regions = regions[:,:,1:]
        dt      = dt[:,:,1:]
    if 'bottom' not in faces:
        regions = regions[:,:,:-1]
        dt      = dt[:,:,:-1]
    if 'front' not in faces:
        regions = regions[:,1:,:]
        dt      = dt[:,1:,:]
    if 'back' not in faces:
        regions = regions[:,:-1,:]
        dt      = dt[:,1:,:]
    if 'left' not in faces:
        regions = regions[1:,:,:]
        dt      = dt[1:,:,:]
    if 'right' not in faces:
        regions = regions[:-1,:,:]
        dt      = dt[:-1,:,:]
    ##-------------------------------------------------------------------------
    # Make labels contiguous
    regions = make_contiguous(regions)
    return regions,dt
