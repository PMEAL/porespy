from skimage.morphology import disk, square, ball, cube
import scipy as sp
import scipy.ndimage as spim
from tqdm import tqdm
from porespy.tools import extend_slice


def dual(net=None, solid_dt=None):

    print('_'*60)
    print('Extracting bond and site information for dual network')
    from skimage.morphology import disk, square, ball, cube
    solid_regions = net['dual.solid_regions']
    im = net['dual.image']
    Np = sp.size(net['pore.label'])
    s_im = solid_regions*~im              # Shows solid labels on solid region
    solid_on_pore = solid_regions*im      # Expose solid labels on pores
    s_dt = solid_dt

    if s_im.ndim == 2:
        cube = square
        ball = disk

    if ~sp.any(s_im == 0):
        raise Exception('The received image has no solid phase (0\'s)')

    if s_dt is None:
        s_dt = spim.distance_transform_edt(s_im > 0)
        s_dt = spim.gaussian_filter(input=s_dt, sigma=0.5)

    # Solid on Pore Slices
    s_slice = spim.find_objects(s_im)   # Slice of Solid region
    # Slice of Solid on pore regions
    solid_on_pore_slice = spim.find_objects(solid_on_pore)

    # Solid Space array initialization
    Ss = sp.arange(sp.amin(solid_regions), sp.amax(s_im)+1)
    Ns = sp.size(Ss)+Np
    s_coords = sp.zeros((Ns, s_im.ndim), dtype=float)
    s_volume = sp.zeros((Ns, ), dtype=float)
    s_dia_local = sp.zeros((Ns, ), dtype=float)
    s_dia_global = sp.zeros((Ns, ), dtype=float)
    s_label = sp.zeros((Ns, ), dtype=int)
    s_area_surf = sp.zeros((Ns, ), dtype=int)
    st_conns = []
    st_dia_inscribed = []
    st_area = []
    st_perimeter = []
    st_coords = []

    solid_pore_area_surf = sp.zeros((Ns, ), dtype=float)
    for k in tqdm(Ss):
        solid = k - 1
        ext_s_slice = extend_slice(s_slice[solid], s_im.shape)

        sub_im_s = s_im[ext_s_slice]  # Sub image of solid extended slice
        sub_dt_s = s_dt[ext_s_slice]  # Sub dt of solid ex_slice
        # Surface Area Calculation
        sub_im_s_dt = spim.distance_transform_edt(sub_im_s)
        sub_im_s_mask = sub_im_s == k
        sub_im_s_solid_labels = sub_im_s_mask*sub_im_s_dt
        solid_pore_area_surf[solid] = sp.sum(sub_im_s_solid_labels == 1)

        im_solid_s = sub_im_s == k
        dt_solid_s = spim.distance_transform_edt(sp.pad(im_solid_s,
                                                        pad_width=1,
                                                        mode='constant'))
        solid_offset = sp.array([kx.start for kx in ext_s_slice])
        s_label[solid] = k
        s_coords[solid, :] = spim.center_of_mass(im_solid_s) + solid_offset
        s_volume[solid] = sp.sum(im_solid_s)
        s_dia_local[solid] = 2*sp.amax(dt_solid_s)
        s_dia_global[solid] = 2*sp.amax(sub_dt_s)
        s_area_surf[solid] = sp.sum(dt_solid_s == 1)
        im_w_s_throats = spim.binary_dilation(input=im_solid_s,
                                              structure=disk(1))
        im_w_s_throats = im_w_s_throats*sub_im_s
        Sn = sp.unique(im_w_s_throats)[1:] - 1
        for l in Sn:
            if l > solid:
                st_conns.append([solid, l])
                vx_s = sp.where(im_w_s_throats == (l + 1))
                st_dia_inscribed.append(2*sp.amax(sub_dt_s[vx_s]))
                st_perimeter.append(sp.sum(sub_dt_s[vx_s] < 2))
                st_area.append(sp.size(vx_s[0]))
                st_inds = tuple([k+l for k, l in zip(vx_s, solid_offset)])
                temp_s = sp.where(s_dt[st_inds] ==
                                  sp.amax(s_dt[st_inds]))[0][0]
                if s_im.ndim == 2:
                    st_coords.append(tuple((st_inds[0][temp_s],
                                           st_inds[1][temp_s])))
                else:
                    st_coords.append(tuple((st_inds[0][temp_s],
                                           st_inds[1][temp_s],
                                           st_inds[2][temp_s])))

    Nst = len(st_dia_inscribed)  # Get number of throats
    if s_im.ndim == 2:  # If 2D, add 0's in 3rd dimension
        s_coords = sp.vstack((s_coords.T, sp.zeros((Ns, )))).T
        st_coords = sp.vstack((sp.array(st_coords).T, sp.zeros((Nst, )))).T

    v_inv = (net['dual.voxel_size'][0])**(-1)
    voxel_size = net['dual.voxel_size']
    bond_all = Ns
    site_all = len(net['throat.diameter']) + Nst
    p_coords = net['pore.coords'] * v_inv
    s_coords[net['pore.label']-1] = p_coords[net['pore.label']-1]
    bond_coords = s_coords
    pt_coords = ((net['throat.centroid']) * v_inv).tolist()
    site_coords = sp.vstack((pt_coords, st_coords))
    pt_conns = (net['throat.conns']).tolist()
    site_conns_partial = pt_conns + st_conns
    site_conns_full = pt_conns + st_conns + (net['dual.throat_conns']).tolist()
    net['pore.label'].resize(len(s_label), refcheck=False)
    bond_label = s_label + net['pore.label']
    p_volume = net['pore.volume'] * (v_inv)**3
    p_volume.resize(len(s_volume), refcheck=False)
    bond_volume = p_volume + s_volume
    site_volume = site_all
    p_dia_local = net['pore.inscribed_diameter'] * v_inv
    p_dia_local.resize(len(s_dia_local), refcheck=False)
    bond_dia_local = p_dia_local + s_dia_local
    p_dia_global = net['pore.extended_diameter'] * v_inv
    p_dia_global.resize(len(s_dia_global), refcheck=False)
    bond_dia_global = p_dia_global + s_dia_global
    p_area_surf = net['pore.surface_area'] * v_inv**2
    p_area_surf.resize(len(s_area_surf), refcheck=False)
    bond_surface_area = p_area_surf + s_area_surf
    pore_solid_area_surf = net['pore.solid_surface_area'] * v_inv**2
    pore_solid_area_surf.resize(len(solid_pore_area_surf), refcheck=False)
    bond_surf_area_phase = pore_solid_area_surf + solid_pore_area_surf
    pt_dia_inscribed = net['throat.inscribed_diameter'] * v_inv
    site_dia_inscribed = pt_dia_inscribed.tolist() + st_dia_inscribed
    pt_area = net['throat.area'] * v_inv**2
    site_area = pt_area.tolist() + st_area
    pt_perimeter = net['throat.perimeter'] * v_inv
    site_perimeter = pt_perimeter.tolist() + st_perimeter
    vol = net['pore.solid_volume']

    net = {}
    net['pore.all'] = sp.ones((bond_all, ), dtype=bool)
    net['throat.all'] = sp.ones((site_all, ), dtype=bool)
    net['pore.coords'] = sp.copy(bond_coords)*voxel_size
    net['pore.centroid'] = sp.copy(bond_coords)*voxel_size
    net['throat.centroid'] = sp.array(site_coords)*voxel_size
    net['pore.label'] = sp.array(bond_label)-1
    net['pore.label_pore'] = sp.trim_zeros(net['pore.label']-1)
    net['pore.label_solid'] = sp.trim_zeros(s_label-1)
    net['pore.volume'] = sp.copy(bond_volume)*(voxel_size**3)
    net['throat.volume'] = sp.zeros((site_volume, ), dtype=float)
    net['pore.diameter'] = sp.copy(bond_dia_local)*voxel_size
    net['pore.inscribed_diameter'] = sp.copy(bond_dia_local)*voxel_size
    net['pore.equivalent_diameter'] = 2*((3/4*net['pore.volume']/sp.pi)**(1/3))
    net['pore.extended_diameter'] = sp.copy(bond_dia_global)*voxel_size
    net['pore.surface_area'] = sp.copy(bond_surface_area)*(voxel_size)**2
    net['pore.solid_surf_area'] = sp.copy(bond_surf_area_phase)*(voxel_size)**2
    net['pore.solid_volume'] = vol
    net['throat.diameter'] = sp.array(site_dia_inscribed)*voxel_size
    net['throat.inscribed_diameter'] = sp.array(site_dia_inscribed)*voxel_size
    net['throat.area'] = sp.array(site_area)*(voxel_size**2)
    net['throat.perimeter'] = sp.array(site_perimeter)*voxel_size
    net['throat.equivalent_dia'] = (sp.array(site_area)*(voxel_size**2))**(0.5)
    net['throat.conns'] = sp.array(site_conns_partial)
    P12 = net['throat.conns']
    PT1 = sp.sqrt(sp.sum(((bond_coords[P12[:, 0]] -
                           site_coords)*voxel_size)**2, axis=1))
    PT2 = sp.sqrt(sp.sum(((bond_coords[P12[:, 1]] -
                           site_coords)*voxel_size)**2, axis=1))
    net['throat.total_length'] = PT1 + PT2
    PT1 = PT1-bond_dia_local[P12[:, 0]]/2*voxel_size
    PT2 = PT2-bond_dia_local[P12[:, 1]]/2*voxel_size
    net['throat.length'] = PT1 + PT2
    net['throat.conns'] = sp.array(site_conns_full)
    P13 = net['throat.conns']
    dist = (bond_coords[P13[:, 0]]-bond_coords[P13[:, 1]])*voxel_size
    net['throat.direct_length'] = sp.sqrt(sp.sum(dist**2, axis=1))

    return net
