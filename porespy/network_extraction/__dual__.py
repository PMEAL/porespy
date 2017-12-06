from skimage.morphology import disk, square, ball, cube
import scipy as sp
import scipy.ndimage as spim
from tqdm import tqdm
from porespy.tools import extend_slice


def dual(solid_regions, pore_regions, im, pore_dt=None, solid_dt=None,
         voxel_size=1):
    # Pore on Solid
    merge = sp.amax(pore_regions)
    p_im = pore_regions*im                # Shows pore labels on pore region
    pore_on_solid = pore_regions*(~im)    # Expose pores labels on solids
    # Solid on Pore
    solid_regions = solid_regions+(merge)
    s_im = solid_regions*~im              # Shows solid labels on solid region
    solid_on_pore = solid_regions*im      # Expose solid labels on pores
    s_dt = solid_dt
    p_dt = pore_dt
    if p_im.ndim == 2:
        cube = square
        ball = disk

    if ~sp.any(p_im == 0):
        raise Exception('The received image has no solid phase (0\'s)')

    if p_dt is None:
        p_dt = spim.distance_transform_edt(p_im > 0)
        p_dt = spim.gaussian_filter(input=p_dt, sigma=0.5)

    if s_im.ndim == 2:
        cube = square
        ball = disk

    if ~sp.any(s_im == 0):
        raise Exception('The received image has no solid phase (0\'s)')

    if s_dt is None:
        s_dt = spim.distance_transform_edt(s_im > 0)
        s_dt = spim.gaussian_filter(input=s_dt, sigma=0.5)

    # Pore on Solid Slices
    p_slice = spim.find_objects(p_im)   # Slice of Pore region
    # Slice of Pore on solid regions
    pore_on_solid_slice = spim.find_objects(pore_on_solid)
    # Solid on Pore Slices
    s_slice = spim.find_objects(s_im)   # Slice of Solid region
    # Slice of Solid on pore regions
    solid_on_pore_slice = spim.find_objects(solid_on_pore)

    # %% Pore space array initialization
    # Initialize arrays
    Ps = sp.arange(1, sp.amax(p_im)+1)
    Np = sp.size(Ps)
    p_coords = sp.zeros((Np, p_im.ndim), dtype=float)
    p_volume = sp.zeros((Np, ), dtype=float)
    p_dia_local = sp.zeros((Np, ), dtype=float)
    p_dia_global = sp.zeros((Np, ), dtype=float)
    p_label = sp.zeros((Np, ), dtype=int)
    p_area_surf = sp.zeros((Np, ), dtype=int)
    p_area_solid = sp.zeros((Np, ), dtype=int)
    pt_conns = []
    pt_dia_inscribed = []
    pt_area = []
    pt_perimeter = []
    pt_coords = []
    pore_solid_conns = []
    pore_solid_area_surf = sp.zeros((Np, ), dtype=int)
    pore_solid_volume = sp.zeros((Np, ), dtype=int)

    # Start extracting size information for pores and throats
    for i in tqdm(Ps):
        pore = i - 1
    #        if slices[pore] is None:
    #            continue
        ext_p_slice = extend_slice(p_slice[pore], p_im.shape)
        ext_pore_on_solid_slice = extend_slice(pore_on_solid_slice[pore],
                                               p_im.shape)
        sub_im = p_im[ext_p_slice]  # Sub image of pore extended slice
        # Sub distance transform of pore extended slice
        sub_dt = p_dt[ext_p_slice]
        # Chunk of solid connected with pore
        # Sub image of solid extended slide
        sub_sim = pore_on_solid[ext_pore_on_solid_slice]
        # Sub distance transform of solid extended slide
        sub_sdt = spim.distance_transform_edt(sub_sim)
        sub_sdt = sub_sdt == 1  # Finding only surface region of solid
        # Detecting chunk of solid connected with ith pore
        solid_im = (sub_sdt*sub_sim) == i
        pore_solid_area_surf[pore] = sp.sum(solid_im)
        pore_solid_volume[pore] = sp.sum(sub_sim == i)
        # %% Finding Pore Solid Neighbours
        pore_regions_full = pore_regions[ext_pore_on_solid_slice]
        pore_region_mask = pore_regions[ext_pore_on_solid_slice] == i
        solid_regions_full = solid_regions[ext_pore_on_solid_slice]
        solid_labels_on_pore = ((pore_region_mask*pore_regions_full != 0) *
                                solid_regions_full)
        neigbhour_solid_labels = sp.unique(solid_labels_on_pore)
        neigbhour_solid_labels = sp.delete(neigbhour_solid_labels,
                                           sp.where(neigbhour_solid_labels ==
                                                    0))
        neigbhour_solid_labels = neigbhour_solid_labels-1
        for solid_label in neigbhour_solid_labels:
            pore_solid_conns.append([pore, solid_label])
        # %% Old Code of porespy
        # True/False form Sub image of ith pore extended slice
        pore_im = sub_im == i
        pore_dt = spim.distance_transform_edt(sp.pad(pore_im, pad_width=1,
                                                     mode='constant'))
        s_offset = sp.array([ix.start for ix in ext_p_slice])
        p_label[pore] = i
        p_coords[pore, :] = spim.center_of_mass(pore_im) + s_offset
        p_volume[pore] = sp.sum(pore_im)
        p_dia_local[pore] = 2*sp.amax(pore_dt)
        p_dia_global[pore] = 2*sp.amax(sub_dt)
        p_area_surf[pore] = sp.sum(pore_dt == 1)
        p_area_solid[pore] = sp.sum(sub_dt == 1)
        im_w_throats = spim.binary_dilation(input=pore_im, structure=disk(1))
        im_w_throats = im_w_throats*sub_im
        Pn = sp.unique(im_w_throats)[1:] - 1
        for j in Pn:
            if j > pore:
                pt_conns.append([pore, j])
                vx = sp.where(im_w_throats == (j + 1))
                pt_dia_inscribed.append(2*sp.amax(sub_dt[vx]))
                pt_perimeter.append(sp.sum(sub_dt[vx] < 2))
                pt_area.append(sp.size(vx[0]))
                t_inds = tuple([i+j for i, j in zip(vx, s_offset)])
                temp = sp.where(p_dt[t_inds] == sp.amax(p_dt[t_inds]))[0][0]
                if p_im.ndim == 2:
                    pt_coords.append(tuple((t_inds[0][temp],
                                           t_inds[1][temp])))
                else:
                    pt_coords.append(tuple((t_inds[0][temp],
                                           t_inds[1][temp],
                                           t_inds[2][temp])))
    # Clean up values
    Nt = len(pt_dia_inscribed)  # Get number of throats
    if p_im.ndim == 2:  # If 2D, add 0's in 3rd dimension
        p_coords = sp.vstack((p_coords.T, sp.zeros((Np, )))).T
        pt_coords = sp.vstack((sp.array(pt_coords).T, sp.zeros((Nt, )))).T

    print('_'*60)
    print('Extracting Solid and Solid throats information from image')
    # %% Solid Space array initialization
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
    solid_pore_conns = []
    solid_pore_area_surf = sp.zeros((Ns, ), dtype=float)
    for k in tqdm(Ss):
        solid = k - 1
        ext_s_slice = extend_slice(s_slice[solid], s_im.shape)
        ext_solid_on_pore_slice = extend_slice(solid_on_pore_slice[solid],
                                               s_im.shape)
        sub_im_s = s_im[ext_s_slice]  # Sub image of solid extended slice
        sub_dt_s = s_dt[ext_s_slice]  # Sub dt of solid ex_slice
        # %% Surface Area Calculation
        sub_im_s_dt = spim.distance_transform_edt(sub_im_s)
        sub_im_s_mask = sub_im_s == k
        sub_im_s_solid_labels = sub_im_s_mask*sub_im_s_dt
        solid_pore_area_surf[solid] = sp.sum(sub_im_s_solid_labels == 1)

        # %% Finding Solid pore Neighbours
        solid_region_full = solid_regions[ext_solid_on_pore_slice]
        solid_region_mask = solid_regions[ext_solid_on_pore_slice] == k
        pore_region_full = pore_regions[ext_solid_on_pore_slice]
        pore_labels_on_solid = ((solid_region_mask*solid_region_full != 0) *
                                pore_region_full)
        neigbhour_pore_labels = sp.unique(pore_labels_on_solid)
        neigbhour_pore_labels = sp.delete(neigbhour_pore_labels,
                                          sp.where(neigbhour_pore_labels == 0))
        neigbhour_pore_labels = neigbhour_pore_labels-1
        for pore_label in neigbhour_pore_labels:
            solid_pore_conns.append([solid, pore_label])

        # %% New Code of porespy for solid extraction
        # True/False form Sub image of kth solid extended slice
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

    bond_all = Ns
    site_all = Nt + Nst
    s_coords[Ps-1] = p_coords[Ps-1]
    bond_coords = s_coords
    site_coords = sp.vstack((pt_coords, st_coords))
    site_conns_partial = pt_conns + st_conns
    site_conns_full = pt_conns + st_conns+pore_solid_conns
    p_label.resize(len(s_label), refcheck=False)
    bond_label = s_label+p_label
    p_volume.resize(len(s_volume), refcheck=False)
    bond_volume = p_volume + s_volume
    site_volume = site_all
    p_dia_local.resize(len(s_dia_local), refcheck=False)
    bond_dia_local = p_dia_local + s_dia_local
    p_dia_global.resize(len(s_dia_global), refcheck=False)
    bond_dia_global = p_dia_global + s_dia_global
    p_area_surf.resize(len(s_area_surf), refcheck=False)
    bond_surface_area = p_area_surf + s_area_surf
    pore_solid_area_surf.resize(len(solid_pore_area_surf), refcheck=False)
    bond_surf_area_phase = pore_solid_area_surf + solid_pore_area_surf
    site_dia_inscribed = pt_dia_inscribed + st_dia_inscribed
    site_area = pt_area + st_area
    site_perimeter = pt_perimeter + st_perimeter

    net = {}
    net['pore.all'] = sp.ones((bond_all, ), dtype=bool)
    net['throat.all'] = sp.ones((site_all, ), dtype=bool)
    net['pore.coords'] = sp.copy(bond_coords)*voxel_size
    net['pore.centroid'] = sp.copy(bond_coords)*voxel_size
    net['throat.centroid'] = sp.array(site_coords)*voxel_size
    net['pore.label'] = sp.array(bond_label)-1
    net['pore.label_pore'] = sp.trim_zeros(sp.array(p_label))-1
    net['pore.label_solid'] = sp.trim_zeros(sp.array(s_label))-1
    net['pore.volume'] = sp.copy(bond_volume)*(voxel_size**3)
    net['throat.volume'] = sp.zeros((site_volume, ), dtype=float)
    net['pore.diameter'] = sp.copy(bond_dia_local)*voxel_size
    net['pore.inscribed_diameter'] = sp.copy(bond_dia_local)*voxel_size
    net['pore.equivalent_diameter'] = 2*((3/4*net['pore.volume']/sp.pi)**(1/3))
    net['pore.extended_diameter'] = sp.copy(bond_dia_global)*voxel_size
    net['pore.surface_area'] = sp.copy(bond_surface_area)*(voxel_size)**2
    net['pore.solid_surf_area'] = sp.copy(bond_surf_area_phase)*(voxel_size)**2
    net['pore.solid_volume'] = sp.copy(pore_solid_volume)*(voxel_size**3)
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
