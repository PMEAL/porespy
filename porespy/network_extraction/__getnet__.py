import scipy as sp
import scipy.ndimage as spim
from scipy.spatial.distance import cdist
from porespy.tools import extend_slice
from tqdm import tqdm


def extract_pore_network(im, pore_regions=None, dt=None, voxel_size=1,
                         dual=False, solid_regions=None):
    r"""
    Analyzes an image that has been partitioned into pore regions and extracts
    the pore and throat geometry as well as network connectivity.

    Parameters
    ----------
    im : ND-array
        An image of the pore space partitioned into individual pore regions.
        Note that this image must have zeros indicating the solid phase.

    dt : ND-array
        The distance transform of the pore space.  If not given it will be
        calculated, but it can save time to provide one if available.

    voxel_size : scalar
        The resolution of the image, expressed as the length of one side of a
        voxel, so the volume of a voxel would be **voxel_size**-cubed.  The
        default is 1, which is useful when overlaying the PNM on the original
        image since the scale of the image is alway 1 unit lenth per voxel.

    pore_rgions : ND-array
        A ND-array the same size as ``dt`` with regions belonging to each peak
        labelled.

    dual : Bool
        If True dual network extraction parameters will be calculated.In such
        case solid_regions are mandatory input.
        If False only single phase network extraction will be processed.

    solid_regions : ND-array
        A ND-array the same size as ``dt`` with regions belonging to each peak
        labelled.

    Returns
    -------
    A dictionary containing all the pore and throat size data, as well as the
    network topological information.  The dictionary names use the OpenPNM
    convention (i.e. 'pore.coords', 'throat.conns') so it may be converted
    directly to an OpenPNM network object using the ``update`` command.
    It also find neccessary parameters to extract dual network if dual is set
    to True
    """
    print('_'*60)
    print('Extracting pore and throat information from image')
    from skimage.morphology import disk, square, ball, cube
    if im.ndim == 2:
        cube = square
        ball = disk

    if ~sp.any(im == 0):
        raise Exception('The received image has no solid phase (0\'s)')

    if dt is None:
        dt = spim.distance_transform_edt(im > 0)
        dt = spim.gaussian_filter(input=dt, sigma=0.5)

    # Pore on Solid
    if (dual is True) and (solid_regions is None):
        raise Exception('Please provide solid regions map for' +
                        ' dual newtwork extraction')

    merge = sp.amax(pore_regions)
    solid_regions = solid_regions + merge
    p_on_s = pore_regions*(~im)    # Expose pores labels on solid
    p_on_s_slice = spim.find_objects(p_on_s)  # Slice of Pore on solid regions
    p_im = pore_regions*im                # Shows pore labels on pore region

    # Get 'slices' into im for each pore region
    slices = spim.find_objects(p_im)

    # Initialize arrays
    Ps = sp.arange(1, sp.amax(p_im)+1)
    Np = sp.size(Ps)
    p_coords = sp.zeros((Np, p_im.ndim), dtype=float)
    p_volume = sp.zeros((Np, ), dtype=float)
    p_dia_local = sp.zeros((Np, ), dtype=float)
    p_dia_global = sp.zeros((Np, ), dtype=float)
    p_label = sp.zeros((Np, ), dtype=int)
    p_area_surf = sp.zeros((Np, ), dtype=int)
    t_conns = []
    t_dia_inscribed = []
    t_area = []
    t_perimeter = []
    t_coords = []
    p_solid_area_surf = sp.zeros((Np, ), dtype=int)
    p_solid_volume = sp.zeros((Np, ), dtype=int)
    pore_solid_conns = []
    # Start extracting size information for pores and throats
    for i in tqdm(Ps):
        pore = i - 1
#        if slices[pore] is None:
#            continue
        s = extend_slice(slices[pore], p_im.shape)
        sub_im = p_im[s]
        sub_dt = dt[s]

        if dual is True:

            # This calculates chunk of solid properties connected with pore
            ext_p_on_s_slice = extend_slice(p_on_s_slice[pore], p_im.shape)
            # Sub image of solid extended slice
            sub_sim = p_on_s[ext_p_on_s_slice]
            # Sub distance transform of solid extended slide
            sub_sdt = spim.distance_transform_edt(sub_sim)
            sub_sdt = sub_sdt == 1  # Finding only surface region of solid
            # Detecting chunk of solid connected with ith pore
            solid_im = (sub_sdt*sub_sim) == i
            p_solid_area_surf[pore] = sp.sum(solid_im)
            p_solid_volume[pore] = sp.sum(sub_sim == i)

            # Finding Pore Solid Neighbours
            pore_regions_full = pore_regions[ext_p_on_s_slice]
            pore_region_mask = pore_regions[ext_p_on_s_slice] == i
            solid_regions_full = solid_regions[ext_p_on_s_slice]
            solid_labels_on_pore = ((pore_region_mask * pore_regions_full != 0)
                                    * solid_regions_full)
            neigbhour_solid = sp.unique(solid_labels_on_pore)
            neigbhour_solid = sp.delete(neigbhour_solid,
                                        sp.where(neigbhour_solid == 0))
            neigbhour_solid = neigbhour_solid-1
            for solid_label in neigbhour_solid:
                pore_solid_conns.append([pore, solid_label])

        pore_im = sub_im == i
        pore_dt = spim.distance_transform_edt(sp.pad(pore_im, pad_width=1,
                                                     mode='constant'))
        s_offset = sp.array([i.start for i in s])
        p_label[pore] = i
        p_coords[pore, :] = spim.center_of_mass(pore_im) + s_offset
        p_volume[pore] = sp.sum(pore_im)
        p_dia_local[pore] = 2*sp.amax(pore_dt)
        p_dia_global[pore] = 2*sp.amax(sub_dt)
        p_area_surf[pore] = sp.sum(pore_dt == 1)
        im_w_throats = spim.binary_dilation(input=pore_im, structure=ball(1))
        im_w_throats = im_w_throats*sub_im
        Pn = sp.unique(im_w_throats)[1:] - 1
        for j in Pn:
            if j > pore:
                t_conns.append([pore, j])
                vx = sp.where(im_w_throats == (j + 1))
                t_dia_inscribed.append(2*sp.amax(sub_dt[vx]))
                t_perimeter.append(sp.sum(sub_dt[vx] < 2))
                t_area.append(sp.size(vx[0]))
                t_inds = tuple([i+j for i, j in zip(vx, s_offset)])
                temp = sp.where(dt[t_inds] == sp.amax(dt[t_inds]))[0][0]
                if p_im.ndim == 2:
                    t_coords.append(tuple((t_inds[0][temp],
                                           t_inds[1][temp])))
                else:
                    t_coords.append(tuple((t_inds[0][temp],
                                           t_inds[1][temp],
                                           t_inds[2][temp])))
    # Clean up values
    Nt = len(t_dia_inscribed)  # Get number of throats
    if p_im.ndim == 2:  # If 2D, add 0's in 3rd dimension
        p_coords = sp.vstack((p_coords.T, sp.zeros((Np, )))).T
        t_coords = sp.vstack((sp.array(t_coords).T, sp.zeros((Nt, )))).T

    # Start creating dictionary of pore network information
    net = {}
    net['pore.all'] = sp.ones((Np, ), dtype=bool)
    net['throat.all'] = sp.ones((Nt, ), dtype=bool)
    net['pore.coords'] = sp.copy(p_coords)*voxel_size
    net['pore.centroid'] = sp.copy(p_coords)*voxel_size
    net['throat.centroid'] = sp.array(t_coords)*voxel_size
    net['throat.conns'] = sp.array(t_conns)
    net['pore.label'] = sp.array(p_label)
    net['pore.volume'] = sp.copy(p_volume)*(voxel_size**3)
    net['throat.volume'] = sp.zeros((Nt, ), dtype=float)
    net['pore.diameter'] = sp.copy(p_dia_local)*voxel_size
    net['pore.inscribed_diameter'] = sp.copy(p_dia_local)*voxel_size
    net['pore.equivalent_diameter'] = 2*((3/4*net['pore.volume']/sp.pi)**(1/3))
    net['pore.extended_diameter'] = sp.copy(p_dia_global)*voxel_size
    net['pore.surface_area'] = sp.copy(p_area_surf)*(voxel_size)**2
    net['throat.diameter'] = sp.array(t_dia_inscribed)*voxel_size
    net['throat.inscribed_diameter'] = sp.array(t_dia_inscribed)*voxel_size
    net['throat.area'] = sp.array(t_area)*(voxel_size**2)
    net['throat.perimeter'] = sp.array(t_perimeter)*voxel_size
    net['throat.equivalent_diameter'] = ((sp.array(t_area)
                                         * (voxel_size**2))**(0.5))
    P12 = net['throat.conns']
    PT1 = (sp.sqrt(sp.sum(((p_coords[P12[:, 0]]-t_coords)
                           * voxel_size)**2, axis=1)))
    PT2 = (sp.sqrt(sp.sum(((p_coords[P12[:, 1]]-t_coords)
                           * voxel_size)**2, axis=1)))
    net['throat.total_length'] = PT1 + PT2
    PT1 = PT1-p_dia_local[P12[:, 0]]/2*voxel_size
    PT2 = PT2-p_dia_local[P12[:, 1]]/2*voxel_size
    net['throat.length'] = PT1 + PT2
    dist = (p_coords[P12[:, 0]]-p_coords[P12[:, 1]])*voxel_size
    net['throat.direct_length'] = sp.sqrt(sp.sum(dist**2, axis=1))

    if dual is True:
        net['pore.solid_surface_area'] = sp.copy((p_solid_area_surf)
                                                 * (voxel_size)**2)
        net['pore.solid_volume'] = sp.copy(p_solid_volume)*(voxel_size**3)
        net['dual.throat_conns'] = sp.array(pore_solid_conns)
        net['dual.solid_regions'] = solid_regions
        net['dual.pore_regions'] = pore_regions
        net['dual.image'] = im
        net['dual.voxel_size'] = sp.array([voxel_size], dtype=float)

    return net
