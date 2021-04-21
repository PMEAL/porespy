import numpy as np
import scipy.ndimage as spim
from edt import edt
from loguru import logger
from skimage.morphology import ball, disk

from porespy import settings
from porespy.metrics import region_interface_areas, region_surface_areas
from porespy.tools import extend_slice, get_tqdm, make_contiguous

tqdm = get_tqdm()


def regions_to_network(regions, phases=None, voxel_size=1, accuracy='standard'):
    r"""
    Analyzes an image that has been partitioned into pore regions and extracts
    the pore and throat geometry as well as network connectivity.

    Parameters
    ----------
    regions : ND-array
        An image of the material partitioned into individual regions.
        Zeros in this image are ignored.
    phases : ND-array, optional
        An image indicating to which phase each voxel belongs. The returned
        network contains a 'pore.phase' array with the corresponding value.
        If not given a value of 1 is assigned to every pore.
    voxel_size : scalar (default = 1)
        The resolution of the image, expressed as the length of one side of a
        voxel, so the volume of a voxel would be **voxel_size**-cubed.
    accuracy : string
        Controls how accurately certain properties are calculated. Options are:

        'standard' (default)
            Computes the surface areas and perimeters by simply counting
            voxels.  This is *much* faster but does not properly account
            for the rough, voxelated nature of the surfaces.
        'high'
            Computes surface areas using the marching cube method, and
            perimeters using the fast marching method.  These are substantially
            slower but better account for the voxelated nature of the images.

    Returns
    -------
    net : dict
        A dictionary containing all the pore and throat size data, as well as
        the network topological information.  The dictionary names use the
        OpenPNM convention (i.e. 'pore.coords', 'throat.conns').

    Notes
    -----
    The meaning of each of the values returned in ``net`` are outlined below:

    - 'pore.region_label': The region labels corresponding to the watershed
    extraction. The pore indices and regions labels will be offset by 1, so
    pore 0 will be region 1.

    'throat.conns'
        An *Nt-by-2* array indicating which pores are connected to each other
    'pore.region_label'
        Mapping of regions in the watershed segmentation to pores in the
        network
    'pore.local_peak'
        The coordinates of the location of the maxima of the distance transform
        performed on the pore region in isolation
    'pore.global_peak'
        The coordinates of the location of the maxima of the distance transform
        performed on the full image
    'pore.geometric_centroid'
        The center of mass of the pore region as calculated by
        ``skimage.measure.center_of_mass``
    'throat.global_peak'
        The coordinates of the location of the maxima of the distance transform
        performed on the full image
    'pore.region_volume'
        The volume of the pore region computed by summing the voxels
    'pore.volume'
        The volume of the pore found by as volume of a mesh obtained from the
        marching cubes algorithm
    'pore.surface_area'
        blah
    'throat.cross_sectional_area'
        blah
    'throat.perimeter'
        blah
    'pore.inscribed_diameter'
        blah
    'pore.extended_diameter'
        blah
    'throat.inscribed_diameter'
        blah
    'throat.total_length'
        blah
    'throat.direct_length'
        blah

    """
    logger.trace('Extracting pore/throat information')

    im = make_contiguous(regions)
    struc_elem = disk if im.ndim == 2 else ball

    if phases is None:
        phases = (im > 0).astype(int)
    if im.size != phases.size:
        raise Exception('regions and phase are different sizes, probably ' +
                        'because boundary regions were not added to phases')
    dt = edt(im > 0)

    # Get 'slices' into im for each pore region
    slices = spim.find_objects(im)

    # Initialize arrays
    Ps = np.arange(1, np.amax(im)+1)
    Np = np.size(Ps)
    p_coords_cm = np.zeros((Np, im.ndim), dtype=float)
    p_coords_dt = np.zeros((Np, im.ndim), dtype=float)
    p_coords_dt_global = np.zeros((Np, im.ndim), dtype=float)
    p_volume = np.zeros((Np, ), dtype=float)
    p_dia_local = np.zeros((Np, ), dtype=float)
    p_dia_global = np.zeros((Np, ), dtype=float)
    p_label = np.zeros((Np, ), dtype=int)
    p_area_surf = np.zeros((Np, ), dtype=int)
    p_phase = np.zeros((Np, ), dtype=int)
    # The number of throats is not known at the start, so lists are used
    # which can be dynamically resized more easily.
    t_conns = []
    t_dia_inscribed = []
    t_area = []
    t_perimeter = []
    t_coords = []

    # Start extracting size information for pores and throats
    for i in tqdm(Ps, **settings.tqdm):
        pore = i - 1
        if slices[pore] is None:
            continue
        s = extend_slice(slices[pore], im.shape)
        sub_im = im[s]
        sub_dt = dt[s]
        pore_im = sub_im == i
        padded_mask = np.pad(pore_im, pad_width=1, mode='constant')
        pore_dt = edt(padded_mask)
        s_offset = np.array([i.start for i in s])
        p_label[pore] = i
        p_coords_cm[pore, :] = spim.center_of_mass(pore_im) + s_offset
        temp = np.vstack(np.where(pore_dt == pore_dt.max()))[:, 0]
        p_coords_dt[pore, :] = temp + s_offset
        p_phase[pore] = (phases[s]*pore_im).max()
        temp = np.vstack(np.where(sub_dt == sub_dt.max()))[:, 0]
        p_coords_dt_global[pore, :] = temp + s_offset
        p_volume[pore] = np.sum(pore_im)
        p_dia_local[pore] = 2*np.amax(pore_dt)
        p_dia_global[pore] = 2*np.amax(sub_dt)
        # The following is overwritten if use_marching_cubes is True
        p_area_surf[pore] = np.sum(pore_dt == 1)
        im_w_throats = spim.binary_dilation(input=pore_im, structure=struc_elem(1))
        im_w_throats = im_w_throats*sub_im
        Pn = np.unique(im_w_throats)[1:] - 1
        for j in Pn:
            if j > pore:
                t_conns.append([pore, j])
                vx = np.where(im_w_throats == (j + 1))
                t_dia_inscribed.append(2*np.amax(sub_dt[vx]))
                # The following is overwritten if accuracy is set to 'high'
                t_perimeter.append(np.sum(sub_dt[vx] < 2))
                # The following is overwritten if accuracy is set to 'high'
                t_area.append(np.size(vx[0]))
                p_area_surf[pore] -= np.size(vx[0])
                t_inds = tuple([i+j for i, j in zip(vx, s_offset)])
                temp = np.where(dt[t_inds] == np.amax(dt[t_inds]))[0][0]
                t_coords.append(tuple([t_inds[k][temp] for k in range(im.ndim)]))

    # Clean up values
    p_coords = p_coords_cm
    Nt = len(t_dia_inscribed)  # Get number of throats
    if im.ndim == 2:  # If 2D, add 0's in 3rd dimension
        p_coords = np.vstack((p_coords_cm.T, np.zeros((Np, )))).T
        t_coords = np.vstack((np.array(t_coords).T, np.zeros((Nt, )))).T

    net = {}
    # Define all the fundamental stuff
    net['throat.conns'] = np.array(t_conns)
    net['pore.coords'] = np.array(p_coords)
    net['pore.all'] = np.ones_like(net['pore.coords'][:, 0], dtype=bool)
    net['throat.all'] = np.ones_like(net['throat.conns'][:, 0], dtype=bool)
    net['pore.region_label'] = np.array(p_label)
    net['pore.region_volume'] = np.copy(p_volume)*(voxel_size**3)
    net['pore.phase'] = np.array(p_phase, dtype=int)
    net['throat.phases'] = net['pore.phase'][net['throat.conns']]
    # Extract the geometric stuff
    net['pore.local_peak'] = np.copy(p_coords_dt)*voxel_size
    net['pore.global_peak'] = np.copy(p_coords_dt_global)*voxel_size
    net['pore.geometric_centroid'] = np.copy(p_coords_cm)*voxel_size
    net['throat.global_peak'] = np.array(t_coords)*voxel_size
    net['pore.inscribed_diameter'] = np.copy(p_dia_local)*voxel_size
    net['pore.extended_diameter'] = np.copy(p_dia_global)*voxel_size
    net['throat.inscribed_diameter'] = np.array(t_dia_inscribed)*voxel_size
    P12 = net['throat.conns']
    PT1 = np.sqrt(np.sum(((p_coords[P12[:, 0]]-t_coords)*voxel_size)**2,
                         axis=1))
    PT2 = np.sqrt(np.sum(((p_coords[P12[:, 1]]-t_coords)*voxel_size)**2,
                         axis=1))
    net['throat.total_length'] = PT1 + PT2
    PT1 = PT1-p_dia_local[P12[:, 0]]/2*voxel_size
    PT2 = PT2-p_dia_local[P12[:, 1]]/2*voxel_size
    dist = (p_coords[P12[:, 0]]-p_coords[P12[:, 1]])*voxel_size
    net['throat.direct_length'] = np.sqrt(np.sum(dist**2, axis=1))
    if (accuracy == 'high') and (im.ndim == 3):
        areas = region_surface_areas(regions=im)
        net['pore.surface_area'] = areas * voxel_size**2
        interface_area = region_interface_areas(regions=im, areas=areas,
                                                voxel_size=voxel_size)
        net['throat.cross_sectional_area'] = interface_area.area * voxel_size**2
        # This will be implemented in a different PR
        net['pore.volume'] = np.copy(p_volume)*(voxel_size**3)
        # This will be implemented in a different PR
        net['throat.perimeter'] = np.array(t_perimeter)*voxel_size
    else:
        net['pore.surface_area'] = np.copy(p_area_surf)*(voxel_size)**2
        net['throat.cross_sectional_area'] = np.array(t_area)
        net['pore.volume'] = np.copy(p_volume)*(voxel_size**3)
        net['throat.perimeter'] = np.array(t_perimeter)*voxel_size

    return net
