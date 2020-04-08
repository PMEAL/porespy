import sys
import numpy as np
import openpnm as op
from tqdm import tqdm
import scipy.ndimage as spim
from porespy.tools import extend_slice
import openpnm.models.geometry as op_gm


def regions_to_network(im, dt=None, voxel_size=1):
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

    Returns
    -------
    A dictionary containing all the pore and throat size data, as well as the
    network topological information.  The dictionary names use the OpenPNM
    convention (i.e. 'pore.coords', 'throat.conns') so it may be converted
    directly to an OpenPNM network object using the ``update`` command.

    """
    print('-'*60)
    print('Extracting pore and throat information from image')
    from skimage.morphology import disk, ball
    struc_elem = disk if im.ndim == 2 else ball

    # if ~np.any(im == 0):
    #     raise Exception('The received image has no solid phase (0\'s)')

    if dt is None:
        dt = spim.distance_transform_edt(im > 0)
        dt = spim.gaussian_filter(input=dt, sigma=0.5)

    # Get 'slices' into im for each pore region
    slices = spim.find_objects(im)

    # Initialize arrays
    Ps = np.arange(1, np.amax(im)+1)
    Np = np.size(Ps)
    p_coords = np.zeros((Np, im.ndim), dtype=float)
    p_volume = np.zeros((Np, ), dtype=float)
    p_dia_local = np.zeros((Np, ), dtype=float)
    p_dia_global = np.zeros((Np, ), dtype=float)
    p_label = np.zeros((Np, ), dtype=int)
    p_area_surf = np.zeros((Np, ), dtype=int)
    t_conns = []
    t_dia_inscribed = []
    t_area = []
    t_perimeter = []
    t_coords = []
    # dt_shape = np.array(dt.shape)

    # Start extracting size information for pores and throats
    for i in tqdm(Ps, file=sys.stdout):
        pore = i - 1
        if slices[pore] is None:
            continue
        s = extend_slice(slices[pore], im.shape)
        sub_im = im[s]
        sub_dt = dt[s]
        pore_im = sub_im == i
        padded_mask = np.pad(pore_im, pad_width=1, mode='constant')
        pore_dt = spim.distance_transform_edt(padded_mask)
        s_offset = np.array([i.start for i in s])
        p_label[pore] = i
        p_coords[pore, :] = spim.center_of_mass(pore_im) + s_offset
        p_volume[pore] = np.sum(pore_im)
        p_dia_local[pore] = (2*np.amax(pore_dt)) - np.sqrt(3)
        p_dia_global[pore] = 2*np.amax(sub_dt)
        p_area_surf[pore] = np.sum(pore_dt == 1)
        im_w_throats = spim.binary_dilation(input=pore_im, structure=struc_elem(1))
        im_w_throats = im_w_throats*sub_im
        Pn = np.unique(im_w_throats)[1:] - 1
        for j in Pn:
            if j > pore:
                t_conns.append([pore, j])
                vx = np.where(im_w_throats == (j + 1))
                t_dia_inscribed.append(2*np.amax(sub_dt[vx]))
                t_perimeter.append(np.sum(sub_dt[vx] < 2))
                t_area.append(np.size(vx[0]))
                t_inds = tuple([i+j for i, j in zip(vx, s_offset)])
                temp = np.where(dt[t_inds] == np.amax(dt[t_inds]))[0][0]
                if im.ndim == 2:
                    t_coords.append(tuple((t_inds[0][temp],
                                           t_inds[1][temp])))
                else:
                    t_coords.append(tuple((t_inds[0][temp],
                                           t_inds[1][temp],
                                           t_inds[2][temp])))
    # Clean up values
    Nt = len(t_dia_inscribed)  # Get number of throats
    if im.ndim == 2:  # If 2D, add 0's in 3rd dimension
        p_coords = np.vstack((p_coords.T, np.zeros((Np, )))).T
        t_coords = np.vstack((np.array(t_coords).T, np.zeros((Nt, )))).T

    net = {}
    net['pore.all'] = np.ones((Np, ), dtype=bool)
    net['throat.all'] = np.ones((Nt, ), dtype=bool)
    net['pore.coords'] = np.copy(p_coords)*voxel_size
    net['pore.centroid'] = np.copy(p_coords)*voxel_size
    net['throat.centroid'] = np.array(t_coords)*voxel_size
    net['throat.conns'] = np.array(t_conns)
    net['pore.label'] = np.array(p_label)
    net['pore.volume'] = np.copy(p_volume)*(voxel_size**3)
    net['throat.volume'] = np.zeros((Nt, ), dtype=float)
    net['pore.diameter'] = np.copy(p_dia_local)*voxel_size
    net['pore.inscribed_diameter'] = np.copy(p_dia_local)*voxel_size
    net['pore.equivalent_diameter'] = 2*((3/4*net['pore.volume']/np.pi)**(1/3))
    net['pore.extended_diameter'] = np.copy(p_dia_global)*voxel_size
    net['pore.surface_area'] = np.copy(p_area_surf)*(voxel_size)**2
    net['throat.diameter'] = np.array(t_dia_inscribed)*voxel_size
    net['throat.inscribed_diameter'] = np.array(t_dia_inscribed)*voxel_size
    net['throat.area'] = np.array(t_area)*(voxel_size**2)
    net['throat.perimeter'] = np.array(t_perimeter)*voxel_size
    net['throat.equivalent_diameter'] = (np.array(t_area) * (voxel_size**2))**0.5
    P12 = net['throat.conns']
    PT1 = np.sqrt(np.sum(((p_coords[P12[:, 0]]-t_coords) * voxel_size)**2, axis=1))
    PT2 = np.sqrt(np.sum(((p_coords[P12[:, 1]]-t_coords) * voxel_size)**2, axis=1))
    net['throat.total_length'] = PT1 + PT2
    PT1 = PT1-p_dia_local[P12[:, 0]]/2*voxel_size
    PT2 = PT2-p_dia_local[P12[:, 1]]/2*voxel_size
    net['throat.length'] = PT1 + PT2
    dist = (p_coords[P12[:, 0]]-p_coords[P12[:, 1]])*voxel_size
    net['throat.direct_length'] = np.sqrt(np.sum(dist**2, axis=1))
    # Make a dummy openpnm network to get the conduit lengths
    pn = op.network.GenericNetwork()
    pn.update(net)
    pn.add_model(propname='throat.endpoints',
                 model=op_gm.throat_endpoints.spherical_pores,
                 pore_diameter='pore.inscribed_diameter',
                 throat_diameter='throat.inscribed_diameter')
    pn.add_model(propname='throat.conduit_lengths',
                 model=op_gm.throat_length.conduit_lengths)
    pn.add_model(propname='pore.area',
                 model=op_gm.pore_area.sphere)
    net['throat.endpoints.head'] = pn['throat.endpoints.head']
    net['throat.endpoints.tail'] = pn['throat.endpoints.tail']
    net['throat.conduit_lengths.pore1'] = pn['throat.conduit_lengths.pore1']
    net['throat.conduit_lengths.pore2'] = pn['throat.conduit_lengths.pore2']
    net['throat.conduit_lengths.throat'] = pn['throat.conduit_lengths.throat']
    net['pore.area'] = pn['pore.area']
    prj = pn.project
    prj.clear()
    wrk = op.Workspace()
    wrk.close_project(prj)

    return net
