import numpy as np
import openpnm as op
import scipy.ndimage as spim
from skimage.segmentation import find_boundaries
from skimage.morphology import ball, cube, disk, square
from skimage.segmentation import relabel_sequential
from porespy.tools import make_contiguous
from porespy.tools import _create_alias_map, overlay
from porespy.tools import insert_cylinder
from porespy.tools import zero_corners
from porespy.generators import borders
from porespy import settings
from porespy.tools import get_tqdm
from loguru import logger
tqdm = get_tqdm()


def map_to_regions(regions, values):
    r"""
    Maps pore values from a network onto the image from which it was extracted

    Parameters
    ----------
    regions : ND-array
        An image of the pore space partitioned into regions and labeled

    values : array_like
        An array containing the numerical values to insert into each region.
        The value at location *n* will be inserted into the image where
        ``regions`` is *n+1*.  This mismatch is caused by the fact that 0's
        in the ``regions`` image is assumed to be the backgroung phase, while
        pore index 0 is valid.

    Notes
    -----
    This function assumes that the array of pore values are indexed starting
    at location 0, while in the region image 0's indicate background phase and
    the region indexing starts at 1.  That is, region 1 corresponds to pore 0.

    """
    values = np.array(values).flatten()
    if np.size(values) != regions.max() + 1:
        raise Exception('Number of values does not match number of regions')
    im = np.zeros_like(regions)
    im = values[regions]
    return im


def add_boundary_regions(regions=None, faces=['front', 'back', 'left',
                                              'right', 'top', 'bottom']):
    r"""
    Given an image partitioned into regions, pads specified faces with new
    regions

    Parameters
    ----------
    regions : ND-array
        An image of the pore space partitioned into regions and labeled
    faces : list of strings
        The faces of ``regions`` which should have boundaries added.  Options
        are:

        *'right'* - Adds boundaries to the x=0 face (``im[0, :, :]``)

        *'left'* - Adds boundaries to the x=X face (``im[-1, :, :]``)

        *'front'* - Adds boundaries to the y=0 face (``im[:, ), :]``)

        *'back'* - Adds boundaries to the x=0 face (``im[:, -1, :]``)

        *'bottom'* - Adds boundaries to the x=0 face (``im[:, :, 0]``)

        *'top'* - Adds boundaries to the x=0 face (``im[:, :, -1]``)

        The default is all faces.

    Returns
    -------
    image : ND-array
        A copy of ``regions`` with the specified boundaries added, so will be
        slightly larger in each direction where boundaries were added.

    """
    if faces is None:
        return regions

    if regions.ndim not in [2, 3]:
        raise Exception("add_boundary_regions works only on 2D and 3D images")

    if regions.ndim == 2:
        if set(["bottom", "top"]).intersection(faces):
            raise Exception("For 2D images, use 'left', 'right', 'front', 'back' labels")

    # Map which image slice corresponds to which face
    face_to_slice = {
        "left": 0, "right": -1, "front": 0, "back": -1, "bottom": 0, "top": -1
    }
    # Map face label to corresponding axis
    face_to_axis = {
        "left": 0, "right": 0, "front": 1, "back": 1, "bottom": 2, "top": 2
    }

    ndim = regions.ndim
    indices = []
    # Note: pad_width format: each elem is [pad_before, pad_after]
    pad_width = [[0, 0] for i in range(ndim)]
    # 1. Create indices for boundary faces, ex. bottom" => [:, :, -1]
    # Note: slice(None) is equivalent to ":" in fancy indexing, ex. [0, 0, :]
    # 2. populate pad_width based on labels
    for face in faces:
        temp = [slice(None) for i in range(ndim)]
        axis = face_to_axis[face]
        plane = face_to_slice[face]
        temp[axis] = plane
        indices.append(tuple(temp))
        pad_width[axis][plane] = 1      # Pad each face by 1 pixel

    regions = np.pad(regions, pad_width=pad_width, mode="edge")

    # Increment boundary regions to distinguish from internal regions
    for idx in indices:
        # Only increment non-background regions (i.e. != 0)
        non_background = regions[idx] != 0
        regions[idx][non_background] += regions.max()

    # Remove connections between boundary regions
    for idx in indices:
        regions[idx] *= ~find_boundaries(regions[idx], mode="outer")

    # Pad twice to make boundary regions 3-pixel thick -> required for marching_cube
    pw = np.array(pad_width) * 1
    regions = np.pad(regions, pad_width=pw * 2, mode="edge")

    # Convert pad-induced corners to 0
    zero_corners(regions, pw * 3)

    # Make labels contiguous
    regions = relabel_sequential(regions, offset=1)[0]

    return regions


def add_boundary_regions2(regions, pad_width=3):
    r"""
    Add boundary regions on specified faces of an image

    Parameters
    ----------
    regions : ND-image
        An image containing labelled regions, such as a watershed segmentation
    pad_width : array_like
        Number of layers to add to the beginnign and end of each axis. This argument
        is handled the same as ``pad_width`` in the ``np.pad`` function. An scalar
        adds the same amount to the beginning and end of each axis. [A, B] adds A to
        the beginning of each axis and B to the ends.  [[A, B], ..., [C, D]] adds
        A to the beginning and B to the end of the first axis, and so on.
        The default is to add 3 voxels on each axis.

    Returns
    -------
    padded_regions : ND-array
        An image with new regions padded on each side of the specified
        width.

    """
    if pad_width == 0:
        return regions
    # Parse user specified padding
    faces = np.array(pad_width)
    if faces.size == 1:
        faces = np.array([[faces, faces]]*regions.ndim)
    elif faces.size == regions.ndim:
        faces = np.vstack([faces]*2)
    else:
        pass
    t = faces.max()
    mx = regions.max()
    # Put a border around each region so padded regions are isolated
    bd = find_boundaries(regions, connectivity=regions.ndim, mode='inner')
    # Pad by t in all directions, this will be trimmed down later
    face_regions = np.pad(regions*(~bd), pad_width=t, mode='edge')
    # Set corners to 0 so regions don't connect across faces
    edges = borders(shape=face_regions.shape, mode='edges', thickness=t)
    face_regions[edges] = 0
    # Extract a mask of just the faces
    mask = borders(shape=face_regions.shape, mode='faces', thickness=t)
    # Relabel regions on faces
    new_regions = spim.label(face_regions*mask)[0] + mx*(face_regions > 0)
    new_regions[~mask] = regions.flatten()
    # Trim image down to user specified size
    s = tuple([slice(t-ax[0], -(t-ax[1]) or None) for ax in faces])
    new_regions = new_regions[s]
    new_regions = make_contiguous(new_regions)
    return new_regions


def _generate_voxel_image(network, pore_shape, throat_shape, max_dim=200):
    r"""
    Generates a 3d numpy array from a network model.

    Parameters
    ----------
    network : OpenPNM GenericNetwork
        Network from which voxel image is to be generated
    pore_shape : str
        Shape of pores in the network, valid choices are "sphere", "cube"
    throat_shape : str
        Shape of throats in the network, valid choices are "cylinder", "cuboid"
    max_dim : int
        Number of voxels in the largest dimension of the network

    Returns
    -------
    im : ND-array
        Voxelated image corresponding to the given pore network model

    Notes
    -----
    (1) The generated voxel image is labeled with 0s, 1s and 2s signifying
    solid phase, pores, and throats respectively.

    """
    xyz = network["pore.coords"]
    cn = network["throat.conns"]

    # Distance bounding box from the network by a fixed amount
    delta = network["pore.diameter"].mean() / 2
    if isinstance(network, op.network.Cubic):
        delta = network._spacing.mean() / 2

    # Shift everything to avoid out-of-bounds
    extra_clearance = int(max_dim * 0.05)

    # Transform points to satisfy origin at (0, 0, 0)
    xyz0 = xyz.min(axis=0) - delta
    xyz += -xyz0
    res = (xyz.ptp(axis=0).max() + 2 * delta) / max_dim
    shape = np.rint((xyz.max(axis=0) + delta) / res).astype(int) + 2 * extra_clearance

    # Transforming from real coords to matrix coords
    xyz = np.rint(xyz / res).astype(int) + extra_clearance
    pore_radi = np.rint(network["pore.diameter"] * 0.5 / res).astype(int)
    throat_radi = np.rint(network["throat.diameter"] * 0.5 / res).astype(int)

    im_pores = np.zeros(shape, dtype=np.uint8)
    im_throats = np.zeros_like(im_pores)

    if pore_shape == "cube":
        pore_elem = cube
        rp = pore_radi * 2 + 1  # +1 since num_voxel must be odd
        rp_max = int(2 * round(delta / res)) + 1
    if pore_shape == "sphere":
        pore_elem = ball
        rp = pore_radi
        rp_max = int(round(delta / res))
    if throat_shape == "cuboid":
        raise Exception("Not yet implemented, try 'cylinder'.")

    # Generating voxels for pores
    for i, pore in enumerate(tqdm(network.Ps, **settings.tqdm)):
        elem = pore_elem(rp[i])
        try:
            im_pores = overlay(im1=im_pores, im2=elem, c=xyz[i])
        except ValueError:
            elem = pore_elem(rp_max)
            im_pores = overlay(im1=im_pores, im2=elem, c=xyz[i])
    # Get rid of pore overlaps
    im_pores[im_pores > 0] = 1

    # Generating voxels for throats
    for i, throat in enumerate(tqdm(network.Ts, **settings.tqdm)):
        try:
            im_throats = insert_cylinder(
                im_throats, r=throat_radi[i], xyz0=xyz[cn[i, 0]], xyz1=xyz[cn[i, 1]])
        except ValueError:
            im_throats = insert_cylinder(
                im_throats, r=rp_max, xyz0=xyz[cn[i, 0]], xyz1=xyz[cn[i, 1]])
    # Get rid of throat overlaps
    im_throats[im_throats > 0] = 1

    # Subtract pore-throat overlap from throats
    im_throats = (im_throats.astype(bool) * ~im_pores.astype(bool)).astype(np.uint8)
    im = im_pores * 1 + im_throats * 2

    return im[extra_clearance:-extra_clearance,
              extra_clearance:-extra_clearance,
              extra_clearance:-extra_clearance]


def generate_voxel_image(network, pore_shape="sphere", throat_shape="cylinder",
                         max_dim=None, rtol=0.1):
    r"""
    Generates voxel image from an OpenPNM network object.

    Parameters
    ----------
    network : OpenPNM GenericNetwork
        Network from which voxel image is to be generated
    pore_shape : str
        Shape of pores in the network, valid choices are "sphere", "cube"
    throat_shape : str
        Shape of throats in the network, valid choices are "cylinder", "cuboid"
    max_dim : int
        Number of voxels in the largest dimension of the network
    rtol : float
        Stopping criteria for finding the smallest voxel image such that
        further increasing the number of voxels in each dimension by 25% would
        improve the predicted porosity of the image by less that ``rtol``

    Returns
    -------
    im : ndarray
        Voxelated image corresponding to the given pore network model

    Notes
    -----
    (1) The generated voxelated image is labeled with 0s, 1s and 2s signifying
    solid phase, pores, and throats respectively.

    (2) If max_dim is not provided, the method calculates it such that the
    further increasing it doesn't change porosity by much.

    """
    logger.trace("Generating voxel image from pore network")
    # If max_dim is provided, generate voxel image using max_dim
    if max_dim is not None:
        return _generate_voxel_image(
            network, pore_shape, throat_shape, max_dim=max_dim)
    max_dim = 200
    # If max_dim is not provided, find best max_dim that predicts porosity
    err = 100
    eps_old = 200
    while err > rtol:
        logger.debug(f"Maximum dimension: {max_dim} voxels")
        im = _generate_voxel_image(
            network, pore_shape, throat_shape, max_dim=max_dim)
        eps = im.astype(bool).sum() / np.prod(im.shape)
        err = abs(1 - eps / eps_old)
        eps_old = eps
        max_dim = int(max_dim * 1.25)
    logger.debug(f"Converged at max_dim = {max_dim} voxels")
    return im


def add_phase_interconnections(net, snow_partitioning_n, voxel_size=1,
                               marching_cubes_area=False,
                               alias=None):
    r"""
    Connects networks of two or more phases together by interconnecting
    neighbouring nodes of different phases.

    The resulting network can be used for the study of transport and kinetics
    at interphase of two phases.

    Parameters
    ----------
    network : 2D or 3D network
        A dictoionary containing structural information of two or more
        phases networks. The dictonary format must be same as porespy
        region_to_network function.
    snow_partitioning_n : tuple
        The output generated by snow_partitioning_n function. The tuple should
        have phases_max_labels and original image of material.
    voxel_size : scalar
        The resolution of the image, expressed as the length of one side of a
        voxel, so the volume of a voxel would be **voxel_size**-cubed.  The
        default is 1, which is useful when overlaying the PNM on the original
        image since the scale of the image is alway 1 unit lenth per voxel.
    marching_cubes_area : bool
        If ``True`` then the surface area and interfacial area between regions
        will be causing the marching cube algorithm. This is a more accurate
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
    A dictionary containing network information of individual and connected
    networks. The dictionary names use the OpenPNM convention so it may be
    converted directly to an OpenPNM network object using the ``update``
    command.

    """
    # Get alias if provided by user
    im = snow_partitioning_n.im
    al = _create_alias_map(im, alias=alias)
    # Find interconnection and interfacial area between ith and jth phases
    conns1 = net['throat.conns'][:, 0]
    conns2 = net['throat.conns'][:, 1]
    label = net['pore.label'] - 1

    num = snow_partitioning_n.phase_max_label
    num = [0, *num]
    phases_num = np.unique(im * 1)
    phases_num = np.trim_zeros(phases_num)
    for i0, i1 in enumerate(phases_num):
        loc1 = np.logical_and(conns1 >= num[i0], conns1 < num[i0 + 1])
        loc2 = np.logical_and(conns2 >= num[i0], conns2 < num[i0 + 1])
        loc3 = np.logical_and(label >= num[i0], label < num[i0 + 1])
        net['throat.{}'.format(al[i1])] = loc1 * loc2
        net['pore.{}'.format(al[i1])] = loc3
        if i1 == phases_num[-1]:
            loc4 = np.logical_and(conns1 < num[-1], conns2 >= num[-1])
            loc5 = label >= num[-1]
            net['throat.boundary'] = loc4
            net['pore.boundary'] = loc5
        for j0, j1 in enumerate(phases_num):
            if j0 > i0:
                pi_pj_sa = np.zeros_like(label, dtype=float)
                loc6 = np.logical_and(conns2 >= num[j0], conns2 < num[j0 + 1])
                pi_pj_conns = loc1 * loc6
                net['throat.{}_{}'.format(al[i1], al[j1])] = pi_pj_conns
                if any(pi_pj_conns):
                    # Calculates phase[i] interfacial area that connects with
                    # phase[j] and vice versa
                    p_conns = net['throat.conns'][:, 0][pi_pj_conns]
                    s_conns = net['throat.conns'][:, 1][pi_pj_conns]
                    ps = net['throat.area'][pi_pj_conns]
                    p_sa = np.bincount(p_conns, ps)
                    # trim zeros at head/tail position to avoid extra bins
                    p_sa = np.trim_zeros(p_sa)
                    i_index = np.arange(min(p_conns), max(p_conns) + 1)
                    j_index = np.arange(min(s_conns), max(s_conns) + 1)
                    s_pa = np.bincount(s_conns, ps)
                    s_pa = np.trim_zeros(s_pa)
                    pi_pj_sa[i_index] = p_sa
                    pi_pj_sa[j_index] = s_pa
                    # Calculates interfacial area using marching cube method
                    if marching_cubes_area:
                        ps_c = net['throat.area'][pi_pj_conns]
                        p_sa_c = np.bincount(p_conns, ps_c)
                        p_sa_c = np.trim_zeros(p_sa_c)
                        s_pa_c = np.bincount(s_conns, ps_c)
                        s_pa_c = np.trim_zeros(s_pa_c)
                        pi_pj_sa[i_index] = p_sa_c
                        pi_pj_sa[j_index] = s_pa_c
                    net[f'pore.{al[i1]}_{al[j1]}_area'] = pi_pj_sa * voxel_size ** 2
    return net


def label_boundary_cells(network=None, boundary_faces=None):
    r"""
    Takes 2D or 3D network and assign labels to boundary pores

    Parameters
    ----------
    network : dictionary
        A dictionary as produced by the SNOW network extraction algorithms
        containing edge/vertex, site/bond, node/link information.

    boundary_faces : list of strings
        The user can choose ‘left’, ‘right’, ‘top’, ‘bottom’, ‘front’ and
        ‘back’ face labels to assign boundary nodes. If no label is
        assigned then all six faces will be selected as boundary nodes
        automatically which can be trimmed later on based on user requirements.

    Returns
    -------
    The same dictionar s pass ing, but containing boundary nodes labels.  For
    example network['pore.left'], network['pore.right'], network['pore.top'],
    network['pore.bottom'] etc.

    Notes
    -----
    The dictionary names use the OpenPNM convention so it may be converted
    directly to an OpenPNM network object using the ``update`` command.

    """
    f = boundary_faces
    if f is not None:
        coords = network['pore.coords']
        condition = coords[~network['pore.boundary']]
        dic = {'left': 0, 'right': 0, 'front': 1, 'back': 1,
               'top': 2, 'bottom': 2}
        if all(coords[:, 2] == 0):
            dic['top'] = 1
            dic['bottom'] = 1
        for i in f:
            if i in ['left', 'front', 'bottom']:
                network['pore.{}'.format(i)] = (
                    coords[:, dic[i]] < min(condition[:, dic[i]])
                )
            elif i in ['right', 'back', 'top']:
                network['pore.{}'.format(i)] = (
                    coords[:, dic[i]] > max(condition[:, dic[i]])
                )

    return network


def label_phases(
        network,
        alias={1: 'void', 2: 'solid'}):
    r"""
    """
    conns = network['throat.conns']
    for i in alias.keys():
        pore_i_hits = network['pore.phase'] == i
        network['pore.' + alias[i]] = pore_i_hits
        for j in alias.keys():
            name = 'throat.' + '_'.join(sorted([alias[i], alias[j]]))
            pore_j_hits = network['pore.phase'] == j
            throat_hits = pore_i_hits[conns[:, 0]] * pore_j_hits[conns[:, 1]]
            throat_hits += pore_i_hits[conns[:, 1]] * pore_j_hits[conns[:, 0]]
            if np.any(throat_hits):
                if name not in network.keys():
                    network[name] = np.zeros_like(conns[:, 0], dtype=bool)
                network[name] += throat_hits
    return network


def label_boundaries(
        network,
        labels=[['left', 'right'], ['front', 'back'], ['top', 'bottom']],
        tol=1e-9):
    r"""
    """
    crds = network['pore.coords']
    extents = [[crds[:, i].min(), crds[:, i].max()] for i in range(len(crds[0, :]))]
    network['pore.boundary'] = np.zeros_like(crds[:, 0], dtype=bool)
    for i, axis in enumerate(labels):
        for j, face in enumerate(axis):
            try:
                hits = crds[:, i] == extents[i][j]
                network['pore.boundary'] += hits
                network['pore.' + labels[i][j]] = hits
            except TypeError:
                continue
    return network
