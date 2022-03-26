import numpy as np
import openpnm as op
import scipy.ndimage as spim
from skimage.segmentation import find_boundaries
from skimage.morphology import ball, cube, disk, square
from porespy.tools import make_contiguous
from porespy.tools import overlay, extend_slice
from porespy.tools import insert_cylinder
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
    regions : ndarray
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

    Examples
    --------
    `Click here
    <https://porespy.org/examples/networks/reference/map_to_regions.html>`_
    to view online example.

    """
    values = np.array(values).flatten()
    if np.size(values) != regions.max():
        raise Exception('Number of values does not match number of regions')
    im = np.zeros_like(regions)
    im = values[regions-1]
    im = im*(regions > 0)
    return im


def add_boundary_regions(regions, pad_width=3):
    r"""
    Add boundary regions on specified faces of an image

    Parameters
    ----------
    regions : ndarray
        An image containing labelled regions, such as a watershed segmentation
    pad_width : array_like
        Number of layers to add to the beginning and end of each axis. This
        argument is handled similarly to the ``pad_width`` in the ``np.pad``
        function. A scalar adds the same amount to the beginning and end of
        each axis. [A, B] adds A to the beginning of each axis and B to the
        ends.  [[A, B], ..., [C, D]] adds A to the beginning and B to the
        end of the first axis, and so on. The default is to add 3 voxels on
        both ends of each axis.  One exception is is [A, B, C] which A to
        the beginning and end of the first axis, and so on. This extra option
        is useful for putting 0 on some axes (i.e. [3, 0, 0]).

    Returns
    -------
    padded_regions : ndarray
        An image with new regions padded on each side of the specified
        width.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/networks/reference/add_boundary_regions.html>`_
    to view online example.

    """
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
    Generates a 3d numpy array from an OpenPNM network

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
    im : ndarray
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
        delta = op.topotools.get_spacing(network).mean() / 2

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
    Generate a voxel image from an OpenPNM network object

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

    Examples
    --------
    `Click here
    <https://porespy.org/examples/networks/reference/generator_voxel_image.html>`_
    to view online example.

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


def label_phases(
        network,
        alias={1: 'void', 2: 'solid'}):
    r"""
    Create pore and throat labels based on 'pore.phase' values

    Parameters
    ----------
    network : dict
        The network stored as a dictionary as returned from the
        ``regions_to_network`` function
    alias : dict
        A mapping between integer values in 'pore.phase' and string labels.
        The default is ``{1: 'void', 2: 'solid'}`` which will result in the
        labels ``'pore.void'`` and ``'pore.solid'``, as well as
        ``'throat.solid_void'``, ``'throat.solid_solid'``, and
        ``'throat.void_void'``.  The reverse labels are also added for
        convenience like ``throat.void_solid``.

    Returns
    -------
    network : dict
        The same ``network`` as passed in but with new boolean arrays added
        for the phase labels.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/networks/reference/label_phases.html>`_
    to view online example.

    """
    conns = network['throat.conns']
    for i in alias.keys():
        pore_i_hits = network['pore.phase'] == i
        network['pore.' + alias[i]] = pore_i_hits
        for j in alias.keys():
            pore_j_hits = network['pore.phase'] == j
            throat_hits = pore_i_hits[conns[:, 0]] * pore_j_hits[conns[:, 1]]
            throat_hits += pore_i_hits[conns[:, 1]] * pore_j_hits[conns[:, 0]]
            if np.any(throat_hits):
                name = 'throat.' + '_'.join([alias[i], alias[j]])
                if name not in network.keys():
                    network[name] = np.zeros_like(conns[:, 0], dtype=bool)
                network[name] += throat_hits
    return network


def label_boundaries(
        network,
        labels=[['left', 'right'], ['front', 'back'], ['top', 'bottom']],
        tol=1e-9):
    r"""
    Create boundary pore labels based on proximity to axis extrema

    Parameters
    ----------
    network : dict
        The network stored as a dictionary as returned from the
        ``regions_to_network`` function
    labels : list of lists
        A 3-element list, with each element containing a pair of strings
        indicating the label to apply to the beginning and end of each axis.
        The default is ``[['left', 'right'], ['front', 'back'],
        ['top', 'bottom']]`` which will apply the label ``'left'`` to all
        pores with the minimum x-coordinate, and ``'right'`` to the pores
        with the maximum x-coordinate, and so on.

    Returns
    -------
    network : dict
        The same ``network`` as passed in but with new boolean arrays added
        for the boundary labels.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/networks/reference/label_boundaries.html>`_
    to view online example.

    """
    crds = network['pore.coords']
    extents = [[crds[:, i].min(), crds[:, i].max()]
               for i in range(len(crds[0, :]))]
    network['pore.boundary'] = np.zeros_like(crds[:, 0], dtype=bool)
    for i, axis in enumerate(labels):
        for j, face in enumerate(axis):
            if face:
                hits = crds[:, i] == extents[i][j]
                network['pore.boundary'] += hits
                network['pore.' + labels[i][j]] = hits
    return network
