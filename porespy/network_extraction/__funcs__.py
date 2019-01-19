import scipy as sp
import numpy as np
import openpnm as op
from porespy.tools import make_contiguous
from skimage.segmentation import find_boundaries
from skimage.morphology import ball, cube, disk
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm


def map_to_regions(regions, values):
    r"""
    Maps pore values from a network onto the image from which it was extracted

    This function assumes that the pore numbering in the network has remained
    unchanged from the region labels in the partitioned image.

    Parameters
    ----------
    regions : ND-array
        An image of the pore space partitioned into regions and labeled

    values : array_like
        An array containing the numerical values to insert into each region.
        The value at location *n* will be inserted into the image where
        ``regions`` is *n+1*.  This mis-match is caused by the fact that 0's
        in the ``regions`` image is assumed to be the backgroung phase, while
        pore index 0 is valid.

    Notes
    -----
    This function assumes that the array of pore values are indexed starting
    at location 0, while in the region image 0's indicate background phase and
    the region indexing starts at 1.  That is, region 1 corresponds to pore 0.

    """
    values = sp.array(values).flatten()
    if sp.size(values) != regions.max() + 1:
        raise Exception('Number of values does not match number of regions')
    im = sp.zeros_like(regions)
    im = values[regions]
    return im


def add_boundary_regions(regions=None, faces=['front', 'back', 'left',
                                              'right', 'top', 'bottom']):
    # -------------------------------------------------------------------------
    # Edge pad segmentation and distance transform
    if faces is not None:
        regions = sp.pad(regions, 1, 'edge')
        # ---------------------------------------------------------------------
        if regions.ndim == 3:
            # Remove boundary nodes interconnection
            regions[:, :, 0] = regions[:, :, 0] + regions.max()
            regions[:, :, -1] = regions[:, :, -1] + regions.max()
            regions[0, :, :] = regions[0, :, :] + regions.max()
            regions[-1, :, :] = regions[-1, :, :] + regions.max()
            regions[:, 0, :] = regions[:, 0, :] + regions.max()
            regions[:, -1, :] = regions[:, -1, :] + regions.max()
            regions[:, :, 0] = (~find_boundaries(regions[:, :, 0],
                                                 mode='outer'))*regions[:, :, 0]
            regions[:, :, -1] = (~find_boundaries(regions[:, :, -1],
                                                  mode='outer'))*regions[:, :, -1]
            regions[0, :, :] = (~find_boundaries(regions[0, :, :],
                                                 mode='outer'))*regions[0, :, :]
            regions[-1, :, :] = (~find_boundaries(regions[-1, :, :],
                                                  mode='outer'))*regions[-1, :, :]
            regions[:, 0, :] = (~find_boundaries(regions[:, 0, :],
                                                 mode='outer'))*regions[:, 0, :]
            regions[:, -1, :] = (~find_boundaries(regions[:, -1, :],
                                                  mode='outer'))*regions[:, -1, :]
            # -----------------------------------------------------------------
            regions = sp.pad(regions, 2, 'edge')

            # Remove unselected faces
            if 'front' not in faces:
                regions = regions[:, 3:, :]  # y
            if 'back' not in faces:
                regions = regions[:, :-3, :]
            if 'left' not in faces:
                regions = regions[3:, :, :]  # x
            if 'right' not in faces:
                regions = regions[:-3, :, :]
            if 'bottom' not in faces:
                regions = regions[:, :, 3:]  # z
            if 'top' not in faces:
                regions = regions[:, :, :-3]

        elif regions.ndim == 2:
            # Remove boundary nodes interconnection
            regions[0, :] = regions[0, :] + regions.max()
            regions[-1, :] = regions[-1, :] + regions.max()
            regions[:, 0] = regions[:, 0] + regions.max()
            regions[:, -1] = regions[:, -1] + regions.max()
            regions[0, :] = (~find_boundaries(regions[0, :],
                                              mode='outer'))*regions[0, :]
            regions[-1, :] = (~find_boundaries(regions[-1, :],
                                               mode='outer'))*regions[-1, :]
            regions[:, 0] = (~find_boundaries(regions[:, 0],
                                              mode='outer'))*regions[:, 0]
            regions[:, -1] = (~find_boundaries(regions[:, -1],
                                               mode='outer'))*regions[:, -1]
            # -----------------------------------------------------------------
            regions = sp.pad(regions, 2, 'edge')

            # Remove unselected faces
            if 'left' not in faces:
                regions = regions[3:, :]  # x
            if 'right' not in faces:
                regions = regions[:-3, :]
            if 'front' not in faces and 'bottom' not in faces:
                regions = regions[:, 3:]  # y
            if 'back' not in faces and 'top' not in faces:
                regions = regions[:, :-3]
        else:
            print('add_boundary_regions works only on 2D and 3D images')
        # ---------------------------------------------------------------------
        # Make labels contiguous
        regions = make_contiguous(regions)
    else:
        regions = regions

    return regions


def overlay(im1, im2, c):
    r"""
    Overlays im2 onto im1, given voxel coords of center of im2 in im1.

    Parameters
    ----------
    im1 : 3D numpy array
        Original voxelated image

    im2 : 3D numpy array
        Template voxelated image

    r : int
        Radius of the cylinder

    Returns
    -------
    im1 : 3D numpy array
        Original voxelated image overlayed with the template

    """
    shape = im2.shape

    for ni in shape:
        if ni % 2 == 0:
            raise Exception("Structuring element must be odd-voxeled...")

    nx, ny, nz = [(ni - 1) // 2 for ni in shape]
    cx, cy, cz = c

    im1[cx-nx:cx+nx+1, cy-ny:cy+ny+1, cz-nz:cz+nz+1] += im2

    return im1


def add_cylinder_to(im, xyz0, xyz1, r):
    r"""
    Overlays a cylinder of given radius onto a given 3d image.

    Parameters
    ----------
    im : 3D numpy array
        Original voxelated image

    xyz0, xyz1 : 3 by 1 numpy array-like
        Voxel coordinates of the two end points of the cylinder

    r : int
        Radius of the cylinder

    Returns
    -------
    im : 3D numpy array
        Original voxelated image overlayed with the cylinder

    """
    # Converting coordinates to numpy array
    xyz0, xyz1 = [np.array(xyz).astype(int) for xyz in (xyz0, xyz1)]
    r = int(r)
    L = np.abs(xyz0 - xyz1).max() + 1
    xyz_line = [np.linspace(xyz0[i], xyz1[i], L).astype(int) for i in range(3)]

    xyz_min = np.min(xyz_line, axis=1) - r
    xyz_max = np.max(xyz_line, axis=1) + r
    shape_template = xyz_max - xyz_min + 1
    template = np.zeros(shape=shape_template)

    # Shortcut for orthogonal cylinders
    if (xyz0 == xyz1).sum() == 2:
        unique_dim = [xyz0[i] != xyz1[i] for i in range(3)].index(True)
        shape_template[unique_dim] = 1
        template_2D = disk(radius=r).reshape(shape_template)
        template = np.repeat(template_2D, repeats=L, axis=unique_dim)
        xyz_min[unique_dim] += r
        xyz_max[unique_dim] += -r
    else:
        xyz_line_in_template_coords = [xyz_line[i] - xyz_min[i] for i in range(3)]
        template[tuple(xyz_line_in_template_coords)] = 1
        template = distance_transform_edt(template == 0) <= r

    im[xyz_min[0]:xyz_max[0]+1,
       xyz_min[1]:xyz_max[1]+1,
       xyz_min[2]:xyz_max[2]+1] += template

    return im


def _generate_voxel_image(network, pore_shape, throat_shape, max_dim=200,
                          verbose=1):
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
    im : 3D numpy array
        Voxelated image corresponding to the given pore network model

    Notes
    -----
    (1) The generated voxelated image is labeled with 0s, 1s and 2s signifying
    solid phase, pores, and throats respectively.

    """
    xyz = network["pore.coords"]
    cn = network["throat.conns"]

    # Distance bounding box from the network by a fixed amount
    delta = network["pore.diameter"].mean() / 2
    if isinstance(network, op.network.Cubic):
        delta = network._spacing.mean() / 2

    # Shift everything to avoid out-of-bounds
    delta_bounds = int(max_dim * 0.1)

    # Transform points to satisfy origin at (0, 0, 0)
    xyz0 = xyz.min(axis=0) - delta
    xyz += -xyz0
    res = xyz.max(axis=0).max() / max_dim
    shape = np.rint((xyz.max(axis=0) + delta) / res).astype(int) + delta_bounds

    # Transforming from real coords to matrix coords
    xyz = np.rint(xyz / res).astype(int) + delta_bounds // 2
    pore_radi = np.rint(network["pore.diameter"] * 0.5 / res).astype(int)
    throat_radi = np.rint(network["throat.diameter"] * 0.5 / res).astype(int)

    im_pores = np.zeros(shape, dtype=np.uint8)
    im_throats = np.zeros_like(im_pores)

    if pore_shape is "cube":
        pore_elem = cube
        rp = pore_radi * 2 + 1  # +1 since num_voxel must be odd
        rp_max = int(2 * round(delta / res)) + 1
    if pore_shape is "sphere":
        pore_elem = ball
        rp = pore_radi
        rp_max = int(round(delta / res))
    if throat_shape is "cuboid":
        raise Exception("Not yet implemented, try 'cylinder'.")

    # Generating voxels for pores
    for i, pore in enumerate(tqdm(network.pores(), disable=not verbose,
                                  desc="Generating pores  ")):
        elem = pore_elem(rp[i])
        try:
            im_pores = overlay(im1=im_pores, im2=elem, c=xyz[i])
        except ValueError:
            elem = pore_elem(rp_max)
            im_pores = overlay(im1=im_pores, im2=elem, c=xyz[i])
    # Get rid of pore overlaps
    im_pores[im_pores > 0] = 1

    # Generating voxels for throats
    for i, throat in enumerate(tqdm(network.throats(), disable=not verbose,
                                    desc="Generating throats")):
        try:
            im_throats = add_cylinder_to(im_throats, r=throat_radi[i],
                                         xyz0=xyz[cn[i, 0]], xyz1=xyz[cn[i, 1]])
        except ValueError:
            im_throats = add_cylinder_to(im_throats, r=rp_max,
                                         xyz0=xyz[cn[i, 0]], xyz1=xyz[cn[i, 1]])
    # Get rid of throat overlaps
    im_throats[im_throats > 0] = 1

    # Subtract pore-throat overlap from throats
    im_throats = (im_throats.astype(bool) * ~im_pores.astype(bool)).astype(sp.uint8)
    im = im_pores * 1 + im_throats * 2

    return im[delta_bounds//2:-delta_bounds//2,
              delta_bounds//2:-delta_bounds//2,
              delta_bounds//2:-delta_bounds//2]

    return im


def generate_voxel_image(network, pore_shape="sphere", throat_shape="cylinder",
                         max_dim=None, verbose=1, rtol=0.1):
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

    tol_percent : int
        Relative percent change in porosity if increasing the size of the voxel
        image by 25%. See Notes.

    Returns
    -------
    im : 3D numpy array
        Voxelated image corresponding to the given pore network model

    Notes
    -----
    (1) The generated voxelated image is labeled with 0s, 1s and 2s signifying
    solid phase, pores, and throats respectively.

    (2) If max_dim is not provided, the method calculates it such that the
    further increasing it doesn't change porosity by much.

    """
    print("\n" + "-" * 44, flush=True)
    print("| Generating voxel image from pore network |", flush=True)
    print("-" * 44, flush=True)

    # If max_dim is provided, generate voxel image using max_dim
    if max_dim is not None:
        return _generate_voxel_image(network, pore_shape, throat_shape,
                                     max_dim=max_dim, verbose=verbose)
    else:
        max_dim = 200

    # If max_dim is not provided, find best max_dim that predicts porosity
    eps_old = 200
    err = 100  # percent

    while err > rtol:
        im = _generate_voxel_image(network, pore_shape, throat_shape,
                                   max_dim=max_dim, verbose=verbose)
        eps = im.astype(bool).sum() / sp.prod(im.shape)

        err = abs(1 - eps/eps_old)
        eps_old = eps
        max_dim = int(max_dim * 1.25)

    if verbose:
        print(f"\nConverged at max_dim = {max_dim} voxels.\n")

    return im
