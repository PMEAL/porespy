import numpy as np
from scipy import ndimage as spim
import scipy.ndimage as nd
from porespy.networks import generate_voxel_image
from pyevtk.hl import imageToVTK


def dict_to_vtk(data, path='./dictvtk', voxel_size=1, origin=(0, 0, 0)):
    r"""
    Accepts multiple images as a dictionary and compiles them into a vtk file

    Parameters
    ----------
    data : dict
        A dictionary of *key: value* pairs, where the *key* is the name of the
        scalar property stored in each voxel of the array stored in the
        corresponding *value*.
    path : string
        Path to output file
    voxel_size : int
        The side length of the voxels (voxels  are cubic)
    origin : float
        data origin (according to selected voxel size)

    Notes
    -----
    Outputs a vtk, vtp or vti file that can opened in ParaView
    """
    vs = voxel_size
    for entry in data:
        if data[entry].dtype == bool:
            data[entry] = data[entry].astype(np.int8)
        if data[entry].flags['C_CONTIGUOUS']:
            data[entry] = np.ascontiguousarray(data[entry])
    imageToVTK(path, cellData=data, spacing=(vs, vs, vs), origin=origin)


def to_openpnm(net, filename):
    r"""
    Save the result of the `snow` network extraction function in a format
    suitable for opening in OpenPNM.

    Parameters
    ----------
    net : dict
        The dictionary object produced by the network extraction functions

    filename : string or path object
        The name and location to save the file, which will have `.net` file
        extension.

    """
    from openpnm.network import GenericNetwork
    # Convert net dict to an openpnm Network
    pn = GenericNetwork()
    pn.update(net)
    pn.project.save_project(filename)
    ws = pn.project.workspace
    ws.close_project(pn.project)


def to_vtk(im, path='./voxvtk', divide=False, downsample=False, voxel_size=1,
           vox=False):
    r"""
    Converts an array to a vtk file.

    Parameters
    ----------
    im : 3D image
        The image of the porous material
    path : string
        Path to output file
    divide : bool
        vtk files can get very large, this option allows you for two output
        files, divided at z = half. This allows for large data sets to be
        imaged without loss of information
    downsample : bool
        very large images acan be downsampled to half the size in each
        dimension, this doubles the effective voxel size
    voxel_size : int
        The side length of the voxels (voxels  are cubic)
    vox : bool
        For an image that is binary (1's and 0's) this reduces the file size by
        using int8 format (can also be used to reduce file size when accuracy
        is not necessary ie: just visulization)

    Notes
    -----
    Outputs a vtk, vtp or vti file that can opened in paraview
    """
    if len(im.shape) == 2:
        im = im[:, :, np.newaxis]
    if im.dtype == bool:
        vox = True
    if vox:
        im = im.astype(np.int8)
    vs = voxel_size
    if divide:
        split = np.round(im.shape[2]/2).astype(np.int)
        im1 = im[:, :, 0:split]
        im2 = im[:, :, split:]
        imageToVTK(path+'1', cellData={'im': np.ascontiguousarray(im1)},
                   spacing=(vs, vs, vs))
        imageToVTK(path+'2', origin=(0.0, 0.0, split*vs),
                   cellData={'im': np.ascontiguousarray(im2)},
                   spacing=(vs, vs, vs))
    elif downsample:
        im = spim.interpolation.zoom(im, zoom=0.5, order=0, mode='reflect')
        imageToVTK(path, cellData={'im': np.ascontiguousarray(im)},
                   spacing=(2*vs, 2*vs, 2*vs))
    else:
        imageToVTK(path, cellData={'im': np.ascontiguousarray(im)},
                   spacing=(vs, vs, vs))


def to_palabos(im, filename, solid=0):
    r"""
    Converts an ND-array image to a text file that Palabos can read in as a
    geometry for Lattice Boltzmann simulations. Uses a Euclidean distance
    transform to identify solid voxels neighboring fluid voxels and labels
    them as the interface.

    Parameters
    ----------
    im : ND-array
        The image of the porous material

    filename : string
        Path to output file

    solid : int
        The value of the solid voxels in the image used to convert image to
        binary with all other voxels assumed to be fluid.

    Notes
    -----
    File produced contains 3 values: 2 = Solid, 1 = Interface, 0 = Pore
    Palabos will run the simulation applying the specified pressure drop from
    x = 0 to x = -1.

    """
    # Create binary image for fluid and solid phases
    bin_im = im == solid
    # Transform to integer for distance transform
    bin_im = bin_im.astype(int)
    # Distance Transform computes Euclidean distance in lattice units to
    # Nearest fluid for every solid voxel
    dt = nd.distance_transform_edt(bin_im)
    dt[dt > np.sqrt(2)] = 2
    dt[(dt > 0)*(dt <= np.sqrt(2))] = 1
    dt = dt.astype(int)
    # Write out data
    with open(filename, 'w') as f:
        out_data = dt.flatten().tolist()
        f.write('\n'.join(map(repr, out_data)))


def openpnm_to_im(network, pore_shape="sphere", throat_shape="cylinder",
                  max_dim=None, verbose=1, rtol=0.1):
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
    im : ND-array
        Voxelated image corresponding to the given pore network model

    Notes
    -----
    (1) The generated voxelated image is labeled with 0s, 1s and 2s signifying
    solid phase, pores, and throats respectively.

    (2) If max_dim is not provided, the method calculates it such that the
    further increasing it doesn't change porosity by much.

    """
    return generate_voxel_image(network, pore_shape=pore_shape,
                                throat_shape=throat_shape, max_dim=max_dim,
                                verbose=verbose, rtol=rtol)
