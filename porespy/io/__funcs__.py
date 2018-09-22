import pickle
import numpy as np
from scipy import ndimage as spim
from pyevtk.hl import imageToVTK
import scipy.ndimage as nd
import skimage.io as io
from pathlib import Path


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

    Output
    ------
    File: vtk, vtp or vti file that can opened in ParaView
    """
    vs = voxel_size
    for entry in data:
        if data[entry].flags['C_CONTIGUOUS']:
            data[entry] = np.ascontiguousarray(data[entry])
    imageToVTK(path, cellData=data, spacing=(vs, vs, vs), origin=origin)


def to_openpnm(net, filename):
    r"""
    Save the result of the `extract_pore_network` function to a file that is
    suitable for opening in OpenPNM.

    Parameters
    ----------
    net : dict
        The dictionary object produced by `extract_pore_network`
    filename : string or path object
        The name and location to save the file, which will have `.net` file
        extension.

    """
    p = Path(filename)
    p = p.resolve()
    # If extension not part of filename
    if p.suffix == '':
        p = p.with_suffix('.net')
    with open(p, 'wb') as f:
        pickle.dump(net, f)


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

    Output
    ------
    File: vtk, vtp or vti file that can opened in paraview
    """
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

    Output
    -------
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
