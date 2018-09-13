import numpy as np
from scipy import ndimage as spim
from porespy.io.evtk import hl as bp
import scipy.ndimage as nd


def to_vtk(im, path='./voxvtk', divide=False, downsample=False, voxel_size=1,
           vox=False):
    r"""
    Wrapper for the pyevtk
    Copyright 2010 - 2016 Paulo A. Herrera. All rights reserved. (see /evtk
    folder for complete license information)

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
        bp.imageToVTK(path+'1', cellData={'im': np.ascontiguousarray(im1)},
                      spacing=(vs, vs, vs))
        bp.imageToVTK(path+'2', origin=(0.0, 0.0, split*vs),
                      cellData={'im': np.ascontiguousarray(im2)},
                      spacing=(vs, vs, vs))
    elif downsample:
        im = spim.interpolation.zoom(im, zoom=0.5, order=0, mode='reflect')
        bp.imageToVTK(path, cellData={'im': np.ascontiguousarray(im)},
                      spacing=(2*vs, 2*vs, 2*vs))
    else:
        bp.imageToVTK(path, cellData={'im': np.ascontiguousarray(im)},
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
