import numpy as np
from scipy import ndimage as spim
from porespy.export.evtk import hl as bp
import scipy.ndimage as nd


def vox2vtk(im, path='./voxvtk', divide=False, downsample=False, voxel_size=1):
    im = im.astype(np.int)
    vs = voxel_size
    if divide == True:
        split = np.round(im.shape[2]/2).astype(np.int)
        im1 = im[:, :, 0:split]
        im2 = im[:, :, split:]
        bp.imageToVTK(path+'1', cellData={'vox': np.ascontiguousarray(im1)},
                      spacing=(vs, vs, vs))
        bp.imageToVTK(path+'2', origin=(0.0, 0.0, split),
                      cellData={'vox': np.ascontiguousarray(im2)},
                      spacing=(vs, vs, vs))
    elif downsample == True:
        im = spim.interpolation.zoom(im, zoom=0.5, order=0)
        bp.imageToVTK(path, cellData={'vox': np.ascontiguousarray(im)},
                      spacing=(vs, vs, vs))
    else:
        bp.imageToVTK(path, cellData={'vox': np.ascontiguousarray(im)},
                      spacing=(vs, vs, vs))
            
    
def im2vtk(im, path='./imvtk', divide=False, downsample=False):
    if divide == True:
        split = np.round(im.shape[2]/2).astype(np.int)
        im1 = im[:, :, 0:split]
        im2 = im[:, :, split:]
        bp.imageToVTK(path+'1', cellData={'im': np.ascontiguousarray(im1)})
        bp.imageToVTK(path+'2', origin=(0.0, 0.0, split), cellData={'im': np.ascontiguousarray(im2)})
    elif downsample == True:
        im = spim.interpolation.zoom(im, zoom=0.5, order=0)
        bp.imageToVTK(path, cellData={'im': np.ascontiguousarray(im)})
    else:
        bp.imageToVTK(path, cellData={'im': np.ascontiguousarray(im)})


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
    x, y, z = np.shape(dt)
    f = open(filename, 'w')
    for k in range(z):
        for j in range(y):
            for i in range(x):
                f.write(str(dt[i, j, k])+'\n')
    f.close()
    