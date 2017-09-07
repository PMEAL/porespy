import numpy as np
import scipy.ndimage as nd


def to_vtk(im, filename):
    r"""
    Converts an ND-array image to a file suitable for visualization in VTK
    compatible applications such as ParaView.

    """
    pass


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
