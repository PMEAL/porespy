import numpy as np
from skimage import io
import porespy as ps


# either generate an image or read and existing one
im = io.imread('./image_to_read.tif', as_gray=False)
im = ~np.array(im, dtype=bool)

# this should generate an "blockMeshDict" file
ps.io.to_openfoam(im, label_boundaries=True, path='./openfoam_case/system/')

# to generate the mesh, simply run "blockMesh" command from a terminal after
# doing "cd" to the "openfoam_case"
