import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as spim
import porespy.export.hl as bp


def vox2vtk(im, path='./voxvtk', divide=False):
    im = im.astype(np.int)
    if divide == True:
        split = np.round(im.shape[2]/2).astype(np.int)
        im1 = im[:, :, 0:split]
        im2 = im[:, :, split:]
        bp.imageToVTK(path+'1', cellData={'vox': np.ascontiguousarray(im1)})
        bp.imageToVTK(path+'2', cellData={'vox': np.ascontiguousarray(im2)})
    else:
        bp.imageToVTK(path, cellData={'vox': np.ascontiguousarray(im)})
    

def im2vtk(im, path='./imvtk'):
    bp.imageToVTK(path, cellData={'im': np.ascontiguousarray(im)})
    