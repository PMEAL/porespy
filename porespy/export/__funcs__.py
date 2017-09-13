import numpy as np
from scipy import ndimage as spim
from porespy.export.evtk import hl as bp


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
