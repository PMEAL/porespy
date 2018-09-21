import scipy as sp
from porespy.tools import make_contiguous
from skimage.segmentation import find_boundaries


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
            # ---------------------------------------------------------------------
            regions = sp.pad(regions, 2, 'edge')

            # Remove unselected faces
            if 'top' not in faces:
                regions = regions[:, 3:, :]
            if 'bottom' not in faces:
                regions = regions[:, :-3, :]
            if 'front' not in faces:
                regions = regions[3:, :, :]
            if 'back' not in faces:
                regions = regions[:-3, :, :]
            if 'left' not in faces:
                regions = regions[:, :, 3:]
            if 'right' not in faces:
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
            # ---------------------------------------------------------------------
            regions = sp.pad(regions, 2, 'edge')

            # Remove unselected faces
            if 'top' not in faces:
                regions = regions[3:, :]
            if 'bottom' not in faces:
                regions = regions[:-3, :]
            if 'left' not in faces:
                regions = regions[:, 3:]
            if 'right' not in faces:
                regions = regions[:, :-3]
        else:
            print('add_boundary_regions works only on 2D and 3D images')
        # ---------------------------------------------------------------------
        # Make labels contiguous
        regions = make_contiguous(regions)
    else:
        regions = regions

    return regions
