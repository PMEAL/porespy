from porespy.network_extraction import regions_to_network, add_boundary_regions
from porespy.filters import snow_partitioning
from porespy.tools import make_contiguous
import scipy as sp


def snow(im, voxel_size=1, boundary_faces=['top', 'bottom', 'left',
                                           'right', 'front', 'back']):
    r"""
    Analyzes an image that has been partitioned into void and solid regions
    and extracts the void and solid phase geometry as well as network
    connectivity.

    Parameters
    ----------
    im : ND-array
        Binary image in the Boolean form with True’s as void phase and False’s
        as solid phase.

    voxel_size : scalar
        The resolution of the image, expressed as the length of one side of a
        voxel, so the volume of a voxel would be **voxel_size**-cubed.  The
        default is 1, which is useful when overlaying the PNM on the original
        image since the scale of the image is alway 1 unit lenth per voxel.

    boundary_faces : list of strings
        Boundary faces labels are provided to assign hypothetical boundary
        nodes having zero resistance to transport process. For cubical
        geometry, the user can choose ‘left’, ‘right’, ‘top’, ‘bottom’,
        ‘front’ and ‘back’ face labels to assign boundary nodes. If no label is
        assigned then all six faces will be selected as boundary nodes
        automatically which can be trimmed later on based on user requirements.

    Returns
    -------
    A dictionary containing the void phase size data, as well as the network
    topological information.  The dictionary names use the OpenPNM
    convention (i.e. 'pore.coords', 'throat.conns') so it may be converted
    directly to an OpenPNM network object using the ``update`` command.
    """

    regions = snow_partitioning(im=im, return_all=True)
    im = regions.im
    dt = regions.dt
    regions = regions.regions
    b_num = sp.amax(regions)
    regions = add_boundary_regions(regions=regions, faces=boundary_faces)
    f = boundary_faces
    if f is not None:
        if im.ndim == 2:
            faces = [(int('left' in f)*3, int('right' in f)*3),
                     (int(('front') in f)*3 or int(('bottom') in f)*3,
                      int(('back') in f)*3 or int(('top') in f)*3)]

        if im.ndim == 3:
            faces = [(int('left' in f)*3, int('right' in f)*3),
                     (int('front' in f)*3, int('back' in f)*3),
                     (int('top' in f)*3, int('bottom' in f)*3)]
        dt = sp.pad(dt, pad_width=faces, mode='edge')
        im = sp.pad(im, pad_width=faces, mode='edge')
    else:
        dt = dt
    regions = regions*im
    regions = make_contiguous(regions)
    net = regions_to_network(im=regions, dt=dt, voxel_size=voxel_size)
    boundary_labels = net['pore.label'] > b_num
    loc1 = net['throat.conns'][:, 0] < b_num
    loc2 = net['throat.conns'][:, 1] >= b_num
    pore_labels = net['pore.label'] <= b_num
    loc3 = net['throat.conns'][:, 0] < b_num
    loc4 = net['throat.conns'][:, 1] < b_num
    net['pore.boundary'] = boundary_labels
    net['throat.boundary'] = loc1 * loc2
    net['pore.internal'] = pore_labels
    net['throat.internal'] = loc3 * loc4
    # -------------------------------------------------------------------------
    # label boundary pore faces
    if f is not None:
        coords = net['pore.coords']
        condition = coords[net['pore.internal']]
        dic = {'left': 0, 'right': 0, 'front': 1, 'back': 1,
               'top': 2, 'bottom': 2}
        if all(coords[:, 2] == 0):
            dic['top'] = 1
            dic['bottom'] = 1
        print(dic)
        for i in f:
            if i in ['left', 'front', 'bottom']:
                net['pore.{}'.format(i)] = (coords[:, dic[i]] <
                                            min(condition[:, dic[i]]))
            elif i in ['right', 'back', 'top']:
                net['pore.{}'.format(i)] = (coords[:, dic[i]] >
                                            max(condition[:, dic[i]]))
    return net
