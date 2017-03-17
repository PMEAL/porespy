import scipy as sp
import porespy as ps
import scipy.ndimage as spim
import sys


def extract_pore_network(im, dt=None):
    r"""
    Analyzes an image that has been partitioned into pore regions and extracts
    the pore and throat geometry as well as network connectivity.

    Parameters
    ----------
    im : ND-array
        An image of the pore space partitioned into individual pore regions.
        Note that this image must have zeros indicating the solid phase.

    dt : ND-array
        The distance transform of the pore space.  If not given it will be
        calculated, but it can save time to provide one if available.

    Returns
    -------
    A dictionary containing all the pore and throat size data, as well as the
    network topological information.  The dictionary names use the OpenPNM
    convention (i.e. 'pore.coords', 'throat.conns') so it may be converted
    directly to an OpenPNM network object using the ``update`` command.
    """
    print('_'*60)
    print('Extracting pore and throat information from image')
    from skimage.morphology import disk, square, ball, cube
    if im.ndim == 2:
        cube = square
        ball = disk

    if ~sp.any(im == 0):
        raise Exception('The received image has no solid phase (0\'s)')

    if dt is None:
        dt = spim.distance_transform_edt(im > 0)

    # Get 'slices' into im for each pore region
    slices = spim.find_objects(im)

    # Initialize arrays
    Ps = sp.arange(1, sp.amax(im)+1)
    Np = sp.size(Ps)
    p_coords = sp.zeros((Np, im.ndim), dtype=float)
    p_volume = sp.zeros((Np, ), dtype=float)
    p_diameter = sp.zeros((Np, ), dtype=float)
    p_label = sp.zeros((Np, ), dtype=int)
    p_vxls = []
    t_conns = []
    t_diameter = []
    t_vxls = []
    t_coords = []

    # Start extracting size information for pores and throats
    for i in Ps:
        pore = i - 1
        s = extend_slices(slices[pore], im.shape)
        sub_im = im[s]
        s_offset = sp.array([i.start for i in s])
        vx = sp.where(sub_im == i)
        p_inds = tuple([i+j for i, j in zip(vx, s_offset)])
        p_vxls.append(p_inds)
        p_label[pore] = i
        p_coords[pore, :] = sp.mean(p_vxls[pore], axis=1)
        p_volume[pore] = sp.size(p_vxls[pore][0])
        p_diameter[pore] = 2*sp.amax(dt[p_vxls[pore]])
        im_w_throats = spim.binary_dilation(input=sub_im == i,
                                            structure=ball(1))
        im_w_throats = im_w_throats*sub_im
        Pn = sp.unique(im_w_throats)[1:] - 1
        for j in Pn:
            if j > pore:
                t_conns.append([pore, j])
                vx = sp.where(im_w_throats == (j + 1))
                t_inds = p_inds = tuple([i+j for i, j in zip(vx, s_offset)])
                t_vxls.append(t_inds)
                t_diameter.append(2*sp.amax(dt[t_inds]))
                temp = sp.where(dt[t_inds] == sp.amax(dt[t_inds]))[0][0]
                if im.ndim == 2:
                    t_coords.append(tuple((t_inds[0][temp], t_inds[1][temp])))
                else:
                    t_coords.append(tuple((t_inds[0][temp], t_inds[1][temp], t_inds[2][temp])))
    # Clean up values
    Nt = len(t_vxls)  # Get number of throats
    if im.ndim == 2:  # If 2D, add 0's in 3rd dimension
        p_coords = sp.vstack((p_coords.T, sp.zeros((Np, )))).T
        t_coords = sp.vstack((sp.array(t_coords).T, sp.zeros((Nt, )))).T

    # Start creating dictionary of pore network information
    net = {}
    net['pore.all'] = sp.ones((Np, ), dtype=bool)
    net['throat.all'] = sp.ones((Nt, ), dtype=bool)
    net['pore.coords'] = p_coords
    net['throat.coords'] = sp.array(t_coords)
    net['throat.conns'] = sp.array(t_conns)
#    net['pore.voxels'] = sp.array(p_vxls)
#    net['throat.voxels'] = sp.array(t_vxls)
    net['pore.label'] = sp.array(p_label)
    net['pore.volume'] = sp.copy(p_volume)
    net['throat.volume'] = sp.zeros((Nt, ), dtype=float)
    net['pore.diameter'] = sp.copy(p_diameter)
    net['pore.equivalent_diameter'] = (3/4*p_volume)**(1/3)*2
    net['throat.diameter'] = sp.array(t_diameter)
    P12 = net['throat.conns']
    PT1 = sp.sqrt(sp.sum((p_coords[P12[:, 0]] - t_coords)**2, axis=1))
    PT2 = sp.sqrt(sp.sum((p_coords[P12[:, 1]] - t_coords)**2, axis=1))
    net['throat.length'] = PT1 + PT2
    PP = sp.sqrt(sp.sum((p_coords[P12[:, 0]] - p_coords[P12[:, 1]])**2, axis=1))
    net['throat.length1'] = PP

    return net


def extend_slices(slices, shape):
    a = []
    for s, dim in zip(slices, shape):
        start = 0
        stop = dim
        if s.start > 0:
            start = s.start - 1
        if s.stop < dim:
            stop = s.stop + 1
        a.append(slice(start, stop, None))
    return a
