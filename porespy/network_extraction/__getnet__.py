import scipy as sp
import scipy.ndimage as spim


def extract_pore_network(im):
    r"""
    Analyzes an image that has been partitioned into pore regions and extracts
    the pore and throat geometry as well as network connectivity.

    Returns
    -------
    A dictionary containing all the pore and throat sizes, as well as the
    network topological information.  The dictionary names use the OpenPNM
    convention (i.e. 'pore.coords', 'throat.conns') so it may be converted
    directly to an OpenPNM network object using the ``update`` command.
    """
    from skimage.morphology import disk, square, ball, cube
    if im.ndim == 2:
        cube = square
        ball = disk

    # Process input image for determination of sizes
    dt = spim.distance_transform_edt(im > 0)
    mn = spim.minimum_filter(im + (im == 0)*2*sp.amax(im), footprint=ball(1))
    mx = spim.maximum_filter(im, footprint=ball(1))
    im_w_throats = sp.copy(im)
    inds = (mn != im)
    im_w_throats[inds] = mn[inds]
    inds = (mx != im)
    im_w_throats[inds] = mx[inds]
    im_w_throats *= im > 0

    # Start extracting size information for pores and throats
    Ps = sp.unique(im)[1:]
    Np = sp.size(Ps)
    p_coords = sp.zeros((Np, im.ndim), dtype=int)
    p_volume = sp.zeros((Np, ), dtype=int)
    p_diameter = sp.zeros((Np, ), dtype=int)
    p_label = sp.zeros((Np, ), dtype=int)
    p_vxls = sp.empty((Np, ), dtype=object)
    t_conns = []
    t_diameter = []
    t_vxls = []
    t_coords = []
    for i in Ps:
        pore = i - 1
        p_vxls[pore] = sp.where(im == i)
        # Get pore info
        p_label[pore] = i
        p_coords[pore, :] = sp.mean(p_vxls[pore], axis=1)
        p_volume[pore] = sp.size(p_vxls[pore][0])
        p_diameter[pore] = 2*sp.amax(dt[p_vxls[pore]])
        # Get throat info
        Pn = sp.unique(im_w_throats[p_vxls[pore]]) - 1
        for j in Pn:
            if j > pore:
                t_conns.append([pore, j])
                temp = im_w_throats[p_vxls[pore]] == (j + 1)
                inds = tuple((p_vxls[pore][0][temp], p_vxls[pore][1][temp], p_vxls[pore][2][temp]))
                t_vxls.append(inds)
                t_diameter.append(2*sp.amax(dt[inds]))
                temp = sp.where(dt[inds] == sp.amax(dt[inds]))[0][0]
                t_coords.append(tuple((inds[0][temp], inds[1][temp], inds[2][temp])))
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
    net['throat.conns'] = sp.array(t_conns)
    net['pore.label'] = sp.array(p_label)
    net['pore.volume'] = p_volume
    net['throat.volume'] = sp.zeros((Nt, ), dtype=float)
    net['pore.diameter'] = p_diameter
    net['throat.diameter'] = sp.array(t_diameter)
    net['throat.coords'] = sp.array(t_coords)
    P12 = net['throat.conns']
    PT1 = sp.sqrt(sp.sum((p_coords[P12[:, 0]] - t_coords)**2, axis=1))
    PT2 = sp.sqrt(sp.sum((p_coords[P12[:, 1]] - t_coords)**2, axis=1))
    net['throat.length'] = PT1 + PT2
    PP = sp.sqrt(sp.sum((p_coords[P12[:, 0]] - p_coords[P12[:, 1]])**2, axis=1))
    net['throat.length1'] = PP

    return net
