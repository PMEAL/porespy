import scipy as sp
import scipy.ndimage as spim
from skimage.morphology import ball, disk, square, cube
import OpenPNM as op

def randomize_colors(im):
    new_colors = sp.random.randint(1, im.max(), im.max()+1)
    new_colors[0] = 0
    new_im = new_colors[im]
    return new_im

def feature_size(labeled_image, N=None):
    if N == None:
        N = sp.amax(labeled_image)
    sizes = sp.zeros_like(labeled_image, dtype=int)
    for i in range(1, N+1):
        crds = sp.where(labeled_image == i)
        sizes[crds] = crds[0].size
    return sizes

def flood_max(im, max_iter=100):
    if get_dims(im) == 2:
        strel = square(3)
    else:
        strel = cube(3)
    mask = im != 0
    im_new = sp.copy(im)
    im_old = sp.zeros_like(im,dtype=int)
    i = 0
    while sp.any(im_new - im_old) and (i < max_iter):
        im_old = im_new
        im_new = spim.maximum_filter(im_new, footprint=strel)
        im_new[~mask] = 0
        i += 1
    return im_new

def flood_min(im, max_iter=100):
    if get_dims(im) == 2:
        strel = square(3)
    else:
        strel = cube(3)
    immax = sp.amax(im)*2
    mask = im != 0
    im_new = sp.copy(im)
    im_new[~mask] = immax
    im_old = sp.zeros_like(im,dtype=int)
    i = 0
    while sp.any(im_new - im_old) and (i < max_iter):
        im_old = im_new
        im_new = spim.minimum_filter(im_new, footprint=strel)
        im_new[~mask] = immax
        i += 1
    im_new[~mask] = 0
    return im_new

def concentration_transform(im):
    net = op.Network.Cubic(shape=im.shape)
    net.fromarray(im, propname='pore.void')
    net.fromarray(~im, propname='pore.solid')
    geom = op.Geometry.GenericGeometry(network=net, pores=net.Ps, throats=net.Ts)
    phase = op.Phases.GenericPhase(network=net)
    phys = op.Physics.GenericPhysics(network=net, phase=phase, geometry=geom)
    phys['throat.diffusive_conductance'] = 1
    phys['pore.A1'] = 1
    phys['pore.A2'] = 2
    phys.models.add(propname='pore.sink',
                    model=op.Physics.models.generic_source_term.linear,
                    A1='pore.A1', A2='pore.A2')
    alg = op.Algorithms.FickianDiffusion(network=net, phase=phase)
    alg.set_boundary_conditions(bctype='Neumann', bcvalue=-1, pores=net.pores('void'))
    alg.set_boundary_conditions(bctype='Dirichlet', bcvalue=0, pores=net.pores('solid'))
#    alg.set_source_term(source_name='pore.sink', pores=net.pores('solid'))
    alg.run()
    ct = net.asarray(alg['pore.mole_fraction']).squeeze()
    return ct

def get_weighted_markers(im, Rs):
    if im.ndim == 2:
        strel = disk
    elif im.ndim == 3:
        strel = ball
    Rs = sp.sort(Rs)
    dt = spim.distance_transform_edt(im)
    weighted_markers = sp.zeros_like(im, dtype=float)
    for r in Rs:
        mx = spim.maximum_filter(input=dt, footprint=strel(r))
        markers = (mx==dt)*im
        pts = sp.where(markers)
        weighted_markers[pts[0], pts[1]] = markers[pts[0], pts[1]]*r
    return weighted_markers

def find_edges(im, strel=None):
    if strel == None:
        if im.ndim == 2:
            strel = disk(1)
        elif im.ndim == 3:
            strel = ball(1)
    temp = spim.convolve(input=im, weights=strel)/sp.sum(strel)
    temp = im != temp
    return temp

def get_border(im, thickness=1):
    ndims = get_dims(im)
    t = thickness
    border = sp.ones_like(im, dtype=bool)
    if ndims == 2:
        border[t:-t, t:-t] = False
    if ndims == 3:
        border[t:-t, t:-t, t:-t] = False
    return border

def fill_border(im, thickness=1, value=1):
    border = get_border(im, thickness=thickness)
    coords = sp.where(border)
    im[coords] = value
    return im

def get_dims(im):
    if im.ndim == 2:
        return 2
    if (im.ndim == 3) and (im.shape[2] == 1):
        return 2
    if im.ndim == 3:
        return 3

def rotate_image_and_repeat(im):
    # Find all markers in distance transform
    weighted_markers = sp.zeros_like(im, dtype=float)
    for phi in sp.arange(0, 45, 2):
        temp = spim.rotate(im.astype(int), angle=phi, order=5, mode='constant', cval=1)
        temp = get_weighted_markers(temp, Rs)
        temp = spim.rotate(temp.astype(int), angle=-phi, order=0, mode='constant', cval=0)
        X_lo = sp.floor(temp.shape[0]/2-im.shape[0]/2).astype(int)
        X_hi = sp.floor(temp.shape[0]/2+im.shape[0]/2).astype(int)
        weighted_markers += temp[X_lo:X_hi, X_lo:X_hi]
    return weighted_markers

def remove_isolated_voxels(im, conn=None):
    if im.ndim == 2:
        if conn == 4:
            strel = disk(1)
        elif conn in [None, 8]:
            strel = square(3)
    elif im.ndim == 3:
        if conn == 6:
            strel = ball(1)
        elif conn in [None, 26]:
            strel = cube(3)
    filtered_im = sp.copy(im)
    id_regions, num_ids = spim.label(filtered_im, structure=strel)
    id_sizes = sp.array(spim.sum(im, id_regions, range(num_ids + 1)))
    area_mask = (id_sizes == 1)
    filtered_im[area_mask[id_regions]] = 0
    return filtered_im
