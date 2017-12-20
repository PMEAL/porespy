import scipy as sp
import scipy.ndimage as spim
from scipy.spatial.distance import cdist
from porespy.tools import extend_slice
from tqdm import tqdm
from porespy.network_extraction import extract_pore_network


def extract_dual_network(im, pore_regions=None, solid_regions=None,
                         voxel_size=1):

    r"""
    Analyzes an image that has been partitioned into pore and solid regions
    and extracts the pore and solid phase geometry as well as network
    connectivity.

    Parameters
    ----------
    im : ND-array
        An image of the pore space partitioned into individual pore regions.
        Note that this image must have zeros indicating the solid phase.

    pore_regions : ND-array
        A ND-array the with regions belonging to each peak labelled in
        pore phase.

    solid_regions : ND-array
        A ND-array the with regions belonging to each peak labelled in
        solid phase.

    voxel_size : scalar
        The resolution of the image, expressed as the length of one side of a
        voxel, so the volume of a voxel would be **voxel_size**-cubed.  The
        default is 1, which is useful when overlaying the PNM on the original
        image since the scale of the image is alway 1 unit lenth per voxel.

    Returns
    -------
    A dictionary containing all the pore and solid phase size data, as well as
    the network topological information.  The dictionary names use the OpenPNM
    convention (i.e. 'pore.coords', 'throat.conns') so it may be converted
    directly to an OpenPNM network object using the ``update`` command.
    """
    pore_region = pore_regions*im
    solid_region = solid_regions*~im
    solid_num = sp.amax(pore_regions)
    solid_mask = solid_region > 0
    solid_region = solid_region + solid_num
    solid_region = solid_region * solid_mask
    pore_dt = spim.distance_transform_edt(pore_region > 0)
    pore_dt = spim.gaussian_filter(pore_dt, sigma=0.4)
    solid_dt = spim.distance_transform_edt(solid_region > 0)
    solid_dt = spim.gaussian_filter(solid_dt, sigma=0.4)
    dt = pore_dt + solid_dt
    regions = pore_region + solid_region
    net = extract_pore_network(im=regions, dt=dt, voxel_size=voxel_size)
    p_on_s = pore_regions*(~im)    # Expose pores labels on solid
    s_on_p = (solid_regions + solid_num) * im  # Expose solid labels on pores
    pv = sp.unique(p_on_s, return_counts=True)[1]
    pv = pv[1:]
    sv = sp.unique(s_on_p, return_counts=True)[1]
    sv = sv[1:]
    p_solid_volume = sp.concatenate((pv, sv))
    loc1 = net['throat.conns'][:, 0] < solid_num
    loc2 = net['throat.conns'][:, 1] >= solid_num
    pore_solid_labels = loc1 * loc2
    loc3 = net['throat.conns'][:, 0] >= solid_num
    solid_solid_labels = loc3 * loc2
    loc4 = net['throat.conns'][:, 1] < solid_num
    pore_pore_labels = loc1 * loc4
    p_conns = net['throat.conns'][:, 0][pore_solid_labels]
    ps = net['throat.area'][pore_solid_labels]
    p_sa = sp.bincount(p_conns, ps)
    s_conns = net['throat.conns'][:, 1][pore_solid_labels]
    s_sa = sp.bincount(s_conns, ps)
    s_sa = sp.trim_zeros(s_sa)
    p_solid_surf = sp.concatenate((p_sa, s_sa))

    net['pore.solid_volume'] = p_solid_volume * voxel_size**3
    net['pore.solid_surface_area'] = p_solid_surf * voxel_size**2
    net['pore.pore_labels'] = pore_pore_labels
    net['pore.solid_labels'] = pore_solid_labels
    net['pore.solid_solid_labels'] = solid_solid_labels

    return net
