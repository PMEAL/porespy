from porespy.network_extraction import regions_to_network, add_boundary_regions
from porespy.filters import snow_partitioning


def snow(im, voxel_size=1, boundary_faces=['top', 'bottom', 'left',
                                           'right', 'front', 'back']):
    r"""
    """
    regions = snow_partitioning(im=im, return_all=True)
    im = regions.im
    dt = regions.dt
    regions = regions.regions
    regions = add_boundary_regions(regions=regions, faces=boundary_faces)
    net = regions_to_network(im=regions*im, dt=dt, voxel_size=voxel_size)
    return net
