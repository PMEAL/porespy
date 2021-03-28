import numpy as np
from porespy.networks import regions_to_network
from porespy.networks import add_boundary_regions2
from porespy.networks import label_phases, label_boundaries
from porespy.filters import snow_partitioning
from collections import namedtuple


def snow2(
        phases,
        phase_alias=None,
        boundary_width=3,
        boundary_labels=None,
        accuracy='standard',
        voxel_size=1,
        return_all=False,
        ):
    r"""
    Applies the SNOW algorithm to each phase indicated in ``phases``.

    This function is a combination of ``snow`` [1]_, ``snow_dual`` [2]_
    and ``snow_n`` [3]_ from previous versions.

    Parameters
    ----------
    phases : ND-image
        An image indicating the phase(s) of interest. A watershed is produced
        for each integer value in ``phases`` (except 0's).  Tthese are
        combined into a single image and one network is extracted using
        ``regions_to_network``.
    phase_alias : dict
        A mapping between integer values in ``phases`` and phase name.
        For instance, asssuming a two-phase image, ``{1: 'void', 2: 'solid'}``
        will result in the labels ``'pore.void'`` and ``'pore.solid'``,
        as well as ``'throat.solid_void'``, ``'throat.solid_solid'``, and
        ``'throat.void_void'``. If not provided, labels are assumed to as
        ``{1: 'phase1', 2: 'phase2, etc}``
    boundary_width : int, list of ints, or list of lists
        Number of voxels to add to the beginning and end of each axis. This
        argument is handled the same as ``pad_width`` in the ``np.pad``
        function. A scalar adds the same amount to the beginning and end of
        each axis. [A, B] adds A to the beginning of each axis and B to the
        ends.  [[A, B], ..., [C, D]] adds A to the beginning and B to the
        end of the first axis, and so on. The default is to add 3 voxels on
        both ends of all axes.
    boundary_labels : list of lists
        A 3-element list, with each element containing a pair of strings
        indicating the label to apply to the beginning and end of each axis.
        For instance, ``[['left', 'right'], ['front', 'back'],
        ['top', 'bottom']]`` will will apply the label ``'left'`` to all pores
        with the minimum x-coordinate, and ``'right'`` to the pores with the
        maximum x-coordinate, and so on.
    accuracy : string
        Controls how accurately certain properties are calculated during the
        analysis of regions in the ``regions_to_network`` function.
        Options are:

        'standard' (default)
            Computes the surface areas and perimeters by simply counting
            voxels. This is *much* faster but does not properly account
            for the rough, voxelated nature of the surfaces.
        'high'
            Computes surface areas using the marching cube method, and
            perimeters using the fast marching method. These are substantially
            slower but better account for the voxelated nature of the images.

    voxel_size : scalar (default = 1)
        The resolution of the image, expressed as the length of one side of a
        voxel, so the volume of a voxel would be **voxel_size**-cubed.
    return_all : boolean
        If set to ``True`` a named tuple is returned containing the
        padded image, the pore regions, and the dictionary containing the
        extracted network. The default is ``False``, so only the network is
        returned.

    Returns
    -------
    network : dict or named-tuple
        If ``return_all = False`` then a dictionary containing the extracted
        network data is returned.  If ``return_all = True`` then a
        *named-tuple* is returned with the padded ``phases`` image, the
        watershed segmentated ``regions``, and the ``network`` dictionary.

    References
    ----------
    .. [1] Gostick JT. Versatile and efficient pore network extraction
       method using marker-based watershed segmentation. Phys. Rev. E 96,
       023307 (2017)
    .. [2] Khan ZA, Tranter TG, Agnaou M, Elkamel A, and Gostick JT, Dual
       network extraction algorithm to investigate multiple transport
       processes in porous materials: Image-based modeling of pore and grain-
       scale processes. Computers and Chemical Engineering. 123(6), 64-77
       (2019)
    .. [3] Khan ZA, García-Salaberri PA, Heenan T, Jervis R, Shearing P,
       Brett D, Elkamel A, Gostick JT, Probing the structure-performance
       relationship of lithium-ion battery cathodes using pore-networks
       extracted from three-phase tomograms. Journal of the Electrochemical
       Society. 167(4), 040528 (2020)
    """
    regions = np.zeros_like(phases)
    for i in range(phases.max()):
        phase = phases == (i + 1)
        snow = snow_partitioning(im=phase, randomize=False, return_all=True,
                                 sigma=0.4, r_max=4)
        regions += snow.regions + regions.max()*(snow.regions > 0)
    regions = add_boundary_regions2(regions, pad_width=boundary_width)
    phases = np.pad(phases, pad_width=boundary_width, mode='edge')
    net = regions_to_network(regions, phases=phases)
    if phase_alias is None:
        phase_alias = {i+1: 'phase' + str(i+1) for i in range(phases.max())}
    net = label_phases(net, alias=phase_alias)
    if boundary_labels is None:
        boundary_labels = [['left', 'right'],
                           ['front', 'back'],
                           ['top', 'bottom'] * (phases.ndim > 2)]
    net = label_boundaries(net, labels=boundary_labels)
    if return_all:  # Collect result in a tuple for return
        result = namedtuple('snow2', ('network', 'regions', 'phases'))
        result.network = net
        result.regions = regions
        result.phases = phases
    else:
        result = net
    return result
