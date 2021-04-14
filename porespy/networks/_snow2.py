import numpy as np
from porespy.networks import regions_to_network
from porespy.networks import add_boundary_regions
from porespy.networks import label_phases, label_boundaries
from porespy.filters import snow_partitioning, snow_partitioning_parallel
from collections import namedtuple


def snow2(phases,
          phase_alias=None,
          boundary_width=3,
          boundary_labels=None,
          accuracy='standard',
          voxel_size=1,
          return_all=False,
          parallelization={},):
    r"""
    Applies the SNOW algorithm to each phase indicated in ``phases``.

    This function is a combination of ``snow`` [1]_, ``snow_dual`` [2]_,
    ``snow_n`` [3]_, and ``snow_parallel`` [4]_ from previous versions.

    Parameters
    ----------
    phases : ND-image
        An image indicating the phase(s) of interest. A watershed is produced
        for each integer value in ``phases`` (except 0's). These are then
        combined into a single image and one network is extracted using
        ``regions_to_network``.
    phase_alias : dict
        A mapping between integer values in ``phases`` and phase name, used
        to add labels to the network. For instance, asssuming a two-phase
        image, ``{1: 'void', 2: 'solid'}`` will result in the labels
        ``'pore.void'`` and ``'pore.solid'``, as well as
        ``'throat.solid_void'``, ``'throat.solid_solid'``, and
        ``'throat.void_void'``. If not provided, alias is assumed as
        ``{1: 'phase1', 2: 'phase2, ...}``.  Phase labels can also be applied
           afterward using ``label_phases``.
    boundary_width : depends
        Number of voxels to add to the beginning and end of each axis. This
        argument is handled the same as ``pad_width`` in the ``np.pad``
        function. A scalar adds the same amount to the beginning and end of
        each axis. ``[A, B]`` adds A to the beginning of each axis and B to the
        ends.  ``[[A, B], ..., [C, D]]`` adds A to the beginning and B to the
        end of the first axis, and so on. The default is to add 3 voxels on
        both ends of all axes.
    boundary_labels : list of lists
        A 3-element list with each element containing a pair of strings
        indicating the label to apply to the beginning and end of each axis.
        For instance, ``[['left', 'right'], ['front', 'back'],
        ['top', 'bottom']]`` will apply the label ``'left'`` to all pores
        with the minimum x-coordinate, and ``'right'`` to the pores with the
        maximum x-coordinate, and so on.  Boundary labels can also be applied
        afterward using ``label_boundaries``.
    accuracy : string
        Controls how accurately certain properties are calculated during the
        analysis of regions in the ``regions_to_network`` function.
        Options are:

        'standard' (default)
            Computes the surface areas and perimeters by simply counting
            voxels. This is *much* faster but does not properly account
            for the rough voxelated nature of the surfaces.
        'high'
            Computes surface areas using the marching cube method, and
            perimeters using the fast marching method. These are substantially
            slower but better account for the voxelated nature of the images.

    voxel_size : scalar (default = 1)
        The resolution of the image, expressed as the length of one side of a
        voxel, so the volume of a voxel would be **voxel_size**-cubed.
    return_all : boolean
        If set to ``True`` a named tuple is returned containing the
        padded ``phases`` image, the pore ``regions``, and a dictionary
        containing the extracted ``network``. The default is ``False``,
        so only the network dictionary is returned.
    parallelization : dict
        The arguments for controlling the parallization of the watershed
        function are rolled into this dictionary, otherwise the function
        signature would become too complex. Refer to the docstring of
        ``snow_partitioning_parallel`` for complete details. If no values
        are provided then the defaults for that function are used.
        To disable parallelization pass ``parallel=None``, which will invoke
        the standard ``snow_partitioning``.

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
       method using marker-based watershed segmentation. Phys. Rev. E. 96,
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
    .. [4] Khan ZA, Elkamel A, Gostick JT, Efficient extraction of pore
       networks from massive tomograms via geometric domain decomposition.
       Advances in Water Resources. 145(Nov), 103734 (2020)
    """
    regions = np.zeros_like(phases, dtype=int)
    for i in range(phases.max()):
        phase = phases == (i + 1)
        if parallelization is not None:
            snow = snow_partitioning_parallel(im=phase,
                                              return_all=True,
                                              sigma=0.4,
                                              r_max=4,
                                              **parallelization)
        else:
            snow = snow_partitioning(im=phase,
                                     return_all=True,
                                     sigma=0.4,
                                     r_max=4)
        # Note: Using snow.regions > 0 here instead of phase is needed to
        # handle a bug in snow_partitioning, see issue #169 and #430
        regions += snow.regions + regions.max()*(snow.regions > 0)
    # Inspect and clean-up boundary_width argument
    boundary_width = _parse_pad_width(boundary_width, phases.shape)
    # If boundaries were specified, pad the images accordingly
    if np.any(boundary_width):
        regions = add_boundary_regions(regions, pad_width=boundary_width)
        phases = np.pad(phases, pad_width=boundary_width, mode='edge')
    # Perform actual extractcion on all regions
    net = regions_to_network(regions,
                             phases=phases,
                             accuracy=accuracy,
                             voxel_size=voxel_size)
    # If image is multiphase, label pores/throats accordingly
    if phases.max() > 1:
        phase_alias = _parse_phase_alias(phase_alias, phases)
        net = label_phases(net, alias=phase_alias)
    # If boundaries were added, label them accordingly
    if np.any(boundary_width):
        boundary_labels = _parse_boundary_labels(boundary_labels,
                                                 pad_width=boundary_width)
        net = label_boundaries(net, labels=boundary_labels)
    if return_all:  # Collect result in a tuple for return
        result = namedtuple('snow2', ('network', 'regions', 'phases'))
        result.network = net
        result.regions = regions
        result.phases = phases
    else:
        result = net
    return result


def _parse_phase_alias(alias, phases):
    r"""
    """
    if alias is None:
        alias = {i+1: 'phase' + str(i+1) for i in range(phases.max())}
    for i in range(phases.max()):
        if i+1 not in alias.keys():
            alias[i+1] = 'phase'+str(i+1)
    return alias


def _parse_pad_width(pad_width, shape):
    r"""
    """
    shape = np.array(shape)
    pw = np.array(pad_width)
    # Deal with integer value
    if pw.size == 1:
        pad_width = [[pad_width, pad_width]]*len(shape)
        pw = np.array(pad_width)
    elif pw.size == 2:
        pad_width = [pad_width]*len(shape)
        pw = np.array(pad_width)
    elif (pw.size == 3) and (shape.size == 3):
        pad_width = [pad_width]*2
        pw = np.array(pad_width).T
    elif (pw.size == 3) and (shape.size == 2):
        raise Exception(f'Not sure how to interpret {pad_width} on a 2D image')
    return pw.squeeze()


def _parse_boundary_labels(boundary_labels, pad_width):
    r"""
    """
    if boundary_labels is None:
        boundary_labels = [['left', 'right'],
                           ['front', 'back'],
                           ['bottom', 'top']]
    bl = []
    for r in range(pad_width.shape[1]):
        temp = []
        for c in [0, 1]:
            if pad_width[r, c] > 0:
                temp.append(boundary_labels[r][c])
        bl.append(temp)
    return bl
