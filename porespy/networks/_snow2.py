import numpy as np
from porespy.networks import regions_to_network
from porespy.networks import add_boundary_regions
from porespy.networks import label_phases, label_boundaries
from porespy.filters import snow_partitioning, snow_partitioning_parallel
from porespy.tools import Results
from loguru import logger


def snow2(phases,
          phase_alias=None,
          boundary_width=3,
          accuracy='standard',
          voxel_size=1,
          sigma=0.4,
          r_max=4,
          peaks=None,
          parallelization={},):
    r"""
    Applies the SNOW algorithm to each phase indicated in ``phases``.

    This function is a combination of ``snow`` [1]_, ``snow_dual`` [2]_,
    ``snow_n`` [3]_, and ``snow_parallel`` [4]_ from previous versions.

    Parameters
    ----------
    phases : ndarray
        An image indicating the phase(s) of interest. A watershed is
        produced for each integer value in ``phases`` (except 0's). These
        are then combined into a single image and one network is extracted
        using ``regions_to_network``.
    phase_alias : dict
        A mapping between integer values in ``phases`` and phase name
        used to add labels to the network. For instance, asssuming a
        two-phase image, ``{1: 'void', 2: 'solid'}`` will result in the
        labels ``'pore.void'`` and ``'pore.solid'``, as well as
        ``'throat.solid_void'``, ``'throat.solid_solid'``, and
        ``'throat.void_void'``. If not provided, aliases are assumed to be
        ``{1: 'phase1', 2: 'phase2, ...}``.  Phase labels can also be
        applied afterward using ``label_phases``.
    boundary_width : depends
        Number of voxels to add to the beginning and end of each axis.
        This argument can either be a scalar or a list. If a scalar is
        passed, it will be applied to the beginning and end of all axes.
        If a list, you can specify the number of voxels for each axis
        individually. Here are some examples:

            - [0, 3, 0]: 3 voxels only applied to the y-axis.

            - [0, [0, 3], 0]: 3 voxels only applied to the end of y-axis.

            - [0, [3, 0], 0]: 3 voxels only applied to the beginning of y-axis.

        The default is to add 3 voxels on both ends of all axes. For each
        boundary width that is not 0, a label will automatically be
        applied indicating which end of which axis (i.e. ``'xmin'`` and
        ``'xmax'``).
    accuracy : string
        Controls how accurately certain properties are calculated during
        the analysis of regions in the ``regions_to_network`` function.
        Options are:

            - 'standard' (default)
                Computes the surface areas and perimeters by simply
                counting voxels. This is *much* faster but does not
                properly account for the rough voxelated nature
                of the surfaces.

            - 'high'
                Computes surface areas using the marching cube
                method, and perimeters using the fast marching method. These
                are substantially slower but better account for the
                voxelated nature of the images.

    voxel_size : scalar (default = 1)
        The resolution of the image, expressed as the length of one side
        of a voxel, so the volume of a voxel would be **voxel_size**-cubed.
    r_max : int
        The radius of the spherical structuring element to use in the
        Maximum filter stage that is used to find peaks. The default is 4.
    sigma : float
        The standard deviation of the Gaussian filter used in step 1. The
        default is 0.4.  If 0 is given then the filter is not applied.
    peaks : ndarray, optional
        Optionally, it is possible to supply an array containing peaks, which
        are used as markers in the watershed segmentation. If a boolean array
        is received (``True`` indicating peaks), then ``scipy.ndimage.label``
        with cubic connectivity is used to label them. If an integer array is
        received then it is assumed the peaks have already been labelled.
        This allows for comparison of peak finding algorithms for instance.
        If this argument is provided, then ``r_max`` and ``sigma`` are ignored
        since these are specfically used in the peak finding process. This
        array should contain peaks for all phases, and they are masked by
        the ``phases`` argument. If ``peaks`` are provided the parallelization
        is disabled.
    parallelization : dict
        The arguments for controlling the parallelization of the watershed
        function are rolled into this dictionary, otherwise the function
        signature would become too complex. Refer to the docstring of
        ``snow_partitioning_parallel`` for complete details. If no values
        are provided then the defaults for that function are used here.
        To disable parallelization pass ``parallel=None``, which will
        invoke the standard ``snow_partitioning`` or ``snow_partitioning_n``.
        If ``peaks`` are provided the parallelization is disabled.

    Returns
    -------
    network : Results object
        A custom object is returned with the following data added as attributes:

        - 'phases'
            The original ``phases`` image with any padding applied

        - 'regions'
            The watershed segmentation of the image, including boundary
            regions if padding was applied

        - 'network'
            A dictionary containing all the extracted network properties in
            OpenPNM format ('pore.coords', 'throat.conns', etc).

    References
    ----------
    .. [1] Gostick JT. Versatile and efficient pore network extraction
       method using marker-based watershed segmentation. Phys. Rev. E. 96,
       023307 (2017)
    .. [2] Khan ZA, Tranter TG, Agnaou M, Elkamel A, and Gostick JT, Dual
       network extraction algorithm to investigate multiple transport
       processes in porous materials: Image-based modeling of pore and
       grain-scale processes. Computers and Chemical Engineering. 123(6),
       64-77 (2019)
    .. [3] Khan ZA, García-Salaberri PA, Heenan T, Jervis R, Shearing P,
       Brett D, Elkamel A, Gostick JT, Probing the structure-performance
       relationship of lithium-ion battery cathodes using pore-networks
       extracted from three-phase tomograms. Journal of the
       Electrochemical Society. 167(4), 040528 (2020)
    .. [4] Khan ZA, Elkamel A, Gostick JT, Efficient extraction of pore
       networks from massive tomograms via geometric domain decomposition.
       Advances in Water Resources. 145(Nov), 103734 (2020)

    Examples
    --------
    `Click here
    <https://porespy.org/examples/networks/reference/snow2.html>`_
    to view online example.

    """
    # Parallel snow does not accept peaks, so if they are provided,
    # disable parallelization
    phases = phases.astype(int)
    if phase_alias is not None:
        vals = phase_alias.keys()
    else:
        vals = np.unique(phases)
        vals = vals[vals > 0]
    if peaks is not None:
        parallelization = None
    regions = None
    for i in vals:
        logger.info(f"Processing phase {i}...")
        phase = phases == i
        pk = None if peaks is None else peaks*phase
        if parallelization is not None:
            snow = snow_partitioning_parallel(
                im=phase, sigma=sigma, r_max=r_max, **parallelization)
        else:
            snow = snow_partitioning(im=phase, sigma=sigma, r_max=r_max,
                                     peaks=pk)
        if regions is None:
            regions = np.zeros_like(snow.regions, dtype=int)
        # Note: Using snow.regions > 0 here instead of phase is needed to
        # handle a bug in snow_partitioning, see issue #169 and #430
        regions += snow.regions + regions.max()*(snow.regions > 0)
    if phases.shape != regions.shape:
        logger.warning(f"Image was cropped to {regions.shape} during watershed")
        for ax in range(phases.ndim):
            phases = np.swapaxes(phases, 0, ax)
            phases = phases[:regions.shape[ax], ...]
            phases = np.swapaxes(phases, 0, ax)
    # Inspect and clean-up boundary_width argument
    boundary_width = _parse_pad_width(boundary_width, phases.shape)
    # If boundaries were specified, pad the images accordingly
    if np.any(boundary_width):
        regions = add_boundary_regions(regions, pad_width=boundary_width)
        phases = np.pad(phases, pad_width=boundary_width, mode='edge')
    # Perform actual extractcion on all regions
    net = regions_to_network(
        regions, phases=phases, accuracy=accuracy, voxel_size=voxel_size)
    # If image is multiphase, label pores/throats accordingly
    if phases.max() > 1:
        phase_alias = _parse_phase_alias(phase_alias, phases)
        net = label_phases(net, alias=phase_alias)
    # If boundaries were added, label them accordingly
    if np.any(boundary_width):
        W = boundary_width.flatten()
        L = ['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'][:phases.ndim*2]
        L = [L[i]*int(W[i] > 0) for i in range(len(L))]
        L = np.reshape(L, newshape=boundary_width.shape)
        net = label_boundaries(net, labels=L)
    result = Results()
    result.network = net
    result.regions = regions
    result.phases = phases
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
    ndim = len(shape)
    pad_width = np.atleast_1d(np.array(pad_width, dtype=object))

    if np.size(pad_width) == 1:
        pad_width = np.tile(pad_width.item(), ndim).astype(object)
    if len(pad_width) != ndim:
        raise Exception(f"pad_width must be scalar or {ndim}-element list")

    tmp = []
    for elem in pad_width:
        if np.size(elem) == 1:
            tmp.append(np.tile(np.array(elem).item(), 2))
        elif np.size(elem) == 2 and np.ndim(elem) == 1:
            tmp.append(elem)
        else:
            raise Exception("pad_width components can't have 2+ elements")

    return np.array(tmp)
