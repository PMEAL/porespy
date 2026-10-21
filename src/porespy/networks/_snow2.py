import logging
import warnings

import numpy as np

from porespy.filters import snow_partitioning, snow_partitioning_parallel
from porespy.tools import Results, get_edt

from ._funcs import add_boundary_regions, label_boundaries, label_phases
from ._getnet_orig import regions_to_network
from ._getnet_para import regions_to_network_parallel

__all__ = ["snow2", "_parse_pad_width"]


edt = get_edt()
logger = logging.getLogger(__name__)


def estimate_overlap_and_chunk(im):
    divs = [2 for i in range(im.ndim)]

    shape = []
    for i in range(im.ndim):
        shape.append(divs[i] * (im.shape[i] // divs[i]))

    if tuple(shape) != im.shape:
        for i in range(im.ndim):
            im = im.swapaxes(0, i)
            im = im[: shape[i], ...]
            im = im.swapaxes(i, 0)

    chunk_shape = (np.array(shape) / np.array(divs)).astype(int)
    dt = edt((im > 0))
    overlap = dt.max()

    return overlap, chunk_shape


def _parse_parallel_kw(parallel_kw, parallel_extraction_kw=None):
    r"""
    Parse ``parallel_kw`` into ``(watershed_kw, extraction_kw)``.

    Supports the legacy flat style (``{'divs': 2, ...}``) for backward
    compatibility, and the new nested style
    (``{'watershed': {...}, 'extraction': {...}}``).
    """
    _NESTED_KEYS = {'watershed', 'extraction'}

    if parallel_kw is None:
        watershed_kw = None
        extraction_kw = None
    elif _NESTED_KEYS & parallel_kw.keys():
        # New nested style: 'watershed' and/or 'extraction' sub-dicts
        watershed_kw = parallel_kw.get('watershed', None)
        extraction_kw = parallel_kw.get('extraction', None)
    else:
        # Legacy flat style: entire dict is forwarded to the watershed step
        watershed_kw = parallel_kw
        extraction_kw = None

    if parallel_extraction_kw is not None:
        warnings.warn(
            "parallel_extraction_kw is deprecated; pass extraction settings via "
            "parallel_kw={'extraction': {...}} instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        extraction_kw = parallel_extraction_kw

    return watershed_kw, extraction_kw


def snow2(
    phases,
    phase_alias=None,
    boundary_width=3,
    accuracy='standard',
    voxel_size=1,
    sigma=0.4,
    r_max=4,
    peaks=None,
    porosity_map=None,
    parallel_kw={},
    parallel_extraction_kw=None,
):
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

        ========== ============================================================
        Value      Description
        ========== ============================================================
        'standard' Computes the surface areas and perimeters by simply counting
                   voxels. This is *much* faster but does not properly account
                   for the rough voxelated nature of the surfaces.
        'high'     Computes surface areas using the marching cube method, and
                   perimeters using the fast marching method. These are
                   substantially slower but better account for the voxelated
                   nature of the images.
        ========== ============================================================

    voxel_size : tuple (default = (1, 1, 1))
        The resolution of the image, expressed as the length of the sides of a
        voxel, so the volume of a voxel would be the product of **voxel_size**
        coords.
    r_max : int
        The radius of the spherical structuring element to use in the
        Maximum filter stage that is used to find peaks. The default is 4.
    sigma : float
        The standard deviation of the Gaussian filter used in step 1. The
        default is 0.4.  If 0 is given then the filter is not applied.
    peaks : ndarray, optional
        Optionally, it is possible to supply an array containing peaks, which
        are used as markers in the watershed segmentation. If a boolean array
        is received (``True`` indicating peaks), then connected-component labeling
        with cubic connectivity is used. If an integer array is
        received then it is assumed the peaks have already been labelled.
        This allows for comparison of peak finding algorithms for instance.
        If this argument is provided, then ``r_max`` and ``sigma`` are ignored
        since these are specfically used in the peak finding process. This
        array should contain peaks for all phases, and they are masked by
        the ``phases`` argument. If ``peaks`` are provided the parallelization
        is disabled.
    parallel_kw : dict or None
        Controls parallelization for both the watershed and network extraction
        steps. Pass ``None`` to run everything serially.

        **Nested style** (recommended): use ``'watershed'`` and/or
        ``'extraction'`` sub-dicts to configure each step independently::

            parallel_kw = {
                'watershed':  {'divs': 4, 'overlap': None, 'cores': None},
                'extraction': {'threads': 4},
            }

        Omitting a sub-key disables that step's parallelism. An empty sub-dict
        (``{}``) uses defaults for that step.

        **Legacy flat style** (backward compatible): a dict containing only
        watershed keys (``'divs'``, ``'overlap'``, ``'cores'``) is forwarded
        directly to the watershed step and leaves extraction serial::

            parallel_kw = {'divs': 4}  # equivalent to {'watershed': {'divs': 4}}

        Watershed sub-dict keys:

        ========== ============================================================
        Key        Description
        ========== ============================================================
        'divs'     Number of divisions per axis (scalar or list).
        'overlap'  Overlap between chunks; inferred automatically if omitted.
        'cores'    Number of worker processes (``None`` = all available).
        ========== ============================================================

        Extraction sub-dict keys (3D only, requires ``pyedt``):

        ========== ============================================================
        Key        Description
        ========== ============================================================
        'threads'  Number of numba threads (defaults to ~half available cores).
        ========== ============================================================

    parallel_extraction_kw : dict or None, optional
        *Deprecated.* Use ``parallel_kw={'extraction': {...}}`` instead.
        Kept for backward compatibility; a ``DeprecationWarning`` is raised
        when this parameter is used.

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
    <https://porespy.org/examples/networks/reference/snow2.html>`__
    to view online example.

    """
    # Parallel snow does not accept peaks, so if they are provided,
    # disable parallelization
    phases = phases.astype(int)
    watershed_kw, extraction_kw = _parse_parallel_kw(parallel_kw, parallel_extraction_kw)
    if phase_alias is not None:
        vals = phase_alias.keys()
    else:
        vals = np.unique(phases)
        vals = vals[vals > 0]
    if peaks is not None:
        watershed_kw = None
    regions = None
    for i in vals:
        logger.info(f"Processing phase {i}...")
        phase = phases == i
        pk = None if peaks is None else peaks*phase
        overlap, chunk = estimate_overlap_and_chunk(phase)
        # TODO: this may not be the overlap the user provides!
        if (overlap > (chunk//2 - 1)).any():
            watershed_kw = None
            logger.warning("Disabling paralelization as overlap exceeds than chunk size.")
        if watershed_kw is not None:
            snow = snow_partitioning_parallel(
                im=phase,
                sigma=sigma,
                r_max=r_max,
                parallel_kw=watershed_kw,
            )
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
        if porosity_map is not None:
            porosity_map = np.pad(porosity_map, pad_width=boundary_width, mode='edge')
    # Perform actual extractcion on all regions
    if extraction_kw is None:
        net = regions_to_network(
            regions,
            phases=phases,
            accuracy=accuracy,
            voxel_size=voxel_size,
            porosity_map=porosity_map,
        )
    else:
        if regions.ndim != 3:
            raise Exception("Parallel network extraction is only supported for 3D images")
        vs = _normalize_voxel_size(voxel_size, regions.ndim)
        net = regions_to_network_parallel(
            regions,
            phases=phases,
            accuracy=accuracy,
            voxel_size=vs,
            porosity_map=porosity_map,
            **extraction_kw,
        )
    # If image is multiphase, label pores/throats accordingly
    if phases.max() > 1:
        phase_alias = _parse_phase_alias(phase_alias, phases)
        net = label_phases(net, alias=phase_alias)
    # If boundaries were added, label them accordingly
    if np.any(boundary_width):
        W = boundary_width.flatten()
        L = ['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'][:phases.ndim*2]
        L = [L[i]*int(W[i] > 0) for i in range(len(L))]
        L = np.reshape(L, shape=boundary_width.shape)
        net = label_boundaries(net, labels=L)
    result = Results()
    result.network = net
    result.regions = regions
    result.phases = phases
    return result


def _normalize_voxel_size(voxel_size, ndim):
    r"""Widen ``voxel_size`` to an ``ndim``-tuple of floats."""
    if np.isscalar(voxel_size):
        return (float(voxel_size),) * ndim
    return tuple(float(v) for v in voxel_size)


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

    return np.array(tmp, dtype=int)
