import inspect
import logging

import numpy as np
import numpy.typing as npt
import scipy.ndimage as spim
import scipy.spatial as sptl
import scipy.stats as spst
from deprecated import deprecated
from numba import njit
from scipy import fft as sp_ft
from skimage.measure import regionprops
from skimage.morphology import ball, disk, skeletonize, footprint_rectangle

from porespy.generators import faces
from porespy.filters import (
    local_thickness,
    pc_to_seq,
    find_closed_pores,
    find_surface_pores,
)
from porespy.tools import (
    Results,
    _check_for_singleton_axes,
    get_edt,
    get_slices_slabs,
    get_tqdm,
    settings,
)


__all__ = [
    "bond_number",
    "boxcount",
    "chord_counts",
    "chord_length_distribution",
    "find_h",
    "find_porosity_threshold",
    "is_percolating",
    "lineal_path_distribution",
    "pore_size_distribution",
    "radial_density_distribution",
    "porosity",
    "find_porosity_threshold",
    "porosity_profile",
    "satn_profile",
    "two_point_correlation",
    "percolating_porosity",
    "phase_fraction",
    "pc_map_to_pc_curve",
    "percolating_porosity",
    "phase_fraction",
    "pore_size_distribution",
    "porosity",
    "porosity_by_type",
    "porosity_profile",
    "radial_density_distribution",
    "satn_profile",
    "two_point_correlation",
]


edt = get_edt()
tqdm = get_tqdm()
logger = logging.getLogger(__name__)
strel = {
    2: {"min": disk(1), "max": footprint_rectangle((3, 3))},
    3: {"min": ball(1), "max": footprint_rectangle((3, 3, 3))}
}


def porosity_by_type(im, conn='min'):
    r"""
    Computes different types of porosity in an image including total, closed, and
    surface

    Parameters
    ----------
    im : ndarray
        An image of the void space with 1 (or `True`) representing the phase of
        interest.  The bulk volume will be computed as the sum of 0's and 1's.
    conn : str
        Can be either `'min'` or `'max'` and controls the shape of the structuring
        element used to determine voxel connectivity.  The default if `'min'` which
        imposes the strictest criteria, so that voxels must share a face to be
        considered connected.

    Returns
    -------
    results
        A dataclass-like object with the following attributes:

        ==========  ================================================================
        Attribute   Description
        ==========  ================================================================
        `total`     The total fraction of the image which is void phase
        `closed`    The fraction of the image which consists isolated voids
        `surface`   The fraction of the image which are pores only the surfaces
        ==========  ================================================================

            Examples
    --------
    `Click here
    <https://porespy.org/examples/metrics/reference/porosity_by_type.html>`_
    to view online example.
    """
    Vb = np.sum((im == 1) + (im == 0), dtype=np.float64)
    temp = im == 1
    eps_total = np.sum(temp, dtype=np.float64)/Vb
    temp = find_closed_pores(im, conn=conn)
    eps_closed = np.sum(temp, dtype=np.float64)/Vb
    temp = find_surface_pores(im, conn=conn)
    eps_surface = np.sum(temp, dtype=np.float64)/Vb

    r = Results()
    r.total = eps_total
    r.closed = eps_closed
    r.surface = eps_surface
    return r


def is_percolating(im, axis=None, inlets=None, outlets=None, conn='min'):
    r"""
    Determines if a percolating path exists across the domain (in the specified
    direction) or between given inlets and outlets.

    Parameters
    ----------
    im : ndarray
        Image of the void space with `True` indicating void space.
    axis : int
        The axis along which percolation is checked. If `None` (default) then
        percolation is checked in all dimensions.
    conn : str
        Can be either `'min'` or `'max'` and controls the shape of the structuring
        element used to determine voxel connectivity.  The default if `'min'` which
        imposes the strictest criteria, so that voxels must share a face to be
        considered connected.

    Returns
    -------
    percolating : bool or list of bools
        A boolean value indicating if the domain percolates in the given direction.
        If `axis=None` then all directions are checked and the result is returned
        as a list like `[True, False, True]` indicating that the domain percolates
        in the `x` and `z` directions, but not `y`.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/metrics/reference/is_percolating.html>`_
    to view online example.

    """
    if (inlets is not None) and (outlets is not None):
        pass
    elif axis is not None:
        im = np.swapaxes(im, 0, axis) == 1
        inlets = np.zeros_like(im)
        inlets[0, ...] = True
        inlets *= im
        outlets = np.zeros_like(im)
        outlets[-1, ...] = True
        outlets *= im
    else:
        ans = []
        for ax in range(im.ndim):
            ans.append(is_percolating(im, axis=ax, conn=conn))
        return ans

    labels, N = spim.label(im, structure=strel[im.ndim][conn])
    a = np.unique(labels[inlets])
    a = a[a > 0]
    b = np.unique(labels[outlets])
    b = b[b > 0]
    hits = np.isin(a, b)
    return np.any(hits)


def find_porosity_threshold(im, axis=0, dt=None, conn="min"):
    r"""
    Finds the porosity of the image at the percolation threshold

    This function progressively dilates the solid and reports the porosity at the
    step just before there are no percolating paths (in the specified direction)

    Parameters
    ----------
    im : ndarray
        Image of the void space with `True` indicating void space
    axis : int
        The axis along which percolation is checked
    dt : ndarray
        The distance transform of the void space. If not provide it will be computed
        so providing one can save time if it is available.
    conn : str
        Can be either `'min'` or `'max'` and controls the shape of the structuring
        element used to determine voxel connectivity.  The default if `'min'` which
        imposes the strictest criteria, so that voxels must share a face to be
        considered connected.

    Returns
    -------
    results
        A Results object with the following attributes:

        ================ ===========================================================
        Attribute        Description
        ================ ===========================================================
        eps_orig         The total porosity of the original image, including closed
                         and surface pores
        eps_orig_perc    The percolating porosity of the original image (i.e. with
                         closed and surface pores filled)
        eps_thresh       The total porosity of the image just before the percolation
                         threshold was reached (i.e at the point where one
                         additional dilation would result in no connected void
                         space.)
        eps_thresh_perc  The percolating porosity (with closed and surface pores
                         filled) just before the percolation threshold was reached
                         (i.e at the point where one additional dilation would
                         result in no connected void space.)
        eps_thresh_post  The total porosity after the percolation threshold was
                         reached (i.e. one step *after* the dilation which
                         resulted in no connected pore space)
        R                The threshold to apply to the distance transform to
                         obtain the percolating image (i.e. im = dt >= R)
        ================ ===========================================================

    Examples
    --------
    `Click here
    <https://porespy.org/examples/metrics/reference/find_porosity_threshold.html>`_
    to view online example.

    """
    if axis is None:
        raise Exception("axis must be specified")

    def _check_percolation(dt, R, step, axis, conn):
        while True:
            im2 = dt >= R
            check = np.array(is_percolating(im2, axis=axis, conn=conn))
            if not np.all(check):
                break
            R += step
        return R

    if dt is None:
        dt = edt(im)

    # Take large steps first, then medium and small steps to find final value faster
    R = _check_percolation(dt, R=1, step=10, axis=axis, conn=conn)
    R = _check_percolation(dt, R=max(1, R - 10), step=4, axis=axis, conn=conn)
    R = _check_percolation(dt, R=max(1, R - 4), step=1, axis=axis, conn=conn)

    im2 = dt >= (R - 1)
    im3 = dt >= R

    r = Results()
    r.eps_orig = porosity(im)
    r.eps_orig_perc = percolating_porosity(im, axis=axis, conn=conn)
    r.eps_thresh = porosity(im2)
    r.eps_thresh_perc = percolating_porosity(im2, axis=axis, conn=conn)
    r.eps_thresh_post = porosity(im3)
    r.R = R - 1
    return r


def percolating_porosity(im, axis=0, inlets=None, outlets=None, conn="min"):
    r"""
    Finds volume fraction of void space which belongs to percolating paths
    across the domain in the direction specified.

    Parameters
    ----------
    im : ndarray
        Image of the void space with `True` indicating void space
    axis : int
        The axis along which percolation is checked
    conn : str
        Can be either `'min'` or `'max'` and controls the shape of the structuring
        element used to determine voxel connectivity.  The default if `'min'` which
        imposes the strictest criteria, so that voxels must share a face to be
        considered connected.
    inlets, outlets : ndarrays, optional
        Boolean arrays indicating the locations of the inlets and outlets. These
        are useful if the domain is not cubic or if special inlet and outlet
        locations are desired.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/metrics/reference/percolating_porosity.html>`_
    to view online example.
    """
    se = strel[im.ndim][conn]
    if (inlets is None) and (outlets is None):
        inlets = faces(im.shape, inlet=axis)
        outlets = faces(im.shape, outlet=axis)
    labels, N = spim.label(im, structure=se)
    a = np.unique(labels[inlets])
    a = a[a > 0]
    b = np.unique(labels[outlets])
    b = b[b > 0]
    hits = np.intersect1d(a, b)
    im3 = np.isin(labels, hits)
    eps = im3.sum()/im3.size
    return eps


def boxcount(im, bins=10):
    r"""
    Calculates the fractal dimension of an image using the tiled box counting
    method [1a]_

    Parameters
    ----------
    im : ndarray
        A boolean image of the porous material with `True` values indicating the
        phase of interest.
    bins : int or array_like, optional
        The number of box sizes to use. The default is 10 sizes logarithmically
        spaced between 1 and ``min(im.shape)``. If an array is provided, this is
        used directly.

    Returns
    -------
    results : dataclass-like
        An object possessing the following attributes:

        ========== =================================================================
        Attribute  Description
        ========== =================================================================
        size       An array containing the specific box sizes used
        count      An array containing the number of boxes of each size that
                   contain both solid and void
        slope      The gradient of ``count``. This has the same number of elements
                   as ``count``.
        ========== =================================================================

    References
    ----------
    .. [1a] Read more about box counting on `Wikipedia
       <https://en.wikipedia.org/wiki/Box_counting>`_

    Examples
    --------
    `Click here
    <https://porespy.org/examples/metrics/reference/boxcount.html>`_
    to view online example.

    """
    im = np.array(im, dtype=bool)

    if len(im.shape) != 2 and len(im.shape) != 3:
        raise Exception("Image must be 2-dimensional or 3-dimensional")

    if isinstance(bins, int):
        Ds = np.unique(np.logspace(1, np.log10(min(im.shape)), bins).astype(int))
    else:
        Ds = np.array(bins).astype(int)

    N = []
    desc = inspect.currentframe().f_code.co_name  # Get current func name
    for d in tqdm(Ds, desc=desc, **settings.tqdm):
        result = 0
        for i in range(0, im.shape[0], d):
            for j in range(0, im.shape[1], d):
                if len(im.shape) == 2:
                    temp = im[i:i + d, j:j + d]
                    result += np.any(temp)
                    result -= np.all(temp)
                else:
                    for k in range(0, im.shape[2], d):
                        temp = im[i:i + d, j:j + d, k:k + d]
                        result += np.any(temp)
                        result -= np.all(temp)
        N.append(result)
    slope = -1 * np.gradient(np.log(np.array(N)), np.log(Ds))
    data = Results()
    data.size = Ds
    data.count = N
    data.slope = slope
    return data


def porosity(im, mask=None, fill_closed=False, fill_surface=False):
    r"""
    Calculates the porosity of an image assuming 1's are void space and 0's
    are solid phase.

    Parameters
    ----------
    im : ndarray
        Image of the void space with 1's indicating void phase (or ``True``)
        and 0's indicating the solid phase (or ``False``). All other values
        are ignored (see Notes).
    mask : ndarray
        An image the same size as `im` with `True` values indicting the domain. This
        argument is optional, but can be provided for images that don't fill the
        entire array, like cylindrical cores.  Note that setting values in `im`
        to 2 will also exclude them from consideration so provides the same effect
        as `mask`, but providing a `mask` is usually much easier.
    fill_closed : bool (default = `False`)
        A flag to indicate if closed pores (not connected to any image boundary)
        should be filled or not before computing the porosity.
    fill_surface : bool (default = `False`)
        A flag to indicate if surface pores connected only to one surface should be
        filled or not before computing the porosity.

    Returns
    -------
    porosity : float
        Calculated as the sum of all 1's divided by the sum of all 1's and 0's.

    See Also
    --------
    phase_fraction
    find_outer_region

    Notes
    -----
    This function assumes void is represented by 1 and solid by 0, and all
    other values are ignored.  This is useful, for example, for images of
    cylindrical cores, where all voxels outside the core are labelled with 2.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/metrics/reference/porosity.html>`__
    to view online example.

    """
    im = im.copy()
    if mask is not None:
        im = np.array(im, dtype=np.int64) * mask
    if fill_closed or fill_surface:
        closed_pores = im * find_closed_pores(im)
        surface_pores = im * find_surface_pores(im) * ~closed_pores
        if fill_closed:
            im[closed_pores] = 0
        if fill_surface:
            im[surface_pores] = 0
    Vp = np.sum(im == 1, dtype=np.int64)
    Vs = np.sum(im == 0, dtype=np.int64)
    e = Vp / (Vs + Vp)
    return e


def porosity_profile(im, axis=0, span=1, mode="tile", voxel_size=None):
    r"""
    Computes the porosity profile along the specified axis

    Parameters
    ----------
    im : ndarray
        The volumetric image for which to calculate the porosity profile.  All
        voxels with a value of 1 (or ``True``) are considered as void.
    axis : int
        The axis along which to profile should be measured
    span : int (Default = 1)
        The thickness of layers to include in the moving average calculation.
    mode : str (Default = 'tile')
        How the moving average should be applied. Options are:

        ======== ==============================================================
        mode     description
        ======== ==============================================================
        'tile'   The average is computed for discrete non-overlapping
                 tiles of a size given by ``span``
        'slide'  The average is computed in a moving window starting at
                 ``span/2`` and sliding by a single voxel. This method
                 provides more data points but is slower.
        ======== ==============================================================

    voxel_size : scalar, optional
        If given, the returned ``position`` values are scaled by this factor.
        If ``None`` (default), positions are returned in voxel units.

    Returns
    -------
    results : dataclass
        Results is a custom class with the following attributes:

        ============= =========================================================
        Attribute     Description
        ============= =========================================================
        position      The position along the given axis at which porosity
                      values are computed, corresponding to the middle of each
                      slice, whether the mode was `'tile'` or `'slide'`.
                      The units are in voxels unless ``voxel_size`` is given.
        porosity      The local porosity value at each position.
        ============= =========================================================

    Returns
    -------
    result : 1D-array
        A 1D-array of porosity along the specified axis

    Examples
    --------
    `Click here
    <https://porespy.org/examples/metrics/reference/porosity_profile.html>`__
    to view online example.

    """
    if axis >= im.ndim:
        raise Exception("axis out of range")
    slices = get_slices_slabs(im=im, axis=axis, span=span, mode=mode)
    eps = np.zeros(len(slices))
    z = np.zeros_like(eps)
    for i, s in enumerate(slices):
        num = (im[s] == 1).sum(dtype=np.float64)
        denom = ((im[s] == 1) + (im[s] == 0)).sum(dtype=np.float64)
        eps[i] = num / denom
        z[i] = (s[axis].start + s[axis].stop) / 2
    if voxel_size is not None:
        z = z * voxel_size
    results = Results()
    results.position = z
    results.porosity = eps
    return results


def radial_density_distribution(dt, bins=10, log=False, voxel_size=1):
    r"""
    Computes radial density function by analyzing the histogram of voxel
    values in the distance transform.  This function is defined by
    Torquato [1b]_ as:

        .. math::

            \int_0^\infty P(r)dr = 1.0

    where *P(r)dr* is the probability of finding a voxel at a lying at a radial
    distance between *r* and *dr* from the solid interface.  This is equivalent
    to a probability density function (*pdf*)

    The cumulative distribution is defined as:

        .. math::

            F(r) = \int_r^\infty P(r)dr

    which gives the fraction of pore-space with a radius larger than *r*. This
    is equivalent to the cumulative distribution function (*cdf*).

    Parameters
    ----------
    dt : ndarray
        A distance transform of the pore space.  Note that it is recommended to apply
        ``find_dt_artifacts`` to this image first, and set potentially
        erroneous values to 0 with ``dt[mask] = 0`` where
        ``mask = porespy.filters.find_dt_artifaces(dt)``.
    bins : int or array_like
        This number of bins (if int) or the location of the bins (if array).
        This argument is passed directly to Scipy's ``histogram`` function so
        see that docstring for more information.  The default is 10 bins, which
        reduces produces a relatively smooth distribution.
    log : boolean
        If ``True`` the size data is converted to log (base-10)
        values before processing.  This can help to plot wide size
        distributions or to better visualize the radii in the small size region.
        Note that you should not anti-log the radii values in the returned
        ``tuple``, since the binning is performed on the logged radii values.
    voxel_size : scalar
        The size of a voxel side in preferred units.  The default is 1, so the
        user can apply the scaling to the returned results after the fact.

    Returns
    -------
    result : Results object
        A custom object with the following data added as named attributes:

        ============== =======================================================
        Attribute      Description
        ============== =======================================================
        *R* or *LogR*  Radius, equivalent to ``bin_centers``
        *pdf*          Probability density function
        *cdf*          Cumulative density function
        *bin_centers*  The center point of each bin
        *bin_edges*    Locations of bin divisions, including 1 more value than
                       the number of bins
        *bin_widths*   Useful for passing to the ``width`` argument of
                       ``matplotlib.pyplot.bar``
        ============== =======================================================

    Notes
    -----
    Torquato refers to this as the *pore-size density function*, and mentions
    that it is also known as the *pore-size distribution function*.  These
    terms are avoided here since they have specific connotations in porous
    media analysis.

    References
    ----------
    .. [1b] Torquato, S. Random Heterogeneous Materials: Mircostructure and
       Macroscopic Properties. Springer, New York (2002) - See page 48 & 292

    Examples
    --------
    `Click here
    <https://porespy.org/examples/metrics/reference/radial_density.html>`__
    to view online example.

    """
    im = np.copy(dt)
    x = im[im > 0].flatten()
    if log:
        x = np.log10(x)
    h = np.histogram(x, bins=bins, density=True)
    h = _parse_histogram(h=h, voxel_size=voxel_size)
    rdf = Results()
    rdf[f"{log * 'Log' + 'R'}"] = h.bin_centers
    rdf.pdf = h.pdf
    rdf.cdf = h.cdf
    rdf.relfreq = h.relfreq
    rdf.bin_centers = h.bin_centers
    rdf.bin_edges = h.bin_edges
    rdf.bin_widths = h.bin_widths
    return rdf


def lineal_path_distribution(im, bins=10, voxel_size=1, log=False):
    r"""
    Determines the probability that a point lies within a certain distance
    of the opposite phase *along a specified direction*

    This relates directly the radial density function defined by Torquato [1c]_,
    but instead of reporting the probability of lying within a stated distance
    to the nearest solid in any direction, it considers only linear distances
    along orthogonal directions.The benefit of this is that anisotropy can be
    detected in materials by performing the analysis in multiple orthogonal
    directions.

    Parameters
    ----------
    im : ndarray
        An image with each voxel containing the distance to the nearest solid
        along a linear path, as produced by ``distance_transform_lin``.
    bins : int or array_like
        The number of bins or a list of specific bins to use
    voxel_size : scalar
        The side length of a voxel.  This is used to scale the chord lengths
        into real units.  Note this is applied *after* the binning, so
        ``bins``, if supplied, should be in terms of voxels, not length units.
    log : boolean
        If ``True`` (default) the size data is converted to log (base-10)
        values before processing.  This can help to plot wide size
        distributions or to better visualize data in the small size region.
        Note that you should not anti-log the radii values in the returned
        ``results``, since the binning is performed on the logged radii values.

    Returns
    -------
    result : Results object
        A custom object with the following data added as named attributes:

        =============== =============================================================
        Description     Attribute
        =============== =============================================================
        *L* or *LogL*   Length, equivalent to ``bin_centers``
        *pdf*           Probability density function
        *cdf*           Cumulative density function
        *relfreq*       Relative frequency chords in each bin.  The sum of all bin
                        heights is 1.0.  For the cumulative relative, use *cdf*
                        which is already normalized to 1.
        *bin_centers*   The center point of each bin
        *bin_edges*     Locations of bin divisions, including 1 more value than
                        the number of bins
        *bin_widths*    Useful for passing to the ``width`` argument of
                        ``matplotlib.pyplot.bar``
        =============== =============================================================

    References
    ----------
    .. [1c] Torquato, S. Random Heterogeneous Materials: Microstructure and
       Macroscopic Properties. Springer, New York (2002)

    Examples
    --------
    `Click here
    <https://porespy.org/examples/metrics/reference/linearl_path_distribution.html>`__
    to view online example.

    """
    x = im[im > 0]
    if log:
        x = np.log10(x)
    h = list(np.histogram(x, bins=bins, density=True))
    h = _parse_histogram(h=h, voxel_size=voxel_size)
    cld = Results()
    cld[f"{log * 'Log' + 'L'}"] = h.bin_centers
    cld.pdf = h.pdf
    cld.cdf = h.cdf
    cld.relfreq = h.relfreq
    cld.bin_centers = h.bin_centers
    cld.bin_edges = h.bin_edges
    cld.bin_widths = h.bin_widths
    return cld


def chord_length_distribution(
    im,
    bins=10,
    log=False,
    voxel_size=1,
    normalization="count",
):
    r"""
    Determines the distribution of chord lengths in an image containing chords.

    Parameters
    ----------
    im : ndarray
        An image with chords drawn in the pore space, as produced by
        ``apply_chords`` or ``apply_chords_3d``.  ``im`` can be either boolean,
        in which case each chord will be identified using ``scipy.ndimage.label``,
        or numerical values in case it is assumed that chords have already been
        identified and labeled. In both cases, the size of each chord will be
        computed as the number of voxels belonging to each labelled region.
    bins : scalar or array_like
        If a scalar is given it is interpreted as the number of bins to use,
        and if an array is given they are used as the bins directly.
    log : boolean
        If ``True`` (default) the size data is converted to log (base-10)
        values before processing.  This can help to plot wide size
        distributions or to better visualize the in the small size region.
        Note that you should not anti-log the radii values in the returned
        ``tuple``, since the binning is performed on the logged radii values.
    normalization : string
        Indicates how to normalize the bin heights.  Options are:

        *'count' or 'number'*
            (default) This simply counts the number of chords in each bin in
            the normal sense of a histogram.  This is the rigorous definition
            according to Torquato [1d]_.

        *'length'*
            This multiplies the number of chords in each bin by the
            chord length (i.e. bin size).  The normalization scheme accounts for
            the fact that long chords are less frequent than shorter chords,
            thus giving a more balanced distribution.

    voxel_size : scalar
        The size of a voxel side in preferred units.  The default is 1, so the
        user can apply the scaling to the returned results after the fact.

    Returns
    -------
    result : Results object
        A custom object with the following data added as named attributes:

        =============== =============================================================
        Attribute       Description
        =============== =============================================================
        *L* or *LogL*   Chord length, equivalent to ``bin_centers``
        *pdf*           Probability density function
        *cdf*           Cumulative density function
        *relfreq*       Relative frequency chords in each bin.  The sum of all bin
                        heights is 1.0.  For the cumulative relative, use *cdf*
                        which is already normalized to 1.
        *bin_centers*   The center point of each bin
        *bin_edges*     Locations of bin divisions, including 1 more value than
                        the number of bins
        *bin_widths*    Useful for passing to the ``width`` argument of
                        ``matplotlib.pyplot.bar``
        =============== =============================================================

    References
    ----------
    .. [1d] Torquato, S. Random Heterogeneous Materials: Microstructure and
       Macroscopic Properties. Springer, New York (2002) - See page 45 & 292

    Examples
    --------
    `Click here
    <https://porespy.org/examples/metrics/reference/chord_length_distribution.html>`__
    to view online example.

    """
    x = chord_counts(im)
    if bins is None:
        bins = np.array(range(0, x.max() + 2)) * voxel_size
    x = x * voxel_size
    if log:
        x = np.log10(x)
    if normalization == "length":
        h = list(np.histogram(x, bins=bins, density=False))
        # Scale bin heigths by length
        h[0] = h[0] * (h[1][1:] + h[1][:-1]) / 2
        # Normalize h[0] manually
        h[0] = h[0] / h[0].sum(dtype=np.int64) / (h[1][1:] - h[1][:-1])
    elif normalization in ["number", "count"]:
        h = np.histogram(x, bins=bins, density=True)
    else:
        raise Exception("Unsupported normalization:", normalization)
    h = _parse_histogram(h)
    cld = Results()
    cld[f"{log * 'Log' + 'L'}"] = h.bin_centers
    cld.pdf = h.pdf
    cld.cdf = h.cdf
    cld.relfreq = h.relfreq
    cld.bin_centers = h.bin_centers
    cld.bin_edges = h.bin_edges
    cld.bin_widths = h.bin_widths
    return cld


def pore_size_distribution(im, bins=10, log=True, voxel_size=1):
    r"""
    Calculate a pore-size distribution based on the image produced by the
    ``porosimetry`` or ``local_thickness`` functions.

    Parameters
    ----------
    im : ndarray
        The array of containing the sizes of the largest sphere that overlaps
        each voxel.  Obtained from either ``porosimetry`` or
        ``local_thickness``.
    bins : scalar or array_like
        Either an array of bin sizes to use, or the number of bins that should
        be automatically generated that span the data range.
    log : boolean
        If ``True`` (default) the size data is converted to log (base-10)
        values before processing.  This can help to plot wide size
        distributions or to better visualize the in the small size region.
        Note that you should not anti-log the radii values in the returned
        ``tuple``, since the binning is performed on the logged radii values.
    voxel_size : scalar
        The size of a voxel side in preferred units.  The default is 1, so the
        user can apply the scaling to the returned results after the fact.

    Returns
    -------
    result : Results object
        A custom object with the following data added as named attributes:

        =============== =============================================================
        Attribute       Description
        =============== =============================================================
        *R* or *logR*   Radius, equivalent to ``bin_centers``
        *pdf*           Probability density function
        *cdf*           Cumulative density function
        *satn*          Phase saturation in differential form.  For the cumulative
                        saturation, just use *cfd* which is already normalized to 1.
        *bin_centers*   The center point of each bin
        *bin_edges*     Locations of bin divisions, including 1 more value than
                        the number of bins
        *bin_widths*    Useful for passing to the ``width`` argument of
                        ``matplotlib.pyplot.bar``
        =============== =============================================================

    Examples
    --------
    `Click here
    <https://porespy.org/examples/metrics/reference/pore_size_distribution.html>`__
    to view online example.

    """
    im = im.flatten()
    vals = im[im > 0] * voxel_size
    if log:
        vals = np.log10(vals)
    h = _parse_histogram(np.histogram(vals, bins=bins, density=True))
    cld = Results()
    cld[f"{log * 'Log' + 'R'}"] = h.bin_centers
    cld.pdf = h.pdf
    cld.cdf = h.cdf
    cld.satn = h.relfreq
    cld.bin_centers = h.bin_centers
    cld.bin_edges = h.bin_edges
    cld.bin_widths = h.bin_widths
    return cld


def two_point_correlation_bf(im, spacing=10):
    r"""
    Calculates the two-point correlation function using brute-force (see Notes)

    Parameters
    ----------
    im : ndarray
        The image of the void space on which the 2-point correlation is
        desired.
    spacing : int
        The space between points on the regular grid that is used to
        generate the correlation (see Notes).

    Returns
    -------
    result : Results object
        A custom object with the following data added as named attributes:

        'distance'
            The distance between two points. The distance values are binned
            as: ``bins = range(start=0, stop=np.amin(im.shape)/2, stride=spacing)``

        'probability'
            The probability that two points of the stated separation distance
            are within the same phase

    Notes
    -----
    The brute-force approach means overlaying a grid of equally spaced points
    onto the image, calculating the distance between each and every pair of
    points, then counting the instances where both pairs lie in the void space.

    This approach uses a distance matrix so can consume memory very quickly for
    large 3D images and/or close spacing.  It is recommended to avoid this.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/metrics/reference/two_point_correlation_bf.html>`__
    to view online example.

    """
    _check_for_singleton_axes(im)
    if im.ndim == 2:
        pts = np.meshgrid(range(0, im.shape[0], spacing), range(0, im.shape[1], spacing))
        crds = np.vstack([pts[0].flatten(), pts[1].flatten()]).T
    elif im.ndim == 3:
        pts = np.meshgrid(
            range(0, im.shape[0], spacing),
            range(0, im.shape[1], spacing),
            range(0, im.shape[2], spacing),
        )
        crds = np.vstack([pts[0].flatten(), pts[1].flatten(), pts[2].flatten()]).T
    dmat = sptl.distance.cdist(XA=crds, XB=crds)
    hits = im[tuple(pts)].flatten()
    dmat = dmat[hits, :]
    h1 = np.histogram(dmat, bins=range(0, int(np.amin(im.shape) / 2), spacing))
    dmat = dmat[:, hits]
    h2 = np.histogram(dmat, bins=h1[1])
    tpcf = Results()
    tpcf.distance = h2[1][:-1]
    tpcf.probability = h2[0] / h1[0]
    return tpcf


def _radial_profile(autocorr, bins, pf=None, voxel_size=1):
    r"""
    Helper functions to calculate the radial profile of the autocorrelation

    Masks the image in radial segments from the center and averages the values
    The distance values are normalized and 100 bins are used as default.

    Parameters
    ----------
    autocorr : ndarray
        The image of autocorrelation produced by FFT
    r_max : int or float
        The maximum radius in pixels to sum the image over
    bins : ndarray
        The edges of the bins to use in summing the radii, ** must be in voxels
    pf : float
        the phase fraction (porosity) of the image, used for scaling the
        normalized autocorrelation down to match the two-point correlation
        definition as given by Torquato
    voxel_size : scalar
        The size of a voxel side in preferred units.  The default is 1, so the
        user can apply the scaling to the returned results after the fact.

    Returns
    -------
    result : tpcf


    """
    if len(autocorr.shape) == 2:
        adj = np.reshape(autocorr.shape, [2, 1, 1])
        # use np.round otherwise with odd image sizes, the mask generated can
        # be zero, resulting in Div/0 error
        inds = np.indices(autocorr.shape) - np.round(adj / 2)
        dt = np.sqrt(inds[0] ** 2 + inds[1] ** 2)
    elif len(autocorr.shape) == 3:
        adj = np.reshape(autocorr.shape, [3, 1, 1, 1])
        # use np.round otherwise with odd image sizes, the mask generated can
        # be zero, resulting in Div/0 error
        inds = np.indices(autocorr.shape) - np.round(adj / 2)
        dt = np.sqrt(inds[0] ** 2 + inds[1] ** 2 + inds[2] ** 2)
    else:
        raise Exception("Image dimensions must be 2 or 3")
    if np.max(bins) > np.max(dt):
        msg = (
            "Bins specified distances exceeding maximum radial distance for"
            " image size. Radial distance cannot exceed distance from center"
            " of image to corner."
        )
        raise Exception(msg)

    bin_size = bins[1:] - bins[:-1]
    radial_sum = _get_radial_sum(dt, bins, bin_size, autocorr)
    # Return normalized bin and radially summed autoc
    norm_autoc_radial = radial_sum / np.max(autocorr)
    h = [norm_autoc_radial, bins]
    h = _parse_histogram(h, voxel_size=1)
    tpcf = Results()
    tpcf.distance = h.bin_centers * voxel_size
    tpcf.bin_centers = h.bin_centers * voxel_size
    tpcf.bin_edges = h.bin_edges * voxel_size
    tpcf.bin_widths = h.bin_widths * voxel_size
    tpcf.probability = norm_autoc_radial
    tpcf.probability_scaled = norm_autoc_radial * pf
    tpcf.pdf = h.pdf * pf
    tpcf.relfreq = h.relfreq
    return tpcf


@njit(parallel=False)  # pragma: no cover
def _get_radial_sum(dt, bins, bin_size, autocorr):
    radial_sum = np.zeros_like(bins[:-1], dtype=np.float64)
    for i, r in enumerate(bins[:-1]):
        mask = (dt <= r) * (dt > (r - bin_size[i]))
        radial_sum[i] = np.sum(np.ravel(autocorr)[np.ravel(mask)], dtype=np.int64) / np.sum(
            mask, dtype=np.int64
        )
    return radial_sum


def two_point_correlation(im, voxel_size=1, bins=100):
    r"""
    Calculate the two-point correlation function using Fourier transforms

    Parameters
    ----------
    im : ndarray
        The image of the void space on which the 2-point correlation is
        desired, in which the phase of interest is labelled as True
    voxel_size : scalar
        The size of a voxel side in preferred units.  The default is 1, so
        the user can apply the scaling to the returned results after the
        fact.
    bins : scalar or array_like
        Either an array of bin sizes to use, or the number of bins that
        should be automatically generated that span the data range. The
        maximum value of the bins, if passed as an array, cannot exceed
        the distance from the center of the image to the corner.

    Returns
    -------
    result : tpcf
        A dataclass-like object with following named attributes:

        =========================== =================================================
        Attribute                   Description
        =========================== =================================================
        *distance*                  The distance between two points, equivalent to
                                    bin_centers
        *bin_centers*               The center point of each bin. See distance
        *bin_edges*                 Locations of bin divisions, including 1 more
                                    value than the number of bins
        *bin_widths*                Useful for passing to the ``width`` argument of
                                    ``matplotlib.pyplot.bar``
        *probability_normalized*    The probability that two points of the stated
                                    separation distance are within the same phase
                                    normalized to 1 at r = 0
        *probability*               The probability that two points of the stated
                                    separation distance are within the same phase
                                    scaled to the phase fraction at r = 0
        *pdf*                       Same as probability
        =========================== =================================================


    Notes
    -----
    The fourier transform approach utilizes the fact that the
    autocorrelation function is the inverse FT of the power spectrum
    density. For background read the Scipy fftpack docs and for a good
    explanation `see this thesis
    <https://www.ucl.ac.uk/~ucapikr/projects/KamilaSuankulova_BSc_Project.pdf>`_.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/metrics/reference/two_point_correlation.html>`__
    to view online example.

    """
    # Get the number of CPUs available to parallel process Fourier transforms
    cpus = settings.ncores
    # Get the phase fraction of the image
    pf = porosity(im)
    if isinstance(bins, int):
        # Calculate half lengths of the image
        r_max = (np.ceil(np.min(np.shape(im))) / 2).astype(int)
        # Get the bin-size - ensures it will be at least 1
        bin_size = int(np.ceil(r_max / bins))
        # Calculate the bin divisions, equivalent to bin_edges
        bins = np.arange(0, r_max + bin_size, bin_size)
    # set the number of parallel processors to use:
    with sp_ft.set_workers(cpus):
        # Fourier Transform and shift image
        F = sp_ft.ifftshift(sp_ft.rfftn(sp_ft.fftshift(im)))
        # Compute Power Spectrum
        P = np.absolute(F**2)
        # Auto-correlation is inverse of Power Spectrum
        autoc = np.absolute(sp_ft.ifftshift(sp_ft.irfftn(sp_ft.fftshift(P))))
    tpcf = _radial_profile(autoc, bins, pf=pf, voxel_size=voxel_size)
    return tpcf


def _parse_histogram(h, voxel_size=1, density=True):
    delta_x = h[1]
    P = h[0]
    bin_widths = delta_x[1:] - delta_x[:-1]
    temp = P * (bin_widths)
    C = np.cumsum(temp[-1::-1])[-1::-1]
    S = P * (bin_widths)
    if not density:
        P /= np.max(P)
        temp_sum = np.sum(P * bin_widths)
        C /= temp_sum
        S /= temp_sum

    bin_edges = delta_x * voxel_size
    bin_widths = (bin_widths) * voxel_size
    bin_centers = ((delta_x[1:] + delta_x[:-1]) / 2) * voxel_size
    hist = Results()
    hist.pdf = P
    hist.cdf = C
    hist.relfreq = S
    hist.bin_centers = bin_centers
    hist.bin_edges = bin_edges
    hist.bin_widths = bin_widths
    return hist


def chord_counts(im):
    r"""
    Find the length of each chord in the supplied image

    Parameters
    ----------
    im : ndarray
        An image containing chords drawn in the void space.

    Returns
    -------
    result : 1D-array
        A 1D array with one element for each chord, containing its length.

    Notes
    ----
    The returned array can be passed to ``plt.hist`` to plot the histogram,
    or to ``np.histogram`` to get the histogram data directly. Another useful
    function is ``np.bincount`` which gives the number of chords of each
    length in a format suitable for ``plt.plot``.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/reference/metrics/chord_counts.html>`__
    to view online example.

    """
    labels, N = spim.label(im > 0)
    props = regionprops(labels)
    chord_lens = np.array([i.filled_area for i in props], dtype=int)
    return chord_lens


def phase_fraction(im, normed=True):
    r"""
    Calculate the fraction of each phase in an image

    Parameters
    ----------
    im : ndarray
        An ndarray containing integer values
    normed : boolean
        If ``True`` (default) the returned values are normalized by the total
        number of voxels in image, otherwise the voxel count of each phase is
        returned.

    Returns
    -------
    result : 1D-array
        A array of length max(im) with each element containing the number of
        voxels found with the corresponding label.

    See Also
    --------
    porosity

    Examples
    --------
    `Click here
    <https://porespy.org/examples/metrics/reference/phase_fraction.html>`__
    to view online example.

    """
    if im.dtype == bool:
        im = im.astype(int)
    labels = np.unique(im)
    results = {}
    for label in labels:
        results[label] = np.sum(im == label, dtype=np.int64) * (1 / im.size if normed else 1)
    return results


@deprecated
def pc_curve(im, pc, seq=None):
    r"""
    Produces a Pc-Snwp curve given a map of meniscus radii or capillary
    pressures at which each voxel was invaded

    Parameters
    ----------
    im : ndarray
        The voxel image of the porous media with ``True`` values indicating
        the void space
    pc : ndarray
        An image containing the capillary pressures at which each voxel was
        invaded during an invasion experiment. This image can be produced
        using `size_to_pc` if not available.
    seq : ndarray, optional
        An image containing invasion sequence values, such as that returned
        from the ``ibip`` function. The curve is generated by scanning from
        lowest to highest values and computing the corresponding saturation.

    Returns
    -------
    pc_curve : Results object
        A custom object with the following data added as named attributes:

        ==================  ===================================================
        Attribute           Description
        ==================  ===================================================
        pc                  The capillary pressure, either as given in
                            ``pc`` or computed from ``sizes`` (see
                            Notes).
        snwp                The fraction of void space filled by non-wetting
                            phase at each pressure in ``pc``
        ==================  ===================================================

    Examples
    --------
    `Click here
    <https://porespy.org/examples/metrics/reference/pc_curve.html>`__
    to view online example.

    """
    Ps = np.unique(pc[im])
    # Utilize the fact that -inf and +inf will be at locations 0 & -1 in Ps
    if Ps[-1] == np.inf:
        Ps[-1] = Ps[-2] * 2
    if Ps[0] == -np.inf:
        Ps[0] = Ps[1] - np.abs(Ps[1] / 2)
    else:
        # Add a point at begining to ensure curve starts a 0, if no residual
        Ps = np.hstack((Ps[0] - np.abs(Ps[0] / 2), Ps))
    y = []
    Vp = im.sum(dtype=np.int64)
    temp = pc[im]
    desc = inspect.currentframe().f_code.co_name  # Get current func name
    for p in tqdm(Ps, desc=desc, **settings.tqdm):
        y.append((temp <= p).sum(dtype=np.int64) / Vp)
    pc_curve = Results()
    pc_curve.pc = Ps
    pc_curve.snwp = np.array(y)
    return pc_curve


def pc_map_to_pc_curve(
    pc,
    im,
    seq=None,
    mode="drainage",
    pc_min=None,
    pc_max=None,
    fix_ends=True,
):
    r"""
    Converts a pc map into a capillary pressure curve

    Parameters
    ----------
    pc : ndarray
        A numpy array with each voxel containing the capillary pressure at which
        it was invaded. `-inf` indicates voxels which are filled with non-wetting
        fluid at all pressures, and `+inf` indicates voxels that are filled by
        wetting fluid at all pressures. Values in the solid phase are masked by
        `im` so are ignored.
    im : ndarray
        A numpy array with `True` values indicating the void space and `False`
        elsewhere. This is necessary to define the total void volume of the domain
        when computing the saturation.
    seq : ndarray, optional
        A numpy array with each voxel containing the sequence at which it was
        invaded. This is required when analyzing results from injection simulations
        since the pressures in `pc` do not correspond to the sequence in which
        they were filled.
    mode : str
        Indicates whether the invasion was a drainage or an imbibition process.
        Options are 'drainage' and 'imbibition'.
    fix_ends : bool (default is `True`)
        If `True` (default) this puts values at + and - infinity corresponding to
        maximum and minimum non-wetting phase saturations. This helps when plotting
        as it adds plateaus.
    pc_min, pc_max : float
        Minimum and maximum values to clip the capillary pressures. This is useful
        if the minimum or maximum capillary pressure values are -/+ infinity, which
        means they do not show up when plotting.  Using `pc_min=1` and `pc_max=1e6`
        for instance, will make plateaus render when plotting.

    Returns
    -------
    results : dataclass-like
        A dataclass like object with the following attributes:

        ================== =========================================================
        Attribute          Description
        ================== =========================================================
        pc                 The capillary pressure
        snwp               The fraction of void space filled by non-wetting
                           phase at each pressure in ``pc``
        ================== =========================================================

    Notes
    -----
    To use this function with the results of `porosimetry` or `ibip` the sizes map
    must be converted to a capillary pressure map first.  `drainage` and `invasion`
    both return capillary pressure maps which can be passed directly as `pc`.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/metrics/reference/pc_map_to_pc_curve.html>`_
    to view online example.
    """
    pc = np.copy(pc)

    if seq is None:
        seq = pc_to_seq(im=im, pc=pc, mode=mode)
        # Or stand alone code
        # if mode.startswith('dr'):
        #     seq = np.digitize(x=pc.flatten(), bins=np.unique(pc[im]))
        # elif mode.startswith('imb'):
        #     seq = np.digitize(x=pc.flatten(), bins=np.flip(np.unique(pc)))
        # seq = np.reshape(seq, im.shape)

    if mode.startswith("dr"):
        seq = seq.astype(float)
        seq[pc == np.inf] = np.inf
        seq[pc == -np.inf] = -np.inf
        # This could be done with pc instead of seq, but using seq makes it work
        # for injection as well as drainage
        vals, index, counts = np.unique(seq[im], return_index=True, return_counts=True)
        pcs = pc[im][index]
        # If trapping present, don't include last counts in cumsum
        mask = pcs < np.inf
        snwp = np.cumsum(counts[mask]) / im.sum()
        snwp = np.hstack((snwp, [snwp[-1]]*sum(~mask)))

        if fix_ends:
            if pcs[0] > -np.inf:  # Fix lower left side
                pcs = np.hstack((pcs[0], pcs))
                snwp = np.hstack((0.0, snwp))
            if (pcs[-1] < np.inf) and (snwp[-1] < 1):
                pcs = np.hstack((pcs, np.inf))
                snwp = np.hstack((snwp, snwp[-1]))

    elif mode.startswith("imb"):
        seq = seq.astype(float)
        seq[pc == np.inf] = np.inf  # Set residual pixels in seq to inf
        seq[pc == -np.inf] = -np.inf  # Set trapped pixels in seql to -inf
        vals, index, counts = np.unique(seq[im], return_index=True, return_counts=True)
        pcs = pc[im][index]
        # Move +/-inf to opposite ends of pcs, and upate counts accordingly
        idx = np.argsort(pcs)[-1::-1]
        pcs = pcs[idx]
        counts = counts[idx]

        mask = pcs > -np.inf
        snwp = 1 - np.cumsum(counts[mask]) / im.sum()
        snwp = np.hstack((snwp, [snwp[-1]]*sum(~mask)))
        if fix_ends:
            if pcs[0] < np.inf:
                pcs = np.hstack((pcs[0], pcs))
                snwp = np.hstack((1.0, snwp))
            if (pcs[-1] > -np.inf) and (snwp[-1] > 0):
                pcs = np.hstack((pcs, -np.inf))
                snwp = np.hstack((snwp, snwp[-1]))

    # Apply clipping to Pc values
    if pc_min or pc_max:
        pcs = np.clip(pcs, a_min=pc_min, a_max=pc_max)
        if pc_min and pcs.min() > pc_min:
            pcs = np.hstack((pc_min, pcs))
            snwp = np.hstack((snwp[0], snwp))
        if pc_max and pcs.min() < pc_max:
            pcs = np.hstack((pcs, pc_max))
            snwp = np.hstack((snwp, snwp[-1]))

    results = Results()
    results.pc = pcs
    results.snwp = snwp
    return results


def satn_profile(satn, s=None, im=None, axis=0, span=10, mode="tile",
                 voxel_size=None):
    r"""
    Computes a saturation profile from an image of fluid invasion

    Parameters
    ----------
    satn : ndarray
        An image with each voxel indicating the saturation upon its
        invasion.  0's are treated as solid and -1's are treated as uninvaded
        void space.
    s : scalar
        The global saturation value for which the profile is desired. If `satn` is
        a pre-thresholded boolean image then this is ignored, in which case `im`
        is required.
    im : ndarray
        A boolean image with `True` values indicating the void phase. This is used
        to compute the void volume if `satn` is given as a pre-thresholded boolean
        mask.
    axis : int
        The axis along which to profile should be measured
    span : int
        The width of layers to include in the moving average saturation
        calculation.
    mode : str
        How the moving average should be applied. Options are:

        ======== ==============================================================
        mode     description
        ======== ==============================================================
        'tile'   The average is computed for discrete non-overlapping
                 tiles of a size given by ``span``
        'slide'  The average is computed in a moving window starting at
                 ``span/2`` and sliding by a single voxel. This method
                 provides more data points but is slower.
        ======== ==============================================================

    voxel_size : scalar, optional
        If given, the returned ``position`` values are scaled by this factor.
        If ``None`` (default), positions are returned in voxel units.

    Returns
    -------
    results : dataclass
        Results is a custom porespy class with the following attributes:

        ============= =========================================================
        Attribute     Description
        ============= =========================================================
        position      The position along the given axis at which saturation
                      values are computed.  The units are in voxels unless
                      ``voxel_size`` is given.
        saturation    The local saturation value at each position.
        ============= =========================================================

    Examples
    --------
    `Click here
    <https://porespy.org/examples/metrics/reference/satn_profile.html>`__
    to view online example.
    """
    span = max(1, span)
    if s is None:
        if satn.dtype != bool:
            msg = "Must specify a target saturation if saturation map is provided"
            raise Exception(msg)
        s = 2  # Will find ALL voxels, then > 0 will limit to only True ones
        satn = satn.astype(int)
        satn[satn == 0] = -1
        satn[~im] = 0
    else:
        msg = "The maximum saturation in the image is less than the given threshold"
        if satn.max() < s:
            raise Exception(msg)

    slices = get_slices_slabs(im=satn, axis=axis, span=span, mode=mode)
    y = np.zeros(len(slices))
    z = np.zeros_like(y)
    for i, slab in enumerate(slices):
        void = satn[slab] != 0
        nwp = (satn[slab] <= s) * (satn[slab] > 0)
        y[i] = nwp.sum(dtype=np.int64) / void.sum(dtype=np.int64)
        z[i] = (slab[axis].start + slab[axis].stop) / 2

    if voxel_size is not None:
        z = z * voxel_size
    results = Results()
    results.position = z
    results.saturation = y
    return results


def find_h(saturation, position=None, srange=[0.01, 0.99]):
    r"""
    Given a saturation profile, compute the height between given bounds

    Parameters
    ----------
    saturation : array_like
        A list of saturation values as function of ``position``
    position : array_like, optional
        A list of positions corresponding to each saturation.  If not provided
        then each value in ``saturation`` is assumed to be separated by 1 voxel.
    srange : list
        The minimum and maximum value of saturation to consider as the start
        and end of the profile

    Returns
    -------
    A dataclass-like object with the following attributes:

        =========== ================================================================
        Attribute   Description
        =========== ================================================================
        `zmax`      The position where the saturation first exceeds `smax`
        `zmin`      The position where the saturation first exceeds `smin`
        `smax`      The value defining the start of the saturation profile
        `smin`      The value defining the end of the saturation profile
        `h`         The total distance in voxels between `zmax` and `zmin`
        `valid`     A flag indicating whether the requested saturation difference
                    (between `smin` and `smax`) was found.
        =========== ================================================================

    See Also
    --------
    satn_profile

    Notes
    -----
    The `satn_profile` function can be used to obtain the ``saturation``
    and `position` from an image, such as a displacement map produced by
    `drainage` or `imbibition`.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/metrics/reference/find_h.html>`__
    to view online example.

    """
    r = Results()
    r.valid = True
    # First ensure saturation generally descends from left to right
    if np.mean(saturation[:10]) < np.mean(saturation[-10:]):
        saturation = np.flip(saturation, axis=0)
    # Ensure requested saturation limits actually exist
    if (min(srange) < min(saturation)) or (max(srange) > max(saturation)):
        srange = max(min(srange), min(saturation)), min(max(srange), max(saturation))
        r.valid = False
        logger.warning(
            f"The requested saturation range was adjusted to {srange} to accomodate data"
        )
    # Find zmax
    x = saturation >= max(srange)
    zmax = np.where(x)[0][-1]
    y = saturation <= min(srange)
    zmin = np.where(y)[0][0]
    # If position array was given, index into it
    if position is not None:
        zmax = position[zmax]
        zmin = position[zmin]

    # Add remaining data to results object
    r.zmax = zmax
    r.zmin = zmin
    r.smax = max(srange)
    r.smin = min(srange)
    r.h = abs(zmax - zmin)

    return r


def bond_number(
    im: npt.NDArray,
    delta_rho: float,
    g: float,
    sigma: float,
    voxel_size: float,
    source: str = "lt",
    method: str = "median",
    mask_source: bool = False,
    use_diameter: bool = False,
):
    r"""
    Computes the Bond number for an image

    Parameters
    ----------
    im : ndarray
        The image of the domain with `True` values indicating the phase of interest
    delta_rho : float
        The difference in the density of the non-wetting and wetting phase
    g : float
        The gravitational constant for the system
    sigma : float
        The surface tension of the fluid pair
    voxel_size : float
        The size of the voxels
    source : str
        The source of the pore size values to use when computing the characteristic
        length *R*. Options are:

        ============== =============================================================
        Option         Description
        ============== =============================================================
        dt             Uses the distance transform
        lt             Uses the local thickness
        ============== =============================================================

    method : str
        The method to use for finding the characteristic length *R* from the
        values in `source`. Options are:

        ============== =============================================================
        Option         Description
        ============== =============================================================
        mean           The arithmetic mean (using `numpy.mean`)
        min (or amin)  The minimum value (using `numpy.amin`)
        max (or amax)  The maximum value (using `numpy.amax`)
        mode           The mode of the values (using `scipy.stats.mode`)
        gmean          The geometric mean of the values (using `scipy.stats.gmean`)
        hmean          The harmonic mean of the values (using `scipy.stats.hmean`)
        pmean          The power mean of the values (using `scipy.stats.pmean`)
        ============== =============================================================

    mask_source : bool (default is `False`)
        If `True` then the distance values in `source` are masked by the skeleton
        before computing the average value using the specified `method`. This
        requires computing the skeleton which can take a few moments.
    use_diameter : bool (default is `False`)
        If `True` then the characteristic size obtaine from `source` is multiplied by
        2 to convert radius to diameter.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/metrics/reference/bond_number.html>`_
    to view online example.
    """
    if mask_source is True:
        mask = skeletonize(im)
    else:
        mask = im

    if source == "dt":
        dvals = edt(im)
    elif source == "lt":
        dvals = local_thickness(im)
    else:
        raise Exception(f"Unrecognized source {source}")

    if method in ["median", "mean", "amin", "amax"]:
        f = getattr(np, method)
    elif method in ["min", "max"]:
        f = getattr(np, "a" + method)
    elif method in ["pmean", "hmean", "gmean", "mode"]:
        f = getattr(spst, method)
    else:
        raise Exception(f"Unrecognized method {method}")
    R = f(dvals[mask])
    if use_diameter:
        R = 2 * R
    Bo = abs(delta_rho * g * (R * voxel_size) ** 2 / sigma)
    return Bo
