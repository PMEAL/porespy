import numpy as np
import scipy.ndimage as spim
import scipy.spatial as sptl
from scipy import fftpack as sp_ft
from skimage.measure import regionprops
from porespy.tools import extend_slice, mesh_region, ps_round
from porespy.filters import find_dt_artifacts
from porespy.tools import _check_for_singleton_axes
from skimage import measure
from loguru import logger
from porespy.tools import extend_slice, mesh_region, ps_round
from porespy.tools import _check_for_singleton_axes
from porespy.tools import Results
from porespy import settings
from porespy.tools import get_tqdm
tqdm = get_tqdm()


def representative_elementary_volume(im, npoints=1000):
    r"""
    Calculates the porosity of an image as a function subdomain size.

    This function extracts a specified number of subdomains of random size,
    then finds their porosity.

    Parameters
    ----------
    im : ndarray
        The image of the porous material
    npoints : int
        The number of randomly located and sized boxes to sample.  The default
        is 1000.

    Returns
    -------
    result : Results object
        A custom object with the following data added as named attributes:

        'volume'
            The total volume of each cubic subdomain tested
        'porosity'
            The porosity of each subdomain tested

        These attributes can be conveniently plotted by passing the Results
        object to matplotlib's ``plot`` function using the
        \* notation: ``plt.plot(*result, 'b.')``.  The resulting plot is
        similar to the sketch given by Bachmat and Bear [1]

    Notes
    -----
    This function is frustratingly slow.  Profiling indicates that all the time
    is spent on scipy's ``sum`` function which is needed to sum the number of
    void voxels (1's) in each subdomain.

    References
    ----------
    [1] Bachmat and Bear. On the Concept and Size of a Representative
    Elementary Volume (Rev), Advances in Transport Phenomena in Porous Media
    (1987)

    Examples
    --------
    `Click here
    <https://porespy.org/examples/metrics/howtos/representative_elementary_volume.html>`_
    to view online example.

    """
    # TODO: this function is a prime target for parallelization since the
    # ``npoints`` are calculated independenlty.
    im_temp = np.zeros_like(im)
    crds = np.array(np.random.rand(npoints, im.ndim) * im.shape, dtype=int)
    pads = np.array(np.random.rand(npoints) * np.amin(im.shape) / 2 + 10, dtype=int)
    im_temp[tuple(crds.T)] = True
    labels, N = spim.label(input=im_temp)
    slices = spim.find_objects(input=labels)
    porosity = np.zeros(shape=(N,), dtype=float)
    volume = np.zeros(shape=(N,), dtype=int)
    for i in tqdm(np.arange(0, N), **settings.tqdm):
        s = slices[i]
        p = pads[i]
        new_s = extend_slice(s, shape=im.shape, pad=p)
        temp = im[new_s]
        Vp = np.sum(temp)
        Vt = np.size(temp)
        porosity[i] = Vp / Vt
        volume[i] = Vt
    profile = Results()
    profile.volume = volume
    profile.porosity = porosity
    return profile


def porosity_profile(im, axis=0):
    r"""
    Computes the porosity profile along the specified axis

    Parameters
    ----------
    im : ndarray
        The volumetric image for which to calculate the porosity profile
    axis : int
        The axis (0, 1, or 2) along which to calculate the profile.  For
        instance, if `axis` is 0, then the porosity in each YZ plane is
        calculated and returned as 1D array with 1 value for each X position.

    Returns
    -------
    result : 1D-array
        A 1D-array of porosity along the specified axis

    Examples
    --------
    `Click here
    <https://porespy.org/examples/metrics/howtos/porosity_profile.html>`_
    to view online example.

    """
    if axis >= im.ndim:
        raise Exception('axis out of range')
    im = np.atleast_3d(im)
    a = set(range(im.ndim)).difference(set([axis]))
    a1, a2 = a
    prof = np.sum(np.sum(im, axis=a2), axis=a1) / (im.shape[a2] * im.shape[a1])
    return prof


def radial_density_distribution(dt, bins=10, log=False, voxel_size=1):
    r"""
    Computes radial density function by analyzing the histogram of voxel
    values in the distance transform.  This function is defined by
    Torquato [1] as:

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
        A distance transform of the pore space (the ``edt`` package is
        recommended).  Note that it is recommended to apply
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
        distributions or to better visualize the in the small size region.
        Note that you should not anti-log the radii values in the retunred
        ``tuple``, since the binning is performed on the logged radii values.
    voxel_size : scalar
        The size of a voxel side in preferred units.  The default is 1, so the
        user can apply the scaling to the returned results after the fact.

    Returns
    -------
    result : Results object
        A custom object with the following data added as named attributes:

        *R* or *LogR*
            Radius, equivalent to ``bin_centers``
        *pdf*
            Probability density function
        *cdf*
            Cumulative density function
        *bin_centers*
            The center point of each bin
        *bin_edges*
            Locations of bin divisions, including 1 more value than
            the number of bins
        *bin_widths*
            Useful for passing to the ``width`` argument of
            ``matplotlib.pyplot.bar``

    Notes
    -----
    Torquato refers to this as the *pore-size density function*, and mentions
    that it is also known as the *pore-size distribution function*.  These
    terms are avoided here since they have specific connotations in porous
    media analysis.

    References
    ----------
    [1] Torquato, S. Random Heterogeneous Materials: Mircostructure and
    Macroscopic Properties. Springer, New York (2002) - See page 48 & 292

    Examples
    --------
    `Click here
    <https://porespy.org/examples/metrics/howtos/radial_density.html>`_
    to view online example.

    """
    im = np.copy(dt)
    x = im[im > 0].flatten()
    if log:
        x = np.log10(x)
    h = np.histogram(x, bins=bins, density=True)
    h = _parse_histogram(h=h, voxel_size=voxel_size)
    rdf = Results()
    rdf[f"{log*'Log' + 'R'}"] = h.bin_centers
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

    This relates directly the radial density function defined by Torquato [1],
    but instead of reporting the probability of lying within a stated distance
    to the nearest solid in any direciton, it considers only linear distances
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
        Note that you should not anti-log the radii values in the retunred
        ``results``, since the binning is performed on the logged radii values.

    Returns
    -------
    result : Results object
        A custom object with the following data added as named attributes:

        *L* or *LogL*
            Length, equivalent to ``bin_centers``
        *pdf*
            Probability density function
        *cdf*
            Cumulative density function
        *relfreq*
            Relative frequency chords in each bin.  The sum of all bin
            heights is 1.0.  For the cumulative relativce, use *cdf* which is
            already normalized to 1.
        *bin_centers*
            The center point of each bin
        *bin_edges*
            Locations of bin divisions, including 1 more value than
            the number of bins
        *bin_widths*
            Useful for passing to the ``width`` argument of
            ``matplotlib.pyplot.bar``

    References
    ----------
    [1] Torquato, S. Random Heterogeneous Materials: Mircostructure and
    Macroscopic Properties. Springer, New York (2002)

    Examples
    --------
    `Click here
    <https://porespy.org/examples/metrics/howtos/linear_density.html>`_
    to view online example.

    """
    x = im[im > 0]
    if log:
        x = np.log10(x)
    h = list(np.histogram(x, bins=bins, density=True))
    h = _parse_histogram(h=h, voxel_size=voxel_size)
    cld = Results()
    cld[f"{log*'Log' + 'L'}"] = h.bin_centers
    cld.pdf = h.pdf
    cld.cdf = h.cdf
    cld.relfreq = h.relfreq
    cld.bin_centers = h.bin_centers
    cld.bin_edges = h.bin_edges
    cld.bin_widths = h.bin_widths
    return cld


def chord_length_distribution(im, bins=10, log=False, voxel_size=1,
                              normalization='count'):
    r"""
    Determines the distribution of chord lengths in an image containing chords.

    Parameters
    ----------
    im : ndarray
        An image with chords drawn in the pore space, as produced by
        ``apply_chords`` or ``apply_chords_3d``.  ``im`` can be either boolean,
        in which case each chord will be identified using ``scipy.ndimage.label``,
        or numerical values in case it is assumed that chords have already been
        identifed and labeled. In both cases, the size of each chord will be
        computed as the number of voxels belonging to each labelled region.
    bins : scalar or array_like
        If a scalar is given it is interpreted as the number of bins to use,
        and if an array is given they are used as the bins directly.
    log : boolean
        If ``True`` (default) the size data is converted to log (base-10)
        values before processing.  This can help to plot wide size
        distributions or to better visualize the in the small size region.
        Note that you should not anti-log the radii values in the retunred
        ``tuple``, since the binning is performed on the logged radii values.
    normalization : string
        Indicates how to normalize the bin heights.  Options are:

        *'count' or 'number'*
            (default) This simply counts the number of chords in each bin in
            the normal sense of a histogram.  This is the rigorous definition
            according to Torquato [1].
        *'length'*
            This multiplies the number of chords in each bin by the
            chord length (i.e. bin size).  The normalization scheme accounts for
            the fact that long chords are less frequent than shorert chords,
            thus giving a more balanced distribution.

    voxel_size : scalar
        The size of a voxel side in preferred units.  The default is 1, so the
        user can apply the scaling to the returned results after the fact.

    Returns
    -------
    result : Results object
        A custom object with the following data added as named attributes:

        *L* or *LogL*
            Chord length, equivalent to ``bin_centers``
        *pdf*
            Probability density function
        *cdf*
            Cumulative density function
        *relfreq*
            Relative frequency chords in each bin.  The sum of all bin
            heights is 1.0.  For the cumulative relativce, use *cdf* which is
            already normalized to 1.
        *bin_centers*
            The center point of each bin
        *bin_edges*
            Locations of bin divisions, including 1 more value than
            the number of bins
        *bin_widths*
            Useful for passing to the ``width`` argument of
            ``matplotlib.pyplot.bar``

    References
    ----------
    [1] Torquato, S. Random Heterogeneous Materials: Mircostructure and
    Macroscopic Properties. Springer, New York (2002) - See page 45 & 292

    Examples
    --------
    `Click here
    <https://porespy.org/examples/metrics/howtos/chord_length_distribution.html>`_
    to view online example.

    """
    x = chord_counts(im)
    if bins is None:
        bins = np.array(range(0, x.max() + 2)) * voxel_size
    x = x * voxel_size
    if log:
        x = np.log10(x)
    if normalization == 'length':
        h = list(np.histogram(x, bins=bins, density=False))
        h[0] = h[0] * (h[1][1:] + h[1][:-1]) / 2  # Scale bin heigths by length
        h[0] = h[0] / h[0].sum() / (h[1][1:] - h[1][:-1])  # Normalize h[0] manually
    elif normalization in ['number', 'count']:
        h = np.histogram(x, bins=bins, density=True)
    else:
        raise Exception('Unsupported normalization:', normalization)
    h = _parse_histogram(h)
    cld = Results()
    cld[f"{log*'Log' + 'L'}"] = h.bin_centers
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
        Note that you should not anti-log the radii values in the retunred
        ``tuple``, since the binning is performed on the logged radii values.
    voxel_size : scalar
        The size of a voxel side in preferred units.  The default is 1, so the
        user can apply the scaling to the returned results after the fact.

    Returns
    -------
    result : Results object
        A custom object with the following data added as named attributes:

        *R* or *logR*
            Radius, equivalent to ``bin_centers``
        *pdf*
            Probability density function
        *cdf*
            Cumulative density function
        *satn*
            Phase saturation in differential form.  For the cumulative
            saturation, just use *cfd* which is already normalized to 1.
        *bin_centers*
            The center point of each bin
        *bin_edges*
            Locations of bin divisions, including 1 more value than
            the number of bins
        *bin_widths*
            Useful for passing to the ``width`` argument of
            ``matplotlib.pyplot.bar``

    Notes
    -----
    (1) To ensure the returned values represent actual sizes you can manually
    scale the input image by the voxel size first (``im *= voxel_size``)

    plt.bar(psd.R, psd.satn, width=psd.bin_widths, edgecolor='k')

    Examples
    --------
    `Click here
    <https://porespy.org/examples/metrics/howtos/pore_size_distribution.html>`_
    to view online example.

    """
    im = im.flatten()
    vals = im[im > 0] * voxel_size
    if log:
        vals = np.log10(vals)
    h = _parse_histogram(np.histogram(vals, bins=bins, density=True))
    cld = Results()
    cld[f"{log*'Log' + 'R'}"] = h.bin_centers
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
            as:
        $$ bins = range(start=0, stop=np.amin(im.shape)/2, stride=spacing) $$

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
    <https://porespy.org/examples/metrics/howtos/two_point_correlation_bf.html>`_
    to view online example.

    """
    _check_for_singleton_axes(im)
    if im.ndim == 2:
        pts = np.meshgrid(range(0, im.shape[0], spacing),
                          range(0, im.shape[1], spacing))
        crds = np.vstack([pts[0].flatten(),
                          pts[1].flatten()]).T
    elif im.ndim == 3:
        pts = np.meshgrid(range(0, im.shape[0], spacing),
                          range(0, im.shape[1], spacing),
                          range(0, im.shape[2], spacing))
        crds = np.vstack([pts[0].flatten(),
                          pts[1].flatten(),
                          pts[2].flatten()]).T
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


def _radial_profile(autocorr, r_max, nbins=100):
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

    Returns
    -------
    result : named_tuple
        A named tupling containing an array of ``bins`` of radial position
        and an array of ``counts`` in each bin.

    """
    if len(autocorr.shape) == 2:
        adj = np.reshape(autocorr.shape, [2, 1, 1])
        inds = np.indices(autocorr.shape) - adj / 2
        dt = np.sqrt(inds[0]**2 + inds[1]**2)
    elif len(autocorr.shape) == 3:
        adj = np.reshape(autocorr.shape, [3, 1, 1, 1])
        inds = np.indices(autocorr.shape) - adj / 2
        dt = np.sqrt(inds[0]**2 + inds[1]**2 + inds[2]**2)
    else:
        raise Exception('Image dimensions must be 2 or 3')
    bin_size = np.int(np.ceil(r_max / nbins))
    bins = np.arange(bin_size, r_max, step=bin_size)
    radial_sum = np.zeros_like(bins)
    for i, r in enumerate(bins):
        # Generate Radial Mask from dt using bins
        mask = (dt <= r) * (dt > (r - bin_size))
        radial_sum[i] = np.sum(autocorr[mask]) / np.sum(mask)
    # Return normalized bin and radially summed autoc
    norm_autoc_radial = radial_sum / np.max(autocorr)
    tpcf = Results()
    tpcf.distance = bins
    tpcf.probability = norm_autoc_radial
    return tpcf


def two_point_correlation(im):
    r"""
    Calculate the two-point correlation function using Fourier transforms

    Parameters
    ----------
    im : ndarray
        The image of the void space on which the 2-point correlation is
        desired.

    Returns
    -------
    result : Results object
        A custom object with the following data added as named attributes:

        'distance'
            The distance between two points.

        'probability'
            The probability that two points of the stated separation distance
            are within the same phase

    Notes
    -----
    The fourier transform approach utilizes the fact that the
    autocorrelation function is the inverse FT of the power spectrum
    density. For background read the Scipy fftpack docs and for a good
    explanation `see this thesis
    <https://www.ucl.ac.uk/~ucapikr/projects/KamilaSuankulova_BSc_Project.pdf>`_

    Examples
    --------
    `Click here
    <https://porespy.org/examples/metrics/howtos/two_point_correlation_fft.html>`_
    to view online example.

    """
    # Calculate half lengths of the image
    hls = (np.ceil(np.shape(im)) / 2).astype(int)
    # Fourier Transform and shift image
    F = sp_ft.ifftshift(sp_ft.fftn(sp_ft.fftshift(im)))
    # Compute Power Spectrum
    P = np.absolute(F**2)
    # Auto-correlation is inverse of Power Spectrum
    autoc = np.absolute(sp_ft.ifftshift(sp_ft.ifftn(sp_ft.fftshift(P))))
    tpcf = _radial_profile(autoc, r_max=np.min(hls))
    return tpcf


def _parse_histogram(h, voxel_size=1):
    delta_x = h[1]
    P = h[0]
    temp = P * (delta_x[1:] - delta_x[:-1])
    C = np.cumsum(temp[-1::-1])[-1::-1]
    S = P * (delta_x[1:] - delta_x[:-1])
    bin_edges = delta_x * voxel_size
    bin_widths = (delta_x[1:] - delta_x[:-1]) * voxel_size
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
    <https://porespy.org/examples/howtos/metrics/chord_counts.html>`_
    to view online example.

    """
    labels, N = spim.label(im > 0)
    props = regionprops(labels)
    chord_lens = np.array([i.filled_area for i in props])
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
    <https://porespy.org/examples/metrics/howtos/phase_fraction.html>`_
    to view online example.

    """
    if im.dtype == bool:
        im = im.astype(int)
    labels = np.unique(im)
    results = {}
    for label in labels:
        results[label] = np.sum(im == label) * (1 / im.size if normed else 1)
    return results


def pc_curve_from_ibip(seq, sizes, im=None, sigma=0.072, theta=180, voxel_size=1,
                       stepped=True):
    r"""
    Produces a Pc-Snwp curve from the output of ``ibip``

    Parameters
    ----------
    seq : ndarray
        The image containing the invasion sequence values returned from the
        ``ibip`` function.
    sizes : ndarray
        The image containing the invasion size values returned from ``ibip``
    im : ndarray
        The voxel image of the porous media.  It not provided then the void
        space is assumed to be ``im = ~(seq == 0)``.
    sigma : float
        The surface tension of the fluid-fluid system of interest
    theta : float
        The contact angle measured through the invading phase in degrees
    voxel_size : float
        The voxel resolution of the image
    stepped : boolean
        If ``True`` (default) the returned data has steps between each point
        instead of connecting points directly with sloped lines.

    Returns
    -------
    pc_curve : Results object
        A custom object with the following data added as named attributes:

        'pc'
            The capillary pressure, computed using the Washburn equation with
            the given fluid properties

        'snwp'
            the fraction of void space filled by non-wetting phase.

        If ``stepped`` was set to ``True`` then the values include the corners
        of the steps, which may be helpful for plotting.

    """
    if im is None:
        im = ~(seq == 0)
    seqs = np.unique(seq)[1:]
    x = []
    y = []
    with tqdm(seqs, **settings.tqdm) as pbar:
        for n in seqs:
            pbar.update()
            mask = seq == n
            # The following assumes only one size found, which was confirmed
            r = sizes[mask][0]*voxel_size
            pc = -2*sigma*np.cos(np.deg2rad(theta))/r
            x.append(pc)
            snwp = ((seq <= n)*(seq > 0)*(im == 1)).sum()/im.sum()
            y.append(snwp)
    if stepped:
        pc = x.copy()
        snwp = y.copy()
        for i in range(0, len(x)-1):
            j = 2*i + 1
            pc.insert(j, x[i+1])
            snwp.insert(j, y[i])
        x = pc
        y = snwp
    pc_curve = Results()
    pc_curve.pc = x
    pc_curve.snwp = y
    return pc_curve


def pc_curve_from_mio(sizes, im=None, sigma=0.072, theta=180, voxel_size=1,
                      stepped=True):
    r"""
    Produces a Pc-Snwp curve from the output of ``porosimetry``

    Parameters
    ----------
    sizes : ndarray
        The image of invasion sizes returned from ``porosimetry``
    im : ndarray
        The voxel image of the porous media.  It not provided then the void
        space is assumed to be ``im = ~(sizes == 0)``.
    sigma : float
        The surface tension of the fluid-fluid system of interest
    theta : float
        The contact angle measured through the invading phase in degrees
    voxel_size : float
        The voxel resolution of the image
    stepped : boolean
        If ``True`` (default) the returned data has steps between each point
        instead of connecting points directly with sloped lines.

    Returns
    -------
    pc_curve : Results object
        A custom object with the following data added as named attributes:

        'pc'
            The capillary pressure, computed using the Washburn equation with
            the given fluid properties

        'snwp'
            the fraction of void space filled by non-wetting phase.

        If ``stepped`` was set to ``True`` then the values include the corners
        of the steps, which may be helpful for plotting.

    """
    if im is None:
        im = ~(sizes == 0)
    sz = np.unique(sizes)[:0:-1]
    x = []
    y = []
    with tqdm(sz, **settings.tqdm) as pbar:
        for n in sz:
            pbar.update()
            r = n*voxel_size
            pc = -2*sigma*np.cos(np.deg2rad(theta))/r
            x.append(pc)
            snwp = ((sizes >= n)*(im == 1)).sum()/im.sum()
            y.append(snwp)
    if stepped:
        pc = x.copy()
        snwp = y.copy()
        for i in range(0, len(x)-1):
            j = 2*i + 1
            pc.insert(j, x[i+1])
            snwp.insert(j, y[i])
        x = pc
        y = snwp
    pc_curve = Results()
    pc_curve.pc = x
    pc_curve.snwp = y
    return pc_curve


def porosity(im):
    r"""
    Calculates the porosity of an image assuming 1's are void space and 0's are
    solid phase.

    All other values are ignored, so this can also return the relative
    fraction of a phase of interest in multiphase images.

    Parameters
    ----------
    im : ndarray
        Image of the void space with 1's indicating void phase (or ``True``)
        and 0's indicating the solid phase (or ``False``).

    Returns
    -------
    porosity : float
        Calculated as the sum of all 1's divided by the sum of all 1's and 0's.
        Note that the denominator is *not* the total image size, so putting
        values of 2 in some voxels, for instance, will remove those voxels
        from consideration.

    See Also
    --------
    phase_fraction
    find_outer_region

    Notes
    -----
    This function assumes void is represented by 1 and solid by 0, and all
    other values are ignored.  This is useful, for example, for images of
    cylindrical cores, where all voxels outside the core are labelled with 2.

    Alternatively, images can be processed with ``find_disconnected_voxels``
    to get an image of only blind pores.  This can then be added to the orignal
    image such that blind pores have a value of 2, thus allowing the
    calculation of accessible porosity, rather than overall porosity.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/metrics/howtos/porosity.html>`_
    to view online example.

    """
    im = np.array(im, dtype=int)
    Vp = np.sum(im == 1)
    Vs = np.sum(im == 0)
    e = Vp / (Vs + Vp)
    return e
