import numpy as np
import warnings
from skimage.measure import regionprops
import scipy.ndimage as spim
import scipy.spatial as sptl
from edt import edt
from porespy.tools import extend_slice, mesh_region
from porespy.filters import find_dt_artifacts
from collections import namedtuple
from tqdm import tqdm
from scipy import fftpack as sp_ft
from skimage import measure


def representative_elementary_volume(im, npoints=1000):
    r"""
    Calculates the porosity of the image as a function subdomain size.  This
    function extracts a specified number of subdomains of random size, then
    finds their porosity.

    Parameters
    ----------
    im : ND-array
        The image of the porous material
    npoints : int
        The number of randomly located and sized boxes to sample.  The default
        is 1000.

    Returns
    -------
    result : named_tuple
        A tuple containing the *volume* and *porosity* of each subdomain
        tested in arrays ``npoints`` long.  They can be accessed as
        attributes of the tuple.  They can be conveniently plotted
        by passing the tuple to matplotlib's ``plot`` function using the
        \* notation: ``plt.plot(*result, 'b.')``.  The resulting plot is
        similar to the sketch given by Bachmat and Bear [1]

    Notes
    -----
    This function is frustratingly slow.  Profiling indicates that all the time
    is spent on scipy's ``sum`` function which is needed to sum the number of
    void voxels (1's) in each subdomain.

    Also, this function is a prime target for parallelization since the
    ``npoints`` are calculated independenlty.

    References
    ----------
    [1] Bachmat and Bear. On the Concept and Size of a Representative
    Elementary Volume (Rev), Advances in Transport Phenomena in Porous Media
    (1987)

    """
    im_temp = np.zeros_like(im)
    crds = np.array(np.random.rand(npoints, im.ndim)*im.shape, dtype=int)
    pads = np.array(np.random.rand(npoints)*np.amin(im.shape)/2+10, dtype=int)
    im_temp[tuple(crds.T)] = True
    labels, N = spim.label(input=im_temp)
    slices = spim.find_objects(input=labels)
    porosity = np.zeros(shape=(N,), dtype=float)
    volume = np.zeros(shape=(N,), dtype=int)
    for i in tqdm(np.arange(0, N)):
        s = slices[i]
        p = pads[i]
        new_s = extend_slice(s, shape=im.shape, pad=p)
        temp = im[new_s]
        Vp = np.sum(temp)
        Vt = np.size(temp)
        porosity[i] = Vp/Vt
        volume[i] = Vt
    profile = namedtuple('profile', ('volume', 'porosity'))
    profile.volume = volume
    profile.porosity = porosity
    return profile


def porosity_profile(im, axis=0):
    r"""
    Returns a porosity profile along the specified axis

    Parameters
    ----------
    im : ND-array
        The volumetric image for which to calculate the porosity profile
    axis : int
        The axis (0, 1, or 2) along which to calculate the profile.  For
        instance, if `axis` is 0, then the porosity in each YZ plane is
        calculated and returned as 1D array with 1 value for each X position.

    Returns
    -------
    result : 1D-array
        A 1D-array of porosity along the specified axis
    """
    if axis >= im.ndim:
        raise Exception('axis out of range')
    im = np.atleast_3d(im)
    a = set(range(im.ndim)).difference(set([axis]))
    a1, a2 = a
    prof = np.sum(np.sum(im, axis=a2), axis=a1)/(im.shape[a2]*im.shape[a1])
    return prof


def radial_density(im, bins=10, voxel_size=1):
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
    im : ND-array
        Either a binary image of the pore space with ``True`` indicating the
        pore phase (or phase of interest), or a pre-calculated distance
        transform which can save time.
    bins : int or array_like
        This number of bins (if int) or the location of the bins (if array).
        This argument is passed directly to Scipy's ``histogram`` function so
        see that docstring for more information.  The default is 10 bins, which
        reduces produces a relatively smooth distribution.
    voxel_size : scalar
        The size of a voxel side in preferred units.  The default is 1, so the
        user can apply the scaling to the returned results after the fact.

    Returns
    -------
    result : named_tuple
        A named-tuple containing several 1D arrays:

        *R* - radius, equivalent to ``bin_centers``

        *pdf* - probability density function

        *cdf* - cumulative density function

        *bin_centers* - the center point of each bin

        *bin_edges* - locations of bin divisions, including 1 more value than
        the number of bins

        *bin_widths* - useful for passing to the ``width`` argument of
        ``matplotlib.pyplot.bar``

    Notes
    -----
    This function should not be taken as a pore size distribution in the
    explict sense, but rather an indicator of the sizes in the image.  The
    distance transform contains a very skewed number of voxels with small
    values near the solid walls.  Nonetheless, it does provide a useful
    indicator and it's mathematical formalism is handy.

    Torquato refers to this as the *pore-size density function*, and mentions
    that it is also known as the *pore-size distribution function*.  These
    terms are avoided here since they have specific connotations in porous
    media analysis.

    References
    ----------
    [1] Torquato, S. Random Heterogeneous Materials: Mircostructure and
    Macroscopic Properties. Springer, New York (2002) - See page 48 & 292
    """
    if im.dtype == bool:
        im = edt(im)
    mask = find_dt_artifacts(im) == 0
    im[mask] = 0
    x = im[im > 0].flatten()
    h = np.histogram(x, bins=bins, density=True)
    h = _parse_histogram(h=h, voxel_size=voxel_size)
    rdf = namedtuple('radial_density_function',
                     ('R', 'pdf', 'cdf', 'bin_centers', 'bin_edges',
                      'bin_widths'))
    return rdf(h.bin_centers, h.pdf, h.cdf, h.bin_centers, h.bin_edges,
               h.bin_widths)


def porosity(im):
    r"""
    Calculates the porosity of an image assuming 1's are void space and 0's are
    solid phase.

    All other values are ignored, so this can also return the relative
    fraction of a phase of interest in trinary or multiphase images.

    Parameters
    ----------
    im : ND-array
        Image of the void space with 1's indicating void phase (or True) and
        0's indicating the solid phase (or False).

    Returns
    -------
    porosity : float
        Calculated as the sum of all 1's divided by the sum of all 1's and 0's.

    See Also
    --------
    phase_fraction

    Notes
    -----
    This function assumes void is represented by 1 and solid by 0, and all
    other values are ignored.  This is useful, for example, for images of
    cylindrical cores, where all voxels outside the core are labelled with 2.

    Alternatively, images can be processed with ``find_disconnected_voxels``
    to get an image of only blind pores.  This can then be added to the orignal
    image such that blind pores have a value of 2, thus allowing the
    calculation of accessible porosity, rather than overall porosity.

    """
    im = np.array(im, dtype=int)
    Vp = np.sum(im == 1)
    Vs = np.sum(im == 0)
    e = Vp/(Vs + Vp)
    return e


def two_point_correlation_bf(im, spacing=10):
    r"""
    Calculates the two-point correlation function using brute-force (see Notes)

    Parameters
    ----------
    im : ND-array
        The image of the void space on which the 2-point correlation is desired
    spacing : int
        The space between points on the regular grid that is used to generate
        the correlation (see Notes)

    Returns
    -------
    result : named_tuple
        A tuple containing the x and y data for plotting the two-point
        correlation function, using the *args feature of matplotlib's plot
        function.  The x array is the distances between points and the y array
        is corresponding probabilities that points of a given distance both
        lie in the void space. The distance values are binned as follows:
        ``bins = range(start=0, stop=np.amin(im.shape)/2, stride=spacing)``

    Notes
    -----
    The brute-force approach means overlaying a grid of equally spaced points
    onto the image, calculating the distance between each and every pair of
    points, then counting the instances where both pairs lie in the void space.

    This approach uses a distance matrix so can consume memory very quickly for
    large 3D images and/or close spacing.
    """
    if im.ndim != im.squeeze().ndim:
        warnings.warn('Input image conains a singleton axis:' + str(im.shape)
                      + ' Reduce dimensionality with np.squeeze(im) to avoid'
                      + ' unexpected behavior.')
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
    h1 = np.histogram(dmat, bins=range(0, int(np.amin(im.shape)/2), spacing))
    dmat = dmat[:, hits]
    h2 = np.histogram(dmat, bins=h1[1])
    tpcf = namedtuple('two_point_correlation_function',
                      ('distance', 'probability'))
    return tpcf(h2[1][:-1], h2[0]/h1[0])


def _radial_profile(autocorr, r_max, nbins=100):
    r"""
    Helper functions to calculate the radial profile of the autocorrelation
    Masks the image in radial segments from the center and averages the values
    The distance values are normalized and 100 bins are used as default.

    Parameters
    ----------
    autocorr : ND-array
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
        inds = np.indices(autocorr.shape) - adj/2
        dt = np.sqrt(inds[0]**2 + inds[1]**2)
    elif len(autocorr.shape) == 3:
        adj = np.reshape(autocorr.shape, [3, 1, 1, 1])
        inds = np.indices(autocorr.shape) - adj/2
        dt = np.sqrt(inds[0]**2 + inds[1]**2 + inds[2]**2)
    else:
        raise Exception('Image dimensions must be 2 or 3')
    bin_size = np.int(np.ceil(r_max/nbins))
    bins = np.arange(bin_size, r_max, step=bin_size)
    radial_sum = np.zeros_like(bins)
    for i, r in enumerate(bins):
        # Generate Radial Mask from dt using bins
        mask = (dt <= r) * (dt > (r-bin_size))
        radial_sum[i] = np.sum(autocorr[mask])/np.sum(mask)
    # Return normalized bin and radially summed autoc
    norm_autoc_radial = radial_sum/np.max(autocorr)
    tpcf = namedtuple('two_point_correlation_function',
                      ('distance', 'probability'))
    return tpcf(bins, norm_autoc_radial)


def two_point_correlation_fft(im):
    r"""
    Calculates the two-point correlation function using fourier transforms

    Parameters
    ----------
    im : ND-array
        The image of the void space on which the 2-point correlation is desired

    Returns
    -------
    result : named_tuple
        A tuple containing the x and y data for plotting the two-point
        correlation function, using the *args feature of matplotlib's plot
        function.  The x array is the distances between points and the y array
        is corresponding probabilities that points of a given distance both
        lie in the void space.

    Notes
    -----
    The fourier transform approach utilizes the fact that the autocorrelation
    function is the inverse FT of the power spectrum density.
    For background read the Scipy fftpack docs and for a good explanation see:
    http://www.ucl.ac.uk/~ucapikr/projects/KamilaSuankulova_BSc_Project.pdf
    """
    # Calculate half lengths of the image
    hls = (np.ceil(np.shape(im))/2).astype(int)
    # Fourier Transform and shift image
    F = sp_ft.ifftshift(sp_ft.fftn(sp_ft.fftshift(im)))
    # Compute Power Spectrum
    P = np.absolute(F**2)
    # Auto-correlation is inverse of Power Spectrum
    autoc = np.absolute(sp_ft.ifftshift(sp_ft.ifftn(sp_ft.fftshift(P))))
    tpcf = _radial_profile(autoc, r_max=np.min(hls))
    return tpcf


def pore_size_distribution(im, bins=10, log=True, voxel_size=1):
    r"""
    Calculate a pore-size distribution based on the image produced by the
    ``porosimetry`` or ``local_thickness`` functions.

    Parameters
    ----------
    im : ND-array
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
        Note that you can anti-log the radii values in the retunred ``tuple``,
        but the binning is performed on the logged radii values.
    voxel_size : scalar
        The size of a voxel side in preferred units.  The default is 1, so the
        user can apply the scaling to the returned results after the fact.

    Returns
    -------
    result : named_tuple
        A named-tuple containing several values:

        *R* or *logR* - radius, equivalent to ``bin_centers``

        *pdf* - probability density function

        *cdf* - cumulative density function

        *satn* - phase saturation in differential form.  For the cumulative
        saturation, just use *cfd* which is already normalized to 1.

        *bin_centers* - the center point of each bin

        *bin_edges* - locations of bin divisions, including 1 more value than
        the number of bins

        *bin_widths* - useful for passing to the ``width`` argument of
        ``matplotlib.pyplot.bar``

    Notes
    -----
    (1) To ensure the returned values represent actual sizes you can manually
    scale the input image by the voxel size first (``im *= voxel_size``)

    plt.bar(psd.R, psd.satn, width=psd.bin_widths, edgecolor='k')

    """
    im = im.flatten()
    vals = im[im > 0]*voxel_size
    if log:
        vals = np.log10(vals)
    h = _parse_histogram(np.histogram(vals, bins=bins, density=True))
    psd = namedtuple('pore_size_distribution',
                     (log*'log' + 'R', 'pdf', 'cdf', 'satn',
                      'bin_centers', 'bin_edges', 'bin_widths'))
    return psd(h.bin_centers, h.pdf, h.cdf, h.relfreq,
               h.bin_centers, h.bin_edges, h.bin_widths)


def _parse_histogram(h, voxel_size=1):
    delta_x = h[1]
    P = h[0]
    temp = P*(delta_x[1:] - delta_x[:-1])
    C = np.cumsum(temp[-1::-1])[-1::-1]
    S = P*(delta_x[1:] - delta_x[:-1])
    bin_edges = delta_x * voxel_size
    bin_widths = (delta_x[1:] - delta_x[:-1]) * voxel_size
    bin_centers = ((delta_x[1:] + delta_x[:-1])/2) * voxel_size
    psd = namedtuple('histogram', ('pdf', 'cdf', 'relfreq',
                                   'bin_centers', 'bin_edges', 'bin_widths'))
    return psd(P, C, S, bin_centers, bin_edges, bin_widths)


def chord_counts(im):
    r"""
    Finds the length of each chord in the supplied image and returns a list
    of their individual sizes

    Parameters
    ----------
    im : ND-array
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
    """
    labels, N = spim.label(im > 0)
    props = regionprops(labels)
    chord_lens = np.array([i.filled_area for i in props])
    return chord_lens


def linear_density(im, bins=25, voxel_size=1, log=False):
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
    im : ND-array
        An image with each voxel containing the distance to the nearest solid
        along a linear path, as produced by ``distance_transform_lin``.
    bins : int or array_like
        The number of bins or a list of specific bins to use
    voxel_size : scalar
        The side length of a voxel.  This is used to scale the chord lengths
        into real units.  Note this is applied *after* the binning, so
        ``bins``, if supplied, should be in terms of voxels, not length units.

    Returns
    -------
    result : named_tuple

    References
    ----------
    [1] Torquato, S. Random Heterogeneous Materials: Mircostructure and
    Macroscopic Properties. Springer, New York (2002)

    """
    x = im[im > 0]
    h = list(np.histogram(x, bins=bins, density=True))
    h = _parse_histogram(h=h, voxel_size=voxel_size)
    cld = namedtuple('linear_density_function',
                     ('L', 'pdf', 'cdf', 'relfreq',
                      'bin_centers', 'bin_edges', 'bin_widths'))
    return cld(h.bin_centers, h.pdf, h.cdf, h.relfreq,
               h.bin_centers, h.bin_edges, h.bin_widths)


def chord_length_distribution(im, bins=None, log=False, voxel_size=1,
                              normalization='count'):
    r"""
    Determines the distribution of chord lengths in an image containing chords.

    Parameters
    ----------
    im : ND-image
        An image with chords drawn in the pore space, as produced by
        ``apply_chords`` or ``apply_chords_3d``.

        ``im`` can be either boolean, in which case each chord will be
        identified using ``scipy.ndimage.label``, or numerical values in which
        case it is assumed that chords have already been identifed and labeled.
        In both cases, the size of each chord will be computed as the number
        of voxels belonging to each labelled region.

    bins : scalar or array_like
        If a scalar is given it is interpreted as the number of bins to use,
        and if an array is given they are used as the bins directly.

    log : Boolean
        If true, the logarithm of the chord lengths will be used, which can
        make the data more clear.

    normalization : string
        Indicates how to normalize the bin heights.  Options are:

        *'count' or 'number'* - (default) This simply counts the number of
        chords in each bin in the normal sense of a histogram.  This is the
        rigorous definition according to Torquato [1].

        *'length'* - This multiplies the number of chords in each bin by the
        chord length (i.e. bin size).  The normalization scheme accounts for
        the fact that long chords are less frequent than shorert chords,
        thus giving a more balanced distribution.

    voxel_size : scalar
        The size of a voxel side in preferred units.  The default is 1, so the
        user can apply the scaling to the returned results after the fact.

    Returns
    -------
    result : named_tuple
        A tuple containing the following elements, which can be retrieved by
        attribute name:

        *L* or *logL* - chord length, equivalent to ``bin_centers``

        *pdf* - probability density function

        *cdf* - cumulative density function

        *relfreq* - relative frequency chords in each bin.  The sum of all bin
        heights is 1.0.  For the cumulative relativce, use *cdf* which is
        already normalized to 1.

        *bin_centers* - the center point of each bin

        *bin_edges* - locations of bin divisions, including 1 more value than
        the number of bins

        *bin_widths* - useful for passing to the ``width`` argument of
        ``matplotlib.pyplot.bar``

    References
    ----------
    [1] Torquato, S. Random Heterogeneous Materials: Mircostructure and
    Macroscopic Properties. Springer, New York (2002) - See page 45 & 292
    """
    x = chord_counts(im)
    if bins is None:
        bins = np.array(range(0, x.max()+2))*voxel_size
    x = x*voxel_size
    if log:
        x = np.log10(x)
    if normalization == 'length':
        h = list(np.histogram(x, bins=bins, density=False))
        h[0] = h[0]*(h[1][1:]+h[1][:-1])/2  # Scale bin heigths by length
        h[0] = h[0]/h[0].sum()/(h[1][1:]-h[1][:-1])  # Normalize h[0] manually
    elif normalization in ['number', 'count']:
        h = np.histogram(x, bins=bins, density=True)
    else:
        raise Exception('Unsupported normalization:', normalization)
    h = _parse_histogram(h)
    cld = namedtuple('chord_length_distribution',
                     (log*'log' + 'L', 'pdf', 'cdf', 'relfreq',
                      'bin_centers', 'bin_edges', 'bin_widths'))
    return cld(h.bin_centers, h.pdf, h.cdf, h.relfreq,
               h.bin_centers, h.bin_edges, h.bin_widths)


def region_interface_areas(regions, areas, voxel_size=1, strel=None):
    r"""
    Calculates the interfacial area between all pairs of adjecent regions

    Parameters
    ----------
    regions : ND-array
        An image of the pore space partitioned into individual pore regions.
        Note that zeros in the image will not be considered for area
        calculation.
    areas : array_like
        A list containing the areas of each regions, as determined by
        ``region_surface_area``.  Note that the region number and list index
        are offset by 1, such that the area for region 1 is stored in
        ``areas[0]``.
    voxel_size : scalar
        The resolution of the image, expressed as the length of one side of a
        voxel, so the volume of a voxel would be **voxel_size**-cubed.  The
        default is 1.
    strel : array_like
        The structuring element used to blur the region.  If not provided,
        then a spherical element (or disk) with radius 1 is used.  See the
        docstring for ``mesh_region`` for more details, as this argument is
        passed to there.

    Returns
    -------
    result : named_tuple
        A named-tuple containing 2 arrays. ``conns`` holds the connectivity
        information and ``area`` holds the result for each pair.  ``conns`` is
        a N-regions by 2 array with each row containing the region number of an
        adjacent pair of regions.  For instance, if ``conns[0, 0]`` is 0 and
        ``conns[0, 1]`` is 5, then row 0 of ``area`` contains the interfacial
        area shared by regions 0 and 5.

    """
    print('-'*60)
    print('Finding interfacial areas between each region')
    from skimage.morphology import disk, ball
    im = regions.copy()
    if im.ndim != im.squeeze().ndim:
        warnings.warn('Input image conains a singleton axis:' + str(im.shape)
                      + ' Reduce dimensionality with np.squeeze(im) to avoid'
                      + ' unexpected behavior.')
    # cube_elem = square if im.ndim == 2 else cube
    ball_elem = disk if im.ndim == 2 else ball
    # Get 'slices' into im for each region
    slices = spim.find_objects(im)
    # Initialize arrays
    Ps = np.arange(1, np.amax(im)+1)
    sa = np.zeros_like(Ps, dtype=float)
    sa_combined = []  # Difficult to preallocate since number of conns unknown
    cn = []
    # Start extracting area from im
    for i in tqdm(Ps):
        reg = i - 1
        if slices[reg] is not None:
            s = extend_slice(slices[reg], im.shape)
            sub_im = im[s]
            mask_im = sub_im == i
            sa[reg] = areas[reg]
            im_w_throats = spim.binary_dilation(input=mask_im,
                                                structure=ball_elem(1))
            im_w_throats = im_w_throats*sub_im
            Pn = np.unique(im_w_throats)[1:] - 1
            for j in Pn:
                if j > reg:
                    cn.append([reg, j])
                    merged_region = im[(min(slices[reg][0].start,
                                            slices[j][0].start)):
                                       max(slices[reg][0].stop,
                                           slices[j][0].stop),
                                       (min(slices[reg][1].start,
                                            slices[j][1].start)):
                                       max(slices[reg][1].stop,
                                           slices[j][1].stop)]
                    merged_region = ((merged_region == reg + 1)
                                     + (merged_region == j + 1))
                    mesh = mesh_region(region=merged_region, strel=strel)
                    sa_combined.append(mesh_surface_area(mesh))
    # Interfacial area calculation
    cn = np.array(cn)
    ia = 0.5 * (sa[cn[:, 0]] + sa[cn[:, 1]] - sa_combined)
    ia[ia <= 0] = 1
    result = namedtuple('interfacial_areas', ('conns', 'area'))
    result.conns = cn
    result.area = ia * voxel_size**2
    return result


def region_surface_areas(regions, voxel_size=1, strel=None):
    r"""
    Extracts the surface area of each region in a labeled image.

    Optionally, it can also find the the interfacial area between all
    adjoining regions.

    Parameters
    ----------
    regions : ND-array
        An image of the pore space partitioned into individual pore regions.
        Note that zeros in the image will not be considered for area
        calculation.
    voxel_size : scalar
        The resolution of the image, expressed as the length of one side of a
        voxel, so the volume of a voxel would be **voxel_size**-cubed.  The
        default is 1.
    strel : array_like
        The structuring element used to blur the region.  If not provided,
        then a spherical element (or disk) with radius 1 is used.  See the
        docstring for ``mesh_region`` for more details, as this argument is
        passed to there.

    Returns
    -------
    result : list
        A list containing the surface area of each region, offset by 1, such
        that the surface area of region 1 is stored in element 0 of the list.

    """
    print('-'*60)
    print('Finding surface area of each region')
    im = regions.copy()
    # Get 'slices' into im for each pore region
    slices = spim.find_objects(im)
    # Initialize arrays
    Ps = np.arange(1, np.amax(im)+1)
    sa = np.zeros_like(Ps, dtype=float)
    # Start extracting marching cube area from im
    for i in tqdm(Ps):
        reg = i - 1
        if slices[reg] is not None:
            s = extend_slice(slices[reg], im.shape)
            sub_im = im[s]
            mask_im = sub_im == i
            mesh = mesh_region(region=mask_im, strel=strel)
            sa[reg] = mesh_surface_area(mesh)
    result = sa * voxel_size**2
    return result


def mesh_surface_area(mesh=None, verts=None, faces=None):
    r"""
    Calculates the surface area of a meshed region

    Parameters
    ----------
    mesh : tuple
        The tuple returned from the ``mesh_region`` function
    verts : array
        An N-by-ND array containing the coordinates of each mesh vertex
    faces : array
        An N-by-ND array indicating which elements in ``verts`` form a mesh
        element.

    Returns
    -------
    surface_area : float
        The surface area of the mesh, calculated by
        ``skimage.measure.mesh_surface_area``

    Notes
    -----
    This function simply calls ``scikit-image.measure.mesh_surface_area``, but
    it allows for the passing of the ``mesh`` tuple returned by the
    ``mesh_region`` function, entirely for convenience.
    """
    if mesh:
        verts = mesh.verts
        faces = mesh.faces
    else:
        if (verts is None) or (faces is None):
            raise Exception('Either mesh or verts and faces must be given')
    surface_area = measure.mesh_surface_area(verts, faces)
    return surface_area


def phase_fraction(im, normed=True):
    r"""
    Calculates the number (or fraction) of each phase in an image

    Parameters
    ----------
    im : ND-array
        An ND-array containing integer values
    normed : Boolean
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

    """
    if im.dtype == bool:
        im = im.astype(int)
    elif im.dtype != int:
        raise Exception('Image must contain integer values for each phase')
    labels = np.arange(0, np.amax(im)+1)
    results = np.zeros_like(labels)
    for i in labels:
        results[i] = np.sum(im == i)
    if normed:
        results = results/im.size
    return results
