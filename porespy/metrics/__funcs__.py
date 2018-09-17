import scipy as sp
import numpy as np
from skimage.segmentation import clear_border
from skimage.feature import peak_local_max
import scipy.ndimage as spim
import scipy.spatial as sptl
from porespy.tools import get_border, extract_subsection, extend_slice
from porespy.filters import apply_chords
from collections import namedtuple
from tqdm import tqdm
from scipy import fftpack as sp_ft


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
    A tuple containing the ND-arrays: The subdomain *volume* and its
    *porosity*.  Each of these arrays is ``npoints`` long.  They can be
    conveniently plotted by passing the tuple to matplotlib's ``plot`` function
    using the \* notation: ``plt.plot(*the_tuple, 'b.')``.  The resulting plot
    is similar to the sketch given by Bachmat and Bear [1]

    Notes
    -----
    This function is frustratingly slow.  Profiling indicates that all the time
    is spent on scipy's ``sum`` function which is needed to sum the number of
    void voxels (1's) in each subdomain.

    Also, this function is primed for parallelization since the ``npoints`` are
    calculated independenlty.

    References
    ----------
    [1] Bachmat and Bear. On the Concept and Size of a Representative
    Elementary Volume (Rev), Advances in Transport Phenomena in Porous Media
    (1987)

    """
    im_temp = sp.zeros_like(im)
    crds = sp.array(sp.rand(npoints, im.ndim)*im.shape, dtype=int)
    pads = sp.array(sp.rand(npoints)*sp.amin(im.shape)/2+10, dtype=int)
    im_temp[tuple(crds.T)] = True
    labels, N = spim.label(input=im_temp)
    slices = spim.find_objects(input=labels)
    porosity = sp.zeros(shape=(N,), dtype=float)
    volume = sp.zeros(shape=(N,), dtype=int)
    for i in tqdm(sp.arange(0, N)):
        s = slices[i]
        p = pads[i]
        new_s = extend_slice(s, shape=im.shape, pad=p)
        temp = im[new_s]
        Vp = sp.sum(temp)
        Vt = sp.size(temp)
        porosity[i] = Vp/Vt
        volume[i] = Vt
    profile = namedtuple('profile', ('volume', 'porosity'))
    profile.volume = volume
    profile.porosity = porosity
    return profile


def porosity_profile(im, axis):
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

    """
    if axis > im.ndim:
        raise Exception('axis out of range')
    im = np.atleast_3d(im)
    a = set(range(im.ndim)).difference(set([axis]))
    a1, a2 = a
    prof = np.sum(np.sum(im, axis=a2), axis=a1)/(im.shape[a2]*im.shape[a1])
    return prof*100


def radial_density(im, bins=10, voxel_size=1):
    r"""
    Computes radial density function by analyzing the histogram of voxel
    values in the distance transform.  This function is defined by
    Torquato [1] as:

        .. math::

            \int_0^\infty P(r)dr = 1.0

    where *P(r)dr* is the probability of finding a voxel at a lying at a radial
    distance between *r* and *dr* from the solid interface.

    The cumulative distribution is defined as:

        .. math::

            F(r) = \int_r^\infty P(r)dr

    which gives the fraction of pore-space with a radius larger than *r*.

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
    A named-tuple containing several 1D arrays: ``R `` is the radius of the
    voxels (or x-axis of a pore-size density plot).  ``P`` is the radial
    density function, and ``F`` is the complementary cumulative distribution
    function.

    Notes
    -----
    This function should not be taken as a pore size distribution in the
    explict sense, but rather an indicator of the sizes in the image.  The
    distance transform contains a very skewed number of voxels with small
    values near the solid walls.  Nonetheless, it does provide a useful
    indicator and it's mathematical formalism is handy.

    Torquato refers to this as the pore-size density function, and mentions
    that it is also known as the pore-size distribution function.  These
    terms are avoided here since they have very specific connotations, and
    this function does not satisfy them.

    References
    ----------
    [1] Torquato, S. Random Heterogeneous Materials: Mircostructure and
    Macroscopic Properties. Springer, New York (2002) - See page 48 & 292
    """
    if im.dtype == bool:
        im = spim.distance_transform_edt(im)
    h = sp.histogram(a=im[im > 0], bins=bins, density=True)
    R = h[1]
    P = h[0]
    temp = P*(R[1:]-R[:-1])
    F = sp.cumsum(temp[-1::-1])[-1::-1]
    R = R*voxel_size
    rdf = namedtuple('psdf', ('R', 'P', 'F'))
    rdf.R = R
    rdf.P = P
    rdf.F = F
    return rdf


def porosity(im):
    r"""
    Calculates the porosity of an image assuming 1's are void space and 0's are
    solid phase.  All other values are ignored.

    Parameters
    ----------
    im : ND-array
        Image of the void space with 1's indicating void space (or True) and
        0's indicating the solid phase (or False).

    Returns
    -------
    porosity : float
        Calculated as the sum of all 1's divided by the sum of all 1's and 0's.

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
    im = sp.array(im, dtype=int)
    Vp = sp.sum(im == 1)
    Vs = sp.sum(im == 0)
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
    A tuple containing the x and y data for plotting the two-point correlation
    function, using the *args feature of matplotlib's plot function.  The x
    array is the distances between points and the y array is corresponding
    probabilities that points of a given distance both lie in the void space.

    The distance values are binned as follows:

        bins = range(start=0, stop=sp.amin(im.shape)/2, stride=spacing)

    Notes
    -----
    The brute-force approach means overlaying a grid of equally spaced points
    onto the image, calculating the distance between each and every pair of
    points, then counting the instances where both pairs lie in the void space.

    This approach uses a distance matrix so can consume memory very quickly for
    large 3D images and/or close spacing.
    """
    if im.ndim == 2:
        pts = sp.meshgrid(range(0, im.shape[0], spacing),
                          range(0, im.shape[1], spacing))
        crds = sp.vstack([pts[0].flatten(),
                          pts[1].flatten()]).T
    elif im.ndim == 3:
        pts = sp.meshgrid(range(0, im.shape[0], spacing),
                          range(0, im.shape[1], spacing),
                          range(0, im.shape[2], spacing))
        crds = sp.vstack([pts[0].flatten(),
                          pts[1].flatten(),
                          pts[2].flatten()]).T
    dmat = sptl.distance.cdist(XA=crds, XB=crds)
    hits = im[pts].flatten()
    dmat = dmat[hits, :]
    h1 = sp.histogram(dmat, bins=range(0, int(sp.amin(im.shape)/2), spacing))
    dmat = dmat[:, hits]
    h2 = sp.histogram(dmat, bins=h1[1])
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
    """
    if len(autocorr.shape) == 2:
        adj = sp.reshape(autocorr.shape, [2, 1, 1])
        inds = sp.indices(autocorr.shape) - adj/2
        dt = sp.sqrt(inds[0]**2 + inds[1]**2)
    elif len(autocorr.shape) == 3:
        adj = sp.reshape(autocorr.shape, [3, 1, 1, 1])
        inds = sp.indices(autocorr.shape) - adj/2
        dt = sp.sqrt(inds[0]**2 + inds[1]**2 + inds[2]**2)
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
    A tuple containing the x and y data for plotting the two-point correlation
    function, using the *args feature of matplotlib's plot function.  The x
    array is the distances between points and the y array is corresponding
    probabilities that points of a given distance both lie in the void space.

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
    P = sp.absolute(F**2)
    # Auto-correlation is inverse of Power Spectrum
    autoc = sp.absolute(sp_ft.ifftshift(sp_ft.ifftn(sp_ft.fftshift(P))))
    tpcf = _radial_profile(autoc, r_max=np.min(hls))
    return tpcf


def pore_size_distribution(im):
    r"""
    Calculate drainage curve based on the image produced by the
    ``porosimetry`` function.

    Parameters
    ----------
    im : ND-array
        The array of entry sizes as produced by the ``porosimetry`` function.

    Returns
    -------
    Rp, Snwp: Two arrays containing (a) the radius of the penetrating
    sphere (in voxels) and (b) the volume fraction of pore phase voxels
    that are accessible from the specfied inlets.

    Notes
    -----
    This function normalizes the invading phase saturation by total pore
    volume of the dry image, which is assumed to be all voxels with a value
    equal to 1.  To do porosimetry on images with large outer regions,
    use the ```find_outer_region``` function then set these regions to 0 in
    the input image.  In future, this function could be adapted to apply
    this check by default.

    """
    sizes = sp.unique(im)
    R = []
    Snwp = []
    Vp = sp.sum(im > 0)
    for r in sizes[1:]:
        R.append(r)
        Snwp.append(sp.sum(im >= r))
    Snwp = [s/Vp for s in Snwp]
    data = namedtuple('xy_data', ('radius', 'saturation'))
    return data(R, Snwp)


def chord_length_counts(im):
    r"""
    Determines the length of each chord in the supplied image by looking at
    its size.

    Parameters
    ----------
    im : ND-array
        An image containing chords drawn in the void space.

    Returns
    -------
    A 1D array with one element for each chord, containing its length.

    Notes
    ----
    The returned array can be passed to ``plt.hist`` to plot the histogram,
    or to ``sp.histogram`` to get the histogram data directly. Another useful
    function is ``sp.bincount`` which gives the number of chords of each
    length in a format suitable for ``plt.plot``.
    """
    labels, N = spim.label(im > 0)
    slices = spim.find_objects(labels)
    chord_lens = sp.zeros(N, dtype=int)
    for i in range(len(slices)):
        s = slices[i]
        chord_lens[i] = sp.amax([item.stop-item.start for item in s])
    return chord_lens


def chord_length_distribution(im, bins=25, log=False):
    r"""
    Determines the distribution of chord lengths in a image containing chords.

    Parameters
    ----------
    im : ND-image
        An image with chords drawn in the pore space, as produced by
        ``apply_chords`` or ``apply_chords_3d``.

    bins : scalar or array_like
        If a scalar is given it is interpreted as the number of bins to use,
        and if an array is given they are used as the bins directly.

    log : Boolean
        If true, the logarithm of the chord lengths will be used, which can
        make the data more clear.

    Returns
    -------
    A tuple containing the ``chord_length_bins``, and four separate pieces of
    information: ``cumulative_chord_count`` and ``cumulative_chord_length``,
    as well as the ``differenial_chord_count`` and
    ``differential_chord_length``.
    """
    h = chord_length_counts(im)
    if log:
        h = sp.log10(h)
    y_num, x = sp.histogram(h, bins=bins, density=True)
    y_len, x = sp.histogram(h, bins=bins, weights=h, density=True)
    y_num_cum = sp.cumsum((y_num*(x[1:]-x[:-1]))[::-1])[::-1]
    y_len_cum = sp.cumsum((y_len*(x[1:]-x[:-1]))[::-1])[::-1]
    data = namedtuple('chord_distribution', ('chord_length_bins',
                                             'cumulative_chord_count',
                                             'cumulative_chord_length',
                                             'differential_chord_count',
                                             'differential_chord_length'))
    return data(x[:-1], y_num_cum, y_len_cum, y_num, y_len)


def cld_helper(im, axes=(0, 1, 2), spacing=0, trim_edges=True):
    r"""
    Takes a binary image and returns a weighted coord length distribution in
    the three principle directions as presented in (ref my paper).

    Parameters
    ----------
    im : ND-image
        An image of the porous material with phase pof interest marked as True

    axes : tuple or int (default = (0,1,2))
        A tuple containing the desired direction for the chords to  be
        applied and counted. This argument is identical to the axis argument in
        filters.apply_chords except it accepts a single integer or a tuple of
        integers.

    spacing : int (default = 0)
        Chords are automatically separated by 1 voxel and this argument
        increases the separation.

    trim_edges : bool (default = True)
        Whether or not to remove chords that touch the edges of the image.
        These chords are artifically shortened, so skew the chord length
        distribution

    Returns
    -------
    A list of 1D arrays representing the chord length distibutions for each
    axis as specified by axes. The length of each array is equal to 1 + the
    size of the largest chord in that direction:
        For example if the longest chord in x has length 100, then length(cx) =
        101.
    The chord lengths have also been normalized according to the total length
    of chords in that direction:
        For example if the total chords drawn in the x direction is equal to
        1000, and there are 3 chords of length 4, then cx[3] = 0.12. In this
        way the integral of the curve provided is always 1.
    """
    if isinstance(axes, int):
        axes = (axes,)
    res = []
    for ax in axes:
        c = apply_chords(im, axis=ax, spacing=spacing, trim_edges=trim_edges)
        c = chord_length_counts(c)
        c = np.bincount(c)
        c = c*range(c.size)/(c*range(c.size)).sum()
        res.append(c)
    return res
