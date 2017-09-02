import scipy as sp
import numpy as np
from skimage.segmentation import clear_border
from skimage.feature import peak_local_max
import scipy.ndimage as spim
import scipy.spatial as sptl
from porespy.tools import get_border, extract_subsection, extend_slice
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


def radial_distribution(im, bins=10):
    r"""

    References
    ----------
    [1] Torquato, S. Random Heterogeneous Materials: Mircostructure and
    Macroscopic Properties. Springer, New York (2002)
    """
    print("Sorry, but this function is not implemented yet")


def lineal_path(im):
    r"""

    References
    ----------
    [1] Torquato, S. Random Heterogeneous Materials: Mircostructure and
    Macroscopic Properties. Springer, New York (2002)
    """
    print("Sorry, but this function is not implemented yet")


def pore_size_density(im, bins=10, voxel_size=1):
    r"""
    Computes the histogram of the distance transform as an estimator of the
    pore sizes in the image.

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
    A tuple containing two 1D arrays: ``radius `` is the radius of the voxels,
    and ``count`` is the number of voxels that are within R of the solid.

    Notes
    -----
    This function should not be taken as a pore size distribution in the
    explict sense, but rather an indicator of the sizes in the image.

    References
    ----------
    [1] Torquato, S. Random Heterogeneous Materials: Mircostructure and
    Macroscopic Properties. Springer, New York (2002) - See page 292
    """
    if im.dtype == bool:
        im = spim.distance_transform_edt(im)
    hist = sp.histogram(a=im[im > 0], bins=bins)
    n = hist[0]/sp.sum(im > 0)
    r = hist[1][:-1]*voxel_size
    rdf = namedtuple('rdf', ('radius', 'count'))
    return rdf(r, n)


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

    References
    ----------
    [1] Torquato, S. Random Heterogeneous Materials: Mircostructure and
    Macroscopic Properties. Springer, New York (2002)
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
    inds = sp.indices(autocorr.shape) - sp.reshape(autocorr.shape, [2, 1, 1])/2
    dt = sp.sqrt(inds[0]**2 + inds[1]**2)
    bin_size = np.int(np.ceil(r_max/nbins))
    bins = np.arange(bin_size, r_max, step=bin_size)
    radial_sum = np.zeros_like(bins)
    for i, r in enumerate(bins):
        # Generate Radial Mask from dt using bins
        mask = (dt <= r) * (dt > (r-bin_size))
        radial_sum[i] = np.sum(autocorr[mask])/np.sum(mask)
    # Return normalized bin and radially summed autoc
    norm_bins = bins/np.max(bins)
    norm_autoc_radial = radial_sum/np.max(radial_sum)
    tpcf = namedtuple('two_point_correlation_function',
                      ('distance', 'probability'))
    return tpcf(norm_bins, norm_autoc_radial)


def two_point_correlation_fft(image, pad=False):
    r"""
    Calculates the two-point correlation function using fourier transforms

    Parameters
    ----------
    im : ND-array
        The image of the void space on which the 2-point correlation is desired

    pad : bool
        The image is padded with Trues or 1's depending on dtype around border

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
    hls = (np.ceil(np.shape(image))/2).astype(int)
    if pad:
        # Pad image boundaries with ones
        dtype = image.dtype
        ish = np.shape(image)
        off = hls + ish
        if len(ish) == 2:
            pad_im = np.ones(shape=[2*ish[0], 2*ish[1]], dtype=dtype)
            pad_im[hls[0]:off[0], hls[1]:off[1]] = image
        elif len(ish) == 3:
            pad_im = np.ones(shape=[2*ish[0], 2*ish[1], 2*ish[2]], dtype=dtype)
            pad_im[hls[0]:off[0], hls[1]:off[1], hls[2]:off[2]] = image
        image = pad_im
    # Fourier Transform and shift image
    F = sp_ft.ifftshift(sp_ft.fftn(sp_ft.fftshift(image)))
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


def chord_length_distribution(im):
    r"""
    Determines the length of each chord in the supplied image by looking at
    its size.

    Parameters
    ----------
    im : ND-array
        An image containing chords drawn in the void space.

    Returns
    -------
    A 1D array with one element for each chord, containing the length.

    Notes
    ----
    The returned array can be passed to ```plt.hist``` to plot the histogram,
    or to ```sp.histogram``` to get the histogram data directly. Another useful
    function is ```sp.bincount``` which gives the number of chords of each
    length in a format suitable for ```plt.plot```.
    """
    labels, N = spim.label(im)
    slices = spim.find_objects(labels)
    chord_lens = sp.zeros(N, dtype=int)
    for i in range(len(slices)):
        s = slices[i]
        chord_lens[i] = sp.amax([item.stop-item.start for item in s])
    return chord_lens
