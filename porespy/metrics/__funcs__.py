import scipy as sp
from skimage.segmentation import clear_border
from skimage.feature import peak_local_max
import scipy.ndimage as spim
import scipy.spatial as sptl
from porespy.tools import get_border, extract_subsection, extend_slice


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
    for i in sp.arange(0, N):
        s = slices[i]
        p = pads[i]
        new_s = extend_slice(s, shape=im.shape, pad=p)
        temp = im[new_s]
        Vp = sp.sum(temp)
        Vt = sp.size(temp)
        porosity[i] = Vp/Vt
        volume[i] = Vt
    profile = tuple((volume, porosity))
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
    A list containing two 1D arrays: ``r`` is the radius of the voxels, and
    ``n`` is the number of voxels that are within ``r`` of the solid.

    Notes
    -----
    This function should not be taken as a pore size distribution in the
    explict sense, but rather an indicator of the sizes in the image.

    See Also
    --------
    feature_size
    porosimetry

    References
    ----------
    [1] Torquato, S. Random Heterogeneous Materials: Mircostructure and
    Macroscopic Properties. Springer, New York (2002) - See page 292
    """
    if im.dtype == bool:
        im = spim.distance_transform_edt(im)
    rdf = sp.histogram(a=im[im > 0], bins=bins)
    n = rdf[0]/sp.sum(im > 0)
    r = rdf[1][:-1]*voxel_size
    rdf = [r, n]
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
    large 3D image and/or close spacing.
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
    h2 = (h2[1][:-1], h2[0]/h1[0])
    return h2


def apply_chords(im, spacing=0, axis=0, trim_edges=True):
    r"""
    Adds chords to the void space in the specified direction.  The chords are
    separated by 1 voxel plus the provided spacing.

    Parameters
    ----------
    im : ND-array
        An image of the porous material with void marked as True.

    spacing : int (default = 0)
        Chords are automatically separated by 1 voxel and this argument
        increases the separation.

    axis : int (default = 0)
        The axis along which the chords are drawn.

    trim_edges : bool (default = True)
        Whether or not to remove chords that touch the edges of the image.
        These chords are artifically shortened, so skew the chord length
        distribution

    Returns
    -------
    An ND-array of the same size as ```im``` with True values indicating the
    chords.

    See Also
    --------
    apply_chords_3D

    """
    if spacing < 0:
        raise Exception('Spacing cannot be less than 0')
    dims1 = sp.arange(0, im.ndim)
    dims2 = sp.copy(dims1)
    dims2[axis] = 0
    dims2[0] = axis
    im = sp.moveaxis(a=im, source=dims1, destination=dims2)
    im = sp.atleast_3d(im)
    ch = sp.zeros_like(im, dtype=bool)
    if im.ndim == 2:
        ch[:, ::2+spacing, ::2+spacing] = 1
    if im.ndim == 3:
        ch[:, ::4+2*spacing, ::4+2*spacing] = 1
    chords = im*ch
    chords = sp.squeeze(chords)
    if trim_edges:
        temp = clear_border(spim.label(chords == 1)[0]) > 0
        chords = temp*chords
    chords = sp.moveaxis(a=chords, source=dims1, destination=dims2)
    return chords


def apply_chords_3D(im, spacing=0, trim_edges=True):
    r"""
    Adds chords to the void space in all three principle directions.  The
    chords are seprated by 1 voxel plus the provided spacing.  Chords in the X,
    Y and Z directions are labelled 1, 2 and 3 resepctively.

    Parameters
    ----------
    im : ND-array
        A 3D image of the porous material with void space marked as True.

    spacing : int (default = 0)
        Chords are automatically separed by 1 voxel on all sides, and this
        argument increases the separation.

    trim_edges : bool (default = True)
        Whether or not to remove chords that touch the edges of the image.
        These chords are artifically shortened, so skew the chord length
        distribution

    Returns
    -------
    An ND-array of the same size as ```im``` with values of 1 indicating
    x-direction chords, 2 indicating y-direction chords, and 3 indicating
    z-direction chords.

    Notes
    -----
    The chords are separated by a spacing of at least 1 voxel so that tools
    that search for connected components, such as ``scipy.ndimage.label`` can
    detect individual chords.

    See Also
    --------
    apply_chords

    """
    if im.ndim < 3:
        raise Exception('Must be a 3D image to use this function')
    if spacing < 0:
        raise Exception('Spacing cannot be less than 0')
    ch = sp.zeros_like(im, dtype=int)
    ch[:, ::4+2*spacing, ::4+2*spacing] = 1  # X-direction
    ch[::4+2*spacing, :, 2::4+2*spacing] = 2  # Y-direction
    ch[2::4+2*spacing, 2::4+2*spacing, :] = 3  # Z-direction
    chords = ch*im
    if trim_edges:
        temp = clear_border(spim.label(chords > 0)[0]) > 0
        chords = temp*chords
    return chords


def feature_size_distribution(im, bins=None):
    r"""
    Given an image containing the size of the feature to which each voxel
    belongs (as produced by ```simulations.feature_size```), this determines
    the total volume of each feature and returns a tuple containing *radii* and
    *counts* suitable for plotting.

    Parameters
    ----------
    im : array_like
        An array containing the local feature size

    Returns
    -------
    Tuple containing radii, counts
        Two arrays containing the radii of the largest spheres, and the number
        of voxels that are encompassed by spheres of each radii.

    Notes
    -----
    The term *foreground* is used since this function can be applied to both
    pore space or the solid, whichever is set to True.

    """
    inds = sp.where(im > 0)
    bins = sp.unique(im)[1:]
    hist = sp.histogram(a=im[inds], bins=bins)
    radii = hist[1][0:-1]
    counts = hist[0]
    return radii, counts


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


def local_thickness(im):
    r"""
    For each voxel, this functions calculates the radius of the largest sphere
    that both engulfs the voxel and fits entirely within the foreground. This
    is not the same as a simple distance transform, which finds the largest
    sphere that could be *centered* on each voxel.

    Parameters
    ----------
    im : array_like
        A binary image with the phase of interest set to True

    Returns
    -------
    An image with the pore size values in each voxel

    Notes
    -----
    The term *foreground* is used since this function can be applied to both
    pore space or the solid, whichever is set to True.

    """
    from skimage.morphology import cube
    if im.ndim == 2:
        from skimage.morphology import square as cube
    dt = spim.distance_transform_edt(im)
    sizes = sp.unique(sp.around(dt, decimals=0))
    im_new = sp.zeros_like(im, dtype=float)
    denom = int(len(sizes)/52+1)
    print('_'*60)
    print("Performing Image Opening")
    n = min(len(sizes), 52)
    print('0%|'+'-'*n+'|100%')
    print('  |', end='')
    for i in range(len(sizes)):
        if sp.mod(i, denom) == 0:
            print('|', end='')
        r = sizes[i]
        im_temp = dt >= r
        im_temp = spim.distance_transform_edt(~im_temp) <= r
        im_new[im_temp] = r
    print('|')
    # Trim outer edge of features to remove noise
    im_new = spim.binary_erosion(input=im, structure=cube(1))*im_new
    return im_new
