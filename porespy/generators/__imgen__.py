import porespy as ps
import numpy as np
import scipy.spatial as sptl
import scipy.ndimage as spim
from porespy.tools import norm_to_uniform, ps_ball, ps_disk
from typing import List



def insert_shape(im, element, center=None, corner=None, value=1,
                 mode='overwrite'):
    r"""
    Inserts sub-image into a larger image at the specified location.

    If the inserted image extends beyond the boundaries of the image it will
    be cropped accordingly.

    Parameters
    ----------
    im : ND-array
        The image into which the sub-image will be inserted
    element : ND-array
        The sub-image to insert
    center : tuple
        Coordinates indicating the position in the main image where the
        inserted imaged will be centered.  If ``center`` is given then
        ``corner`` cannot be specified.  Note that ``center`` can only be
        used if all dimensions of ``element`` are odd, otherwise the meaning
        of center is not defined.
    corner : tuple
        Coordinates indicating the position in the main image where the
        lower corner (i.e. [0, 0, 0]) of the inserted image should be anchored.
        If ``corner`` is given then ``corner`` cannot be specified.
    value : scalar
        A scalar value to apply to the sub-image.  The default is 1.
    mode : string
        If 'overwrite' (default) the inserted image replaces the values in the
        main image.  If 'overlay' the inserted image is added to the main
        image.  In both cases the inserted image is multiplied by ``value``
        first.

    Returns
    -------
    im : ND-array
        A copy of ``im`` with the supplied element inserted.

    """
    im = im.copy()
    if im.ndim != element.ndim:
        raise Exception('Image shape ' + str(im.shape)
                        + ' and element shape ' + str(element.shape)
                        + ' do not match')
    s_im = []
    s_el = []
    if (center is not None) and (corner is None):
        for dim in range(im.ndim):
            r, d = np.divmod(element.shape[dim], 2)
            if d == 0:
                raise Exception('Cannot specify center point when element ' +
                                'has one or more even dimension')
            lower_im = np.amax((center[dim] - r, 0))
            upper_im = np.amin((center[dim] + r + 1, im.shape[dim]))
            s_im.append(slice(lower_im, upper_im))
            lower_el = np.amax((lower_im - center[dim] + r, 0))
            upper_el = np.amin((upper_im - center[dim] + r,
                                element.shape[dim]))
            s_el.append(slice(lower_el, upper_el))
    elif (corner is not None) and (center is None):
        for dim in range(im.ndim):
            L = int(element.shape[dim])
            lower_im = np.amax((corner[dim], 0))
            upper_im = np.amin((corner[dim] + L, im.shape[dim]))
            s_im.append(slice(lower_im, upper_im))
            lower_el = np.amax((lower_im - corner[dim], 0))
            upper_el = np.amin((upper_im - corner[dim],
                                element.shape[dim]))
            s_el.append(slice(min(lower_el, upper_el), upper_el))
    else:
        raise Exception('Cannot specify both corner and center')

    if mode == 'overlay':
        im[tuple(s_im)] = im[tuple(s_im)] + element[tuple(s_el)]*value
    elif mode == 'overwrite':
        im[tuple(s_im)] = element[tuple(s_el)]*value
    else:
        raise Exception('Invalid mode ' + mode)
    return im


def RSA(im: np.array, radius: int, volume_fraction: int = 1,
        mode: str = 'extended'):
    r"""
    Generates a sphere or disk packing using Random Sequential Addition

    This which ensures that spheres do not overlap but does not guarantee they
    are tightly packed.

    Parameters
    ----------
    im : ND-array
        The image into which the spheres should be inserted.  By accepting an
        image rather than a shape, it allows users to insert spheres into an
        already existing image.  To begin the process, start with an array of
        zero such as ``im = np.zeros([200, 200], dtype=bool)``.
    radius : int
        The radius of the disk or sphere to insert.
    volume_fraction : scalar
        The fraction of the image that should be filled with spheres.  The
        spheres are addeds 1's, so each sphere addition increases the
        ``volume_fraction`` until the specified limit is reach.
    mode : string
        Controls how the edges of the image are handled.  Options are:

        'extended' - Spheres are allowed to extend beyond the edge of the image

        'contained' - Spheres are all completely within the image

        'periodic' - The portion of a sphere that extends beyond the image is
        inserted into the opposite edge of the image (Not Implemented Yet!)

    Returns
    -------
    image : ND-array
        A copy of ``im`` with spheres of specified radius *added* to the
        background.

    Notes
    -----
    Each sphere is filled with 1's, but the center is marked with a 2.  This
    allows easy boolean masking to extract only the centers, which can be
    converted to coordinates using ``scipy.where`` and used for other purposes.
    The obtain only the spheres, use``im = im == 1``.

    This function adds spheres to the background of the received ``im``, which
    allows iteratively adding spheres of different radii to the unfilled space.

    References
    ----------
    [1] Random Heterogeneous Materials, S. Torquato (2001)

    """
    # Note: The 2D vs 3D splitting of this just me being lazy...I can't be
    # bothered to figure it out programmatically right now
    # TODO: Ideally the spheres should be added periodically
    print(78*'―')
    print('RSA: Adding spheres of size ' + str(radius))
    d2 = len(im.shape) == 2
    mrad = 2*radius
    if d2:
        im_strel = ps_disk(radius)
        mask_strel = ps_disk(mrad)
    else:
        im_strel = ps_ball(radius)
        mask_strel = ps_ball(mrad)
    if np.any(im > 0):
        # Dilate existing objects by im_strel to remove pixels near them
        # from consideration for sphere placement
        mask = ps.tools.fftmorphology(im > 0, im_strel > 0, mode='dilate')
        mask = mask.astype(int)
    else:
        mask = np.zeros_like(im)
    if mode == 'contained':
        mask = _remove_edge(mask, radius)
    elif mode == 'extended':
        pass
    elif mode == 'periodic':
        raise Exception('Periodic edges are not implemented yet')
    else:
        raise Exception('Unrecognized mode: ' + mode)
    vf = im.sum()/im.size
    free_spots = np.argwhere(mask == 0)
    i = 0
    while vf <= volume_fraction and len(free_spots) > 0:
        choice = np.random.randint(0, len(free_spots), size=1)
        if d2:
            [x, y] = free_spots[choice].flatten()
            im = _fit_strel_to_im_2d(im, im_strel, radius, x, y)
            mask = _fit_strel_to_im_2d(mask, mask_strel, mrad, x, y)
            im[x, y] = 2
        else:
            [x, y, z] = free_spots[choice].flatten()
            im = _fit_strel_to_im_3d(im, im_strel, radius, x, y, z)
            mask = _fit_strel_to_im_3d(mask, mask_strel, mrad, x, y, z)
            im[x, y, z] = 2
        free_spots = np.argwhere(mask == 0)
        vf = im.sum()/im.size
        i += 1
    if vf > volume_fraction:
        print('Volume Fraction', volume_fraction, 'reached')
    if len(free_spots) == 0:
        print('No more free spots', 'Volume Fraction', vf)
    return im


def bundle_of_tubes(shape: List[int], spacing: int):
    r"""
    Create a 3D image of a bundle of tubes, in the form of a rectangular
    plate with randomly sized holes through it.

    Parameters
    ----------
    shape : list
        The size the image, with the 3rd dimension indicating the plate
        thickness.  If the 3rd dimension is not given then a thickness of
        1 voxel is assumed.

    spacing : scalar
        The center to center distance of the holes.  The hole sizes will be
        randomly distributed between this values down to 3 voxels.

    Returns
    -------
    image : ND-array
        A boolean array with ``True`` values denoting the pore space
    """
    shape = np.array(shape)
    if np.size(shape) == 1:
        shape = np.full((3, ), int(shape))
    if np.size(shape) == 2:
        shape = np.hstack((shape, [1]))
    temp = np.zeros(shape=shape[:2])
    Xi = np.ceil(np.linspace(spacing/2,
                             shape[0]-(spacing/2)-1,
                             int(shape[0]/spacing)))
    Xi = np.array(Xi, dtype=int)
    Yi = np.ceil(np.linspace(spacing/2,
                             shape[1]-(spacing/2)-1,
                             int(shape[1]/spacing)))
    Yi = np.array(Yi, dtype=int)
    temp[tuple(np.meshgrid(Xi, Yi))] = 1
    inds = np.where(temp)
    for i in range(len(inds[0])):
        r = np.random.randint(1, (spacing/2))
        try:
            s1 = slice(inds[0][i]-r, inds[0][i]+r+1)
            s2 = slice(inds[1][i]-r, inds[1][i]+r+1)
            temp[s1, s2] = ps_disk(r)
        except ValueError:
            odd_shape = np.shape(temp[s1, s2])
            temp[s1, s2] = ps_disk(r)[:odd_shape[0], :odd_shape[1]]
    im = np.broadcast_to(array=np.atleast_3d(temp), shape=shape)
    return im


def polydisperse_spheres(shape: List[int], porosity: float, dist,
                         nbins: int = 5, r_min: int = 5):
    r"""
    Create an image of randomly place, overlapping spheres with a distribution
    of radii.

    Parameters
    ----------
    shape : list
        The size of the image to generate in [Nx, Ny, Nz] where Ni is the
        number of voxels in each direction.  If shape is only 2D, then an
        image of polydisperse disks is returns

    porosity : scalar
        The porosity of the image, defined as the number of void voxels
        divided by the number of voxels in the image. The specified value
        is only matched approximately, so it's suggested to check this value
        after the image is generated.

    dist : scipy.stats distribution object
        This should be an initialized distribution chosen from the large number
        of options in the ``scipy.stats`` submodule.  For instance, a normal
        distribution with a mean of 20 and a standard deviation of 10 can be
        obtained with ``dist = scipy.stats.norm(loc=20, scale=10)``

    nbins : scalar
        The number of discrete sphere sizes that will be used to generate the
        image.  This function generates  ``nbins`` images of monodisperse
        spheres that span 0.05 and 0.95 of the possible values produced by the
        provided distribution, then overlays them to get polydispersivity.

    Returns
    -------
    image : ND-array
        A boolean array with ``True`` values denoting the pore space
    """
    shape = np.array(shape)
    if np.size(shape) == 1:
        shape = np.full((3, ), int(shape))
    Rs = dist.interval(np.linspace(0.05, 0.95, nbins))
    Rs = np.vstack(Rs).T
    Rs = (Rs[:-1] + Rs[1:])/2
    Rs = np.clip(Rs.flatten(), a_min=r_min, a_max=None)
    phi_desired = 1 - (1 - porosity)/(len(Rs))
    im = np.ones(shape, dtype=bool)
    for r in Rs:
        phi_im = im.sum() / np.prod(shape)
        phi_corrected = 1 - (1 - phi_desired) / phi_im
        temp = overlapping_spheres(shape=shape, radius=r, porosity=phi_corrected)
        im = im * temp
    return im


def voronoi_edges(shape: List[int], radius: int, ncells: int,
                  flat_faces: bool = True):
    r"""
    Create an image of the edges in a Voronoi tessellation

    Parameters
    ----------
    shape : array_like
        The size of the image to generate in [Nx, Ny, Nz] where Ni is the
        number of voxels in each direction.

    radius : scalar
        The radius to which Voronoi edges should be dilated in the final image.

    ncells : scalar
        The number of Voronoi cells to include in the tesselation.

    flat_faces : Boolean
        Whether the Voronoi edges should lie on the boundary of the
        image (True), or if edges outside the image should be removed (False).

    Returns
    -------
    image : ND-array
        A boolean array with ``True`` values denoting the pore space

    """
    print(78*'―')
    print('voronoi_edges: Generating', ncells, 'cells')
    shape = np.array(shape)
    if np.size(shape) == 1:
        shape = np.full((3, ), int(shape))
    im = np.zeros(shape, dtype=bool)
    base_pts = np.random.rand(ncells, 3)*shape
    if flat_faces:
        # Reflect base points
        Nx, Ny, Nz = shape
        orig_pts = base_pts
        base_pts = np.vstack((base_pts,
                              [-1, 1, 1] * orig_pts + [2.0*Nx, 0, 0]))
        base_pts = np.vstack((base_pts,
                              [1, -1, 1] * orig_pts + [0, 2.0*Ny, 0]))
        base_pts = np.vstack((base_pts,
                              [1, 1, -1] * orig_pts + [0, 0, 2.0*Nz]))
        base_pts = np.vstack((base_pts, [-1, 1, 1] * orig_pts))
        base_pts = np.vstack((base_pts, [1, -1, 1] * orig_pts))
        base_pts = np.vstack((base_pts, [1, 1, -1] * orig_pts))
    vor = sptl.Voronoi(points=base_pts)
    vor.vertices = np.around(vor.vertices)
    vor.vertices *= (np.array(im.shape)-1) / np.array(im.shape)
    vor.edges = _get_Voronoi_edges(vor)
    for row in vor.edges:
        pts = vor.vertices[row].astype(int)
        if np.all(pts >= 0) and np.all(pts < im.shape):
            line_pts = line_segment(pts[0], pts[1])
            im[tuple(line_pts)] = True
    im = spim.distance_transform_edt(~im) > radius
    return im


def _get_Voronoi_edges(vor):
    r"""
    Given a Voronoi object as produced by the scipy.spatial.Voronoi class,
    this function calculates the start and end points of eeach edge in the
    Voronoi diagram, in terms of the vertex indices used by the received
    Voronoi object.

    Parameters
    ----------
    vor : scipy.spatial.Voronoi object

    Returns
    -------
    A 2-by-N array of vertex indices, indicating the start and end points of
    each vertex in the Voronoi diagram.  These vertex indices can be used to
    index straight into the ``vor.vertices`` array to get spatial positions.
    """
    edges = [[], []]
    for facet in vor.ridge_vertices:
        # Create a closed cycle of vertices that define the facet
        edges[0].extend(facet[:-1]+[facet[-1]])
        edges[1].extend(facet[1:]+[facet[0]])
    edges = np.vstack(edges).T  # Convert to scipy-friendly format
    mask = np.any(edges == -1, axis=1)  # Identify edges at infinity
    edges = edges[~mask]  # Remove edges at infinity
    edges = np.sort(edges, axis=1)  # Move all points to upper triangle
    # Remove duplicate pairs
    edges = edges[:, 0] + 1j*edges[:, 1]  # Convert to imaginary
    edges = np.unique(edges)  # Remove duplicates
    edges = np.vstack((np.real(edges), np.imag(edges))).T  # Back to real
    edges = np.array(edges, dtype=int)
    return edges


def lattice_spheres(shape: List[int], radius: int, offset: int = 0,
                    lattice: str = 'sc'):
    r"""
    Generates a cubic packing of spheres in a specified lattice arrangement

    Parameters
    ----------
    shape : list
        The size of the image to generate in [Nx, Ny, Nz] where N is the
        number of voxels in each direction.  For a 2D image, use [Nx, Ny].

    radius : scalar
        The radius of spheres (circles) in the packing

    offset : scalar
        The amount offset (+ or -) to add between sphere centers.

    lattice : string
        Specifies the type of lattice to create.  Options are:

        'sc' - Simple Cubic (default)

        'fcc' - Face Centered Cubic

        'bcc' - Body Centered Cubic

        For 2D images, 'sc' gives a square lattice and both 'fcc' and 'bcc'
        give a triangular lattice.

    Returns
    -------
    image : ND-array
        A boolean array with ``True`` values denoting the pore space
    """
    print(78*'―')
    print('lattice_spheres: Generating ' + lattice + ' lattice')
    r = radius
    shape = np.array(shape)
    if np.size(shape) == 1:
        shape = np.full((3, ), int(shape))
    im = np.zeros(shape, dtype=bool)
    im = im.squeeze()

    # Parse lattice type
    lattice = lattice.lower()
    if im.ndim == 2:
        if lattice in ['sc']:
            lattice = 'sq'
        if lattice in ['bcc', 'fcc']:
            lattice = 'tri'

    if lattice in ['sq', 'square']:
        spacing = 2*r
        s = int(spacing/2) + np.array(offset)
        coords = np.mgrid[r:im.shape[0]-r:2*s,
                          r:im.shape[1]-r:2*s]
        im[coords[0], coords[1]] = 1
    elif lattice in ['tri', 'triangular']:
        spacing = 2*np.floor(np.sqrt(2*(r**2))).astype(int)
        s = int(spacing/2) + offset
        coords = np.mgrid[r:im.shape[0]-r:2*s,
                          r:im.shape[1]-r:2*s]
        im[coords[0], coords[1]] = 1
        coords = np.mgrid[s+r:im.shape[0]-r:2*s,
                          s+r:im.shape[1]-r:2*s]
        im[coords[0], coords[1]] = 1
    elif lattice in ['sc', 'simple cubic', 'cubic']:
        spacing = 2*r
        s = int(spacing/2) + np.array(offset)
        coords = np.mgrid[r:im.shape[0]-r:2*s,
                          r:im.shape[1]-r:2*s,
                          r:im.shape[2]-r:2*s]
        im[coords[0], coords[1], coords[2]] = 1
    elif lattice in ['bcc', 'body cenetered cubic']:
        spacing = 2*np.floor(np.sqrt(4/3*(r**2))).astype(int)
        s = int(spacing/2) + offset
        coords = np.mgrid[r:im.shape[0]-r:2*s,
                          r:im.shape[1]-r:2*s,
                          r:im.shape[2]-r:2*s]
        im[coords[0], coords[1], coords[2]] = 1
        coords = np.mgrid[s+r:im.shape[0]-r:2*s,
                          s+r:im.shape[1]-r:2*s,
                          s+r:im.shape[2]-r:2*s]
        im[coords[0], coords[1], coords[2]] = 1
    elif lattice in ['fcc', 'face centered cubic']:
        spacing = 2*np.floor(np.sqrt(2*(r**2))).astype(int)
        s = int(spacing/2) + offset
        coords = np.mgrid[r:im.shape[0]-r:2*s,
                          r:im.shape[1]-r:2*s,
                          r:im.shape[2]-r:2*s]
        im[coords[0], coords[1], coords[2]] = 1
        coords = np.mgrid[r:im.shape[0]-r:2*s,
                          s+r:im.shape[1]-r:2*s,
                          s+r:im.shape[2]-r:2*s]
        im[coords[0], coords[1], coords[2]] = 1
        coords = np.mgrid[s+r:im.shape[0]-r:2*s,
                          s:im.shape[1]-r:2*s,
                          s+r:im.shape[2]-r:2*s]
        im[coords[0], coords[1], coords[2]] = 1
        coords = np.mgrid[s+r:im.shape[0]-r:2*s,
                          s+r:im.shape[1]-r:2*s,
                          s:im.shape[2]-r:2*s]
        im[coords[0], coords[1], coords[2]] = 1
    im = ~(spim.distance_transform_edt(~im) < r)
    return im


def overlapping_spheres(shape: List[int], radius: int, porosity: float,
                        iter_max: int = 10, tol: float = 0.01):
    r"""
    Generate a packing of overlapping mono-disperse spheres

    Parameters
    ----------
    shape : list
        The size of the image to generate in [Nx, Ny, Nz] where Ni is the
        number of voxels in the i-th direction.

    radius : scalar
        The radius of spheres in the packing.

    porosity : scalar
        The porosity of the final image, accurate to the given tolerance.

    iter_max : int
        Maximum number of iterations for the iterative algorithm that improves
        the porosity of the final image to match the given value.

    tol : float
        Tolerance for porosity of the final image compared to the given value.

    Returns
    -------
    image : ND-array
        A boolean array with ``True`` values denoting the pore space

    Notes
    -----
    This method can also be used to generate a dispersion of hollows by
    treating ``porosity`` as solid volume fraction and inverting the
    returned image.

    """
    shape = np.array(shape)
    if np.size(shape) == 1:
        shape = np.full((3, ), int(shape))
    ndim = (shape != 1).sum()
    s_vol = ps_disk(radius).sum() if ndim == 2 else ps_ball(radius).sum()

    bulk_vol = np.prod(shape)
    N = int(np.ceil((1 - porosity)*bulk_vol/s_vol))
    im = np.random.random(size=shape)

    # Helper functions for calculating porosity: phi = g(f(N))
    f = lambda N: spim.distance_transform_edt(im > N/bulk_vol) < radius
    g = lambda im: 1 - im.sum() / np.prod(shape)

    # # Newton's method for getting image porosity match the given
    # w = 1.0                         # Damping factor
    # dN = 5 if ndim == 2 else 25     # Perturbation
    # for i in range(iter_max):
    #     err = g(f(N)) - porosity
    #     d_err = (g(f(N+dN)) - g(f(N))) / dN
    #     if d_err == 0:
    #         break
    #     if abs(err) <= tol:
    #         break
    #     N2 = N - int(err/d_err)   # xnew = xold - f/df
    #     N = w * N2 + (1-w) * N

    # Bisection search: N is always undershoot (bc. of overlaps)
    N_low, N_high = N, 4*N
    for i in range(iter_max):
        N = np.mean([N_high, N_low], dtype=int)
        err = g(f(N)) - porosity
        if err > 0:
            N_low = N
        else:
            N_high = N
        if abs(err) <= tol:
            break

    return ~f(N)


def generate_noise(shape: List[int], porosity=None, octaves: int = 3,
                   frequency: int = 32, mode: str = 'simplex'):
    r"""
    Generate a field of spatially correlated random noise using the Perlin
    noise algorithm, or the updated Simplex noise algorithm.

    Parameters
    ----------
    shape : array_like
        The size of the image to generate in [Nx, Ny, Nz] where N is the
        number of voxels.

    porosity : float
        If specified, this will threshold the image to the specified value
        prior to returning.  If no value is given (the default), then the
        scalar noise field is returned.

    octaves : int
        Controls the *texture* of the noise, with higher octaves giving more
        complex features over larger length scales.

    frequency : array_like
        Controls the relative sizes of the features, with higher frequencies
        giving larger features.  A scalar value will apply the same frequency
        in all directions, given an isotropic field; a vector value will
        apply the specified values along each axis to create anisotropy.

    mode : string
        Which noise algorithm to use, either ``'simplex'`` (default) or
        ``'perlin'``.

    Returns
    -------
    image : ND-array
        If porosity is given, then a boolean array with ``True`` values
        denoting the pore space is returned.  If not, then normally
        distributed and spatially correlated randomly noise is returned.

    Notes
    -----
    This method depends the a package called 'noise' which must be
    compiled. It is included in the Anaconda distribution, or a platform
    specific binary can be downloaded.

    See Also
    --------
    porespy.tools.norm_to_uniform

    """
    try:
        import noise
    except ModuleNotFoundError:
        raise Exception("The noise package must be installed")
    shape = np.array(shape)
    if np.size(shape) == 1:
        Lx, Ly, Lz = np.full((3, ), int(shape))
    elif len(shape) == 2:
        Lx, Ly = shape
        Lz = 1
    elif len(shape) == 3:
        Lx, Ly, Lz = shape
    if mode == 'simplex':
        f = noise.snoise3
    else:
        f = noise.pnoise3
    frequency = np.atleast_1d(frequency)
    if frequency.size == 1:
        freq = np.full(shape=[3, ], fill_value=frequency[0])
    elif frequency.size == 2:
        freq = np.concatenate((frequency, [1]))
    else:
        freq = np.array(frequency)
    im = np.zeros(shape=[Lx, Ly, Lz], dtype=float)
    for x in range(Lx):
        for y in range(Ly):
            for z in range(Lz):
                im[x, y, z] = f(x=x/freq[0], y=y/freq[1], z=z/freq[2],
                                octaves=octaves)
    im = im.squeeze()
    if porosity:
        im = norm_to_uniform(im, scale=[0, 1])
        im = im < porosity
    return im


def blobs(shape: List[int], porosity: float = 0.5, blobiness: int = 1):
    """
    Generates an image containing amorphous blobs

    Parameters
    ----------
    shape : list
        The size of the image to generate in [Nx, Ny, Nz] where N is the
        number of voxels

    porosity : float
        If specified, this will threshold the image to the specified value
        prior to returning.  If ``None`` is specified, then the scalar noise
        field is converted to a uniform distribution and returned without
        thresholding.

    blobiness : int or list of ints(default = 1)
        Controls the morphology of the blobs.  A higher number results in
        a larger number of small blobs.  If a list is supplied then the blobs
        are anisotropic.

    Returns
    -------
    image : ND-array
        A boolean array with ``True`` values denoting the pore space

    See Also
    --------
    norm_to_uniform

    """
    blobiness = np.array(blobiness)
    shape = np.array(shape)
    if np.size(shape) == 1:
        shape = np.full((3, ), int(shape))
    sigma = np.mean(shape)/(40*blobiness)
    im = np.random.random(shape)
    im = spim.gaussian_filter(im, sigma=sigma)
    im = norm_to_uniform(im, scale=[0, 1])
    if porosity:
        im = im < porosity
    return im


def cylinders(shape: List[int], radius: int, ncylinders: int,
              phi_max: float = 0, theta_max: float = 90, length: float = None):
    r"""
    Generates a binary image of overlapping cylinders.  This is a good
    approximation of a fibrous mat.

    Parameters
    ----------
    shape : list
        The size of the image to generate in [Nx, Ny, Nz] where N is the
        number of voxels. 2D images are not permitted.
    radius : scalar
        The radius of the cylinders in voxels
    ncylinders : scalar
        The number of cylinders to add to the domain. Adjust this value to
        control the final porosity, which is not easily specified since
        cylinders overlap and intersect different fractions of the domain.
    theta_max : scalar
        A value between 0 and 90 that controls the amount of rotation *in the*
        XY plane, with 0 meaning all cylinders point in the X-direction, and
        90 meaning they are randomly rotated about the Z axis by as much
        as +/- 90 degrees.
    phi_max : scalar
        A value between 0 and 90 that controls the amount that the cylinders
        lie *out of* the XY plane, with 0 meaning all cylinders lie in the XY
        plane, and 90 meaning that cylinders are randomly oriented out of the
        plane by as much as +/- 90 degrees.
    length : scalar
        The length of the cylinders to add.  If ``None`` (default) then the
        cylinders will extend beyond the domain in both directions so no ends
        will exist. If a scalar value is given it will be interpreted as the
        Euclidean distance between the two ends of the cylinder.  Note that
        one or both of the ends *may* still lie outside the domain, depending
        on the randomly chosen center point of the cylinder.

    Returns
    -------
    image : ND-array
        A boolean array with ``True`` values denoting the pore space
    """
    shape = np.array(shape)
    if np.size(shape) == 1:
        shape = np.full((3, ), int(shape))
    elif np.size(shape) == 2:
        raise Exception("2D cylinders don't make sense")
    if length is None:
        R = np.sqrt(np.sum(np.square(shape))).astype(int)
    else:
        R = length/2
    im = np.zeros(shape)
    # Adjust max angles to be between 0 and 90
    if (phi_max > 90) or (phi_max < 0):
        raise Exception('phi_max must be betwen 0 and 90')
    if (theta_max > 90) or (theta_max < 0):
        raise Exception('theta_max must be betwen 0 and 90')
    n = 0
    while n < ncylinders:
        # Choose a random starting point in domain
        x = np.random.rand(3)*shape
        # Chose a random phi and theta within given ranges
        phi = (np.pi/2 - np.pi*np.random.rand())*phi_max/90
        theta = (np.pi/2 - np.pi*np.random.rand())*theta_max/90
        X0 = R*np.array([np.cos(phi)*np.cos(theta),
                         np.cos(phi)*np.cos(theta),
                         np.cos(phi)])
        [X0, X1] = [x + X0, x - X0]
        crds = line_segment(X0, X1)
        lower = ~np.any(np.vstack(crds).T < [0, 0, 0], axis=1)
        upper = ~np.any(np.vstack(crds).T >= shape, axis=1)
        valid = upper*lower
        if np.any(valid):
            im[crds[0][valid], crds[1][valid], crds[2][valid]] = 1
            n += 1
    im = np.array(im, dtype=bool)
    dt = spim.distance_transform_edt(~im) < radius
    return ~dt


def line_segment(X0, X1):
    r"""
    Calculate the voxel coordinates of a straight line between the two given
    end points

    Parameters
    ----------
    X0 and X1 : array_like
        The [x, y] or [x, y, z] coordinates of the start and end points of
        the line.

    Returns
    -------
    coords : list of lists
        A list of lists containing the X, Y, and Z coordinates of all voxels
        that should be drawn between the start and end points to create a solid
        line.
    """
    X0 = np.around(X0).astype(int)
    X1 = np.around(X1).astype(int)
    if len(X0) == 3:
        L = np.amax(np.absolute([[X1[0]-X0[0]], [X1[1]-X0[1]], [X1[2]-X0[2]]])) + 1
        x = np.rint(np.linspace(X0[0], X1[0], L)).astype(int)
        y = np.rint(np.linspace(X0[1], X1[1], L)).astype(int)
        z = np.rint(np.linspace(X0[2], X1[2], L)).astype(int)
        return [x, y, z]
    else:
        L = np.amax(np.absolute([[X1[0]-X0[0]], [X1[1]-X0[1]]])) + 1
        x = np.rint(np.linspace(X0[0], X1[0], L)).astype(int)
        y = np.rint(np.linspace(X0[1], X1[1], L)).astype(int)
        return [x, y]


def _fit_strel_to_im_2d(im, strel, r, x, y):
    r"""
    Helper function to add a structuring element to a 2D image.
    Used by RSA. Makes sure if center is less than r pixels from edge of image
    that the strel is sliced to fit.
    """
    elem = strel.copy()
    x_dim, y_dim = im.shape
    x_min = x-r
    x_max = x+r+1
    y_min = y-r
    y_max = y+r+1
    if x_min < 0:
        x_adj = -x_min
        elem = elem[x_adj:, :]
        x_min = 0
    elif x_max > x_dim:
        x_adj = x_max - x_dim
        elem = elem[:-x_adj, :]
    if y_min < 0:
        y_adj = -y_min
        elem = elem[:, y_adj:]
        y_min = 0
    elif y_max > y_dim:
        y_adj = y_max - y_dim
        elem = elem[:, :-y_adj]
    ex, ey = elem.shape
    im[x_min:x_min+ex, y_min:y_min+ey] += elem
    return im


def _fit_strel_to_im_3d(im, strel, r, x, y, z):
    r"""
    Helper function to add a structuring element to a 2D image.
    Used by RSA. Makes sure if center is less than r pixels from edge of image
    that the strel is sliced to fit.
    """
    elem = strel.copy()
    x_dim, y_dim, z_dim = im.shape
    x_min = x-r
    x_max = x+r+1
    y_min = y-r
    y_max = y+r+1
    z_min = z-r
    z_max = z+r+1
    if x_min < 0:
        x_adj = -x_min
        elem = elem[x_adj:, :, :]
        x_min = 0
    elif x_max > x_dim:
        x_adj = x_max - x_dim
        elem = elem[:-x_adj, :, :]
    if y_min < 0:
        y_adj = -y_min
        elem = elem[:, y_adj:, :]
        y_min = 0
    elif y_max > y_dim:
        y_adj = y_max - y_dim
        elem = elem[:, :-y_adj, :]
    if z_min < 0:
        z_adj = -z_min
        elem = elem[:, :, z_adj:]
        z_min = 0
    elif z_max > z_dim:
        z_adj = z_max - z_dim
        elem = elem[:, :, :-z_adj]
    ex, ey, ez = elem.shape
    im[x_min:x_min+ex, y_min:y_min+ey, z_min:z_min+ez] += elem
    return im


def _remove_edge(im, r):
    r'''
    Fill in the edges of the input image.
    Used by RSA to ensure that no elements are placed too close to the edge.
    '''
    edge = np.ones_like(im)
    if len(im.shape) == 2:
        sx, sy = im.shape
        edge[r:sx-r, r:sy-r] = im[r:sx-r, r:sy-r]
    else:
        sx, sy, sz = im.shape
        edge[r:sx-r, r:sy-r, r:sz-r] = im[r:sx-r, r:sy-r, r:sz-r]
    return edge
