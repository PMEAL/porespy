import numpy as np
from edt import edt
import porespy as ps
import scipy.spatial as sptl
import scipy.ndimage as spim
from numba import njit, jit
from porespy.tools import norm_to_uniform, ps_ball, ps_disk, get_border
from porespy.tools import insert_sphere, fftmorphology
from typing import List
from numpy import array
from tqdm import tqdm
from skimage.morphology import ball, disk


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
                raise Exception('Cannot specify center point when element '
                                + 'has one or more even dimension')
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


def RSA(im: array, radius: int, volume_fraction: int = 1, n_max: int = None,
        mode: str = 'contained'):
    r"""
    Generates a sphere or disk packing using Random Sequential Addition

    This algorithm ensures that spheres do not overlap but does not
    guarantee they are tightly packed.

    This function adds spheres to the background of the received ``im``, which
    allows iteratively adding spheres of different radii to the unfilled space,
    be repeatedly passing in the result of previous calls to RSA.

    Parameters
    ----------
    im : ND-array
        The image into which the spheres should be inserted.  By accepting an
        image rather than a shape, it allows users to insert spheres into an
        already existing image.  To begin the process, start with an array of
        zeros such as ``im = np.zeros([200, 200, 200], dtype=bool)``.
    radius : int
        The radius of the disk or sphere to insert.
    volume_fraction : scalar (default is 1.0)
        The fraction of the image that should be filled with spheres.  The
        spheres are added as 1's, so each sphere addition increases the
        ``volume_fraction`` until the specified limit is reach.  Note that if
        ``n_max`` is reached first, then ``volume_fraction`` will not be
        acheived.
    n_max : int (default is 10,000)
        The maximum number of spheres to add.  By default the value of
        ``n_max`` is high so that the addition of spheres will go indefinately
        until ``volume_fraction`` is met, but specifying a smaller value
        will halt addition after the given number of spheres are added.
    mode : string (default is 'contained')
        Controls how the edges of the image are handled.  Options are:

        'contained' - Spheres are all completely within the image

        'extended' - Spheres are allowed to extend beyond the edge of the
        image.  In this mode the volume fraction will be less that requested
        since some spheres extend beyond the image, but their entire volume
        is counted as added for computational efficiency.

    Returns
    -------
    image : ND-array
        A handle to the input ``im`` with spheres of specified radius
        *added* to the background.

    Notes
    -----
    This function uses Numba to speed up the search for valid sphere insertion
    points.  It seems that Numba does not look at the state of the scipy
    random number generator, so setting the seed to a known value has no
    effect on the output of this function. Each call to this function will
    produce a unique image.  If you wish to use the same realization multiple
    times you must save the array (e.g. ``numpy.save``).

    References
    ----------
    [1] Random Heterogeneous Materials, S. Torquato (2001)

    """
    print(80*'-')
    print(f'RSA: Adding spheres of size {radius}')
    im = im.astype(bool)
    if n_max is None:
        n_max = 10000
    vf_final = volume_fraction
    vf_start = im.sum()/im.size
    print('Initial volume fraction:', vf_start)
    if im.ndim == 2:
        template_lg = ps_disk(radius*2)
        template_sm = ps_disk(radius)
    else:
        template_lg = ps_ball(radius*2)
        template_sm = ps_ball(radius)
    vf_template = template_sm.sum()/im.size
    # Pad image by the radius of large template to enable insertion near edges
    im = np.pad(im, pad_width=2*radius, mode='edge')
    # Depending on mode, adjust mask to remove options around edge
    if mode == 'contained':
        border = get_border(im.shape, thickness=2*radius, mode='faces')
    elif mode == 'extended':
        border = get_border(im.shape, thickness=radius+1, mode='faces')
    else:
        raise Exception('Unrecognized mode: ', mode)
    # Remove border pixels
    im[border] = True
    # Dilate existing objects by strel to remove pixels near them
    # from consideration for sphere placement
    print('Dilating foreground features by sphere radius')
    dt = edt(im == 0)
    options_im = (dt >= radius)
    # ------------------------------------------------------------------------
    # Begin inserting the spheres
    vf = vf_start
    free_sites = np.flatnonzero(options_im)
    i = 0
    while (vf <= vf_final) and (i < n_max) and (len(free_sites) > 0):
        c, count = _make_choice(options_im, free_sites=free_sites)
        # The 100 below is arbitrary and may change performance
        if count > 100:
            # Regenerate list of free_sites
            print('Regenerating free_sites after', i, 'iterations')
            free_sites = np.flatnonzero(options_im)
        if all(np.array(c) == -1):
            break
        s_sm = tuple([slice(x - radius, x + radius + 1, None) for x in c])
        s_lg = tuple([slice(x - 2*radius, x + 2*radius + 1, None) for x in c])
        im[s_sm] += template_sm  # Add ball to image
        options_im[s_lg][template_lg] = False  # Update extended region
        vf += vf_template
        i += 1
    print('Number of spheres inserted:', i)
    # ------------------------------------------------------------------------
    # Get slice into returned image to retain original size
    s = tuple([slice(2*radius, d-2*radius, None) for d in im.shape])
    im = im[s]
    vf = im.sum()/im.size
    print('Final volume fraction:', vf)
    return im


@njit
def _make_choice(options_im, free_sites):
    r"""
    This function is called by _begin_inserting to find valid insertion points

    Parameters
    ----------
    options_im : ND-array
        An array with ``True`` at all valid locations and ``False`` at all
        locations where a sphere already exists PLUS a region of radius R
        around each sphere since these points are also invalid insertion
        points.
    free_sites : array_like
        A 1D array containing valid insertion indices.  This list is used to
        select insertion points from a limited which occasionally gets
        smaller.

    Returns
    -------
    coords : list
        The XY or XYZ coordinates of the next insertion point
    count : int
        The number of attempts that were needed to find valid point.  If
        this value gets too high, a short list of ``free_sites`` should be
        generated in the calling function.

    """
    choice = False
    count = 0
    upper_limit = len(free_sites)
    max_iters = upper_limit*20
    if options_im.ndim == 2:
        coords = [-1, -1]
        Nx, Ny = options_im.shape
        while not choice:
            if count >= max_iters:
                coords = [-1, -1]
                break
            ind = np.random.randint(0, upper_limit)
            # This numpy function is not supported by numba yet
            # c1, c2 = np.unravel_index(free_sites[ind], options_im.shape)
            # So using manual unraveling
            coords[1] = free_sites[ind] % Ny
            coords[0] = (free_sites[ind] // Ny) % Nx
            choice = options_im[coords[0], coords[1]]
            count += 1
    if options_im.ndim == 3:
        coords = [-1, -1, -1]
        Nx, Ny, Nz = options_im.shape
        while not choice:
            if count >= max_iters:
                coords = [-1, -1, -1]
                break
            ind = np.random.randint(0, upper_limit)
            # This numpy function is not supported by numba yet
            # c1, c2, c3 = np.unravel_index(free_sites[ind], options_im.shape)
            # So using manual unraveling
            coords[2] = free_sites[ind] % Nz
            coords[1] = (free_sites[ind] // Nz) % Ny
            coords[0] = (free_sites[ind] // (Nz * Ny)) % Nx
            choice = options_im[coords[0], coords[1], coords[2]]
            count += 1
    return coords, count


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
    im = edt(~im) > radius
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
    im = ~(edt(~im) < r)
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
    def f(N):
        return edt(im > N/bulk_vol) < radius

    def g(im):
        r"""Returns fraction of 0s, given a binary image"""
        return 1 - im.sum() / np.prod(shape)

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


def perlin_noise(shape: List[int], porosity=None, octaves: int = 3,
                 frequency: List[int] = 2, persistence: float = 0.5):
    r"""
    Generate a Perlin noise field

    Parameters
    ----------
    shape : array_like
        The shape of the desired image
    frequncy : array_like
        Controls the frequency of the noise, with higher values leading to
        smaller features or more tightly spaced undulations in the brightness.
    porosity : float
        If specified, the returned image will be thresholded to the specified
        porosity.  If not provided, the greyscale noise is returned (default).
    octaves : int
        Controls the texture of the noise, with higher values giving more
        comlex features of larger length scales.
    persistence : float
        Controls how prominent each successive octave is.  Shoul be a number
        less than 1.

    Returns
    -------
    An ND-array of the specified ``shape``.  If ``porosity`` is not given
    then the array contains greyscale values distributed normally about 0.
    Use ``porespy.tools.norm_to_uniform`` to create an well-scale image for
    thresholding.  If ``porosity`` is given then these steps are done
    internally and a boolean image is returned.

    Notes
    -----
    The implementation used here is a bit fussy about the values of
    ``frequency`` and ``octaves``.  (1) the image ``shape`` must an integer
    multiple of ``frequency`` in each direction, and (2) ``frequency`` to the
    power of ``octaves`` must be less than or equal the``shape`` in each
    direction.  Exceptions are thrown if these conditions are not met.

    References
    ----------
    This implementation is taken from Pierre Vigier's
    `Github repo <https://github.com/pvigier/perlin-numpy>`_

    """
    # Parse args
    shape = np.array(shape)
    if shape.size == 1:  # Assume 3D
        shape = np.ones(3, dtype=int)*shape
    res = np.array(frequency)
    if res.size == 1:  # Assume shape as shape
        res = np.ones(shape.size, dtype=int)*res

    # Check inputs for various sins
    if res.size != shape.size:
        raise Exception('shape and res must have same dimensions')
    if np.any(np.mod(shape, res) > 0):
        raise Exception('res must be a multiple of shape along each axis')
    if np.any(shape/res**octaves < 1):
        raise Exception('(res[i])**octaves must be <= shape[i]')
    check = shape/(res**octaves)
    if np.any(check % 1):
        raise Exception("Image size must be factor of res**octaves")

    # Generate noise
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in tqdm(range(octaves)):
        if noise.ndim == 2:
            noise += amplitude * _perlin_noise_2D(shape, frequency*res)
        elif noise.ndim == 3:
            noise += amplitude * _perlin_noise_3D(shape, frequency*res)
        frequency *= 2
        amplitude *= persistence

    if porosity is not None:
        noise = norm_to_uniform(noise, scale=[0, 1])
        noise = noise > porosity

    return noise


def _perlin_noise_3D(shape, res):
    def f(t):
        return 6*t**5 - 15*t**4 + 10*t**3

    delta = res / shape
    d = shape // res
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1], 0:res[2]:delta[2]]
    grid = grid.transpose(1, 2, 3, 0) % 1
    # Gradients
    theta = 2*np.pi*np.random.rand(*(res + 1))
    phi = 2*np.pi*np.random.rand(*(res + 1))
    gradients = np.stack((np.sin(phi)*np.cos(theta),
                          np.sin(phi)*np.sin(theta),
                          np.cos(phi)), axis=3)
    g000 = gradients[0:-1, 0:-1, 0:-1]
    g000 = g000.repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g100 = gradients[1:, 0:-1, 0:-1]
    g100 = g100.repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g010 = gradients[0:-1, 1:, 0:-1]
    g010 = g010.repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g110 = gradients[1:, 1:, 0:-1]
    g110 = g110.repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g001 = gradients[0:-1, 0:-1, 1:]
    g001 = g001.repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g101 = gradients[1:, 0:-1, 1:]
    g101 = g101.repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g011 = gradients[0:-1, 1:, 1:]
    g011 = g011.repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g111 = gradients[1:, 1:, 1:]
    g111 = g111.repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    # Ramps
    n000 = np.sum(np.stack((grid[..., 0],
                            grid[..., 1],
                            grid[..., 2]), axis=3)*g000, 3)
    n100 = np.sum(np.stack((grid[..., 0]-1,
                            grid[..., 1],
                            grid[..., 2]), axis=3)*g100, 3)
    n010 = np.sum(np.stack((grid[..., 0],
                            grid[..., 1]-1,
                            grid[..., 2]), axis=3)*g010, 3)
    n110 = np.sum(np.stack((grid[..., 0]-1,
                            grid[..., 1]-1,
                            grid[..., 2]), axis=3)*g110, 3)
    n001 = np.sum(np.stack((grid[..., 0],
                            grid[..., 1],
                            grid[..., 2]-1), axis=3)*g001, 3)
    n101 = np.sum(np.stack((grid[..., 0]-1,
                            grid[..., 1],
                            grid[..., 2]-1), axis=3)*g101, 3)
    n011 = np.sum(np.stack((grid[..., 0],
                            grid[..., 1]-1,
                            grid[..., 2]-1), axis=3)*g011, 3)
    n111 = np.sum(np.stack((grid[..., 0]-1,
                            grid[..., 1]-1,
                            grid[..., 2]-1), axis=3)*g111, 3)
    # Interpolation
    t = f(grid)
    n00 = n000*(1-t[..., 0]) + t[..., 0]*n100
    n10 = n010*(1-t[..., 0]) + t[..., 0]*n110
    n01 = n001*(1-t[..., 0]) + t[..., 0]*n101
    n11 = n011*(1-t[..., 0]) + t[..., 0]*n111
    n0 = (1-t[..., 1])*n00 + t[..., 1]*n10
    n1 = (1-t[..., 1])*n01 + t[..., 1]*n11
    return ((1-t[..., 2])*n0 + t[..., 2]*n1)


def _perlin_noise_2D(shape, res):
    def f(t):
        return 6*t**5 - 15*t**4 + 10*t**3

    delta = res / shape
    d = shape // res
    grid = np.mgrid[0:res[0]:delta[0],
                    0:res[1]:delta[1]].transpose(1, 2, 0) % 1

    # Gradients
    angles = 2*np.pi*np.random.rand(res[0]+1, res[1]+1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = gradients[0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:, 1:].repeat(d[0], 0).repeat(d[1], 1)

    # Ramps
    n00 = np.sum(np.dstack((grid[..., 0], grid[..., 1])) * g00, 2)
    n10 = np.sum(np.dstack((grid[..., 0]-1, grid[..., 1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[..., 0], grid[..., 1]-1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[..., 0]-1, grid[..., 1]-1)) * g11, 2)

    # Interpolation
    t = f(grid)
    n0 = n00*(1-t[:, :, 0]) + t[:, :, 0]*n10
    n1 = n01*(1-t[:, :, 0]) + t[:, :, 0]*n11

    return np.sqrt(2)*((1-t[:, :, 1])*n0 + t[:, :, 1]*n1)


def blobs(shape: List[int], porosity: float = 0.5, blobiness: int = 1,
          **kwargs):
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
    parallel = kwargs.pop('parallel', False)
    divs = kwargs.pop('divs', 2)
    cores = kwargs.pop('cores', None)
    if np.size(shape) == 1:
        shape = np.full((3, ), int(shape))
    sigma = np.mean(shape)/(40*blobiness)
    im = np.random.random(shape)
    if parallel:
        # TODO: The determination of the overlap should be done rigorously
        im = ps.filters.chunked_func(func=spim.gaussian_filter,
                                     input=im, sigma=sigma,
                                     divs=divs, cores=cores, overlap=10)
    else:
        im = spim.gaussian_filter(im, sigma=sigma)
    im = norm_to_uniform(im, scale=[0, 1])
    if porosity:
        im = im < porosity
    return im


def cylinders(shape: List[int], radius: int, ncylinders: int,
              phi_max: float = 0, theta_max: float = 90, length: float = None):
    r"""
    Generates a binary image of overlapping cylinders.

    This is a good approximation of a fibrous mat.

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
    phi_max : scalar
        A value between 0 and 90 that controls the amount that the cylinders
        lie *out of* the XY plane, with 0 meaning all cylinders lie in the XY
        plane, and 90 meaning that cylinders are randomly oriented out of the
        plane by as much as +/- 90 degrees.
    theta_max : scalar
        A value between 0 and 90 that controls the amount of rotation *in the*
        XY plane, with 0 meaning all cylinders point in the X-direction, and
        90 meaning they are randomly rotated about the Z axis by as much
        as +/- 90 degrees.
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
    # Find hypotenuse of domain from [0,0,0] to [Nx,Ny,Nz]
    H = np.sqrt(np.sum(np.square(shape))).astype(int)
    if length is None:  # Assume cylinders span domain if length not given
        length = 2*H
    R = min(int(length/2), 2*H)  # Trim given length to 2H if too long
    # Adjust max angles to be between 0 and 90
    if (phi_max > 90) or (phi_max < 0):
        raise Exception('phi_max must be betwen 0 and 90')
    if (theta_max > 90) or (theta_max < 0):
        raise Exception('theta_max must be betwen 0 and 90')
    # Create empty image for inserting into
    im = np.zeros(shape, dtype=bool)
    n = 0
    L = min(H, R)
    with tqdm(range(1, ncylinders)) as pbar:
        while n < ncylinders:
            # Choose a random starting point in domain
            x = np.random.rand(3)*(shape + 2*L)
            # Chose a random phi and theta within given ranges
            phi = (np.pi/2 - np.pi*np.random.rand())*phi_max/90
            theta = (np.pi/2 - np.pi*np.random.rand())*theta_max/90
            X0 = R*np.array([np.cos(phi)*np.cos(theta),
                             np.cos(phi)*np.sin(theta),
                             np.sin(phi)])
            [X0, X1] = [x + X0, x - X0]
            crds = line_segment(X0, X1)
            lower = ~np.any(np.vstack(crds).T < [L, L, L], axis=1)
            upper = ~np.any(np.vstack(crds).T >= shape + L, axis=1)
            valid = upper*lower
            if np.any(valid):
                im[crds[0][valid] - L, crds[1][valid] - L, crds[2][valid] - L] = 1
                n += 1
                pbar.update()
    im = np.array(im, dtype=bool)
    dt = edt(~im) < radius
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
