import porespy as ps
import scipy as sp
import scipy.spatial as sptl
import scipy.ndimage as spim
from skimage.segmentation import find_boundaries
from skimage.morphology import ball, disk, square, cube
from tqdm import tqdm
from porespy.tools import norm_to_uniform


def insert_shape(im, center, element, value=1):
    r"""
    """
    if im.ndim != element.ndim:
        raise Exception('Image shape ' + str(im.shape) +
                        ' and element shape ' + str(element.shape) +
                        ' do not match')
    s_im = []
    s_el = []
    for dim in range(im.ndim):
        r = int(element.shape[dim]/2)
        lower_im = sp.amax((center[dim] - r, 0))
        upper_im = sp.amin((center[dim] + r + 1, im.shape[dim]))
        s_im.append(slice(lower_im, upper_im))
        lower_el = sp.amax((lower_im - center[dim] + r, 0))
        upper_el = sp.amin((upper_im - center[dim] + r, element.shape[dim]))
        s_el.append(slice(lower_el, upper_el))
    im[s_im] = im[s_im] + element[s_el]*value
    return im


def RSA(im, radius, volume_fraction=1, mode='extended'):
    r"""
    Generates a sphere or disk packing using Random Sequential Addition, which
    ensures that spheres do not overlap but does not guarantee they are
    tightly packed.

    Each sphere is filled with 1's, and the center is marked with a 2.  This
    allows easy boolean masking to extract only the centers, which can be
    converted to coordinates using ``scipy.where`` and used for other purposes.

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
    mrad = 2*radius + 1
    if d2:
        im_strel = disk(radius)
        mask_strel = disk(mrad)
    else:
        im_strel = ball(radius)
        mask_strel = ball(mrad)
    if sp.any(im > 0):
        mask = ps.tools.fftmorphology(im > 0, im_strel > 0, mode='dilate')
        mask = mask.astype(int)
    else:
        mask = sp.zeros_like(im)
    if mode == 'contained':
        mask = _remove_edge(mask, radius)
    elif mode == 'extended':
        pass
    elif mode == 'periodic':
        raise Exception('Periodic edges are not implemented yet')
    else:
        raise Exception('Unrecognized mode: ' + mode)
    vf = im.sum()/im.size
    free_spots = sp.argwhere(mask == 0)
    i = 0
    while vf <= volume_fraction and len(free_spots) > 0:
        choice = sp.random.randint(0, len(free_spots), size=1)
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
        free_spots = sp.argwhere(mask == 0)
        vf = im.sum()/im.size
        i += 1
    if vf > volume_fraction:
        print('Volume Fraction', volume_fraction, 'reached')
    if len(free_spots) == 0:
        print('No more free spots', 'Volume Fraction', vf)
    return im


def bundle_of_tubes(shape, spacing):
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
    A boolean array with True values denoting the pore space
    """
    shape = sp.array(shape)
    if sp.size(shape) == 1:
        shape = sp.full((3, ), int(shape))
    if sp.size(shape) == 2:
        shape = sp.hstack((shape, [1]))
    temp = sp.zeros(shape=shape[:2])
    Xi = sp.ceil(sp.linspace(spacing/2,
                             shape[0]-(spacing/2)-1,
                             shape[0]/spacing))
    Xi = sp.array(Xi, dtype=int)
    Yi = sp.ceil(sp.linspace(spacing/2,
                             shape[1]-(spacing/2)-1,
                             shape[1]/spacing))
    Yi = sp.array(Yi, dtype=int)
    temp[sp.meshgrid(Xi, Yi)] = 1
    inds = sp.where(temp)
    for i in range(len(inds[0])):
        r = sp.random.randint(1, (spacing/2))
        try:
            s1 = slice(inds[0][i]-r, inds[0][i]+r+1)
            s2 = slice(inds[1][i]-r, inds[1][i]+r+1)
            temp[s1, s2] = disk(r)
        except ValueError:
            odd_shape = sp.shape(temp[s1, s2])
            temp[s1, s2] = disk(r)[:odd_shape[0], :odd_shape[1]]
    im = sp.broadcast_to(array=sp.atleast_3d(temp), shape=shape)
    return im


def polydisperse_spheres(shape, porosity, dist, nbins=5):
    r"""
    Create an image of spheres with a distribution of radii.

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
        This should be an initialized distribution the large number of options
        in the scipy.stats submodule.  For instance, a normal distribution with
        a mean of 20 and a standard deviation of 10 can be obtain with
        ``dist = scipy.stats.norm(loc=20, scale=10)``

    nbins : scalar
        The number of discrete sphere sizes that will be used to generate the
        image.  This function generates  ``nbins`` images of monodisperse
        spheres that span 0.05 and 0.95 of the possible values produced by the
        provided distribution, then overlays them to get polydispersivity.

    Returns
    -------
    A boolean array with True values denoting the pore space
    """
    shape = sp.array(shape)
    if sp.size(shape) == 1:
        shape = sp.full((3, ), int(shape))
    Rs = dist.interval(sp.linspace(0.05, 0.95, nbins))
    Rs = sp.vstack(Rs).T
    Rs = (Rs[:-1] + Rs[1:])/2
    Rs = Rs.flatten()
    phi = 1 - (1 - porosity)/(len(Rs))
    im = sp.ones(shape, dtype=bool)
    for r in Rs:
        temp = overlapping_spheres(shape=shape, radius=r, porosity=phi)
        im = im*temp
    return im


def voronoi_edges(shape, radius, ncells, flat_faces=True):
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
    A boolean array with True values denoting the pore space

    """
    print(78*'―')
    print('voronoi_edges: Generating', ncells, ' cells')
    shape = sp.array(shape)
    if sp.size(shape) == 1:
        shape = sp.full((3, ), int(shape))
    im = sp.zeros(shape, dtype=bool)
    base_pts = sp.rand(ncells, 3)*shape
    if flat_faces:
        # Reflect base points
        Nx, Ny, Nz = shape
        orig_pts = base_pts
        base_pts = sp.vstack((base_pts, [-1, 1, 1]*orig_pts +
                                        [2.0*Nx, 0, 0]))
        base_pts = sp.vstack((base_pts, [1, -1, 1]*orig_pts +
                                        [0, 2.0*Ny, 0]))
        base_pts = sp.vstack((base_pts, [1, 1, -1]*orig_pts +
                                        [0, 0, 2.0*Nz]))
        base_pts = sp.vstack((base_pts, [-1, 1, 1]*orig_pts))
        base_pts = sp.vstack((base_pts, [1, -1, 1]*orig_pts))
        base_pts = sp.vstack((base_pts, [1, 1, -1]*orig_pts))
    vor = sptl.Voronoi(points=base_pts)
    vor.vertices = sp.around(vor.vertices)
    vor.vertices *= (sp.array(im.shape)-1)/sp.array(im.shape)
    vor.edges = _get_Voronoi_edges(vor)
    for row in vor.edges:
        pts = vor.vertices[row].astype(int)
        if sp.all(pts >= 0) and sp.all(pts < im.shape):
            line_pts = line_segment(pts[0], pts[1])
            im[line_pts] = True
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
    edges = sp.vstack(edges).T  # Convert to scipy-friendly format
    mask = sp.any(edges == -1, axis=1)  # Identify edges at infinity
    edges = edges[~mask]  # Remove edges at infinity
    edges = sp.sort(edges, axis=1)  # Move all points to upper triangle
    # Remove duplicate pairs
    edges = edges[:, 0] + 1j*edges[:, 1]  # Convert to imaginary
    edges = sp.unique(edges)  # Remove duplicates
    edges = sp.vstack((sp.real(edges), sp.imag(edges))).T  # Back to real
    edges = sp.array(edges, dtype=int)
    return edges


def lattice_spheres(shape, radius, offset=0, lattice='sc'):
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

        'sc' : Simple Cubic (default)
        'fcc' : Face Centered Cubic
        'bcc' : Body Centered Cubic

        For 2D images, 'sc' gives a square lattice and both 'fcc' and 'bcc'
        give a triangular lattice.

    Returns
    -------
    A boolean array with True values denoting the pore space
    """
    print(78*'―')
    print('lattice_spheres: Generating ' + lattice + ' lattice')
    r = radius
    shape = sp.array(shape)
    if sp.size(shape) == 1:
        shape = sp.full((3, ), int(shape))
    im = sp.zeros(shape, dtype=bool)
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
        s = int(spacing/2) + sp.array(offset)
        coords = sp.mgrid[r:im.shape[0]-r:2*s,
                          r:im.shape[1]-r:2*s]
        im[coords[0], coords[1]] = 1
    elif lattice in ['tri', 'triangular']:
        spacing = 2*sp.floor(sp.sqrt(2*(r**2))).astype(int)
        s = int(spacing/2) + offset
        coords = sp.mgrid[r:im.shape[0]-r:2*s,
                          r:im.shape[1]-r:2*s]
        im[coords[0], coords[1]] = 1
        coords = sp.mgrid[s+r:im.shape[0]-r:2*s,
                          s+r:im.shape[1]-r:2*s]
        im[coords[0], coords[1]] = 1
    elif lattice in ['sc', 'simple cubic', 'cubic']:
        spacing = 2*r
        s = int(spacing/2) + sp.array(offset)
        coords = sp.mgrid[r:im.shape[0]-r:2*s,
                          r:im.shape[1]-r:2*s,
                          r:im.shape[2]-r:2*s]
        im[coords[0], coords[1], coords[2]] = 1
    elif lattice in ['bcc', 'body cenetered cubic']:
        spacing = 2*sp.floor(sp.sqrt(4/3*(r**2))).astype(int)
        s = int(spacing/2) + offset
        coords = sp.mgrid[r:im.shape[0]-r:2*s,
                          r:im.shape[1]-r:2*s,
                          r:im.shape[2]-r:2*s]
        im[coords[0], coords[1], coords[2]] = 1
        coords = sp.mgrid[s+r:im.shape[0]-r:2*s,
                          s+r:im.shape[1]-r:2*s,
                          s+r:im.shape[2]-r:2*s]
        im[coords[0], coords[1], coords[2]] = 1
    elif lattice in ['fcc', 'face centered cubic']:
        spacing = 2*sp.floor(sp.sqrt(2*(r**2))).astype(int)
        s = int(spacing/2) + offset
        coords = sp.mgrid[r:im.shape[0]-r:2*s,
                          r:im.shape[1]-r:2*s,
                          r:im.shape[2]-r:2*s]
        im[coords[0], coords[1], coords[2]] = 1
        coords = sp.mgrid[r:im.shape[0]-r:2*s,
                          s+r:im.shape[1]-r:2*s,
                          s+r:im.shape[2]-r:2*s]
        im[coords[0], coords[1], coords[2]] = 1
        coords = sp.mgrid[s+r:im.shape[0]-r:2*s,
                          s:im.shape[1]-r:2*s,
                          s+r:im.shape[2]-r:2*s]
        im[coords[0], coords[1], coords[2]] = 1
        coords = sp.mgrid[s+r:im.shape[0]-r:2*s,
                          s+r:im.shape[1]-r:2*s,
                          s:im.shape[2]-r:2*s]
        im[coords[0], coords[1], coords[2]] = 1
    im = ~(spim.distance_transform_edt(~im) < r)
    return im


def overlapping_spheres(shape, radius, porosity):
    r"""
    Generate a packing of overlapping mono-disperse spheres

    Parameters
    ----------
    shape : list
        The size of the image to generate in [Nx, Ny, Nz] where Ni is the
        number of voxels in the *i*th direction.

    radius : scalar
        The radius of spheres in the packing.

    porosity : scalar
        The porosity of the final image.  This number is approximated by
        the method so the returned result may not have exactly the
        specified value.

    Returns
    -------
    A boolean array with True values denoting the pore space

    Notes
    -----
    This method can also be used to generate a dispersion of hollows by
    treating ``porosity`` as solid volume fraction and inverting the
    returned image.
    """
    shape = sp.array(shape)
    if sp.size(shape) == 1:
        shape = sp.full((3, ), int(shape))
    if sp.size(shape) == 2:
        s_vol = sp.sum(disk(radius))
    if sp.size(shape) == 3:
        s_vol = sp.sum(ball(radius))
    bulk_vol = sp.prod(shape)
    N = int(sp.ceil((1 - porosity)*bulk_vol/s_vol))
    im = sp.random.random(size=shape) > (N/bulk_vol)
    im = spim.distance_transform_edt(im) < radius
    return ~im


def noise(shape, porosity=None, octaves=3, frequency=32, mode='simplex'):
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
        Which noise algorithm to use, either \'simplex\' (default) or
        \'perlin\'.

    Returns
    -------
    If porosity is given, then a boolean array with ``True`` values denoting
    the pore space is returned.  If not, then normally distributed and
    spatially correlated randomly noise is returned.

    Notes
    -----
    This method depends the a package called 'noise' which must be
    compiled. It is included in the Anaconda distribution, or a platform
    specific binary can be downloaded.

    See Also
    --------
    norm_to_uniform

    """
    try:
        import noise
    except ModuleNotFoundError:
        raise Exception("The noise package must be installed")
    shape = sp.array(shape)
    if sp.size(shape) == 1:
        Lx, Ly, Lz = sp.full((3, ), int(shape))
    elif len(shape) == 2:
        Lx, Ly = shape
        Lz = 1
    elif len(shape) == 3:
        Lx, Ly, Lz = shape
    if mode == 'simplex':
        f = noise.snoise3
    else:
        f = noise.pnoise3
    frequency = sp.atleast_1d(frequency)
    if frequency.size == 1:
        freq = sp.full(shape=[3, ], fill_value=frequency[0])
    elif frequency.size == 2:
        freq = sp.concatenate((frequency, [1]))
    else:
        freq = sp.array(frequency)
    im = sp.zeros(shape=[Lx, Ly, Lz], dtype=float)
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


def blobs(shape, porosity=0.5, blobiness=1):
    """
    Generates an image containing amorphous blobs

    Parameters
    ----------
    shape : list
        The size of the image to generate in [Nx, Ny, Nz] where N is the
        number of voxels

    porosity : float
        If specified, this will threshold the image to the specified value
        prior to returning.  If no value is given (the default), then the
        scalar noise field is returned.

    blobiness : array_like (default = 1)
        Controls the morphology of the blobs.  A higher number results in
        a larger number of small blobs.  If a vector is supplied then the blobs
        are anisotropic.

    Returns
    -------
    If porosity is given, then a boolean array with ``True`` values denoting
    the pore space is returned.  If not, then normally distributed and
    spatially correlated randomly noise is returned.

    See Also
    --------
    norm_to_uniform

    """
    blobiness = sp.array(blobiness)
    shape = sp.array(shape)
    if sp.size(shape) == 1:
        shape = sp.full((3, ), int(shape))
    sigma = sp.mean(shape)/(40*blobiness)
    im = sp.random.random(shape)
    im = spim.gaussian_filter(im, sigma=sigma)
    if porosity:
        im = norm_to_uniform(im, scale=[0, 1])
        im = im < porosity
    return im


def cylinders(shape, radius, nfibers, phi_max=0, theta_max=90):
    r"""
    Generates a binary image of overlapping cylinders.  This is a good
    approximation of a fibrous mat.

    Parameters
    ----------
    phi_max : scalar
        A value between 0 and 90 that controls the amount that the fibers
        lie out of the XY plane, with 0 meaning all fibers lie in the XY
        plane, and 90 meaning that fibers are randomly oriented out of the
        plane by as much as +/- 90 degrees.

    theta_max : scalar
        A value between 0 and 90 that controls the amount rotation in the
        XY plane, with 0 meaning all fibers point in the X-direction, and
        90 meaning they are randomly rotated about the Z axis by as much
        as +/- 90 degrees.

    Returns
    -------
    A boolean array with True values denoting the pore space
    """
    shape = sp.array(shape)
    if sp.size(shape) == 1:
        shape = sp.full((3, ), int(shape))
    elif sp.size(shape) == 2:
        raise Exception("2D fibers don't make sense")
    im = sp.zeros(shape)
    R = sp.sqrt(sp.sum(sp.square(shape)))
    n = 0
    while n < nfibers:
        x = sp.rand(3)*shape
        phi = sp.deg2rad(90 + 90*(0.5 - sp.rand())*phi_max/90)
        theta = sp.deg2rad(180 - 90*(0.5 - sp.rand())*2*theta_max/90)
        X0 = R*sp.array([sp.sin(theta)*sp.cos(phi),
                         sp.sin(theta)*sp.sin(phi),
                         sp.cos(theta)])
        [X0, X1] = [X0 + x, -X0 + x]
        crds = line_segment(X0, X1)
        lower = ~sp.any(sp.vstack(crds).T < [0, 0, 0], axis=1)
        upper = ~sp.any(sp.vstack(crds).T >= shape, axis=1)
        valid = upper*lower
        if sp.any(valid):
            im[crds[0][valid], crds[1][valid], crds[2][valid]] = 1
            n += 1
    im = sp.array(im, dtype=bool)
    dt = spim.distance_transform_edt(~im) < radius
    return ~dt


def line_segment(X0, X1):
    r"""
    Calculate the voxel coordinates of a straight line between the two given
    end points

    Parameters
    ----------
    X0 and X1 : array_like
        The [x, y, z] coordinates of the start and end points of the line.

    Returns
    -------
        A list of lists containing the X, Y, and Z coordinates of all voxels
        that should be drawn between the start and end points to create a solid
        line.
    """
    X0 = sp.around(X0)
    X1 = sp.around(X1)
    L = sp.amax(sp.absolute([[X1[0]-X0[0]], [X1[1]-X0[1]], [X1[2]-X0[2]]])) + 1
    x = sp.rint(sp.linspace(X0[0], X1[0], L)).astype(int)
    y = sp.rint(sp.linspace(X0[1], X1[1], L)).astype(int)
    z = sp.rint(sp.linspace(X0[2], X1[2], L)).astype(int)
    return [x, y, z]


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
    edge = sp.ones_like(im)
    if len(im.shape) == 2:
        sx, sy = im.shape
        edge[r:sx-r, r:sy-r] = im[r:sx-r, r:sy-r]
    else:
        sx, sy, sz = im.shape
        edge[r:sx-r, r:sy-r, r:sz-r] = im[r:sx-r, r:sy-r, r:sz-r]
    return edge
