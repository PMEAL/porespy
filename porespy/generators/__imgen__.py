import scipy as sp
import scipy.spatial as sptl
import scipy.ndimage as spim
import scipy.sparse as sprs
from skimage.segmentation import find_boundaries
from skimage.morphology import ball, disk, square, watershed


def polydisperse_disks(shape, porosity, dist, bins=5):
    r"""
    """
    shape = sp.array(shape)
    Rs = dist.interval(sp.linspace(1/bins, 0.95, bins))
    Rs = sp.vstack(Rs).T
    Rs = (Rs[:-1] + Rs[1:])/2
    Rs = Rs.flatten()
    phi = 1 - (1 - porosity)/(len(Rs))
    im = sp.ones(shape, dtype=bool)
    for r in Rs:
        temp = overlapping_disks(shape=shape, radius=r, porosity=phi)
        im = im*temp
    return im


def voronoi_edges(shape, edge_radius, ncells, flat_faces=True):
    r"""
    Create an image of the edges in a Voronoi tessellation

    Parameters
    ----------
    shape : array_like
        The size of the image to generate in [Nx, Ny, Nz] where Ni is the
        number of voxels in each direction.

    edge_radius : scalar
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
    shape = sp.array(shape)
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
    im = spim.distance_transform_edt(~im) > edge_radius
    return im


def _get_Voronoi_edges(vor):
    coords = [[], []]
    for facet in vor.ridge_vertices:
        # Create a closed cycle of vertices that define the facet
        coords[0].extend(facet[:-1]+[facet[-1]])
        coords[1].extend(facet[1:]+[facet[0]])
    conns = sp.vstack(coords).T  # Convert to scipy friendly format
    conns = sp.sort(conns, axis=1)  # Move all points to upper triangle
    mask = sp.any(conns == -1, axis=1)
    conns = conns[~mask]
    adjmat = sprs.coo_matrix((sp.ones_like(conns[:, 0]),
                             (conns[:, 0], conns[:, 1])))
    adjmat = adjmat.tocsr().tocoo()  # Remove duplicates by casting to csr
    edges = sp.vstack((adjmat.row, adjmat.col)).T
    return edges


def circle_pack(shape, radius, offset=0, packing='square'):
    r"""
    Generates a 2D packing of circles

    Parameters
    ----------
    shape : list
        The size of the image to generate in [Nx, Ny] where N is the
        number of voxels in each direction

    radius : scalar
        The radius of circles in the packing (in pixels)

    offset : scalar
        The amount offset (+ or -) to add between pore centers (in pixels).

    packing : string
        Specifies the type of cubic packing to create.  Options are
        'square' (default) and 'triangular'.

    Returns
    -------
    A boolean array with True values denoting the pore space
    """
    r = radius
    if sp.size(shape) == 1:
        shape = sp.full((3, ), int(shape))
    elif (sp.size(shape) == 3) or (1 in shape):
        raise Exception("This function only produces 2D images, " +
                         "try \'sphere_pack\'")
    im = sp.zeros(shape, dtype=bool)
    if packing.startswith('s'):
        spacing = 2*r
        s = int(spacing/2) + sp.array(offset)
        coords = sp.mgrid[r:im.shape[0]-r:2*s,
                      r:im.shape[1]-r:2*s]
        im[coords[0], coords[1]] = 1
    if packing.startswith('t'):
        spacing = 2*sp.floor(sp.sqrt(2*(r**2))).astype(int)
        s = int(spacing/2) + offset
        coords = sp.mgrid[r:im.shape[0]-r:2*s,
                          r:im.shape[1]-r:2*s]
        im[coords[0], coords[1]] = 1
        coords = sp.mgrid[s+r:im.shape[0]-r:2*s,
                          s+r:im.shape[1]-r:2*s]
        im[coords[0], coords[1]] = 1
    im = spim.distance_transform_edt(~im) >= r
    return im


def sphere_pack(shape, radius, offset=0, packing='sc'):
    r"""
    Generates a cubic packing of spheres

    Parameters
    ----------
    shape : list
        The size of the image to generate in [Nx, Ny, Nz] where N is the
        number of voxels in each direction.

    radius : scalar
        The radius of spheres in the packing

    offset : scalar
        The amount offset (+ or -) to add between pore centers.

    packing : string
        Specifies the type of cubic packing to create.  Options are:

        'sc' : Simple Cubic (default)
        'fcc' : Face Centered Cubic
        'bcc' : Body Centered Cubic

    Returns
    -------
    A boolean array with True values denoting the pore space
    """
    r = radius
    if sp.size(shape) == 1:
        shape = sp.full((3, ), int(shape))
    elif (sp.size(shape) == 2) or (1 in shape):
        raise Exception("This function only produces 3D images, " +
                         "try \'circle_pack\'")
    im = sp.zeros(shape, dtype=bool)
    if packing.startswith('s'):
        spacing = 2*r
        s = int(spacing/2) + sp.array(offset)
        coords = sp.mgrid[r:im.shape[0]-r:2*s,
                      r:im.shape[1]-r:2*s,
                      r:im.shape[2]-r:2*s]
        im[coords[0], coords[1], coords[2]] = 1
    elif packing.startswith('b'):
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
    elif packing.startswith('f'):
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
    im = spim.distance_transform_edt(~im) > r
    return im


def overlapping_spheres(shape, radius, porosity):
    r"""
    Generate a packing of overlapping mono-disperse spheres

    Parameters
    ----------
    shape : list
        The size of the image to generate in [Nx, Ny, Nz] where N is the
        number of voxels.

    radius : scalar
        The radius of spheres in the packing

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
    if sp.size(shape) == 1:
        shape = sp.full((3, ), int(shape))
    s_vol = 4/3*sp.pi*radius**3
    bulk_vol = sp.prod(shape)
    N = (1 - porosity)*bulk_vol/s_vol
    im = sp.random.random(size=shape) > (N/bulk_vol)
    im = spim.distance_transform_edt(im) < radius
    return ~im


def overlapping_disks(shape, radius, porosity):
    r"""
    Generate a packing of overlapping mono-disperse disk/circles

    Parameters
    ----------
    shape : list
        The size of the image to generate in [Nx, Ny] where N is the
        number of voxels.

    radius : scalar
        The radius of spheres in the packing

    porosity : scalar
        The porosity of the final image.  This number is approximated by
        the method so the returned result may not have exactly the
        specified value.

    Returns
    -------
    A boolean array with True values denoting the pore space

    Notes
    -----
    1. This method can also be used to generate a dispersion of hollows by
    treating ``porosity`` as solid volume fraction and inverting the
    returned image.

    """
    if sp.size(shape) == 1:
        shape = sp.full((2, ), int(shape))
    elif sp.size(shape) == 3:
        raise Exception("For 3D images use \'overlapping_spheres\'")
    s_vol = sp.pi*radius**2
    bulk_vol = sp.prod(shape)
    N = (1 - porosity)*bulk_vol/s_vol
    im = sp.random.random(size=shape) > (N/bulk_vol)
    im = spim.distance_transform_edt(im) < radius
    return ~im


def blobs(shape, porosity=0.5, blobiness=1):
    """
    Generates an image containing amorphous blobs

    Parameters
    ----------
    shape : list
        The size of the image to generate in [Nx, Ny, Nz] where N is the
        number of voxels

    porosity : scalar (default = 0.5)
        The porosity of the final image.  This number is approximated by
        the function so the returned result may not have exactly the
        specified value.

    blobiness : array_like (default = 1)
        Controls the morphology of the blobs.  A higher number results in
        a larger number of small blobs.  If a vector is supplied then the blobs
        are anisotropic.

    Returns
    -------
    A boolean array with True values denoting the pore space

    """
    blobiness = sp.array(blobiness)
    shape = sp.array(shape)
    if sp.size(shape) == 1:
        shape = sp.full((3, ), int(shape))
    sigma = sp.mean(shape)/(40*blobiness)
    mask = sp.random.random(shape)
    mask = spim.gaussian_filter(mask, sigma=sigma)
    hist = sp.histogram(mask, bins=1000)
    cdf = sp.cumsum(hist[0])/sp.size(mask)
    xN = sp.where(cdf >= porosity)[0][0]
    im = mask <= hist[1][xN]
    return im


def fibers(shape, radius, nfibers, phi_max=0, theta_max=90):
    r"""
    Generates a binary image of overlapping fibers.

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
