import numpy as np
import inspect as insp
from edt import edt
import porespy as ps
from numba import njit
import scipy.spatial as sptl
import scipy.ndimage as spim
import scipy.stats as spst
from deprecated import deprecated
from porespy.tools import norm_to_uniform, ps_ball, ps_disk, get_border
from porespy.tools import extract_subsection
from porespy.tools import insert_sphere
from porespy import settings
from typing import List
from loguru import logger
tqdm = ps.tools.get_tqdm()


def insert_shape(im, element, center=None, corner=None, value=1, mode="overwrite"):
    r"""
    Inserts sub-image into a larger image at the specified location.

    If the inserted image extends beyond the boundaries of the image it will
    be cropped accordingly.

    Parameters
    ----------
    im : ndarray
        The image into which the sub-image will be inserted
    element : ndarray
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
        If ``center`` is given then ``corner`` cannot be specified.
    value : scalar
        A scalar value to apply to the sub-image.  The default is 1.
    mode : string
        If 'overwrite' (default) the inserted image replaces the values in the
        main image.  If 'overlay' the inserted image is added to the main
        image.  In both cases the inserted image is multiplied by ``value``
        first.

    Returns
    -------
    im : ndarray
        A copy of ``im`` with the supplied element inserted.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/generators/reference/insert_shape.html>`_
    to view online example.

    """
    im = im.copy()
    if im.ndim != element.ndim:
        msg = f"Image shape {im.shape} and element shape {element.shape} do not match"
        raise Exception(msg)

    s_im = []
    s_el = []
    if (center is not None) and (corner is None):
        for dim in range(im.ndim):
            r, d = np.divmod(element.shape[dim], 2)
            if d == 0:
                raise Exception(
                    "Cannot specify center point when element "
                    + "has one or more even dimension"
                )
            lower_im = np.amax((center[dim] - r, 0))
            upper_im = np.amin((center[dim] + r + 1, im.shape[dim]))
            s_im.append(slice(lower_im, upper_im))
            lower_el = np.amax((lower_im - center[dim] + r, 0))
            upper_el = np.amin((upper_im - center[dim] + r, element.shape[dim]))
            s_el.append(slice(lower_el, upper_el))
    elif (corner is not None) and (center is None):
        for dim in range(im.ndim):
            L = int(element.shape[dim])
            lower_im = np.amax((corner[dim], 0))
            upper_im = np.amin((corner[dim] + L, im.shape[dim]))
            s_im.append(slice(lower_im, upper_im))
            lower_el = np.amax((lower_im - corner[dim], 0))
            upper_el = np.amin((upper_im - corner[dim], element.shape[dim]))
            s_el.append(slice(min(lower_el, upper_el), upper_el))
    else:
        raise Exception("Cannot specify both corner and center")

    if mode == "overlay":
        im[tuple(s_im)] = im[tuple(s_im)] + element[tuple(s_el)] * value
    elif mode == "overwrite":
        im[tuple(s_im)] = element[tuple(s_el)] * value
    else:
        raise Exception("Invalid mode " + mode)
    return im


@deprecated("This function has been renamed to rsa (lowercase to meet pep8")
def RSA(*args, **kwargs):
    return rsa(*args, **kwargs)


def rsa(im_or_shape: np.array,
        r: int,
        volume_fraction: int = 1,
        clearance: int = 0,
        n_max: int = 100000,
        mode: str = "contained",
        return_spheres: bool = False,
        smooth: bool = True):
    r"""
    Generates a sphere or disk packing using Random Sequential Addition

    Parameters
    ----------
    im_or_shape : ndarray or list
        To provide flexibility, this argument accepts either an image into
        which the spheres are inserted, or a shape which is used to create an
        empty image.  In both cases the spheres are added as ``True`` values
        to the background.  Since ``True`` is considered the pore space, then
        the added spheres represent holes.
    r : int
        The radius of the disk or sphere to insert.
    volume_fraction : scalar (default is 1.0)
        The fraction of the image that should be filled with spheres.  The
        spheres are added as ``True``'s, so each sphere addition increases the
        ``volume_fraction`` until the specified limit is reached.  Note that if
        ``n_max`` is reached first, then ``volume_fraction`` will not be
        acheived.  Also, ``volume_fraction`` is not counted correctly if the
        ``mode`` is ``'extended'``.
    clearance : int (optional, default = 0)
        The amount of space to put between each sphere. Negative values are
        acceptable to create overlaps, so long as ``abs(clearance) < r``.
    n_max : int (default is 100,000)
        The maximum number of spheres to add.  Using a low value may halt
        the addition process prior to reaching the specified
        ``volume_fraction``.  If ``None`` is given, then no limit is used.
    mode : string (default is 'contained')
        Controls how the edges of the image are handled.  Options are:

        'contained'
            Spheres are all completely within the image
        'extended'
            Spheres are allowed to extend beyond the edge of the
            image.  In this mode the volume fraction will be less that
            requested since some spheres extend beyond the image, but their
            entire volume is counted as added for computational efficiency.

    return_spheres : bool
        If ``True`` then an image containing only the spheres is returned
        rather than the input image with the spheres added, which is the
        default behavior.
    smooth : bool
        Indicates whether balls should have smooth faces (``True``) or should
        include the bumps on the extremities (``False``).

    Returns
    -------
    image : ndarray
        An image with spheres of specified radius *added* to the background.

    See Also
    --------
    pseudo_gravity_packing
    pseudo_electrostatic_packing

    Notes
    -----
    This algorithm ensures that spheres do not overlap but does not
    guarantee they are tightly packed.

    This function adds spheres to the background of the received ``im``, which
    allows iteratively adding spheres of different radii to the unfilled space
    by repeatedly passing in the result of previous calls to RSA.

    References
    ----------
    [1] Random Heterogeneous Materials, S. Torquato (2001)

    Examples
    --------
    `Click here
    <https://porespy.org/examples/generators/reference/rsa.html>`_
    to view online example.

    """
    logger.debug(f"RSA: Adding spheres of size {r}")
    if len(im_or_shape) <= 3:
        im = np.zeros(shape=im_or_shape, dtype=bool)
    else:
        im = im_or_shape
    im = im.astype(bool)
    if return_spheres:
        im_temp = np.copy(im)
    if n_max is None:
        n_max = np.inf
    vf_final = volume_fraction
    vf_start = im.sum() / im.size
    logger.debug(f"Initial volume fraction: {vf_start}")
    if im.ndim == 2:
        template_lg = ps_disk((r + clearance) * 2)
        template_sm = ps_disk(r, smooth=smooth)
    else:
        template_lg = ps_ball((r + clearance) * 2)
        template_sm = ps_ball(r, smooth=smooth)
    vf_template = template_sm.sum() / im.size
    # Pad image by the radius of large template to enable insertion near edges
    im = np.pad(im, pad_width=2 * (r + clearance), mode="edge")
    # Depending on mode, adjust mask to remove options around edge
    if mode == "contained":
        border = get_border(im.shape, thickness=2 * (r + clearance),
                            mode="faces")
    elif mode == "extended":
        border = get_border(im.shape, thickness=(r + clearance) + 1,
                            mode="faces")
    else:
        raise Exception("Unrecognized mode: ", mode)
    # Remove border pixels
    im[border] = True
    # Dilate existing objects by strel to remove pixels near them
    # from consideration for sphere placement
    logger.trace("Dilating foreground features by sphere radius")
    dt = edt(im == 0)
    options_im = dt >= r
    # Begin inserting the spheres
    vf = vf_start
    free_sites = np.flatnonzero(options_im)
    i = 0
    while (vf <= vf_final) and (i < n_max) and (len(free_sites) > 0):
        c, count = _make_choice(options_im, free_sites=free_sites)
        # The 100 below is arbitrary and may change performance
        if count > 100:
            # Regenerate list of free_sites
            logger.debug(f"Regenerating `free_sites` after {i} iterations")
            free_sites = np.flatnonzero(options_im)
        if all(np.array(c) == -1):
            break
        s_sm = tuple([slice(x - r, x + r + 1, None) for x in c])
        s_lg = tuple([slice(x - 2 * (r + clearance),
                            x + 2 * (r + clearance) + 1, None) for x in c])
        im[s_sm] += template_sm  # Add ball to image
        options_im[s_lg][template_lg] = False  # Update extended region
        vf += vf_template
        i += 1
    logger.trace(f"Number of spheres inserted: {i}")
    # Get slice into returned image to retain original size
    s = tuple([slice(2 * (r + clearance), d - 2 * (r + clearance), None)
               for d in im.shape])
    im = im[s]
    vf = im.sum() / im.size
    logger.debug("Final volume fraction:", vf)
    if return_spheres:
        im = im * (~im_temp)
    return im


@njit
def _make_choice(options_im, free_sites):
    r"""
    This function is called by _begin_inserting to find valid insertion
    points.

    Parameters
    ----------
    options_im : ndarray
        An array with ``True`` at all valid locations and ``False`` at all
        locations where a sphere already exists PLUS a region of radius R
        around each sphere since these points are also invalid insertion
        points.
    free_sites : array_like
        A 1D array containing valid insertion indices.  This list is used
        to select insertion points from a limited which occasionally gets
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
    maxiter = upper_limit * 20
    if options_im.ndim == 2:
        coords = [-1, -1]
        Nx, Ny = options_im.shape
        while not choice:
            if count >= maxiter:
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
            if count >= maxiter:
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


def bundle_of_tubes(shape: List[int], spacing: int, distribution=None, smooth=True):
    r"""
    Create a 3D image of a bundle of tubes, in the form of a rectangular
    plate with randomly sized holes through it.

    Parameters
    ----------
    shape : list
        The size the image, with the 3rd dimension indicating the plate
        thickness.  If the 3rd dimension is not given then a thickness of
        1 voxel is assumed.
    spacing : int
        The center to center distance of the holes.  The hole sizes will
        be distributed between this values down to 3 voxels.
    distribution : scipy.stats object
        A handle to a scipy stats object with the desired parameters.

    Returns
    -------
    image : ndarray
        A boolean array with ``True`` values denoting the pore space

    Examples
    --------
    `Click here
    <https://porespy.org/examples/generators/reference/bundle_of_tubes.html>`_
    to view online example.

    """
    shape = np.array(shape)
    if len(shape) == 2:
        shape = np.hstack((shape, [1]))
    shape2 = shape[shape > 1]
    im = ~lattice_spheres(shape=shape2,
                          r=1,
                          offset=0.5*spacing,
                          spacing=spacing,
                          lattice='sc')
    N = im.sum()
    if distribution is None:
        # +1 below is because randint 4.X gives a max of 3
        distribution = spst.randint(low=3, high=int(spacing/2 + 1))
        Rs = distribution.rvs(N)
    else:
        Rs = distribution.rvs(N)
        Rs = np.around(np.clip(Rs, a_min=1, a_max=spacing/2), decimals=0).astype(int)
    temp = np.zeros_like(im)
    inds = np.where(im)
    for i in range(len(inds[0])):
        c = np.hstack([j[i] for j in inds])
        temp = insert_sphere(im=temp, c=c, r=Rs[i])
    # Add 3rd dimension back
    temp = np.tile(np.atleast_3d(temp), [1, 1, shape[2]])
    return temp


def polydisperse_spheres(shape: List[int],
                         porosity: float,
                         dist,
                         nbins: int = 5,
                         r_min: int = 5):
    r"""
    Create an image of randomly placed, overlapping spheres with a
    distribution of radii.

    Parameters
    ----------
    shape : list
        The size of the image to generate in [Nx, Ny, Nz] where Ni is the
        number of voxels in each direction.  If shape is only 2D, then an
        image of polydisperse disks is returns
    porosity : float
        The porosity of the image, defined as the number of void voxels
        divided by the number of voxels in the image. The specified value
        is only matched approximately, so it's suggested to check this
        value after the image is generated.
    dist : scipy.stats distribution object
        This should be an initialized distribution chosen from the large
        number of options in the ``scipy.stats`` submodule.  For instance,
        a normal distribution with a mean of 20 and a standard deviation
        of 10 can be obtained with
        ``dist = scipy.stats.norm(loc=20, scale=10)``
    nbins : int
        The number of discrete sphere sizes that will be used to generate
        the image. This function generates ``nbins`` images of
        monodisperse spheres that span 0.05 and 0.95 of the possible
        values produced by the provided distribution, then overlays them
        to get polydispersivity.

    Returns
    -------
    image : ndarray
        A boolean array with ``True`` values denoting the pore space

    Examples
    --------
    `Click here
    <https://porespy.org/examples/generators/reference/polydisperse_spheres.html>`_
    to view online example.

    """
    shape = np.array(shape)
    if np.size(shape) == 1:
        shape = np.full((3,), int(shape))
    Rs = dist.interval(np.linspace(0.05, 0.95, nbins))
    Rs = np.vstack(Rs).T
    Rs = (Rs[:-1] + Rs[1:]) / 2
    Rs = np.clip(Rs.flatten(), a_min=r_min, a_max=None)
    phi_desired = 1 - (1 - porosity) / (len(Rs))
    im = np.ones(shape, dtype=bool)
    for r in Rs:
        phi_im = im.sum() / np.prod(shape)
        phi_corrected = 1 - (1 - phi_desired) / phi_im
        temp = overlapping_spheres(shape=shape, r=r, porosity=phi_corrected)
        im = im * temp
    return im


def voronoi_edges(shape: List[int], ncells: int, r: int = 0,
                  flat_faces: bool = True):
    r"""
    Create an image from the edges of a Voronoi tessellation.

    Parameters
    ----------
    shape : array_like
        The size of the image to generate in [Nx, Ny, <Nz>] where Ni is the
        number of voxels in each direction.  If Nz is not given a 2D image
        is returned.
    ncells : int
        The number of Voronoi cells to include in the tesselation.
    radius : int, optional
        The radius to which Voronoi edges should be dilated in the final
        image.  If not given then edges are a single pixel/voxel thick.
    flat_faces : bool
        Whether the Voronoi edges should lie on the boundary of the
        image (``True``), or if edges outside the image should be removed
        (``False``).

    Returns
    -------
    image : ndarray
        A boolean array with ``True`` values denoting the pore space

    Examples
    --------
    `Click here
    <https://porespy.org/examples/generators/reference/voronoi_edges.html>`_
    to view online example.

    """
    logger.trace(f"Generating {ncells} cells")
    shape = np.array(shape)
    if np.size(shape) == 1:
        shape = np.full((3,), int(shape))
    im = np.zeros(shape, dtype=bool)
    base_pts = tuple(np.around(np.random.rand(ncells, im.ndim) * (shape-1),
                               decimals=0).T.astype(int))
    im[tuple(base_pts)] = True
    pw = [(s, s) for s in im.shape]
    if flat_faces:
        im = np.pad(im, pad_width=pw, mode='symmetric')
    else:
        im = np.pad(im, pad_width=pw, mode='constant', constant_values=0)
    base_pts = np.where(im)
    vor = sptl.Voronoi(points=np.vstack(base_pts).T)
    vor.vertices = np.around(vor.vertices)
    vor.vertices *= (np.array(im.shape) - 1) / np.array(im.shape)
    vor.edges = _get_Voronoi_edges(vor)
    im = np.zeros_like(im)
    for row in vor.edges:
        pts = np.around(vor.vertices[row], decimals=0).astype(int)
        if np.all(pts >= 0) and np.all(pts < im.shape):
            line_pts = line_segment(pts[0], pts[1])
            im[tuple(line_pts)] = True
    im = extract_subsection(im=im, shape=shape)
    im = edt(~im) > r
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
    A 2-by-N array of vertex indices, indicating the start and end points
    of each vertex in the Voronoi diagram.  These vertex indices can be
    used to index straight into the ``vor.vertices`` array to get spatial
    positions.

    """
    edges = [[], []]
    for facet in vor.ridge_vertices:
        # Create a closed cycle of vertices that define the facet
        edges[0].extend(facet[:-1] + [facet[-1]])
        edges[1].extend(facet[1:] + [facet[0]])
    edges = np.vstack(edges).T  # Convert to scipy-friendly format
    mask = np.any(edges == -1, axis=1)  # Identify edges at infinity
    edges = edges[~mask]  # Remove edges at infinity
    edges = np.sort(edges, axis=1)  # Move all points to upper triangle
    # Remove duplicate pairs
    edges = edges[:, 0] + 1j * edges[:, 1]  # Convert to imaginary
    edges = np.unique(edges)  # Remove duplicates
    edges = np.vstack((np.real(edges), np.imag(edges))).T  # Back to real
    edges = np.array(edges, dtype=int)
    return edges


def lattice_spheres(shape: List[int],
                    r: int,
                    spacing: int = None,
                    offset: int = None,
                    smooth: bool = True,
                    lattice: str = "sc"):
    r"""
    Generate a cubic packing of spheres in a specified lattice arrangement.

    Parameters
    ----------
    shape : list
        The size of the image to generate in [Nx, Ny, Nz] where N is
        the number of voxels in each direction. For a 2D image, use
        [Nx, Ny].
    radius : int
        The radius of spheres (circles) in the packing.
    spacing : int or List[int]
        The spacing between unit cells. If the spacing is too small then
        spheres may overlap. If an ``int`` is given it will be applied
        in all directions, while a list of ``int`` will be interpreted
        to apply along each axis.
    offset : int or List[int]
        The amount offset to add between sphere centers and the edges of
        the image. A single ``int`` will be applied in all directions,
        while a list of ``int`` will be interpreted to apply along each
        axis.
    smooth : bool, default=True
        If ``True`` (default) the outer extremities of the sphere will
        not have the little bumps on each face.
    lattice : str
        Specifies the type of lattice to create. Options are:

        ======= ===============================================================
        option  description
        ======= ===============================================================
        'sc'    Simple cubic (default), works in both 2D and 3D.
        'tri'   Triangular, only works in 2D
        'fcc'   Face centered cubic, only works in 3D
        'bcc'   Body centered cubic, only works on 3D
        ======= ===============================================================

    Returns
    -------
    image : ndarray
        A boolean array with ``True`` values denoting the pore space.

    Notes
    -----
    For 2D images, 'sc' gives a square lattice and both 'fcc' and
    'bcc' give a triangular lattice.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/generators/reference/lattice_spheres.html>`_
    to view online example.

    """
    logger.debug(f"Generating {lattice} lattice")
    shape = np.array(shape)
    im = np.zeros(shape, dtype=bool)

    # Parse lattice type
    lattice = lattice.lower()
    if im.ndim == 2:
        if lattice in ['sc', 'square', 'cubic', 'simple cubic']:
            lattice = 'sq'
        elif lattice in ['tri', 'triangular']:
            lattice = 'tri'
        else:
            raise Exception(f'Unrecognized mode: {lattice}')
    else:
        if lattice in ['sc', 'cubic', 'simple cubic']:
            lattice = 'sc'
        elif lattice in ['bcc', 'body centered cubic']:
            lattice = 'bcc'
        elif lattice in ['fcc', 'face centered cubic']:
            lattice = 'fcc'
        else:
            raise Exception(f'Unrecognized mode: {lattice}')

    # Parse offset and spacing args
    if spacing is None:
        spacing = 2*r
    if isinstance(spacing, (int, float)):
        spacing = int(spacing)
        spacing = [spacing]*im.ndim
    spacing = np.array(spacing, dtype=int)
    if offset is None:
        offset = r
    if isinstance(offset, (int, float)):
        offset = int(offset)
        offset = [offset]*im.ndim
    offset = np.array(offset, dtype=int)

    if lattice == 'sq':
        im[offset[0]::spacing[0],
           offset[1]::spacing[1]] = True
    elif lattice == 'tri':
        im[offset[0]::spacing[0],
           offset[1]::spacing[1]] = True
        im[offset[0]+int(spacing[0]/2)::spacing[0],
           offset[1]+int(spacing[1]/2)::spacing[1]] = True
    elif lattice == 'sc':
        im[offset[0]::spacing[0],
           offset[1]::spacing[1],
           offset[2]::spacing[2]] = True
    elif lattice == 'bcc':
        im[offset[0]::spacing[0],
           offset[1]::spacing[1],
           offset[2]::spacing[2]] = True
        im[offset[0]+int(spacing[0]/2)::spacing[0],
           offset[1]+int(spacing[1]/2)::spacing[1],
           offset[2]+int(spacing[2]/2)::spacing[2]] = True
    elif lattice == 'fcc':
        im[offset[0]::spacing[0],
           offset[1]::spacing[1],
           offset[2]::spacing[2]] = True
        # xy-plane
        im[offset[0]+int(spacing[0]/2)::spacing[0],
           offset[1]+int(spacing[1]/2)::spacing[1],
           offset[2]::spacing[2]] = True
        # xz-plane
        im[offset[0]+int(spacing[0]/2)::spacing[0],
           offset[1]::spacing[1],
           offset[2]+int(spacing[2]/2)::spacing[2]] = True
        # yz-plane
        im[offset[0]::spacing[0],
           offset[1]+int(spacing[1]/2)::spacing[1],
           offset[2]+int(spacing[2]/2)::spacing[2]] = True
    if smooth:
        im = ~(edt(~im) < r)
    else:
        im = ~(edt(~im) <= r)
    return im


def overlapping_spheres(shape: List[int],
                        r: int,
                        porosity: float,
                        maxiter: int = 10,
                        tol: float = 0.01):
    r"""
    Generate a packing of overlapping mono-disperse spheres

    Parameters
    ----------
    shape : list
        The size of the image to generate in [Nx, Ny, Nz] where Ni is the
        number of voxels in the i-th direction.
    r : scalar
        The radius of spheres in the packing.
    porosity : scalar
        The porosity of the final image, accurate to the given tolerance.
    maxiter : int
        Maximum number of iterations for the iterative algorithm that improves
        the porosity of the final image to match the given value.
    tol : float
        Tolerance for porosity of the final image compared to the given value.

    Returns
    -------
    image : ndarray
        A boolean array with ``True`` values denoting the pore space

    Notes
    -----
    This method can also be used to generate a dispersion of hollows by
    treating ``porosity`` as solid volume fraction and inverting the
    returned image.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/generators/reference/overlapping_spheres.html>`_
    to view online example.

    """
    shape = np.array(shape)
    if np.size(shape) == 1:
        shape = np.full((3, ), int(shape))
    ndim = (shape != 1).sum()
    s_vol = ps_disk(r).sum() if ndim == 2 else ps_ball(r).sum()

    bulk_vol = np.prod(shape)
    N = int(np.ceil((1 - porosity) * bulk_vol / s_vol))
    im = np.random.random(size=shape)

    # Helper functions for calculating porosity: phi = g(f(N))
    def f(N):
        return edt(im > N / bulk_vol) < r

    def g(im):
        r"""Returns fraction of 0s, given a binary image"""
        return 1 - im.sum() / np.prod(shape)

    # # Newton's method for getting image porosity match the given
    # w = 1.0                         # Damping factor
    # dN = 5 if ndim == 2 else 25     # Perturbation
    # for i in range(maxiter):
    #     err = g(f(N)) - porosity
    #     d_err = (g(f(N+dN)) - g(f(N))) / dN
    #     if d_err == 0:
    #         break
    #     if abs(err) <= tol:
    #         break
    #     N2 = N - int(err/d_err)   # xnew = xold - f/df
    #     N = w * N2 + (1-w) * N

    # Bisection search: N is always undershoot (bc. of overlaps)
    N_low, N_high = N, 4 * N
    for i in range(maxiter):
        N = np.mean([N_high, N_low], dtype=int)
        err = g(f(N)) - porosity
        if err > 0:
            N_low = N
        else:
            N_high = N
        if abs(err) <= tol:
            break

    return ~f(N)


def blobs(shape: List[int], porosity: float = 0.5, blobiness: int = 1,
          divs: int = 1):
    """
    Generates an image containing amorphous blobs

    Parameters
    ----------
    shape : list
        The size of the image to generate in [Nx, Ny, Nz] where N is the
        number of voxels
    porosity : float
        If specified, this will threshold the image to the specified value
        prior to returning.  If ``None`` is specified, then the scalar
        noise field is converted to a uniform distribution and returned
        without thresholding.
    blobiness : int or list of ints(default = 1)
        Controls the morphology of the blobs.  A higher number results in
        a larger number of small blobs.  If a list is supplied then the
        blobs are anisotropic.
    divs : int or array_like
        The number of times to divide the image for parallel processing.
        If ``1`` then parallel processing does not occur.  ``2`` is
        equivalent to ``[2, 2, 2]`` for a 3D image.  The number of cores
        used is specified in ``porespy.settings.ncores`` and defaults to
        all cores.

    Returns
    -------
    image : ndarray
        A boolean array with ``True`` values denoting the pore space

    See Also
    --------
    norm_to_uniform

    Notes
    -----
    This function generates random noise, the applies a gaussian blur to
    the noise with a sigma controlled by the blobiness argument as:

        $$ np.mean(shape) / (40 * blobiness) $$

    The value of 40 was chosen so that a ``blobiness`` of 1 gave a
    reasonable result.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/generators/reference/blobs.html>`_
    to view online example.

    """
    if isinstance(shape, int):
        shape = [shape]*3
    if len(shape) == 1:
        shape = [shape[0]]*3
    shape = np.array(shape)
    if isinstance(blobiness, int):
        blobiness = [blobiness]*len(shape)
    blobiness = np.array(blobiness)
    parallel = False
    if isinstance(divs, int):
        divs = [divs]*len(shape)
    if max(divs) > 1:
        parallel = True
        logger.info(f'Performing {insp.currentframe().f_code.co_name} in parallel')
    sigma = np.mean(shape) / (40 * blobiness)
    im = np.random.random(shape)
    if parallel:
        overlap = max([int(s*4) for s in sigma])
        im = ps.filters.chunked_func(func=spim.gaussian_filter,
                                     input=im, sigma=sigma,
                                     divs=divs, overlap=overlap)
    else:
        im = spim.gaussian_filter(im, sigma=sigma)
    im = norm_to_uniform(im, scale=[0, 1])
    if porosity:
        im = im < porosity
    return im


def _cylinders(shape: List[int],
               r: int,
               ncylinders: int,
               phi_max: float = 0,
               theta_max: float = 90,
               length: float = None,
               verbose: bool = True):
    r"""
    Generates a binary image of overlapping cylinders.

    This is a good approximation of a fibrous mat.

    Parameters
    ----------
    shape : list
        The size of the image to generate in [Nx, Ny, Nz] where N is the
        number of voxels. 2D images are not permitted.
    r : int
        The radius of the cylinders in voxels
    ncylinders : int
        The number of cylinders to add to the domain. Adjust this value to
        control the final porosity, which is not easily specified since
        cylinders overlap and intersect different fractions of the domain.
    phi_max : int
        A value between 0 and 90 that controls the amount that the
        cylinders lie *out of* the XY plane, with 0 meaning all cylinders
        lie in the XY plane, and 90 meaning that cylinders are randomly
        oriented out of the plane by as much as +/- 90 degrees.
    theta_max : int
        A value between 0 and 90 that controls the amount of rotation
        *in the* XY plane, with 0 meaning all cylinders point in the
        X-direction, and 90 meaning they are randomly rotated about the Z
        axis by as much as +/- 90 degrees.
    length : int
        The length of the cylinders to add.  If ``None`` (default) then
        the cylinders will extend beyond the domain in both directions so
        no ends will exist. If a scalar value is given it will be
        interpreted as the Euclidean distance between the two ends of the
        cylinder. Note that one or both of the ends *may* still lie
        outside the domain, depending on the randomly chosen center point
        of the cylinder.

    Returns
    -------
    image : ndarray
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
        length = 2 * H
    R = min(int(length / 2), 2 * H)  # Trim given length to 2H if too long
    # Adjust max angles to be between 0 and 90
    if (phi_max > 90) or (phi_max < 0):
        raise Exception('phi_max must be betwen 0 and 90')
    if (theta_max > 90) or (theta_max < 0):
        raise Exception('theta_max must be betwen 0 and 90')
    # Create empty image for inserting into
    im = np.zeros(shape, dtype=bool)
    n = 0
    L = min(H, R)
    # Disable tqdm if called from another tqdm to prevent double pbars
    tqdm_settings = settings.tqdm.copy()
    if not settings.tqdm["disable"]:
        tqdm_settings = {**settings.tqdm, **{'disable': not verbose}}
    with tqdm(ncylinders, **tqdm_settings) as pbar:
        while n < ncylinders:
            # Choose a random starting point in domain
            x = np.random.rand(3) * (shape + 2 * L)
            # Chose a random phi and theta within given ranges
            phi = (np.pi / 2 - np.pi * np.random.rand()) * phi_max / 90
            theta = (np.pi / 2 - np.pi * np.random.rand()) * theta_max / 90
            X0 = R * np.array([np.cos(phi) * np.cos(theta),
                               np.cos(phi) * np.sin(theta),
                               np.sin(phi)])
            [X0, X1] = [x + X0, x - X0]
            crds = line_segment(X0, X1)
            lower = ~np.any(np.vstack(crds).T < [L, L, L], axis=1)
            upper = ~np.any(np.vstack(crds).T >= shape + L, axis=1)
            valid = upper * lower
            if np.any(valid):
                im[crds[0][valid] - L, crds[1][valid] - L, crds[2][valid] - L] = 1
                n += 1
                pbar.update()
    im = np.array(im, dtype=bool)
    dt = edt(~im) < r
    return ~dt


def cylinders(shape: List[int],
              r: int,
              ncylinders: int = None,
              porosity: float = None,
              phi_max: float = 0,
              theta_max: float = 90,
              length: float = None,
              maxiter: int = 3):
    r"""
    Generates a binary image of overlapping cylinders given porosity OR
    number of cylinders.

    This is a good approximation of a fibrous mat.

    Parameters
    ----------
    shape : list
        The size of the image to generate in [Nx, Ny, Nz] where N is the
        number of voxels. 2D images are not permitted.
    r : scalar
        The radius of the cylinders in voxels
    ncylinders : int
        The number of cylinders to add to the domain. Adjust this value to
        control the final porosity, which is not easily specified since
        cylinders overlap and intersect different fractions of the domain.
    porosity : float
        The targeted value for the porosity of the generated mat. The
        function uses an algorithm for predicted the number of required
        number of cylinder, and refines this over a certain number of
        fractional insertions (according to the 'iterations' input).
    phi_max : int
        A value between 0 and 90 that controls the amount that the
        cylinders lie *out of* the XY plane, with 0 meaning all cylinders
        lie in the XY plane, and 90 meaning that cylinders are randomly
        oriented out of the plane by as much as +/- 90 degrees.
    theta_max : int
        A value between 0 and 90 that controls the amount of rotation
        *in the* XY plane, with 0 meaning all cylinders point in the
        X-direction, and 90 meaning they are randomly rotated about the Z
        axis by as much as +/- 90 degrees.
    length : int
        The length of the cylinders to add.  If ``None`` (default) then
        the cylinders will extend beyond the domain in both directions so
        no ends will exist. If a scalar value is given it will be
        interpreted as the Euclidean distance between the two ends of the
        cylinder. Note that one or both of the ends *may* still lie
        outside the domain, depending on the randomly chosen center point
        of the cylinder.
    maxiter : int
        The number of fractional fiber insertions used to target the
        requested porosity. By default a value of 3 is used (and this is
        typically effective in getting very close to the targeted
        porosity), but a greater number can be input to improve the
        achieved porosity.

    Returns
    -------
    image : ndarray
        A boolean array with ``True`` values denoting the pore space

    Notes
    -----
    The cylinders_porosity function works by estimating the number of
    cylinders needed to be inserted into the domain by estimating
    cylinder length, and exploiting the fact that, when inserting any
    potentially overlapping objects randomly into a volume v_total (which
    has units of pixels and is equal to dimx x dimy x dimz, for example),
    such that the total volume of objects added to the volume is v_added
    (and includes any volume that was inserted but overlapped with already
    occupied space), the resulting porosity will be equal to
    exp(-v_added/v_total).

    After intially estimating the cylinder number and inserting a small
    fraction of the estimated number, the true cylinder volume is
    calculated, the estimate refined, and a larger fraction of cylinders
    inserted. This is repeated a number of times according to the
    ``maxiter`` argument, yielding an image with a porosity close to
    the goal.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/generators/reference/cylinders.html>`_
    to view online example.

    """
    if ncylinders is not None:
        im = _cylinders(
            shape=shape,
            r=r,
            ncylinders=ncylinders,
            phi_max=phi_max,
            theta_max=theta_max,
            length=length,
        )
        return im

    if porosity is None:
        raise Exception("'ncylinders' and 'porosity' can't be both None")

    # if maxiter < 3:
    #     raise Exception("Iterations must be greater than or equal to 3")

    vol_total = float(np.prod(shape))

    def get_num_pixels(porosity):
        r"""
        Helper method to calculate number of pixels given a porosity
        """
        return -np.log(porosity) * vol_total

    # Crudely estimate fiber length as cube root of product of dims
    length_estimate = vol_total ** (1 / 3) if length is None else length

    # Rough fiber volume estimate
    vol_fiber = length_estimate * np.pi * r * r
    n_pixels_to_add = get_num_pixels(porosity)

    # Rough estimate of n_fibers
    n_fibers_added = 0
    # Calculate fraction of fibers to be added in each iteration.
    subdif = 0.8 / np.sum(np.arange(1, maxiter) ** 2)
    fractions = [0.2]
    for i in range(1, maxiter):
        fractions.append(fractions[i - 1] + (maxiter - i) ** 2 * subdif)

    im = np.ones(shape, dtype=bool)
    for frac in tqdm(fractions, **settings.tqdm):
        n_fibers_total = n_pixels_to_add / vol_fiber
        n_fibers = int(np.ceil(frac * n_fibers_total) - n_fibers_added)
        if n_fibers > 0:
            im = im & _cylinders(shape, r, n_fibers,
                                 phi_max, theta_max, length,
                                 verbose=False)
        n_fibers_added += n_fibers
        # Update parameters for next iteration
        porosity = ps.metrics.porosity(im)
        vol_added = get_num_pixels(porosity)
        vol_fiber = vol_added / n_fibers_added

    logger.debug(f"{n_fibers_added} fibers added to reach target porosity.")

    return im


def line_segment(X0, X1):
    r"""
    Calculate the voxel coordinates of a straight line between the two
    given end points

    Parameters
    ----------
    X0 and X1 : array_like
        The [x, y] or [x, y, z] coordinates of the start and end points of
        the line.

    Returns
    -------
    coords : list of lists
        A list of lists containing the X, Y, and Z coordinates of all
        voxels that should be drawn between the start and end points to
        create a solid line.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/generators/reference/line_segment.html>`_
    to view online example.

    """
    X0 = np.around(X0).astype(int)
    X1 = np.around(X1).astype(int)
    if len(X0) == 3:
        L = np.amax(np.absolute([[X1[0] - X0[0]], [X1[1] - X0[1]], [X1[2] - X0[2]]])) + 1
        x = np.rint(np.linspace(X0[0], X1[0], L)).astype(int)
        y = np.rint(np.linspace(X0[1], X1[1], L)).astype(int)
        z = np.rint(np.linspace(X0[2], X1[2], L)).astype(int)
        return [x, y, z]
    else:
        L = np.amax(np.absolute([[X1[0] - X0[0]], [X1[1] - X0[1]]])) + 1
        x = np.rint(np.linspace(X0[0], X1[0], L)).astype(int)
        y = np.rint(np.linspace(X0[1], X1[1], L)).astype(int)
        return [x, y]
