import numpy as np
import porespy as ps
from porespy.tools import get_edt


__all__ = [
    "fibers_2D",
    "fibers_1D",
    "fibers_3D",
]


edt = get_edt()


def _fiber_directions(n, phi_max, theta_max):
    """Sample unoriented axes uniformly in the specified angular window."""
    theta = np.random.uniform(-np.deg2rad(theta_max), np.deg2rad(theta_max), n)
    z = np.random.uniform(-np.sin(np.deg2rad(phi_max)),
                          np.sin(np.deg2rad(phi_max)), n)
    xy = np.sqrt(1 - z*z)
    directions = np.column_stack((xy*np.cos(theta), xy*np.sin(theta), z))
    # This perpendicular basis remains well defined even near the poles.
    transverse = np.column_stack((-np.sin(theta), np.cos(theta), np.zeros(n)))
    return directions, transverse


def fibers_3D(shape, r, n=None, porosity=None, phi_max=90, theta_max=90,
              maxiter=10, seed=None):
    r"""
    Generate overlapping fibers with spatially uniform random placement.

    Parameters
    ----------
    shape : sequence of three ints
        Image dimensions in [Nx, Ny, Nz] order.
    r : float
        Positive fiber radius in voxels.
    n : int, optional
        Number of candidate lines to sample. Some lines miss the image, as in
        ``fibers_2D``. Specify exactly one of ``n`` and ``porosity``.
    porosity : float, optional
        Target void fraction, strictly between zero and one. The achieved
        porosity is approximate because fibers overlap and are voxelized.
    phi_max : float
        Maximum elevation above or below the XY plane, in degrees, from 0 to
        90. Zero confines fiber axes to XY planes. Elevation is sampled
        uniformly in its sine, rather than uniformly in angle.
    theta_max : float
        Maximum in-plane rotation away from the X direction, in degrees,
        from 0 to 90. Zero confines axes to the XZ plane. Both angular limits
        set to zero give X-aligned fibers; both set to 90 give isotropic axes.
    maxiter : int
        Maximum number of batches used to approach a target porosity. Stops
        when the void fraction is within 0.01 of the target or below it.
    seed : int, optional
        Seed for NumPy's global random generator. If omitted, the current
        global state is used, as in the other fiber generators.

    Returns
    -------
    im : ndarray
        Boolean image with True indicating void and False indicating fibers.

    Notes
    -----
    For each direction, line offsets are sampled uniformly in a perpendicular
    disk enclosing the image and its radius padding. Sampling disk radius
    proportional to the square root of a uniform variate gives uniform area
    density. Lines are allowed to miss the image, avoiding the position and
    direction bias of repeatedly sampling until a line intersects it.

    Lines extend beyond the padded image, are rasterized, and are thickened
    with a distance transform. The output therefore has voxel discretization
    error, especially for small radii. Finite samples need not have exactly
    uniform density or exactly isotropic orientation.

    Examples
    --------
    >>> from porespy.beta import fibers_3D
    >>> im = fibers_3D([50, 50, 50], r=2, n=100, seed=0)
    >>> mat = fibers_3D([50, 50, 50], r=2, n=100, phi_max=0, seed=0)
    """
    shape = np.asarray(shape)
    if (shape.shape != (3,) or not np.all(np.isfinite(shape))
            or np.any(shape <= 0) or np.any(shape != np.floor(shape))):
        raise ValueError('shape must contain three positive integers')
    shape = shape.astype(int)
    if not np.isscalar(r) or not np.isfinite(r) or r <= 0:
        raise ValueError('r must be positive and finite')
    for name, angle in [('phi_max', phi_max), ('theta_max', theta_max)]:
        if not np.isscalar(angle) or not np.isfinite(angle) or not 0 <= angle <= 90:
            raise ValueError(f'{name} must be between 0 and 90 degrees')
    if (n is None) == (porosity is None):
        raise ValueError('Specify exactly one of n and porosity')
    if n is not None and (not isinstance(n, (int, np.integer)) or n < 0):
        raise ValueError('n must be a nonnegative integer')
    if porosity is not None and (not np.isscalar(porosity)
                                or not np.isfinite(porosity)
                                or not 0 < porosity < 1):
        raise ValueError('porosity must be strictly between 0 and 1')
    if not isinstance(maxiter, (int, np.integer)) or maxiter < 1:
        raise ValueError('maxiter must be a positive integer')
    if seed is not None:
        np.random.seed(seed)

    pad = int(np.ceil(r)) + 1
    padded_shape = shape + 2*pad
    center = (padded_shape - 1)/2
    radius = np.linalg.norm(padded_shape/2)
    lines = np.zeros(padded_shape, dtype=bool)
    crop = tuple(slice(pad, -pad) for _ in shape)
    im = np.ones(shape, dtype=bool)
    total = 0
    batch = n if n is not None else max(
        1, int(np.ceil(-0.5*np.log(porosity)*radius**2/r**2)))

    for _ in range(1 if n is not None else maxiter):
        directions, transverse = _fiber_directions(batch, phi_max, theta_max)
        offsets = radius*np.sqrt(np.random.random(batch))
        angles = np.random.uniform(0, 2*np.pi, batch)
        for direction, u, offset, angle in zip(
                directions, transverse, offsets, angles):
            v = np.cross(direction, u)
            point = center + offset*(np.cos(angle)*u + np.sin(angle)*v)
            coords = np.asarray(ps.generators.line_segment(
                point - radius*direction, point + radius*direction))
            valid = np.all((coords >= 0) & (coords < padded_shape[:, None]), axis=0)
            lines[tuple(coords[:, valid])] = True
        total += batch
        if lines.any():
            im = (edt(~lines) >= r)[crop]
        if porosity is None:
            break
        current = im.mean()
        if current <= porosity + 0.01:
            break
        # Estimate the observed attenuation per candidate line. Retain the
        # continuum estimate when the first batch happened to miss the image.
        rate = -np.log(current)/total if current < 1 else r**2/radius**2
        batch = max(1, int(np.ceil(0.8*np.log(current/porosity)/rate)))
    return im


def fibers_1D(shape, r, n, overlap=None):
    r"""
    Generates randomly located fibers which are all oriented in z-direction

    Parameters
    ----------
    shape : list
        The shape of the image to generate.  This must be 3D.
    r : int
        The radius of the fibers to generate
    n : int
        The number of fibers to draw

    Returns
    -------
    im : ndarray
        An `ndarray` of the given shape with fibers indicated by `False` and void
        by `True`.

    Notes
    -----
    ..  [1] Tomadakis MM, Sotirchos SV. Effective diffusivities and conductivities
        of random dispersions of nonoverlapping and partially overlapping
        unidirectional fibers. Journal of Chemical Physics. 99, 9820–9827 (1993).
        `doi <https://doi.org/10.1063/1.465464>`_

    """
    from porespy.generators import random_spheres
    if overlap is None:
        im = np.ones(np.array(shape[:2]) + r, dtype=bool)
        locs = np.random.randint((0, 0), shape[:2], (n, 2))
        im[tuple(locs.T)] = False
        dt = edt(im)
        im = dt >= (r + 0.001)
        im = im[r:-r, r:-r]
    else:
        im = ~random_spheres(
            np.array(shape[:2]) + r,
            r=0,
            clearance=-overlap,
            phi=1,
            maxiter=n,
            edges='extended',
        )
        im = im[r:-r, r:-r]
    if len(shape) == 3:
        im = np.tile(im, [shape[2], 1, 1])
        im = np.rollaxis(im, 0, 3)
    return im


def fibers_2D(shape, r, n=None, porosity=None):
    r"""
    Generates randomly located fibers which are randomly oriented in the x-y plane
    but aligned in the z-plane

    Parameters
    ----------
    shape : list
        The shape of the image to generate.  This must be 3D.
    r : int
        The radius of the fibers to generate
    porosity : float
        The radius of the fibers

    Returns
    -------
    im : ndarray
        An `ndarray` of the given shape with fibers indicated by `False` and void
        by `True`.

    Notes
    -----
    Drawing randomly oriented fibers with a uniform density is a surprizingly
    tricky task. The problem was recently discussed and a solution proposed by
    Beckman et al [1]_.

    ..  [1] Beckman IP, Beckman PM, Cho H, Riveros G. Modeling uniform random
        distributions of nonwoven fibers for computational analysis of
        composite materials. Composite Structures. 301(12), 116242 (2022).
        `doi <https://doi.org/10.1016/j.compstruct.2022.116242>`_
    """

    def add_n_lines(im, r, n):
        x1, y1 = im.shape[0]/2, im.shape[1]/2
        Lmax = (x1**2 + y1**2)**0.5
        for _ in range(n):
            phi = np.random.rand()*2*np.pi
            L = 2*(0.5 - np.random.rand())*Lmax
            x2 = x1 + L*np.cos(phi)
            y2 = y1 + L*np.sin(phi)
            x3 = x2 + 2*Lmax*np.cos(phi + np.pi/2)
            y3 = y2 + 2*Lmax*np.sin(phi + np.pi/2)
            x4 = x2 + 2*Lmax*np.cos(phi - np.pi/2)
            y4 = y2 + 2*Lmax*np.sin(phi - np.pi/2)
            X = ps.generators.line_segment((x3, y3), (x4, y4))
            mask = (X[0] >= 0)*(X[0] < im.shape[0])*(X[1] >= 0)*(X[1] < im.shape[1])
            X = X[0][mask], X[1][mask]
            if im.ndim == 3:
                X = X[0], X[1], np.random.randint(im.shape[2])
            im[tuple(X)] = True
        if r is not None:
            dt = edt(im == 0)
            im = dt >= r
        return im

    if (n is None) and (porosity is not None):
        shape = np.array(shape) + 2*r  # Add 2r so rounded fiber ends fall outside im
        porosity_orig = porosity
        im = np.zeros(shape, dtype=bool)
        iters = np.around(np.log(porosity)/(-4*r**2/im.size**(0.5))).astype(int)
        n = iters*10
        im = add_n_lines(im=im, r=r, n=n)
        porosity = 1 - (im.sum()/im.size - porosity_orig)
        while porosity < .99:
            iters = np.around(np.log(porosity)/(-2*r**2/im.size**(0.5))).astype(int)
            n = iters*10
            if n == 0:
                break
            im2 = np.zeros(shape, dtype=bool)
            im2 = add_n_lines(im=im2, r=r, n=n)
            im *= im2
            porosity = 1 - (im.sum()/im.size - porosity_orig)
        im = im[r:-r, r:-r, r:-r]  # Remove padding
    else:
        im = np.zeros(shape)
        im = add_n_lines(im=im, r=r, n=n)

    return im


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fibs = fibers_2D([200, 200, 100], r=5, porosity=0.85)
    print(f"Porosity: {fibs.sum()/fibs.size}")

    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(ps.visualization.xray(fibs > 0, axis=2))
    ax[1].imshow(ps.visualization.xray(fibs > 0, axis=1).T)
    ax[2].imshow(ps.visualization.xray(fibs > 0, axis=0).T)
