import numpy as np
import porespy as ps
from porespy.tools import get_edt


__all__ = [
    "fibers",
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


def fibers(shape, r, n=None, porosity=None, phi_max=90, theta_max=90,
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
        Number of candidate lines to sample. Some lines miss the image.
        Specify exactly one of ``n`` and ``porosity``.
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
    >>> from porespy.beta import fibers
    >>> im = fibers([50, 50, 50], r=2, n=100, seed=0)
    >>> mat = fibers([50, 50, 50], r=2, n=100, phi_max=0, seed=0)
    >>> aligned = fibers([50, 50, 50], r=2, n=100,
    ...                  phi_max=0, theta_max=0, seed=0)
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
