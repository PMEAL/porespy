import numpy as np
from numba import njit, prange

__all__ = [
    '_make_disk',
    '_make_disks',
    '_make_ball',
    '_make_balls',
    '_insert_disk_at_points',
    '_insert_disk_at_point',
    '_insert_disk_at_points_parallel',
    '_insert_disks_at_points',
    '_insert_disks_at_points_serial',
    '_insert_disks_at_points_parallel',
    '_insert_shape_at_point',
    '_insert_shape_at_points',
    '_insert_shape_at_points_parallel',
    'insert_shape_at_points',
    'points_to_spheres',
]


# Mode codes for the `_insert_shape_at_point*` primitives. Kept as ints so
# numba can dispatch in the inner loop without string handling.
_MODE_PRESERVE = 0
_MODE_OVERWRITE = 1
_MODE_ADD = 2

_MODE_LOOKUP = {
    'preserve': _MODE_PRESERVE,
    'overwrite': _MODE_OVERWRITE,
    'add': _MODE_ADD,
    'overlay': _MODE_ADD,
}


def points_to_spheres(im):
    r"""
    Inserts disks/spheres into an image at locations indicated by non-zero values

    Parameters
    ----------
    im : ndarray
        The image containing nonzeros indicating the locations to insert
        spheres. If the non-zero values are ``bool``, then the maximal size is
        found and used. If the non-zeros are ``int`` then these values are used
        as the radii.

    Returns
    -------
    spheres : ndarray
        A `bool` array with disks/spheres inserted at each nonzero location in
        ``im``.
    """
    from scipy.spatial import distance_matrix
    if im.ndim == 3:
        x, y, z = np.where(im > 0)
        coords = np.vstack((x, y, z))
    else:
        x, y = np.where(im > 0)
        coords = np.vstack((x, y))
    if im.dtype == bool:
        dmap = distance_matrix(coords.T, coords.T)
        mask = dmap < 1
        dmap[mask] = np.inf
        r = np.around(dmap.min(axis=0)/2, decimals=0).astype(int)
    else:
        if im.ndim == 3:
            r = im[x, y, z].flatten()
        else:
            r = im[x, y].flatten()
    im_spheres = np.zeros_like(im, dtype=bool)
    im_spheres = _insert_disks_at_points_parallel(
        im_spheres,
        coords=coords,
        radii=r,
        v=True,
        smooth=False,
    )
    return im_spheres


@njit(parallel=True)
def _insert_disks_at_points_parallel(im, coords, radii, v, smooth=True,
                                     overwrite=False):  # pragma: no cover
    npts = len(coords[0])
    if im.ndim == 2:
        xlim, ylim = im.shape
        for i in prange(npts):
            r = radii[i]
            pt = coords[:, i]
            for a, x in enumerate(range(pt[0]-r, pt[0]+r+1)):
                if (x >= 0) and (x < xlim):
                    for b, y in enumerate(range(pt[1]-r, pt[1]+r+1)):
                        if (y >= 0) and (y < ylim):
                            R = ((a - r)**2 + (b - r)**2)**0.5
                            if (R <= r)*(~smooth) or (R < r)*(smooth):
                                if overwrite or (im[x, y] == 0):
                                    im[x, y] = v
    elif im.ndim == 3:
        xlim, ylim, zlim = im.shape
        for i in prange(npts):
            r = radii[i]
            pt = coords[:, i]
            for a, x in enumerate(range(pt[0]-r, pt[0]+r+1)):
                if (x >= 0) and (x < xlim):
                    for b, y in enumerate(range(pt[1]-r, pt[1]+r+1)):
                        if (y >= 0) and (y < ylim):
                            for c, z in enumerate(range(pt[2]-r, pt[2]+r+1)):
                                if (z >= 0) and (z < zlim):
                                    R = ((a - r)**2 + (b - r)**2 + (c - r)**2)**0.5
                                    if (R <= r)*(~smooth) or (R < r)*(smooth):
                                        if overwrite or (im[x, y, z] == 0):
                                            im[x, y, z] = v
    return im


@njit
def _insert_disks_at_points_serial(im, coords, radii, v, smooth=True,
                                   overwrite=False):  # pragma: no cover
    r"""
    Insert spheres (or disks) of specified radii into an ND-image at given locations.

    This function uses numba to accelerate the process, and does not overwrite
    any existing values (i.e. only writes to locations containing zeros).

    Parameters
    ----------
    im : ND-array
        The image into which the spheres/disks should be inserted. This is an
        'in-place' operation.
    coords : ND-array
        The center point of each sphere/disk in an array of shape
        ``ndim by npts``
    radii : array_like
        The radii of the spheres/disks to add.
    v : scalar
        The value to insert
    smooth : boolean, optional
        If ``True`` (default) then the spheres/disks will not have the litte
        nibs on the surfaces.
    overwrite : boolean, optional
        If ``True`` then the inserted spheres overwrite existing values.  The
        default is ``False``.

    """
    npts = len(coords[0])
    if im.ndim == 2:
        xlim, ylim = im.shape
        for i in range(npts):
            r = radii[i]
            pt = coords[:, i]
            for a, x in enumerate(range(pt[0]-r, pt[0]+r+1)):
                if (x >= 0) and (x < xlim):
                    for b, y in enumerate(range(pt[1]-r, pt[1]+r+1)):
                        if (y >= 0) and (y < ylim):
                            R = ((a - r)**2 + (b - r)**2)**0.5
                            if (R <= r)*(~smooth) or (R < r)*(smooth):
                                if overwrite or (im[x, y] == 0):
                                    im[x, y] = v
    elif im.ndim == 3:
        xlim, ylim, zlim = im.shape
        for i in range(npts):
            r = radii[i]
            pt = coords[:, i]
            for a, x in enumerate(range(pt[0]-r, pt[0]+r+1)):
                if (x >= 0) and (x < xlim):
                    for b, y in enumerate(range(pt[1]-r, pt[1]+r+1)):
                        if (y >= 0) and (y < ylim):
                            for c, z in enumerate(range(pt[2]-r, pt[2]+r+1)):
                                if (z >= 0) and (z < zlim):
                                    R = ((a - r)**2 + (b - r)**2 + (c - r)**2)**0.5
                                    if (R <= r)*(~smooth) or (R < r)*(smooth):
                                        if overwrite or (im[x, y, z] == 0):
                                            im[x, y, z] = v
    return im


@njit(parallel=False)
def _insert_shape_at_point(im, coords, element, v, mode):  # pragma: no cover
    r"""
    Stamp ``element`` into ``im`` centered at a single coordinate.

    Only writes at locations where ``element`` is non-zero, so the stamp acts
    as its own mask. ``element`` must have odd-length sides; the centre is at
    ``element.shape[d] // 2`` along each axis.

    Parameters
    ----------
    im : ND-array
        The image to stamp into. Modified in place.
    coords : array_like
        The centre coordinate of length ``ndim``.
    element : ND-array
        The sub-image to stamp. Must match the dimensionality of ``im``.
    v : scalar
        The value written at active pixels (multiplied by ``element[a, b, ...]``
        to keep weighted stamps working).
    mode : int
        One of ``_MODE_PRESERVE`` (only writes where ``im`` is 0),
        ``_MODE_OVERWRITE`` (replaces), or ``_MODE_ADD`` (in-place add).
    """
    if im.ndim == 2:
        xlim, ylim = im.shape
        rx = element.shape[0] // 2
        ry = element.shape[1] // 2
        for a in range(element.shape[0]):
            x = coords[0] - rx + a
            if (x < 0) or (x >= xlim):
                continue
            for b in range(element.shape[1]):
                y = coords[1] - ry + b
                if (y < 0) or (y >= ylim):
                    continue
                e = element[a, b]
                if e == 0:
                    continue
                if mode == 0:
                    if im[x, y] == 0:
                        im[x, y] = e * v
                elif mode == 1:
                    im[x, y] = e * v
                else:
                    im[x, y] += e * v
    elif im.ndim == 3:
        xlim, ylim, zlim = im.shape
        rx = element.shape[0] // 2
        ry = element.shape[1] // 2
        rz = element.shape[2] // 2
        for a in range(element.shape[0]):
            x = coords[0] - rx + a
            if (x < 0) or (x >= xlim):
                continue
            for b in range(element.shape[1]):
                y = coords[1] - ry + b
                if (y < 0) or (y >= ylim):
                    continue
                for c in range(element.shape[2]):
                    z = coords[2] - rz + c
                    if (z < 0) or (z >= zlim):
                        continue
                    e = element[a, b, c]
                    if e == 0:
                        continue
                    if mode == 0:
                        if im[x, y, z] == 0:
                            im[x, y, z] = e * v
                    elif mode == 1:
                        im[x, y, z] = e * v
                    else:
                        im[x, y, z] += e * v
    return im


@njit(parallel=False)
def _insert_shape_at_points(im, coords, element, v, mode):  # pragma: no cover
    r"""
    Stamp ``element`` into ``im`` at every column of ``coords``.

    See ``_insert_shape_at_point`` for the per-pixel semantics. ``coords``
    is shaped ``(ndim, npts)``.
    """
    npts = len(coords[0])
    if im.ndim == 2:
        xlim, ylim = im.shape
        rx = element.shape[0] // 2
        ry = element.shape[1] // 2
        for i in range(npts):
            for a in range(element.shape[0]):
                x = coords[0, i] - rx + a
                if (x < 0) or (x >= xlim):
                    continue
                for b in range(element.shape[1]):
                    y = coords[1, i] - ry + b
                    if (y < 0) or (y >= ylim):
                        continue
                    e = element[a, b]
                    if e == 0:
                        continue
                    if mode == 0:
                        if im[x, y] == 0:
                            im[x, y] = e * v
                    elif mode == 1:
                        im[x, y] = e * v
                    else:
                        im[x, y] += e * v
    elif im.ndim == 3:
        xlim, ylim, zlim = im.shape
        rx = element.shape[0] // 2
        ry = element.shape[1] // 2
        rz = element.shape[2] // 2
        for i in range(npts):
            for a in range(element.shape[0]):
                x = coords[0, i] - rx + a
                if (x < 0) or (x >= xlim):
                    continue
                for b in range(element.shape[1]):
                    y = coords[1, i] - ry + b
                    if (y < 0) or (y >= ylim):
                        continue
                    for c in range(element.shape[2]):
                        z = coords[2, i] - rz + c
                        if (z < 0) or (z >= zlim):
                            continue
                        e = element[a, b, c]
                        if e == 0:
                            continue
                        if mode == 0:
                            if im[x, y, z] == 0:
                                im[x, y, z] = e * v
                        elif mode == 1:
                            im[x, y, z] = e * v
                        else:
                            im[x, y, z] += e * v
    return im


@njit(parallel=True)
def _insert_shape_at_points_parallel(im, coords, element, v,
                                     mode):  # pragma: no cover
    r"""
    Parallel variant of ``_insert_shape_at_points`` (``prange`` over points).
    """
    npts = len(coords[0])
    if im.ndim == 2:
        xlim, ylim = im.shape
        rx = element.shape[0] // 2
        ry = element.shape[1] // 2
        for i in prange(npts):
            for a in range(element.shape[0]):
                x = coords[0, i] - rx + a
                if (x < 0) or (x >= xlim):
                    continue
                for b in range(element.shape[1]):
                    y = coords[1, i] - ry + b
                    if (y < 0) or (y >= ylim):
                        continue
                    e = element[a, b]
                    if e == 0:
                        continue
                    if mode == 0:
                        if im[x, y] == 0:
                            im[x, y] = e * v
                    elif mode == 1:
                        im[x, y] = e * v
                    else:
                        im[x, y] += e * v
    elif im.ndim == 3:
        xlim, ylim, zlim = im.shape
        rx = element.shape[0] // 2
        ry = element.shape[1] // 2
        rz = element.shape[2] // 2
        for i in prange(npts):
            for a in range(element.shape[0]):
                x = coords[0, i] - rx + a
                if (x < 0) or (x >= xlim):
                    continue
                for b in range(element.shape[1]):
                    y = coords[1, i] - ry + b
                    if (y < 0) or (y >= ylim):
                        continue
                    for c in range(element.shape[2]):
                        z = coords[2, i] - rz + c
                        if (z < 0) or (z >= zlim):
                            continue
                        e = element[a, b, c]
                        if e == 0:
                            continue
                        if mode == 0:
                            if im[x, y, z] == 0:
                                im[x, y, z] = e * v
                        elif mode == 1:
                            im[x, y, z] = e * v
                        else:
                            im[x, y, z] += e * v
    return im


def insert_shape_at_points(im, coords, element, value=1, mode='preserve'):
    r"""
    Insert a sub-image (``element``) at one or more coordinates.

    A generic version of ``_insert_disk_at_points``: the radius is replaced
    by an arbitrary stamp. Active pixels are those where ``element`` is
    non-zero, so the stamp doubles as its own mask.

    Parameters
    ----------
    im : ND-array
        The image into which ``element`` is stamped. Modified in place.
    coords : array_like
        Coordinates at which to stamp ``element``. Either a 1-D array of
        length ``ndim`` (single point) or a 2-D array of shape
        ``(ndim, npts)``.
    element : ND-array
        The sub-image to stamp. Each side must be odd so the centre is well
        defined.
    value : scalar, optional
        Value written at active pixels (multiplied by ``element``). Default
        is 1.
    mode : str, optional
        One of ``'preserve'`` (only writes where ``im`` is currently 0),
        ``'overwrite'`` (replaces), or ``'add'`` / ``'overlay'`` (in-place
        add). Default is ``'preserve'``.

    Returns
    -------
    im : ND-array
        The same array passed in, modified in place.
    """
    if mode not in _MODE_LOOKUP:
        raise ValueError(
            f"Invalid mode {mode!r}; must be one of {sorted(set(_MODE_LOOKUP))}"
        )
    coords = np.asarray(coords)
    if coords.ndim == 1:
        return _insert_shape_at_point(im, coords, element, value,
                                      _MODE_LOOKUP[mode])
    return _insert_shape_at_points(im, coords, element, value,
                                   _MODE_LOOKUP[mode])


@njit(parallel=False)
def _insert_disk_at_point(im, coords, r, v,
                          smooth=True, overwrite=False):  # pragma: no cover
    r"""
    Insert a disk/ball of radius ``r`` at a single coordinate.

    Thin wrapper around ``_insert_shape_at_point`` that builds the disk/ball
    stencil. See that function for the per-pixel semantics.
    """
    if im.ndim == 2:
        s = _make_disk(r, smooth)
    else:
        s = _make_ball(r, smooth)
    mode = 1 if overwrite else 0
    return _insert_shape_at_point(im, coords, s, v, mode)


@njit(parallel=False)
def _insert_disk_at_points(im, coords, r, v,
                           smooth=True, overwrite=False):  # pragma: no cover
    r"""
    Insert disks/balls of radius ``r`` at every column of ``coords``.

    Thin wrapper around ``_insert_shape_at_points``.
    """
    if im.ndim == 2:
        s = _make_disk(r, smooth)
    else:
        s = _make_ball(r, smooth)
    mode = 1 if overwrite else 0
    return _insert_shape_at_points(im, coords, s, v, mode)


@njit(parallel=False)
def _insert_disk_at_points_parallel(im, coords, r, v, smooth=True,
                                    overwrite=False):  # pragma: no cover
    r"""
    Parallel variant of ``_insert_disk_at_points``. The wrapper itself
    is serial; the per-point parallelism lives inside the inner call.
    """
    if im.ndim == 2:
        s = _make_disk(r, smooth)
    else:
        s = _make_ball(r, smooth)
    mode = 1 if overwrite else 0
    return _insert_shape_at_points_parallel(im, coords, s, v, mode)


@njit(parallel=False)
def _insert_disks_at_points(im, coords, radii, v, smooth=True,
                            overwrite=False):  # pragma: no cover
    r"""
    Insert spheres (or disks) of specified radii into an ND-image at given locations.

    This function uses numba to accelerate the process, and does not overwrite
    any existing values (i.e. only writes to locations containing zeros).

    Parameters
    ----------
    im : ND-array
        The image into which the spheres/disks should be inserted. This is an
        'in-place' operation.
    coords : ND-array
        The center point of each sphere/disk in an array of shape
        ``ndim by npts``
    radii : array_like
        The radii of the spheres/disks to add.
    v : scalar
        The value to insert
    smooth : boolean, optional
        If ``True`` (default) then the spheres/disks will not have the litte
        nibs on the surfaces.
    overwrite : boolean, optional
        If ``True`` then the inserted spheres overwrite existing values.  The
        default is ``False``.

    """
    npts = len(coords[0])
    if im.ndim == 2:
        xlim, ylim = im.shape
        for i in range(npts):
            r = radii[i]
            s = _make_disk(r, smooth)
            pt = coords[:, i]
            for a, x in enumerate(range(pt[0]-r, pt[0]+r+1)):
                if (x >= 0) and (x < xlim):
                    for b, y in enumerate(range(pt[1]-r, pt[1]+r+1)):
                        if (y >= 0) and (y < ylim):
                            if s[a, b] == 1:
                                if overwrite or (im[x, y] == 0):
                                    im[x, y] = v
    elif im.ndim == 3:
        xlim, ylim, zlim = im.shape
        for i in range(npts):
            r = radii[i]
            s = _make_ball(r, smooth)
            pt = coords[:, i]
            for a, x in enumerate(range(pt[0]-r, pt[0]+r+1)):
                if (x >= 0) and (x < xlim):
                    for b, y in enumerate(range(pt[1]-r, pt[1]+r+1)):
                        if (y >= 0) and (y < ylim):
                            for c, z in enumerate(range(pt[2]-r, pt[2]+r+1)):
                                if (z >= 0) and (z < zlim):
                                    if s[a, b, c] == 1:
                                        if overwrite or (im[x, y, z] == 0):
                                            im[x, y, z] = v
    return im


@njit(parallel=False)
def _make_disk(r, smooth=True):  # pragma: no cover
    r"""
    Generate a circular structuring element of the given radius

    Parameters
    ----------
    r : int
        The radius of the desired disk
    smooth : boolean
        If ``True`` (default) then the disk will not have the litte
        nibs on the surfaces.

    Returns
    -------
    disk : ndarray
        A numpy array of 1 and 0 suitable for use as a structuring element
    """
    s = np.zeros((2*r+1, 2*r+1), dtype=type(r))
    if smooth:
        thresh = r - 0.001
    else:
        thresh = r
    for i in range(2*r+1):
        for j in range(2*r+1):
            if ((i - r)**2 + (j - r)**2)**0.5 <= thresh:
                s[i, j] = 1
    return s


@njit(parallel=False)
def _make_ball(r, smooth=True):  # pragma: no cover
    r"""
    Generate a spherical structuring element of the given radius

    Parameters
    ----------
    r : int
        The radius of the desired ball
    smooth : boolean
        If ``True`` (default) then the ball will not have the litte
        nibs on the surfaces.

    Returns
    -------
    ball : ndarray
        A numpy array of 1 and 0 suitable for use as a structuring element
    """
    s = np.zeros((2*r+1, 2*r+1, 2*r+1), dtype=type(r))
    if smooth:
        thresh = r - 0.001
    else:
        thresh = r
    for i in range(2*r+1):
        for j in range(2*r+1):
            for k in range(2*r+1):
                if ((i - r)**2 + (j - r)**2 + (k - r)**2)**0.5 <= thresh:
                    s[i, j, k] = 1
    return s


@njit(parallel=False)
def _make_disks(r, smooth=True):  # pragma: no cover
    r"""
    Returns a list of disks from size 0 to ``r``

    Parameters
    ----------
    r : int
        The size of the largest disk to generate
    smooth : bool
        Indicates whether the disks should include the nibs (``False``) on
        the surface or not (``True``).  The default is ``True``.

    Returns
    -------
    disks : list of ND-arrays
        A list containing the disk images, with the disk of radius R at index
        R of the list, meaning it can be accessed as ``disks[R]``.

    """
    disks = []
    for val in range(0, r):
        disk = _make_disk(val, smooth)
        disks.append(disk)
    return disks


@njit(parallel=False)
def _make_balls(r, smooth=True):  # pragma: no cover
    r"""
    Returns a list of balls from size 0 to ``r``

    Parameters
    ----------
    r : int
        The size of the largest ball to generate
    smooth : bool
        Indicates whether the balls should include the nibs (``False``) on
        the surface or not (``True``).  The default is ``True``.

    Returns
    -------
    balls : list of ND-arrays
        A list containing the ball images, with the ball of radius R at index
        R of the list, meaning it can be accessed as ``balls[R]``.

    """
    balls = []
    for val in range(0, r):
        ball = _make_ball(val, smooth)
        balls.append(ball)
    return balls


if __name__ == "__main__":
    import numpy as np

    from porespy.tools import tic, toc

    np.random.seed(0)
    im = np.random.rand(400, 400, 400) > 0.995
    coords = np.where(im)
    rs = np.random.randint(5, 10, len(coords[0]))

    im2 = np.zeros_like(im)
    # Call function once to trigger jit before timing
    im2 = _insert_disk_at_points(im=im2, coords=np.vstack(coords),
                                 r=10, v=1, smooth=True)
    im2 = np.zeros_like(im)
    tic()
    im2 = _insert_disk_at_points(im=im2, coords=np.vstack(coords),
                                 r=10, v=1, smooth=True)
    t = toc(quiet=True)
    print(f"Single radii, serial: {t}")

    im2 = np.zeros_like(im)
    im2 = _insert_disk_at_points_parallel(im=im2, coords=np.vstack(coords),
                                          r=10, v=1, smooth=True)
    im2 = np.zeros_like(im)
    tic()
    im2 = _insert_disk_at_points_parallel(im=im2, coords=np.vstack(coords),
                                          r=10, v=1, smooth=True)
    t = toc(quiet=True)
    print(f"Single radii, parallel: {t}")

    im3 = np.zeros_like(im)
    im3 = _insert_disks_at_points(im=im3, coords=np.vstack(coords),
                                         radii=rs, v=1)
    im3 = np.zeros_like(im)
    tic()
    im3 = _insert_disks_at_points(im=im3, coords=np.vstack(coords),
                                         radii=rs, v=1)
    t = toc(quiet=True)
    print(f"Multiple radii, legacy: {t}")

    im3 = np.zeros_like(im)
    im3 = _insert_disks_at_points_serial(im=im3, coords=np.vstack(coords), radii=rs, v=1)
    im3 = np.zeros_like(im)
    tic()
    im3 = _insert_disks_at_points_serial(im=im3, coords=np.vstack(coords), radii=rs, v=1)
    t = toc(quiet=True)
    print(f"Multiple radii, new: {t}")

    im4 = np.zeros_like(im)
    im4 = _insert_disks_at_points_parallel(im=im4, coords=np.vstack(coords),
                                           radii=rs, v=1)
    im4 = np.zeros_like(im)
    tic()
    im4 = _insert_disks_at_points_parallel(im=im4, coords=np.vstack(coords),
                                           radii=rs, v=1)
    t = toc(quiet=True)
    print(f"Multiple radii, parallel: {t}")
