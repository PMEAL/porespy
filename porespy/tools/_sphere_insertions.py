import numpy as np
import numba


__all__ = [
    '_make_disk',
    '_make_disks',
    '_make_ball',
    '_make_balls',
    '_insert_disk_at_points',
    '_insert_disks_at_points',
]


@numba.jit(nopython=True, parallel=False)
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


@numba.jit(nopython=True, parallel=False)
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


@numba.jit(nopython=True, parallel=False)
def _insert_disk_at_points(im, coords, r, v,
                           smooth=True, overwrite=False):  # pragma: no cover
    r"""
    Insert spheres (or disks) into the given ND-image at given locations

    This function uses numba to accelerate the process, and does not
    overwrite any existing values (i.e. only writes to locations containing
    zeros).

    Parameters
    ----------
    im : ND-array
        The image into which the spheres/disks should be inserted. This is an
        'in-place' operation.
    coords : ND-array
        The center point of each sphere/disk in an array of shape
        ``ndim by npts``
    r : int
        The radius of all the spheres/disks to add. It is assumed that they
        are all the same radius.
    v : scalar
        The value to insert
    smooth : boolean
        If ``True`` (default) then the spheres/disks will not have the litte
        nibs on the surfaces.

    """
    npts = len(coords[0])
    if im.ndim == 2:
        xlim, ylim = im.shape
        s = _make_disk(r, smooth)
        for i in range(npts):
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
        s = _make_ball(r, smooth)
        for i in range(npts):
            pt = coords[:, i]
            for a, x in enumerate(range(pt[0]-r, pt[0]+r+1)):
                if (x >= 0) and (x < xlim):
                    for b, y in enumerate(range(pt[1]-r, pt[1]+r+1)):
                        if (y >= 0) and (y < ylim):
                            for c, z in enumerate(range(pt[2]-r, pt[2]+r+1)):
                                if (z >= 0) and (z < zlim):
                                    if (s[a, b, c] == 1):
                                        if overwrite or (im[x, y, z] == 0):
                                            im[x, y, z] = v
    return im


@numba.jit(nopython=True, parallel=False)
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


@numba.jit(nopython=True, parallel=False)
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


@numba.jit(nopython=True, parallel=False)
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
