import numpy as np
from numba import get_num_threads, get_thread_id, njit, prange

__all__ = [
    '_make_disk',
    '_make_disks',
    '_make_ball',
    '_make_balls',
    '_make_axial_extent_lookup',
    '_get_axial_extent',
    '_insert_disks_at_indices_parallel',
    '_insert_disks_at_indices_parallel_direct',
    '_insert_disks_at_indices_parallel_merged',
    '_use_merged_intervals',
    '_insert_disk_at_points',
    '_insert_disk_at_point',
    '_insert_disk_at_points_parallel',
    '_insert_disks_at_points',
    '_insert_disks_at_points_serial',
    '_insert_disks_at_points_parallel',
    'points_to_spheres',
]


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


@njit
def _make_axial_extent_lookup(max_radius):
    """Build the lookup used to rasterize circular scan-line spans."""
    ceil_distance = np.empty(int(max_radius)**2 + 1, dtype=np.int32)
    for distance_squared in range(len(ceil_distance)):
        ceil_distance[distance_squared] = int(
            np.ceil(np.sqrt(distance_squared)))
    return ceil_distance


@njit(inline='always')
def _get_axial_extent(distance_squared, ceil_distance, smooth):
    if smooth:
        if distance_squared <= 0:
            return -1
        return int(ceil_distance[distance_squared]) - 1
    if distance_squared < 0:
        return -1
    extent = int(ceil_distance[distance_squared])
    if extent**2 > distance_squared:
        extent -= 1
    return extent


def _insert_disks_at_indices_parallel(
    im,
    indices,
    dt,
    ceil_distance,
    smooth=True,
    overwrite=False,
    fixed_radius=-1,
):  # pragma: no cover
    """Insert disks from flat indices using direct or merged scan-line writes."""
    if overwrite and _use_merged_intervals(im, indices, dt, fixed_radius):
        return _insert_disks_at_indices_parallel_merged(
            im=im,
            indices=indices,
            dt=dt,
            ceil_distance=ceil_distance,
            smooth=smooth,
            fixed_radius=fixed_radius,
        )
    return _insert_disks_at_indices_parallel_direct(
        im=im,
        indices=indices,
        dt=dt,
        ceil_distance=ceil_distance,
        smooth=smooth,
        overwrite=overwrite,
        fixed_radius=fixed_radius,
    )


@njit
def _use_merged_intervals(im, indices, dt, fixed_radius=-1):
    """Estimate whether merging scan-line intervals will reduce write work."""
    if len(indices) == 0:
        return False
    nsamples = min(len(indices), 256)
    estimated_intervals = 0
    if im.ndim == 2:
        ylim = im.shape[1]
        for q in range(nsamples):
            ind = indices[q * len(indices) // nsamples]
            i = ind // ylim
            j = ind - i * ylim
            r = fixed_radius if fixed_radius >= 0 else int(dt[i, j])
            estimated_intervals += 2 * r + 1
        nrows = im.shape[0]
    else:
        ylim, zlim = im.shape[1:]
        stride0 = ylim * zlim
        for q in range(nsamples):
            ind = indices[q * len(indices) // nsamples]
            i = ind // stride0
            rem = ind - i * stride0
            j = rem // zlim
            k = rem - j * zlim
            r = fixed_radius if fixed_radius >= 0 else int(dt[i, j, k])
            diameter = 2 * r + 1
            estimated_intervals += diameter**2
        nrows = im.shape[0] * im.shape[1]
    # Row buffers and their initialization cost more than direct writes for
    # sparse disks. Benchmarks place the crossover near 256 generated intervals
    # per output row, while strongly overlapping spheres exceed this by orders
    # of magnitude.
    return estimated_intervals * len(indices) >= 256 * nrows * nsamples


@njit(parallel=True)
def _insert_disks_at_indices_parallel_direct(
    im,
    indices,
    dt,
    ceil_distance,
    smooth=True,
    overwrite=False,
    fixed_radius=-1,
    value=True,
):  # pragma: no cover
    """Insert disks from flat indices using direct scan-line writes."""
    npts = len(indices)
    if im.ndim == 2:
        xlim, ylim = im.shape
        for q in prange(npts):
            ind = indices[q]
            i = ind // ylim
            j = ind - i * ylim
            r = fixed_radius if fixed_radius >= 0 else int(dt[i, j])
            radius_squared = r**2
            for x in range(max(0, i - r), min(i + r + 1, xlim)):
                dx = x - i
                y_extent = _get_axial_extent(
                    radius_squared - dx**2,
                    ceil_distance,
                    smooth,
                )
                y_start = max(0, j - y_extent)
                y_stop = min(j + y_extent + 1, ylim)
                if overwrite:
                    im[x, y_start:y_stop] = value
                else:
                    for y in range(y_start, y_stop):
                        if not im[x, y]:
                            im[x, y] = value
    elif im.ndim == 3:
        xlim, ylim, zlim = im.shape
        stride0 = ylim * zlim
        for q in prange(npts):
            ind = indices[q]
            i = ind // stride0
            rem = ind - i * stride0
            j = rem // zlim
            k = rem - j * zlim
            r = fixed_radius if fixed_radius >= 0 else int(dt[i, j, k])
            radius_squared = r**2
            for x in range(max(0, i - r), min(i + r + 1, xlim)):
                dx = x - i
                yz_extent = _get_axial_extent(
                    radius_squared - dx**2,
                    ceil_distance,
                    smooth,
                )
                for y in range(
                    max(0, j - yz_extent),
                    min(j + yz_extent + 1, ylim),
                ):
                    dy = y - j
                    z_extent = _get_axial_extent(
                        radius_squared - dx**2 - dy**2,
                        ceil_distance,
                        smooth,
                    )
                    z_start = max(0, k - z_extent)
                    z_stop = min(k + z_extent + 1, zlim)
                    if overwrite:
                        im[x, y, z_start:z_stop] = value
                    else:
                        for z in range(z_start, z_stop):
                            if not im[x, y, z]:
                                im[x, y, z] = value
    return im


@njit(parallel=True)
def _insert_disks_at_indices_parallel_merged(
    im,
    indices,
    dt,
    ceil_distance,
    smooth=True,
    fixed_radius=-1,
):  # pragma: no cover
    """Insert disks by merging consecutive overlapping scan-line intervals."""
    nthreads = get_num_threads()
    if im.ndim == 2:
        xlim, ylim = im.shape
        starts = np.full((nthreads, xlim), ylim, dtype=np.int32)
        stops = np.zeros((nthreads, xlim), dtype=np.int32)
        for q in prange(len(indices)):
            thread = get_thread_id()
            ind = indices[q]
            i = ind // ylim
            j = ind - i * ylim
            r = fixed_radius if fixed_radius >= 0 else int(dt[i, j])
            radius_squared = r**2
            for x in range(max(0, i - r), min(i + r + 1, xlim)):
                dx = x - i
                extent = _get_axial_extent(
                    radius_squared - dx**2,
                    ceil_distance,
                    smooth,
                )
                start = max(0, j - extent)
                stop = min(j + extent + 1, ylim)
                if start >= stop:
                    continue
                old_start = starts[thread, x]
                old_stop = stops[thread, x]
                if old_start == ylim:
                    starts[thread, x] = start
                    stops[thread, x] = stop
                elif (start <= old_stop) and (stop >= old_start):
                    starts[thread, x] = min(start, old_start)
                    stops[thread, x] = max(stop, old_stop)
                else:
                    im[x, old_start:old_stop] = True
                    starts[thread, x] = start
                    stops[thread, x] = stop
        for q in prange(nthreads * xlim):
            thread = q // xlim
            x = q - thread * xlim
            start = starts[thread, x]
            if start < ylim:
                im[x, start:stops[thread, x]] = True
    elif im.ndim == 3:
        xlim, ylim, zlim = im.shape
        stride0 = ylim * zlim
        nrows = xlim * ylim
        starts = np.full((nthreads, nrows), zlim, dtype=np.int32)
        stops = np.zeros((nthreads, nrows), dtype=np.int32)
        for q in prange(len(indices)):
            thread = get_thread_id()
            ind = indices[q]
            i = ind // stride0
            rem = ind - i * stride0
            j = rem // zlim
            k = rem - j * zlim
            r = fixed_radius if fixed_radius >= 0 else int(dt[i, j, k])
            radius_squared = r**2
            for x in range(max(0, i - r), min(i + r + 1, xlim)):
                dx = x - i
                yz_extent = _get_axial_extent(
                    radius_squared - dx**2,
                    ceil_distance,
                    smooth,
                )
                for y in range(
                    max(0, j - yz_extent),
                    min(j + yz_extent + 1, ylim),
                ):
                    dy = y - j
                    z_extent = _get_axial_extent(
                        radius_squared - dx**2 - dy**2,
                        ceil_distance,
                        smooth,
                    )
                    start = max(0, k - z_extent)
                    stop = min(k + z_extent + 1, zlim)
                    if start >= stop:
                        continue
                    row = x * ylim + y
                    old_start = starts[thread, row]
                    old_stop = stops[thread, row]
                    if old_start == zlim:
                        starts[thread, row] = start
                        stops[thread, row] = stop
                    elif (start <= old_stop) and (stop >= old_start):
                        starts[thread, row] = min(start, old_start)
                        stops[thread, row] = max(stop, old_stop)
                    else:
                        im[x, y, old_start:old_stop] = True
                        starts[thread, row] = start
                        stops[thread, row] = stop
        for q in prange(nthreads * nrows):
            thread = q // nrows
            row = q - thread * nrows
            start = starts[thread, row]
            if start < zlim:
                x = row // ylim
                y = row - x * ylim
                im[x, y, start:stops[thread, row]] = True
    return im


@njit(parallel=True)
def _insert_disks_at_points_parallel(im, coords, radii, v, smooth=True,
                                     overwrite=False):  # pragma: no cover
    npts = len(coords[0])
    max_radius = 0
    for i in range(npts):
        max_radius = max(max_radius, int(radii[i]))
    ceil_distance = _make_axial_extent_lookup(max_radius)
    if im.ndim == 2:
        xlim, ylim = im.shape
        for i in prange(npts):
            r = int(radii[i])
            pt = coords[:, i]
            radius_squared = r**2
            for x in range(max(0, pt[0] - r), min(pt[0] + r + 1, xlim)):
                dx = x - pt[0]
                y_extent = _get_axial_extent(
                    radius_squared - dx**2,
                    ceil_distance,
                    smooth,
                )
                for y in range(
                    max(0, pt[1] - y_extent),
                    min(pt[1] + y_extent + 1, ylim),
                ):
                    if overwrite or (im[x, y] == 0):
                        im[x, y] = v
    elif im.ndim == 3:
        xlim, ylim, zlim = im.shape
        for i in prange(npts):
            r = int(radii[i])
            pt = coords[:, i]
            radius_squared = r**2
            for x in range(max(0, pt[0] - r), min(pt[0] + r + 1, xlim)):
                dx = x - pt[0]
                yz_extent = _get_axial_extent(
                    radius_squared - dx**2,
                    ceil_distance,
                    smooth,
                )
                for y in range(
                    max(0, pt[1] - yz_extent),
                    min(pt[1] + yz_extent + 1, ylim),
                ):
                    dy = y - pt[1]
                    z_extent = _get_axial_extent(
                        radius_squared - dx**2 - dy**2,
                        ceil_distance,
                        smooth,
                    )
                    for z in range(
                        max(0, pt[2] - z_extent),
                        min(pt[2] + z_extent + 1, zlim),
                    ):
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
def _insert_disk_at_point(im, coords, r, v,
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
        The center point of the sphere/disk
    r : int
        The radius of all the spheres/disks to add. It is assumed that they
        are all the same radius.
    v : scalar
        The value to insert
    smooth : boolean
        If ``True`` (default) then the spheres/disks will not have the litte
        nibs on the surfaces.

    """
    if im.ndim == 2:
        xlim, ylim = im.shape
        s = _make_disk(r, smooth)
        pt = coords
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
        pt = coords
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


@njit(parallel=False)
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


@njit(parallel=True)
def _insert_disk_at_points_parallel(im, coords, r, v, smooth=True,
                                    overwrite=False):  # pragma: no cover
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
        for i in prange(npts):
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
        for i in prange(npts):
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
