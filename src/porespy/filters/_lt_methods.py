import inspect
import logging
from typing import Literal

import numpy as np
import numpy.typing as npt
from numba import njit, prange
from skimage.morphology import ball, disk, footprint_rectangle

from porespy.tools import (
    _insert_disk_at_points,
    get_edt,
    get_tqdm,
    ps_round,
    settings,
)

edt = get_edt()
tqdm = get_tqdm()
logger = logging.getLogger(__name__)
strel = {
    2: {'min': disk(1), 'max': footprint_rectangle((3, 3))},
    3: {'min': ball(1), 'max': footprint_rectangle((3, 3, 3))}
}

__all__ = [
    "local_thickness_bf",
    "local_thickness_imj",
    "local_thickness_dt",
    "local_thickness_conv",
    "local_thickness",
    "porosimetry",
]


def _index_dtype(n):
    """Smallest unsigned int dtype that holds values `0..n` inclusive."""
    if n <= np.iinfo(np.uint8).max:
        return np.uint8
    if n <= np.iinfo(np.uint16).max:
        return np.uint16
    return np.uint32


def porosimetry(
    im: npt.NDArray,
    dt: npt.NDArray = None,
    inlets: npt.NDArray = None,
    sizes: int = None,
    method: Literal['dsi', 'fft', 'dt'] = 'dt',
    smooth: bool = True,
    return_indices: bool = False,
):
    r"""
    Each location is assigned the radius of the largest sphere that can reach it
    from the given inlets.

    This function is essentially a local thickness filter but with access limitations
    so represents a form of porosimetry

    Parameters
    ----------
    im : ndarray
        Boolean image of the porous material
    dt : ndarray, optional
        The distance transform of the image
    inlets : ndarray, optional
        A boolean array the same sizes a `im`, with `True` values indicating the
        inlet locations. If not provided then all faces are used.
    method : str
        Which method to use to compute the local thickness. Options are:

        ======== ===================================================================
        Method   Description
        ======== ===================================================================
        'dt'     Uses distance transforms to perform erosion and dilation for each
                 radius in the image
        'dsi'     Uses brute-force to inserts spheres at each voxel
        'conv'   Uses FFT-based convolution to perform erosion and dilation for
                 each radius in the image
        ======== ===================================================================

    sizes : array_like or scalar
        This is only used if the method is `dt` or `conv`. If a list of values is
        provided they are used directly. If a scalar is provided then that number
        of points spanning the min and max of the distance transform are used.
        If `None`, then all the unique values in the distance transform are used,
        which may become time consuming. This can be sped up if `dt` is provided
        and rounded to the nearest integer first.
    smooth : bool, optional
        Indicates if protrusions should be removed from the faces of the spheres
        or not. Default is `True`.
    return_indices : bool, optional
        If `True`, return `(sizes, indices)` instead of a float image. `sizes` is
        a 1D array of applied radii with a leading `0` for unfilled voxels, and
        `indices` is an unsigned-int image such that `sizes[indices]` reproduces
        the float result. Avoids allocating a full float64 image, which matters
        for large tomograms. Only supported for `method='dt'`. Default is `False`.

    Returns
    -------
    sizes : ndarray
        In image with each voxel value indicating the largest overlapping sphere
        which can reach it from the given inlets. If `return_indices` is `True`,
        returns `(sizes, indices)` instead.

    See Also
    --------
    local_thickness
    drainage

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/porosimetry.html>`_
    to view online example.

    """
    if return_indices and method != 'dt':
        raise NotImplementedError(
            f"`return_indices=True` is only supported with `method='dt'`, got {method!r}"
        )
    if inlets is None:
        from porespy.generators import borders
        inlets = borders(im.shape, mode='faces')
    if dt is None:
        dt = edt(im)
    if sizes is None:
        sizes = np.unique(dt[im])
    if method == 'dt':
        from porespy.simulations import drainage_dt
        drn = drainage_dt(im=im, inlets=inlets, steps=sizes, smooth=smooth,
                          return_indices=return_indices)
    elif method in ['dsi', 'bf']:
        from porespy.simulations import drainage_bf
        drn = drainage_bf(im=im, inlets=inlets, steps=sizes, smooth=smooth)
    if method in ['fft', 'conv']:
        from porespy.simulations import drainage_conv
        drn = drainage_conv(im=im, inlets=inlets, steps=sizes, smooth=smooth)
    if return_indices:
        return drn.bins, drn.im_seq
    return drn.im_size


def local_thickness(
    im: npt.NDArray,
    dt: npt.NDArray = None,
    method: Literal['bf', 'imj', 'conv', 'dt'] = 'dt',
    smooth: bool = True,
    mask: npt.NDArray = None,
    approx: bool = False,
    sizes: int = 25,
    return_indices: bool = False,
):
    r"""
    Insert a maximally inscribed sphere at every pixel labelled by sphere radius

    This is a wrapper method for computing local thickness via a variety of
    different methods.

    Parameters
    ----------
    im : ndarray
        Boolean image of the porous material
    dt : ndarray, optional
        The distance transform of the image
    method : str
        Which method to use to compute the local thickness. Options are:

        ======== ===================================================================
        Method   Description
        ======== ===================================================================
        'dt'     Uses distance transforms to perform erosion and dilation for each
                 radius in the image
        'bf'     Uses brute-force to inserts spheres at each voxel
        'imj'    Uses the brute-force method but reduces the number of insertion
                 sites by 80-90% to speed up the process
        'conv'   Uses FFT-based convolution to perform erosion and dilation for
                 each radius in the image
        ======== ===================================================================

    sizes : array_like or scalar
        This is only used if the method is `dt` or `conv`. If a list of values is
        provided they are used directly. If a scalar is provided then that number
        of points spanning the min and max of the distance transform are used.
        If `None`, then all the unique values in the distance transform are used,
        which may become time consuming. This can be sped up if `dt` is provided
        and rounded to the nearest integer first.
    smooth : bool, optional
        Indicates if protrusions should be removed from the faces of the spheres
        or not. Default is `True`.
    mask : ndarray, optional
        This is only used if the method is `bf` or `imj`.  A boolean mask indicating
        which sites to insert spheres at. If not provided then all `True` values in
        `im` are used.
    approx : bool, optional
        This is only used if the method is `imj`. If `True` the algorithm is more
        aggressive at skipping voxels to process, which speeds things up, but this
        sacrifices accuracy in terms of a voxel-by-voxel match with the reference
        implementation. The default is `False`, meaning full accuracy is the default.
    return_indices : bool, optional
        If `True`, return `(sizes, indices)` instead of a float image. `sizes` is
        a 1D array of applied radii with a leading `0` for unfilled voxels, and
        `indices` is an unsigned-int image such that `sizes[indices]` reproduces
        the float result. Avoids allocating a full float64 image, which matters
        for large tomograms. Only supported for `method='dt'`. Default is `False`.

    Returns
    -------
    lt : ndarray
        The local thickness of the image with each voxel labelled according to the
        radius of the largest sphere which overlaps it. If `return_indices` is
        `True`, returns `(sizes, indices)` instead.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/local_thickness.html>`_
    to view online example.
    """

    if return_indices and method != 'dt':
        raise NotImplementedError(
            f"`return_indices=True` is only supported with `method='dt'`, got {method!r}"
        )

    if method == 'dt':
        lt = local_thickness_dt(
            im=im, dt=dt, sizes=sizes, smooth=smooth, return_indices=return_indices
        )
    elif method == 'imj':
        lt = local_thickness_imj(im=im, dt=dt, smooth=smooth)[0]
    elif method == 'bf':
        lt = local_thickness_bf(im=im, dt=dt, smooth=smooth)
    elif method == 'conv':
        lt = local_thickness_conv(im=im, dt=dt, sizes=sizes, smooth=smooth)
    else:
        raise Exception(f"Unrecognized method {method}")
    return lt


def local_thickness_bf(im, dt=None, mask=None, smooth=True):
    r"""
    Insert a maximally inscribed sphere at every pixel labelled by sphere radius

    Parameters
    ----------
    im : ndarray
        Boolean image of the porous material
    dt : ndarray, optional
        The distance transform of the image
    mask : ndarray, optional
        A boolean mask indicating which sites to insert spheres at. If not provided
        then all `True` values in `im` are used.
    smooth : bool, optional
        Indicates if protrusions should be removed from the faces of the spheres
        or not. Default is `True`.

    Returns
    -------
    lt : ndarray
        The local thickness of the image with each voxel labelled according to the
        radius of the largest sphere which overlaps it

    Notes
    -----
    This function uses brute force, meaning that is inserts spheres at every single
    pixel or voxel in the void phase without making any attempt to reduce the number
    of insertion sites. This provides a reference implementation for comparing
    accuracy of other methods.

    """
    if dt is None:
        dt = edt(im)
    if mask is None:
        mask = im
    args = np.argsort(dt.flatten())
    inds = np.vstack(np.unravel_index(args, dt.shape)).T
    if im.ndim == 2:
        lt = _run2D_bf(im, dt, mask, inds, smooth)
    elif im.ndim == 3:
        lt = _run3D_bf(im, dt, mask, inds, smooth)
    return lt


@njit
def _run2D_bf(im, dt, mask, inds, smooth):
    im2 = np.zeros(im.shape, dtype=float)
    # if im2.ndim == 2:
    for idx in inds:
        i = idx[0]
        j = idx[1]
        idx = np.array([[i, j]]).T
        r = dt[i, j]
        if mask[i, j]:
            im2 = _insert_disk_at_points(
                im=im2, coords=idx, r=int(r), v=r, overwrite=True, smooth=smooth)
    return im2


@njit
def _run3D_bf(im, dt, mask, inds, smooth):
    im3 = np.zeros(im.shape, dtype=float)
    for idx in inds:
        i = idx[0]
        j = idx[1]
        k = idx[2]
        idx = np.array([[i, j, k]]).T
        r = dt[i, j, k]
        if mask[i, j, k]:
            im3 = _insert_disk_at_points(
                im=im3, coords=idx, r=int(r), v=r, overwrite=True, smooth=smooth)
    return im3


def local_thickness_imj(im, dt=None, smooth=False, approx=False):
    r"""
    Insert a maximally inscribed sphere at every pixel labelled by sphere radius

    Parameters
    ----------
    im : ndarray
        Boolean image of the porous material
    dt : ndarray, optional
        The distance transform of the image
    smooth : bool, optional
        Indicates if protrusions should be removed from the faces of the spheres
        or not. Default is `True`.
    approx : bool, optional
        If `True` the algorithm is more aggressive at skipping voxels to process,
        which speeds things up, but this sacrifices accuracy in terms of a
        voxel-by-voxel match with the reference implementation. The default is
        `False`, meaning full accuracy is the default.

    Returns
    -------
    lt : ndarray
        The local thickness of the image with each voxel labelled according to the
        radius of the largest sphere which overlaps it

    Notes
    -----
    This version uses some logic to only insert spheres at locations which
    are not fully overlapped by larger spheres to reduce the number of insertions
    """
    if dt is None:
        dt = edt(im)

    # Sort dt to scan sites from largest to smallest
    args = np.argsort(dt.flatten())[-1::-1]
    ijk = np.vstack(np.unravel_index(args, dt.shape)).T

    # Call jitted function to draw spheres
    if im.ndim == 2:
        lt = _run2D(im, dt, ijk, smooth, approx)
    elif im.ndim == 3:
        lt = _run3D(im, dt, ijk, smooth, approx)

    return lt


@njit(parallel=True)
def _run2D(im, dt, ijk, smooth, approx):
    valid = np.copy(im)
    lt = np.zeros(im.shape, dtype=float)
    used = np.copy(lt)
    count = 0
    for idx in ijk:
        i = idx[0]
        j = idx[1]
        rval = dt[i, j]
        r = int(rval)
        # Since entries in ijk are sorted by size, once we reach an entry with
        # r = 0, then we know all remain entries will also be 0 so we can stop
        if r == 0:
            break
        # Only process if point has not yet been engulfed on previous step
        if valid[i, j]:
            used[i, j] = 1.0
            # Scan neighborhood around current pixel
            mn = r_to_inds_2d(r)
            for row in prange(len(mn[0])):
                m = mn[0][row] - r
                n = mn[1][row] - r
                if ((i + m) >= 0) and ((i + m) < im.shape[0]) \
                        and ((j + n) >= 0) and ((j + n) < im.shape[1]):
                    # Draw spheres within L of point (i, j)
                    L = r - ((m)**2 + (n)**2)**0.5 + 1
                    if (lt[i+m, j+n] == 0) and (L > 1 if smooth else L >= 1):
                        lt[i+m, j+n] = rval
                    # Use ints here since it's about actual sphere sizes
                    # not exact distances between pixel centers
                    if approx:
                        if int(dt[i+m, j+n]) <= int(L):
                            valid[i+m, j+n] = False
                    else:
                        if int(dt[i+m, j+n]) < int(L):
                            valid[i+m, j+n] = False
            count += 1
    return lt, count, used


@njit(parallel=True)
def _run3D(im, dt, ijk, smooth, approx):
    valid = np.copy(im)
    lt = np.zeros(im.shape, dtype=float)
    used = np.copy(lt)
    count = 0
    for idx in ijk:
        i = idx[0]
        j = idx[1]
        k = idx[2]
        rval = dt[i, j, k]
        r = int(rval)
        # Since entries in ijk are sorted by size, once we reach an entry with
        # r = 0, then we know all remain entries will also be 0 so we can stop
        if r == 0:
            break
        # Only process if point has not yet been engulfed on a previous step
        if valid[i, j, k]:
            used[i, j, k] = True
            # Scan neighborhood around current voxel
            mno = r_to_inds_3d(r)
            for row in prange(len(mno[0])):
                m = mno[0][row] - r
                n = mno[1][row] - r
                o = mno[2][row] - r
                if ((i + m) >= 0) and ((i + m) < im.shape[0]) \
                    and ((j + n) >= 0) and ((j + n) < im.shape[1]) \
                        and ((k + o) >= 0) and ((k + o) < im.shape[2]):
                    # Draw spheres within L of point (i, j, k)
                    L = r - (m**2 + n**2 + o**2)**0.5 + 1
                    if (lt[i+m, j+n, k+o] == 0) and \
                            (L > 1 if smooth else L >= 1):
                        lt[i+m, j+n, k+o] = rval
                    # Use ints here since it's about actual sphere
                    # sizes not exact distances between pixel centers
                    if approx:
                        if int(dt[i+m, j+n, k+o]) <= int(L):
                            valid[i+m, j+n, k+o] = False
                    else:
                        if int(dt[i+m, j+n, k+o]) < int(L):
                            valid[i+m, j+n, k+o] = False
            count += 1
    return lt, count, used


def r_to_inds(r, ndim):
    m = np.meshgrid(*[np.arange(2*r+1) for _ in range(ndim)])
    inds = np.vstack([n.flatten() for n in m]).T
    return inds


@njit
def r_to_inds_3d(r):
    size = 2*r + 1
    xx = np.empty(shape=(size**3), dtype=np.int_)
    yy = np.empty_like(xx)
    zz = np.empty_like(xx)
    for i in range(size):
        for j in range(size):
            for k in range(size):
                xx[i*size**2 + j*size + k] = i
                yy[i*size**2 + j*size + k] = j
                zz[i*size**2 + j*size + k] = k
    return xx, yy, zz


@njit
def r_to_inds_2d(r):
    size = 2*r + 1
    xx = np.empty(shape=(size**2), dtype=np.int_)
    yy = np.empty_like(xx)
    for i in range(size):
        for j in range(size):
            xx[i*size + j] = i
            yy[i*size + j] = j
    return xx, yy


def local_thickness_conv(
    im: npt.NDArray,
    dt: npt.NDArray = None,
    sizes: int = 25,
    smooth: bool = True,
):
    r"""
    Calculates the radius of the largest sphere that overlaps each voxel while
    fitting entirely within the void space.

    Parameters
    ----------
    im : ndarray
        A binary image with the phase of interest set to `True`
    dt : ndarray
        The distance transform of the void space. If not provided it will be computed
        but providing it saves time. Note that rounding and/or converting the values
        to integers and using `sizes=None` can save time by limiting the number of
        sizes that are used.
    sizes : array_like or scalar
        The sizes to insert. If a list of values is provided they are
        used directly. If a scalar is provided then that number of points
        spanning the min and max of the distance transform are used. If `None`, the
        all the unique values in the distance transform are used, which may become
        time consuming.
    smooth : bool, optional
        Indicates if protrusions should be removed from the faces of the spheres
        or not. Default is `True`.

    Returns
    -------
    image : ndarray
        A copy of `im` with the pore size values in each voxel

    Notes
    -----
    The way local thickness is found in PoreSpy differs from the
    traditional method (i.e. used in ImageJ
    `<https://imagej.net/Local_Thickness>`_).

    """
    from porespy.filters import fftmorphology

    im = np.squeeze(im)

    if dt is None:
        dt = edt(im > 0)

    if sizes is None:
        sizes = np.unique(dt[im])
    elif isinstance(sizes, int):
        sizes = np.logspace(start=np.log10(np.amax(dt)), stop=0, num=sizes)
    else:
        sizes = np.unique(sizes)[-1::-1]

    imresults = np.zeros(np.shape(im))
    desc = inspect.currentframe().f_code.co_name  # Get current func name
    for r in tqdm(sizes, desc=desc, **settings.tqdm):
        imtemp = dt >= r
        if np.any(imtemp):
            se = ps_round(r, ndim=im.ndim, smooth=smooth)
            imtemp = fftmorphology(imtemp, se, mode="dilation")
            imresults[(imresults == 0) * imtemp] = r

    return imresults


def local_thickness_dt(
    im: npt.NDArray,
    dt: npt.NDArray = None,
    sizes: int = 25,
    smooth: bool = True,
    return_indices: bool = False,
):
    r"""
    Calculates the radius of the largest sphere that overlaps each voxel while
    fitting entirely within the void space.

    Parameters
    ----------
    im : ndarray
        A binary image with the phase of interest set to `True`
    dt : ndarray
        The distance transform of the void space. If not provided it will be computed
        but providing it saves time. Note that rounding and/or converting the values
        to integers and using `sizes=None` can save time by limiting the number of
        sizes that are used.
    sizes : array_like or scalar
        The sizes to insert. If a list of values is provided they are
        used directly. If a scalar is provided then that number of points
        spanning the min and max of the distance transform are used. If `None`, then
        all the unique values in the distance transform are used, which may become
        time consuming.
    smooth : bool, optional
        Indicates if protrusions should be removed from the faces of the spheres
        or not. Default is `True`.
    return_indices : bool, optional
        If `True`, return `(sizes, indices)` where `sizes` is the 1D array of
        applied sizes (with a leading `0` for unfilled voxels) and `indices` is
        a small unsigned-int image such that `sizes[indices]` reproduces the
        usual float result. This avoids allocating a full float64 image, which
        matters for large tomograms. Default is `False`.

    Returns
    -------
    image : ndarray
        A copy of `im` with the pore size values in each voxel. If
        `return_indices` is `True`, returns `(sizes, indices)` instead.

    """
    im = np.squeeze(im)

    if dt is None:
        dt = edt(im > 0)

    # Parse given sizes
    if sizes is None:
        sizes = np.unique(dt[im])
    elif isinstance(sizes, int):
        sizes = np.logspace(start=np.log10(np.amax(dt)), stop=0, num=sizes)
    else:
        sizes = np.unique(sizes)[-1::-1]

    if return_indices:
        # Store i+1 in a compact unsigned-int image; 0 marks unfilled voxels.
        im_results = np.zeros(np.shape(im), dtype=_index_dtype(len(sizes)))
    else:
        im_results = np.zeros(np.shape(im))
    desc = inspect.currentframe().f_code.co_name  # Get current func name
    for i, r in enumerate(tqdm(sizes, desc=desc, **settings.tqdm)):
        im_temp = dt >= r  # Perform erosion
        if np.any(im_temp):
            # Perform dilation
            im_temp = edt(~im_temp) < r if smooth else edt(~im_temp) <= r
            # Add values to im_results
            im_results[(im_results == 0) * im_temp] = (i + 1) if return_indices else r

    if return_indices:
        # Prepend 0 so that `sizes[indices]` recovers the float result.
        return np.concatenate(([0.0], sizes)), im_results
    return im_results


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from localthickness import local_thickness as loct

    import porespy as ps

    im = ~ps.generators.random_spheres([150, 150, 150], r=10, clearance=10, seed=0)
    dt = edt(im)
    ps.tools.tic()
    lt1, count, used = local_thickness_imj(im, dt=dt, smooth=True, approx=True)
    t1 = ps.tools.toc(quiet=True)
    ps.tools.tic()
    lt2, count, used = local_thickness_imj(im, dt=dt, smooth=True, approx=False)
    t2 = ps.tools.toc(quiet=True)
    ps.tools.tic()
    lt3 = local_thickness_dt(im, dt=dt, sizes=np.unique(dt[im].astype(int)))
    t3 = ps.tools.toc(quiet=True)
    ps.tools.tic()
    lt4 = loct(im)
    t4 = ps.tools.toc(quiet=True)
    print("Times are:")
    print(f" Reference: {t2}")
    print(f" New Method: {t1}")
    print(f" PoreSpy: {t3}")
    print(f" Dahl: {t4}")
    print("Errors are:")
    print(f" New Method: {np.sum(lt2 != lt1)/im.sum()}")
    print(f" PoreSpy: {np.sum(lt2 != lt3)/im.sum()}")
    print(f" Dahl: {np.sum(lt2 != lt4)/im.sum()}")
    print(f"New method used {round(count/im.sum()*100, 2)}% of pixels")

    if im.ndim == 2:
        fig, ax = plt.subplots(1, 4)
        # ax[0].imshow(lt2 / im)
        ax[0].set_title('Reference')
        ax[0].axis('off')
        ax[1].imshow(lt1 / im)
        ax[1].set_title('New Method')
        ax[1].axis('off')
        ax[2].imshow(lt3 / im)
        ax[2].set_title('PoreSpy')
        ax[2].axis('off')
        ax[3].imshow(lt4 / im)
        ax[3].set_title('Dahl')
        ax[3].axis('off')
