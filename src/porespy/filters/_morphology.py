import numpy as np
import numpy.typing as npt
from scipy.signal import fftconvolve

from porespy.tools import (
    _find_sphere_interface,
    _get_sphere_indices,
    _insert_disks_at_indices_parallel,
    _make_axial_extent_lookup,
    get_edt,
    ps_round,
)

__all__ = [
    'erode',
    'dilate',
]


edt = get_edt()


def erode(
    im: npt.NDArray,
    r: int,
    dt: npt.NDArray = None,
    method: str = 'dt',
    smooth: bool = True,
):
    r"""
    Perform erosion with a round structuring element

    Parameters
    ----------
    im : ndarray
        A boolean image with the foreground (to be eroded) indicated by `True`
    r : int
        The radius of the round structuring element to use
    dt : ndarray
        The distance transform of the foreground. If not provided it will be
        computed. This argument is only relevant if `method='dt'`.
    smooth : boolean
        If `True` (default) the single voxel protrusion on the face of the
        structuring element are removed.
    method : str
        Controls which method is used. Options are:

        ================= =============================================================
        method            Description
        ================= =============================================================
        `'dt'`            Uses a distance transform to find all voxels within `r` of
                          the background, then removes them to affect an erosion
        `'conv'`          Uses a FFT based convolution to find all voxels within `r`
                          of the background (voxels with a value smaller than the sum
                          of the structuring element), then removes them to affect an
                          erosion.
        `'brute_force'`   Draws spheres at the phase interface using optimized
                          scan-line insertion. Also accepts 'bf'. Requires a
                          nonnegative integer radius and a 2D or 3D image.
        ================= =============================================================

    Returns
    -------
    erosion : ndarray
        An image the same size as `im` with the foreground eroded by the specified
        amount.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/erode.html>`_
    to view online example.
    """
    if method == 'dt':
        if dt is None:
            dt = edt(im)
        if dt.dtype == int:
            tmp = ~(dt <= r)
        else:
            tmp = dt >= r if smooth else dt > r
        ero = tmp * im
    elif method in ('brute_force', 'bf'):
        ero = ~_dilate_spheres(~(im == 1), r, smooth)
    elif method.startswith('conv'):
        se = ps_round(r=r, ndim=im.ndim, smooth=smooth)
        ero = ~(fftconvolve(~im, se, mode='same') > 0.1)
    return ero


def dilate(
    im: npt.NDArray,
    r: int,
    dt: npt.NDArray = None,
    method: str = 'dt',
    smooth: bool = True,
):
    r"""
    Perform dilation with a round structuring element

    Parameters
    ----------
    im : ndarray
        A boolean image with the foreground (to be dilated) indicated by `True`
    r : int
        The radius of the round structuring element to use
    dt : ndarray
        The distance transform of the foreground. If not provided it will be
        computed. This argument is only relevant if `method='dt'`.
    smooth : boolean
        If `True` (default) the single voxel protrusion on the face of the
        structuring element are removed.
    method : str
        Controls which method is used. Options are:

        ================= =============================================================
        method            Description
        ================= =============================================================
        `'dt'`            Uses a distance transform to find all voxels within `r` of
                          the foreground, then adds them to affect a dilation
        `'conv'`          Using a FFT based convolution to find all voxels within `r`
                          of the foreground (voxels with a value larger than 0), then adds
                          them to affect a dilation.
        `'brute_force'`   Draws spheres at the phase interface using optimized
                          scan-line insertion. Also accepts 'bf'. Requires a
                          nonnegative integer radius and a 2D or 3D image.
        ================= =============================================================

    Returns
    -------
    dilation : ndarray
        An image the same size as `im` with the foreground eroded by the specified
        amount.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/dilate.html>`_
    to view online example.
    """
    im = im == 1
    if method == 'dt':
        if dt is None:
            dt = edt(~im)
        dil = dt < r if smooth else dt <= r
        dil += im
    elif method in ('brute_force', 'bf'):
        dil = _dilate_spheres(im, r, smooth)
    elif method.startswith('conv'):
        se = ps_round(r=r, ndim=im.ndim, smooth=smooth)
        dil = fftconvolve(im, se, mode='same') > 0.1
    return dil


def _dilate_spheres(im, r, smooth):
    """Dilate a binary phase using the shared interface and sphere renderer."""
    if im.ndim not in (2, 3):
        raise ValueError('brute_force requires a 2D or 3D image')
    if not np.isscalar(r) or not np.isfinite(r) or r < 0 or r != int(r):
        raise ValueError('brute_force requires a nonnegative integer radius')
    r = int(r)
    result = np.array(im, dtype=bool, order='C', copy=True)
    if r == 0:
        return result
    interface = _find_sphere_interface(result, np.empty_like(result))
    indices = _get_sphere_indices(interface)
    if indices.size:
        _insert_disks_at_indices_parallel(
            im=result,
            indices=indices,
            dt=result,
            ceil_distance=_make_axial_extent_lookup(r),
            smooth=smooth,
            overwrite=True,
            fixed_radius=r,
        )
    return result


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import porespy as ps

    edt = ps.tools.get_edt()

    im = ps.generators.blobs([200, 200], porosity=0.6, seed=5)
    dt = edt(im)

    ero_smooth = erode(im=im, r=5, method='conv', smooth=True).astype(int)
    ero_smooth[~im] = -1

    ero1 = erode(im=im, r=5, dt=dt, method='dt', smooth=True).astype(int)
    ero1[~im] = -1

    ero_not_smooth = erode(im=im, r=5, method='conv', smooth=False).astype(int)
    ero_not_smooth[~im] = -1

    ero3 = erode(im=im, r=5, dt=dt, method='dt', smooth=False).astype(int)
    ero3[~im] = -1

    # fig, ax = plt.subplots(1, 2)
    # ax[0].imshow(im)
    # ax[1].imshow(dt)

    fig, ax = plt.subplots(2, 2)
    ax[0][0].imshow(ero_smooth)
    ax[0][0].set_title('Conv, Smooth')
    ax[0][1].imshow(ero1)
    ax[0][1].set_title('DT, float')
    ax[1][0].imshow(ero_not_smooth)
    ax[1][0].set_title('Conv, Not Smooth')
    ax[1][1].imshow(ero3)
    ax[1][1].set_title('DT, float')


    assert np.all(ero_smooth == ero1)
    assert np.all(ero_not_smooth == ero3)
