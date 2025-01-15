import numpy as np
import numpy.typing as npt
from porespy.generators import ramp
try:
    from pyedt import edt
except ModuleNotFoundError:
    from edt import edt


__all__ = [
    'capillary_transform',
]


def capillary_transform(
    im: npt.NDArray,
    dt: npt.NDArray = None,
    sigma: float = 1.0,
    theta: float = 180,
    g: float = 0.0,
    rho_wp: float = 0.0,
    rho_nwp: float = 0.0,
    voxel_size: float = 1.0,
    spacing: float = None,
):
    r"""
    Uses the Washburn equation to convert distance transform values to a capillary
    transform.

    Parameters
    ----------
    im : ndarray
        A boolean image describing the porous medium with ``True`` values indicating
        the phase of interest.
    dt : ndarray, optional
        The distance transform of the void phase. If not provided it will be
        calculated, so some time can be save if a pre-computed array is already
        available.
    sigma : scalar (default = 1.0)
        The surface tension of the fluid-fluid interface.
    theta : scalar (default = 180)
        The contact angle of the fluid-fluid-solid system, in degrees.
    g : scalar (default = 0)
        The gravitational constant acting on the fluids. Gravity is assumed to act
        toward the x=0 axis.  To have gravity act in different directions just
        use `np.swapaxes(im, 0, ax)` where `ax` is the desired direction.
    delta_rho : scalar (default = 0.0)
        The density difference between the fluids.
    voxel_size : scalar (default = 0.0)
        The resolution of the image
    spacing : scalar (default = None)
        If a 2D image is provided, this value is used to compute the second
        radii of curvature.  Setting it to `inf` will make the calculation truly
        2D since only one radii of curvature is considered. Setting it to `None`
        will force the calculation to be 3D.  If `im` is 3D this argument is
        ignored.

    Notes
    -----
    All physical properties should be in self-consistent units, and it is strongly
    recommended to use SI for everything.

    """
    delta_rho = rho_nwp - rho_wp
    if dt is None:
        dt = edt(im)
    if (im.ndim == 2) and (spacing is not None):
        pc = -sigma*np.cos(np.deg2rad(theta))*(1/(dt*voxel_size) + 2/spacing)
    else:
        pc = -2*sigma*np.cos(np.deg2rad(theta))/(dt*voxel_size)
    if delta_rho > 0:
        h = ramp(im.shape, inlet=0, outlet=im.shape[0], axis=0)*voxel_size
        pc = pc + delta_rho*g*h
    elif delta_rho < 0:
        h = ramp(im.shape, inlet=im.shape[0], outlet=0, axis=0)*voxel_size
        pc = pc + delta_rho*g*h
    return pc
