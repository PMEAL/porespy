import numpy as np
import numpy.typing as npt

from porespy.tools import get_edt

__all__ = ['capillary_transform']


edt = get_edt()


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
        The contact angle of the fluid-fluid-solid system, in degrees, measured through
        the non-wetting phase.  It must be >90.
    g : scalar (default = 0)
        The gravitational constant acting on the fluids. Gravity is assumed to act
        toward the x=0 axis. To have gravity act in different directions use
        `np.swapaxes(im, 0, ax)` where `ax` is the desired direction.
    rho_nwp : scalar
        The density of the non-wetting fluid
    rho_wp : scalar
        The density of the wetting fluid
    voxel_size : scalar (default = 0.0)
        The resolution of the image
    spacing : scalar (default = None)
        If a 2D image is provided, this value is used to compute the second
        radii of curvature.  Setting it to `np.inf` will make the calculation truly
        2D since only one radii of curvature is considered. Setting it to `None`
        will force the calculation to be 3D.  If `im` is 3D this argument is
        ignored.  This should be in units of physical length, not voxels.

    Notes
    -----
    All physical properties should be in self-consistent units, and it is strongly
    recommended to use SI for everything.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/capillary_transform.html>`__
    to view online example.

    """
    from porespy.generators import ramp

    if np.any(theta < 90) or np.any(theta > 180):
        raise Exception('The contact angle must be between 90 and 180')
    delta_rho = rho_nwp - rho_wp
    if dt is None:
        dt = edt(im)
    if (im.ndim == 2) and (spacing is not None):
        pc = -sigma*np.cos(np.deg2rad(theta))*(1/(dt*voxel_size) + 2/spacing)
    else:
        pc = -2*sigma*np.cos(np.deg2rad(theta))/(dt*voxel_size)
    h = ramp(im.shape, inlet=0, outlet=im.shape[0], axis=0)*voxel_size
    pc = pc + delta_rho*g*h
    return pc
