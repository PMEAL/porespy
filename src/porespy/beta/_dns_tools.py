import numpy as np
from scipy.ndimage import convolve1d

from porespy.filters import trim_nonpercolating_paths
from porespy.generators import faces

__all__ = ["flux", "tau_from_cmap"]


def flux(c, axis, k=None):
    """
    Computes the layer-by-layer diffusive flux in a given direction.

    Parameters
    ----------
    c : ndarray
        The concentration field
    axis : int
        The axis along which the flux is computed
    k : ndarray
        The conductivity field

    Returns
    -------
    J : ndarray
        The layer-by-layer flux in the given direction

    """
    k = np.ones_like(c) if k is None else np.array(k)
    # Compute the gradient of the concentration field using forward diff
    dcdX = convolve1d(c, weights=np.array([-1.0, 1.0]), axis=axis)
    # dcdX @ outlet is incorrect due to forward diff -> use backward
    _fix_gradient_outlet(dcdX, axis)
    # Compute the conductivity at the faces using resistors in series
    k_face = 1 / convolve1d(1 / k, weights=np.array([0.5, 0.5]), axis=axis)
    # Normalize gradient by the conductivity to get the physical flux
    J = dcdX * k_face
    return J


def tau_from_cmap(c, im, axis):
    """
    Computes the tortuosity factor from a concentration field.

    Parameters
    ----------
    c : ndarray
        The concentration field
    im : ndarray
        The binary image of the porous medium
    axis : int
        The axis along which tortuosity is computed

    Returns
    -------
    tau : float
        The tortuosity factor along the given axis

    """
    im = _trim_nonpercolating_paths(im, axis=axis)
    # Use the image as conductivity matrix (solid = 0, fluid = 1)
    k = im.astype(c.dtype)
    # Find transport length and cross-sectional area
    L = im.shape[axis]
    A = np.prod(im.shape) / L
    # Find the average inlet and outlet concentration
    cA, cB = _get_BC_values(c, im, axis)
    # Compute the point-wise flux in the given direction
    J = flux(c, axis=axis, k=k)
    # Calculate the net flux for each layer in the given direction
    normal_axes = tuple(i for i in range(im.ndim) if i != axis)
    rate = J.sum(axis=normal_axes)
    # NOTE: L-1 because c is stored at cell centers
    Deff = rate.mean() * (L-1) / A / (cA-cB)
    eps = im.sum(dtype=np.int64) / im.size
    return eps / Deff


def _fix_gradient_outlet(J, axis):
    """Replaces the gradient @ outlet with that of the penultimate layer."""
    J_outlet = _slice_view(J, -1, axis)
    J_penultimate_layer = _slice_view(J, -2, axis)
    J_outlet[:] = J_penultimate_layer


def _slice_view(a, idx, axis):
    """Returns a slice view of the array along the given axis."""
    # Example: _slice_view(a, i=5, axis=1) -> a[:, 5, :]
    sl = [slice(None)] * a.ndim
    sl[axis] = idx
    return a[tuple(sl)]


def _get_BC_values(c, im, axis):
    """Returns the inlet and outlet concentration values."""
    cA = c.take(0, axis=axis)   # c @ inlet
    cB = c.take(-1, axis=axis)  # c @ outlet
    mask_inlet = im.take(0, axis=axis)
    mask_outlet = im.take(-1, axis=axis)
    cA = cA[mask_inlet].mean()
    cB = cB[mask_outlet].mean()
    return cA, cB


def _trim_nonpercolating_paths(im, axis):
    """Removes non-percolating paths from the image."""
    inlets = faces(im.shape, inlet=axis)
    outlets = faces(im.shape, outlet=axis)
    im = trim_nonpercolating_paths(im, inlets=inlets, outlets=outlets)
    return im
