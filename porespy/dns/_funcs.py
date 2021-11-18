import numpy as np
import openpnm as op
from auto_all import start_all, end_all
from porespy.filters import trim_nonpercolating_paths
from porespy.tools import Results
from loguru import logger
from porespy.generators import faces

ws = op.Workspace()
start_all()  # All functions below here, and above end_all() will be imported


def tortuosity(im, axis):
    r"""
    Calculates the tortuosity of image in the specified direction.

    Parameters
    ----------
    im : ndarray
        The binary image to analyze with ``True`` indicating phase of interest
    axis : int
        The axis along which to apply boundary conditions

    Returns
    -------
    results : Results objects
        The following values are computed and returned as attributes:

        - tortuosity
            Calculated using the ``effective_porosity`` as
            :math:`\tau = \frac{D_{AB}}{D_{eff}} \cdot \varepsilon`.

        - effective_porosity
            Porosity of the image after applying ``trim_nonpercolating_paths``.
            This removes disconnected voxels which cause singular matrices.

        - original_porosity
            Porosity of the as-received the image

        - formation_factor
            Found as :math:`D_{AB}/D_{eff}`.

        - concentration
            An image containing the concentration values from the
            simulation.

    """
    if axis > (im.ndim - 1):
        raise Exception(f"'axis' must be <= {im.ndim}")

    # Obtain original porosity
    eps0 = im.sum() / im.size
    # Remove floating pores
    inlets = faces(im.shape, inlet=axis)
    outlets = faces(im.shape, outlet=axis)
    im = trim_nonpercolating_paths(im, inlets=inlets, outlets=outlets)
    # Check if prosity is changed after trimmimg floating pores
    eps = im.sum() / im.size
    if eps < eps0:  # pragma: no cover
        logger.warning(f'True porosity is {eps:.2f}, filled {eps0 - eps:.5f}'
                       ' volume fraction of the image for it to percolate.')

    # Generate a Cubic network to be used as an orthogonal grid
    net = op.network.CubicTemplate(template=im, spacing=1.0)
    # Create a dummy phase
    phase = op.phases.GenericPhase(network=net)
    phase['throat.diffusive_conductance'] = 1.0
    # Run Fickian Diffusion on the image
    fd = op.algorithms.FickianDiffusion(network=net, phase=phase)
    # Choose axis of concentration gradient
    inlets = net.coords[:, axis] <= 1
    outlets = net.coords[:, axis] >= im.shape[axis] - 1
    # Boundary conditions on concentration
    cL, cR = 1.0, 0.0
    fd.set_value_BC(pores=inlets, values=cL)
    fd.set_value_BC(pores=outlets, values=cR)
    fd.settings.update({'solver_family': 'scipy', 'solver_type': 'cg'})
    fd.run()
    # Calculate molar flow rate, effective diffusivity and tortuosity
    rate_in = fd.rate(pores=inlets)[0]
    rate_out = fd.rate(pores=outlets)[0]
    if not np.allclose(-rate_out, rate_in):  # pragma: no cover
        raise Exception('Something went wrong, inlet and outlet rates do not match!')
    dC = cL - cR
    L = im.shape[axis]
    A = np.prod(im.shape) / L
    # L-1 because BCs are put inside the domain, see issue #495
    Deff = rate_in * (L-1)/A / dC
    tau = eps / Deff

    result = Results()
    result.tortuosity = tau
    result.formation_factor = 1 / Deff
    result.original_porosity = eps0
    result.effective_porosity = eps
    conc = np.zeros(im.size, dtype=float)
    conc[net['pore.template_indices']] = fd['pore.concentration']
    result.concentration = conc.reshape(im.shape)

    # Free memory
    ws.close_project(net.project)

    return result


end_all()  # All functions aove here, and below start_all() will be imported
