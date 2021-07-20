import numpy as np
import openpnm as op
from porespy.filters import trim_nonpercolating_paths
from porespy.tools import Results
from loguru import logger
from porespy.generators import faces


def tortuosity(im, axis):
    r"""
    Calculate the tortuosity of image in the specified direction

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
    # removing floating pores
    IN = faces(im.shape, inlet=axis)
    OUT = faces(im.shape, outlet=axis)
    im = trim_nonpercolating_paths(im, inlets=IN, outlets=OUT)
    # porosity is changed because of trimmimg floating pores
    eps = im.sum() / im.size
    if eps < eps0:  # pragma: no cover
        logger.warning(f'True porosity is {eps:.2f}, filled {eps0 - eps:.5f}'
                       ' volume fraction of the image for it to percolate.')
    # cubic network generation
    net = op.network.CubicTemplate(template=im, spacing=1)
    # Adding phase
    water = op.phases.Water(network=net)
    water['throat.diffusive_conductance'] = 1  # dummy value
    # Running Fickian Diffusion
    fd = op.algorithms.FickianDiffusion(network=net, phase=water)
    # Choosing axis of concentration gradient
    inlets = net['pore.coords'][:, axis] <= 1
    outlets = net['pore.coords'][:, axis] >= im.shape[axis]-1
    # Boundary conditions on concentration
    C_in = 1.0
    C_out = 0.0
    fd.set_value_BC(pores=inlets, values=C_in)
    fd.set_value_BC(pores=outlets, values=C_out)
    fd.settings['solver_family'] = 'scipy'
    fd.settings['solver_type'] = 'cg'
    fd.run()
    # Calculating molar flow rate, effective diffusivity and tortuosity
    rate_out = fd.rate(pores=outlets)[0]
    rate_in = fd.rate(pores=inlets)[0]
    if not np.allclose(-rate_out, rate_in):  # pragma: no cover
        raise Exception('Something went wrong, inlet and outlet rate do not match')
    delta_C = C_in - C_out
    L = im.shape[axis]
    A = np.prod(im.shape) / L
    N_A = A / (L-1) * delta_C  # -1 because BCs are put inside domain, see #495
    Deff = rate_in / N_A
    tau = eps / Deff
    result = Results()
    result.tortuosity = tau
    result.formation_factor = 1 / Deff
    result.original_porosity = eps0
    result.effective_porosity = eps
    conc = np.zeros([im.size, ], dtype=float)
    conc[net['pore.template_indices']] = fd['pore.concentration']
    conc = np.reshape(conc, newshape=im.shape)
    result.concentration = conc
    return result
