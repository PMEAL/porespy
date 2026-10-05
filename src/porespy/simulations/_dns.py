import logging
import time

import numpy as np

from porespy.filters import trim_nonpercolating_paths
from porespy.generators import faces
from porespy.tools import Results

logger = logging.getLogger(__name__)


__all__ = ["tortuosity_fd"]


def tortuosity_fd(im, axis, solver=None, tol=None, ftol=1e-3):
    r"""
    Calculates the tortuosity of image in the specified direction.

    Parameters
    ----------
    im : ndarray
        The binary image to analyze with ``True`` indicating phase of interest
    axis : int
        The axis along which to apply boundary conditions
    solver : openpnm Solver, optional
        A pre-built OpenPNM solver. When given, a single solve is performed
        and ``ftol`` is used only as a post-hoc convergence check.
    tol : float, optional
        Residual relative tolerance passed to the default PyAMG solver. If
        given, a single solve at this tolerance is performed (no iterative
        tightening) and ``ftol`` is used only as a post-hoc check. Ignored
        when ``solver`` is given.
    ftol : float, optional
        Target relative inlet/outlet flux mismatch. When neither ``solver``
        nor ``tol`` is given, the residual tolerance is tightened iteratively
        until the achieved flux balance falls under ``ftol``. Default is
        ``1e-3``.

    Returns
    -------
    results : Results object
        The following values are computed and returned as attributes:

        =================== ===================================================
        Attribute           Description
        =================== ===================================================
        tortuosity          Calculated using the ``effective_porosity`` as
                            :math:`\tau = \frac{D_{AB}}{D_{eff}} \cdot
                            \varepsilon`.
        effective_porosity  Porosity of the image after applying
                            ``trim_nonpercolating_paths``.  This removes
                            disconnected voxels which cause singular matrices.
        original_porosity   Porosity of the as-received the image
        formation_factor    found as :math:`D_{AB}/D_{eff}`.
        im_conc             An image containing the concentration values from
                            the simulation.
        converged           Whether the achieved inlet/outlet flux mismatch
                            falls under ``ftol``. Users who need a finer
                            picture (e.g. layer-by-layer flux constancy) can
                            run ``porespy.simulations.flux`` on ``im_conc``.
        =================== ===================================================

    Examples
    --------
    `Click here
    <https://porespy.org/examples/simulations/reference/tortuosity_fd.html>`__
    to view online example.

    """
    import openpnm as op
    import pyamg

    from ._dns_tools import flux

    ws = op.Workspace()

    if axis > (im.ndim - 1):
        raise Exception(f"'axis' must be <= {im.ndim}")
    openpnm_v3 = op.__version__.startswith("3")

    # Obtain original porosity
    eps0 = im.sum(dtype=np.int64) / im.size

    # Remove floating pores
    inlets = faces(im.shape, inlet=axis)
    outlets = faces(im.shape, outlet=axis)
    im = trim_nonpercolating_paths(im, inlets=inlets, outlets=outlets)
    # Check if porosity is changed after trimming floating pores
    eps = im.sum(dtype=np.int64) / im.size
    if not eps:
        raise Exception("No pores remain after trimming floating pores")
    if eps < eps0:  # pragma: no cover
        logger.warning("Found non-percolating regions, were filled to percolate")

    # Generate a Cubic network to be used as an orthogonal grid
    net = op.network.CubicTemplate(template=im, spacing=1.0)
    if openpnm_v3:
        phase = op.phase.Phase(network=net)
    else:
        phase = op.phases.GenericPhase(network=net)
    phase["throat.diffusive_conductance"] = 1.0
    # Run Fickian Diffusion on the image
    fd = op.algorithms.FickianDiffusion(network=net, phase=phase)
    # Choose axis of concentration gradient
    inlets = net.coords[:, axis] <= 1
    outlets = net.coords[:, axis] >= im.shape[axis] - 1
    # Boundary conditions on concentration
    cL, cR = 1.0, 0.0
    fd.set_value_BC(pores=inlets, values=cL)
    fd.set_value_BC(pores=outlets, values=cR)

    template_indices = net["pore.template_indices"]
    normal_axes = tuple(i for i in range(im.ndim) if i != axis)

    def flux_mismatch(x):
        # Inlet/outlet relative flux mismatch from the layer-summed flux of
        # the porespy `flux` helper. Matches what a user would compute
        # themselves on `result.im_conc`.
        conc = np.zeros(im.size, dtype=float)
        conc[template_indices] = x
        c = conc.reshape(im.shape)
        rate = flux(c, axis=axis, k=im).sum(axis=normal_axes)
        return abs(rate[0] - rate[-1]) / max(abs(rate[0]), abs(rate[-1]))

    t = time.perf_counter_ns()
    if openpnm_v3:
        fd._update_A_and_b()
        A = fd.A.tocsr()
        b = fd.b
        if solver is None and tol is None:
            # Build the multigrid hierarchy once and reuse it as we tighten
            # the residual tolerance. The calibrated starting tol of
            # `ftol*0.01` clears `ftol` in 2-3 iterations on typical inputs.
            ml = pyamg.ruge_stuben_solver(A)
            cur_tol = ftol * 0.01
            x0 = None
            converged = False
            mismatch = np.inf
            for _ in range(5):
                fd.x, info = ml.solve(b, x0=x0, tol=cur_tol, return_info=True)
                if info:
                    raise Exception(f"Solver failed to converge, exit code: {info}")
                x0 = fd.x
                mismatch = flux_mismatch(fd.x)
                if mismatch <= ftol:
                    converged = True
                    break
                cur_tol *= 0.1
            if not converged:  # pragma: no cover
                logger.warning(
                    f"Could not meet ftol={ftol:.1e}; final flux "
                    f"mismatch {mismatch:.2e}"
                )
        else:
            if solver is None:
                solver = op.solvers.PyamgRugeStubenSolver(tol=tol)
            fd.x, info = solver.solve(A, b)
            if info:
                raise Exception(f"Solver failed to converge, exit code: {info}")
            converged = flux_mismatch(fd.x) <= ftol
            if not converged:  # pragma: no cover
                logger.warning(
                    f"Inlet/outlet flux mismatch {flux_mismatch(fd.x):.2e} "
                    f"exceeds ftol={ftol:.1e}"
                )
    else:
        fd.settings.update({"solver_family": "scipy", "solver_type": "cg"})
        fd.run()
        converged = True
    t = time.perf_counter_ns() - t

    # Calculate molar flow rate, effective diffusivity and tortuosity
    r_in = fd.rate(pores=inlets)[0]
    dC = cL - cR
    L = im.shape[axis]
    A_geom = np.prod(im.shape) / L
    # L-1 because BCs are put inside the domain, see issue #495
    Deff = r_in * (L - 1) / A_geom / dC
    tau = eps / Deff

    # Attach useful parameters to Results object
    result = Results()
    result.im = im
    result.tortuosity = tau
    result.formation_factor = 1 / Deff
    result.original_porosity = eps0
    result.effective_porosity = eps
    conc = np.zeros(im.size, dtype=float)
    conc[template_indices] = fd["pore.concentration"]
    result.im_conc = conc.reshape(im.shape)
    result.time = t/1e9
    result.converged = bool(converged)

    # Free memory
    ws.close_project(net.project)

    return result
