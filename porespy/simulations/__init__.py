r"""

Collection of functions for performing numerical simulations on images
######################################################################

This module contains routines for performing simulations directly on images.

.. currentmodule:: porespy

.. autosummary::
   :template: mybase.rst
   :toctree: generated/

    simulations.drainage
    simulations.tortuosity_fd
    simulations.rw
    simulations.calc_gas_props
    simulations.compute_steps
    simulations.steps_to_displacements
    simulations.effective_diffusivity_rw
    simulations.tortuosity_rw
    simulations.plot_deff
    simulations.plot_tau
    simulations.plot_msd
    simulations.rw_to_image

"""

from ._drainage import *
from ._dns import *
from ._ibip import ibip
from ._ibip_gpu import ibip_gpu
from .rw_simulation import *
from .rw_post import *
