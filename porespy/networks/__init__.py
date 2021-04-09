r"""

Networks
########

**Obtain Network Representations**

Contains functions for analysing images as pore networks.

.. currentmodule:: porespy

.. autosummary::
   :template: mybase.rst
   :toctree: generated/

   networks.snow
   networks.snow_dual
   networks.map_to_regions
   networks.regions_to_network
   networks.add_boundary_regions
   networks.generate_voxel_image
   networks.maximal_ball

"""

__all__ = [
    "add_boundary_regions",
    "map_to_regions",
    "generate_voxel_image",
    "label_boundary_cells",
    "add_phase_interconnections",
    "regions_to_network",
    "snow",
    "snow_dual",
    "snow_n",
    "maximal_ball"]

from .__funcs__ import add_boundary_regions
from .__funcs__ import map_to_regions
from .__funcs__ import generate_voxel_image
from .__funcs__ import label_phases
from .__funcs__ import label_boundaries
from .__getnet__ import regions_to_network
from .__utils__ import _net_dict
from ._snow2 import snow2
from .__maximal_ball__ import maximal_ball
