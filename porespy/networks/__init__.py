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

from .__funcs__ import (add_boundary_regions, generate_voxel_image,
                        label_boundaries, label_phases, map_to_regions)
from .__getnet__ import regions_to_network
from .__maximal_ball__ import maximal_ball
from .__utils__ import _net_dict
from ._snow2 import _parse_pad_width, snow2

# .. autofunction:: add_boundary_regions
# .. autofunction:: connect_network_phases
# .. autofunction:: generate_voxel_image
# .. autofunction:: label_boundary_cells
# .. autofunction:: map_to_regions
# .. autofunction:: add_phase_interconnections
# .. autofunction:: regions_to_network
# .. autofunction:: snow
# .. autofunction:: snow_dual
# .. autofunction:: snow_n
# .. autofunction:: maximal_ball
