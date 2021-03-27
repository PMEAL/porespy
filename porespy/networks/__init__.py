r"""

Networks
########

**Obtain Network Representations**

Contains functions for analysing images as pore networks.

.. autosummary::
   :toctree: generated/

   snow
   snow_dual
   map_to_regions
   regions_to_network
   add_boundary_regions
   generate_voxel_image
   maximal_ball

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
from .__funcs__ import label_boundary_cells
from .__funcs__ import add_phase_interconnections
from .__getnet__ import regions_to_network
from .__utils__ import _net_dict
from .__snow__ import snow
from .__snow_dual__ import snow_dual
from .__snow_n__ import snow_n
from .__maximal_ball__ import maximal_ball


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
