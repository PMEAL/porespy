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

from ._funcs import add_boundary_regions
from ._funcs import map_to_regions
from ._funcs import generate_voxel_image
from ._funcs import label_phases
from ._funcs import label_boundaries
from ._getnet import regions_to_network
from ._utils import _net_dict
from ._snow2 import snow2
from ._snow2 import _parse_pad_width
from ._maximal_ball import maximal_ball
