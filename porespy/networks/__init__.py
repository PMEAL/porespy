r"""

===============================================================================
networks
===============================================================================

**Obtain Network Representations**

Contains functions for analysing images as pore networks

.. autosummary::

    porespy.networks.add_boundary_regions
    porespy.networks.generate_voxel_image
    porespy.networks.map_to_regions
    porespy.networks.maximal_ball
    porespy.networks.regions_to_network
    porespy.networks.snow2

.. autofunction:: add_boundary_regions
.. autofunction:: generate_voxel_image
.. autofunction:: map_to_regions
.. autofunction:: regions_to_network
.. autofunction:: snow2
.. autofunction:: maximal_ball


"""

from .__funcs__ import add_boundary_regions
from .__funcs__ import map_to_regions
from .__funcs__ import generate_voxel_image
from .__funcs__ import label_phases
from .__funcs__ import label_boundaries
from .__funcs__ import _parse_pad_width
from .__getnet__ import regions_to_network
from .__utils__ import _net_dict
from ._snow2 import snow2
from .__maximal_ball__ import maximal_ball
