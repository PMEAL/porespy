r"""

===============================================================================
networks
===============================================================================

**Obtain Network Representations**

Contains functions for analysing images as pore networks

.. autosummary::

    porespy.networks.add_boundary_regions
    porespy.networks.add_boundary_regions2
    porespy.networks.snow
    porespy.networks.regions_to_network
    porespy.networks.map_to_regions
    porespy.networks.generate_voxel_image
    porespy.networks.maximal_ball

.. autofunction:: add_boundary_regions
.. autofunction:: add_boundary_regions2
.. autofunction:: connect_network_phases
.. autofunction:: generate_voxel_image
.. autofunction:: label_boundary_cells
.. autofunction:: map_to_regions
.. autofunction:: add_phase_interconnections
.. autofunction:: regions_to_network
.. autofunction:: snow
.. autofunction:: snow_n
.. autofunction:: maximal_ball


"""

from .__funcs__ import add_boundary_regions
from .__funcs__ import add_boundary_regions2
from .__funcs__ import map_to_regions
from .__funcs__ import generate_voxel_image
from .__funcs__ import label_boundary_cells
from .__funcs__ import add_phase_interconnections
from .__funcs__ import label_phases
from .__funcs__ import label_boundaries
from .__getnet__ import regions_to_network
from .__utils__ import _net_dict
from .__snow__ import snow
from .__snow_n__ import snow_n
from ._snow2 import snow2
from .__maximal_ball__ import maximal_ball
