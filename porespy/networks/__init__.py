r"""

===============================================================================
networks
===============================================================================

**Obtain Network Representations**

Contains functions for analysing images as pore networks

.. autosummary::

    porespy.networks.add_boundary_regions
    porespy.networks.snow
    porespy.networks.snow_dual
    porespy.networks.regions_to_network
    porespy.networks.map_to_regions
    porespy.networks.generate_voxel_image

.. autofunction:: snow
.. autofunction:: snow_dual
.. autofunction:: regions_to_network
.. autofunction:: add_boundary_regions
.. autofunction:: map_to_regions
.. autofunction:: generate_voxel_image

"""

from .__funcs__ import add_boundary_regions
from .__funcs__ import map_to_regions
from .__funcs__ import generate_voxel_image
from .__getnet__ import regions_to_network
from .__snow__ import snow
from .__snow_dual__ import snow_dual
