r"""

===============================================================================
network extraction
===============================================================================

**Obtain Network Representations**

Contains functions for extracting pore networks from images.

.. autosummary::

    porespy.network_extraction.snow
    porespy.network_extraction.snow_dual
    porespy.network_extraction.regions_to_network
    porespy.network_extraction.map_to_regions
    porespy.network_extraction.generate_voxel_image

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
