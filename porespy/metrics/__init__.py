r"""

===============================================================================
metrics
===============================================================================

**Extract Quantitative Information**

This submodule contains functions for determining key metrics about an image.
Typically these are applied to an image after applying a filter, but a few
functions can be applied directly to the binary image.

.. autosummary::

    porespy.metrics.chord_counts
    porespy.metrics.chord_length_distribution
    porespy.metrics.linear_density
    porespy.metrics.mesh_surface_area
    porespy.metrics.phase_fraction
    porespy.metrics.pore_size_distribution
    porespy.metrics.porosity
    porespy.metrics.porosity_profile
    porespy.metrics.props_to_image
    porespy.metrics.props_to_DataFrame
    porespy.metrics.radial_density
    porespy.metrics.region_interface_areas
    porespy.metrics.region_surface_areas
    porespy.metrics.regionprops_3D
    porespy.metrics.representative_elementary_volume
    porespy.metrics.two_point_correlation_bf
    porespy.metrics.two_point_correlation_fft

.. autofunction:: chord_counts
.. autofunction:: chord_length_distribution
.. autofunction:: linear_density
.. autofunction:: mesh_surface_area
.. autofunction:: phase_fraction
.. autofunction:: pore_size_distribution
.. autofunction:: porosity
.. autofunction:: porosity_profile
.. autofunction:: props_to_image
.. autofunction:: props_to_DataFrame
.. autofunction:: radial_density
.. autofunction:: region_interface_areas
.. autofunction:: region_surface_areas
.. autofunction:: regionprops_3D
.. autofunction:: representative_elementary_volume
.. autofunction:: two_point_correlation_bf
.. autofunction:: two_point_correlation_fft

"""

from .__regionprops__ import regionprops_3D
from .__regionprops__ import props_to_DataFrame
from .__regionprops__ import props_to_image
from .__funcs__ import chord_counts
from .__funcs__ import chord_length_distribution
from .__funcs__ import linear_density
from .__funcs__ import pore_size_distribution
from .__funcs__ import radial_density
from .__funcs__ import porosity
from .__funcs__ import porosity_profile
from .__funcs__ import representative_elementary_volume
from .__funcs__ import two_point_correlation_bf
from .__funcs__ import two_point_correlation_fft
from .__funcs__ import region_surface_areas
from .__funcs__ import region_interface_areas
from .__funcs__ import mesh_surface_area
from .__funcs__ import phase_fraction
