r"""

Metrics
#######

**Extract Quantitative Information**

This submodule contains functions for determining key metrics about an
image. Typically these are applied to an image after applying a filter,
but a few functions can be applied directly to the binary image.

.. autosummary::
   :toctree: generated/

    chord_counts
    chord_length_distribution
    lineal_path_distribution
    mesh_surface_area
    phase_fraction
    pore_size_distribution
    porosity
    porosity_profile
    prop_to_image
    props_to_DataFrame
    radial_density_distribution
    region_interface_areas
    region_surface_areas
    regionprops_3D
    representative_elementary_volume
    two_point_correlation_bf
    two_point_correlation_fft

"""

from .__regionprops__ import regionprops_3D
from .__regionprops__ import props_to_DataFrame
from .__regionprops__ import prop_to_image
from .__funcs__ import chord_counts
from .__funcs__ import chord_length_distribution
from .__funcs__ import lineal_path_distribution
from .__funcs__ import pore_size_distribution
from .__funcs__ import radial_density_distribution
from .__funcs__ import porosity
from .__funcs__ import porosity_profile
from .__funcs__ import representative_elementary_volume
from .__funcs__ import two_point_correlation_bf
from .__funcs__ import two_point_correlation_fft
from .__funcs__ import region_surface_areas
from .__funcs__ import region_interface_areas
from .__funcs__ import mesh_surface_area
from .__funcs__ import phase_fraction
