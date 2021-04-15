r"""

Metrics
#######

**Extract Quantitative Information**

This submodule contains functions for determining key metrics about an
image. Typically these are applied to an image after applying a filter,
but a few functions can be applied directly to the binary image.

.. currentmodule:: porespy

.. autosummary::
   :template: mybase.rst
   :toctree: generated/

    metrics.chord_counts
    metrics.chord_length_distribution
    metrics.lineal_path_distribution
    metrics.mesh_surface_area
    metrics.phase_fraction
    metrics.pore_size_distribution
    metrics.porosity
    metrics.porosity_profile
    metrics.prop_to_image
    metrics.props_to_DataFrame
    metrics.radial_density_distribution
    metrics.region_interface_areas
    metrics.region_surface_areas
    metrics.regionprops_3D
    metrics.representative_elementary_volume
    metrics.two_point_correlation_bf
    metrics.two_point_correlation_fft

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
from .__funcs__ import pc_curve_from_ibip
from .__funcs__ import pc_curve_from_mio
from ._fractal_dims import boxcount
