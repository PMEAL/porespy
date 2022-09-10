r"""

Collection of functions for extracting quantitative information from images
###########################################################################

This submodule contains functions for determining key metrics about an
image. Typically these are applied to an image after applying a filter,
but a few functions can be applied directly to the binary image.

.. currentmodule:: porespy

.. autosummary::
   :template: mybase.rst
   :toctree: generated/

    metrics.boxcount
    metrics.chord_counts
    metrics.chord_length_distribution
    metrics.find_h
    metrics.lineal_path_distribution
    metrics.mesh_surface_area
    metrics.mesh_volume
    metrics.pc_curve
    metrics.pc_curve_from_ibip
    metrics.pc_curve_from_mio
    metrics.phase_fraction
    metrics.pore_size_distribution
    metrics.porosity
    metrics.porosity_profile
    metrics.prop_to_image
    metrics.props_to_DataFrame
    metrics.radial_density_distribution
    metrics.region_interface_areas
    metrics.region_surface_areas
    metrics.region_volumes
    metrics.regionprops_3D
    metrics.representative_elementary_volume
    metrics.satn_profile
    metrics.two_point_correlation

"""

from ._regionprops import regionprops_3D
from ._regionprops import props_to_DataFrame
from ._regionprops import prop_to_image
from ._funcs import chord_counts
from ._funcs import chord_length_distribution
from ._funcs import find_h
from ._funcs import lineal_path_distribution
from ._funcs import pore_size_distribution
from ._funcs import radial_density_distribution
from ._funcs import porosity
from ._funcs import porosity_profile
from ._funcs import representative_elementary_volume
from ._funcs import satn_profile
from ._funcs import two_point_correlation
from ._funcs import phase_fraction
from ._funcs import pc_curve
from ._funcs import pc_curve_from_ibip
from ._funcs import pc_curve_from_mio
from ._fractal_dims import boxcount
from ._meshtools import mesh_surface_area
from ._meshtools import mesh_volume
from ._meshtools import region_interface_areas
from ._meshtools import region_surface_areas
from ._meshtools import region_volumes
