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

from ._regionprops import *
from ._funcs import *
from ._meshtools import *
