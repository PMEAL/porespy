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
    metrics.geometrical_tortuosity
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

from .__funcs__ import (chord_counts, chord_length_distribution,
                        lineal_path_distribution, mesh_surface_area,
                        pc_curve_from_ibip, pc_curve_from_mio, phase_fraction,
                        pore_size_distribution, porosity, porosity_profile,
                        radial_density_distribution, region_interface_areas,
                        region_surface_areas, representative_elementary_volume,
                        two_point_correlation_bf, two_point_correlation_fft)
from .__regionprops__ import prop_to_image, props_to_DataFrame, regionprops_3D
from ._fractal_dims import boxcount
from ._geometrical_tortuosity import geometrical_tortuosity
