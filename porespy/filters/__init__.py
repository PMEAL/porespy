r"""

Filters
#######

**Highlight Features of Interest**

This module contains a variety of functions for altering images based on
the structural characteristics, such as pore sizes.  A definition of a
*filter* is a function that returns an image the shape as the original
image, but with altered values.

.. currentmodule:: porespy

.. autosummary::
   :template: mybase.rst
   :toctree: generated/

    filters.apply_chords
    filters.apply_chords_3D
    filters.apply_padded
    filters.chunked_func
    filters.distance_transform_lin
    filters.fftmorphology
    filters.fill_blind_pores
    filters.find_disconnected_voxels
    filters.find_dt_artifacts
    filters.find_peaks
    filters.flood
    filters.hold_peaks
    filters.local_thickness
    filters.nphase_border
    filters.porosimetry
    filters.prune_branches
    filters.reduce_peaks
    filters.region_size
    filters.snow_partitioning
    filters.snow_partitioning_n
    filters.snow_partitioning_parallel
    filters.trim_disconnected_blobs
    filters.trim_extrema
    filters.trim_floating_solid
    filters.trim_nearby_peaks
    filters.trim_nonpercolating_paths
    filters.trim_saddle_points
    filters.trim_small_clusters


"""

from . import imagej
from .__funcs__ import (apply_chords, apply_chords_3D, apply_padded,
                        chunked_func, distance_transform_lin, fftmorphology,
                        fill_blind_pores, find_disconnected_voxels,
                        find_dt_artifacts, flood, flood_func, hold_peaks,
                        local_thickness, nphase_border, porosimetry,
                        prune_branches, region_size, trim_disconnected_blobs,
                        trim_extrema, trim_floating_solid,
                        trim_nonpercolating_paths, trim_small_clusters)
from .__ibip__ import find_trapped_regions, ibip
from ._nlmeans import nl_means_layered
from ._snows import (find_peaks, reduce_peaks, snow_partitioning,
                     snow_partitioning_n, snow_partitioning_parallel,
                     trim_nearby_peaks, trim_saddle_points)
