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
    filters.size_to_seq
    filters.size_to_satn
    filters.seq_to_satn
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

from .__funcs__ import apply_chords
from .__funcs__ import apply_chords_3D
from .__funcs__ import apply_padded
from .__funcs__ import chunked_func
from .__funcs__ import distance_transform_lin
from .__funcs__ import fftmorphology
from .__funcs__ import fill_blind_pores
from .__funcs__ import find_disconnected_voxels
from .__funcs__ import find_dt_artifacts
from .__funcs__ import flood
from .__funcs__ import flood_func
from .__funcs__ import hold_peaks
from .__funcs__ import local_thickness
from .__funcs__ import nphase_border
from .__funcs__ import porosimetry
from .__funcs__ import prune_branches
from .__funcs__ import region_size
from .__funcs__ import trim_disconnected_blobs
from .__funcs__ import trim_extrema
from .__funcs__ import trim_floating_solid
from .__funcs__ import trim_nonpercolating_paths
from .__funcs__ import trim_small_clusters
from ._snows import snow_partitioning
from ._snows import snow_partitioning_n
from ._snows import snow_partitioning_parallel
from ._snows import find_peaks
from ._snows import reduce_peaks
from ._snows import trim_nearby_peaks
from ._snows import trim_saddle_points
from ._size_seq_satn import size_to_seq
from ._size_seq_satn import size_to_satn
from ._size_seq_satn import seq_to_satn
from ._nlmeans import nl_means_layered
from . import imagej
from .__ibip__ import ibip
from .__ibip__ import find_trapped_regions
