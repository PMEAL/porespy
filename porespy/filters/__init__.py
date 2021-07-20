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
    filters.flood_func
    filters.hold_peaks
    filters.local_thickness
    filters.nphase_border
    filters.nl_means_layered
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

from ._funcs import apply_chords
from ._funcs import apply_chords_3D
from ._funcs import apply_padded
from ._funcs import chunked_func
from ._funcs import distance_transform_lin
from ._funcs import fill_blind_pores
from ._funcs import find_disconnected_voxels
from ._funcs import find_dt_artifacts
from ._funcs import flood
from ._funcs import flood_func
from ._funcs import hold_peaks
from ._funcs import local_thickness
from ._funcs import nphase_border
from ._funcs import porosimetry
from ._funcs import prune_branches
from ._funcs import region_size
from ._funcs import trim_disconnected_blobs
from ._funcs import trim_extrema
from ._funcs import trim_floating_solid
from ._funcs import trim_nonpercolating_paths
from ._funcs import trim_small_clusters
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
from ._fftmorphology import fftmorphology
from . import imagej
from ._ibip import ibip
from ._ibip import find_trapped_regions
from ._ibip_gpu import ibip_gpu