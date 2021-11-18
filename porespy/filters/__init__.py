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

from ._funcs import *
from ._ibip import *
from ._snows import *
from ._size_seq_satn import *
from ._ibip_gpu import ibip_gpu
from ._nlmeans import nl_means_layered
from ._fftmorphology import fftmorphology
from . import imagej