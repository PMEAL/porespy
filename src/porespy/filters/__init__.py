r"""

Collection of functions for altering images based on structural properties
##########################################################################

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
    filters.find_trapped_regions
    filters.flood
    filters.flood_func
    filters.hold_peaks
    filters.ibip
    filters.ibip_gpu
    filters.imagej
    filters.local_thickness
    filters.nl_means_layered
    filters.nphase_border
    filters.pc_to_satn
    filters.porosimetry
    filters.prune_branches
    filters.reduce_peaks
    filters.region_size
    filters.satn_to_seq
    filters.seq_to_satn
    filters.size_to_satn
    filters.size_to_seq
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
from ._fftmorphology import *
from ._funcs import *
from ._nlmeans import *
from ._size_seq_satn import *
from ._snows import *
