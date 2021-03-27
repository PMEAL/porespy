r"""

Filters
#######

**Highlight Features of Interest**

This module contains a variety of functions for altering images based on
the structural characteristics, such as pore sizes.  A definition of a
*filter* is a function that returns an image the shape as the original
image, but with altered values.

.. autosummary::
   :toctree: generated/

    apply_chords
    apply_chords_3D
    apply_padded
    chunked_func
    distance_transform_lin
    fftmorphology
    fill_blind_pores
    find_disconnected_voxels
    find_dt_artifacts
    find_peaks
    flood
    hold_peaks
    local_thickness
    nphase_border
    porosimetry
    prune_branches
    reduce_peaks
    region_size
    snow_partitioning
    snow_partitioning_n
    snow_partitioning_parallel
    trim_disconnected_blobs
    trim_extrema
    trim_floating_solid
    trim_nearby_peaks
    trim_nonpercolating_paths
    trim_saddle_points
    trim_small_clusters

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
from .__funcs__ import find_peaks
from .__funcs__ import flood
from .__funcs__ import hold_peaks
from .__funcs__ import local_thickness
from .__funcs__ import nphase_border
from .__funcs__ import porosimetry
from .__funcs__ import prune_branches
from .__funcs__ import reduce_peaks
from .__funcs__ import region_size
from .__funcs__ import snow_partitioning
from .__funcs__ import snow_partitioning_n
from .__funcs__ import snow_partitioning_parallel
from .__funcs__ import trim_disconnected_blobs
from .__funcs__ import trim_extrema
from .__funcs__ import trim_floating_solid
from .__funcs__ import trim_nonpercolating_paths
from .__funcs__ import trim_nearby_peaks
from .__funcs__ import trim_saddle_points
from .__funcs__ import trim_small_clusters
