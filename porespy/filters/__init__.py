r"""

===============================================================================
filters
===============================================================================

**Highlight Features of Interest**

This module contains a variety of functions for altering images based on the
structural characteristics, such as pore sizes.  A definition of a *filter* is
a function that returns an image the shape as the original image, but with
altered values.

.. autosummary::

    porespy.filters.apply_chords
    porespy.filters.apply_chords_3D
    porespy.filters.apply_padded
    porespy.filters.chunked_func
    porespy.filters.distance_transform_lin
    porespy.filters.fftmorphology
    porespy.filters.fill_blind_pores
    porespy.filters.find_disconnected_voxels
    porespy.filters.find_dt_artifacts
    porespy.filters.find_peaks
    porespy.filters.flood
    porespy.filters.hold_peaks
    porespy.filters.local_thickness
    porespy.filters.nphase_border
    porespy.filters.porosimetry
    porespy.filters.prune_branches
    porespy.filters.reduce_peaks
    porespy.filters.region_size
    porespy.filters.snow_partitioning
    porespy.filters.snow_partitioning_n
    porespy.filters.snow_partitioning_parallel
    porespy.filters.trim_disconnected_blobs
    porespy.filters.trim_extrema
    porespy.filters.trim_floating_solid
    porespy.filters.trim_nearby_peaks
    porespy.filters.trim_nonpercolating_paths
    porespy.filters.trim_saddle_points
    porespy.filters.trim_small_clusters


.. autofunction:: apply_chords
.. autofunction:: apply_chords_3D
.. autofunction:: apply_padded
.. autofunction:: chunked_func
.. autofunction:: distance_transform_lin
.. autofunction:: fftmorphology
.. autofunction:: fill_blind_pores
.. autofunction:: find_disconnected_voxels
.. autofunction:: find_dt_artifacts
.. autofunction:: find_peaks
.. autofunction:: flood
.. autofunction:: hold_peaks
.. autofunction:: local_thickness
.. autofunction:: nphase_border
.. autofunction:: porosimetry
.. autofunction:: prune_branches
.. autofunction:: reduce_peaks
.. autofunction:: region_size
.. autofunction:: snow_partitioning
.. autofunction:: snow_partitioning_n
.. autofunction:: snow_partitioning_parallel
.. autofunction:: trim_disconnected_blobs
.. autofunction:: trim_extrema
.. autofunction:: trim_floating_solid
.. autofunction:: trim_nearby_peaks
.. autofunction:: trim_nonpercolating_paths
.. autofunction:: trim_saddle_points
.. autofunction:: trim_small_clusters

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
