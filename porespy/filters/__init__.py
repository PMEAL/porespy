r"""
=======
Filters
=======

This module contains a variety of functions for altering images based on the
structural characteristics, such as pore sizes.  A definition of a *filter* is
a function that returns an image the shape as the original image, but with
altered values.

+---------------------------+----------------------------------------------------------------------+
| Method                    | Description                                                          |
+===========================+======================================================================+
| apply_chords              | Adds chords to the void space in the specified direction.  The ch... |
+---------------------------+----------------------------------------------------------------------+
| apply_chords_3D           | Adds chords to the void space in all three principle directions. ... |
+---------------------------+----------------------------------------------------------------------+
| distance_transform_lin    | Replaces each void voxel with the linear distance to the nearest ... |
+---------------------------+----------------------------------------------------------------------+
| fftmorphology             | Perform morphological operations on binary images using fft appro... |
+---------------------------+----------------------------------------------------------------------+
| fill_blind_pores          | Fills all pores that are not connected to the edges of the image.    |
+---------------------------+----------------------------------------------------------------------+
| find_disconnected_voxels  | This identifies all pore (or solid) voxels that are not connected... |
+---------------------------+----------------------------------------------------------------------+
| find_peaks                | Returns all local maxima in the distance transform                   |
+---------------------------+----------------------------------------------------------------------+
| flood                     | Floods/fills each region in an image with a single value based on... |
+---------------------------+----------------------------------------------------------------------+
| local_thickness           | For each voxel, this functions calculates the radius of the large... |
+---------------------------+----------------------------------------------------------------------+
| porosimetry               | Performs a porosimetry simulution on the image                       |
+---------------------------+----------------------------------------------------------------------+
| reduce_peaks              | Any peaks that are broad or elongated are replaced with a single ... |
+---------------------------+----------------------------------------------------------------------+
| snow_partitioning         | Partitions the void space into pore regions using a marker-based ... |
+---------------------------+----------------------------------------------------------------------+
| trim_extrema              | Trims local extrema in greyscale values by a specified amount.       |
+---------------------------+----------------------------------------------------------------------+
| trim_floating_solid       | Removes all solid that that is not attached to the edges of the i... |
+---------------------------+----------------------------------------------------------------------+
| trim_nonpercolating_paths | Removes all nonpercolating paths between specified edges             |
+---------------------------+----------------------------------------------------------------------+
| trim_nearby_peaks         | Finds pairs of peaks that are nearer to each other than to the so... |
+---------------------------+----------------------------------------------------------------------+
| trim_saddle_points        | Removes peaks that were mistakenly identified because they lied on a |
+---------------------------+----------------------------------------------------------------------+

.. autofunction:: apply_chords
.. autofunction:: apply_chords_3D
.. autofunction:: distance_transform_lin
.. autofunction:: fftmorphology
.. autofunction:: find_disconnected_voxels
.. autofunction:: find_dt_artifacts
.. autofunction:: fill_blind_pores
.. autofunction:: trim_floating_solid
.. autofunction:: trim_nonpercolating_paths
.. autofunction:: porosimetry
.. autofunction:: local_thickness
.. autofunction:: trim_extrema
.. autofunction:: flood
.. autofunction:: find_peaks
.. autofunction:: trim_saddle_points
.. autofunction:: trim_nearby_peaks
.. autofunction:: region_size
.. autofunction:: snow_partitioning

"""
from .__funcs__ import apply_chords
from .__funcs__ import apply_chords_3D
from .__funcs__ import distance_transform_lin
from .__funcs__ import fftmorphology
from .__funcs__ import fill_blind_pores
from .__funcs__ import find_disconnected_voxels
from .__funcs__ import find_dt_artifacts
from .__funcs__ import find_peaks
from .__funcs__ import flood
from .__funcs__ import hold_peaks
from .__funcs__ import local_thickness
from .__funcs__ import porosimetry
from .__funcs__ import reduce_peaks
from .__funcs__ import region_size
from .__funcs__ import snow_partitioning
from .__funcs__ import trim_extrema
from .__funcs__ import trim_floating_solid
from .__funcs__ import trim_nonpercolating_paths
from .__funcs__ import trim_nearby_peaks
from .__funcs__ import trim_saddle_points
from .__funcs__ import nphase_border
