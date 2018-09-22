r"""
=======
Filters
=======

This module contains a variety of functions for altering image based on the
structural characteristics, such as pore sizes.  A definition of a *filter* is
 a function that returns an image the shape as the original image, but with
 altered values.

.. autofunction:: apply_chords
.. autofunction:: apply_chords_3D
.. autofunction:: fftmorphology
.. autofunction:: find_disconnected_voxels
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
.. autofunction:: reduce_peaks
.. autofunction:: snow_partitioning

"""
from .__funcs__ import apply_chords
from .__funcs__ import apply_chords_3D
from .__funcs__ import fftmorphology
from .__funcs__ import fill_blind_pores
from .__funcs__ import find_disconnected_voxels
from .__funcs__ import find_peaks
from .__funcs__ import flood
from .__funcs__ import local_thickness
from .__funcs__ import porosimetry
from .__funcs__ import reduce_peaks
from .__funcs__ import snow_partitioning
from .__funcs__ import trim_extrema
from .__funcs__ import trim_floating_solid
from .__funcs__ import trim_nonpercolating_paths
from .__funcs__ import trim_nearby_peaks
from .__funcs__ import trim_saddle_points
