r"""
=======
Filters
=======

This module contains a variety of functions for altering image based on the
structural characteristics, such as pore sizes.

.. autofunction:: apply_chords
.. autofunction:: apply_chords_3D
.. autofunction:: fill_blind_pores
.. autofunction:: find_disconnected_voxels
.. autofunction:: flood
.. autofunction:: local_thickness
.. autofunction:: norm_to_uniform
.. autofunction:: porosimetry
.. autofunction:: trim_extrema
.. autofunction:: trim_floating_solid

"""
from .__funcs__ import apply_chords
from .__funcs__ import apply_chords_3D
from .__funcs__ import fill_blind_pores
from .__funcs__ import find_disconnected_voxels
from .__funcs__ import flood
from .__funcs__ import local_thickness
from .__funcs__ import norm_to_uniform
from .__funcs__ import porosimetry
from .__funcs__ import trim_extrema
from .__funcs__ import trim_floating_solid
