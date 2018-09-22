r'''
=====
Tools
=====

This module contains a variety of functions for manipulating images in ways
that do NOT return a modified version of the original image.

.. autofunction:: bbox_to_slices
.. autofunction:: extend_slice
.. autofunction:: extract_subsection
.. autofunction:: extract_cylinder
.. autofunction:: fftmorphology
.. autofunction:: find_outer_region
.. autofunction:: get_border
.. autofunction:: get_planes
.. autofunction:: get_slice
.. autofunction:: in_hull
.. autofunction:: make_contiguous
.. autofunction:: norm_to_uniform
.. autofunction:: randomize_colors
.. autofunction:: subdivide

'''
from .__funcs__ import align_image_with_openpnm
from .__funcs__ import bbox_to_slices
from .__funcs__ import extend_slice
from .__funcs__ import extract_subsection
from .__funcs__ import extract_cylinder
from .__funcs__ import fftmorphology
from .__funcs__ import find_outer_region
from .__funcs__ import get_border
from .__funcs__ import get_planes
from .__funcs__ import get_slice
from .__funcs__ import in_hull
from .__funcs__ import make_contiguous
from .__funcs__ import norm_to_uniform
from .__funcs__ import randomize_colors
from .__funcs__ import subdivide
