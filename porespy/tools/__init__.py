r'''
=====
Tools
=====

This module contains a variety of functions for manipulating images in ways 
that do NOT return a modified version of the original image.  

.. autofunction:: extend_slice
.. autofunction:: extract_subsection
.. autofunction:: extract_cylinder
.. autofunction:: find_outer_region
.. autofunction:: get_border
.. autofunction:: get_slice
.. autofunction:: make_contiguous
.. autofunction:: randomize_colors
.. autofunction:: subdivide


'''

from .__funcs__ import binary_opening_fast
from .__funcs__ import extend_slice
from .__funcs__ import extract_subsection
from .__funcs__ import extract_cylinder
from .__funcs__ import find_outer_region
from .__funcs__ import get_border
from .__funcs__ import get_slice
from .__funcs__ import make_contiguous
from .__funcs__ import randomize_colors
from .__funcs__ import subdivide
