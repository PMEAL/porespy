r'''

===============================================================================
tools
===============================================================================

**Helper Functions**

This module contains a variety of functions for manipulating images in ways
that do NOT return a modified version of the original image.

.. autosummary::

    porespy.tools.align_image_with_openpnm
    porespy.tools.bbox_to_slices
    porespy.tools.extend_slice
    porespy.tools.extract_subsection
    porespy.tools.extract_regions
    porespy.tools.extract_cylinder
    porespy.tools.extract_subsection
    porespy.tools.fftmorphology
    porespy.tools.find_outer_region
    porespy.tools.get_border
    porespy.tools.get_planes
    porespy.tools.insert_cylinder
    porespy.tools.insert_sphere
    porespy.tools.in_hull
    porespy.tools.make_contiguous
    porespy.tools.mesh_region
    porespy.tools.norm_to_uniform
    porespy.tools.overlay
    porespy.tools.ps_disk
    porespy.tools.ps_ball
    porespy.tools.pad_faces
    porespy.tools.randomize_colors
    porespy.tools.seq_to_satn
    porespy.tools.size_to_seq
    porespy.tools.subdivide

.. autofunction:: align_image_with_openpnm
.. autofunction:: bbox_to_slices
.. autofunction:: extend_slice
.. autofunction:: extract_cylinder
.. autofunction:: extract_regions
.. autofunction:: extract_subsection
.. autofunction:: fftmorphology
.. autofunction:: find_outer_region
.. autofunction:: get_border
.. autofunction:: get_planes
.. autofunction:: insert_cylinder
.. autofunction:: insert_sphere
.. autofunction:: in_hull
.. autofunction:: make_contiguous
.. autofunction:: mesh_region
.. autofunction:: norm_to_uniform
.. autofunction:: overlay
.. autofunction:: ps_disk
.. autofunction:: ps_ball
.. autofunction:: pad_faces
.. autofunction:: randomize_colors
.. autofunction:: seq_to_satn
.. autofunction:: size_to_seq
.. autofunction:: subdivide

'''

from .__funcs__ import align_image_with_openpnm
from .__funcs__ import bbox_to_slices
from .__funcs__ import _create_alias_map
from .__funcs__ import extend_slice
from .__funcs__ import extract_cylinder
from .__funcs__ import extract_subsection
from .__funcs__ import extract_regions
from .__funcs__ import fftmorphology
from .__funcs__ import find_outer_region
from .__funcs__ import get_border
from .__funcs__ import get_planes
from .__funcs__ import insert_cylinder
from .__funcs__ import insert_sphere
from .__funcs__ import in_hull
from .__funcs__ import make_contiguous
from .__funcs__ import mesh_region
from .__funcs__ import norm_to_uniform
from .__funcs__ import overlay
from .__funcs__ import randomize_colors
from .__funcs__ import ps_disk
from .__funcs__ import ps_ball
from .__funcs__ import pad_faces
from .__funcs__ import seq_to_satn
from .__funcs__ import size_to_seq
from .__funcs__ import subdivide
