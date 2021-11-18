r"""

Tools
#####

**Helper Functions**

This module contains a variety of functions for manipulating images in
ways do NOT return a modified version of the original image.

.. currentmodule:: porespy

.. autosummary::
   :template: mybase.rst
   :toctree: generated/

   tools.align_image_with_openpnm
   tools.bbox_to_slices
   tools.extend_slice
   tools.extract_subsection
   tools.extract_regions
   tools.extract_cylinder
   tools.extract_subsection
   tools.find_outer_region
   tools.get_border
   tools.get_planes
   tools.insert_cylinder
   tools.insert_sphere
   tools.in_hull
   tools.isolate_object
   tools.make_contiguous
   tools.marching_map
   tools.mesh_region
   tools.norm_to_uniform
   tools.overlay
   tools.ps_ball
   tools.ps_disk
   tools.ps_rect
   tools.ps_round
   tools.randomize_colors
   tools.recombine
   tools.subdivide
   tools.unpad

"""

from ._funcs import *
from ._utils import *
