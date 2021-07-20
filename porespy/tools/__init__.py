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

__all__ = [
    "align_image_with_openpnm",
    "bbox_to_slices",
    "extend_slice",
    "extract_cylinder",
    "extract_subsection",
    "extract_regions",
    "find_outer_region",
    "get_border",
    "get_planes",
    "insert_cylinder",
    "insert_sphere",
    "in_hull",
    "isolate_object",
    "make_contiguous",
    "marching_map",
    "mesh_region",
    "norm_to_uniform",
    "overlay",
    "ps_ball",
    "ps_disk",
    "ps_rect",
    "ps_round",
    "randomize_colors",
    "recombine",
    "subdivide",
    "unpad",
    "sanitize_filename",
    "get_tqdm",
    "show_docstring"]

from ._funcs import align_image_with_openpnm
from ._funcs import bbox_to_slices
from ._funcs import extend_slice
from ._funcs import extract_cylinder
from ._funcs import extract_subsection
from ._funcs import extract_regions
from ._funcs import find_outer_region
from ._funcs import get_border
from ._funcs import get_planes
from ._funcs import insert_cylinder
from ._funcs import insert_sphere
from ._funcs import in_hull
from ._funcs import isolate_object
from ._funcs import marching_map
from ._funcs import make_contiguous
from ._funcs import mesh_region
from ._funcs import norm_to_uniform
from ._funcs import overlay
from ._funcs import randomize_colors
from ._funcs import recombine
from ._funcs import ps_ball
from ._funcs import ps_disk
from ._funcs import ps_rect
from ._funcs import ps_round
from ._funcs import subdivide
from ._utils import sanitize_filename
from ._utils import get_tqdm
from ._utils import show_docstring
from ._utils import Results
from ._funcs import _check_for_singleton_axes
from ._unpad import unpad
