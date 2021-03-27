r"""

Tools
#####

**Helper Functions**

This module contains a variety of functions for manipulating images in
ways do NOT return a modified version of the original image.

.. autosummary::
   :toctree: generated/

   align_image_with_openpnm
   bbox_to_slices
   extend_slice
   extract_subsection
   extract_regions
   extract_cylinder
   extract_subsection
   fftmorphology
   find_outer_region
   get_border
   get_planes
   insert_cylinder
   insert_sphere
   in_hull
   make_contiguous
   mesh_region
   norm_to_uniform
   overlay
   ps_ball
   ps_disk
   ps_rect
   ps_round
   pad_faces
   randomize_colors
   seq_to_satn
   size_to_seq
   subdivide
   zero_corners

"""

__all__ = [
    "align_image_with_openpnm",
    "bbox_to_slices",
    "extend_slice",
    "extract_cylinder",
    "extract_subsection",
    "extract_regions",
    "fftmorphology",
    "find_outer_region",
    "get_border",
    "get_planes",
    "insert_cylinder",
    "insert_sphere",
    "in_hull",
    "make_contiguous",
    "mesh_region",
    "norm_to_uniform",
    "overlay",
    "randomize_colors",
    "ps_ball",
    "ps_disk",
    "ps_rect",
    "ps_round",
    "pad_faces",
    "seq_to_satn",
    "size_to_seq",
    "subdivide",
    "zero_corners",
    "sanitize_filename",
    "get_tqdm",
    "show_docstring"]

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
from .__funcs__ import ps_ball
from .__funcs__ import ps_disk
from .__funcs__ import ps_rect
from .__funcs__ import ps_round
from .__funcs__ import pad_faces
from .__funcs__ import seq_to_satn
from .__funcs__ import size_to_seq
from .__funcs__ import subdivide
from .__funcs__ import zero_corners
from .__funcs__ import sanitize_filename
from .__utils__ import get_tqdm
from .__utils__ import show_docstring
from .__funcs__ import _check_for_singleton_axes
