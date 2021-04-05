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
   tools.fftmorphology
   tools.find_outer_region
   tools.get_border
   tools.get_planes
   tools.insert_cylinder
   tools.insert_sphere
   tools.in_hull
   tools.make_contiguous
   tools.mesh_region
   tools.norm_to_uniform
   tools.overlay
   tools.ps_ball
   tools.ps_disk
   tools.ps_rect
   tools.ps_round
   tools.pad_faces
   tools.randomize_colors
   tools.seq_to_satn
   tools.size_to_seq
   tools.subdivide
   tools.zero_corners

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
