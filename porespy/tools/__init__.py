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
   tools.unpad
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
    "ps_ball",
    "ps_disk",
    "ps_rect",
    "ps_round",
    "pad_faces",
    "randomize_colors",
    "seq_to_satn",
    "size_to_seq",
    "subdivide",
    "unpad",
    "zero_corners",
    "sanitize_filename",
    "get_tqdm",
    "show_docstring"]

from .__funcs__ import (_check_for_singleton_axes, _create_alias_map,
                        align_image_with_openpnm, bbox_to_slices, extend_slice,
                        extract_cylinder, extract_regions, extract_subsection,
                        fftmorphology, find_outer_region, get_border,
                        get_planes, in_hull, insert_cylinder, insert_sphere,
                        isolate_object, make_contiguous, marching_map,
                        mesh_region, norm_to_uniform, overlay, pad_faces,
                        ps_ball, ps_disk, ps_rect, ps_round, randomize_colors,
                        sanitize_filename, seq_to_satn, size_to_satn,
                        size_to_seq, subdivide, zero_corners)
from .__utils__ import get_tqdm, show_docstring
from ._unpadfunc import unpad
