r"""

Collection of helper functions for manipulating images
######################################################

This module contains a variety of functions for manipulating images in
ways that do NOT return a modified version of the original image.

.. currentmodule:: porespy

.. autosummary::
   :template: mybase.rst
   :toctree: generated/

    tools.Results
    tools.align_image_with_openpnm
    tools.bbox_to_slices
    tools.extend_slice
    tools.extract_cylinder
    tools.extract_regions
    tools.extract_subsection
    tools.find_outer_region
    tools.get_border
    tools.get_planes
    tools.get_tqdm
    tools.in_hull
    tools.insert_cylinder
    tools.insert_sphere
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
    tools.sanitize_filename
    tools.show_docstring
    tools.subdivide
    tools.unpad

"""

from ._funcs import *
from ._utils import *
from ._funcs import _check_for_singleton_axes
from ._unpad import unpad
from ._sphere_insertions import *


def _get_version():
    from porespy.__version__ import __version__ as ver
    suffix = ".dev0"
    if ver.endswith(suffix):
        ver = ver[:-len(suffix)]
    return ver
