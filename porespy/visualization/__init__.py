r"""

Visualization
#############

**Create Basic Views**

This module contains functions for quickly visualizing 3D images in 2D
views.

.. currentmodule:: porespy

.. autosummary::
   :template: mybase.rst
   :toctree: generated/

   visualization.sem
   visualization.show_planes
   visualization.xray
   visualization.show_3D
   visualization.imshow
   visualization.bar
   visualization.show_mesh
   visualization.set_mpl_style

"""

__all__ = [
    "sem",
    "show_planes",
    "xray",
    "show_3D",
    "imshow",
    "bar",
    "show_mesh",
    "set_mpl_style"]

from .__funcs__ import satn_to_movie, set_mpl_style
from .__plots__ import bar, imshow, show_mesh
from .__views__ import sem, show_3D, show_planes, xray
