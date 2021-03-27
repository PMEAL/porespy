r"""

Visualization
#############

**Create Basic Views**

This module contains functions for quickly visualizing 3D images in 2D
views.

.. autosummary::
   :toctree: generated/

   sem
   show_planes
   xray
   show_3D
   imshow
   bar
   show_mesh
   set_mpl_style

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

from .__views__ import sem, show_planes, xray, show_3D
from .__plots__ import imshow, bar, show_mesh
from .__funcs__ import set_mpl_style
