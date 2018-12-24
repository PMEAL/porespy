r'''
=============
Visualization
=============

This module contains functions for quickly visualizing 3D images in 2D views.

.. autofunction:: sem
.. autofunction:: xray
.. autofunction:: set_mpl_style
.. autofunction:: show_mesh


'''


from .__views__ import sem
from .__views__ import show_planes
from .__views__ import xray
from .__plots__ import show_mesh
from .__funcs__ import set_mpl_style
