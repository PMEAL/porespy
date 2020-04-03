r'''

===============================================================================
visualization
===============================================================================

**Create Basic Views**

This module contains functions for quickly visualizing 3D images in 2D views.

.. autosummary::

    porespy.visualization.sem
    porespy.visualization.xray
    porespy.visualization.show_3D
    porespy.visualization.set_mpl_style
    porespy.visualization.show_mesh

.. autofunction:: sem
.. autofunction:: xray
.. autofunction:: show_3D
.. autofunction:: set_mpl_style
.. autofunction:: show_mesh

'''


from .__views__ import sem
from .__views__ import show_planes
from .__views__ import xray
from .__views__ import show_3D
from .__plots__ import show_mesh
from .__funcs__ import set_mpl_style
