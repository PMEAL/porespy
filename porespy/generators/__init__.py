r"""

===============================================================================
generators
===============================================================================

**Generate Artificial Images**

This module contains a variety of functions for generating artificial images
of porous materials, generally for testing, validation, debugging, and
illustration purposes.

.. autosummary::

    porespy.generators.blobs
    porespy.generators.bundle_of_tubes
    porespy.generators.cylindrical_plug
    porespy.generators.cylinders
    porespy.generators.fractal_noise
    porespy.generators.insert_shape
    porespy.generators.lattice_spheres
    porespy.generators.line_segment
    porespy.generators.overlapping_spheres
    porespy.generators.polydisperse_spheres
    porespy.generators.pseudo_electrostatic_packing
    porespy.generators.pseudo_gravity_packing
    porespy.generators.RSA
    porespy.generators.voronoi_edges

.. autofunction:: blobs
.. autofunction:: bundle_of_tubes
.. autofunction:: cylindrical_plug
.. autofunction:: cylinders
.. autofunction:: fractal_noise
.. autofunction:: insert_shape
.. autofunction:: lattice_spheres
.. autofunction:: line_segment
.. autofunction:: overlapping_spheres
.. autofunction:: polydisperse_spheres
.. autofunction:: pseudo_electrostatic_packing
.. autofunction:: pseudo_gravity_packing
.. autofunction:: RSA
.. autofunction:: voronoi_edges

"""

from .__imgen__ import blobs
from .__imgen__ import bundle_of_tubes
from .__imgen__ import cylinders
from .__imgen__ import insert_shape
from .__imgen__ import lattice_spheres
from .__imgen__ import line_segment
from .__imgen__ import overlapping_spheres
from .__imgen__ import polydisperse_spheres
from .__imgen__ import RSA
from .__imgen__ import voronoi_edges
from .__gravity__ import pseudo_gravity_packing
from .__electrostatic__ import pseudo_electrostatic_packing
from .__cylinder__ import cylindrical_plug
from ._noise import fractal_noise
from ._borders import *
