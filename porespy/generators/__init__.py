r"""

Generators
##########

**Generate Artificial Images**

This module contains a variety of functions for generating artificial
images of porous materials, generally for testing, validation,
debugging, and illustration purposes.

.. autosummary::
   :toctree: generated/

    blobs
    bundle_of_tubes
    cylindrical_plug
    cylinders
    fractal_noise
    insert_shape
    lattice_spheres
    line_segment
    overlapping_spheres
    polydisperse_spheres
    pseudo_electrostatic_packing
    pseudo_gravity_packing
    RSA
    voronoi_edges

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
