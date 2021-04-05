r"""

Generators
##########

**Generate Artificial Images**

This module contains a variety of functions for generating artificial
images of porous materials, generally for testing, validation,
debugging, and illustration purposes.

.. currentmodule:: porespy

.. autosummary::
   :template: mybase.rst
   :toctree: generated/

    generators.blobs
    generators.bundle_of_tubes
    generators.cylindrical_plug
    generators.cylinders
    generators.fractal_noise
    generators.insert_shape
    generators.lattice_spheres
    generators.line_segment
    generators.overlapping_spheres
    generators.polydisperse_spheres
    generators.pseudo_electrostatic_packing
    generators.pseudo_gravity_packing
    generators.RSA
    generators.voronoi_edges

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
