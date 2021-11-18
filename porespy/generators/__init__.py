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
    generators.random_cantor_dust
    generators.sierpinski_foam
    generators.voronoi_edges

"""

from ._imgen import *
from ._pseudo_packings import *
from ._noise import fractal_noise
from ._borders import *
from ._fractals import *
