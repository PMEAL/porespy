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

from ._imgen import blobs
from ._imgen import bundle_of_tubes
from ._imgen import cylinders
from ._imgen import insert_shape
from ._imgen import lattice_spheres
from ._imgen import line_segment
from ._imgen import overlapping_spheres
from ._imgen import polydisperse_spheres
from ._imgen import RSA
from ._imgen import voronoi_edges
from ._gravity import pseudo_gravity_packing
from ._electrostatic import pseudo_electrostatic_packing
from ._cylinder import cylindrical_plug
from ._noise import fractal_noise
from ._borders import *
from ._fractals import random_cantor_dust
from ._fractals import sierpinski_foam
