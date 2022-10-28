r"""

Collection of functions for generating synthetic images
#######################################################

This module contains a variety of functions for generating artificial
images of porous materials, generally for testing, validation,
debugging, and illustration purposes.

.. currentmodule:: porespy

.. autosummary::
   :template: mybase.rst
   :toctree: generated/

    generators.RSA
    generators.blobs
    generators.borders
    generators.bundle_of_tubes
    generators.cylinders
    generators.cylindrical_plug
    generators.faces
    generators.fractal_noise
    generators.insert_shape
    generators.lattice_spheres
    generators.line_segment
    generators.overlapping_spheres
    generators.polydisperse_spheres
    generators.pseudo_electrostatic_packing
    generators.pseudo_gravity_packing
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
from ._imgen import rsa
from ._imgen import voronoi_edges
from ._pseudo_packings import pseudo_gravity_packing
from ._pseudo_packings import pseudo_electrostatic_packing
from ._cylinder import cylindrical_plug
from ._noise import fractal_noise
from ._borders import *
from ._fractals import random_cantor_dust
from ._fractals import sierpinski_foam
