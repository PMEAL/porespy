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

    generators.rsa
    generators.random_spheres
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

from ._imgen import *
from ._pseudo_packings import *
from ._noise import *
from ._spheres_from_coords import *
from ._borders import *
from ._fractals import *
from ._micromodels import *
