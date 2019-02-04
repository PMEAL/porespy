r'''
===============================================================================
PoreSpy
===============================================================================

Porespy is a collection of functions for analyze voxel based images of porous
materials.  It consists of the following sub-modules:

.. autosummary::

    porespy.generators
    porespy.filters
    porespy.metrics
    porespy.network_extraction
    porespy.tools
    porespy.io
    porespy.visualization

PoreSpy is primarily developed and maintained by the PMEAL group at the
University of Waterloo.  For more information visit pmeal.com or porespy.org.

'''

__version__ = "0.4.2"

from . import metrics
from . import tools
from . import filters
from . import generators
from . import simulations
from . import network_extraction
from . import visualization
from . import io
