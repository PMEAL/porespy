r"""
#######
PoreSpy
#######

PoreSpy is a package for performing image analysis on volumetric images of
porous materials.

PoreSpy consists of several key modules. Each module is consisted of
several functions. Here, you'll find a comprehensive documentation of the
modules, occasionally with basic embedded examples on how to use them.

"""

from .tools._utils import Settings as _Settings

settings = _Settings()

from . import tools
from . import filters
from . import metrics
from . import networks
from . import generators
from . import simulations
from . import visualization
from . import io
# The dns module will be deprecated in V3, in favor of simulations
from . import dns

from .visualization import imshow

import numpy as _np
_np.seterr(divide='ignore', invalid='ignore')

__version__ = tools._get_version()
