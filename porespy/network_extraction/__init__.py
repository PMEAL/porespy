r"""
This module contains a variety of network extraction algorithms
"""

from .__snow__ import SNOW_peaks
from .__funcs__ import trim_extrema, partition_pore_space, all_peaks
from .__getnet__ import extract_pore_network