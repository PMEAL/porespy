r'''
The pore space image should be 1's for void and 0's for solid
'''
from .__imio__ import ImageIO
from .__main__ import ChordLengthDistribution as CLD
from .__main__ import TwoPointCorrelation as TPC
from .__rev__ import REV

imopen = ImageIO.imopen
del ImageIO
