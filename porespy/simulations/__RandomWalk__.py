# -*- coding: utf-8 -*-
"""
@author: Tom Tranter, Matt Lam, Matt Kok, Jeff Gostick
PMEAL lab, University of Waterloo, Ontario, Canada

Random Walker Code
"""

import pytrax


class RandomWalk(pytrax.RandomWalk):
    r'''
    Wrapper class for the pytrax RandomWalk class
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
