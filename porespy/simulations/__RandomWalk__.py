# -*- coding: utf-8 -*-
"""
@author: Tom Tranter, Matt Lam, Matt Kok, Jeff Gostick
PMEAL lab, University of Waterloo, Ontario, Canada

Random Walker Code
"""

import pytrax
import logging


class RandomWalk(pytrax.__RandomWalk__.RandomWalk):
    r'''
    Wrapper class for the pytrax RandomWalk class

    The RandomWalk class implements a simple vectorized version of a random
    walker. The image that is analyzed can be 2 or 3 dimensional and the run
    method can take an arbitrary number of steps and walkers.
    Walker starting positions can be set to the same point or to different ones
    chosen at random.
    The image is duplicated and flipped a number of times for visualization as
    this represents the real path the walker would have taken if it had not
    been confined to the bounds of the image.
    The mean square displacement is calculated and the gradient of the msd
    when plotted over time is equal to 1/tau, the tortuosity factor.
    The image data and walker co-ordinates can be exported for visualization
    in paraview.
    A simple 2d slice can also be viewed directly using matplotlib.
    Currently walkers do not travel along diagonals.
    Running walkers in parallel by setting num_proc is possible to speed up
    calculations as wach walker path is completely independent of the others.
    '''

    def __init__(self, image=None, seed=False):
        r'''
        Get image info and make a bigger periodically flipped image for viz

        Parameters
        ----------
        image: ndarray of int
            2D or 3D image with 1 denoting pore space and 0 denoting solid

        seed: bool
            Determines whether to seed the random number generators so that
            Simulation is repeatable. Warning - This results in only semi-
            random walks so should only be used for debugging

        Examples
        --------

        Creating a RandomWalk object:

        >>> import porespy as ps
        >>> import pytrax as pt
        >>> im = ps.generators.blobs([100, 100])
        >>> rw = pt.RandomWalk(im)
        >>> rw.run(nt=1000, nw=1)
        '''
        if image is not None:
            super().__init__(image, seed)
        else:
            logging.error('Please instantiate class with an image')
