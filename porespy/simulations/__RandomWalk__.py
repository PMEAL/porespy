# -*- coding: utf-8 -*-
"""
@author: Tom Tranter

Random Walker Code
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import porespy as ps
from porespy.tools.__funcs__ import do_profile
import os
from tqdm import tqdm


class RandomWalk():
    r'''
    The RandomWalk class implements a simple vectorized version of a random
    walker. The image that is analyzed can be 2 or 3 dimensional and the run
    method can take an arbitrary number of steps and walkers.
    Walker starting positions can be set to the same point or to different ones
    chosen at random.
    The image is duplicated and flipped a number of times for visualization as
    this represents the real path the walker would have taken if it had not
    been confined to the bounds of the image.
    The mean square displacement is calculated and the gradient of the msd
    when plotted over time is equal to 1/tortuosity
    The image data and walker co-ordinates can be exported for visualization
    in paraview.
    A simple 2d slice can also be viewed directly using matplotlib.
    Currently walkers do not travel along diagonals.
    '''

    def __init__(self, image, offset=1, seed=False):
        r'''
        Get image info and make a bigger periodically flipped image for viz

        Parameters
        ----------
        image: ndarray of int
            2D or 3D image with 1 denoting pore space and 0 denoting solid

        offset: int (default = 1)
            The number of image offsets to start the real walkers in along each
            axis. The big image is flipped and tiled twice this many times so
            that walkers start in the middle.
        seed: bool
            Determines whether to seed the random number generators so that
            Simulation is repeatable

        Examples
        --------

        Creating a RandomWalk object:

        >>> import porespy as ps
        >>> im = ps.generators.blobs([100, 100])
        >>> rw = ps.simulations.RandomWalk(im, offset=1)
        >>> rw.run(nt=1000, nw=100)
        '''
        self.im = image
        self.shape = np.array(np.shape(self.im))
        self.dim = len(self.shape)
        self.offset = offset
        self.solid_value = 0
        self.seed = seed
        self._get_wall_map(self.im)

    def _transform_coord(self, coord=None, reflection=None):
        r'''
        Transform a coordinate from the original image to a reflected image

        Parameters
        ----------
        coord: ndarray of int
            coordinates in the original image
        reflection: ndarray of int
            number of times to shift the coordinate into a reflected image.
            An odd number results in a reflection and even results in a shift
        '''
        t_coord = coord.copy()
        for ax in range(self.dim):
            rs = reflection[:, ax] % 2 == 1
            t_coord[rs, ax] = self.shape[ax] - 1 - coord[rs, ax]
            t_coord[:, ax] += reflection[:, ax]*self.shape[ax]
        return t_coord

    def _build_big_image(self, num_copies=0):
        r'''
        Build the big image by flipping and stacking along each axis a number
        of times

        Parameters
        ----------
        num_copies: int
            the number of times to copy the image along each axis
        '''
        big_im = self.im.copy()
        func = [np.vstack, np.hstack, np.dstack]
        temp_im = self.im.copy()
        for ax in range(self.dim):
            flip_im = np.flip(temp_im, ax)
            for c in range(num_copies):
                if c % 2 == 0:
                    big_im = func[ax]((big_im, flip_im))
                else:
                    big_im = func[ax]((big_im, temp_im))
            temp_im = big_im.copy()
        return big_im

    def _rand_start(self, image, num=1):
        r'''
        Get a number of start points in the pore space of the image

        Parameters
        ----------
        image: ndarray of int
            2D or 3D image with 1 denoting pore space and 0 denoting solid
        num: int
            number of unique starting points to return
        '''
        inds = np.argwhere(image != self.solid_value)
        if self.seed:
            np.random.seed(1)
        try:
            choice = np.random.choice(np.arange(0, len(inds), 1),
                                      num,
                                      replace=False)
        except ValueError:
            choice = np.random.choice(np.arange(0, len(inds), 1),
                                      num,
                                      replace=True)
        return inds[choice]

    def _get_wall_map(self, image):
        r'''
        Function savesc a wall map and movement vectors.
        This is referred to later when random walker moves in a particular
        direction to detect where the walls are.

        Parameters
        ----------
        image: ndarray of int
            2D or 3D image with 1 denoting pore space and 0 denoting solid
        '''
        # Make boolean map where solid is True
        solid = image.copy() == self.solid_value
        solid = solid.astype(bool)
        moves = []
        for axis in range(self.dim):
            ax_list = []
            for direction in [-1, 1]:
                # Store the direction of the step in an array for later use
                step = np.arange(0, self.dim, 1, dtype=int) == axis
                step = step.astype(int) * direction
                ax_list.append(step)
            moves.append(ax_list)
        # Save inverse of the solid wall map for fluid map
        self.wall_map = ~solid
        self.moves = np.asarray(moves)

    def check_wall(self, walkers, move):
        r'''
        The walkers are an array of coordinates of the image,
        the wall map is a boolean map of the image rolled in each direction.
        directions is an array referring to the movement up or down an axis
        and is used to increment the walker coordinates if a wall is not met

        Parameters
        ----------
        walkers: ndarray of int and shape [nw, dim]
            the current coordinates of the walkers
        move: ndarray of int and shape [nw, dim]
            the vector of the next move to be made by the walker
        inds: array of int and shape [nw]
            the index of the wall map corresponding to the move vector
        '''
        next_move = walkers + move
        if self.dim == 2:
            move_ok = self.wall_map[next_move[:, 0],
                                    next_move[:, 1]]
        elif self.dim == 3:
            move_ok = self.wall_map[next_move[:, 0],
                                    next_move[:, 1],
                                    next_move[:, 2]]
        # Cancel moves that hit walls - effectively walker travels half way
        # across, hits a wall, bounces back and results in net zero movement
        if np.any(~move_ok):
            move[~move_ok] = 0
        return move

    def check_edge(self, walkers, axis, move, real):
        r'''
        Check to see if next move passes out of the domain
        If so, zero walker move and update the real velocity direction.
        Walker has remained stationary in the small image but tranisioned
        between real and reflected domains.
        Parameters
        ----------
        walkers: ndarray of int and shape [nw, dim]
            the current coordinates of the walkers
        move: ndarray of int and shape [nw, dim]
            the vector of the next move to be made by the walker
        inds: array of int and shape [nw]
            the index of the wall map corresponding to the move vector
        '''
        next_move = walkers + move
        move_real = move.copy()
        # Divide walkers into two groups, those moving postive and negative
        # Check lower edge
        axis = axis.flatten()
        w_id = np.arange(len(walkers))
        shift = np.zeros_like(axis)
        # Check lower edge
        l_hit = next_move[w_id, axis] < 0
        shift[l_hit] = -1
        # Check upper edge
        u_hit = next_move[w_id, axis] >= self.shape[axis]
        shift[u_hit] = 1
        # Combine again and update arrays
        hit = np.logical_or(l_hit, u_hit)

        if np.any(hit) > 0:
            ax = axis[hit]
            real[hit, ax] *= -1
            # walker in the original image stays stationary
            move[hit, ax] = 0
            # walker in the real image passes through an interface between
            # original and flipped along the axis of travel
            # the transition step is reversed as it the reality of travel
            # both cancel to make the real walker follow the initial move
            move_real[hit, ax] *= -1
        return move, move_real, real

    def _get_starts(self, same_start=False):
        r'''
        Start walkers in the pore space at random location
        same_start starts all the walkers at the same spot if True and at
        different ones if False
        Parameters
        ----------
        same_start: bool
            determines whether to start all the walkers at the same coordinate
        '''
        if not same_start:
            walkers = self._rand_start(self.im, num=self.nw)
        else:
            w = self._rand_start(self.im, num=1).flatten()
            walkers = np.tile(w, (self.nw, 1))
        # Start the real walkers in the middle of the big image
        reflection = np.ones_like(walkers)*int(self.offset)
        walkers_real = self._transform_coord(walkers, reflection)
        return walkers, walkers_real

#   Uncomment the line below to profile the run method
#    @do_profile(follow=[_get_wall_map, check_wall, check_edge])
    def run(self, nt=1000, nw=1, same_start=False):
        r'''
        Main run loop over nt timesteps and nw walkers.
        same_start starts all the walkers at the same spot if True and at
        different ones if False.

        Parameters
        ----------
        nt: int (default = 1000)
            the number of timesteps to run the simulation for
        nw: int (default = 1)
            he vector of the next move to be made by the walker
        same_start: bool
            determines whether to start all the walkers at the same coordinate
        debug_mode: string (default None) options ('save', 'load')
            save: saves the walker starts, and movement vectors
            load: loads the saved info enabling the same random walk to be
                  run multiple times which can be useful for debug
        '''
        self.nt = int(nt)
        self.nw = int(nw)
        # Get starts
        walkers, walkers_real = self._get_starts(same_start)
        # Save starts
        self.start = walkers.copy()
        self.start_real = walkers_real.copy()
        # Array to keep track of whether the walker is travelling in a real
        # or reflected image in each axis
        # Offsetting the walker start positions in the real image an odd
        # Number of times starts them in a reflected image
        real = np.ones_like(walkers)
        if self.offset % 2 == 1:
            real *= -1
        if self.seed:
            # Generate a seed for each timestep
            np.random.seed(1)
            seeds = np.random.randint(0, self.nw, self.nt)
        real_coords = np.ndarray([self.nt, self.nw, self.dim], dtype=int)
        for t in tqdm(range(nt), desc='Running Walk'):
            # Random velocity update
            # Randomly select an axis to move along for each walker
            if self.seed:
                np.random.seed(seeds[t])
            ax = np.random.randint(0, self.dim, self.nw)
            # Randomly select a direction positive = 1, negative = 0 index
            if self.seed:
                np.random.seed(seeds[-t])
            pn = np.random.randint(0, 2, self.nw)
            # Get the movement
            m = self.moves[ax, pn]
            # Reflected velocity (if edge is hit)
            m, mr, real = self.check_edge(walkers, ax, m, real)
            # Check for hitting walls
            mw = self.check_wall(walkers, m)
            # Reflected velocity in real direction
            walkers_real += mw*real
            real_coords[t] = walkers_real.copy()
            walkers += mw

        self.real = real
        self.real_coords = real_coords
        self.walkers = walkers
        self.walkers_real = walkers_real

    def calc_msd(self):
        r'''
        Calculate the mean square displacement
        '''
        disp = self.real_coords[:, :, :] - self.real_coords[0, :, :]
        self.axial_sq_disp = disp**2
        self.sq_disp = np.sum(disp**2, axis=2)
        self.msd = np.mean(self.sq_disp, axis=1)
        self.axial_msd = np.mean(self.axial_sq_disp, axis=1)

    def _add_linear_plot(self, x, y):
        r'''
        Helper method to add a line to the msd plot
        '''
        a, res, _, _ = np.linalg.lstsq(x, y)
        from scipy.stats import linregress
        [slope,
         intercept,
         r_value,
         p_value,
         std_err] = linregress(y, (a*x).flatten())
        rsq = r_value**2
        label = ('Tau: ' + str(np.around(1/a[0], 3)) +
                 ', R^2: ' + str(np.around(rsq, 3)))
        print(label)
        plt.plot(a*x, '--', label=label)

    def plot_msd(self):
        r'''
        Plot the mean square displacement for all walkers vs timestep
        And include a least squares regression fit.
        '''
        self.calc_msd()
        plt.figure()
        plt.plot(self.msd, '-', label='msd')
        x = np.arange(0, self.nt, 1)[:, np.newaxis]
        print('#'*30)
        print('Mean Square Displacement Data:')
        self._add_linear_plot(x, self.msd)
        for ax in range(self.dim):
            print('Axis ' + str(ax) + ' Square Displacement Data:')
            data = self.axial_msd[:, ax]*self.dim
            plt.plot(data, '-', label='axis '+str(ax))
            self._add_linear_plot(x, data)
        plt.legend()

    def _check_big_bounds(self):
        r'''
        Helper function to check whether the big image is big enough to show
        all the walks
        '''
        big_shape = np.asarray(np.shape(self.im_big))
        ok = True
        if np.sum(self.real_coords < 0):
            ok = False
        if np.sum(self.real_coords - big_shape >= 0):
            ok = False
        return ok

    def export_walk(self, image=None, path=None, sub='data', prefix='rw_',
                    stride=10):
        r'''
        Export big image to vti and walker coords to vtu every stride number of
        steps

        Parameters
        ----------
        image: ndarray of int size[im*self.offset*2]
            the walkers in the big image
        path: string (default = None)
            the filepath to save the data, defaults to current working dir
        prefix: string (default = 'rw_)
            a string prefix for all the data
        stride: int (default = 10)
            used to export the coordinates every number of strides
        '''
        # This function may have been called with plot function with a pre-
        # populated image of a subset of walkers
        # if not then export all walkers
        if image is None:
            image = self._fill_im_big()
        if self.dim == 2:
            image = image[:, :, np.newaxis]
        if path is None:
            path = os.getcwd()
        if sub is not None:
            subdir = os.path.join(path, sub)
            # if it doesn't exist, make it
            if not os.path.exists(subdir):
                os.makedirs(subdir)
            path = subdir
        im_fp = os.path.join(path, prefix+'image')
        ps.export.evtk.imageToVTK(im_fp,
                                  cellData={'pore_space': image})
        zf = np.int(np.ceil(np.log10(self.nt*10)))
        time_data = np.ascontiguousarray(np.zeros(self.nw, dtype=int))
        if self.dim == 2:
            z_coords = np.ascontiguousarray(np.ones(self.nw, dtype=int))
        coords = self.real_coords
        for t in np.arange(0, self.nt, stride, dtype=int):
            x_coords = np.ascontiguousarray(coords[t, :, 0])
            y_coords = np.ascontiguousarray(coords[t, :, 1])
            if self.dim == 3:
                z_coords = np.ascontiguousarray(coords[t, :, 2])
            wc_fp = os.path.join(path, prefix+'coords_'+str(t).zfill(zf))
            ps.export.evtk.pointsToVTK(path=wc_fp,
                                       x=x_coords,
                                       y=y_coords,
                                       z=z_coords,
                                       data={'time': time_data})
            time_data += 1

    def _fill_im_big(self, w_id=None, data='t'):
        r'''
        Fill up a copy of the big image with walker data.
        Move untrodden pore space to index -1 and solid to -2

        Parameters
        ----------
        w_id: array of int of any length (default = None)
            the indices of the walkers to plot. If None then all are shown
        data: string (options are 't' or 'w')
            t fills image with timestep, w fills image with walker index
        '''
        # Check to see whether im_big exists, if not then make it
        if not hasattr(self, 'im_big'):
            self.im_big = self._build_big_image(self.offset*2)

        big_im = self.im_big.copy().astype(int)
        big_im -= 2
        if w_id is None:
            w_id = np.arange(0, self.nw, 1, dtype=int)
        else:
            w_id = np.array([w_id])
        indices = np.indices(np.shape(self.real_coords))
        coords = self.real_coords
        if data == 't':
            # Get timestep indices
            d = indices[0, :, w_id, 0].T
        else:
            # Get walker indices
            d = indices[1, :, w_id, 0].T
        if self.dim == 3:
            big_im[coords[:, w_id, 0],
                   coords[:, w_id, 1],
                   coords[:, w_id, 2]] = d
        else:
            big_im[coords[:, w_id, 0],
                   coords[:, w_id, 1]] = d

        return big_im

    def plot_walk(self, w_id=None, slice_ind=None, data='t', export=False,
                  export_stride=10):
        r'''
        Plot the walker paths in the big image. If 3d show a slice along the
        last axis at the slice index slice_ind. For 3d walks, a better option
        for viewing results is to export and view files in paraview.

        Parameters
        ----------
        w_id: array of int and any length (default = None)
            the indices of the walkers to plot. If None then all are shown
        slice_ind: int (default = None)
            the index of the slice to take along the last axis if image is 3d
            If None then the middle slice is taken
        data: string (options are 't' or 'w')
            t fills image with timestep, w fills image with walker index
        export: bool (default = False)
            Determines whether to export the image and walker steps to vti and
            vtu, respectively. Saves a bit of time filling image up again
            separately, if export data is required. Saves in cwd.
        stride: int (default = 10)
            used if export to output the coordinates every number of strides

        '''
        self.im_big = self._build_big_image(self.offset*2)
        if self._check_big_bounds():
            big_im = self._fill_im_big(w_id=w_id, data=data).astype(int)
            plt.figure()
            if export:
                self.export_walk(image=big_im, stride=export_stride)
            if self.dim == 3:
                if slice_ind is None:
                    slice_ind = int(np.shape(big_im)[2]/2)
                big_im = big_im[:, :, slice_ind]
            masked_array = np.ma.masked_where(big_im == self.solid_value-2,
                                              big_im)
            cmap = matplotlib.cm.brg
            cmap.set_bad(color='black')
            plt.imshow(masked_array, cmap=cmap)

        else:
            print('Walk exceeds big image size! consider reducing nt ' +
                  'or increasing the starting offset')
