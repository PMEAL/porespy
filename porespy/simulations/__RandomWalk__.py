# -*- coding: utf-8 -*-
"""
@author: Tom Tranter, Matt Lam, Matt Kok, Jeff Gostick
PMEAL lab, University of Waterloo, Ontario, Canada

Random Walker Code
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import porespy as ps
from porespy.tools.__funcs__ import do_profile
from tqdm import tqdm
import os
import time
import csv
from concurrent.futures import ProcessPoolExecutor
import gc
cmap = matplotlib.cm.viridis


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
    when plotted over time is equal to 1/tau, the tortuosity factor.
    The image data and walker co-ordinates can be exported for visualization
    in paraview.
    A simple 2d slice can also be viewed directly using matplotlib.
    Currently walkers do not travel along diagonals.
    Running walkers in parallel by setting num_proc is possible to speed up
    calculations as wach walker path is completely independent of the others.
    '''

    def __init__(self, image, seed=False):
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
        >>> im = ps.generators.blobs([100, 100])
        >>> rw = ps.simulations.RandomWalk(im)
        >>> rw.run(nt=1000, nw=100)
        '''
        self.im = image
        self.shape = np.array(np.shape(self.im))
        self.dim = len(self.shape)
        self.solid_value = 0
        self.seed = seed
        self._get_wall_map(self.im)
        self.data = {}

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
        return ~move_ok

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

        if np.any(hit):
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
        return walkers

    def _run_walk(self, walkers):
        r'''
        Run the walk in self contained way to enable parallel processing for
        batches of walkers
        '''
        # Number of walkers in this batch
        nw = len(walkers)
        walkers = np.asarray(walkers)
        wr = walkers.copy()
        # Array to keep track of whether the walker is travelling in a real
        # or reflected image in each axis
        real = np.ones_like(walkers)
#        real_coords = np.ndarray([self.nt, nw, self.dim], dtype=int)
        real_coords = []
        for t in range(self.nt):
            # Random velocity update
            # Randomly select an axis to move along for each walker
            if self.seed:
                np.random.seed(self.seeds[t])
            ax = np.random.randint(0, self.dim, nw)
            # Randomly select a direction positive = 1, negative = 0 index
            if self.seed:
                np.random.seed(self.seeds[-t])
            pn = np.random.randint(0, 2, nw)
            # Get the movement
            m = self.moves[ax, pn]
            # Reflected velocity (if edge is hit)
            m, mr, real = self.check_edge(walkers, ax, m, real)
            # Check for wall hits and zero both movements
            # Cancel moves that hit walls - effectively walker travels half way
            # across, hits a wall, bounces back and results in net zero move
            wall_hit = self.check_wall(walkers, m)
            if np.any(wall_hit):
                m[wall_hit] = 0
                mr[wall_hit] = 0
            # Reflected velocity in real direction
            wr += mr*real
            walkers += m
            if t % self.stride == 0:
                real_coords.append(wr.copy())
        return real_coords

    # Uncomment the line below to profile the run method
    # Only works for single process
#    @do_profile(follow=[_run_walk, check_wall, check_edge])
    def run(self, nt=1000, nw=1, same_start=False, stride=1, num_proc=None):
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
        stride: int
            save coordinate data every stride number of timesteps
        num_proc: int (default None - uses half available)
            number of concurrent processes to start running
        '''
        self.nt = int(nt)
        self.nw = int(nw)
        self.stride = stride
        record_t = int(self.nt/stride)
        # Get starts
        walkers = self._get_starts(same_start)
        if self.seed:
            # Generate a seed for each timestep
            np.random.seed(1)
            self.seeds = np.random.randint(0, self.nw, self.nt)
        real_coords = np.ndarray([record_t, self.nw, self.dim], dtype=int)
        # Default to run in parallel with half the number of available procs
        if num_proc is None:
            num_proc = int(os.cpu_count()/2)
        if num_proc > 1 and self.nw >= num_proc:
            # Run in parallel over multiple CPUs
            batches = self._chunk_walkers(walkers, num_proc)
            pool = ProcessPoolExecutor(max_workers=num_proc)
            mapped_coords = list(pool.map(self._run_walk, batches))
            pool.shutdown()
            del pool
            # Put coords back together
            si = 0
            for mc in mapped_coords:
                mnw = np.shape(mc)[1]
                real_coords[:, si: si + mnw, :] = mc.copy()
                si = si + mnw
        else:
            # Run in serial
            real_coords = np.asarray(self._run_walk(walkers.tolist()))

        self.real_coords = real_coords

    def _chunk_walkers(self, walkers, num_chunks):
        r'''
        Helper function to divide the walkers into batches for pool-processing
        '''
        num_walkers = len(walkers)
        n = int(np.floor(num_walkers / num_chunks))
        l = walkers.tolist()
        chunks = [l[i:i + n] for i in range(0, len(l), n)]
        return chunks

    def calc_msd(self):
        r'''
        Calculate the mean square displacement
        '''
        disp = self.real_coords[:, :, :] - self.real_coords[0, :, :]
        self.axial_sq_disp = disp**2
        self.sq_disp = np.sum(disp**2, axis=2)
        self.msd = np.mean(self.sq_disp, axis=1)
        self.axial_msd = np.mean(self.axial_sq_disp, axis=1)

    def _add_linear_plot(self, x, y, descriptor=None, color='k'):
        r'''
        Helper method to add a line to the msd plot
        '''
        a, res, _, _ = np.linalg.lstsq(x, y)
        tau = 1/a[0]
        SStot = np.sum((y - y.mean())**2)
        rsq = 1 - (np.sum(res)/SStot)
        label = ('Tau: ' + str(np.around(tau, 3)) +
                 ', R^2: ' + str(np.around(rsq, 3)))
        print(label)
        plt.plot(x, a[0]*x, color+'--', label=label)
        self.data[descriptor + '_tau'] = tau
        self.data[descriptor + '_rsq'] = rsq

    def plot_msd(self):
        r'''
        Plot the mean square displacement for all walkers vs timestep
        And include a least squares regression fit.
        '''
        self.calc_msd()
        self.data = {}
        fig, ax = plt.subplots(figsize=[6, 6])
        ax.set(aspect=1, xlim=(0, self.nt), ylim=(0, self.nt))
        x = np.arange(0, self.nt, self.stride)[:, np.newaxis]
        plt.plot(x, self.msd, 'k-', label='msd')
        print('#'*30)
        print('Square Displacement:')
        self._add_linear_plot(x, self.msd, 'Mean', color='k')
        colors = ['r', 'g', 'b']
        for ax in range(self.dim):
            print('Axis ' + str(ax) + ' Square Displacement Data:')
            data = self.axial_msd[:, ax]*self.dim
            plt.plot(x, data, colors[ax]+'-', label='asd '+str(ax))
            self._add_linear_plot(x, data, 'axis_'+str(ax), colors[ax])
        plt.legend()

    def _check_big_bounds(self):
        r'''
        Helper function to check the maximum displacement and return the number
        of image copies needed to build a big image large enough to display all
        the walks
        '''
        max_disp = np.max(np.max(np.abs(self.real_coords), axis=0), axis=0)+1
        num_domains = np.ceil(max_disp / self.shape)
        num_copies = int(np.max(num_domains))-1
        return num_copies

    def export_walk(self, image=None, path=None, sub='data', prefix='rw_',
                    sample=1):
        r'''
        Export big image to vti and walker coords to vtu

        Parameters
        ----------
        image: ndarray of int size (Default is None)
            Can be used to export verisons of the image
        path: string (default = None)
            the filepath to save the data, defaults to current working dir
        prefix: string (default = 'rw_)
            a string prefix for all the data
        sample: int (default = 1)
            used to down-sample the number of walkers to export by this factor
        '''
        if path is None:
            path = os.getcwd()
        if sub is not None:
            subdir = os.path.join(path, sub)
            # if it doesn't exist, make it
            if not os.path.exists(subdir):
                os.makedirs(subdir)
            path = subdir
        if image is not None:
            if len(np.shape(image)) == 2:
                image = image[:, :, np.newaxis]
            im_fp = os.path.join(path, prefix+'image')
            ps.export.evtk.imageToVTK(im_fp,
                                      cellData={'image_data': image})
        # number of zeros to fill the file index
        zf = np.int(np.ceil(np.log10(self.nt*10)))
        w_id = np.arange(0, self.nw, sample)
        nw = len(w_id)
        time_data = np.ascontiguousarray(np.zeros(nw, dtype=int))
        if self.dim == 2:
            z_coords = np.ascontiguousarray(np.ones(nw, dtype=int))
        coords = self.real_coords
        for t in range(np.shape(coords)[0]):
            st = self.stride*t
            time_data.fill(st)
            x_coords = np.ascontiguousarray(coords[t, w_id, 0])
            y_coords = np.ascontiguousarray(coords[t, w_id, 1])
            if self.dim == 3:
                z_coords = np.ascontiguousarray(coords[t, w_id, 2])
            wc_fp = os.path.join(path, prefix+'coords_'+str(st).zfill(zf))
            ps.export.evtk.pointsToVTK(path=wc_fp,
                                       x=x_coords,
                                       y=y_coords,
                                       z=z_coords,
                                       data={'time': time_data})

    def _build_big_image(self, num_copies=0):
        r'''
        Build the big image by flipping and stacking along each axis a number
        of times on both sides of the image to keep the original in the center

        Parameters
        ----------
        num_copies: int
            the number of times to copy the image along each axis
        '''
        big_im = self.im.copy()
        func = [np.vstack, np.hstack, np.dstack]
        temp_im = self.im.copy()
        for ax in tqdm(range(self.dim), desc='building big image'):
            flip_im = np.flip(temp_im, ax)
            for c in range(num_copies):
                if c % 2 == 0:
                    # Place one flipped copy either side
                    big_im = func[ax]((big_im, flip_im))
                    big_im = func[ax]((flip_im, big_im))
                else:
                    # Place one original copy either side
                    big_im = func[ax]((big_im, temp_im))
                    big_im = func[ax]((temp_im, big_im))
            # Update image to copy for next axis
            temp_im = big_im.copy()
        return big_im

    def _fill_im_big(self, w_id=None, t_id=None, data='t'):
        r'''
        Fill up a copy of the big image with walker data.
        Move untrodden pore space to index -1 and solid to -2

        Parameters
        ----------
        w_id: array of int of max length num_walkers (default = None)
            the indices of the walkers to plot. If None then all are shown
        t_id: array of int of max_length num_timesteps/stride (default = None)
            the indices of the timesteps to plot. If None then all are shown
        data: string (options are 't' or 'w', other)
            t fills image with timestep, w fills image with walker index,
            any other value places a 1 signifiying that a walker is at this
            coordinate.
        '''
        offset = self._check_big_bounds()
        if not hasattr(self, 'im_big'):
            self.im_big = self._build_big_image(offset)
        big_im = self.im_big.copy().astype(int)
        big_im -= 2
        # Number of stored timesteps, walkers dimensions
        [nst, nsw, nd] = np.shape(self.real_coords)
        if w_id is None:
            w_id = np.arange(0, nsw, 1, dtype=int)
        else:
            w_id = np.array([w_id])
        if t_id is None:
            t_id = np.arange(0, nst, 1, dtype=int)
        else:
            t_id = np.array([t_id])
        indices = np.indices(np.shape(self.real_coords))
        coords = self.real_coords + offset*self.shape
        if data == 't':
            # Get timestep indices
            d = indices[0, :, w_id, 0].T
        elif data == 'w':
            # Get walker indices
            d = indices[1, :, w_id, 0].T
        else:
            # Fill with 1 where there is a walker
            d = np.ones_like(indices[0, :, w_id, 0].T, dtype=int)
        if self.dim == 3:
            big_im[coords[:, w_id, 0][t_id],
                   coords[:, w_id, 1][t_id],
                   coords[:, w_id, 2][t_id]] = d[t_id]
        else:
            big_im[coords[:, w_id, 0][t_id],
                   coords[:, w_id, 1][t_id]] = d[t_id]

        return big_im

    def _save_fig(self, figname='test.png', dpi=600):
        r'''
        Wrapper for saving figure in journal format
        '''
        plt.figaspect(1)
        plt.savefig(filename=figname, dpi=dpi, facecolor='w', edgecolor='w',
                    format='png', bbox_inches='tight', pad_inches=0.0)

    def plot_walk_2d(self, w_id=None, data='t', check_solid=False):
        r'''
        Plot the walker paths in a big image that shows real and reflected
        domains. The starts are temporarily shifted and then put back

        Parameters
        ----------
        w_id: array of int and any length (default = None)
            the indices of the walkers to plot. If None then all are shown
        data: string (options are 't' or 'w')
            t fills image with timestep, w fills image with walker index

        '''
        if self.dim == 3:
            print('Method is not implemented for 3d images')
            print('Please use export for visualizing 3d walks in paraview')
        else:
            offset = self._check_big_bounds()
            if not hasattr(self, 'im_big'):
                self.im_big = self._build_big_image(offset)
            sb = np.sum(self.im_big == self.solid_value)
            big_im = self._fill_im_big(w_id=w_id, data=data).astype(int)
            sa = np.sum(big_im == self.solid_value - 2)
            fig, ax = plt.subplots(figsize=[6, 6])
            ax.set(aspect=1)
            solid = big_im == self.solid_value-2
            solid = solid.astype(float)
            solid[np.where(solid == 0)] = np.nan
            porous = big_im == self.solid_value-1
            porous = porous.astype(float)
            porous[np.where(porous == 0)] = np.nan
            plt.imshow(big_im, cmap=cmap)
            # Make Solid Black
            plt.imshow(solid, cmap='binary', vmin=0, vmax=1)
            # Make Untouched Porous White
            plt.imshow(porous, cmap='gist_gray', vmin=0, vmax=1)
            if check_solid:
                print('Solid pixel match?', sb == sa, sb, sa)

    def axial_density_plot(self, time=None, axis=None, bins=None):
        r'''
        Plot the walker density summed along an axis at a given time

        Parameters
        ----------
        time: int
            the index in stride time. If the run method was set with a stride
            of 10 then putting time=2 will visualize the 20th timestep as this
            is the second stored time-step (after 0)
        axis: int - Used only for 3D walks
            The axis over which to sum, produces a slice in the plane normal to
            this axis.

        '''
        copies = self._check_big_bounds()
        im_bins = (copies + 1)*np.asarray(self.shape)
        t_coords = self.real_coords[time, :, :]
        if self.dim == 3:
            if axis is None:
                axis = 2
            axes = np.arange(0, self.dim, 1)
            [a, b] = axes[axes != axis]
            [na, nb] = im_bins[axes != axis]
        else:
            [a, b] = [0, 1]
            [na, nb] = im_bins
        a_coords = t_coords[:, a]
        b_coords = t_coords[:, b]
        fig, ax = plt.subplots(figsize=[6, 6])
        ax.set(aspect=1)
        if bins is None:
            bins = max([na, nb])
        plt.hist2d(a_coords, b_coords, bins=bins, cmin=1, cmap=cmap)
        plt.colorbar()

    def run_analytics(self, ws, ts, fname='analytics.csv', num_proc=None):
        r'''
        Run run a number of times saving info
        Warning - this method may take some time to complete!

        Parameters
        ----------
        ws: list
            list of number of walkers to run
        ts: int (default = 4)
            list of number of walkers to run
        fname: string
            file name - must end '.csv'
        '''
        with open(fname, 'w', newline='\n') as f:
            self.data['sim_nw'] = 0
            self.data['sim_nt'] = 0
            self.data['sim_time'] = 0
            w = csv.DictWriter(f, self.data)
            header_written = False
            for nw in ws:
                for nt in ts:
                    print('Running Analytics for:')
                    print('Number of Walkers: ' + str(nw))
                    print('Number of Timesteps: ' + str(nt))
                    start_time = time.time()
                    self.run(nt, nw, same_start=False, stride=10,
                             num_proc=num_proc)
                    sim_time = time.time() - start_time
                    print('Completed in: ' + str(sim_time))
                    self.plot_msd()
                    plt.title('Walkers: ' + str(nw) + ' Steps: ' + str(nt))
                    self.data['sim_nw'] = nw
                    self.data['sim_nt'] = nt
                    self.data['sim_time'] = sim_time
                    w = csv.DictWriter(f, self.data)
                    if not header_written:
                        w.writeheader()
                        header_written = True
                    w.writerow(self.data)
                    gc.collect()
