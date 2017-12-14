# -*- coding: utf-8 -*-
"""
@author: Tom Tranter, Matt Lam, Matt Kok, Jeff Gostick
PMEAL lab, University of Waterloo, Ontario, Canada

Random Walker Code
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolors
import porespy as ps
from porespy.tools.__funcs__ import do_profile
import os
import time
import csv
from concurrent.futures import ProcessPoolExecutor


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
            Simulation is repeatable

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
        real_coords = np.ndarray([self.nt, nw, self.dim], dtype=int)
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
            real_coords[t] = wr.copy()
            walkers += m
        return real_coords

    # Uncomment the line below to profile the run method
    # Only works for single process
#    @do_profile(follow=[_run_walk, check_wall, check_edge])
    def run(self, nt=1000, nw=1, same_start=False, num_proc=None):
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
        num_proc: int (default None - uses half available)
            number of concurrent processes to start running
        '''
        self.nt = int(nt)
        self.nw = int(nw)
        # Get starts
        walkers = self._get_starts(same_start)
        if self.seed:
            # Generate a seed for each timestep
            np.random.seed(1)
            self.seeds = np.random.randint(0, self.nw, self.nt)
        real_coords = np.ndarray([self.nt, self.nw, self.dim], dtype=int)
        # Default to run in parallel with half the number of available procs
        if num_proc is None:
            num_proc = int(os.cpu_count()/2)
        if num_proc > 1:
            # Run in parallel over multiple CPUs
            batches = self._chunk_walkers(walkers, num_proc)
            with ProcessPoolExecutor(max_workers=num_proc) as pool:
                mapped_coords = list(pool.map(self._run_walk, batches))
            # Put coords back together
            si = 0
            for mc in mapped_coords:
                mnw = np.shape(mc)[1]
                real_coords[:, si: si + mnw, :] = mc.copy()
                si = si + mnw
        else:
            # Run in serial
            real_coords = self._run_walk(walkers.tolist())

        self.real_coords = real_coords

    def _chunk_walkers(self, walkers, num_chunks):
        r'''
        Helper function to divide the walkers into batches for pool-processing
        '''
        num_walkers = len(walkers)
        n = int(np.floor(num_walkers / num_chunks))
        l = walkers.tolist()
        return [l[i:i + n] for i in range(0, len(l), n)]

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
        plt.plot(a[0]*x, color+'--', label=label)
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
        plt.plot(self.msd, 'k-', label='msd')
        x = np.arange(0, self.nt, 1)[:, np.newaxis]
        print('#'*30)
        print('Square Displacement:')
        self._add_linear_plot(x, self.msd, 'Mean', color='k')
        colors = ['r', 'g', 'b']
        for ax in range(self.dim):
            print('Axis ' + str(ax) + ' Square Displacement Data:')
            data = self.axial_msd[:, ax]*self.dim
            plt.plot(data, colors[ax]+'-', label='asd '+str(ax))
            self._add_linear_plot(x, data, 'axis_'+str(ax), colors[ax])
        plt.legend()

    def _check_big_bounds(self):
        r'''
        Helper function to check the maximum displacement and return the number
        of image copies needed to build a big image large enough to display all
        the walks
        '''
        max_disp = np.max(np.max(np.abs(self.real_coords), axis=0), axis=0)
        num_domains = np.ceil(max_disp / self.shape)
        num_copies = int(np.max(num_domains))-1
        return num_copies

    def export_walk(self, image=None, path=None, sub='data', prefix='rw_',
                    stride=10):
        r'''
        Export big image to vti and walker coords to vtu every stride number of
        steps

        Parameters
        ----------
        image: ndarray of int size (Default is None)
            Can be used to export verisons of the image
        path: string (default = None)
            the filepath to save the data, defaults to current working dir
        prefix: string (default = 'rw_)
            a string prefix for all the data
        stride: int (default = 10)
            used to export the coordinates every number of strides
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
        for ax in range(self.dim):
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

    def _fill_im_big(self, w_id=None, coords=None, data='t'):
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
        big_im = self.im_big.copy().astype(int)
        big_im -= 2
        if w_id is None:
            w_id = np.arange(0, self.nw, 1, dtype=int)
        else:
            w_id = np.array([w_id])
        indices = np.indices(np.shape(self.real_coords))
        if coords is None:
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
            self.im_big = self._build_big_image(offset)
            sb = np.sum(self.im_big == self.solid_value)
            coords = self.real_coords + self.shape*offset
            big_im = self._fill_im_big(w_id=w_id,
                                       data=data,
                                       coords=coords).astype(int)
            sa = np.sum(big_im == self.solid_value - 2)
#            fs = (np.asarray(np.shape(big_im))/200).tolist()
            fig, ax = plt.subplots(figsize=[6, 6])
            ax.set(aspect=1)
            solid = big_im == self.solid_value-2
            solid = solid.astype(float)
            solid[np.where(solid == 0)] = np.nan
            porous = big_im == self.solid_value-1
            porous = porous.astype(float)
            porous[np.where(porous == 0)] = np.nan
            cmap = matplotlib.cm.viridis
#            cmap.set_bad(color='black')
            plt.imshow(big_im, cmap=cmap)
            plt.imshow(solid, cmap='binary', vmin=0, vmax=1)
            plt.imshow(porous, cmap='gist_gray', vmin=0, vmax=1)
            if check_solid:
                print('Solid pixel match?', sb == sa, sb, sa)

    def run_analytics(self, lw=2, uw=4, lt=2, ut=4):
        r'''
        Run run a number of times saving info
        Warning - this method may take some time to complete!

        Parameters
        ----------
        lw: int (default = 2)
            the lower power of 10 to use for nummber of walkers
        uw: int (default = 4)
            the upper power of 10 to use for nummber of walkers
        lt: int (default = 2)
            the lower power of 10 to use for nummber of timesteps
        ut: int (default = 4)
            the upper power of 10 to use for nummber of timesteps
        '''
        pow_10w = np.arange(lw, uw+1, 1, dtype=int)
        pow_10t = np.arange(lt, ut+1, 1, dtype=int)
        with open('analytics.csv', 'w', newline='\n') as f:
            self.data['sim_nw'] = 0
            self.data['sim_nt'] = 0
            self.data['sim_time'] = 0
            w = csv.DictWriter(f, self.data)
            w.writeheader()
            for pw in pow_10w:
                for pt in pow_10t:
                    nw = np.power(10, pw)
                    nt = np.power(10, pt)
                    print('Running Analystics for:')
                    print('Number of Walkers: ' + str(nw))
                    print('Number of Timesteps: ' + str(nt))
                    start_time = time.time()
                    self.run(nt, nw, same_start=False)
                    sim_time = time.time() - start_time
                    print('Completed in: ' + str(sim_time))
                    self.plot_msd()
                    plt.title('Walkers: ' + str(nw) + ' Steps: ' + str(nt))
                    self.data['sim_nw'] = nw
                    self.data['sim_nt'] = nt
                    self.data['sim_time'] = sim_time
                    w = csv.DictWriter(f, self.data)
                    w.writerow(self.data)

if __name__ == "__main__":
    plt.close('all')
    image_run = 0
    if image_run == 0:
        # Open space
        im = np.ones([3, 3], dtype=int)
        fname = 'open_'
        num_t = 10000
        num_w = 10000
    elif image_run == 1:
        # Load tau test image
        im = 1 - ps.data.tau()
        fname = 'tau_'
        # Number of time steps and walkers
        num_t = 200000
        num_w = 1000
    elif image_run == 2:
        # Generate a Sierpinski carpet by tiling an image and blanking the
        # Middle tile recursively
        def tileandblank(image, n):
            if n > 0:
                n -= 1
                shape = np.asarray(np.shape(image))
                image = np.tile(image, (3, 3))
                image[shape[0]:2*shape[0], shape[1]:2*shape[1]] = 0
                image = tileandblank(image, n)
            return image

        im = np.ones([1, 1], dtype=int)
        im = tileandblank(im, 5)
        fname = 'sierpinski_'
        # Number of time steps and walkers
        num_t = 5000
        num_w = 100000
    else:
        # Do some blobs
        im = ps.generators.blobs(shape=[300, 300, 300], porosity=0.5,
                                 blobiness=[1, 2, 5]).astype(int)
        fname = 'blobs_'
        # Number of time steps and walkers
        num_t = 10000
        num_w = 100000

    # Track time of simulation
    st = time.time()
    rw = ps.simulations.RandomWalk(im, seed=False)
    rw.run(num_t, num_w, same_start=False)
    print('run time', time.time()-st)
    rw.calc_msd()
    # Plot mean square displacement
    rw.plot_msd()
    rw._save_fig(fname+'msd.png')
    if rw.dim == 2:
        # Plot the longest walk
        rw.plot_walk_2d(w_id=np.argmax(rw.sq_disp[-1, :]), data='w')
        dpi = 600
        rw._save_fig(fname+'longest.png', dpi=dpi)
        # Plot all the walks
        rw.plot_walk_2d(check_solid=True)
        rw._save_fig(fname+'all.png', dpi=dpi)
    else:
        # export to paraview
        rw.export_walk(image=rw.im, stride=1)
#rw.run_analytics(lw=2, uw=3, lt=2, ut=6)
