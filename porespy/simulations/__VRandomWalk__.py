# -*- coding: utf-8 -*-
"""
@author: Tom Tranter

Random Walker Code
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import porespy as ps

plt.close('all')

# Code for profiling stats
# uncomment decorator on the run method to switch profiling on

try:
    from line_profiler import LineProfiler

    def do_profile(follow=[]):
        def inner(func):
            def profiled_func(*args, **kwargs):
                try:
                    profiler = LineProfiler()
                    profiler.add_function(func)
                    for f in follow:
                        profiler.add_function(f)
                    profiler.enable_by_count()
                    return func(*args, **kwargs)
                finally:
                    profiler.print_stats()
            return profiled_func
        return inner

except ImportError:
    def do_profile(follow=[]):
        "Helpful if you accidentally leave in production!"
        def inner(func):
            def nothing(*args, **kwargs):
                return func(*args, **kwargs)
            return nothing
        return inner


class VRandomWalk():

    def __init__(self, image, copies=None):
        r'''
        Get image info and make a bigger periodically flipped image for viz
        '''
        self.im = image
        self.shape = np.array(np.shape(self.im))
        self.dim = len(self.shape)
        if copies is None:
            copies = 4
        self.copies = copies
        self.im_big = self._build_big_image(self.copies)
        self.solid_value = 0

    def _build_big_image(self, num_copies):
        r'''
        Build the big image by flipping and stacking along each axis a number
        of times
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
        Get a single start point in the pore space of the image
        '''
        inds = np.argwhere(image != self.solid_value)
        choice = np.random.choice(np.arange(0, len(inds), 1),
                                  num,
                                  replace=False)
        return inds[choice]

    def _setup_walk(self):
        r'''
        Initialize variables for this walk
        '''
        # Main data array - the walkers coordinates
        self.coords = np.ndarray([self.nt, self.nw, self.dim], dtype=int)
        self.start = self._find_start()
        # The origin of the image the walker is travalling in.
        # When an edge is hit this increments and the walker is now
        # travelling in a real or reflected version of the image.
        # Any hit of an edge switches the reality
        self.origin = self.start.copy()
        self.origin.fill(0)
        # Array to keep track of whether the walker is travelling in a real
        # or reflected image in each axis
        self.real = self.start.copy()
        self.real.fill(1)

    def _get_wall_map(self, image):
        r'''
        Function takes an image and rolls it back and forth on each axis saving
        the results in a wall_map. This is referred to later when random walker
        moves in a particular direction to detect where the walls are.
        '''
        # Make boolean map where solid is True
        solid = image.copy() == self.solid_value
        solid = solid.astype(bool)
        wall_dim = list(self.shape) + [self.dim*2]
        wall_map = np.zeros(wall_dim, dtype=bool)
        index = 0
        moves = []
        indices = []
        for axis in range(self.dim):
            ax_list = []
            for direction in [-1, 1]:
                # Roll the image and get the back in the original shape
                temp = np.roll(solid, -direction, axis)
                if self.dim == 2:
                    wall_map[:, :, index] = temp
                else:
                    wall_map[:, :, :, index] = temp
                # Store the direction of the step in an array for later use
                step = np.arange(0, self.dim, 1, dtype=int) == axis
                step = step.astype(int) * direction
                ax_list.append(step)
                index += 1
            moves.append(ax_list)
            indices.append([index-2, index-1])
        moves = np.asarray(moves)
        indices = np.asarray(indices)
        # Return inverse of wall map for fluid map
        return ~wall_map, moves, indices

    def check_wall(self, walkers, move, inds, wall_map):
        r'''
        The walkers are an array of coordinates of the image,
        the wall map is a boolean map of the image rolled in each direction.
        directions is an array referring to the movement up or down an axis
        and is used to increment the walker coordinates if a wall is not met
        '''
        if self.dim == 2:
            move_ok = wall_map[walkers[:, 0],
                               walkers[:, 1],
                               inds]
        elif self.dim == 3:
            move_ok = wall_map[walkers[:, 0],
                               walkers[:, 1],
                               walkers[:, 2],
                               inds]
        # Cancel moves that hit walls - effectively walker travels half way
        # across, hits a wall, bounces back and results in net zero movement
        if np.any(~move_ok):
            move[~move_ok] = 0
        return move

    def check_edge(self, walkers, axis, move, origin, real):
        r'''
        Check to see if next move passes out of the domain
        If so, zero walker move and update the real velocity direction.
        Walker has remained stationary in the small image but tranisioned
        between real and reflected domains.
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
            origin[hit, ax] *= -1
            origin[hit, ax] += shift[hit]
            real[hit, ax] *= -1
            # walker in the original image stays stationary
            move[hit, ax] = 0
            # walker in the real image passes through an interface between
            # original and flipped along the axis of travel
            # the transition step is reversed as it the reality of travel
            # both cancel to make the real walker follow the initial move
            move_real[hit, ax] *= -1
        return move, move_real, origin, real

    def _get_starts(self, same_start=False):
        r'''
        Start walkers in the pore space at random location
        same_start starts all the walkers at the same spot if True and at
        different ones if False
        '''
        if not same_start:
            walkers = self._rand_start(self.im, num=self.nw)
        else:
            w = self._rand_start(self.im, num=1).flatten()
            walkers = np.tile(w, (self.nw, 1))
        self.start = walkers.copy()
        # Start the real walkers in the middle of the big image
        walkers_real = walkers + self.shape*int(self.copies/2)
        self.start_real = walkers_real.copy()
        return walkers, walkers_real

#    @do_profile(follow=[_get_wall_map, check_wall, check_edge])
    def run(self, nt, nw, same_start=False):
        r'''
        Main run loop over nt timesteps and nw walkers.
        same_start starts all the walkers at the same spot if True and at
        different ones if False.
        '''
        self.nt = int(nt)
        self.nw = int(nw)
        walkers, walkers_real = self._get_starts(same_start)
        wall_map, moves, indices = self._get_wall_map(self.im)
        # The origin of the image the walker is travalling in
        # When an edge is hit this increments and the walker is now
        # travelling in a real or reflected version of the image
        origin = walkers.copy()
        origin.fill(0)
        # Array to keep track of whether the walker is travelling in a real
        # or reflected image in each axis
        real = walkers.copy()
        real.fill(1)
        # Set solid to -1 for vis
        small_im = self.im.copy()
        small_im[small_im == 1] *= -1
        big_im = self.im_big.copy()
        big_im[big_im == 1] *= -1
        real_coords = np.ndarray([self.nt, self.nw, self.dim], dtype=int)
        for t in range(nt):
            # Random velocity update
            # Randomly select an axis to move along for each walker
            ax = np.random.randint(0, self.dim, self.nw)
            # Randomly select a direction positive = 1, negative = 0 index
            pn = np.random.randint(0, 2, self.nw)
            # Get the movement
            m = moves[ax, pn]
            # Get the index of the wall map
            ind = indices[ax, pn]
            # Check for hitting walls
            mw = self.check_wall(walkers, m, ind, wall_map)
            # Reflected velocity (if wall is hit)
            m, mr, origin, real = self.check_edge(walkers,
                                                  ax, mw, origin, real)
            # Check for hitting walls
            # Reflected velocity in real direction
            walkers_real += mr*real
            real_coords[t] = walkers_real.copy()
            walkers += m

        self.origin = origin
        self.real = real
        self.real_coords = real_coords
        self.walkers = walkers
        self.walkers_real = walkers_real
        self.nw = nw
        self.nt = nt

    def shift_origin(self, origin):
        r'''
        Method to shift the start position of the walker based on the
        origin shifts when walls are hit
        '''
        so = origin*(self.shape)
        shifted_origin = so.astype(int)
        return shifted_origin

    def check_disp_match(self):
        r'''
        Check that the displacement matches for the two images
        '''
        sstart = self.shift_origin(self.origin)
        print('#'*30)
        # Figure out whether to calculate start from upper or lower edge
        a = (self.shape - (sstart != 0).astype(int))
        b = (self.real == -1).astype(int)
        adj_start = np.abs(a*b - self.start)
        distance = (self.walkers - adj_start - sstart)*self.real
        real_distance = (self.walkers_real - self.start_real)
        print('Square Distance Match?', np.allclose(distance**2,
                                                    real_distance**2))
        print('#'*30)

    def plot_msd(self):
        r'''
        Plot the mean square displacement for all walkers vs timestep
        And include a least squares regression fit.
        '''
        disp = self.real_coords[:, :, :] - self.real_coords[0, :, :]
        self.sq_disp = np.sum(disp**2, axis=2)
        self.msd = np.mean(self.sq_disp, axis=1)
        plt.figure()
        plt.plot(self.msd, 'k-', label='msd')
        x = np.arange(0, self.nt, 1)[:, np.newaxis]
        a, res, _, _ = np.linalg.lstsq(x, self.msd)
        R_sq = 1 - res / (self.msd.size * self.msd.var())
        label = ('Grad: ' + str(np.around(a[0], 3)) +
                 ', R^2: ' + str(np.around(R_sq[0], 3)))
        plt.plot(a*x, 'r--', label=label)
        plt.legend()
        print('Gradient:', a, 'R^2', R_sq)

    def plot_walk(self, w_id=None, slice_ind=None):
        r'''
        Plot the walker paths in the big image.
        If 3d show a slice along the last axis.
        w_id is the integer index of the walker to plot. If None all are shown.
        slice_ind defaults to the middle of the image.
        '''
        big_im = self.im_big.copy()
        big_im -= 2
        if self.dim == 3 and slice_ind is None:
            slice_ind = int(np.shape(big_im)[2]/2)
        if w_id is None:
            w_id = np.arange(0, self.nw, 1, dtype=int)
        else:
            w_id = np.array([w_id])
        for w in w_id:
            coords = self.real_coords[:, w, :]
            ts = np.arange(0, self.nt, 1, dtype=int)
            try:
                if self.dim == 3:
                    big_im[coords[:, 0], coords[:, 1], coords[:, 2]] = ts
                else:
                    big_im[coords[:, 0], coords[:, 1]] = ts
            except IndexError:
                print('Walk exceeds big image size, consider reducing nt')
        self.big_im = big_im.copy()
        plt.figure()
        if self.dim == 3:
            big_im = big_im[:, :, slice_ind]
        masked_array = np.ma.masked_where(big_im == self.solid_value-2, big_im)
        cmap = matplotlib.cm.brg
        cmap.set_bad(color='black')
        plt.imshow(masked_array, cmap=cmap)


if __name__ == "__main__":
    if 1 == 1:
        # Load tau test image
        im = 1 - ps.data.tau()
    else:
        im = ps.generators.blobs(100).astype(int)

    # Number of time steps and walkers
    num_t = 10000
    num_w = 1
    # Track time of simulation
    st = time.time()
    rw = VRandomWalk(im)
    rw.run(num_t, num_w, same_start=False)
    # Plot mean square displacement
    rw.plot_msd()
    # Plot the longest walk
    rw.plot_walk(w_id=np.argmax(rw.sq_disp[-1, :]))
    # Plot all the walks
#    rw.plot_walk()
    print('sim time', time.time()-st)
