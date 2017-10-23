import numpy as np
import matplotlib.pyplot as plt
from porespy.metrics import porosity
from numba import jit
from mpl_toolkits.mplot3d import Axes3D
from collections import namedtuple


class RandomWalk:
    r"""
    This class generates objects for performing random walk simulations
    on.

    Parameters
    ----------
    im: ndarray of bool
        2D or 3D image
    walkers: int (default = 1000)
        The number of walks to initially perform
    max_steps: int (default = 5000)
        The number of steps to attempt per walk
    stride: int (default = 10)
        The number of steps taken between each coordinate in path_data
    st_frac: float (default = 0.2)
        This fraction of the image (in the centre) is searched for a valid
        starting point for each walk

    Examples
    --------

    Creating a RandomWalk object:

    >>> import porespy as ps
    >>> im = ps.generators.blobs(300)
    >>> rw = ps.simulations.RandomWalk(im, walkers=500, max_steps=3000)

    """

    @jit
    def walk(self, start_point, max_steps, stride=1):
        r"""
        This function performs a single random walk through porous image. It
        returns an array containing the walker path in the image, and the
        walker path in free space.

        Parameters
        ----------
        start_point: array_like
           A sequence containing the indices of a valid starting point. If
           the image is 2D, the first coordinate needs to be zero
        max_steps: int
            The number of steps to attempt per walk
        stride: int
            Number of steps taken between each returned coordinate

        Returns
        --------
        paths: namedtuple
            A tuple containing 2 arrays showing the paths taken by the walker:
            path.pore_space and path.free_space

        Notes
        ------
        This function only performs a single walk through the image. Use
        RandomWalk.-perform_walks() to add more
        """

        ndim = self._ndim
        im = self.im
        if ndim == 3:
            directions = 6
        elif ndim == 2:
            directions = 4
        (z, y, x) = start_point
        if not im[z, y, x]:
            raise ValueError('invalid starting point: not a pore')
        z_max, y_max, x_max = self._shape
        z_free, y_free, x_free = z, y, x
        coords = np.ones((max_steps+1, 3), dtype=int) * (-1)
        free_coords = np.ones((max_steps+1, 3), dtype=int) * (-1)
        coords[0, :] = [z, y, x]
        free_coords[0, :] = [z_free, y_free, x_free]
        steps = 0
        # begin walk
        for step in range(1, max_steps+1):
            x_step, y_step, z_step = 0, 0, 0
            direction = np.random.randint(0, directions)
            if direction == 0:
                x_step += 1
            elif direction == 1:
                x_step -= 1
            elif direction == 2:
                y_step += 1
            elif direction == 3:
                y_step -= 1
            elif direction == 4:
                z_step += 1
            elif direction == 5:
                z_step -= 1
            # checks to make sure image does not go out of bounds
            if x_free+x_step < 0 or y_free+y_step < 0 or z_free+z_step < 0:
                break
            elif (x_free+x_step >= x_max or y_free+y_step >= y_max or
                    z_free+z_step >= z_max):
                break
            else:
                x_free += x_step
                y_free += y_step
                z_free += z_step
            if x+x_step < 0 or y+y_step < 0 or z+z_step < 0:
                break
            elif x+x_step >= x_max or y+y_step >= y_max or z+z_step >= z_max:
                break
            # checks if the step leads to a pore in image
            elif im[z+z_step, y+y_step, x+x_step]:
                x += x_step
                y += y_step
                z += z_step
            steps += 1
            coords[step] = [z, y, x]
            free_coords[step] = [z_free, y_free, x_free]
        path = namedtuple('path', ('pore_space', 'free_space'))
        paths = path(coords[::stride, :], free_coords[::stride, :])
        return paths

    def find_start_point(self, start_frac):
        r"""
        Finds a random valid start point in a porous image, searching in the
        given fraction of the image

        Parameters
        ----------
        start_frac: float
            A value between 0 and 1. Determines what fraction of the image is
            randomly searched for a starting point

        Returns
        --------
        start_point: tuple
            A tuple containing the index of a valid start point.
            If the image is 2D, start_point will have 0 as the first coord
        """

        x_r = self._x_len*start_frac
        y_r = self._y_len*start_frac
        z_r = self._z_len*start_frac
        i = 0
        while True:
            i += 1
            if i > 10000:
                print("failed to find starting point")
                return None
            x = int(self._x_len/2 - x_r/2 + np.random.randint(x_r+1))
            y = int(self._y_len/2 - y_r/2 + np.random.randint(y_r+1))
            z = int(self._z_len/2 - z_r/2 + np.random.randint(z_r+1))
            if self.im[z, y, x]:
                break
        start_point = (z, y, x)
        return start_point

    def perform_walks(self, walkers):
        r"""
        Adds the specified number of walkers to the path_data attribute

        Parameters
        ----------
        walkers: int
            The number of walks to perform. This many rows will be added
            to each of the path_data arrays (path_data.pore_space and
            path_data.free_space)

        Notes
        ------
        This function will continue to use the max_steps and stride value
        stored on the object
        """
        path_data = -1*np.ones((3, walkers, self._max_steps//self._stride+1))
        free_path_data = -1*np.ones((3, walkers,
                                     self._max_steps//self._stride+1))
        paths = namedtuple('path', ('pore_space', 'free_space'))
        for w in range(walkers):
            p = self.find_start_point(self._start_frac)
            path, free_path = self.walk(p, self._max_steps, self._stride)
            path_data[:, w, :] = np.swapaxes(path, 0, 1)
            free_path_data[:, w, :] = np.swapaxes(free_path, 0, 1)
        if self._path_data is None:
            self._path_data = paths(path_data, free_path_data)
        else:
            new_data = np.concatenate((self._path_data.pore_space, path_data),
                                      1)
            new_free_data = np.concatenate((self._path_data.free_space,
                                            free_path_data), 1)
            self._path_data = paths(new_data, new_free_data)
        self._sd_updated = False
        self._sterr_updated = False

    def __init__(self, im, walkers=1000, max_steps=5000, stride=10,
                 start_frac=0.2):
        self.im = np.array(im, ndmin=3)
        self._max_steps = max_steps
        self._stride = stride
        self._start_frac = start_frac
        self.porosity = porosity(im)
        self._ndim = np.ndim(im)
        if self._ndim == 3:
            self._z_len = np.size(im, 0)
            self._y_len = np.size(im, 1)
            self._x_len = np.size(im, 2)
        elif self._ndim == 2:
            self._z_len = 1
            self._y_len = np.size(im, 0)
            self._x_len = np.size(im, 1)
        else:
            raise ValueError('image needs to be 2 or 3 dimensional')
        self._shape = (self._z_len, self._y_len, self._x_len)
        self._path_data = None
        self._sd_data = None
        self._sd_updated = False
        self._sterr_data = None
        self._sterr_updated = False
        self.perform_walks(walkers)

    @property
    def steps(self):
        return np.arange(0, self.max_steps+1, self.stride)

    @property
    def path_data(self):
        return self._path_data

    @property
    def sd_data(self):
        if not self._sd_updated:
            self._get_sq_displacement()
        return self._sd_data

    @property
    def sterr_data(self):
        if not self._sterr_updated:
            self._get_sterr()
        return self._sterr_data

    @property
    def ndim(self):
        return self._ndim

    @property
    def shape(self):
        return self._shape

    @property
    def max_steps(self):
        return self._max_steps

    @property
    def total_walkers(self):
        return np.size(self._path_data.pore_space, 1)

    @property
    def stride(self):
        return self._stride

    @property
    def start_frac(self):
        return self._start_frac

    def _get_sq_displacement(self):
        r"""
        Generates squared displacement arrays for pore_space and free_space
        """
        squared_displacement = namedtuple('square_displacement',
                                          ('pore_space', 'free_space'))
        paths = self._path_data.pore_space
        free_paths = self._path_data.free_space
        sd = np.ones(paths.shape)*-1
        sd_free = np.ones(free_paths.shape)*-1
        for w in range(np.size(paths, 1)):
            path = paths[:, w, :]
            path = path[:, path[0, :] >= 0]
            free_path = free_paths[:, w, :]
            free_path = free_path[:, free_path[0, :] >= 0]
            for i in range(np.size(path, 1)):
                sd[:, w, i] = (path[:, i]-path[:, 0])**2
                sd_free[:, w, i] = (free_path[:, i] - free_path[:, 0])**2
        self._sd_data = squared_displacement(sd, sd_free)
        self._sd_updated = True

    def _get_sterr(self):
        r"""
        """
        standard_error = namedtuple('standard_error',
                                    ('pore_space', 'free_space'))
        sd, sd_f = self.sd_data
        ste = np.zeros((4, np.size(self.steps)))
        ste_f = np.zeros((4, np.size(self.steps)))
        for col in range(np.size(sd, 2)):
            ste[3, col] = np.std(np.sum(sd[:, np.where(sd[0, :, col] >= 0),
                                 col], 0))
            ste_f[3, col] = np.std(np.sum(sd_f[:,
                                   np.where(sd_f[0, :, col] >= 0), col], 0))
            ste[3, col] /= np.sqrt(np.size(np.where(sd[0, :, col] >= 0)))
            ste_f[3, col] /= np.sqrt(np.size(np.where(sd[0, :, col] >= 0)))
        for d in range(3):
            for col in range(np.size(sd, 2)):
                ste[d, col] = np.std(sd[d, np.where(sd[d, :, col] >= 0), col])
                ste_f[d, col] = np.std(sd_f[d, np.where(sd_f[d, :, col] >= 0),
                                       col])
                ste[d, col] /= np.sqrt(np.size(np.where(sd[0, :, col] >= 0)))
                ste_f[d, col] /= np.sqrt(np.size(np.where(sd[0, :, col] >= 0)))
        self._sterr_data = standard_error(ste, ste_f)
        self._sterr_updated = True

    def get_msd(self, direction=None):
        r"""
        """
        d = direction
        sd, sd_f = self.sd_data
        msd = np.zeros(np.size(self.steps))
        msd_f = np.zeros(np.size(self.steps))
        mean_sd = namedtuple('mean_squared_displacement',
                             ('pore_space', 'free_space'))
        if d is None:
            for col in range(np.size(sd, 2)):
                msd[col] = np.mean(np.sum(sd[:, np.where(sd[0, :, col] >= 0),
                                   col], 0))
                msd_f[col] = np.mean(np.sum(sd_f[:,
                                     np.where(sd_f[0, :, col] >= 0), col], 0))
        else:
            for col in range(np.size(sd, 2)):
                msd[col] = np.mean(sd[d, np.where(sd[0, :, col] >= 0), col])
                msd_f[col] = np.mean(sd_f[d, np.where(sd_f[0, :, col] >= 0),
                                     col])
        return mean_sd(msd, msd_f)

    def show_msd(self, step_range=None, direction=None, showstderr=True):
        r"""
        Graphs mean squared displacement in pore space and free space
        vs. number of steps taken

        Parameters
        -----------
        step_range: array_like (optional)
            A sequence with the min and max range of steps to plot. If no
            argument is given, the range will be zero to max_steps
        direction: int (optional)
            The axis along which to show mean squared displacement in. If no
            argument is given, total msd is shown
        showstderr: bool (optional)
            If set to True, the graph will show error bars

        Returns
        --------
        out: namedtuple
        """
        if step_range is None:
            step_range = (0, self.max_steps)
        start, stop = step_range
        d = direction
        start = start//self.stride
        stop = stop//self.stride
        sd, sd_f = self.sd_data
        ste, ste_f = self.sterr_data
        steps = self.steps
        msd, msd_f = self.get_msd(direction)
        slopes = namedtuple('slope', ('pore_space', 'free_space'))
        if d is None:
            ste = ste[3, :]
            ste_f = ste_f[3, :]
        else:
            ste = ste[d, :]
            ste_f = ste_f[d, :]
        p = np.polyfit(steps[start:stop], msd[start:stop], 1)
        p_f = np.polyfit(steps[start:stop], msd_f[start:stop], 1)
        if showstderr:
            plt.errorbar(steps[start:stop], msd[start:stop],
                         yerr=ste[start:stop])
            plt.errorbar(steps[start:stop], msd_f[start:stop],
                         yerr=ste_f[start:stop])
        else:
            plt.plot(steps[start:stop], msd[start:stop])
            plt.plot(steps[start:stop], msd_f[start:stop])
        return slopes(p[0], p_f[0])

    def _get_path(self, walker):
        r"""
        Returns matplotlib figures for show_path and save_path functions
        """
        z, y, x = self.shape
        path = self.path_data.pore_space[:, walker, :]
        path = path[:, path[0, :] >= 0]
        free_path = self.path_data.free_space[:, walker, :]
        free_path = free_path[:, free_path[0, :] >= 0]
        max_i = np.size(path, 1) - 1
        if self._ndim == 3:
            size = (7*x/y, 7*z/y)
            fig_im = plt.figure(num=1, figsize=size)
            fig_im = Axes3D(fig_im)
            fig_im.plot(path[2, :], path[1, :], path[0, :], 'c')
            fig_im.plot([path[2, 0]], [path[1, 0]], [path[0, 0]], 'g.')
            fig_im.plot([path[2, max_i]], [path[1, max_i]], [path[0, max_i]],
                        'r.')
            fig_im.set_xlim3d(0, x)
            fig_im.set_ylim3d(0, y)
            fig_im.set_zlim3d(0, z)
            fig_im.invert_yaxis()
            plt.title('Path in Pore Space')
            fig_free = plt.figure(num=2, figsize=size)
            fig_free = Axes3D(fig_free)
            fig_free.plot(free_path[2, :], free_path[1, :], free_path[0, :],
                          'c')
            fig_free.plot([free_path[2, 0]], [free_path[1, 0]],
                          [free_path[0, 0]], 'g.')
            fig_free.plot([free_path[2, max_i]], [free_path[1, max_i]],
                          [free_path[0, max_i]], 'r.')
            fig_free.set_xlim3d(0, x)
            fig_free.set_ylim3d(0, y)
            fig_free.set_zlim3d(0, z)
            fig_free.invert_yaxis()
            plt.title('Path in Free Space')
        elif self._ndim == 2:
            size = (5, 5)
            fig_im = plt.figure(num=1, figsize=size)
            plt.plot(path[2, :], path[1, :], 'c')
            plt.plot(path[2, 0], path[1, 0], 'g.')
            plt.plot(path[2, max_i], path[1, max_i], 'r.')
            plt.xlim((0, x))
            plt.ylim((0, y))
            plt.gca().invert_yaxis()
            plt.title('Path in Pore Space')
            fig_free = plt.figure(num=2, figsize=size)
            plt.plot(free_path[2, :], free_path[1, :], 'c')
            plt.plot(free_path[2, 0], free_path[1, 0], 'g.')
            plt.plot(free_path[2, max_i], free_path[1, max_i], 'r.')
            plt.xlim((0, x))
            plt.ylim((0, y))
            plt.gca().invert_yaxis()
            plt.title('Path in Free Space')
        return(fig_im, fig_free)

    def show_path(self, walker):
        r"""
        Shows the selected walkers path through pore space and free space
        using a matplotlib figure

        Parameters
        -----------
        walker: int
            The walker whose path will be shown
        """
        self._get_path(walker)
        plt.show()

    def save_path(self, walker, f_name):
        r"""
        Saves the selected walkers path through pore space and free space

        Parameters
        -----------
        walker: int
            The walker whose path will be saved

        f_name: string
            A path to a filename. Note that this function will add the suffix
            pore_space and free_space to differentiate between paths
        """

        fig_im, fig_free = self._get_path(walker)
        plt.savefig(f_name+'free_space.png')
        plt.figure(num=1)
        plt.savefig(f_name+'pore_space.png')
