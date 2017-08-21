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
    im: array of bool
        2D or 3D image
    """

    def __init__(self, im):
        self.im = np.array(im, ndmin=3)
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
        self._shape = (self._z_len, self._y_len, self._x_len)
        self._walk_data = None

    @property
    def walk_data(self):
        return self._walk_data

    @walk_data.setter
    def walk_data(self, walk_data):
        print('Use function get_walk_data to set this property')

    def clear_walk_data(self):
        self._walk_data = None

    @property
    def ndim(self):
        return self._ndim

    @property
    def shape(self):
        return self._shape

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
            If img is 2D, start_point will have z = 0
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

    @jit
    def walk(self, start_point, max_steps, stride=1):
        r"""
        This function performs a single random walk through porous image. It
        returns an array containing the walker path in the image, and the
        walker path in free space.

        Parameters
        ----------
        start_point: array_like
           A tuple, list, or array with index of a valid start point
        max_steps: int
            The number of steps to attempt per walk
        stride: int
            A number greater than zero, determining how many steps are taken
            between each returned coordinate

        Returns
        --------
        paths: namedtuple
            A tuple containing 2 arrays showing the paths taken by the walker:
            path.pore_space and path.free_space
        """

        ndim = self._ndim
        im = self.im
        if ndim == 3:
            directions = 6
        elif ndim == 2:
            directions = 4
        else:
            raise ValueError('im needs to be 2 or 3 dimensions')
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
            if (x_free+x_step >= x_max or y_free+y_step >= y_max or
                    z_free+z_step >= z_max):
                break
            if x+x_step < 0 or y+y_step < 0 or z+z_step < 0:
                break
            if x+x_step >= x_max or y+y_step >= y_max or z+z_step >= z_max:
                break
            x_free += x_step
            y_free += y_step
            z_free += z_step
            # checks if the step leads to a pore in image
            if im[z+z_step, y+y_step, x+x_step]:
                x += x_step
                y += y_step
                z += z_step
            steps += 1
            coords[step] = [z, y, x]
            free_coords[step] = [z_free, y_free, x_free]
        paths = namedtuple('path', ('pore_space', 'open_space'))
        paths = (coords[:steps+1:stride, :], free_coords[:steps+1:stride, :])
        return paths

    def get_walk_data(self, walkers=100, start_frac=0.2, max_steps=3000,
                      stride=20):
        r"""
        Function for calculating squared displacement values for individual
        walkers at specified intervals, in array format

        Parameters
        ----------
        walkers: int
            The number of walks to perform
        start_frac: int
            A value between 0 and 1. Determines what fraction of the image is
            randomly searched for a starting point
        max_steps: int
            The number of steps to attempt per walk

        stride: int
            Number of steps taken between each returned displacement value

        Returns
        --------
        out: tuple
            A tuple containing squared distance arrays for image walks and free
            space walks. Each row in the arrays represents a different walker.
            Each column represents a different step length, with intervals
            determined by stride. If stride is 20, for example, then moving
            right one column means taking 20 steps
        """

        sd = np.ones((3, walkers, int(max_steps//stride+1)))*-1
        sd_free = np.ones((3, walkers, int(max_steps//stride+1)))*-1
        for w in range(walkers):
            start_point = self.find_start_point(start_frac)
            path, free_path = self.walk(start_point, max_steps, stride)
            steps = np.size(path, 0)
            for i in range(steps):
                sd[:, w, i] = (path[i]-path[0])**2
                sd_free[:, w, i] = (free_path[i]-free_path[0])**2
        if self.walk_data is not None:
            sd_prev, sd_free_prev = self.walk_data
            try:
                sd_c = np.concatenate((sd_prev, sd), 1)
                sd_free_c = np.concatenate((sd_free_prev, sd_free), 1)
                self._walk_data = (sd_c, sd_free_c)
            except ValueError:
                self.clear_walk_data()
                print("Warning: walk_data could not be concatenated")
                self._walk_data = (sd, sd_free)
        else:
            self._walk_data = (sd, sd_free)
        return (sd, sd_free)

    def mean_squared_displacement(self, walkers=800, start_frac=0.2,
                                  max_steps=5000):
        r"""
        Function for performing many random walks on an image and
        determining the mean squared displacement values the walker
        travels in both the image and free space

        Parameters
        ----------
        walkers: int
            The number of walks to perform
        start_frac: int
            A value between 0 and 1. Determines what fraction of the image is
            randomly searched for a starting point
        max_steps: int
            The number of steps to attempt per walk

        Returns
        --------
        out: tuple
            A tuple containing the msd values for the image walkers in index
            0 and for the free space walkers in index 1
        """

        sd = np.zeros((walkers, 3))
        sd_free = np.zeros((walkers, 3))
        for w in range(walkers):
            start_point = self.find_start_point(start_frac)
            path, free_path = self.walk(start_point, max_steps)
            steps = np.size(path, 0) - 1
            d = path[steps] - path[0]
            d_free = free_path[steps] - free_path[0]
            sd[w] = d**2
            sd_free[w] = d_free**2
        msd = np.average(sd, 0)
        msd_free = np.average(sd_free, 0)
        return (msd, msd_free)

    def run(self, direction=None, walkers=1000, start_frac=0.2,
            max_steps=5000, stride=10):
        r"""
        Performs one walk for each step length from 1 to max_steps. Graphs
        MSD in free space and MSD in image vs. step length

        Parameters
        ----------
        direction: int
            0, 1, or 2. Determines direction to graph mean square displacement
            in. If direction is None, the total displacement is used.
        walkers: int
            The number of walks to perform
        start_frac: int
            A value between 0 and 1. Determines what fraction of the image is
            randomly searched for a starting point
        max_steps: int
            Maximum number of steps to attempt
        stride: int
            Number of steps taken between each point plotted

        Returns
        --------
        out: tuple
            A tuple containing the slopes of the msd vs step length graphs
        """

        steps = np.arange(0, max_steps+1, stride)
        sd, sd_f = self.get_walk_data(walkers, start_frac, max_steps, stride)
        msd = np.zeros(np.size(steps))
        ste = np.zeros(np.size(steps))
        msd_f = np.zeros(np.size(steps))
        ste_f = np.zeros(np.size(steps))
        if direction is None:
            for col in range(np.size(sd, 2)):
                msd[col] = np.mean(np.sum(sd[:, np.where(sd[0, :, col] >= 0),
                                   col], 0))
                ste[col] = np.std(np.sum(sd[:, np.where(sd[0, :, col] >= 0),
                                  col], 0))
                msd_f[col] = np.mean(np.sum(sd_f[:,
                                     np.where(sd_f[0, :, col] >= 0), col], 0))
                ste_f[col] = np.std(np.sum(sd_f[:,
                                    np.where(sd_f[0, :, col] >= 0), col], 0))
                ste[col] /= np.sqrt(np.size(np.where(sd[0, :, col] >= 0)))
                ste_f[col] /= np.sqrt(np.size(np.where(sd[0, :, col] >= 0)))
        else:
            d = direction
            for col in range(np.size(sd, 2)):
                msd[col] = np.mean(sd[d, np.where(sd[0, :, col] >= 0), col])
                ste[col] = np.std(sd[d, np.where(sd[0, :, col] >= 0), col])
                msd_f[col] = np.mean(sd_f[d, np.where(sd_f[0, :, col] >= 0),
                                     col])
                ste_f[col] = np.std(sd_f[d, np.where(sd_f[0, :, col] >= 0),
                                    col])
                ste[col] /= np.sqrt(np.size(np.where(sd[0, :, col] >= 0)))
                ste_f[col] /= np.sqrt(np.size(np.where(sd[0, :, col] >= 0)))

        p = np.polyfit(steps, msd, 1)
        p_f = np.polyfit(steps, msd_f, 1)
        plt.errorbar(steps, msd, yerr=ste)
        plt.errorbar(steps, msd_f, yerr=ste_f)
        return(p[0], p_f[0])

    def tortuosity(self, direction=None, walkers=1000, start_frac=0.2,
                   max_steps=5000, stride=5):
        r"""
        Calculates tortuosity of the image

        Parameters
        ----------
        start_frac: int
            A value between 0 and 1. Determines what fraction of the image is
            randomly searched for a starting point
        max_steps: int
            Maximum number of steps to attempt
        stride: int
            Number of steps taken between each point plotted in run function

        Returns
        --------
        out: array_like
            Estimation of tortuosity of the image in each direction
        """

        m, m_f = self.run(direction, walkers, start_frac, max_steps, stride)
        tortuosity = m_f/m
        return tortuosity

    def _get_path(self, start_point=None, max_steps=10000, size=None):
        r"""
        Returns matplotlib figures for show_path and save_path functions
        """
        if start_point is None:
            start_point = self.find_start_point(0.2)
        (path, free_path) = self.walk(start_point, max_steps)
        max_i = np.size(path, 0) - 1
        z, y, x = self._z_len, self._y_len, self._x_len

        if self._ndim == 3:
            size = (7*x/y, 7*z/y)
            fig_im = plt.figure(num=1, figsize=size)
            fig_im = Axes3D(fig_im)
            fig_im.plot(path[:, 2], path[:, 1], path[:, 0], 'c')
            fig_im.plot([path[0, 2]], [path[0, 1]], [path[0, 0]], 'g.')
            fig_im.plot([path[max_i, 2]], [path[max_i, 1]], [path[max_i, 0]],
                        'r.')
            fig_im.set_xlim3d(0, x)
            fig_im.set_ylim3d(0, y)
            fig_im.set_zlim3d(0, z)
            fig_im.invert_yaxis()
            plt.title('Path in Porous Image')
            fig_free = plt.figure(num=2, figsize=size)
            fig_free = Axes3D(fig_free)
            fig_free.plot(free_path[:, 2], free_path[:, 1], free_path[:, 0],
                          'c')
            fig_free.plot([free_path[0, 2]], [free_path[0, 1]],
                          [free_path[0, 0]], 'g.')
            fig_free.plot([free_path[max_i, 2]], [free_path[max_i, 1]],
                          [free_path[max_i, 0]], 'r.')
            fig_free.set_xlim3d(0, x)
            fig_free.set_ylim3d(0, y)
            fig_free.set_zlim3d(0, z)
            fig_free.invert_yaxis()
            plt.title('Path in Free Space')
        elif self._ndim == 2:
            if size is None:
                size = (5, 5)
            fig_im = plt.figure(num=1, figsize=size)
            plt.plot(path[:, 2], path[:, 1], 'c')
            plt.plot(path[0, 2], path[0, 1], 'g.')
            plt.plot(path[max_i, 2], path[max_i, 1], 'r.')
            plt.xlim((0, x))
            plt.ylim((0, y))
            plt.gca().invert_yaxis()
            plt.title('Path in Porous Image')
            fig_free = plt.figure(num=2, figsize=size)
            plt.plot(free_path[:, 2], free_path[:, 1], 'c')
            plt.plot(free_path[0, 2], free_path[0, 1], 'g.')
            plt.plot(free_path[max_i, 2], free_path[max_i, 1], 'r.')
            plt.xlim((0, x))
            plt.ylim((0, y))
            plt.gca().invert_yaxis()
            plt.title('Path in Free Space')
        return(fig_im, fig_free)

    def show_path(self, start_point=None, max_steps=10000, size=None):
        r"""
        This function performs a walk on an image and shows the path taken
        by the walker in free space and in the porous image using matplotlib
        plots

        Parameters
        ----------
        start_point: tuple of ints
            Starting coordinates for the walk
        max_steps: int
            The number of steps to attempt in a walk
        size: tuple of ints
            Width, height, in inches. For 2D paths only
        """
        fig_im, fig_free = self._get_path(start_point, max_steps, size)
        plt.show()
        plt.close('all')

    def save_path(self, f_name, start_point=None, max_steps=3000, size=None):
        r"""
        This function performs a walk on an image and saves the path taken by
        the walker in free space and the porous image using matplotlib plots

        Parameters
        -----------
        f_name: string
            A path to a filename. This function will add the suffix Img and
            Free to indicate path in the image vs path in the free space
        start_point: tuple of ints
            Starting coordinates for the walk
        max_steps: int
            The number of steps to attempt in a walk
        size: tuple
            Width, height, in inches. For 2D paths only
        """
        fig_im, fig_free = self._get_path(start_point, max_steps, size)
        plt.savefig(f_name+'free.png')
        plt.figure(num=1)
        plt.savefig(f_name+'img.png')
