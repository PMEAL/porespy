import numpy as np
import matplotlib.pyplot as plt
from porespy.metrics import porosity
from numba import jit
from mpl_toolkits.mplot3d import Axes3D


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
        self.ndim = np.ndim(im)
        if self.ndim == 3:
            self.z_len = np.size(im, 0)
            self.y_len = np.size(im, 1)
            self.x_len = np.size(im, 2)
        elif self.ndim == 2:
            self.z_len = 1
            self.y_len = np.size(im, 0)
            self.x_len = np.size(im, 1)
        self.shape = (self.z_len, self.y_len, self.x_len)
        self.sds = None

    def find_start_point(self, st_frac):
        r"""
        Finds a random valid start point in a porous image, searching in the
        given fraction of the image

        Parameters
        ----------
        st_frac: float
            A value between 0 and 1. Determines what fraction of the image is
            randomly searched for a starting point

        Returns
        --------
        st_point: tuple
            A tuple containing the index of a valid start point.
            If img is 2D, st_point will have z = 0
        """

        x_r = self.x_len*st_frac
        y_r = self.y_len*st_frac
        z_r = self.z_len*st_frac
        i = 0
        while True:
            i += 1
            if i > 10000:
                print("failed to find starting point")
                return None
            x = int(self.x_len/2 - x_r/2 + np.random.randint(x_r+1))
            y = int(self.y_len/2 - y_r/2 + np.random.randint(y_r+1))
            z = int(self.z_len/2 - z_r/2 + np.random.randint(z_r+1))
            if self.im[z, y, x]:
                break
        st_point = (z, y, x)
        return st_point

    @jit
    def walk(self, st_point, maxsteps, stride=1):
        r"""
        This function performs a single random walk through porous image. It
        returns an array containing the walker path in the image, and the
        walker path in free space.

        Parameters
        ----------
        st_point: array_like
           A tuple, list, or array with index of a valid start point
        stride: int
            A number greater than zero, determining how many steps are taken
            between each returned coordinate
        maxsteps: int
            The number of steps to attempt per walk

        Returns
        --------
        paths: tuple
            A tuple containing 2 arrays: paths[0] contains the coordinates of
            the walker's path through the image, and paths[1] contains the
            coords of the walker's path through free space
        """

        ndim = self.ndim
        im = self.im
        if ndim == 3:
            directions = 6
        elif ndim == 2:
            directions = 4
        else:
            raise ValueError('im needs to be 2 or 3 dimensions')
        (z, y, x) = st_point
        if not im[z, y, x]:
            raise ValueError('invalid starting point: not a pore')
        z_max, y_max, x_max = self.shape
        z_free, y_free, x_free = z, y, x
        coords = np.ones((maxsteps+1, 3), dtype=int) * (-1)
        free_coords = np.ones((maxsteps+1, 3), dtype=int) * (-1)
        coords[0, :] = [z, y, x]
        free_coords[0, :] = [z_free, y_free, x_free]
        steps = 0
        # begin walk
        for step in range(1, maxsteps+1):
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

        paths = (coords[:steps+1:stride, :], free_coords[:steps+1:stride, :])
        return paths

    def msd(self, walks=800, st_frac=0.2, maxsteps=5000):
        r"""
        Function for performing many random walks on an image and
        determining the mean squared displacement values the walker
        travels in both the image and free space

        Parameters
        ----------
        walks: int
            The number of walks to perform
        st_frac: int
            A value between 0 and 1. Determines what fraction of the image is
            randomly searched for a starting point
        stride: int
            Value used in walk function
        maxsteps: int
            The number of steps to attempt per walk

        Returns
        --------
        out: tuple
            A tuple containing the msd values for the image walks in index
            0 and for the free space walks in index 1
        """

        sd = np.zeros((walks, 3))
        sd_free = np.zeros((walks, 3))
        for w in range(walks):
            st_point = self.find_start_point(st_frac)
            path, free_path = self.walk(st_point, maxsteps)
            steps = np.size(path, 0) - 1
            d = path[steps] - path[0]
            d_free = free_path[steps] - free_path[0]
            sd[w] = d**2
            sd_free[w] = d_free**2
        msd = np.average(sd, 0)
        msd_free = np.average(sd_free, 0)
        return (msd, msd_free)

    def sd_array(self, walks=100, st_frac=0.2, maxsteps=3000, stride=20):
        r"""
        Function for calculating squared displacement values for individual
        walkers at specified intervals, in array format

        Parameters
        ----------
        walks: int
            The number of walks to perform
        st_frac: int
            A value between 0 and 1. Determines what fraction of the image is
            randomly searched for a starting point
        maxsteps: int
            The number of steps to attempt per walk
        stride: int
            Value used in walk function

        Returns
        --------
        out: tuple
            A tuple containing squared distance arrays for image walks and free
            space walks. Each row in the arrays represents a different walker.
            Each column represents a different step length, with intervals
            determined by stride. If stride is 20, for example, then moving
            right one column means taking 20 steps
        """

        sd = np.zeros((walks, maxsteps//stride+1))
        sd_free = np.zeros((walks, maxsteps//stride+1))
        for w in range(walks):
            st_point = self.find_start_point(st_frac)
            path, free_path = self.walk(st_point, maxsteps, stride)
            steps = np.size(path, 0)
            for i in range(steps):
                sd[w, i] = np.sum((path[i]-path[0])**2)
                sd_free[w, i] = np.sum((free_path[i]-free_path[0])**2)
        if self.sds is not None:
            sd_prev, sd_free_prev = self.sds
            sd = np.concatenate((sd_prev, sd))
            sd_free = np.concatenate((sd_free_prev, sd_free))
        self.sds = (sd, sd_free)
        return (sd, sd_free)

    def msd_graph(self, st_frac=0.2, maxsteps=5000, stride=5):
        r"""
        Performs one walk for each step length from 1 to maxsteps. Graphs
        MSD in free space and MSD in image vs. step length

        Parameters
        ----------
        st_frac: int
            A value between 0 and 1. Determines what fraction of the image is
            randomly searched for a starting point
        maxsteps: int
            Maximum number of steps to attempt
        """

        steps = np.arange(0, maxsteps+1, stride)
        sd, sd_f = self.sd_array(1000, st_frac, maxsteps, stride)
        msd = np.zeros(np.size(steps))
        msd_f = np.zeros(np.size(steps))
        for col in range(np.size(sd, 1)):
            msd[col] = np.mean(sd[np.where(sd[:, col] >= 0), col])
            msd_f[col] = np.mean(sd_f[np.where(sd_f[:, col] >= 0), col])
        p = np.polyfit(steps, msd, 1)
        p_f = np.polyfit(steps, msd_f, 1)
        plt.plot(steps, msd, 'r.', steps, msd_f, 'b.')
        plt.text(maxsteps*0.75, maxsteps*0.9, p_f[0])
        plt.text(maxsteps*0.5, maxsteps*0.1, p[0])

    def error_analysis(self, walks):
        r"""
        Returns an estimation for standard error, as a percentage of the
        mean squared distance calculated

        Parameters
        -----------
        walks: int
            The number of walks to perform

        Returns
        --------
        out: float
            Estimation of standard error as a percentage of the mean squared
            displacement, if a random walk simulation is performed on the given
            image with the specified number of walks
        """
        steps = 2000
        std = np.zeros(21)
        mean = np.zeros(21)
        sd, sd_free = self.sd_array(walks=1000, maxsteps=steps, stride=100)
        for col in range(np.size(sd, 1)):
            stdi = np.std(sd[np.where(sd[:, col] > 0), col])
            meani = np.mean(sd[np.where(sd[:, col] > 0), col])
            std[col] = stdi
            mean[col] = meani
        ste = std/np.sqrt(walks)
        stepct = 100*ste/mean
        return np.mean(stepct[1:12])

    def show_path(self, st_point=None, maxsteps=3000, size=None):
        r"""
        This function performs a walk on an image and shows the path taken
        by the walker in free space and in the porous image using matplotlib
        plots

        Parameters
        ----------
        st_point: tuple of ints
            Starting coordinates for the walk
        maxsteps: int
            The number of steps to attempt in a walk
        size: tuple of ints
            Width, height, in inches. For 2D paths only
        """
        if st_point is None:
            st_point = self.find_start_point(0.2)
        (path, free_path) = self.walk(st_point, maxsteps)
        max_i = np.size(path, 0) - 1
        z, y, x = self.z_len, self.y_len, self.x_len

        if self.ndim == 3:
            size = (7*x/y, 7*z/y)
            fig = plt.figure(figsize=size)
            ax = Axes3D(fig)
            ax.plot(path[:, 2], path[:, 1], path[:, 0], 'c')
            ax.plot([path[0, 2]], [path[0, 1]], [path[0, 0]], 'g.')
            ax.plot([path[max_i, 2]], [path[max_i, 1]], [path[max_i, 0]], 'r.')
            ax.set_xlim3d(0, x)
            ax.set_ylim3d(0, y)
            ax.set_zlim3d(0, z)
            ax.invert_yaxis()
            plt.title('Path in Porous Image')
            plt.show()
            plt.close()
            fig2 = plt.figure(figsize=size)
            ax2 = Axes3D(fig2)
            ax2.plot(free_path[:, 2], free_path[:, 1], free_path[:, 0], 'c')
            ax2.plot([free_path[0, 2]], [free_path[0, 1]], [free_path[0, 0]],
                     'g.')
            ax2.plot([free_path[max_i, 2]], [free_path[max_i, 1]],
                     [free_path[max_i, 0]], 'r.')
            ax2.set_xlim3d(0, x)
            ax2.set_ylim3d(0, y)
            ax2.set_zlim3d(0, z)
            ax2.invert_yaxis()
            plt.title('Path in Free Space')
            plt.show()
            plt.close()
        elif self.ndim == 2:
            if size is None:
                size = (5, 5)
            fig = plt.figure(figsize=size)
            plt.plot(path[:, 2], path[:, 1], 'c')
            plt.plot(path[0, 2], path[0, 1], 'g.')
            plt.plot(path[max_i, 2], path[max_i, 1], 'r.')
            plt.xlim((0, x))
            plt.ylim((0, y))
            plt.gca().invert_yaxis()
            plt.title('Path in Porous Image')
            plt.show()
            plt.close()
            fig2 = plt.figure(figsize=size)
            plt.plot(free_path[:, 2], free_path[:, 1], 'c')
            plt.plot(free_path[0, 2], free_path[0, 1], 'g.')
            plt.plot(free_path[max_i, 2], free_path[max_i, 1], 'r.')
            plt.xlim((0, x))
            plt.ylim((0, y))
            plt.gca().invert_yaxis()
            plt.title('Path in Free Space')
            plt.show()
            plt.close()

    def save_path(self, f_name, st_point=None, maxsteps=3000, size=(5, 5)):
        r"""
        This function performs a walk on an image and saves the path taken by
        the walker in free space and the porous image using matplotlib plots

        Parameters
        -----------
        f_path: string
            A path to a filename. This function will add the suffix Img and
            Free to indicate path in the image vs path in the free space
        maxsteps: int
            The number of steps to attempt in a walk
        size: tuple
            Width, height, in inches. For 2D paths only
        """
        if st_point is None:
            st_point = self.find_start_point(0.2)
        (path, free_path) = self.walk(st_point, maxsteps)
        max_i = np.size(path, 0) - 1
        z, y, x = self.z_len, self.y_len, self.x_len

        if self.ndim == 3:
            size = (7*x/y, 7*z/y)
            fig = plt.figure(figsize=size)
            ax = Axes3D(fig)
            ax.plot(path[:, 2], path[:, 1], path[:, 0], 'c')
            ax.plot([path[0, 2]], [path[0, 1]], [path[0, 0]], 'g.')
            ax.plot([path[max_i, 2]], [path[max_i, 1]], [path[max_i, 0]], 'r.')
            ax.set_xlim3d(0, x)
            ax.set_ylim3d(0, y)
            ax.set_zlim3d(0, z)
            ax.invert_yaxis()
            plt.title('Path in Porous Image')
            plt.savefig(f_name+'Img.png')
            plt.close()
            fig2 = plt.figure(figsize=size)
            ax2 = Axes3D(fig2)
            ax2.plot(free_path[:, 2], free_path[:, 1], free_path[:, 0], 'c')
            ax2.plot([free_path[0, 2]], [free_path[0, 1]], [free_path[0, 0]],
                     'g.')
            ax2.plot([free_path[max_i, 2]], [free_path[max_i, 1]],
                     [free_path[max_i, 0]], 'r.')
            ax2.set_xlim3d(0, x)
            ax2.set_ylim3d(0, y)
            ax2.set_zlim3d(0, z)
            ax2.invert_yaxis()
            plt.title('Path in Free Space')
            plt.savefig(f_name+'Free.png')
            plt.close()
        elif self.ndim == 2:
            if size is None:
                size = (5, 5)
            fig = plt.figure(figsize=size)
            plt.plot(path[:, 2], path[:, 1], 'c')
            plt.plot(path[0, 2], path[0, 1], 'g.')
            plt.plot(path[max_i, 2], path[max_i, 1], 'r.')
            plt.xlim((0, x))
            plt.ylim((0, y))
            plt.gca().invert_yaxis()
            plt.title('Path in Porous Image')
            plt.savefig(f_name+'Img.png')
            plt.close()
            fig2 = plt.figure(figsize=size)
            plt.plot(free_path[:, 2], free_path[:, 1], 'c')
            plt.plot(free_path[0, 2], free_path[0, 1], 'g.')
            plt.plot(free_path[max_i, 2], free_path[max_i, 1], 'r.')
            plt.xlim((0, x))
            plt.ylim((0, y))
            plt.gca().invert_yaxis()
            plt.title('Path in Free Space')
            plt.savefig(f_name+'Free.png', )
            plt.close()
