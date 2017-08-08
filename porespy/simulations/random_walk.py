import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numba import jit


def find_start_point(img, st_frac):
    r"""
    Finds a random valid start point in a porous image, searching in the
    given fraction of the image

    Parameters
    ----------
    img: array_like
        A 2D or 3D binary image on which to perform the walk
    st_frac: float
        A value between 0 and 1. Determines how much of the image is
        randomly searched for a starting point

    Returns
    --------
    st_point: tuple
        A tuple containing the index of a valid start point.
        If img is 2D, st_point will have z = 0
    """

    ndim = np.ndim(img)
    if ndim == 3:
        z_dim, y_dim, x_dim = np.shape(img)
    elif ndim == 2:
        y_dim, x_dim = np.shape(img)
        z_dim = 1
    x_r = x_dim*st_frac
    y_r = y_dim*st_frac
    z_r = z_dim*st_frac
    i = 0
    while True:
        i += 1
        if i > 10000:
            print("failed to find starting point")
            return None
        x = int(x_dim/2 - x_r/2 + np.random.randint(0, x_r))
        y = int(y_dim/2 - y_r/2 + np.random.randint(0, y_r))
        z = int(z_dim/2 - z_r/2 + np.random.randint(0, z_r))
        if ndim == 3:
            if img[z, y, x]:
                break
        elif ndim == 2:
            if img[y, x]:
                z = 0
                break
    st_point = (z, y, x)
    return st_point


@jit
def walk(img, st_point, maxsteps, stride=1):
    r"""
    This function performs a single random walk through porous image. It
    returns an array containing the walker path in the image, and the walker
    path in free space.

    Parameters
    ----------
    img: array_like
        A 2D or 3D binary image on which to perform the walk
    st_point: array_like
        A tuple, list, or array with index of a valid
        start point
    stride: int
        A number greater than zero, determining how many steps are taken
        between each returned coordinate
    maxsteps: int
        The number of steps to attempt per walk. If none is given, a default
        value is calculated

    Returns
    --------
    paths: tuple
        A tuple containing 2 arrays: paths[0] contains the coordinates of
        the walker's path through the image, and paths[1] contains the
        coords of the walker's path through free space
    """
    ndim = np.ndim(img)
    if ndim == 3:
        z, y, x = st_point
        directions = 6
    elif ndim == 2:
        y, x = st_point[1:3]
        z = 0
        directions = 4
        img = np.array([img])
    else:
        raise ValueError('img needs to be 2 or 3 dimensions')
    if not img[z, y, x]:
        raise ValueError('invalid starting point: not a pore')
    z_max, y_max, x_max = np.shape(img)
    x_free, y_free, z_free = x, y, z
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
        if img[z+z_step, y+y_step, x+x_step]:
            x += x_step
            y += y_step
            z += z_step
        steps += 1
        coords[step] = [z, y, x]
        free_coords[step] = [z_free, y_free, x_free]
    paths = (coords[:steps+1:stride, :], free_coords[:steps+1:stride, :])
    return paths


def msd(img, direct=None, walks=800, st_frac=0.2, maxsteps=3000, stride=1):
    r"""
    Function for performing many random walks on an image and determining the
    mean squared displacement values the walker travels in both the image
    and free space

    Parameters
    ----------
    img: array_like
        A binary image on which to perform the walks
    direct: int
        The direction to calculate mean squared displacement in
        (0, 1 or 2). If no argument is given, all msd values are given,
        and can be summed to find total msd
    walks: int
        The number of walks to perform
    st_frac: int
        Value used in find_start_point function
    stride: int
        Value used in walk function
    maxsteps: int
        The number of steps to attempt per walk. If no argument is given, the
        walks will use a default value calculated

    Returns
    --------
    out: tuple
        A tuple containing the msd values for the image walks in index 0 and
        for the free space walks in index 1
    """
    sd = np.zeros((walks, 3))
    sd_free = np.zeros((walks, 3))
    for w in range(walks):
        st_point = find_start_point(img, st_frac)
        path, free_path = walk(img, st_point, maxsteps, stride)
        steps = np.size(path, 0) - 1
        d = path[steps] - path[0]
        d_free = free_path[steps] - free_path[0]
        sd[w] = d**2
        sd_free[w] = d_free**2
    msd = np.average(sd, 0)
    msd_free = np.average(sd_free, 0)
    if direct is None:
        return (msd, msd_free)
    else:
        return (msd[direct], msd_free[direct])


def sd_array(img, walks=100, st_frac=0.2, maxsteps=3000, stride=100,
             previous_sds=None):
    r"""
    Function for outputing squared displacement values for individual walkers
    at every given interval, in array format

    Parameters
    ----------
    img: array_like
        A binary image on which to perform the walks
    walks: int
        The number of walks to perform
    st_frac: int
        Value used in find_start_point function
    stride: int
        Value used in walk function
    maxsteps: int
        The number of steps to attempt per walk. If no argument is given, the
        walks will use a default value calculated in the walk function
    previous_sds: tuple
        A tuple containing previous squared distance arrays. Should contain
        squared distance array for image in index 0, and sd array for free
        space in index 1. If this entry is given, the function will
        concatenate the previous arrays with more walker data, and then
        return them

    Returns
    --------
    out: tuple
        A tuple containing squared distance arrays for image walks and free
        space walks. Each row in the arrays represents a different walker.
        Each column represents a different step length, with intervals
        determined by stride
    """

    sd = np.zeros((walks, maxsteps//stride+1))
    sd_free = np.zeros((walks, maxsteps//stride+1))
    for w in range(walks):
        st_point = find_start_point(img, st_frac)
        path, free_path = walk(img, st_point, maxsteps, stride)
        steps = np.size(path, 0)
        for i in range(steps):
            sd[w, i] = np.sum((path[i]-path[0])**2)
            sd_free[w, i] = np.sum((free_path[i]-free_path[0])**2)
    if previous_sds is not None:
        sd_prev, sd_free_prev = previous_sds
        sd = np.concatenate((sd_prev, sd))
        sd_free = np.concatenate((sd_free_prev, sd_free))
    return (sd, sd_free)


def error_analysis(img, walks):
    r"""
    Returns an estimation for standard error, as a percentage of the
    mean squared distance calculated

    Parameters
    -----------
    img: array_like
        A binary image
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
    sd, sd_free = sd_array(img, 3000, steps, stride=100)
    for col in range(np.size(sd, 1)):
        stdi = np.std(sd[np.where(sd[:, col] > 0), col])
        meani = np.mean(sd[np.where(sd[:, col] > 0), col])
        std[col] = stdi
        mean[col] = meani

    ste = std/np.sqrt(walks)
    stepct = 100*ste/mean
    return np.mean(stepct[1:12])


def show_path(img, st_point=None, maxsteps=3000, size=None):
    r"""
    This function performs a walk on an image and shows the path taken
    by the walker in free space and in the porous image using matplotlib
    plots

    Parameters
    ----------
    img: array_like
        A binary image on which to perform the walk
    st_point: tuple of ints
        Starting coordinates for the walk
    maxsteps: int
        The number of steps to attempt in a walk. If no argument is given, the
        walk will use a default value calculated in the walk function
    size: tuple of ints
        Width, height, in inches. For 2D paths only
    """
    if st_point is None:
        st_point = find_start_point(img, 0.2)
    (path, free_path) = walk(img, st_point, maxsteps)
    max_i = np.size(path, 0) - 1

    if np.ndim(img) == 3:
        z, y, x = np.shape(img)
        if size is None:
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
        ax2.plot([free_path[0, 2]], [free_path[0, 1]], [free_path[0, 0]], 'g.')
        ax2.plot([free_path[max_i, 2]], [free_path[max_i, 1]],
                 [free_path[max_i, 0]], 'r.')
        ax2.set_xlim3d(0, x)
        ax2.set_ylim3d(0, y)
        ax2.set_zlim3d(0, z)
        ax2.invert_yaxis()
        plt.title('Path in Free Space')
        plt.show()
        plt.close()
    elif np.ndim(img) == 2:
        y, x = np.shape(img)
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


def save_path(img, f_name, st_point=None, maxsteps=3000, size=(5, 5)):
    r"""
    This function performs a walk on an image and saves the path taken by
    the walker in free space and the porous image using matplotlib plots

    Parameters
    -----------
    img: array_like
        A binary image on which to perform the walk
    f_path: string
        A path to a filename. This function will add the suffix Img and Free to
        indicate path in the image vs path in the free space
    maxsteps: int
        The number of steps to attempt in a walk. If no argument is given, the
        walk will use a default value calculated in the walk function
    size: tuple
        Width, height, in inches.
    """
    if st_point is None:
        st_point = find_start_point(img, 0.2)
    (path, free_path) = walk(img, st_point, maxsteps)
    max_i = np.size(path, 0) - 1

    if np.ndim(img) == 3:
        z, y, x = np.shape(img)
        if size is None:
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
        ax2.plot([free_path[0, 2]], [free_path[0, 1]], [free_path[0, 0]], 'g.')
        ax2.plot([free_path[max_i, 2]], [free_path[max_i, 1]],
                 [free_path[max_i, 0]], 'r.')
        ax2.set_xlim3d(0, x)
        ax2.set_ylim3d(0, y)
        ax2.set_zlim3d(0, z)
        ax2.invert_yaxis()
        plt.title('Path in Free Space')
        plt.savefig(f_name+'Free.png')
        plt.close()
    elif np.ndim(img) == 2:
        y, x = np.shape(img)
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
