import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def find_start_point(img, st_frac):
    r"""
    Finds a random valid start point in a porous image, searching in the
    given of the image

    Parameters
    ----------
    img: array_like
        A 2D or 3D binary image on which to perform the walk
    st_frac: float
        A value between 0 and 1. Determines how much of the image is
        randomly searched for a starting point

    Notes
    -----
    Tuple size returned depends on dimension of image
    """

    ndim = np.ndim(img)
    if ndim == 3:
        x_dim, y_dim, z_dim = np.shape(img)
    elif ndim == 2:
        x_dim, y_dim = np.shape(img)
        z_dim = 1
    x_m = np.floor(x_dim*st_frac)
    y_m = np.floor(y_dim*st_frac)
    z_m = np.floor(z_dim*st_frac)
    i = 0
    while True:
        i += 1
        if i > 10000:
            print("failed to find starting point")
            return None
        x = int(x_dim/2 - x_m/2 + np.random.randint(0, x_m))
        y = int(y_dim/2 - y_m/2 + np.random.randint(0, y_m))
        z = int(z_dim/2 - z_m/2 + np.random.randint(0, z_m))
        if ndim == 3:
            if img[x, y, z]:
                break
        elif ndim == 2:
            if img[x, y]:
                z = 0
                break
    point = (x, y, z)
    return point


def walk(img, st_point, maxsteps=None):
    r"""
    This function performs a single random walk through porous image. It
    returns an array containing the walker path in the image, and the walker
    path in free space.

    Parameters
    ----------
    img: array_like
        A 2D or 3D binary image on which to perform the walk
    st_point: array_like
        A tuple, list, or array with the x, y, (z) coordinates of a valid
        start point.
    maxsteps: int
        The number of steps to attempt per walk. If none is given, a default
        value is calculated
    """

    if maxsteps is None:
        maxsteps = int(np.cbrt(img.size))*5
    if not img[st_point]:
        raise ValueError('invalid starting point: not a pore')
    ndim = np.ndim(img)
    if ndim == 3:
        x, y, z = st_point
        directions = 6
    elif ndim == 2:
        x, y = st_point[0:2]
        z = 0
        directions = 4
    else:
        raise ValueError('img needs to be 2 or 3 dimensions')
    x_free, y_free, z_free = x, y, z
    coords = np.ones((maxsteps+1, 3), dtype=int) * (-1)
    free_coords = np.ones((maxsteps+1, 3), dtype=int) * (-1)
    coords[0, :] = [x, y, z]
    free_coords[0, :] = [x_free, y_free, z_free]
    # begin walk
    for step in range(1, maxsteps+1):
        x_step = 0
        y_step = 0
        z_step = 0
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
        # try block makes sure image does not go out of bounds
        try:
            if x_free+x_step < 0 or y_free+y_step < 0 or z_free+z_step < 0:
                raise IndexError
            x_free += x_step
            y_free += y_step
            z_free += z_step
            # if statement checks if the step leads to a pore in image
            if img[x+x_step, y+y_step, z+z_step]:
                if x < 0 or y < 0 or z < 0:
                    raise IndexError
                x += x_step
                y += y_step
                z += z_step
            coords[step] = [x, y, z]
            free_coords[step] = [x_free, y_free, z_free]
        except IndexError:
            # if the walker goes out of bounds, set the last element in array
            # to last valid coordinate and break out of loop
            coords[maxsteps] = coords[step-1]
            free_coords[maxsteps] = free_coords[step-1]
            break

    paths = (coords, free_coords)
    return paths


def msd(img, direct=None, walks=500, st_frac=0.2, maxsteps=None):
    r"""
    Function for performing many random walks on an image and determining the
    mean squared displacement values the walker travels in both the image
    and free space.

    Parameters
    ----------
    img: array_like
        A binary image on which to perform the walk
    direct: int
        The direction to calculate mean squared displacement in(0:x, 1:y, 2:z).
        If no argument is given, total msd values are calculated
    maxsteps: int
        The number of steps to attempt per walk
    """

    sd = np.zeros((walks, 3))
    sd_free = np.zeros((walks, 3))
    for w in range(walks):
        st_point = find_start_point(img, st_frac)
        path, free_path = walk(img, st_point, maxsteps)
        d = path[maxsteps] - path[0]
        d_free = free_path[maxsteps] - free_path[0]
        sd[w] = d**2
        sd_free[w] = d_free**2
    msd = np.average(sd, 0)
    msd_free = np.average(sd_free, 0)
    if direct is None:
        return (msd, msd_free)
    else:
        return (msd[direct], msd_free[direct])


def show_path(img, maxsteps=None):
    r"""
    This function performs a walk on an image and shows the path taken
    by the walker in free space and in the porous image.

    Parameters
    ----------
    img: array_like
        A binary image on which to perform the walk
    maxsteps: int
        The number of steps to attempt per walk
    """

    (path, free_path) = walk(img, maxsteps)
    max_index = np.size(path, 0) - 1
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot(path[:, 0], path[:, 1], path[:, 2])
    ax.plot([path[0, 0], path[max_index, 0]], [path[0, 1], path[max_index, 1]],
            [path[0, 2], path[max_index, 2]], 'r+')
    ax.autoscale()
    plt.title('Path in Porous Image')
    plt.show()
    fig2 = plt.figure()
    ax2 = Axes3D(fig2)
    ax2.plot(free_path[:, 0], free_path[:, 1], free_path[:, 2])
    ax2.plot([free_path[0, 0], free_path[max_index, 0]], [free_path[0, 1],
             free_path[max_index, 1]], [path[0, 2], path[max_index, 2]], 'r+')
    ax2.autoscale()
    plt.title('Path in Free Space')
    plt.show()
