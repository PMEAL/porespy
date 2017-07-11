import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def find_start_point(img, st_frac):
    r""" Finds a random valid start point in a porous image, searching in the
    given of the image
    """

    x_dim, y_dim, z_dim = np.shape(img)
    x_m = np.floor(x_dim*st_frac)
    y_m = np.floor(y_dim*st_frac)
    z_m = np.floor(z_dim*st_frac)
    x = int(np.size(img, 0)/2 - x_m/2 + np.random.randint(0, x_m))
    y = int(np.size(img, 1)/2 - y_m/2 + np.random.randint(0, y_m))
    z = int(np.size(img, 2)/2 - z_m/2 + np.random.randint(0, z_m))
    while True:
        if img[x, y, z]:
            break
        else:
            x = int(np.size(img, 0)/2-x_m/2+np.random.randint(0, x_m))
            y = int(np.size(img, 1)/2-y_m/2+np.random.randint(0, y_m))
            z = int(np.size(img, 2)/2-z_m/2+np.random.randint(0, z_m))
    return (x, y, z)


def walk(img, st_point, maxsteps=None):
    r"""This function performs a single random walk through porous image. It
    returns an array containing the walker path in the image, and the walker
    path in free space.
    """

    if maxsteps is None:
        maxsteps = int(np.cbrt(img.size))*100
    # find start point in middle 20 percent of image
    x, y, z = st_point
    x_free, y_free, z_free = x, y, z
    coords = np.ones((maxsteps, 3), dtype=int) * (-1)
    free_coords = np.ones((maxsteps, 3), dtype=int) * (-1)
    coords[0, :] = [x, y, z]
    free_coords[0, :] = [x_free, y_free, z_free]
    # begin walk
    for step in range(maxsteps):
        x_step = 0
        y_step = 0
        z_step = 0
        direction = np.random.randint(0, 6)
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
            break

    paths = (coords, free_coords)
    return paths


def show_path(img, maxsteps=None):
    r"""This function performs a walk on an image and shows the path taken
    by the walker in free space and in the porous image.
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
