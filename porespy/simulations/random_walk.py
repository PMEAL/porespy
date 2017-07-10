def walk(img, maxsteps=None):
    r"""This function performs a single random walk through porous image. It
    returns an array containing the walker path in the image, and the walker
    path in free space. 
    """
    import numpy as np
  
    if maxsteps is None:
        maxsteps = int(np.cbrt(img.size))*2
    start_percent = 20
    coords = []
    free_coords = []
    x_max = np.floor(np.size(img, 0)*start_percent*0.01)
    y_max = np.floor(np.size(img, 1)*start_percent*0.01)
    z_max = np.floor(np.size(img, 2)*start_percent*0.01)
    x = int(np.size(img, 0)/2 - x_max/2 + np.random.randint(0, x_max))
    y = int(np.size(img, 1)/2 - y_max/2 + np.random.randint(0, y_max))
    z = int(np.size(img, 2)/2 - z_max/2 + np.random.randint(0, z_max))
    while True:
        if img[x, y, z]:
            break
        else:
            x = int(np.size(img, 0)/2-x_max/2+np.random.randint(0, x_max))
            y = int(np.size(img, 1)/2-y_max/2+np.random.randint(0, y_max))
            z = int(np.size(img, 2)/2-z_max/2+np.random.randint(0, z_max))
    x_free, y_free, z_free = x, y, z
    coords.append([x, y, z])
    free_coords.append([x, y, z])

    for step in range(maxsteps+1):
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
        try:
            if x_free+x_step < 0 or y_free+y_step < 0 or z_free+z_step < 0:
                raise IndexError
            x_free += x_step
            y_free += y_step
            z_free += z_step
            if img[x+x_step, y+y_step, z+z_step]:
                if x < 0 or y < 0 or z < 0:
                    raise IndexError
                x += x_step
                y += y_step
                z += z_step
            else:
                pass
            coords.append([x, y, z])
            free_coords.append([x_free, y_free, z_free])
        except IndexError:
            break

    path = np.array(coords)
    free_path = np.array(free_coords)
    paths = np.dstack((path, free_path))
    return paths
