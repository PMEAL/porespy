import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def set_mpl_style():

    sfont = 15
    mfont = 15
    lfont = 15

    line_props = {'linewidth': 4,
                  'markersize': 10}
    font_props = {'size': sfont}
    axes_props = {'titlesize': lfont,
                  'labelsize': mfont,
                  'linewidth': 3,
                  'labelpad': 10}
    xtick_props = {'labelsize': sfont,
                   'top': True,
                   'direction': 'in',
                   'major.size': 10,
                   'major.width': 3}
    ytick_props = {'labelsize': sfont,
                   'right': True,
                   'direction': 'in',
                   'major.size': 10,
                   'major.width': 3}
    legend_props = {'fontsize': mfont,
                    'frameon': False}
    figure_props = {'titlesize': sfont}

    plt.rc('font', **font_props)
    plt.rc('lines', **line_props)
    plt.rc('axes', **axes_props)
    plt.rc('xtick', **xtick_props)
    plt.rc('ytick', **ytick_props)
    plt.rc('legend', **legend_props)
    plt.rc('figure', **figure_props)


def satn_to_movie(im, satn, cmap='viridis',
                  c_under='grey', c_over='white',
                  v_under=1e-3, v_over=1.0, fps=10, repeat=None):
    r"""

    Notes
    -----
    To save animation as a file use:
    ``ani.save('image_based_ip.gif', writer='imagemagick', fps=3)``
    """
    import matplotlib.pyplot as plt
    from matplotlib import animation
    # Define nice color map
    cmap = plt.cm.get_cmap(name=cmap)
    cmap.set_over(color=c_over)
    cmap.set_under(color=c_under)

    # Reduce inv_satn image to limited number of values to speed-up movie
    target = np.around(satn, decimals=3)
    seq = np.zeros_like(target)  # Empty image to place frame
    movie = []  # List to append each frame
    fig, ax = plt.subplots(1, 1)
    steps = np.unique(target)[1:]
    with tqdm(steps) as pbar:
        for v in steps:
            pbar.update()
            seq += v*(target == v)
            seq[~im] = target.max() + 10
            frame1 = ax.imshow(seq, vmin=v_under, vmax=v_over,
                               animated=True, cmap=cmap, origin='xy')
            movie.append([frame1])
    if repeat is not None:
        repeat_delay = repeat/100
        repeat = True
    else:
        repeat = False
    ani = animation.ArtistAnimation(fig, movie, interval=int(1000/fps),
                                    blit=True, repeat=repeat,
                                    repeat_delay=repeat_delay)
    return ani
