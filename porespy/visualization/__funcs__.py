import porespy as ps
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from matplotlib import animation
from copy import copy
from porespy import settings


def set_mpl_style():  # pragma: no cover
    r"""
    Prettifies matplotlib's output by adjusting fonts, markersize etc.
    """
    sfont = 12
    mfont = 12
    lfont = 12

    image_props = {'interpolation': 'none',
                   'cmap': 'viridis'}
    line_props = {'linewidth': 2,
                  'markersize': 8,
                  'markerfacecolor': 'w'}
    font_props = {'size': sfont}
    axes_props = {'titlesize': lfont,
                  'labelsize': mfont,
                  'linewidth': 2,
                  'labelpad': 8}
    xtick_props = {'labelsize': sfont,
                   'top': True,
                   'direction': 'in',
                   'major.size': 6,
                   'major.width': 2}
    ytick_props = {'labelsize': sfont,
                   'right': True,
                   'direction': 'in',
                   'major.size': 6,
                   'major.width': 2}
    legend_props = {'fontsize': mfont,
                    'frameon': False}
    figure_props = {'titlesize': sfont,
                    'autolayout': True}

    plt.rc('font', **font_props)
    plt.rc('lines', **line_props)
    plt.rc('axes', **axes_props)
    plt.rc('xtick', **xtick_props)
    plt.rc('ytick', **ytick_props)
    plt.rc('legend', **legend_props)
    plt.rc('figure', **figure_props)
    plt.rc('image', **image_props)

    if ps.settings.notebook:
        import IPython
        IPython.display.set_matplotlib_formats('svg')


def satn_to_movie(im, satn, cmap='viridis',
                  c_under='grey', c_over='white',
                  v_under=1e-3, v_over=1.0, fps=10, repeat=True):
    r"""

    Notes
    -----
    To save animation as a file use:
    ``ani.save('image_based_ip.gif', writer='imagemagick', fps=3)``
    """
    # Define nice color map
    cmap = copy(plt.cm.get_cmap(name=cmap))
    cmap.set_over(color=c_over)
    cmap.set_under(color=c_under)

    # Reduce inv_satn image to limited number of values to speed-up movie
    target = np.around(satn, decimals=3)
    seq = np.zeros_like(target)  # Empty image to place frame
    movie = []  # List to append each frame
    fig, ax = plt.subplots(1, 1)
    steps = np.unique(target)[1:]
    with tqdm(steps, **settings.tqdm) as pbar:
        for v in steps:
            pbar.update()
            seq += v*(target == v)
            seq[~im] = target.max() + 10
            frame1 = ax.imshow(seq, vmin=v_under, vmax=v_over,
                               animated=True, cmap=cmap, origin='lower',
                               interpolation='none')
            movie.append([frame1])
    ani = animation.ArtistAnimation(fig, movie, interval=int(1000/fps),
                                    blit=True, repeat=repeat,
                                    repeat_delay=1.0)
    return ani
