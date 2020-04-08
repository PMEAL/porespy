import matplotlib.pyplot as plt
import numpy as np


def imshow(im, mask=None, fig=None):
    r"""
    Wrapper for matplotlib's imshow that sets ``False`` and 0 values to white
    regardless of colormap.

    Parameters
    ----------
    im : ND-image
        The image to show
    mask : ND-image
        A mask the same size as ``im`` indicating which locations to show
        (denoted by ``True``) and which to convert to white (denoted by
        ``False``)
        which l
    fig : matplotlib figure object
        The figure to place the plot.  If not given one will be created.

    Returns
    -------
    fig : Matplotlib figure handle

    """
    from matplotlib.pyplot import imshow, figure
    if mask is None:
        mask = (im == 0)
    temp = np.ma.masked_array(data=im, mask=mask)
    if fig is None:
        fig = figure()
    imshow(temp)
    return fig


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
