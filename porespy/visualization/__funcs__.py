import porespy as ps
import matplotlib.pyplot as plt


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
