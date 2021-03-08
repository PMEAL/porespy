import matplotlib.pyplot as plt


def set_mpl_style():  # pragma: no cover

    sfont = 14
    mfont = 14
    lfont = 14

    line_props = {'linewidth': 2,
                  'markersize': 8}
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
    figure_props = {'titlesize': sfont}

    plt.rc('font', **font_props)
    plt.rc('lines', **line_props)
    plt.rc('axes', **axes_props)
    plt.rc('xtick', **xtick_props)
    plt.rc('ytick', **ytick_props)
    plt.rc('legend', **legend_props)
    plt.rc('figure', **figure_props)
