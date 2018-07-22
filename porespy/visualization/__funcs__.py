import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy as sp
import scipy.ndimage as spim
from porespy.visualization import sem


def set_mpl_style():

    sfont = 25
    mfont = 35
    lfont = 45

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


def show_slices(im, n=1, visible_phase=0, stride=1):
    r"""
    Show a view of the slices though the material with solids set to visible.

    Parameter
    ---------
    im : ND-image
        The image of the porous material

    n : int (default = 1)
        The number of slices to show in the z-direction.  For ``n = 1``, the
        middle slice is show, for all higher values, the top and bottom are
        always shown, with the remainder evenly spaced through the image.  For
        2D images, this is ignored since there is only one slice.

    visible_phase : int (default = 0)
        This controls which phase is set to visible in the slices.  The default
        is the solid phase (False or 0), but any value can be used, provided
        that it is present in the image.

    stride : int (default = 1)
        Controls how many of the voxels are represented in the image.  If
        set to 1, then every voxel is shown by a marker, which could get
        slow to plot for large images.  Higher values skip voxels to produce
        a more sparse image.

    Returns
    -------
    This function creates a Matplotlib figure and automatically shows it.

    Notes
    -----
    In order to handle 2D and 3D images equally, this function creates a
    scatter plot for each slice and puts markers at the location of the
    ``visible_phase``.  For 2D image, users can just call matplotlib's
    ``imshow`` or ``matshow`` functions to get more standard image.

    """

    if im.ndim == 2:
        slices = [0]
    elif n == 1:
        slices = [int(im.shape[2]/2)]
    else:
        slices = sp.linspace(0, im.shape[2], n).astype(int)
        slices[-1] -= 1
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for n in slices:
        pts = sp.where(im[::stride, ::stride, n] == visible_phase)
        ax.plot(*pts, zs=n, marker='.', linestyle='None')
    plt.axis('equal')


def show_3D(im, visible_phase=0):
    r"""
    Shows a view of the image stack from an angle with solids set to visible.

    Parameters
    ----------
    im : ND-array
        A 3D image of the porous material.

    visible_phase : int (default = 0)
        This controls which phase is set to visible in the slices.  The default
        is the solid phase (False or 0), but any value can be used, provided
        that it is present in the image.

    Notes
    -----
    This function first rotates the image, then passes it to the ``sem`` view.
    The end result is a reasonbly decent view of the stack that is useful to
    ensure everything is looking as it's supposed to.

    """

    im = im == visible_phase
    rot = spim.rotate(input=im, angle=35, axes=[2, 0], order=0,
                      mode='constant', cval=1)
    rot = spim.rotate(input=rot, angle=25, axes=[1, 0], order=0,
                      mode='constant', cval=1)
    plt.imshow(sem(rot), cmap=plt.cm.bone)
    plt.axis('off')
