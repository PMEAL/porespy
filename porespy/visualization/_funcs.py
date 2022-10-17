import porespy as ps
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from matplotlib import animation
from copy import copy
from porespy import settings


__all__ = [
    'set_mpl_style',
    'satn_to_movie',
    'satn_to_panels',
    'prep_for_imshow',
]


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
        IPython.display.set_matplotlib_formats('retina')


def satn_to_movie(im, satn, cmap='viridis',
                  c_under='grey', c_over='white',
                  v_under=1e-3, v_over=1.0, fps=10, repeat=True):
    r"""
    Converts a saturation map into an animation that can be saved

    Parameters
    ----------
    im : ndarray
        The boolean image of the porous media with ``True`` values indicating
        the void space
    satn : ndaray
        The saturation map such as that produced by an invasion or drainage
        algorithm.
    cmap : str
        The name of the matplotlib color map to use. These are listed on
        matplotlib's website
        `here <https://matplotlib.org/stable/gallery/color/colormap_reference.html>`__
    c_under, c_over : str
        The color to insert for values that are less than `v_under`
        (greater than `v_over`).  The string value of colors are given on
        matplotlib's website
        `here <https://matplotlib.org/stable/gallery/color/named_colors.html>`__
    v_under, v_over : scalar
        The values in ``satn`` that should be considered the lower and upper
        threshold, beyond which the colors given in `c_under` and `c_over`
        are used.
    fps : int
        The frames per second to use when generating the movie.  A higher
        number gives a shorter and faster-paced movie.
    repeat : bool
        If ``True`` the produced animation will rerun repeatedly until
        stopped or closed.

    Notes
    -----
    To save animation as a file use:
    ``ani.save('image_based_ip.gif', writer='imagemagick', fps=3)``

    Examples
    --------
    `Click here
    <https://porespy.org/examples/visualization/reference/satn_to_movie.html>`_
    to view online example.
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


def satn_to_panels(satn, im, bins=None, axis=0, slice=None, **kwargs):
    r"""
    Produces a set of images with each panel containing one saturation

    Parameters
    ----------
    satn : ndarray
        An image with each voxel indicating the global saturation at which
        it was invaded.  0 indicates solid and -1 indicates uninvaded.
    im : ndarray
        A boolean image with ``True`` values indicating the void voxels and
        ``False`` for solid.
    bins : int
        Indicates for which saturations images should be made. If an ``int``
        then a list of equally space values between 0 and 1 is generated.
        If ``None`` (default) than all saturation values in the image are used.
    axis : int, optional
        If the image is 3D, a 2D image is extracted at the specified
        ``slice`` taken along this axis. If the image is 2D this is ignored.
    slice : int, optional
        If the image is 3D, a 2D image is extracted from this slice
        along the given ``axis``.  If ``None``, then a slice at the mid-point
        of the axis is returned.  If 2D this is ignored.
    **kwargs : various
        Additional keyword arguments are sent to the ``imshow`` function,
        such as ``interpolation``.

    Returns
    -------
    fig, ax : Matplotlib figure and axis objects
        The same things as returned by ``plt.subplots``

    Examples
    --------
    `Click here
    <https://porespy.org/examples/visualization/reference/satn_to_panels.html>`_
    to view online example.
    """
    def factors(n):
        return sorted(list(set(
            factor for i in range(1, int(n**0.5) + 1) if n % i == 0
            for factor in (i, n//i)
        )))

    if bins is None:
        Ps = np.unique(satn)
    elif isinstance(bins, int):
        Ps = np.linspace(0, 1, bins+1)[1:]
    Ps = Ps[Ps > 0]
    f = factors(len(Ps))
    if len(Ps) < 4:
        m = 1
        n = len(Ps)
    elif len(f) % 2 == 0:
        m, n = f[int(len(f)/2-1)], f[int(len(f)/2)]
    else:
        m = f[int(len(f)/2)]
        n = m
    fig, ax = plt.subplots(m, n)
    ax = np.atleast_2d(ax)
    temp_old = np.zeros_like(im)
    for i, p in enumerate(Ps):
        temp = (satn <= p)*(satn > 0)
        im_data = prep_for_imshow(im=temp*2.0 - temp_old*1.0, mask=im,
                                  axis=axis, slice=slice)
        im_data.pop('vmax')
        [im_data.pop(i) for i in kwargs]
        ax[i // n][i % n].imshow(**im_data, vmax=2, **kwargs)
        ax[i // n][i % n].set_title(str(np.around(temp.sum()/im.sum(),
                                                  decimals=5)))
        temp_old = np.copy(temp)
    return fig, ax


def prep_for_imshow(im, mask=None, axis=0, slice=None):
    r"""
    Adjusts the range of greyscale values in an image to improve visualization
    by ``matplotlib.pyplot.imshow``

    Parameters
    ----------
    im : ndimage
        The image to show. If ``im`` includes ``+inf`` or ``-inf`` values,
        they are converted to 1 above or below the minimum and maximum finite
        values in ``im``, respectively.
    mask : ndimage, optional
        An image of the porous material with ``True`` indicating voxels of
        interest. The ``False`` voxels are excluded from the ``vmax`` and
        ``vmin`` calculation.
    axis : int, optional
        If the image is 3D, a 2D image is returned with the specified
        ``slice`` taken along this axis (default = 0).  If ``None`` then a 3D
        image is returned. If the image is 2D this is ignored.
    slice : int, optional
        If ``im`` is 3D, a 2D image is be returned showing this slice
        along the given ``axis``.  If ``None``, then a slice at the mid-point
        of the axis is returned.  If ``axis`` is ``None`` or the image is 2D
        this is ignored.

    Returns
    -------
    kwargs : dict
        A python dicionary designed to be passed directly to
        ``matplotlib.pyplot.imshow`` using the "\*\*kwargs" features (i.e.
        ``plt.imshow(\*\*data)``).  It contains the following key-value pairs:

        =============== =======================================================
        key               value
        =============== =======================================================
        'X'             The adjusted image with ``+inf`` replaced by
                        ``vmax + 1``, and all solid voxels replacd by
                        ``np.nan`` to show as white in ``imshow``
        'vmax'          The maximum of ``values`` not including ``+inf`` or
                        values in ``False`` voxels in ``mask``.
        'vmin'          The minimum of ``values`` not including ``-inf`` or
                        values in ``False`` voxels in ``mask``.
        'interpolation' Set to 'none' to avoid artifacts in ``imshow``
        'origin'        Set to 'lower' to put (0, 0) on the bottom-left corner
        =============== =======================================================

    Notes
    -----
    If any of the *extra* items are unwanted they can be removed with
    ``del data['interpolation']`` or ``data.pop('interpolation')``.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/visualization/reference/prep_for_imshow.html>`_
    to view online example.

    """
    # If 3D, fetch 2D slice immediately to save memory
    if (im.ndim == 3) and (axis is not None):
        if slice is None:
            slice = int(im.shape[axis]/2)
        # Rotate image to put given axis first, then slice
        im = np.swapaxes(im, 0, axis)[slice, ...]
        if mask is not None:  # Rotate mask as well, then slice
            mask = np.swapaxes(mask, 0, axis)[slice, ...]
    im = im.astype(float)
    if mask is None:
        mask = np.ones_like(im, dtype=bool)

    vmax = np.amax((im*(im < np.inf))[mask])
    im[(im == np.inf)] = vmax + 1
    vmin = np.amin((im*(im > -np.inf))[mask])
    im[(im == -np.inf)] = vmin - 1

    return {'X': im, 'vmin': vmin, 'vmax': vmax,
            'interpolation': 'none', 'origin': 'lower'}
