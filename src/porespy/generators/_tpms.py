import numpy as np


__all__ = [
    'tpms_unit_cell',
    'tile_tpms',
]


def tpms_unit_cell(shape, method='schoen', skew=0.5, phi=0.5):
    r"""
    Generate a boolean image of a triply periodic minimal surface (TPMS).

    One full period of the selected surface is mapped onto a cubic domain of
    ``shape`` voxels per side. The surface field is thresholded to produce a
    solid phase, with the band centre and half-width controlled by ``skew``
    and ``phi`` respectively.

    Parameters
    ----------
    shape : int
        Number of voxels along each side of the cubic output array. The output
        shape is ``(shape, shape, shape)``.
    method : str, optional
        The TPMS geometry to generate. Options are:

        ==============  ==========================================================
        Value           Surface
        ==============  ==========================================================
        ``'schoen'``    Schoen Gyroid (default)
        ``'primitive'`` Schwartz Primitive (P surface)
        ``'diamond'``   Schwartz Diamond (D surface)
        ``'diagonal'``  Diagonal surface
        ``'diamond2'``  Diamond variant (second form)
        ``'lidinoid'``  Lidinoid
        ``'split-p'``   Split-P
        ``'neovius'``   Neovius
        ``'FKS'``       Fischer–Koch S surface
        ``'FRD'``       F-RD surface
        ``'pw-hybrid'`` PW-Hybrid
        ``'iWP'``       Schoen I-WP
        ==============  ==========================================================

    skew : float, optional
        Centre of the isovalue band used for thresholding the TPMS field.
        Shifting this value moves the solid phase toward higher or lower field
        values, effectively skewing which region of the surface is selected.
        Default is 0.5.
    phi : float, optional
        Half-width of the isovalue band. Voxels are set to ``True`` where
        ``skew - phi < v < skew + phi``. Larger values produce a thicker solid
        phase and lower porosity. Default is 0.5.

    Returns
    -------
    im : ndarray of bool
        A 3-D boolean array of shape ``(shape, shape, shape)`` where ``True``
        indicates the solid phase defined by the TPMS level set.

    Notes
    -----
    The TPMS field ``v`` is evaluated on a regular grid spanning
    :math:`[0, 2\pi]` in each direction (one full unit cell). The solid phase
    is the set of voxels satisfying :math:`\text{skew} - \phi < v <
    \text{skew} + \phi`.

    Examples
    --------
    Generate a 100-voxel gyroid image:

    >>> import porespy as ps
    >>> im = gyroid(shape=100, method='schoen', phi=0.5)
    >>> im.shape
    (100, 100, 100)

    """
    step = np.linspace(0, 2*np.pi, shape)
    x, y, z = np.meshgrid(step, step, step)

    if method == 'primitive':
        v = np.cos(x) + np.cos(y) + np.cos(z)
    if method == 'schoen':
        v = np.sin(x)*np.cos(y) + np.sin(y)*np.cos(z) + np.sin(z)*np.cos(x)
    if method == 'diamond':
        v = np.cos(x)*np.cos(y)*np.cos(z) + np.sin(x)*np.sin(y)*np.sin(z)
    if method == 'diagonal':
        v = 2*(np.cos(x)*np.cos(y) + np.cos(y)*np.cos(z) + np.cos(z)*np.cos(x)) \
            - (np.cos(2*x) + np.cos(2*y)+np.cos(2*z))
    if method == 'diamond2':
        v = np.sin(x)*np.sin(y)*np.sin(z) + np.sin(x)*np.cos(y)*np.cos(z) \
            + np.cos(x)*np.sin(y)*np.cos(z) + np.cos(x)*np.cos(y)*np.sin(z)
    if method == 'lidinoid':
        v = np.sin(2*x)*np.cos(y)*np.sin(z) + np.sin(x)*np.sin(2*y)*np.cos(z) \
            + np.cos(x)*np.sin(y)*np.sin(2*z) - np.cos(2*x)*np.cos(2*y) \
            - np.cos(2*y)*np.cos(2*z)-np.cos(2*z)*np.cos(2*x) + 0.3
    if method == 'split-p':
        v = 1.1*(np.sin(2*x)*np.cos(y)*np.sin(z)
                 + np.sin(x)*np.sin(2*y)*np.cos(z)
                 + np.cos(x)*np.sin(y)*np.sin(2*z)) \
            - 0.2*(np.cos(2*x)*np.cos(2*y)
                   + np.cos(2*y)*np.cos(2*z)
                   + np.cos(2*z)*np.cos(2*x)) \
            - 0.4*(np.cos(2*x) + np.cos(2*y) + np.cos(2*z))
    if method == 'neovius':
        v = 3.0 * (np.cos(x) + np.cos(x) + np.cos(z)) \
            + 4.0 * np.cos(x)*np.cos(y)*np.cos(z)
    if method == 'FKS':
        v = np.cos(2*x)*np.sin(y)*np.cos(z) + np.cos(2*y)*np.sin(z)*np.cos(x) \
            + np.cos(2*z)*np.sin(x)*np.cos(y)
    if method == 'FRD':
        v = 4*np.cos(x)*np.cos(y)*np.cos(z) \
            - (np.cos(2*x)*np.cos(2*y) + np.cos(2*x)*np.cos(2*z)
               + np.cos(2*y)*np.cos(2*z))
    if method == 'pw-hybrid':
        v = 4.0*(np.cos(x)*np.cos(y) + np.cos(y)*np.cos(z) + np.cos(z)*np.cos(x)) \
            - 3*np.cos(x)*np.cos(y)*np.cos(z) + 2.4
    if method == 'iWP':
        v = 2.0*(np.cos(x)*np.cos(y) + np.cos(z)*np.cos(x) + np.cos(y)*np.cos(z)) \
            - (np.cos(2*x) + np.cos(2*y) + np.cos(2*z))

    im = (v > (skew - phi))*(v < (skew + phi))
    return im


def tile_tpms(im, n, mode='periodic'):
    r"""
    Tile a TPMS image to create a larger domain.

    Parameters
    ----------
    im : ndarray
        The 3-D boolean image to tile, typically produced by `gyroid`.
    n : int or sequence of int
        Number of repetitions along each axis. A single integer tiles equally
        in all directions; a sequence (e.g. ``(3, 3, 1)``) tiles each axis
        independently.
    mode : str, optional
        Tiling strategy. Options are:

        ============  =========================================================
        Value         Behaviour
        ============  =========================================================
        ``'periodic'`` Repeat the image using simple copy (default).
        ``'reflect'``  Mirror the image at each boundary so that features meet
                       smoothly across tile edges.
        ============  =========================================================

    Returns
    -------
    im2 : ndarray
        Tiled boolean array. Its shape along each axis is
        ``im.shape[i] * n[i]``.

    Examples
    --------
    Generate one gyroid unit cell and tile it 3x3 in-plane:

    >>> im = gyroid(shape=100)
    >>> im2 = tile(im, n=(3, 3, 1), mode='periodic')
    >>> im2.shape
    (300, 300, 100)

    """
    if np.isscalar(n):
        n = (n,) * im.ndim
    if mode == 'periodic':
        im2 = np.tile(im, n)
    elif mode == 'reflect':
        shape = np.array(im.shape)*(np.array(n)-1)
        pw = [(0, shape[i]) for i in range(im.ndim)]
        im2 = np.pad(im, pad_width=pw, mode='reflect')
    return im2


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import porespy as ps
    ps.visualization.set_mpl_style()

    im = tpms_unit_cell(shape=300, phi=0.5, skew=0.5, method='schoen')
    im2 = tile_tpms(im, n=(3, 3, 1), mode='periodic')
    print(im2.shape)
    fig, ax = plt.subplots(2, 2)
    ax[0][0].imshow(ps.visualization.sem(im2, axis=0), cmap=plt.cm.bone, vmax=1.5)
    ax[0][0].axis(False)
    ax[0][1].imshow(ps.visualization.xray(im2, axis=2), cmap=plt.cm.bone)
    ax[0][1].axis(False)
    ax[1][0].imshow(ps.visualization.sem(im2, axis=2).T, cmap=plt.cm.bone, vmax=1.5)
    ax[1][0].axis(False)
    ax[1][1].imshow(ps.visualization.xray(im2, axis=0), cmap=plt.cm.bone)
    ax[1][1].axis(False)
