import logging
import numpy as np
import scipy.ndimage as spim
from porespy.tools import get_tqdm
from porespy import settings


tqdm = get_tqdm()
logger = logging.getLogger(__name__)


__all__ = [
    'random_cantor_dust',
    'sierpinski_foam',
    'sierpinski_foam2',
]


def random_cantor_dust(shape, n: int = 5, p: int = 2, f: float = 0.8, seed: int = None):
    r"""
    Generates an image of random cantor dust

    Parameters
    ----------
    shape : array_like
        The shape of the final image.  If not evenly divisible by $p**n$
        it will be increased to the nearest size that is.
    n : int
        The number of times to iteratively divide the image.
    p : int (default = 2)
        The number of divisions to make on each iteration.
    f : float (default = 0.8)
        The fraction of the set to keep on each iteration.
    seed : int, optional, default = `None`
        Initializes numpy's random number generator to the specified state. If not
        provided, the current global value is used. This means calls to
        ``np.random.state(seed)`` prior to calling this function will be respected.

    Returns
    -------
    dust : ndarray
        A boolean image of a random Cantor dust

    Examples
    --------
    `Click here
    <https://porespy.org/examples/generators/reference/randon_cantor_dust.html>`_
    to view online example.

    """
    if seed is not None:
        np.random.seed(seed)
    # Parse the given shape and adjust if necessary
    shape = np.array(shape)
    trim = np.mod(shape, (p**n))
    if np.any(trim > 0):
        shape = shape - trim + p**n
        logger.warning(f"Requested shape being changed to {shape}")
    im = np.ones(shape, dtype=bool)
    divs = []
    if isinstance(n, int):
        for i in range(1, n):
            divs.append(p**i)
    else:
        for i in n:
            divs.append(p**i)
    for i in tqdm(divs, **settings.tqdm):
        sh = (np.array(im.shape)/i).astype(int)
        mask = np.random.rand(*sh) < f
        mask = spim.zoom(mask, zoom=i, order=0)
        im = im*mask
    return im


def sierpinski_foam2(shape, n: int = 5):
    r"""
    Generates an image of a Sierpinski carpet or foam with independent control of
    image size and number of iterations

    Parameters
    ----------
    shape : array_like
        The shape of the final image to create. To create a 'centered' image,
        the shape should be ``3**n``.
    n : int
        The number of times to iteratively divide the image. This functions starts
        by inserting single voxels, then inserts increasingly large squares/cubes.

    Returns
    -------
    im : ndarray
        A boolean image with ``False`` values inserted at at the center of each
        square (or cubic) sub-section.

    Notes
    -----
    This function may generate a larger image than need then return the center
    portion of the requested ``shape``, so the edges may be clipped from the
    true Sierpinski foam. This can be avoided by setting shape to some multiple
    of ``3**n``.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/generators/reference/sierpinski_foam2.html>`_
    to view online example.

    """
    im = np.zeros(shape, dtype=bool)
    if im.ndim == 2:
        im[1::3, 1::3] = 1
    else:
        im[1::3, 1::3, 1::3] = 1
    i = 1
    pbar = tqdm()
    while i < n:
        if im.ndim == 2:
            mask = np.zeros([3**(i+1), 3**(i+1)], dtype=bool)
            s = 3**(i+1)//3
            mask[s:-s, s:-s] = 1
            t = int(np.ceil(im.shape[0]/mask.shape[0]))
            im2 = np.tile(mask, [t, t])
            im2 = im2[:im.shape[0], :im.shape[1]]
        if im.ndim == 3:
            mask = np.zeros([3**(i+1), 3**(i+1), 3**(i+1)], dtype=bool)
            s = 3**(i+1)//3
            mask[s:-s, s:-s, s:-s] = 1
            t = int(np.ceil(im.shape[0]/mask.shape[0]))
            im2 = np.tile(mask, [t, t, t])
            im2 = im2[:im.shape[0], :im.shape[1], :im.shape[2]]
        im += im2
        i += 1
        pbar.update()
    pbar.close()
    im = im == 0
    return im


def sierpinski_foam(dmin: int = 1, n: int = 5, ndim: int = 2, max_size: int = 1e9):
    r"""
    Generates an image of a Sierpinski carpet or foam

    Parameters
    ----------
    dmin : int
        The size of the smallest square in the final image
    n : int
        The number of times to iteratively tile the image
    ndim : int
        The number of dimensions of the desired image, can be 2 or 3. The
        default value is 2.

    Returns
    -------
    foam : ndarray
        A boolean image of a Sierpinski gasket or foam

    Examples
    --------
    `Click here
    <https://porespy.org/examples/generators/reference/sierpinski_foam.html>`_
    to view online example.

    """
    def _insert_cubes(im, n):
        if n > 0:
            n -= 1
            shape = np.asarray(np.shape(im))
            im = np.tile(im, (3, 3, 3))
            im[shape[0]:2*shape[0], shape[1]:2*shape[1], shape[2]:2*shape[2]] = 0
            if im.size < max_size:
                im = _insert_cubes(im, n)
        return im

    def _insert_squares(im, n):
        if n > 0:
            n -= 1
            shape = np.asarray(np.shape(im))
            im = np.tile(im, (3, 3))
            im[shape[0]:2*shape[0], shape[1]:2*shape[1]] = 0
            if im.size < max_size:
                im = _insert_squares(im, n)
        return im

    im = np.ones([dmin]*ndim, dtype=int)
    if ndim == 2:
        im = _insert_squares(im, n)
    elif ndim == 3:
        im = _insert_cubes(im, n)
    return im
