import numpy as np
import scipy.ndimage as spim
from porespy.tools import get_tqdm
from porespy import settings
from loguru import logger
tqdm = get_tqdm()


def random_cantor_dust(shape, n, p=2, f=0.8):
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


def sierpinski_foam(dmin, n, ndim=2, max_size=1e9):
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
