import imageio
import numpy as np
import os
from zipfile import ZipFile
from porespy.tools import get_tqdm
import scipy.ndimage as spim
from skimage.io import imread_collection


tqdm = get_tqdm()


__all__ = [
    'zip_to_stack',
]


def zip_to_stack(f, trim=True):
    r"""
    Reads a zip file containing 2D slices of a 3D image, and converts to a 3D stack

    Parameters
    ----------
    f : str
        The path and/or file name of the zip archive.  If ``<name>.zip`` is given,
        then it's assumed to be located in the current working directory. Otherwise
        a full path should be given, like ``C:\path\to\file.zip``. Either way, the
        archive is extracted into a folder in the given directory.
    trim : bool
        If ``True`` then a bounding box around the image is found and all excess
        is trimmed.

    Returns
    -------
    im : ndarray
        A 3D numpy array of the imported image.
    """
    fzip = f
    target_dir = fzip.rpartition('.')[0]

    with ZipFile(fzip, 'r') as f:
        f.extractall(target_dir)

    # Method 1: uses skimage and numpy function so is easy to understand
    # filenames = []
    # for f in os.listdir(target_dir):
    #     filenames.append(os.path.join(target_dir, f))
    # files = imread_collection(filenames, conserve_memory=True)
    # im = np.stack(files, axis=2)

    # Method 2: Same speed as 1 but more complex, but allows tqdm progress bar
    test_im = imageio.v2.imread(os.path.join(target_dir, os.listdir(target_dir)[0]))
    im = np.zeros(shape=[test_im.shape[0], test_im.shape[1], len(os.listdir(target_dir))],
                  dtype=test_im.dtype)
    for i, f in enumerate(tqdm(os.listdir(target_dir))):
        im[..., i] = imageio.v2.imread(os.path.join(target_dir, f))

    if trim:
        x, y, z = spim.find_objects(im > 0)[0]
        im = im[x, y, z]

    return im
