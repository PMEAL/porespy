import imageio
import numpy as np
import os
from zipfile import ZipFile
from porespy.tools import get_tqdm


tqdm = get_tqdm()


__all__ = [
    'zip_to_stack',
    'folder_to_stack',
]


def folder_to_stack(target_dir):
    r"""
    Opens all images found in the target directory as single 3D numpy array

    Parameters
    ----------
    target_dir : str or path object
        The location of the folder containing the images.

    Returns
    -------
    im : ndarray
        A 3D numpy array of the imported image.

    Notes
    -----
    The files should be named with leading numerical values indicating their
    layer number, like 001, 002, etc.
    """
    test_im = imageio.v2.imread(os.path.join(target_dir, os.listdir(target_dir)[0]))
    im = np.zeros(shape=[test_im.shape[0],
                         test_im.shape[1],
                         len(os.listdir(target_dir))],
                  dtype=test_im.dtype)
    for i, f in enumerate(tqdm(os.listdir(target_dir))):
        im[..., i] = imageio.v2.imread(os.path.join(target_dir, f))

    return im


def zip_to_stack(f):
    r"""
    Reads a zip file containing 2D slices of a 3D image, and converts to a 3D stack

    Parameters
    ----------
    f : str
        The path and/or file name of the zip archive.  If ``<name>.zip`` is given,
        then it's assumed to be located in the current working directory. Otherwise
        a full path should be given, like ``C:\path\to\file.zip``. Either way, the
        archive is extracted into a folder in the given directory.

    Returns
    -------
    im : ndarray
        A 3D numpy array of the imported image.

    Notes
    -----
    The files should be named with leading numerical values indicating their
    layer number, like 001, 002, etc.
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
    im = np.zeros(shape=[test_im.shape[0],
                         test_im.shape[1],
                         len(os.listdir(target_dir))],
                  dtype=test_im.dtype)
    for i, f in enumerate(tqdm(os.listdir(target_dir))):
        im[..., i] = imageio.v2.imread(os.path.join(target_dir, f))

    return im
