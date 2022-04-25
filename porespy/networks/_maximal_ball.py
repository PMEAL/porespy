import os
import time
import psutil
import subprocess
import numpy as np
from loguru import logger
import imageio


def maximal_ball_wrapper(im, prefix, path_to_exe, voxel_size=1e-6):
    r"""
    Implementing a maximal ball algorithm on an image

    Parameters
    ----------
    im : ndarray
        The image of the porous material.
    prefix : string
        The prefix to append to the filenames (i.e. 'prefix_node1.dat')
    path_to_exe : string
        Path to the maximal ball .exe file (pnextract.exe). See Notes
    voxel_size : scalar
        The size of a voxel side in preferred units.  The default is 1e-6, so the
        user can apply the scaling to the returned results after the fact.

    Notes
    -----
    outputs four DAT files:
        prefix_link1, prefix_link2, prefix_node1, prefix_node2

    This function only runs on Windows since the Windows compatible binary is
    provided by the Imperial College team.

    """
    file = os.path.splitext(prefix)[0]
    imageio.volsave(file + ".tif", np.array(im.astype("uint8")))
    with open(f"{file}.mhd", "w") as f:
        f.write("ObjectType =  Image\n\
                 NDims =	   3 \n\
                 ElementType = MET_UCHAR \n\
                 DimSize = " + str(im.shape[0]) + " " + str(im.shape[1]) + " " +
                 str(im.shape[2]) + "\n\
                 ElementSpacing =" + str(voxel_size*1e6) + " " +
                 str(voxel_size*1e6) + " " + str(voxel_size*1e6) + "\n\
                 Offset = 0   0  0 \n\
                 ElementDataFile = " + file + ".tif")
    subprocess.Popen([path_to_exe, file + ".mhd"])
    time_elapsed = 0
    while _is_running('pnextract'):
        logger.trace('Maximal ball algorithm running for {time_elapsed} s')
        time.sleep(10)
        time_elapsed += 10


def _is_running(process_name):
    r"""
    Check if there is any running process that contains the given name.
    """
    for proc in psutil.process_iter():
        try:
            # Check if process name contains the given name string.
            if process_name.lower() in proc.name().lower():
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return False
