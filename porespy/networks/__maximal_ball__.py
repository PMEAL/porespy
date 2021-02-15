import numpy as np
import os
import imageio
import subprocess
import psutil
import time


def maximal_ball(im, filename, path, voxel_size=1e-6):
    r"""
    Implementing a maximal ball algorithm on an image
    
    Parameters
    ----------
    im : ndarray
        The image of the porous material.
    filename : string
        The name to save the files.
    path : string
        path to the maximal ball .exe file (pnextract.exe)
    voxel_size : scalar
        The size of a voxel side in preferred units.  The default is 1e-6, so the
        user can apply the scaling to the returned results after the fact.
  
    Notes
    -----
    outputs four DAT files:
        filename_link1, filename_link2, filename_node1, filename_node2
        
    """
    file = os.path.splitext(filename)[0]
    imageio.volsave(file + ".tif", np.array(im.astype("uint8")))
    f = open(file + ".mhd", "w")
    f.write("ObjectType =  Image\n\
            NDims =	   3 \n\
            ElementType = MET_UCHAR \n\
            DimSize = " + str(im.shape[0]) + " " + str(im.shape[1]) + " " +
            str(im.shape[2]) + "\n\
            ElementSpacing =" + str(voxel_size*1e6) + " " +
            str(voxel_size*1e6) + " " + str(voxel_size*1e6) + "\n\
            Offset = 0   0  0 \n\
            ElementDataFile = " + file + ".tif")
    f.close()
    subprocess.Popen([path, file + ".mhd"])
    i = 0
    while checkIfProcessRunning('pnextract'):
        print('maximal ball algorithm is running: ' + str(i) + 's')
        time.sleep(10)
        i = i+10

def checkIfProcessRunning(processName):
    r"""
    Check if there is any running process that contains the given name.
    """
    for proc in psutil.process_iter():
        try:
            # Check if process name contains the given name string.
            if processName.lower() in proc.name().lower():
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return False
