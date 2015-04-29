# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 08:46:37 2015

@author: Jeff
"""
path = 'C:\\Users\\Jeff\\Dropbox\\Shared\\LatticeBoltzmann\\XRayImages\\'
file = 'SEGMENTED.1500x1500_XY'
ext = 'tif'

import scipy as sp
import scipy.ndimage as spim
import matplotlib.pyplot as plt
import tifffile as tff

tiffimg = tff.TIFFfile(path+file+'.'+ext)
im = tiffimg.asarray()

# Choose N random center points
N = 50
Cx = sp.random.randint(0,sp.shape(im)[0],N)
Cy = sp.random.randint(0,sp.shape(im)[1],N)
Cz = sp.random.randint(0,sp.shape(im)[2],N)




plt.figure(0)
plt.subplot(1,2,1)
plt.imshow(im)
plt.subplot(1,2,2)
plt.plot(size,porosity,'b.')