# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 08:46:37 2015

@author: Jeff
"""
path = 'C:\\Users\\Jeff\\Dropbox\\Shared\\LatticeBoltzmann\\XRayImages\\Wet\\'
file = 'SEGMENTED.1500x1500_XY'
ext  = 'tif'

import scipy as sp
import scipy.ndimage as spim
import matplotlib.pyplot as plt
import tifffile as tff

tiffimg = tff.TIFFfile(path+file+'.'+ext)
im = tiffimg.asarray()
im = sp.swapaxes(im, 0, 2)
#im = sp.pad(im,pad_width=((0,0),(0,0),(20,20)),mode='constant',constant_values=0)

# Choose N random center points
N = 500
pad = [0.1,0.1,0.45]
Cx = sp.random.randint(pad[0]*sp.shape(im)[0],(1-pad[0])*sp.shape(im)[0],N)
Cy = sp.random.randint(pad[1]*sp.shape(im)[1],(1-pad[1])*sp.shape(im)[1],N)
Cz = sp.random.randint(pad[2]*sp.shape(im)[2],(1-pad[2])*sp.shape(im)[2],N)
C = sp.vstack((Cx,Cy,Cz)).T

#Find maximum radius allowable for each point
Rmax = sp.array(C>sp.array(sp.shape(im))/2)
Rlim = sp.zeros(sp.shape(Rmax))
Rlim[Rmax[:,0],0] = sp.shape(im)[0]
Rlim[Rmax[:,1],1] = sp.shape(im)[1]
Rlim[Rmax[:,2],2] = sp.shape(im)[2]
R = sp.absolute(C-Rlim)
R = R.astype(sp.int_)
Rmin = sp.amin(R,axis=1)

vol = []
size = []
porosity = []
for i in range(0,N):
    for r in sp.arange(Rmin[i],1,-10):
        imtemp = im[C[i,0]-r:C[i,0]+r,C[i,1]-r:C[i,1]+r,C[i,2]-r:C[i,2]+r]
        vol.append(sp.size(imtemp))
        size.append(2*r)
        porosity.append(sp.sum(imtemp==1)/(sp.size(imtemp)))
#        porosity.append(sp.sum(imtemp==2)/(sp.sum(imtemp==0)+sp.sum(imtemp==2)))

plt.figure(0)
plt.style.use('ggplot')
plt.plot(size,porosity,'b.')


sp.savetxt('REV.csv',sp.vstack((size,porosity)).T)