import scipy as sp
import porespy as ps
import matplotlib.pyplot as plt
plt.style.use('ggplot')

path = 'C:\\Users\\Jeff\Dropbox\\Shared\\LatticeBoltzmann\\XRayImages\\'
file = 'SEGMENTED.1500x1500_XY'
ext  = 'tif'

im = 1 - ps.imopen(filename=path+file+'.'+ext)

#[distance, count] = ps.TPC.run(img=im,npoints=5000)
#plt.plot(distance,count)

[porosity,size] = ps.REV.run(im,N=100)
plt.plot(size,porosity,'b.')

#sp.savetxt('rev.csv',sp.vstack((size,porosity)).T)