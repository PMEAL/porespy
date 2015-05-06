import scipy as sp
import scipy.ndimage as spim
import porespy as ps
import matplotlib.pyplot as plt
plt.style.use('ggplot')

#path = 'C:\\Users\\Jeff\Dropbox\\Shared\\LatticeBoltzmann\\XRayImages\\'
#file = 'SEGMENTED.1500x1500_XY'
#ext  = 'tif'
#im = 1 - ps.imopen(filename=path+file+'.'+ext)

im = sp.rand(30,30,30)<0.998
im = spim.distance_transform_bf(im)>4

#[distance, count] = ps.tpc.run(img=im,npoints=5000)
#plt.plot(distance,count)

#[porosity,size] = ps.rev.run(im,N=100)
#plt.plot(size,porosity,'b.')

#sp.savetxt('rev.csv',sp.vstack((size,porosity)).T)