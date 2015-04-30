import scipy as sp
import porespy as ps
import matplotlib.pyplot as plt

path = 'C:\\Users\\Jeff\Dropbox\\Shared\\LatticeBoltzmann\\XRayImages\\'
file = 'SEGMENTED_600x600_WholeThickness'
ext  = 'tif'

im = ps.imopen(filename=path+file+'.'+ext)
rev = ps.REV()
[porosity,size] = rev.run(im,N=300)
plt.style.use('ggplot')
plt.plot(size,porosity,'b.')
sp.savetxt('rev.csv',sp.vstack((size,porosity)).T)