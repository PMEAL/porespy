import porespy as ps

path = 'C:\\Users\\Jeff\Dropbox\\Shared\\LatticeBoltzmann\\XRayImages\\'
file = 'SEGMENTED_600x600_WholeThickness'
ext  = 'tif'

im = ps.imopen(filename=path+file+'.'+ext)
rev = ps.REV()
[porosity,size] = rev.run(im)