import scipy as sp
import scipy.ndimage as spim
import porespy as ps
import matplotlib.pyplot as plt
import timeit

#==============================================================================
# import imageio
# # Import an image file as an ND-array
# path = 'C:\\Users\\Jeff\\Dropbox\\Flash Sync\\Code\\Git\\PoreSpy\\test\\fixtures'
# file = 'voronoi_fibers'
# ext = 'tif'
# im = imageio.mimread(path+'\\'+file+'.'+ext)
#==============================================================================

start_time = timeit.default_timer()

# Generate an image of spheres
im = sp.rand(330, 330, 1) < 0.997
im = spim.distance_transform_bf(im) >= 4
print(timeit.default_timer() - start_time)

#==============================================================================
# # Perform CLD calculation and get image for illustration
# a = ps.cld(image=im)
# cx = a.xdir(spacing=5, trim_edges=True)
# print(timeit.default_timer() - start_time)
# cim = a.get_chords(direction='x', spacing=5, trim_edges=True)
# print(timeit.default_timer() - start_time)
#
# # Plot some slices
# plt.subplot(2, 2, 1)
# plt.imshow(im[:, :, 7])
# plt.subplot(2, 2, 3)
# plt.imshow(im[:, :, 7]*1.0 - cim[:, :, 7]*0.5)
# plt.subplot(2, 2, 2)
# plt.plot(cx)
# plt.subplot(2, 2, 4)
# plt.plot(sp.log10(cx))
#==============================================================================

#==============================================================================
# # Perform REV calculation
# b = ps.rev(image=im)
# vals = b.run(N=500)
# # Plot results
# plt.plot(vals.size, vals.porosity, 'bo')
#==============================================================================

# Perform MIO simulation
self = ps.mio(image=im)
self.run()
