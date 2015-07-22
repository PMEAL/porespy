import scipy as sp
import scipy.ndimage as spim
import porespy as ps
import matplotlib.pyplot as plt
import timeit
start_time = timeit.default_timer()

# Generate an image of spheres using the imgen class
#im = ps.imgen.spheres(shape=100, radius=4, porosity=0.9)
im = ps.imgen.blobs(shape=100, porosity=0.7, blobiness=8)
print(timeit.default_timer() - start_time)

#==============================================================================
# import imageio
# # Import an image file as an ND-array
# path = 'C:\\Users\\Jeff\\Dropbox\\Flash Sync\\Code\\Git\\PoreSpy\\test\\fixtures'
# file = 'voronoi_fibers'
# ext = 'tif'
# im = imageio.mimread(path+'\\'+file+'.'+ext)
#==============================================================================





#==============================================================================
# # Perform PoreSizeFunction calculation
# self = ps.psf(image=im)
# a = self.run()
# plt.loglog(a.distance, a.frequency, 'bo')
#==============================================================================

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

#==============================================================================
# # Perform MIO simulation
# self = ps.mio(image=im)
# self.run()
# pcsnwp = self.drainage_curve()
# im = self.get_fluid_image(size=3)
# plt.imshow(im[:, :, 50]*0.2 + ~self.image[:, :, 50], interpolation='none')
#==============================================================================

#==============================================================================
# # Perform TPC calculation
# self = ps.tpc(image=im)
# a = self.run(npts=200, r=sp.arange(1, 30, 3))
# plt.plot(a.distance, a.probability, 'b-x')
# print(timeit.default_timer() - start_time)
#==============================================================================
