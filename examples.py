import porespy as ps
import scipy as sp
import scipy.ndimage as spim
import matplotlib.pyplot as plt
# import timeit
# start_time = timeit.default_timer()
# print(timeit.default_timer() - start_time)

# Generate an image of spheres using the imgen class
# im = ps.imgen.spheres(shape=100, radius=4, porosity=0.9)
im = ps.generators.blobs(shape=[500, 500], porosity=0.7, blobiness=1)

plt.figure(1)
plt.imshow(im)


# Chord length distributions
chords = ps.filters.apply_chords(im=im, trim_edges=False)
colored_chords = ps.filters.flood(im=chords, mode='size')
h = ps.metrics.chord_length_distribution(chords)
ps.visualization.set_mpl_style()
plt.figure(2)
plt.subplot(2, 2, 1)
plt.imshow(im)
plt.subplot(2, 2, 3)
plt.imshow(chords)
plt.subplot(2, 2, 4)
plt.imshow(colored_chords, cmap=plt.cm.jet)
plt.subplot(2, 2, 2)
plt.bar(h.L, h.pdf, width=h.bin_widths, edgecolor='k')
