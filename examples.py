import porespy as ps
import scipy as sp
import scipy.ndimage as spim
import matplotlib.pyplot as plt

# Generate an image of spheres using the imgen class
im = ps.generators.blobs(shape=[500, 500], porosity=0.7, blobiness=1)
plt.figure(1)
plt.imshow(im)

# Chord length distributions
chords = ps.filters.apply_chords(im=im, trim_edges=False)
colored_chords = ps.filters.region_size(chords)
h = ps.metrics.chord_length_distribution(chords, bins=25)
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
