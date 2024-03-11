import matplotlib.pyplot as plt
import porespy as ps

# Generate an image of spheres using the imgen class
im = ps.generators.blobs(shape=[500, 500], porosity=0.7, blobiness=1)
plt.figure(1)
plt.imshow(im)

# Chord length distribution
chords = ps.filters.apply_chords(im=im, trim_edges=False)
colored_chords = ps.filters.region_size(chords)
h = ps.metrics.chord_length_distribution(chords, bins=25)
ps.visualization.set_mpl_style()
fig, ax = plt.subplots(2, 2)
ax[0][0].imshow(im)
ax[0][1].imshow(chords)
ax[1][0].imshow(colored_chords, cmap=plt.cm.jet)
ax[1][1].bar(h.L, h.pdf, width=h.bin_widths, edgecolor="k")
