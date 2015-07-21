import scipy as sp
import scipy.ndimage as spim


class ChordLengthDistribution(object):
    r"""
    Performs Chord-length distribution analyses

    Examples
    --------
    Generate a test image of a sphere pack:

    >>> import scipy as sp
    >>> import scipy.image as spim
    >>> im = sp.rand(40, 40, 40) < 0.997
    >>> im = spim.distance_transform_bf(im) >= 4

    Import porespy and use it:

    >>> import porespy
    >>> a = porespy.cld(im)
    >>> cx = a.xdir(spacing=5, trim_edges=True)
    >>> cim = a.get_chords(direction='x', spacing=5, trim_edges=True)

    Visualize with Matplotlib

    .. code-block:: python

        import matplotlib as plt
        plt.subplot(2, 2, 1)
        plt.imshow(im[:, :, 7])
        plt.subplot(2, 2, 3)
        plt.imshow(im[:, :, 7]*1.0 - cim[:, :, 7]*0.5)
        plt.subplot(2, 2, 2)
        plt.plot(cx)
        plt.subplot(2, 2, 4)
        plt.plot(sp.log10(cx))

    """

    def __init__(self, image):
        super().__init__()
        image = sp.atleast_3d(image)
        self.image = sp.array(image, dtype=bool)

    def xdir(self, spacing=10, trim_edges=True):
        r'''
        '''
        image = self.image
        temp = self._distribution(image=image,
                                  spacing=spacing,
                                  trim_edges=trim_edges)
        return temp

    def ydir(self, spacing=10, trim_edges=True):
        r'''
        '''
        image = sp.transpose(self.image, [1, 0, 2])
        temp = self._distribution(image=image,
                                  spacing=spacing,
                                  trim_edges=trim_edges)
        return temp

    def zdir(self, spacing=10, trim_edges=True):
        r'''
        '''
        image = sp.transpose(self.image, [2, 1, 0])
        temp = self._distribution(image=image,
                                  spacing=spacing,
                                  trim_edges=trim_edges)
        return temp

    def ndir(self, spacing=10, rotation=0, trim_edges=True):
        r'''
        '''
        image = spim.rotate(self.image, axes=[1, 2], angle=rotation)
        temp = self._distribution(image=image,
                                  spacing=spacing,
                                  trim_edges=trim_edges)
        return temp

    def get_chords(self, direction, spacing=10, trim_edges=True):
        if direction == 'x':
            swap_axes = [0, 1, 2]
        elif direction == 'y':
            swap_axes = [1, 0, 2]
        elif direction == 'z':
            swap_axes = [2, 1, 0]
        image = sp.transpose(self.image, swap_axes)
        image = self._apply_chords(image=image,
                                   spacing=spacing,
                                   trim_edges=trim_edges)
        image = sp.transpose(image, swap_axes)
        return image

    def _apply_chords(self, image, spacing, trim_edges):
        r'''
        This method returns a copy of the image with chords applied, solely for
        visualization purposes.  The actual determination of the chord length
        distribution does not need to this.

        Notes
        -----
        This private method is called by the varioius public methods which
        rotate the image correctly prior to sending, then rotate it back upon
        receipt
        '''
        # Extract size metrics from input image
        [Lx, Ly, Lz] = sp.shape(image)
        start = sp.array(sp.floor(spacing/2), dtype=int)
        Y = sp.arange(start, Ly, spacing)
        Z = sp.arange(start, Lz, spacing)
        temp = sp.zeros([Lx, Ly, Lz], dtype=int)
        # Generate 2D mask of chords in X-dir
        maskX = sp.zeros([Lx, Ly, 1], dtype=bool)
        maskX[:, Y, :] = 1
        # Apply chord mask to specified layers (Z-dir) of input image
        temp[:, :, Z] = image[:, :, Z]*maskX
        if trim_edges:
            temp[[0, -1], :, :] = 1
            temp[:, [0, -1], :] = 1
            L = spim.label(temp)[0]
            ind = sp.where(L == L[0, 0, 0])
            temp[ind] = 0

        return sp.array(temp, dtype=bool)

    def _distribution(self, image, spacing, rotation=0, trim_edges=True):
        r'''
        '''
        # Clean up input image
        img = sp.array(image, ndmin=3, dtype=int)
        # Extract size metrics from input image
        [Lx, Ly, Lz] = sp.shape(img)
        start = sp.array(sp.floor(spacing/2), dtype=int)
        Y = sp.arange(start, Ly, spacing)
        Z = sp.arange(start, Lz, spacing)
        [y, z] = sp.meshgrid(Y, Z, indexing='ij')
        y = y.flatten()
        z = z.flatten()
        bins = sp.zeros(sp.amax(sp.shape(img))+1, dtype=int)
        for yi, zi in zip(y, z):
            a = self._find_blocks(img[:, yi, zi], trim_edges=trim_edges)
            bins[a['length']] += 1
        return bins

    def _find_blocks(self, array, trim_edges=False):
        array = sp.clip(array, a_min=0, a_max=1)
        temp = sp.pad(array, pad_width=1, mode='constant', constant_values=0)
        end_pts = sp.where(sp.ediff1d(temp) == -1)[0]  # Find 1->0 transitions
        end_pts -= 1  # To adjust for 0 padding
        seg_len = sp.cumsum(array)[end_pts]
        seg_len[1:] = seg_len[1:] - seg_len[:-1]
        start_pts = end_pts - seg_len + 1
        a = dict()
        a['start'] = start_pts
        a['end'] = end_pts
        a['length'] = seg_len
        if trim_edges:
            if (a['start'].size > 0) and (a['start'][0] == 0):
                [a.update({item: a[item][1:]}) for item in a]
            if (a['end'].size > 0) and (a['end'][-1] == sp.size(array)-1):
                [a.update({item: a[item][:-1]}) for item in a]
        return a
