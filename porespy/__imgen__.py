import scipy as sp
import scipy.ndimage as spim


class ImageGenerator():
    r"""
    This class contains various methods for generating test images
    """

    @staticmethod
    def spheres(shape, radius, porosity):
        r"""
        Generate a packing of overlapping mono-disperse spheres

        Parameters
        ----------
        shape : list
            The size of the image to generate in [Nx, Ny, Nz] where N is the
            number of voxels.

        radius : scalar
            The radius of spheres in the packing

        porosity : scalar
            The porosity of the final image.  This number is approximated by
            the method so the returned result may not have exactly the
            specified value.

        Notes
        -----
        This method can also be used to generate a dispersion of hollows by
        treating ``porosity`` as solid volume fraction and inverting the
        returned image.
        """
        if sp.size(shape) == 1:
            shape = sp.full((3, ), int(shape))
        im = sp.zeros(shape, dtype=bool)
        while sp.sum(im)/sp.size(im) < (1 - porosity):
            temp = sp.rand(shape[0], shape[1], shape[2]) < 0.9995
            im = im + (spim.distance_transform_edt(temp) < radius)
        return ~im

    @staticmethod
    def blobs(shape, porosity, blobiness=8):
        """
        Generates an image containing amorphous blobs

        Parameters
        ----------
        shape : list
            The size of the image to generate in [Nx, Ny, Nz] where N is the
            number of voxels

        blobiness : scalar
            Controls the morphology of the image.  A higher number results in
            a larger number of smaller blobs.

        porosity : scalar
            The porosity of the final image.  This number is approximated by
            the method so the returned result may not have exactly the
            specified value.

        """
        if sp.size(shape) == 1:
            shape = sp.full((3, ), int(shape))
        [Nx, Ny, Nz] = shape
        sigma = sp.mean(shape)/(4*blobiness)
        mask = sp.rand(Nx, Ny, Nz)
        mask = spim.gaussian_filter(mask, sigma=sigma)
        hist = sp.histogram(mask, bins=1000)
        cdf = sp.cumsum(hist[0])/sp.size(mask)
        xN = sp.where(cdf >= porosity)[0][0]
        im = mask <= hist[1][xN]
        return im
