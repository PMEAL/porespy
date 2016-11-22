import scipy as sp
import scipy.ndimage as spim


class ImageGenerator():
    r"""
    This class contains various methods for generating test images
    """

    @staticmethod
    def sphere_pack(shape, radius, spacing=None, packing='sc'):
        r"""
        Generates a cubic packing of spheres

        Parameters
        ----------
        shape : list
            The size of the image to generate in [Nx, Ny, Nz] where N is the
            number of voxels.

        radius : scalar
            The radius of spheres in the packing

        spacing : scalar
            The spacing between pore centers.  If none is supplied then the
            spheres will be packed to the maximum solid fraction.  If a
            spacing is specified that is closer than the maximum, then the
            spheres will overlap.

        packing : string
            Specifies the type of cubic packing to create.  Options are:

            'sc' : Simple Cubic (default)
            'fcc' : Face Centered Cubic
            'bcc' : Body Centered Cubic

        Returns
        -------
        A boolean array with True values denoting the pore space
        """
        r = radius
        if sp.size(shape) == 1:
            shape = sp.full((3, ), int(shape))
        if sp.size(shape) == 2:
            raise Exception('This method can only produce 3D images')
        im = sp.zeros(shape, dtype=bool)
        if packing.startswith('s'):
            if spacing == None:
                spacing = 2*r
            s = int(spacing/2)
            coords = sp.mgrid[r:im.shape[0]-r:2*s,
                              r:im.shape[1]-r:2*s,
                              r:im.shape[2]-r:2*s]
            im[coords[0], coords[1], coords[2]] = 1
        elif packing.startswith('b'):
            if spacing == None:
                spacing = 2*sp.floor(sp.sqrt(4/3*(r**2))).astype(int)
            s = int(spacing/2)
            coords = sp.mgrid[r:im.shape[0]-r:2*s,
                              r:im.shape[1]-r:2*s,
                              r:im.shape[2]-r:2*s]
            im[coords[0], coords[1], coords[2]] = 1
            coords = sp.mgrid[s+r:im.shape[0]-r:2*s,
                              s+r:im.shape[1]-r:2*s,
                              s+r:im.shape[2]-r:2*s]
            im[coords[0], coords[1], coords[2]] = 1
        elif packing.startswith('f'):
            if spacing == None:
                spacing = 2*sp.floor(sp.sqrt(2*(r**2))).astype(int)
            s = int(spacing/2)
            coords = sp.mgrid[r:im.shape[0]-r:2*s,
                              r:im.shape[1]-r:2*s,
                              r:im.shape[2]-r:2*s]
            im[coords[0], coords[1], coords[2]] = 1
            coords = sp.mgrid[r:im.shape[0]-r:2*s,
                              s+r:im.shape[1]-r:2*s,
                              s+r:im.shape[2]-r:2*s]
            im[coords[0], coords[1], coords[2]] = 1
            coords = sp.mgrid[s+r:im.shape[0]-r:2*s,
                              s:im.shape[1]-r:2*s,
                              s+r:im.shape[2]-r:2*s]
            im[coords[0], coords[1], coords[2]] = 1
            coords = sp.mgrid[s+r:im.shape[0]-r:2*s,
                              s+r:im.shape[1]-r:2*s,
                              s:im.shape[2]-r:2*s]
            im[coords[0], coords[1], coords[2]] = 1
        im = spim.distance_transform_edt(~im) > r
        return im

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

        Returns
        -------
        A boolean array with True values denoting the pore space

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

        Returns
        -------
        A boolean array with True values denoting the pore space

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

    @staticmethod
    def spatially_correlated(shape, porosity, weights=None, strel=None):
        r"""
        Generates pore seeds that are spatailly correlated with their neighbors.

        Parameters
        ----------
        shape : list
            The size of the image to generate in [Nx, Ny, Nz] where N is the
            number of voxels

        porosity : scalar
            The final porosity of the image (fraction of 1's).  The image is
            thresholded then denoised using the same structuring element that
            was used to generate the image.  The final porosity will only be
            approximate.

        weights : list of ints, optional
            The [Nx,Ny,Nz] distances (in number of pores) in each direction that
            should be correlated.

        strel : array_like, optional (in place of weights)
            The option allows full control over the spatial correlation pattern by
            specifying the structuring element to be used in the convolution.

            The array should be a 3D array containing the strength of correlations
            in each direction.  Nonzero values indicate the strength, direction
            and extent of correlations.  The following would achieve a basic
            correlation in the z-direction:

            strel = sp.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]], \
                              [[0, 0, 0], [1, 1, 1], [0, 0, 0]], \
                              [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])

        Returns
        -------
        A boolean array with True values denoting the pore space

        """
        import scipy.ndimage as spim
        # The following will only work on Cubic networks
        x, y, z = shape
        im = sp.rand(x, y, z)
        if strel is None:  # Then generate a strel
            if sum(weights) == 0:
                # If weights of 0 are sent, then skip everything and return rands.
                return im.flatten()
            w = sp.array(weights)
            strel = sp.zeros(w*2+1)
            strel[:, w[1], w[2]] = 1
            strel[w[0], :, w[2]] = 1
            strel[w[0], w[1], :] = 1
        im = spim.convolve(im, strel)
        # Convolution is no longer randomly distributed, so fit a gaussian
        # and find it's seeds
        im = (im - sp.mean(im))/sp.std(im)
        im = 1/2*sp.special.erfc(-im/sp.sqrt(2))
        im = im < 0.9*porosity
        im = spim.median_filter(input=im, size=3)
        return im

    @staticmethod
    def fibers(shape, radius, nfibers, phi_max=0, theta_max=90):
        r"""
        Generates a binary image of overlapping fibers.

        Parameters
        ----------
        phi_max : scalar
            A value between 0 and 90 that controls the amount that the fibers
            lie out of the XY plane, with 0 meaning all fibers lie in the XY
            plane, and 90 meaning that fibers are randomly oriented out of the
            plane by as much as +/- 90 degrees.
        theta_max : scalar
            A value between 0 and 90 that controls the amount rotation in the
            XY plane, with 0 meaning all fibers point in the X-direction, and
            90 meaning they are randomly rotated about the Z axis by as much
            as +/- 90 degrees.

        Returns
        -------
        A boolean array with True values denoting the pore space
        """
        shape = sp.array(shape)
        im = sp.zeros(shape)
        R = sp.sqrt(sp.sum(sp.square(shape)))
        n = 0
        while n < nfibers:
            x = sp.rand(3)*shape
            phi = sp.deg2rad(90 + 90*(0.5 - sp.rand())*phi_max/90)
            theta = sp.deg2rad(180 - 90*(0.5 - sp.rand())*2*theta_max/90)
            X0 = R*sp.array([sp.sin(theta)*sp.cos(phi),
                             sp.sin(theta)*sp.sin(phi),
                             sp.cos(theta)])
            [X0, X1] = [X0 + x, -X0 + x]
            crds = line_segment(X0, X1)
            lower = ~sp.any(sp.vstack(crds).T < [0, 0, 0], axis=1)
            upper = ~sp.any(sp.vstack(crds).T >= shape, axis=1)
            valid = upper*lower
            if sp.any(valid):
                im[crds[0][valid], crds[1][valid], crds[2][valid]] = 1
                n += 1
        im = sp.array(im, dtype=bool)
        dt = spim.distance_transform_edt(~im) < radius
        return ~dt


def line_segment(X0, X1):
    r"""
    Calculate the voxel coordinates of a straight line between the two given
    end points

    Parameters
    ----------
    X0 and X1 : array_like
        The [x, y, z] coordinates of the start and end points of the line.

    Returns
    -------
        A list of lists containing the X, Y, and Z coordinates of all voxels
        that should be drawn between the start and end points to create a solid
        line.
    """
    X0 = sp.around(X0)
    X1 = sp.around(X1)
    L = sp.amax(sp.absolute([[X1[0]-X0[0]], [X1[1]-X0[1]], [X1[2]-X0[2]]])) + 1
    x = sp.rint(sp.linspace(X0[0], X1[0], L)).astype(int)
    y = sp.rint(sp.linspace(X0[1], X1[1], L)).astype(int)
    z = sp.rint(sp.linspace(X0[2], X1[2], L)).astype(int)
    return [x, y, z]
