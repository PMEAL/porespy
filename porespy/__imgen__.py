import scipy as sp
import scipy.spatial as sptl
import scipy.ndimage as spim
from skimage.segmentation import find_boundaries
from skimage.morphology import ball, disk, square, rectangle, watershed


class ImageGenerators():
    r"""
    This class contains various methods for generating test images
    """

    @staticmethod
    @staticmethod
    def circle_pack(shape, radius, offset=0, packing='square'):
        r"""
        Generates a 2D packing of circles

        Parameters
        ----------
        shape : list
            The size of the image to generate in [Nx, Ny] where N is the
            number of voxels in each direction

        radius : scalar
            The radius of circles in the packing (in pixels)

        offset : scalar
            The amount offset (+ or -) to add between pore centers (in pixels).

        packing : string
            Specifies the type of cubic packing to create.  Options are
            'square' (default) and 'triangular'.

        Returns
        -------
        A boolean array with True values denoting the pore space
        """
        r = radius
        if sp.size(shape) == 1:
            shape = sp.full((3, ), int(shape))
        elif (sp.size(shape) == 3) or (1 in shape):
            raise Exception("This function only produces 2D images, " +
                             "try \'sphere_pack\'")
        im = sp.zeros(shape, dtype=bool)
        if packing.startswith('s'):
            spacing = 2*r
            s = int(spacing/2) + sp.array(offset)
            coords = sp.mgrid[r:im.shape[0]-r:2*s,
                          r:im.shape[1]-r:2*s]
            im[coords[0], coords[1]] = 1
        if packing.startswith('t'):
            spacing = 2*sp.floor(sp.sqrt(2*(r**2))).astype(int)
            s = int(spacing/2) + offset
            coords = sp.mgrid[r:im.shape[0]-r:2*s,
                              r:im.shape[1]-r:2*s]
            im[coords[0], coords[1]] = 1
            coords = sp.mgrid[s+r:im.shape[0]-r:2*s,
                              s+r:im.shape[1]-r:2*s]
            im[coords[0], coords[1]] = 1
        im = spim.distance_transform_edt(~im) >= r
        return im

    @staticmethod
    def sphere_pack(shape, radius, offset=0, packing='sc'):
        r"""
        Generates a cubic packing of spheres

        Parameters
        ----------
        shape : list
            The size of the image to generate in [Nx, Ny, Nz] where N is the
            number of voxels in each direction.

        radius : scalar
            The radius of spheres in the packing

        offset : scalar
            The amount offset (+ or -) to add between pore centers.

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
        elif (sp.size(shape) == 2) or (1 in shape):
            raise Exception("This function only produces 3D images, " +
                             "try \'circle_pack\'")
        im = sp.zeros(shape, dtype=bool)
        if packing.startswith('s'):
            spacing = 2*r
            s = int(spacing/2) + sp.array(offset)
            coords = sp.mgrid[r:im.shape[0]-r:2*s,
                          r:im.shape[1]-r:2*s,
                          r:im.shape[2]-r:2*s]
            im[coords[0], coords[1], coords[2]] = 1
        elif packing.startswith('b'):
            spacing = 2*sp.floor(sp.sqrt(4/3*(r**2))).astype(int)
            s = int(spacing/2) + offset
            coords = sp.mgrid[r:im.shape[0]-r:2*s,
                              r:im.shape[1]-r:2*s,
                              r:im.shape[2]-r:2*s]
            im[coords[0], coords[1], coords[2]] = 1
            coords = sp.mgrid[s+r:im.shape[0]-r:2*s,
                              s+r:im.shape[1]-r:2*s,
                              s+r:im.shape[2]-r:2*s]
            im[coords[0], coords[1], coords[2]] = 1
        elif packing.startswith('f'):
            spacing = 2*sp.floor(sp.sqrt(2*(r**2))).astype(int)
            s = int(spacing/2) + offset
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
    def overlapping_spheres(shape, radius, porosity):
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

        blobiness : array_like
            Controls the morphology of the image.  A higher number results in
            a larger number of blobs.  If a vector is supplied then the blobs
            are anisotropic.

        porosity : scalar
            The porosity of the final image.  This number is approximated by
            the method so the returned result may not have exactly the
            specified value.

        Returns
        -------
        A boolean array with True values denoting the pore space

        """
        blobiness = sp.array(blobiness)
        shape = sp.array(shape)
        if sp.size(shape) == 1:
            shape = sp.full((3, ), int(shape))
        sigma = sp.mean(shape)/(40*blobiness)
        mask = sp.random.random(shape)
        mask = spim.gaussian_filter(mask, sigma=sigma)
        hist = sp.histogram(mask, bins=1000)
        cdf = sp.cumsum(hist[0])/sp.size(mask)
        xN = sp.where(cdf >= porosity)[0][0]
        im = mask <= hist[1][xN]
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
