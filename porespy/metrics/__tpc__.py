import sys
import scipy as sp
from collections import namedtuple
from scipy.spatial import cKDTree as kdt
import scipy.ndimage as spim


class two_point_correlation(object):
    r"""
    Computes the 2-point correlation function of an image

    Examples
    --------
    Generate a test image of a sphere pack:

    >>> import scipy as sp
    >>> import scipy.image as spim
    >>> im = sp.rand(100, 100, 100) < 0.997
    >>> im = spim.distance_transform_edf(im) >= 4

    Import porespy and use it:

    >>> import porespy
    >>> a = porespy.tpc(im)
    >>> vals = a.run(npts=250, spacing=2, rmax=20)

    Visualize with Matplotlib

    .. code-block:: python

        import matplotlib as plt
        plt.plot(vals.distance, vals.probability, 'b-o')

    """

    def __init__(self, image):
        super().__init__()
        image = sp.atleast_3d(image)
        self.image = sp.array(image, dtype=bool)

    def run(self, npts=100, r=20, dtransform=None):
        r"""
        Performs the 2-point correlation calculation.

        This method works by selecting a set of **query** points in the void
        space then finding all neighboring points within a specified distance
        of each **query** point that lie in the void space or the solid phase.
        The fraction of points that lie in the void space as a function of
        distance from the query point is returned.

        Parameters
        ----------
        npts : int
            The number of points against which the neighboring points should
            be queried.  The **query** points are randomly selected, so
            repeated calls to run will not necessarily generate identical
            results.  If the results differ too much then ``npts`` should be
            incresed.

        r : scalar or vector
            Controls the radial distance from the query points that are
            considered.  If a scalar is received then a list of sizes between
            1 and ``r`` is generated with a spacing of 1 voxel, otherwise the
            given ``r`` values are used.  It is useful to provide ``r`` values
            to limit the number of points and speed-up the calculation.

        TODO: The methods in here could clearly benefit from proper use of
        itertools, nditer, and other numpy functions.  I can't quite figure
        how to convert meshgrid to vector form.

        """
        if sp.size(r) == 1:
            rmax = r
            sizes = sp.arange(1, rmax)
        else:
            sizes = r
            rmax = r[-1]
        # Extract size metrics from input image
        [Lx, Ly, Lz] = sp.shape(self.image)
        ind = sp.where(self.image == 1)
        temp = sp.random.randint(0, sp.shape(ind)[1], npts)
        i_query = (ind[0][temp], ind[1][temp], ind[2][temp])
        i_void = sp.where(self.image == 1)
        i_solid = sp.where(self.image == 0)

        # Reduce points to only those within rmax of query points
        if dtransform is None:
            imtemp = sp.ones((Lx, Ly, Lz), dtype=bool)
            imtemp[i_query] = False
            dtransform = spim.distance_transform_edt(imtemp)
        mask = dtransform <= rmax
        i_void = sp.where((self.image*mask) == 1)
        i_solid = sp.where(((~self.image)*mask) == 1)

        # Convert matrix into index notation for void and solid phases
        ind_void = sp.vstack((i_void[0].flatten(),
                              i_void[1].flatten(),
                              i_void[2].flatten())).T
        ind_solid = sp.vstack((i_solid[0].flatten(),
                               i_solid[1].flatten(),
                               i_solid[2].flatten())).T
        ind_query = sp.vstack((i_query[0].flatten(),
                               i_query[1].flatten(),
                               i_query[2].flatten())).T

        # Generate kdtrees for void, solid and query points
        dtree_void = kdt(ind_void)
        dtree_solid = kdt(ind_solid)
        dtree_pts = kdt(ind_query)

        # Perform 2-point correlation calculation for range of radii
        print('Checking correlations vs increasing radii')
        print('0%|'+'-'*len(sizes)+'|100%')
        print('  |', end='')
        hits = []
        for r in sizes:
            print('.', end='')
            sys.stdout.flush()
            hits_void = dtree_pts.count_neighbors(other=dtree_void, r=r)
            hits_solid = dtree_pts.count_neighbors(other=dtree_solid, r=r)
            hits.append(hits_void/(hits_solid + hits_void))
        print('|')

        # Store results in namedtuple
        vals = namedtuple('TwoPointCorrelation', ('distance', 'probability'))
        vals.distance = sizes
        vals.probability = hits
        return vals
