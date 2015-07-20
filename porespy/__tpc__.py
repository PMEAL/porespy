import scipy as sp
from collections import namedtuple
from scipy.spatial import cKDTree as kdt


class TwoPointCorrelation(object):

    def __init__(self, image):
        super().__init__()
        image = sp.atleast_3d(image)
        self.image = sp.array(image, dtype=bool)

    def run(self, npts=25, spacing=2, rmax=20):
        img = self.image
        # Extract size metrics from input image
        [Lx, Ly, Lz] = sp.shape(img)

        # Generate kdtree for void and solid space
        i = sp.meshgrid(sp.arange(0, Lx, spacing),
                        sp.arange(0, Ly, spacing),
                        sp.arange(0, Lz, spacing))
        # Convert matrix into index notation
        i_void = sp.where(img[i] == 1)
        ind_void = sp.vstack((i_void[0].flatten(),
                              i_void[1].flatten(),
                              i_void[2].flatten())).T
        i_solid = sp.where(img[i] == 0)
        ind_solid = sp.vstack((i_solid[0].flatten(),
                               i_solid[1].flatten(),
                               i_solid[2].flatten())).T
        # Generate kdtrees
        dtree_void = kdt(ind_void)
        dtree_solid = kdt(ind_solid)

        # Choose npts random base points from ind_void
        ind = ind_void[sp.random.randint(0, sp.shape(ind_void)[0], npts)]
        # Generate kd-tree of base points
        dtree_pts = kdt(ind)

        # Perform 2-point correlation calculation for range of radii
        hits = []
        for r in sp.arange(1, rmax):
            hits_void = dtree_pts.count_neighbors(other=dtree_void, r=r)
            hits_solid = dtree_pts.count_neighbors(other=dtree_solid, r=r)
            hits.append(hits_void/(hits_solid + hits_void))

        # Store results in namedtuple
        vals = namedtuple('TwoPointCorrelation', ('distance', 'probability'))
        vals.distance = sp.arange(1, rmax, 1)
        vals.probability = hits
        return vals
