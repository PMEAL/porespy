import porespy as ps
import numpy as np


radii = np.array([10, 20, 25, 5])
centers = np.array([[0, 10, 3],
                    [20, 20, 13],
                    [40, 25, 55],
                    [60, 0, 89]])
ps.io.sphere_pack_to_comsol('sphere_pack', centers, radii)
