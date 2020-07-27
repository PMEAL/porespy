import sys
sys.path.insert(1, '/home/mehrez/repositories/porespy')
import scipy as sp
import porespy as ps


# porosity values investigated
p = 0.5

# characteristics of the medium
shape = [20, 20, 20]
dist = sp.stats.norm(loc=7, scale=5)
r_min = 5

im = ps.generators.polydisperse_spheres(shape=shape,
                                        porosity=p,
                                        dist=dist,
                                        r_min=r_min)
ps.io.im_to_openfoam(im)
