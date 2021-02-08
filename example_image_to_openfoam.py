import numpy as np
from skimage import io
import sys
sys.path.insert(1, '/home/mehrez/repositories/porespy')
import porespy as ps


file = './spheres.tif'
im = io.imread(file, as_gray=False)[0:20, 0:20, 0:50]
im = ~np.array(im, dtype=bool)
ps.io.to_openfoam(im, label=True)
