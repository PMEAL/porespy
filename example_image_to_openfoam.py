import numpy as np
import porespy as ps
from edt import edt


im = np.random.rand(25, 25, 25) < 0.999
im = edt(im) > 10
im = ~np.array(im, dtype=bool)
ps.io.to_openfoam(im, label=True)
