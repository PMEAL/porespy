import numpy as np
import porespy as ps
import matplotlib.pyplot as plt


im = ps.generators.overlapping_spheres(shape=[250, 250, 250],
                                       radius=6,
                                       porosity=0.5)


