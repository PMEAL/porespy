
import porespy as ps
from porespy.filters.__funcs__ import try_sigma_R
im = ps.generators.blobs(shape=[400, 400], porosity=0.6, blobiness=2)
r_max_arr=[2,4,6,8,12,15,20]
sigma_arr=[0.25,0.35,0.5,0.65]
results=try_sigma_R(im, r_max_arr, sigma_arr,plot_all=True, mask=True)