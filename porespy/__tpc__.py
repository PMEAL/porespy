import scipy as sp

class TwoPointCorrelation(object):
    @staticmethod
    def run(img,npoints=1000,nbins=100):
        r'''
        '''
        from scipy.spatial.distance import cdist
        img = sp.atleast_3d(img)
        # Extract size metrics from input image
        [Lx, Ly, Lz] = sp.shape(img)
        ind = sp.vstack((sp.random.randint(0,Lx,npoints),
                         sp.random.randint(0,Ly,npoints),
                         sp.random.randint(0,Lz,npoints))).T
        phase = img[ind[:,0],ind[:,1],ind[:,2]].flatten()
        ind = ind[phase.astype(bool)]
        # Get distance map of points
        dmap = cdist(ind,ind,'euclidean')
        bin_max = sp.ceil(sp.amax(dmap))
        bin_min = sp.floor(sp.amin(dmap))
        bin_array = sp.linspace(bin_min,bin_max,nbins)
        temp = sp.digitize(dmap.flatten(),bin_array)
        count = sp.bincount(temp)
        distance = sp.arange(bin_min,bin_max,(bin_max-bin_min)/nbins)
        return [distance,count]
        
        
        
class AutoCorrelation():
    @staticmethod
    def run(img):
        from scipy.ndimage.filters import correlate as corr
        import scipy.ndimage as spim
        import scipy as sp
        import numpy as np
        
        space = sp.rand(200,200)>0.99
        space = spim.distance_transform_bf(~space)>=5
        space = space.astype(int)
        
#        result = corr(space, space)
#        result = result[result.size/2:]
        
        space_ps = sp.absolute(np.fft.fftn(space))
        space_ps *= space_ps

        space_ac = np.fft.ifftn(space_ps).real.round()
        space_ac /= space_ac[0, 0]
        
        
        





