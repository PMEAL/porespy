import scipy as sp
from collections import namedtuple


class representative_elementary_volume(object):
    r'''
    Perform a Representative Elementary Volume calculation on porosity

    Examples
    --------
    Generate a test image of a sphere pack:

    >>> import scipy as sp
    >>> import scipy.image as spim
    >>> im = sp.rand(40, 40, 40) < 0.997
    >>> im = spim.distance_transform_bf(im) >= 4

    Import porespy and use it:

    >>> import porespy
    >>> a = porespy.rev(image=im)
    >>> results = a.run(n=200)

    Visualize results using Matplotlib:

    .. code-block:: python

        import matplotlib as plt
        plt.plot(vals.size, vals.porosity, 'bo')

    '''
    def __init__(self, image):
        super().__init__()
        image = sp.atleast_3d(image)
        self.image = sp.array(image, dtype=bool)

    def run(self, N=100):
        r'''
        '''
        im = self.image
        # Create a list of N random points to use as box centers
        pad = [0.1,0.1,0.45]  # Ensure points are near middle
        Cx = sp.random.randint(pad[0]*sp.shape(im)[0],(1-pad[0])*sp.shape(im)[0],N)
        Cy = sp.random.randint(pad[1]*sp.shape(im)[1],(1-pad[1])*sp.shape(im)[1],N)
        Cz = sp.random.randint(pad[2]*sp.shape(im)[2],(1-pad[2])*sp.shape(im)[2],N)
        C = sp.vstack((Cx,Cy,Cz)).T

        # Find maximum radius allowable for each point
        Rmax = sp.array(C>sp.array(sp.shape(im))/2)
        Rlim = sp.zeros(sp.shape(Rmax))
        Rlim[Rmax[:,0],0] = sp.shape(im)[0]
        Rlim[Rmax[:,1],1] = sp.shape(im)[1]
        Rlim[Rmax[:,2],2] = sp.shape(im)[2]
        R = sp.absolute(C-Rlim)
        R = R.astype(sp.int_)
        Rmin = sp.amin(R,axis=1)

        vol = []
        size = []
        porosity = []
        for i in range(0,N):
            for r in sp.arange(Rmin[i],1,-10):
                imtemp = im[C[i,0]-150:C[i,0]+150,C[i,1]-150:C[i,1]+150:,C[i,2]-r:C[i,2]+r]
                vol.append(sp.size(imtemp))
                size.append(2*r)
                porosity.append(sp.sum(imtemp==1)/(sp.size(imtemp)))

        vals = namedtuple('REV', ('porosity', 'size'))
        vals.porosity = porosity
        vals.size = size
        return vals