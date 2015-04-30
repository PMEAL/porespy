import scipy as sp
__all__ =['REV']

class REV(object):
    r'''
    '''
    
    def run(self,im,N=100):
        r'''
        '''
        pad = [0.25,0.25,0.45]
        Cx = sp.random.randint(pad[0]*sp.shape(im)[0],(1-pad[0])*sp.shape(im)[0],N)
        Cy = sp.random.randint(pad[1]*sp.shape(im)[1],(1-pad[1])*sp.shape(im)[1],N)
        Cz = sp.random.randint(pad[2]*sp.shape(im)[2],(1-pad[2])*sp.shape(im)[2],N)
        C = sp.vstack((Cx,Cy,Cz)).T
        
        #Find maximum radius allowable for each point
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
            R = Rmin[i]
            print(i/N)
            for r in sp.arange(Rmin[i],1,-10):
                imtemp = im[C[i,0]-150:C[i,0]+150,C[i,1]-150:C[i,1]+150:,C[i,2]-r:C[i,2]+r]
                vol.append(sp.size(imtemp))
                size.append(2*r)
                porosity.append(sp.sum(imtemp==0)/(sp.size(imtemp)))
        
        return [porosity,size]