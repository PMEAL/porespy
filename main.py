
import scipy as sp
import scipy.ndimage as spim
import scipy.spatial as sptl
import tifffile

class TwoPointCorrelation(object):
    def __init__(self,img,**kwargs):
        pass
    
    def radial(self,img,spacing):
        r'''
        '''
        img = sp.atleast_3d(img)
        # Extract size metrics from input image
        [Lx, Ly, Lz] = sp.shape(img)
        X = sp.arange(0,Lx,spacing)
        Y = sp.arange(0,Ly,spacing)
        Z = sp.arange(0,Lz,spacing)
        pts = sp.meshgrid(X,Y,Z)
        ind = sp.vstack((pts[0].flatten(),pts[1].flatten(),pts[2].flatten())).T  # Add 0.5 here?
        
        radii = sp.linspace(spacing,(Lx**2 + Ly**2 + Lz**2)**0.5,spacing)
        void = sp.reshape(img[pts],(sp.size(img[pts]),))
        kdt_all = sptl.cKDTree(ind)       
        kdt_void = sptl.cKDTree(ind[void])
        hits_all = kdt_void.count_neighbors(kdt_all,r=radii)
        hits_void = kdt_void.count_neighbors(kdt_void,r=radii)
        
    
    def _dist(self,pts):
        from scipy.spatial.distance import cdist
        dmap = cdist(pts,pts,'euclidean')
        return dmap
        
    def _hist(self,dist_map,bins=100):
        bin_max = sp.ceil(sp.amax(dist_map))
        bin_min = sp.floor(sp.amin(dist_map))
        bin_array = sp.linspace(bin_min,bin_max,bins)
        temp = sp.digitize(dist_map.flatten(),bin_array)
        bin_count = sp.bincount(temp)
        return bin_count

class AutoCorrelation(object):
    pass

class ChordLengthDistribution(object):
       
    def xdir(self,img,spacing=10,trim_edges=True):
        r'''
        '''
        temp = self._apply_chords(img=img,spacing=spacing,trim_edges=trim_edges)
        return temp

    def ydir(self,img,spacing=10,trim_edges=True):
        r'''
        '''
        temp = sp.transpose(img,[1,0,2])
        temp = self._apply_chords(img=temp,spacing=spacing,trim_edges=trim_edges)
        temp = sp.transpose(img,[1,0,2])
        return temp
        
    def zdir(self,img,spacing=10,trim_edges=True):
        r'''
        '''
        temp = sp.transpose(img,[2,1,0])
        temp = self._apply_chords(img=temp,spacing=spacing,trim_edges=trim_edges)
        temp = sp.transpose(img,[2,1,0])
        return temp
        
    def _apply_chords(self,img,spacing=10,trim_edges=True):
        r'''
        '''
        # Clean up input image
        img = sp.atleast_3d(img)
        # Extract size metrics from input image
        [Lx, Ly, Lz] = sp.shape(img)
        start = sp.array(sp.floor(spacing/2),dtype=int)
        Y = sp.arange(start,Ly,spacing)
        Z = sp.arange(start,Lz,spacing)
        temp = sp.zeros([Lx,Ly,Lz],dtype=int)
        # Generate 2D mask of chords in X-dir
        maskX = sp.zeros([Lx,Ly,1],dtype=bool)
        maskX[:,Y,:] = 1
        # Apply chord mask to specified layers (Z-dir) of input image
        temp[:,:,Z] = img[:,:,Z]*maskX
        if trim_edges:
            temp[[0,-1],:,:] = 1
            temp[:,[0,-1],:] = 1
            spim.label(temp)
            ind = sp.where(temp==temp[0,0,0])
            temp[ind] = 0
            
        return sp.array(temp,dtype=int)
        
    def _distribution(self,img):
        r'''
        '''
        # Find spacing
        proj = sp.sum(img,axis=1)
        [yi,zi] = sp.where(proj)
        bins = sp.zeros((sp.size(img)),dtype=int)
        for y,z in zip(yi,zi):
            a = self._find_blocks(img[:,y,z])
            bins[a['length']] += 1
        big_bin = sp.where(bins)[0][-1] + 1
        return bins[:big_bin]
        
    def _find_blocks(self,array,ignore_edges=False):
        temp = sp.pad(array,pad_width=1,mode='constant',constant_values=0)
        end_pts = sp.where(sp.ediff1d(temp)==-1)[0] # Find 1->0 transitions
        end_pts -= 1  # To adjust for 0 padding
        seg_len = sp.cumsum(array)[end_pts]
        seg_len[1:] = seg_len[1:] - seg_len[:-1]
        start_pts = end_pts - seg_len + 1
        a = dict()
        a['start'] = start_pts
        a['end'] = end_pts
        a['length'] = seg_len
        if ignore_edges:
            if a['start'][0] == 0:
                [a.update({item:a[item][1:]}) for item in a]
            if a['end'][-1] == sp.size(array):
                [a.update({item:a[item][:-1]}) for item in a]
        return a
        
if __name__ == '__main__':
#    path = 'C:\\Users\\Jeff\\Dropbox\\Public\\'
#    file = 'Xray-trinary(800-1000-1200)'
#    ext = 'tif'
#    img = tifffile.imread(path+file+'.'+ext)
#    img = img[:,:,1200:1800,1200:1800]
#    img = img.swapaxes(2,0)
#    img = img.swapaxes(3,1)
#    img = img[:,:,:,0] < sp.amax(img[:,:,:,0])
#    sp.savez('img_med',img)
#    temp = sp.load('img.npz')
    temp = sp.load('img_med.npz')
    img = temp['arr_0']
    C = ChordLengthDistribution()
    chords = C._apply_chords(img,trim_edges=False)
#    dist = C._distribution(chords)
#    plt.plot(sp.log10(dist))
    
    
    
    
    
    
    
    
    
    
    
    
    