import scipy as _sp

class ImageIO():
    r'''
    '''
    @staticmethod        
    def imopen(filename):
        r'''
        '''
        import tifffile as tff
        tiffimg = tff.TIFFfile(filename)
        im = tiffimg.asarray()
        im = _sp.swapaxes(im, 0, 2)
        return im
        
    