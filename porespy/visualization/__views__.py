import scipy as sp


def sem(im, direction='X'):
    im = sp.array(~im, dtype=int)
    if direction in ['Y', 'y']:
        im = sp.transpose(im, axes=[1,0,2])
    if direction in ['Z', 'z']:
        im = sp.transpose(im, axes=[2,1,0])
    t = im.shape[0]
    depth = sp.reshape(sp.arange(0, t), [t, 1, 1])
    im = im*depth
    im = sp.amax(im, axis=0)
    return im


def xray(im, direction='X'):
    im = sp.array(~im, dtype=int)
    if direction in ['Y', 'y']:
        im = sp.transpose(im, axes=[1,0,2])
    if direction in ['Z', 'z']:
        im = sp.transpose(im, axes=[2,1,0])
    im = sp.sum(im, axis=0)
    return im
