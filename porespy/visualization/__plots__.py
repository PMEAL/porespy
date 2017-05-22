import matplotlib.pyplot as plt
import scipy as sp


def drainage_curve(im):
    r"""
    Calculate drainage curve based on the image produced by the ``porosimetry``
    function.

    Parameters
    ----------
    im : ND-array
        The image returned by ``porespy.simulations.porosimetry``

    Returns
    -------
    Rp, Snwp
    Two arrays

    Notes
    -----
    This function normalizes the invading phase saturation by total pore volume
    of the dry image.

    """
    sizes = sp.unique(im)
    Rp = []
    Snwp = []
    Vp = sp.sum(im > 0)
    for r in sizes[1:]:
        Rp.append(r)
        Snwp.append(sp.sum(im >= r))
    Snwp = [s/Vp for s in Snwp]
    return Rp, Snwp
