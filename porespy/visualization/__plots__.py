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
    This function normalizes the invading phase saturation by the empty pore
    volume at the first invasion step.

    """
    sizes = sp.unique(im)
    Rp = []
    Snwp = []
    for r in sizes[1:]:
        Rp.append(r)
        Snwp.append(sp.sum(im >= r))
    Snwp = [(s-Snwp[-1])/(Snwp[0]-Snwp[-1]) for s in Snwp]
    return Rp, Snwp
