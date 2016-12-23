import matplotlib.pyplot as plt
import scipy as sp


def drainage_curve(im, fig=None, **kwargs):
    r"""
    Output a drainage curve based on the image output from the ``porosimetry``
    function.

    Parameters
    ----------
    im : ND-array

    """
    if fig is None:
        fig = plt.figure()
    sizes = sp.unique(im)
    Rp = []
    Snwp = []
    for r in sizes[1:]:
        Rp.append(r)
        Snwp.append(sp.sum(im >= r))
    Vtot = sp.sum(im > 0)
    Snwp = [s/Vtot for s in Snwp]
    plt.plot(Rp, Snwp, 'b-o')
    return fig
