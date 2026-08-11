import numpy as np
import porespy as ps
from porespy.tools import get_edt


__all__ = [
    "fibers_2D",
    "fibers_1D",
]


edt = get_edt()


def fibers_1D(shape, r, n, overlap=None):
    r"""
    Generates randomly located fibers which are all oriented in z-direction

    Parameters
    ----------
    shape : list
        The shape of the image to generate.  This must be 3D.
    r : int
        The radius of the fibers to generate
    n : int
        The number of fibers to draw

    Returns
    -------
    im : ndarray
        An `ndarray` of the given shape with fibers indicated by `False` and void
        by `True`.

    Notes
    -----
    ..  [1] Tomadakis MM, Sotirchos SV. Effective diffusivities and conductivities
        of random dispersions of nonoverlapping and partially overlapping
        unidirectional fibers. Journal of Chemical Physics. 99, 9820–9827 (1993).
        `doi <https://doi.org/10.1063/1.465464>`_

    """
    from porespy.generators import random_spheres
    if overlap is None:
        im = np.ones(np.array(shape[:2]) + r, dtype=bool)
        locs = np.random.randint((0, 0), shape[:2], (n, 2))
        im[tuple(locs.T)] = False
        dt = edt(im)
        im = dt >= (r + 0.001)
        im = im[r:-r, r:-r]
    else:
        im = ~random_spheres(
            np.array(shape[:2]) + r,
            r=0,
            clearance=-overlap,
            phi=1,
            maxiter=n,
            edges='extended',
        )
        im = im[r:-r, r:-r]
    if len(shape) == 3:
        im = np.tile(im, [shape[2], 1, 1])
        im = np.rollaxis(im, 0, 3)
    return im


def fibers_2D(shape, r, n=None, porosity=None):
    r"""
    Generates randomly located fibers which are randomly oriented in the x-y plane
    but aligned in the z-plane

    Parameters
    ----------
    shape : list
        The shape of the image to generate.  This must be 3D.
    r : int
        The radius of the fibers to generate
    porosity : float
        The radius of the fibers

    Returns
    -------
    im : ndarray
        An `ndarray` of the given shape with fibers indicated by `False` and void
        by `True`.

    Notes
    -----
    Drawing randomly oriented fibers with a uniform density is a surprizingly
    tricky task. The problem was recently discussed and a solution proposed by
    Beckman et al [1]_.

    ..  [1] Beckman IP, Beckman PM, Cho H, Riveros G. Modeling uniform random
        distributions of nonwoven fibers for computational analysis of
        composite materials. Composite Structures. 301(12), 116242 (2022).
        `doi <https://doi.org/10.1016/j.compstruct.2022.116242>`_
    """

    def add_n_lines(im, r, n):
        x1, y1 = im.shape[0]/2, im.shape[1]/2
        Lmax = (x1**2 + y1**2)**0.5
        for _ in range(n):
            phi = np.random.rand()*2*np.pi
            L = 2*(0.5 - np.random.rand())*Lmax
            x2 = x1 + L*np.cos(phi)
            y2 = y1 + L*np.sin(phi)
            x3 = x2 + 2*Lmax*np.cos(phi + np.pi/2)
            y3 = y2 + 2*Lmax*np.sin(phi + np.pi/2)
            x4 = x2 + 2*Lmax*np.cos(phi - np.pi/2)
            y4 = y2 + 2*Lmax*np.sin(phi - np.pi/2)
            X = ps.generators.line_segment((x3, y3), (x4, y4))
            mask = (X[0] >= 0)*(X[0] < im.shape[0])*(X[1] >= 0)*(X[1] < im.shape[1])
            X = X[0][mask], X[1][mask]
            if im.ndim == 3:
                X = X[0], X[1], np.random.randint(im.shape[2])
            im[tuple(X)] = True
        if r is not None:
            dt = edt(im == 0)
            im = dt >= r
        return im

    if (n is None) and (porosity is not None):
        shape = np.array(shape) + 2*r  # Add 2r so rounded fiber ends fall outside im
        porosity_orig = porosity
        im = np.zeros(shape, dtype=bool)
        iters = np.around(np.log(porosity)/(-4*r**2/im.size**(0.5))).astype(int)
        n = iters*10
        im = add_n_lines(im=im, r=r, n=n)
        porosity = 1 - (im.sum()/im.size - porosity_orig)
        while porosity < .99:
            iters = np.around(np.log(porosity)/(-2*r**2/im.size**(0.5))).astype(int)
            n = iters*10
            if n == 0:
                break
            im2 = np.zeros(shape, dtype=bool)
            im2 = add_n_lines(im=im2, r=r, n=n)
            im *= im2
            porosity = 1 - (im.sum()/im.size - porosity_orig)
        im = im[r:-r, r:-r, r:-r]  # Remove padding
    else:
        im = np.zeros(shape)
        im = add_n_lines(im=im, r=r, n=n)

    return im


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fibs = fibers_2D([200, 200, 100], r=5, porosity=0.85)
    print(f"Porosity: {fibs.sum()/fibs.size}")

    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(ps.visualization.xray(fibs > 0, axis=2))
    ax[1].imshow(ps.visualization.xray(fibs > 0, axis=1).T)
    ax[2].imshow(ps.visualization.xray(fibs > 0, axis=0).T)
