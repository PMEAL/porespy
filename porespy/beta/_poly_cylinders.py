import numpy as np
from porespy import settings
from porespy.tools import get_tqdm


tqdm = get_tqdm()


__all__ = [
    'polydisperse_cylinders',
]


def polydisperse_cylinders(
    shape,
    porosity,
    dist,
    voxel_size=1,
    phi_max=0,
    theta_max=90,
    maxiter=3,
    rtol=1e-2,
    seed=None,
):
    r"""
    Generates overlapping cylinders with radii from the given size distribution

    This function works by combining individual images from the `cylinders` function
    for each radius, so **it can be slow**.  For instance, if the distribution spans
    10 different radii, then this function will take approximately 10x longer
    than generating unimodal cylinders.

    Parameters
    ----------
    shape : list
        The shape of the image to generate
    porosity : float
        The target porosity of the final image. This value is achieved by
        iteratively inserting cylinders so the final value will not be exact.
    dist : scipy.stats frozen distribution
        The fiber radius distribution in arbitary units depending on the value of
        `voxel_size`. The way this distribution is used internally is explained in
        more detail in the Notes section.
    voxel_size : scalar
        The length of one side of a voxel. The units should be the same as used to
        obtain `dist`, so if `dist` is in terms of `nm`, then `voxel_size` should
        have units of `nm/voxel`.
    phi_max : scalar
        The maximum amount of 'out of plane' rotation applied to the fibers in units
        of degrees.  A value of 10 will result in the fibers being randomly oriented
        +/- 10 degress out of the XY plane.  The default is 0.
    theta_max : scalar
        The maximum amount of 'in plane' rotation applied to the fibers in units of
        degrees.  A value of 0 means the fibers will all be aligned in the
        x-direction.  A value of 90 degrees means they will be oriented +/- 90
        degrees, providing completely random orientation.
    maxiter : int
        The number of iterations to use when trying to achieve the requested
        porosity. The default is 3.  If the algorithm tends to undershoot porosity
        (i.e. gives 0.4 when 0.5 was requested) try lowering this value to 2 or 1.
        And conversely if the algorithm gives 0.7 when 0.6 was requested by 4 or 5.
    rtol : float
        Controls how close the porosity gets to the target value before stopping.
        The default is 1e-2, or 2%, so if the requested porosity is 0.5, the default
        will stop iterations if the image achieves 0.51 or lower.
    seed : int
        The seed to use in the random number generator.  The default is `None` which
        will produced different results each time.

    Returns
    -------
    cylinders : ndarray
        An ndarray of the requested shape with ``True`` values indicating the void
        space.

    Notes
    -----
    The `scipy.stats` object must be initialized with desired parameter to create
    a *frozen* object, like `dist = spst.gamma(5, 1, 7)`. Then this parameters
    are fixed for all future calls to the object's methods (i.e. `ppf`, `pdf`, etc.)
    The classes in the `stats` module have a very useful `fit` method for
    finding the fitting parameters for a given data set.  For example,
    `params = scipy.stats.gamma.fit(data)` can then be used to initialize the
    frozen distribution as `dist = scipy.stats.gamma(*params)`.

    The `stats` object is used to compute the lower and upper limits on the
    cylinder radii in units of voxels using the `ppf` method as follows:

        `rstart = int(stats.ppf(0.01)/voxel_size)`

        and

        `rstop = int(stats.ppf(0.99)/voxel_size)`

    An array of radius values is then found as `radii = np.arange(rstart, rstop, 1)`
    so all radii are used.

    The `stats` object is then used again to determine the relative fraction of
    each cylinder size to add using the `pdf` function:

        `phi = stats.pdf(r*voxel_size)*bin_width`

    where `r` is in units of voxels and `bin_width` is found as:

        `bin_width = (radii[i+1] -  radii[i])*voxel_size`

    where `radii` is a list of cylinder radii between `rstart` and `rstop`. However,
    since `radii` contains sequential integers `bin_width` is basically equal to
    the `voxel_size`.

    """
    from porespy.generators import cylinders
    if seed is not None:
        np.random.seed(seed)
    fibers = np.ones(shape, dtype=bool)
    e = porosity
    radii = np.arange(
        start=max(1, np.floor(dist.ppf(0.01)/voxel_size)),
        stop=np.ceil(dist.ppf(0.99)/voxel_size),
        step=1,
    ).astype(int)
    iters = 0
    enable_status = settings.tqdm['disable']
    f = 0.5**porosity  # Controls how much of the predicted phi is actually inserted
    while iters < maxiter:
        for i, r in enumerate(tqdm(radii[:-1])):
            settings.tqdm['disable'] = True  # Disable for call to cylinders
            phi = 1 - f*(1-e)*dist.pdf(r*voxel_size)*(radii[i+1] - radii[i])*voxel_size
            tmp = ~cylinders(
                shape=fibers.shape,
                porosity=phi,
                r=r,
                phi_max=phi_max,
                theta_max=theta_max)
            fibers[tmp] = False
            settings.tqdm['disable'] = enable_status  # Set back to user preference
        eps = fibers.sum(dtype=np.int64)/fibers.size
        if (eps < porosity*(1+rtol)):  # If within rtol of target porosity, break
            break
        else:
            e = 1 - (eps - porosity)
            # f = f**0.5
            iters += 1
    return fibers


if __name__ == "__main__":
    import scipy.stats as spst
    import matplotlib.pyplot as plt
    import porespy as ps

    params = (5.65832732e+00, 1.54364793e-05, 7.37705832e+00)
    dist = spst.gamma(*params)
    fibers = polydisperse_cylinders(
        shape=[500, 500, 250],
        porosity=0.75,
        dist=dist,
        voxel_size=5,
        phi_max=5,
        theta_max=90,
        maxiter=3,
        rtol=1e-2,
        seed=0,
    )
    print(fibers.sum()/fibers.size)
    plt.imshow(ps.visualization.sem(fibers, axis=2))
