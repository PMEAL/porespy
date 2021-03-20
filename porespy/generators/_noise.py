from porespy.tools import norm_to_uniform
import multiprocessing
import numpy as np


def fractal_noise(shape, frequency=0.05, octaves=4, gain=0.5, mode='simplex',
                  seed=None, cores=None, uniform=True):
    r"""
    Generate fractal noise which can be thresholded to create binary images
    with realistic structures across scales.

    Parameters
    ----------
    shape : array_like
        The size of the image to generate, can be 2D or 3D.
    frequency : scalar
        Controls the overall scale of the generated noise, with larger
        values giving larger structures.  Default is 0.05.
    octaves : int
        Controls the number of scales across which structures are generated,
        with larger values giving more scale.  Default is 4.
    gain : scalar
        Controls the intensity of successively higher octaves.  Values
        below 1.0 mean the higher octaves are less prominent.  Default is 0.5.
    mode : string
        The type of noise to generate.  Options are:

        - 'perlin' : classic Perlin noise

        - 'simplex' : updated Perlin noise for more realistic textures

        - 'value' : blnear interpolation of white noise

        - 'cubic' : cubic interpolation of white noise

    seed : int
        The seed of the random number generator.  Using the same ``seed``
        between runs will produce the same image.
    cores : int
        The number of cores to use.  This package used by this function has
        implemented SIMD processing which can be spread across cores. The
        default is to use all cores.
    uniform : boolean
        If ``True`` (default) the random values are converted to a uniform
        distribution between 0 and 1, otherwise the result image contains
        the unprocesssed values, which have a 'normal-esque' distribution
        centered on 0.

    Notes
    -----
    This function provides a simplified wrapper for the the functins in
    `pyfastnoisesimd <https://github.com/robbmcleod/pyfastnoisesimd>`_
    package. ``pyfastnoisesimd`` is itself a wrapper for a C-library called
    `FastNoiseSIMD <https://github.com/Auburn/FastNoiseSIMD>`_. To access the
    more elaborate functionality and options of these packages, explore the
    `pyfastnoisesimd documentation
    <https://pyfastnoisesimd.readthedocs.io/en/latest/overview.html>`_.

    For more information on ``simplex noise`` see
    `here <https://en.wikipedia.org/wiki/Simplex_noise>`_.

    For more information on ``perlin noise`` see
    `here <https://en.wikipedia.org/wiki/Perlin_noise>`_.

    For more information on ``value noise`` see
    `here <https://en.wikipedia.org/wiki/Value_noise>`_.

    For more information on ``cubic noise`` see
    `here <https://github.com/jobtalle/CubicNoise>`_.

    """
    try:
        import pyfastnoisesimd as fns
    except ModuleNotFoundError:
        raise Exception('fractal_noise requires pyfastnoisesimd, which can be pip' +
                        'installed if a c-compiler is available on the target ' +
                        'machine')
    if cores is None:
        cores = multiprocessing.cpu_count()
    if seed is None:
        seed = np.random.randint(2**31)
    perlin = fns.Noise(numWorkers=cores)
    if mode == 'simplex':
        perlin.noiseType = fns.NoiseType.SimplexFractal
    elif mode == 'perlin':
        perlin.noiseType = fns.NoiseType.PerlinFractal
    elif mode == 'cubic':
        perlin.noiseType = fns.NoiseType.CubicFractal
    elif mode == 'value':
        perlin.noiseType = fns.NoiseType.ValueFractal
    perlin.frequency = frequency
    perlin.fractal.octaves = octaves
    perlin.fractal.gain = gain
    perlin.fractal.lacunarity = 2
    perlin.perturb.perturbType = fns.PerturbType.NoPerturb
    perlin.seed = seed
    result = perlin.genAsGrid(shape)
    if uniform:
        result = norm_to_uniform(result, scale=[0, 1])
    return result