import time
from porespy import simulations, tools, settings
from porespy.tools import Results
import numpy as np
import openpnm as op
import dask.delayed
import dask
import edt
import pandas as pd
from dask.distributed import Client, LocalCluster
import logging
import porespy as ps


__all__ = [
    'network_to_tau',
    'tortuosity_by_blocks',
    'blocks_to_dataframe',
    'rev_tortuosity',
]


tqdm = tools.get_tqdm()
settings.loglevel = 50


def get_block_sizes(shape, block_size_range=[10, 100]):
    r"""
    Finds all viable block sizes between lower and upper limits

    Parameters
    ----------
    shape : sequence of int
        The [x, y, z] size of the image
    block_size_range : sequence of 2 ints
        The [lower, upper] range of the desired block sizes. Default is [10, 100]

    Returns
    -------
    sizes : ndarray
        All the viable block sizes in the specified range
    """
    Lmin, Lmax = block_size_range
    a = np.ceil(min(shape)/Lmax).astype(int)
    block_sizes = min(shape) // np.arange(a, 9999)  # Generate WAY more than needed
    block_sizes = np.unique(block_sizes[block_sizes >= Lmin])
    return block_sizes


def get_divs(shape, block_size_range=[10, 100]):
    r"""
    Finds all the viable ways the image can be divided between the lower and upper
    limits

    Parameters
    ----------
    shape : sequence of int
        The [x, y, z] size of the image
    block_size_range : sequence of 2 ints
        The [lower, upper] range of the desired block sizes. Default is [10, 100]

    Returns
    -------
    sizes : ndarray
        All the viable block sizes in the specified range
    """
    Lmin, Lmax = block_size_range
    num_blocks = np.unique((min(shape)/np.arange(Lmin, Lmax)).astype(int))
    return num_blocks


def rev_tortuosity(im, block_size_range=[10, 100]):
    r"""
    Computes data for a representative element volume plot based on tortuosity

    Parameters
    ----------
    im : ndarray
        A boolean image of the porous media with `True` values indicating the phase
        of interest.
    block_size_range : sequence of ints
        The upper and lower bounds of the block sizes to use. The defaults are
        10 to 100 voxels.  Placing an upper limit on the size of the blocks can
        avoid time consuming computations, while placing a lower limit can avoid
        computing values for meaninglessly small blocks.

    Returns
    -------
    df : DataFrame
        A `pandas` DataFrame with the tortuosity and volume for each block, along
        with other useful data like the porosity.

    """
    mn, mx = block_size_range
    a = np.ceil(min(im.shape)/mx).astype(int)
    block_size = min(im.shape) // np.arange(a, 9999)  # Generate WAY more than needed
    block_size = block_size[block_size >= mn]  # Trim to given min block size
    tau = []
    for s in tqdm(block_size):
        tau.append(blocks_to_dataframe(im, block_size=s))
    df = pd.concat(tau)
    del df['Throat Number']
    df = df[df.Tortuosity < np.inf]  # inf values mean block did not percolate
    return df


@dask.delayed
def calc_g(im, axis):
    r"""
    Calculates diffusive conductance of an image

    Parameters
    ----------
    image : np.ndarray
        The binary image to analyze with ``True`` indicating phase of interest.
    axis : int
        0 for x-axis, 1 for y-axis, 2 for z-axis.
    """
    try:
        solver = op.solvers.PyamgRugeStubenSolver(tol=1e-6)
        results = simulations.tortuosity_fd(im=im, axis=axis, solver=solver)
    except Exception:
        results = Results()
        results.effective_porosity = 0.0
        results.original_porosity = im.sum()/im.size
        results.tortuosity = np.inf
    L = im.shape[axis]
    A = np.prod(im.shape)/im.shape[axis]
    g = (results.effective_porosity * A) / (results.tortuosity * L)
    results.diffusive_conductance = g
    return (g, results)


def estimate_block_size(im, scale_factor=3, mode='radial'):
    r"""
    Predicts suitable block size based on properties of the image

    Parameters
    ----------
    im : ndarray
        The boolean image of the materials with `True` indicating the void space
    scale_factor : int
        The factor by which to increase the estimating block size to ensure blocks
        are big enough
    mode : str
        Which method to use when estimating the block size. Options are:

        -------- --------------------------------------------------------------------
        mode     description
        -------- --------------------------------------------------------------------
        radial   Uses the maximum of the distance transform
        linear   Uses chords drawn in all three directions, then uses the mean length
        -------- --------------------------------------------------------------------

    Returns
    -------
    block_size : list of ints
        The block size in a 3 directions. If `mode='radial'` these will all be equal.
        If `mode='linear'` they could differ in each direction.  The largest block
        size will be used if passed into another function.

    """
    if mode.startswith('rad'):
        dt = edt.edt(im)
        x = min(dt.max()*scale_factor, min(im.shape)/2)  # Div by 2 is fewest blocks
        size = np.array([x, x, x], dtype=int)
    elif mode.startswith('lin'):
        size = []
        for ax in range(im.ndim):
            crds = ps.filters.apply_chords(im, axis=ax)
            crd_sizes = ps.filters.region_size(crds > 0)
            L = np.mean(crd_sizes[crds > 0])/2  # Divide by 2 to make it a radius
            L = min(L*scale_factor, im.shape[ax]/2)  # Div by 2 is fewest blocks
            size.append(L)
        size = np.array(size, dtype=int)
    else:
        raise Exception('Unsupported mode')
    return size


def block_size_to_divs(shape, block_size):
    shape = np.array(shape)
    divs = (shape/block_size).astype(int)
    return divs


def divs_to_block_size(shape, divs):
    block_size = (np.array(shape)/divs).astype(int)
    return block_size


def blocks_to_dataframe(im, block_size):
    r"""
    Computes the diffusive conductance for each block and returns values in a
    dataframe

    Parameters
    ----------
    im : ndarray
        The boolean image of the materials with `True` indicating the void space
    block_size : int or list of ints
        The size of the blocks to use. If a list is given, the largest value is used
        for all directions.

    Returns
    -------
    df : dataframe
        A `pandas` data frame with the properties for each block on a given row.

    """
    if dask.distributed.client._get_global_client() is None:
        client = Client(LocalCluster(silence_logs=logging.CRITICAL))
    else:
        client = dask.distributed.client._get_global_client()
    if not isinstance(block_size, int):  # divs must be equal, so use biggest block
        block_size = max(block_size)
    df = pd.DataFrame()
    offset = int(block_size/2)
    for axis in tqdm(range(im.ndim)):
        im_temp = np.swapaxes(im, 0, axis)
        im_temp = im_temp[offset:-offset, ...]
        divs = block_size_to_divs(shape=im_temp.shape, block_size=block_size)
        slices = ps.tools.subdivide(im_temp, divs)
        queue = []
        for s in slices:
            queue.append(calc_g(im_temp[s], axis=0))
        results = dask.compute(queue, scheduler=client)[0]
        df_temp = pd.DataFrame()
        df_temp['eps_orig'] = [r[1].original_porosity for r in results]
        df_temp['eps_perc'] = [r[1].effective_porosity for r in results]
        df_temp['g'] = [r[1].diffusive_conductance for r in results]
        df_temp['tau'] = [r[1].tortuosity for r in results]
        df_temp['axis'] = [axis for _ in results]
        df = pd.concat((df, df_temp))
    return df


def network_to_tau(df, im, block_size):
    r"""
    Compute the tortuosity of a network populated with diffusive conductance values
    from the given dataframe.

    Parameters
    ----------
    df : dataframe
        The dataframe returned by the `blocks_to_dataframe` function
    im : ndarray
        The boolean image of the materials with `True` indicating the void space
    block_size : int or list of ints
        The size of the blocks to use. If a list is given, the largest value is used
        for all directions.

    Returns
    -------
    tau : list of floats
        The tortuosity in all three principal directions
    """
    if not isinstance(block_size, int):  # divs must be equal, so use biggest block
        block_size = max(block_size)
    divs = block_size_to_divs(shape=im.shape, block_size=block_size)
    net = op.network.Cubic(shape=divs)
    phase = op.phase.Phase(network=net)
    gx = np.array(df['g'])[df['axis'] == 0]
    gy = np.array(df['g'])[df['axis'] == 1]
    gz = np.array(df['g'])[df['axis'] == 2]
    g = np.hstack((gz, gy, gx))  # throat indices in openpnm are in reverse order!
    if np.any(g == 0):
        g += 1e-20
    phase['throat.diffusive_conductance'] = g
    bcs = {0: {'in': 'left', 'out': 'right'},
           1: {'in': 'front', 'out': 'back'},
           2: {'in': 'top', 'out': 'bottom'}}
    im_temp = ps.filters.fill_blind_pores(im, surface=True)
    e = np.sum(im_temp, dtype=np.int64) / im_temp.size
    D_AB = 1
    tau = []
    for ax in range(im.ndim):
        fick = op.algorithms.FickianDiffusion(network=net, phase=phase)
        fick.set_value_BC(pores=net.pores(bcs[ax]['in']), values=1.0)
        fick.set_value_BC(pores=net.pores(bcs[ax]['out']), values=0.0)
        fick.run()
        rate_inlet = fick.rate(pores=net.pores(bcs[ax]['in']))[0]
        L = im.shape[ax] - block_size
        A = np.prod(im.shape) / im.shape[ax]
        D_eff = rate_inlet * L / (A * (1 - 0))
        tau.append(e * D_AB / D_eff)
    ws = op.Workspace()
    ws.clear()
    return tau


def tortuosity_by_blocks(im, block_size=None):
    r"""
    Computes the tortuosity tensor of an image using the geometric domain
    decomposition method

    Parameters
    ----------
    im : ndarray
        The boolean image of the materials with `True` indicating the void space
    block_size : int or list of ints
        The size of the blocks to use. If a list is given, the largest value is used
        for all directions.

    Returns
    -------
    tau : list of floats
        The tortuosity in all three principal directions
    """
    if block_size is None:
        block_size = estimate_block_size(im, scale_factor=3, mode='radial')
    df = blocks_to_dataframe(im, block_size)
    tau = network_to_tau(df=df, im=im, block_size=block_size)
    return tau


if __name__ =="__main__":
    import porespy as ps
    import numpy as np
    import matplotlib.pyplot as plt
    np.random.seed(1)
    im = ps.generators.cylinders(shape=[200, 200, 200], porosity=0.7, r=3)
    block_size = estimate_block_size(im, scale_factor=3, mode='linear')
    tau = tortuosity_by_blocks(im=im, block_size=block_size)
    print(tau)
