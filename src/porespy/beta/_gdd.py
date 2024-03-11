import time
from porespy import simulations, settings
from porespy.tools import Results
import numpy as np
import openpnm as op
from pandas import DataFrame
import dask.delayed
import dask
import edt

__all__ = ['tortuosity_gdd', 'chunks_to_dataframe']
settings.loglevel = 50


@dask.delayed
def calc_g(image, axis):
    r'''Calculates diffusive conductance of an image.

    Parameters
    ----------
    image : np.ndarray
        The binary image to analyze with ``True`` indicating phase of interest.
    axis : int
        0 for x-axis, 1 for y-axis, 2 for z-axis.
    result: int
        0 for diffusive conductance, 1 for both diffusive conductance
        and results object from Porespy.
    '''
    try:
        # if tortuosity_fd fails, throat is closed off from whichever axis was specified
        results = simulations.tortuosity_fd(im=image, axis=axis)

    except Exception:
        # a is diffusive conductance, b is tortuosity
        a, b = (0, 99)

        return (a, b)

    L = image.shape[axis]
    A = np.prod(image.shape)/image.shape[axis]

    return ((results.effective_porosity * A) / (results.tortuosity * L), results)


def network_calc(image, chunk_size, network, phase, bc, axis):
    r'''Calculates the resistor network tortuosity.

    Parameters
    ----------
    image : np.ndarray
        The binary image to analyze with ``True`` indicating phase of interest.
    chunk_size : np.ndarray
        Contains the size of a chunk in each direction.
    bc : tuple
        Contains the first and second boundary conditions.
    axis : int
        The axis to calculate on.

    Returns
    -------
    tau : Tortuosity of the network in the given dimension
    '''
    fd=op.algorithms.FickianDiffusion(network=network, phase=phase)

    fd.set_value_BC(pores=network.pores(bc[0]), values=1)
    fd.set_value_BC(pores=network.pores(bc[1]), values=0)
    fd.run()

    rate_inlet = fd.rate(pores=network.pores(bc[0]))[0]
    L = image.shape[axis] - chunk_size[axis]
    A = np.prod(image.shape) / image.shape[axis]
    d_eff = rate_inlet * L / (A * (1 - 0))

    e = image.sum() / image.size
    D_AB = 1
    tau = e * D_AB / d_eff

    return tau


def chunking(spacing, divs):
    r'''Returns slices given the number of chunks and chunk sizes.

    Parameters
    ----------
    spacing : float
        Size of each chunk.
    divs : list
        Number of chunks in each direction.

    Returns
    -------
    slices : list
        Contains lists of image slices corresponding to chunks
    '''

    slices = [[
    (int(i*spacing[0]), int((i+1)*spacing[0])),
    (int(j*spacing[1]), int((j+1)*spacing[1])),
    (int(k*spacing[2]), int((k+1)*spacing[2]))]
    for i in range(divs[0])
    for j in range(divs[1])
    for k in range(divs[2])]

    return np.array(slices, dtype=int)


def tortuosity_gdd(im, scale_factor=3, use_dask=True):
    r'''Calculates the resistor network tortuosity.

    Parameters
    ----------
    im : np.ndarray
        The binary image to analyze with ``True`` indicating phase of interest

    chunk_shape : list
        Contains the number of chunks to be made in the x,y,z directions.

    Returns
    -------
    results : list
        Contains tau values for three directions, time stamps, tau values for each chunk
    '''
    t0 = time.perf_counter()

    dt = edt.edt(im)
    print(f'Max distance transform found: {np.round(dt.max(), 3)}')

    # determining the number of chunks in each direction, minimum of 3 is required
    if np.all(im.shape//(scale_factor*dt.max())>np.array([3, 3, 3])):

        # if the minimum is exceeded, then chunk number is validated
        # integer division is required for defining chunk shapes
        chunk_shape=np.array(im.shape//(dt.max()*scale_factor), dtype=int)
        print(f"{chunk_shape} > [3,3,3], using {(im.shape//chunk_shape)} as chunk size.")

    # otherwise, the minimum of 3 in all directions is used
    else:
        chunk_shape=np.array([3, 3, 3])
        print(f"{np.array(im.shape//(dt.max()*scale_factor), dtype=int)} <= [3,3,3], \
using {im.shape[0]//3} as chunk size.")

    t1 = time.perf_counter() - t0

    # determines chunk size
    chunk_size = np.floor(im.shape/np.array(chunk_shape))

    # creates the masked images - removes half of a chunk from both ends of one axis
    x_image = im[int(chunk_size[0]//2): int(im.shape[0] - chunk_size[0] //2), :, :]
    y_image = im[:, int(chunk_size[1]//2): int(im.shape[1] - chunk_size[1] //2), :]
    z_image = im[:, :, int(chunk_size[2]//2): int(im.shape[2] - chunk_size[2] //2)]

    t2 = time.perf_counter()- t0

    # creates the chunks for each masked image
    x_slices = chunking(spacing=chunk_size,
                        divs=[chunk_shape[0]-1, chunk_shape[1], chunk_shape[2]])
    y_slices = chunking(spacing=chunk_size,
                        divs=[chunk_shape[0], chunk_shape[1]-1, chunk_shape[2]])
    z_slices = chunking(spacing=chunk_size,
                        divs=[chunk_shape[0], chunk_shape[1], chunk_shape[2]-1])

    t3 = time.perf_counter()- t0
    # queues up dask delayed function to be run in parallel

    x_gD = [calc_g(x_image[x_slice[0, 0]:x_slice[0, 1],
                           x_slice[1, 0]:x_slice[1, 1],
                           x_slice[2, 0]:x_slice[2, 1],],
                           axis=0) for x_slice in x_slices]

    y_gD = [calc_g(y_image[y_slice[0, 0]:y_slice[0, 1],
                           y_slice[1, 0]:y_slice[1, 1],
                           y_slice[2, 0]:y_slice[2, 1],],
                           axis=1) for y_slice in y_slices]

    z_gD = [calc_g(z_image[z_slice[0, 0]:z_slice[0, 1],
                           z_slice[1, 0]:z_slice[1, 1],
                           z_slice[2, 0]:z_slice[2, 1],],
                           axis=2) for z_slice in z_slices]

    # order of throat creation
    all_values = [z_gD, y_gD, x_gD]

    if use_dask:
        all_results = np.array(dask.compute(all_values), dtype=object).flatten()

    else:
        all_values = np.array(all_values).flatten()
        all_results = []
        for item in all_values:
            all_results.append(item.compute())

        all_results = np.array(all_results).flatten()

    # THIS DOESNT WORK FOR SOME REASON
    # all_gD = all_results[::2]
    # all_tau_unfiltered = all_results[1::2]

    all_gD = [result for result in all_results[::2]]
    all_tau_unfiltered = [result for result in all_results[1::2]]

    all_tau = [result.tortuosity if not isinstance(result, int)
               else result for result in all_tau_unfiltered]

    t4 = time.perf_counter()- t0

    # creates opnepnm network to calculate image tortuosity
    net = op.network.Cubic(chunk_shape)
    air = op.phase.Phase(network=net)

    air['throat.diffusive_conductance']=np.array(all_gD).flatten()

    # calculates throat tau in x, y, z directions
    throat_tau = [
    # x direction
    network_calc(image=im,
                 chunk_size=chunk_size,
                 network=net,
                 phase=air,
                 bc=['left', 'right'],
                 axis=1),

    # y direction
    network_calc(image=im,
                 chunk_size=chunk_size,
                 network=net,
                 phase=air,
                 bc=['front', 'back'],
                 axis=2),

    # z direction
    network_calc(image=im,
                 chunk_size=chunk_size,
                 network=net,
                 phase=air,
                 bc=['top', 'bottom'],
                 axis=0)]

    t5 = time.perf_counter()- t0

    output = Results()
    output.__setitem__('tau', throat_tau)
    output.__setitem__('time_stamps', [t1, t2, t3, t4, t5])
    output.__setitem__('all_tau', all_tau)

    return output


def chunks_to_dataframe(im, scale_factor=3, use_dask=True):
    r'''Calculates the resistor network tortuosity.

    Parameters
    ----------
    im : np.ndarray
        The binary image to analyze with ``True`` indicating phase of interest

    chunk_shape : list
        Contains the number of chunks to be made in the x, y, z directions.

    Returns
    -------
    df : pandas.DataFrame
        Contains throat numbers, tau values, diffusive conductance values, and porosity

    '''
    dt = edt.edt(im)
    print(f'Max distance transform found: {np.round(dt.max(), 3)}')

    # determining the number of chunks in each direction, minimum of 3 is required
    if np.all(im.shape//(scale_factor*dt.max())>np.array([3, 3, 3])):

        # if the minimum is exceeded, then chunk number is validated
        # integer division is required for defining chunk shapes
        chunk_shape=np.array(im.shape//(dt.max()*scale_factor), dtype=int)
        print(f"{chunk_shape} > [3,3,3], using {(im.shape//chunk_shape)} as chunk size.")

    # otherwise, the minimum of 3 in all directions is used
    else:
        chunk_shape=np.array([3, 3, 3])
        print(f"{np.array(im.shape//(dt.max()*scale_factor), dtype=int)} <= [3,3,3], \
using {im.shape[0]//3} as chunk size.")

    # determines chunk size
    chunk_size = np.floor(im.shape/np.array(chunk_shape))

    # creates the masked images - removes half of a chunk from both ends of one axis
    x_image = im[int(chunk_size[0]//2): int(im.shape[0] - chunk_size[0] //2), :, :]
    y_image = im[:, int(chunk_size[1]//2): int(im.shape[1] - chunk_size[1] //2), :]
    z_image = im[:, :, int(chunk_size[2]//2): int(im.shape[2] - chunk_size[2] //2)]

    # creates the chunks for each masked image
    x_slices = chunking(spacing=chunk_size,
                        divs=[chunk_shape[0]-1, chunk_shape[1], chunk_shape[2]])
    y_slices = chunking(spacing=chunk_size,
                        divs=[chunk_shape[0], chunk_shape[1]-1, chunk_shape[2]])
    z_slices = chunking(spacing=chunk_size,
                        divs=[chunk_shape[0], chunk_shape[1], chunk_shape[2]-1])

    # queues up dask delayed function to be run in parallel
    x_gD = [calc_g(x_image[x_slice[0, 0]:x_slice[0, 1],
                           x_slice[1, 0]:x_slice[1, 1],
                           x_slice[2, 0]:x_slice[2, 1],],
                           axis=0) for x_slice in x_slices]

    y_gD = [calc_g(y_image[y_slice[0, 0]:y_slice[0, 1],
                           y_slice[1, 0]:y_slice[1, 1],
                           y_slice[2, 0]:y_slice[2, 1],],
                           axis=1) for y_slice in y_slices]

    z_gD = [calc_g(z_image[z_slice[0, 0]:z_slice[0, 1],
                           z_slice[1, 0]:z_slice[1, 1],
                           z_slice[2, 0]:z_slice[2, 1],],
                           axis=2) for z_slice in z_slices]

    # order of throat creation
    all_values = [z_gD, y_gD, x_gD]

    if use_dask:
        all_results = np.array(dask.compute(all_values), dtype=object).flatten()

    else:
        all_values = np.array(all_values).flatten()
        all_results = []
        for item in all_values:
            all_results.append(item.compute())

        all_results = np.array(all_results).flatten()

    all_gD = [result for result in all_results[::2]]
    all_tau_unfiltered = [result for result in all_results[1::2]]

    all_porosity = [result.effective_porosity if not isinstance(result, int)
                    else result for result in all_tau_unfiltered]
    all_tau = [result.tortuosity if not isinstance(result, int)
               else result for result in all_tau_unfiltered]

    # creates opnepnm network to calculate image tortuosity
    net = op.network.Cubic(chunk_shape)

    df = DataFrame(list(zip(np.arange(net.Nt), all_tau, all_gD, all_porosity)),
                        columns=['Throat Number', 'Tortuosity',
                                 'Diffusive Conductance', 'Porosity'])

    return df


if __name__ =="__main__":
    import porespy as ps
    import numpy as np
    np.random.seed(1)
    im = ps.generators.blobs(shape=[100, 100, 100], porosity=0.7)
    res = ps.simulations.tortuosity_gdd(im=im, scale_factor=3, use_dask=True)
    print(res)

    # np.random.seed(2)
    # im = ps.generators.blobs(shape=[100, 100, 100], porosity=0.7)
    # df = ps.simulations.chunks_to_dataframe(im=im, scale_factor=3)
    # print(df)
