import numpy as np
import openpnm as op
from porespy.filters import trim_nonpercolating_paths
import collections


def tortuosity(im, axis, return_im=False):
    r"""
    Calculates tortuosity of given image in specified direction

    Parameters
    ----------
    im : ND-image
        The binary image to analyze with ``True`` indicating phase of interest
    axis : int
        The axis along which to apply boundary conditions
    return_im : boolean
        If ``True`` then the resulting tuple contains a copy of the input
        image with the concentration profile.

    Returns
    -------
    results :  tuple
        A named-tuple containing the calculated ``tortuosity`` value, the
        ``original_porosity`` of the image, the ``effective_porosity`` (with
        blind and non-percolating regions removed), and ``return_im`` is
        ``True``, then ``image`` will contain a copy of ``im`` with
        concentration values from the simulation.

    """
    if axis > (im.ndim - 1):
        raise Exception("Axis argument is too high")
    # Obtain original porosity
    porosity_orig = im.sum()/im.size
    # removing floating pores
    im = trim_nonpercolating_paths(im, inlet_axis=axis, outlet_axis=axis)
    # porosity is changed because of trimmimg floating pores
    porosity_true = im.sum()/im.size
    if porosity_true < porosity_orig:
        print('Caution, True porosity is:', porosity_true,
              'and volume fraction filled:',
              abs(porosity_orig-porosity_true)*100, '%')
    # cubic network generation
    net = op.network.CubicTemplate(template=im, spacing=1)
    # adding phase
    water = op.phases.Water(network=net)
    water['throat.diffusive_conductance'] = 1  # dummy value
    # running Fickian Diffusion
    fd = op.algorithms.FickianDiffusion(network=net, phase=water)
    # choosing axis of concentration gradient
    inlet = net['pore.coords'][:, axis] <= 1
    outlet = net['pore.coords'][:, axis] >= im.shape[axis]-1
    # boundary conditions on concentration
    fd.set_value_BC(pores=inlet, values=1)
    fd.set_value_BC(pores=outlet, values=0)
    fd.settings['solver_type'] = 'spsolve'
    fd.run()
    # calculating molar flow rate, effective diffusivity and tortuosity
    rate_outlet = fd.rate(pores=outlet)[0]
    tau = porosity_true*(1/abs(rate_outlet))
    result = collections.namedtuple('tortuosity_result', ['tortuosity',
                                                          'original_porosity',
                                                          'effective_porosity',
                                                          'image'])
    result.tortuosity = tau
    result.original_porosity = porosity_orig
    result.effective_porosity = porosity_true
    if return_im:
        conc = np.zeros([im.size, ], dtype=float)
        conc[net['pore.template_indices']] = fd['pore.concentration']
        conc = np.reshape(conc, newshape=im.shape)
        result.image = conc
    else:
        result.image = None
    return result
