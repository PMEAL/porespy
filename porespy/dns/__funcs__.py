import collections
import openpnm as op
import porespy as ps


def tortuosity():
    r"""
    Calculates tortuosity of given image in specified direction
    
    Parameters
    ----------
    im : ND-image
        The binary image to analyze with ``True`` indicating phase of interest
    axis : int
        The axis along which to apply boundary conditions
        
    Returns
    -------
    results : tuple
        A named-tuple containing the ``tortuosity``, ``conc_field``, percolating_porosity, 
        
    """
    if axis > (im.ndim -1):
        raise Exception("Axis argument is too high")
    # removing floating pores
    im = ps.filters.trim_nonpercolating_paths(im, 
                                              inlet_axis=axis, 
                                              outlet_axis=axis)
    # porosity is changed because of trimmimg floating pores
    porosity_true = im.sum()/im.size
    print('Caution, True porosity is:', porosity_true,
          'and volume fraction filled:', abs(p-porosity_true)*100, '%')
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
    Tau = porosity_true*(1/abs(rate_outlet))
    image = collections.namedtuple('image', ['flow_rate', 'eff_D', 'Tortuosity'])
    im_1 = image(rate_outlet, D_eff, Tau)
    return im_1[2]
