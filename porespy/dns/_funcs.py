from deprecated import deprecated
from porespy.simulations import tortuosity as _tortuosity


@deprecated("The dns module has been renamed simulations and tortuosity was moved there")
def tortuosity(*args, **kwargs):
    return _tortuosity(*args, **kwargs)
