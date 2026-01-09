import numpy as np
from edt import edt
import matplotlib.animation as animation
import imageio_ffmpeg
from matplotlib import rcParams
from porespy.filters import (
    fill_closed_pores,
)
from porespy.tools import (
    Results,
)


def physisorption(im):
    # Physical properties of liquid nitrogen
    T = 77
    R = 8.314
    gam = 8.85*10**-3
    vm = 28.5*10**-6
    film_thickness = np.linspace(0.05, 7, 100)
    # Rearranged Harkins-Jura Equation for relative pressure
    relative_pressure = 10**((0.034-13.99/(100*film_thickness**2))/0.4343)

    # Kelvin Cohan equation to determine radius for capilary condensation
    def kelvin_cohan(p, th):
        return -gam*vm/(R*T*np.log(p)) + th

    epsilon = []  # Array which will hold porosity values for each image
    adsorbed_volume = []
    dt = edt(im)  # Euclidian distance transform on initial image
    im_ads = np.zeros_like(im, dtype=int)

    for i in range(np.size(film_thickness)):
        # Find radius of structuring element for the capilary condensation
        p = relative_pressure[i]
        th = film_thickness[i]
        r = kelvin_cohan(p, th)
        capiliary_radius = film_thickness[i] + r

        # Isolating the film around the pores
        film_adsorption = (dt <= th) * im

        # Closing operation using structuring element of radius rn
        im_closed = edt(im * ~film_adsorption) > capiliary_radius
        im_closed_dil = edt(~im_closed) <= capiliary_radius

        adsorbed_volume.append(np.sum(im) - np.sum(im_closed_dil))
        epsilon.append(im_closed_dil.sum() / im_closed_dil.size)
        mask = (im_ads == 0)*(im_closed_dil == 0)
        im_ads[mask] = i

    im_final = im_closed_dil
    desorbed_volume = []
    im_des = np.zeros_like(im, dtype=int)

    for i in range(np.size(film_thickness), 0, -1):
        p = relative_pressure[i-1]
        th = film_thickness[i-1]
        r = kelvin_cohan(p, th)
        capiliary_radius = film_thickness[i-1] + r

        # Opening using structuring element of radius rn
        im_open = edt(im) > capiliary_radius
        im_open_dil = edt(~im_open) <= capiliary_radius

        film = edt(im_open_dil) > film_thickness[i-1]
        film = fill_closed_pores(film)
        desorbed_volume.append(np.sum(im) - np.sum(film))
        mask = (im_ads == 0)*(im_closed_dil == 0)
        im_des[mask] = i-1

    result = Results()
    result.im_ads = im_ads
    result.V_ads = adsorbed_volume
    result.p_ads = relative_pressure
    result.im_des = im_des
    result.V_des = desorbed_volume
    result.p_des = relative_pressure[-1::-1]
    return result


if __name__ == "__main__":
    import porespy as ps
    import matplotlib.pyplot as plt

    im = ps.generators.blobs(shape=[500,500], porosity=0.6, blobiness=3, seed=0)
    im = ps.filters.fill_invalid_pores(im)

    bet = physisorption(im)
    fig, ax = plt.subplots()
    ax.plot(bet.p_ads, bet.V_ads)
    ax.plot(bet.p_des, bet.V_des)
