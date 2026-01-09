import numpy as np
from porespy.filters import (
    fill_closed_pores,
)
from porespy.tools import (
    Results,
    get_edt,
    get_tqdm,
)

tqdm = get_tqdm()
edt = get_edt()


def physisorption(im):
    # Physical properties of liquid nitrogen
    T = 77  # K
    R = 8.314  # J/(mol.K)
    film_thickness = np.linspace(0.05, 7, 100)  # Why 7, and what units?
    # Rearranged Harkins-Jura Equation for relative pressure
    Pr = 10**((0.034-13.99/(100*film_thickness**2))/0.4343)  # Exceeds 1.0

    # Kelvin Cohan equation to determine radius for capilary condensation
    def kelvin_cohan(p, th):
        gam = 8.85*10**-3
        vm = 28.5*10**-6  # Units?
        return -gam*vm/(R*T*np.log(p)) + th

    adsorbed_volume = []
    dt = edt(im)  # Euclidian distance transform on initial image
    im_ads = np.zeros_like(im, dtype=int)

    for i in tqdm(range(np.size(film_thickness))):
        # Find radius of structuring element for the capillary condensation
        p = Pr[i]
        th = film_thickness[i]
        r = kelvin_cohan(p, th)
        capillary_radius = film_thickness[i] + r

        # Isolate the film on pore walls
        film_adsorption = (dt <= th) * im

        # Close film using structuring element of radius rn
        im_closed = edt(im * ~film_adsorption) > capillary_radius
        im_closed_dil = edt(~im_closed) <= capillary_radius

        adsorbed_volume.append(np.sum(im) - np.sum(im_closed_dil))
        mask = (im_ads == 0)*(im_closed_dil == 0)
        im_ads[mask] = i

    im_final = im_closed_dil
    desorbed_volume = []
    im_des = np.zeros_like(im, dtype=int)

    for i in tqdm(range(np.size(film_thickness), 0, -1)):
        p = Pr[i-1]
        th = film_thickness[i-1]
        r = kelvin_cohan(p, th)
        capillary_radius = film_thickness[i-1] + r

        # Opening using structuring element of radius rn
        im_open = edt(im) > capillary_radius
        im_open_dil = edt(~im_open) <= capillary_radius

        film = edt(im_open_dil) > film_thickness[i-1]
        film = fill_closed_pores(film)
        desorbed_volume.append(np.sum(im) - np.sum(film))
        mask = (im_des == 0)*(im_open_dil == 0)*im
        im_des[mask] = i-1

    result = Results()
    result.im_ads = im_ads
    result.V_ads = adsorbed_volume
    result.p_ads = Pr
    result.im_des = im_des
    result.V_des = desorbed_volume
    result.p_des = Pr[-1::-1]
    return result


if __name__ == "__main__":
    import porespy as ps
    import matplotlib.pyplot as plt
    from copy import copy
    import matplotlib.animation as animation
    import imageio_ffmpeg
    from matplotlib import rcParams

    cm = copy(plt.cm.plasma)
    cm.set_under('black')
    cm.set_bad('white')
    cm.set_over('grey')

    im = ps.generators.blobs(shape=[200, 200], porosity=0.6, blobiness=1, seed=0)
    im = ps.filters.fill_invalid_pores(im)

    bet = physisorption(im)
    fig, ax = plt.subplots()
    ax.plot(bet.p_ads, bet.V_ads, label='Adsorb')
    ax.plot(bet.p_des, bet.V_des, label='Desorb')
    ax.legend()

    # Generate animation
    if im.ndim == 2:
        im_ani = bet.im_des.copy()
        N = np.unique(im_ani)
        stk = np.zeros([len(N)]+list(im.shape))
        # Create stack of images to show
        for i, s in enumerate(N):
            mask = im * (im_ani < i) * i
            mask[mask == 0] = max(N)
            mask[~im] = -1
            stk[i, ...] = mask

        fig2, ax2 = plt.subplots()
        animated_image = ax2.imshow(
            stk[0, ...],
            origin='lower',
            cmap=cm,
            vmin=0,
            vmax=max(N),
            interpolation='none',
        )
        ax2.axis(False)

        def update(i):
            animated_image.set_data(stk[i, :])
            return (animated_image,)

        ani = animation.FuncAnimation(
            fig=fig2,
            func=update,
            frames=len(stk),
            interval=60,
            blit=True,
        )
        rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        ani.save("Adsoprtion.mp4", writer='ffmpeg')
