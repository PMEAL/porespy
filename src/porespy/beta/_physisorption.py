import numpy as np
from porespy.generators import borders
from porespy.filters import (
    fill_closed_pores,
    trim_disconnected_voxels,
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
    T = 77.0  # K
    R = 8.314  # J/(mol.K)

    def kelvin_cohan(p, th):
        # Kelvin Cohan equation to determine radius for capilary condensation
        gam = 8.85*10**-3
        vm = 28.5*10**-6  # Units?
        return -gam*vm/(R*T*np.log(p)) + th

    def relative_pressure(thickness):
        # Rearranged Harkins-Jura Equation for relative pressure
        A = 13.99
        B = 0.034
        C = 0.4343
        D = 10*thickness
        PPo = np.exp((B*D**2 - A)/(C*D**2))
        return PPo

    adsorbed_volume = []
    Vpore = np.sum(im)
    dt = edt(im)  # Euclidian distance transform on initial image
    im_ads = np.zeros_like(im, dtype=int)

    sizes1 = np.unique(dt[im].astype(int))
    for thickness in tqdm(sizes1):
        # Find radius of structuring element for the capillary condensation
        p = relative_pressure(thickness)
        r = kelvin_cohan(p, thickness)
        capillary_radius = thickness + r

        # Isolate the film on pore walls
        film = (dt < thickness) * im

        # Close film using structuring element of radius rn
        im_closed = edt(im * ~film) >= capillary_radius
        im_closed_dil = edt(~im_closed) < capillary_radius

        adsorbed_volume.append(Vpore - np.sum(im_closed_dil * im))
        mask = (im_ads == 0)*(im_closed_dil == 0)
        im_ads[mask] = thickness

    desorbed_volume = []
    im_des = np.zeros_like(im, dtype=int)
    boundary = borders(im.shape, mode='faces')
    sizes2 = np.unique(dt.astype(int))[-1::-1]
    for i, thickness in enumerate(tqdm(sizes2)):
        p = relative_pressure(thickness)
        r = kelvin_cohan(p, thickness)
        capillary_radius = thickness + r

        # Opening using structuring element of radius rn
        im_open = im_ads > thickness
        film = trim_disconnected_voxels(im_open, inlets=boundary)

        desorbed_volume.append(Vpore - np.sum(film * im))
        mask = (im_des == 0) * film * im
        im_des[mask] = i

    result = Results()
    result.im_ads = im_ads
    result.V_ads = adsorbed_volume
    result.p_ads = relative_pressure(sizes1)
    result.im_des = im_des
    result.V_des = desorbed_volume
    result.p_des = relative_pressure(sizes2)
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

    im = ps.generators.blobs(shape=[1200, 1200], porosity=0.6, blobiness=2.5, seed=0)
    im = ps.filters.fill_invalid_pores(im)

    bet = physisorption(im)
    fig, ax = plt.subplots()
    ax.plot(bet.p_ads, bet.V_ads, label='Adsorb')
    ax.plot(bet.p_des, bet.V_des, label='Desorb')
    ax.legend()

    # Generate animation
    if im.ndim == 2:
        # im_ani = bet.im_ads.copy()
        im_ani = bet.im_ads.copy()
        N = np.unique(im_ani[im])
        stk = np.zeros([len(N)]+list(im.shape))
        # Create stack of images to show
        for i, s in enumerate(N):
            mask = im * (im_ani < s) * i
            mask = mask.astype(float)
            mask[mask == 0] = max(N) + 1
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
            interval=250,
            blit=True,
        )
        rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        ani.save("Adsoprtion.mp4", writer='ffmpeg')
