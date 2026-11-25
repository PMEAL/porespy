import numpy as np


if __name__ == "__main__":
    from copy import copy
    import porespy as ps
    import matplotlib.pyplot as plt

    edt = ps.tools.get_edt()

    ps.visualization.set_mpl_style()
    cm = copy(plt.cm.plasma)
    cm.set_under('grey')
    # cm.set_over('grey')
    cm.set_bad('k')
    conn = 'min'
    steps = 50

    im = ~ps.generators.random_spheres(
        [400, 400],
        r=10,
        # phi=0.15,
        clearance=10,
        seed=16,
        edges="extended",
        phi=0.25,
    )
    inlets = ps.generators.faces(shape=im.shape, inlet=0)
    outlets = ps.generators.faces(shape=im.shape, outlet=0)
    dt = edt(im)
    pc = ps.filters.capillary_transform(
        im=im,
        dt=dt,
        sigma=0.072,
        theta=180,
        g=0,
        voxel_size=1e-5,
    )

    drn1 = ps.simulations.drainage(
        im=im,
        pc=pc,
        inlets=inlets,
        outlets=outlets,
        steps=steps,
        conn=conn,
    )

    imb2 = ps.simulations.imbibition(
        im=im,
        pc=pc,
        inlets=outlets,
        outlets=inlets,
        residual=drn1.im_trapped,
        steps=steps,
        conn=conn,
    )

    drn2 = ps.simulations.drainage(
        im=im,
        pc=pc,
        inlets=inlets,
        outlets=outlets,
        residual=imb2.im_trapped,
        steps=steps,
        conn=conn,
    )

# %%
    tmp = np.copy(drn2.im_seq).astype(float)
    # tmp[drn2.im_trapped] = tmp.max() + 1
    tmp[~im] = np.nan

    fig, ax = plt.subplots(figsize=[5, 5])
    ax.imshow(tmp, cmap=cm, vmin=0, origin='lower')
    ax.axis(False)
