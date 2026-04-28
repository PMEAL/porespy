r"""BET-style physisorption (adsorption + desorption) on binary images."""

import numpy as np

from porespy.filters import trim_disconnected_voxels
from porespy.generators import borders
from porespy.tools import (
    Results,
    get_edt,
    get_tqdm,
)

__all__ = ["physisorption"]

R = 8.314  # J/(mol.K), universal gas constant

edt = get_edt()
tqdm = get_tqdm()


def _kelvin_cohan_radius_nm(p, t_nm, gamma, vm, T):
    r"""
    Kelvin-Cohan capillary radius in cylindrical pores, in nm.

    Parameters
    ----------
    p : float or ndarray
        Relative pressure ``p/p0`` in (0, 1).
    t_nm : float or ndarray
        Statistical film thickness in nm.
    gamma : float
        Surface tension of the condensed phase in N/m.
    vm : float
        Molar volume of the condensed phase in m^3/mol.
    T : float
        Temperature in K.

    Returns
    -------
    r_nm : ndarray
        Pore radius in nm at which capillary condensation occurs.

    Notes
    -----
    For cylindrical pores ``ln(p/p0) = -gamma*vm/(R*T*(r - t))``,
    rearranged to ``r = -gamma*vm/(R*T*ln(p/p0)) + t``. The
    ``gamma*vm/(R*T*ln p)`` term is in metres in SI units, so we
    multiply by 1e9 to bring it into nm before adding ``t_nm``.
    """
    r_kelvin_nm = -gamma * vm / (R * T * np.log(p)) * 1e9
    return r_kelvin_nm + t_nm


def _harkins_jura_pressure(t_nm, A, B, C):
    r"""
    Relative pressure from film thickness via Harkins-Jura.

    Parameters
    ----------
    t_nm : float or ndarray
        Statistical film thickness in nm.
    A, B, C : float
        Harkins-Jura coefficients tuned to the chosen adsorbate.

    Returns
    -------
    p : ndarray
        Relative pressure ``p/p0`` corresponding to ``t_nm``.

    Notes
    -----
    The standard form is ``log10(p/p0) = B - A/t_A^2`` with ``t_A``
    the film thickness in Angstroms. The factor ``C = log10(e)``
    converts to natural-log form so the result can be exponentiated.
    """
    t_A = 10.0 * t_nm
    return np.exp((B * t_A**2 - A) / (C * t_A**2))


def physisorption(
    im,
    voxel_size=1.0,
    T=77.0,
    gamma=8.85e-3,
    vm=28.5e-6,
    harkins_jura=(13.99, 0.034, 0.4343),
):
    r"""
    Simulates a physisorption isotherm (adsorption + desorption).

    The simulation sweeps a film thickness over the pore space.
    Harkins-Jura maps each thickness to a relative pressure, and
    Kelvin-Cohan maps that pressure to a capillary-condensation
    radius. A morphological closing of the void space by the
    capillary radius identifies the regions that capillary-condense
    at each step (adsorption branch). The desorption branch is
    obtained by tracking which condensed regions remain connected to
    the image boundary as the film thickness shrinks back down.

    Parameters
    ----------
    im : ndarray
        Boolean array with ``True`` in the void space.
    voxel_size : float, optional
        Edge length of one voxel in nm. The thickness sweep and the
        capillary radius are computed in nm, so ``voxel_size`` is
        what ties the image geometry to the physical scale.
    T : float, optional
        Temperature in K. Default is 77 K (boiling point of N2).
    gamma : float, optional
        Surface tension of the adsorbate in N/m. Default is the
        liquid-N2 value at 77 K (8.85e-3 N/m).
    vm : float, optional
        Liquid molar volume of the adsorbate in m^3/mol. Default is
        the liquid-N2 value (28.5e-6 m^3/mol).
    harkins_jura : tuple of float, optional
        ``(A, B, C)`` coefficients of the Harkins-Jura equation.
        Defaults are the canonical N2-at-77 K values.

    Returns
    -------
    results : Results
        A Results object with the following attributes:

        ============ ===========================================================
        Attribute    Description
        ============ ===========================================================
        ``im_ads``   ndarray of floats. For each voxel, the film thickness
                     (in nm) at which it joined the condensed phase during
                     adsorption. Solid voxels are 0.
        ``t_ads``    ndarray of floats. Film thicknesses (in nm) used in
                     the adsorption sweep.
        ``V_ads``    ndarray. Total condensed volume (voxel count) at
                     each adsorption step.
        ``p_ads``    ndarray. Relative pressures matching ``V_ads``.
        ``im_des``   ndarray. For each voxel, the desorption-step index
                     at which it drained.
        ``t_des``    ndarray of floats. Film thicknesses (in nm) used in
                     the desorption sweep.
        ``V_des``    ndarray. Total condensed volume remaining at each
                     desorption step.
        ``p_des``    ndarray. Relative pressures matching ``V_des``.
        ============ ===========================================================

    Notes
    -----
    Defaults reproduce the standard BET conditions (N2 at 77 K). For
    other fluids supply ``T``, ``gamma``, ``vm``, and ``harkins_jura``
    as a coherent set.

    References
    ----------
    Cohan, L. H. *Sorption hysteresis and the vapor pressure of
    concave surfaces.* J. Am. Chem. Soc. 60 (1938) 433-435.
    Harkins, W. D.; Jura, G. *Surfaces of solids. XII.* J. Am. Chem.
    Soc. 66 (1944) 1366-1373.
    """
    im = np.asarray(im, dtype=bool)
    A, B, C = harkins_jura

    def p_of_t(t_nm):
        return _harkins_jura_pressure(t_nm, A, B, C)

    def r_of_p_t(p, t_nm):
        return _kelvin_cohan_radius_nm(p, t_nm, gamma, vm, T)

    def valid_thicknesses(candidates):
        # Keep only thicknesses for which Harkins-Jura returns a
        # physically meaningful pressure in (0, 1).
        candidates = candidates[candidates > 0]
        p = p_of_t(candidates)
        return candidates[(p > 0) & (p < 1)]

    dt = edt(im) * voxel_size  # film thickness in nm
    V_pore = int(np.sum(im))

    t_ads = valid_thicknesses(np.unique(dt[im]))
    im_ads = np.zeros_like(im, dtype=float)
    V_ads = np.empty(t_ads.size, dtype=float)
    for i, t in enumerate(tqdm(t_ads, desc="adsorption")):
        p = p_of_t(t)
        r_cap = r_of_p_t(p, t)
        film = (dt < t) & im
        vapor_eroded = edt(im & ~film) >= r_cap
        vapor = edt(~vapor_eroded) < r_cap
        condensed = ~vapor & im
        V_ads[i] = np.sum(condensed)
        new = (im_ads == 0) & condensed
        im_ads[new] = t

    t_des = valid_thicknesses(np.unique(dt))[::-1]
    im_des = np.zeros_like(im, dtype=int)
    boundary = borders(im.shape, mode="faces")
    V_des = np.empty(t_des.size, dtype=float)
    for i, t in enumerate(tqdm(t_des, desc="desorption")):
        drained = trim_disconnected_voxels(im_ads > t, inlets=boundary) & im
        V_des[i] = V_pore - int(np.sum(drained))
        new = (im_des == 0) & drained
        im_des[new] = i

    result = Results()
    result.im_ads = im_ads
    result.t_ads = t_ads
    result.V_ads = V_ads
    result.p_ads = p_of_t(t_ads)
    result.im_des = im_des
    result.t_des = t_des
    result.V_des = V_des
    result.p_des = p_of_t(t_des)
    return result


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    import porespy as ps

    im = ps.generators.blobs(shape=[300, 300], porosity=0.6, blobiness=2.0, seed=0)
    im = ps.filters.fill_invalid_pores(im)

    bet = physisorption(im, voxel_size=0.5)
    fig, ax = plt.subplots()
    ax.plot(bet.p_ads, bet.V_ads, "o-", label="adsorption")
    ax.plot(bet.p_des, bet.V_des, "s-", label="desorption")
    ax.set_xlabel("Relative pressure $p/p_0$")
    ax.set_ylabel("Adsorbed volume (voxels)")
    ax.legend()
    plt.show()
