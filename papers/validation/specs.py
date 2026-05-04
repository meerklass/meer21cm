# specifications for the simulation
import numpy as np
import matplotlib.pyplot as plt
from meer21cm.plot import plot_map
from meer21cm.util import create_wcs, redshift_to_freq
from astropy.cosmology import Planck18
from scipy.interpolate import interp1d
from meer21cm.telescope import dish_beam_sigma
from meer21cm import MockSimulation
import scipy.signal.windows as windows
from meer21cm.power import bin_3d_to_cy, bin_3d_to_1d

num_pix_x = 120
num_pix_y = 40

# z_min = 0.8
z_min = 0.6
z_max = 0.8
# z_max = 1.1
nu_min = redshift_to_freq(z_max)
nu_max = redshift_to_freq(z_min)
nu_resol = 132812.5
num_ch = int((nu_max - nu_min) / nu_resol)
nu_arr = np.linspace(nu_min, nu_min + (num_ch - 1) * nu_resol, num_ch)

wcs = create_wcs(
    ra_cr=150,
    dec_cr=-2.5,
    ngrid=[num_pix_x, num_pix_y],
    resol=0.5,
)

ra_range = [125, 175]
dec_range = [-10.1, 5]

# dndz_data = np.load("LRG_dndz.npz")
dndz_data = np.load("LRGELG_dndz.npz")
z_bin = dndz_data["z_bin"]
z_count = dndz_data["z_count"]
z_cen = (z_bin[:-1] + z_bin[1:]) / 2
dV_arr = Planck18.differential_comoving_volume(z_cen).value

# LRG3, DESI DR1
# n_gal = 859824 / 5 / 1e9 #Mpc-3
# LRG2, DESI DR1
n_gal = 771875 / 4 / 1e9  # Mpc-3

k1dbins = np.linspace(0.003, 0.2, 25)[1:]
kperpbins = np.linspace(0, 0.048, 17)[2:]
kparabins = np.linspace(0, 0.5, 51)
window_name = "blackmanharris"

z_func = interp1d(
    z_cen, z_count / dV_arr, kind="linear", bounds_error=False, fill_value=0
)
sigma_beam_ch = dish_beam_sigma(13.5, nu_arr)

hi_bias = 1.0
gal_bias = 1.0
sigma_v_hi = 100
sigma_v_gal = 100
omega_hi = 5e-4

grid_scheme = "cic"
sim_upres_transverse = 1 / 2
sim_upres_radial = 1 / 2
ps_downres_transverse = 3
ps_downres_radial = 6


def plot_cy_power(xbins, ybins, pdatacy, pmodcy, vmin_ratio, vmax_ratio):
    arr = np.array(
        [
            np.log10(pdatacy.mean(axis=0).T),
            np.log10(pmodcy.T),
        ]
    )
    vmin = np.nanmin(arr)
    vmax = np.nanmax(arr)
    fig, axes = plt.subplots(1, 3)
    axes[0].pcolormesh(
        xbins,
        ybins,
        np.log10(pdatacy.mean(axis=0).T),
        vmin=vmin,
        vmax=vmax,
    )
    im = axes[1].pcolormesh(
        xbins,
        ybins,
        np.log10(pmodcy.T),
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(im, ax=axes[:-1], location="top", fraction=0.046, pad=0.04)
    im = axes[2].pcolormesh(
        xbins,
        ybins,
        (pdatacy.mean(axis=0).T) / (pmodcy.T),
        vmin=vmin_ratio,
        vmax=vmax_ratio,
        cmap="bwr",
    )
    plt.colorbar(im, ax=axes[2], location="top", fraction=0.046, pad=0.04)
    return fig


def plot_1d_power(
    keff,
    pdatad,
    pmodd,
    ratio_min,
    ratio_max,
):
    keff = np.array(keff)
    pdatad = np.array(pdatad)
    pmodd = np.array(pmodd)
    sel = keff == keff
    keff = keff[sel]
    pdatad = pdatad[:, sel]
    pmodd = pmodd[sel]
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 5),
        sharex=True,
        height_ratios=[2, 1],
    )
    axes[0].errorbar(
        keff,
        pdatad.mean(axis=0) * keff,
        yerr=pdatad.std(axis=0) * keff,
        label="mock",
    )
    axes[0].plot(keff, pmodd * keff, label="model", ls="--")
    axes[0].set_ylim(np.nanmin(pmodd * keff) * 0.7, np.nanmax(pmodd * keff) * 1.2)
    axes[0].legend()
    axes[1].errorbar(
        keff,
        (pdatad.mean(axis=0)) / (pmodd) - 1,
        yerr=(pdatad.std(axis=0)) / (pmodd),
    )
    axes[1].axhline(0, color="black", ls="--")
    axes[1].fill_between(
        np.linspace(keff.min() - 0.005, keff.max() + 0.005, 100),
        -0.05,
        0.05,
        color="black",
        alpha=0.2,
    )
    axes[1].set_xlim(keff.min() - 0.005, keff.max() + 0.005)
    axes[1].set_ylim(ratio_min, ratio_max)
    axes[1].legend()
    return fig


def get_mock(seed):
    mock = MockSimulation(
        wproj=wcs,
        num_pix_x=num_pix_x,
        num_pix_y=num_pix_y,
        ra_range=ra_range,
        dec_range=dec_range,
        nu=nu_arr,
        discrete_source_dndz=z_func,
        seed=seed,
        tracer_bias_2=gal_bias,
        tracer_bias_1=hi_bias,
        mean_amp_1="average_hi_temp",
        omega_hi=omega_hi,
        # sigma_beam_ch=sigma_beam_ch,
        sigma_v_1=sigma_v_hi,
        sigma_v_2=sigma_v_gal,
    )
    mock.taper_func = getattr(windows, window_name)
    return mock


def bin_power_cy(
    power_3d,
    k_perp,
    k_para,
    kperpbins,
    kparabins,
    kweights=None,
):
    pcy_arr = bin_3d_to_cy(
        power_3d,
        k_perp,
        kperpbins,
        vectorize=True,
        weights=kweights,
    )
    pcy_arr = bin_3d_to_cy(
        np.nan_to_num(pcy_arr),
        np.abs(k_para),
        kparabins,
        vectorize=True,
        weights=(1 - np.isnan(pcy_arr))[0].astype("float"),
    )
    return pcy_arr


def bin_power_1d(
    power_3d,
    k_mode,
    k1dbins,
    kweights,
    num_split=None,
):
    if num_split is None:
        p1d, keff, nmodes = bin_3d_to_1d(
            power_3d,
            k_mode,
            k1dbins,
            vectorize=True,
            weights=kweights,
        )
    else:
        p1d = []
        power_3d_arr = np.array_split(power_3d, num_split)
        for i in range(num_split):
            pdata1darr_i, keff, nmodes = bin_3d_to_1d(
                power_3d_arr[i],
                k_mode,
                k1dbins,
                vectorize=True,
                weights=kweights,
            )
            p1d.append(pdata1darr_i)
        p1d = np.concatenate(p1d)
    return p1d, keff, nmodes


def get_cy(
    p3d,
    karr,
    weights=None,
):
    kperp, kpara, kmode, kvec = karr
    if len(p3d.shape) == 3:
        p3d = p3d.copy()[None]
    results = bin_power_cy(
        p3d.mean(0)[None],
        kperp,
        kpara,
        kperpbins,
        kparabins,
        kweights=weights,
    )
    return results[0]


def get_1d(p3d, karr, weights=None, num_split=None):
    kperp, kpara, kmode, kvec = karr
    if len(p3d.shape) == 3:
        p3d = p3d.copy()[None]
        num_split = None
    p1d, k1d, _ = bin_power_1d(
        p3d,
        kmode,
        k1dbins,
        weights,
        num_split=num_split,
    )
    return p1d, k1d
