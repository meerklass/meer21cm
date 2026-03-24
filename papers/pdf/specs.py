# specifications for the simulation
import numpy as np
import matplotlib.pyplot as plt
from meer21cm.plot import plot_map
from meer21cm.util import create_wcs, redshift_to_freq, HiddenPrints
from astropy.cosmology import Planck18
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from meer21cm import MockSimulation
from meer21cm.grid import project_particle_to_regular_grid
import warnings
import os
import sys
import contextlib

mock = MockSimulation()
fiducial_cosmology = mock.fiducial_cosmology
Omfid = fiducial_cosmology["omega_cold"]
Asfid = fiducial_cosmology["As"]
bgfid = 2.0

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
dndz_data = np.load("../validation/LRGELG_dndz.npz")
z_bin = dndz_data["z_bin"]
z_count = dndz_data["z_count"]
z_cen = (z_bin[:-1] + z_bin[1:]) / 2
dV_arr = Planck18.differential_comoving_volume(z_cen)

# LRG3, DESI DR1
# n_gal = 859824 / 5 / 1e9 #Mpc-3
# LRG2, DESI DR1
n_gal = 771875 / 4 / 1e9  # Mpc-3

k1dbins = np.linspace(0.003, 0.2, 25)[1:]
kperpbins = np.linspace(0, 0.048, 17)[2:]
kparabins = np.linspace(0, 0.5, 51)


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
    axes[0].set_ylim((pmodd * keff).min() * 0.7, (pmodd * keff).max() * 1.2)
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


def sim_pdf(params, bins=np.linspace(-1, 2.5, 100), seed=None):
    z_func = interp1d(
        z_cen, z_count / dV_arr, kind="linear", bounds_error=False, fill_value=0
    )
    input_cosmo = fiducial_cosmology.copy()
    input_cosmo.update(
        As=params["As"],
        omega_cold=params["omega_cold"],
    )
    mock = MockSimulation(
        wproj=wcs,
        num_pix_x=num_pix_x,
        num_pix_y=num_pix_y,
        ra_range=ra_range,
        dec_range=dec_range,
        nu=nu_arr,
        discrete_source_dndz=z_func,
        seed=seed,
        tracer_bias_2=params["bgal"],
        sigma_v_2=100,
        true_cosmology=input_cosmo,
    )
    mock.downres_factor_transverse = 1 / 2
    mock.downres_factor_radial = 2
    num_gal = int(mock.survey_volume * n_gal)
    mock.num_discrete_source = num_gal
    _, _, gal_count = project_particle_to_regular_grid(
        mock.mock_tracer_position_in_box,
        mock.box_len,
        mock.box_ndim,
    )
    gal_over_den = gal_count / gal_count.mean() - 1
    sigma_smooth = 8 / mock.cosmo.h
    sigma_kernel = sigma_smooth / mock.box_resol
    smooth_over_den = gaussian_filter(gal_over_den, sigma_kernel)
    smooth_hist, _ = np.histogram(smooth_over_den.flatten(), bins=bins)
    return smooth_hist


def log_likelihood(params, hist_data=None, bins=np.linspace(-1, 2.5, 100)):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        hist_model_arr = sim_pdf(params, bins=bins)
        bin_sel = (hist_model_arr > 0) * (hist_data > 0)
        pdf_model = hist_model_arr / hist_model_arr.sum()
        pdf_data = hist_data / hist_data.sum()
        logl_plus = hist_data.sum() * (
            (
                -pdf_data[bin_sel] * np.log(pdf_data[bin_sel])
                + pdf_data[bin_sel] * np.log(pdf_model[bin_sel])
            ).sum()
        )
    return logl_plus


def log_likelihood_cov(params, hist_data=None, hist_inv_cov=None, bins=np.linspace(-1, 2.5, 100)):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        hist_model_arr = sim_pdf(params, bins=bins)
        logl = (
            -0.5
            * (hist_model_arr - hist_data)
            @ hist_inv_cov
            @ (hist_model_arr - hist_data)
        )
    return logl
