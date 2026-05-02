import matplotlib.pyplot as plt
import numpy as np
from meer21cm.plot import *
from meer21cm.util import create_wcs, pca_clean
from meer21cm.dataanalysis import Specification


def test_plt(test_W, test_nu, test_wproj):
    plt.switch_backend("Agg")
    plot_pixels_along_los(test_W, test_W, zaxis=test_nu[:1])
    plot_eigenspectrum(np.array([1, 2]))
    plot_map(test_W, test_wproj, W=test_W, vmin=0, vmax=1)
    plot_map(test_W, test_wproj, vmin=0, vmax=1)
    plt.close("all")
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(20, 20),
        subplot_kw={"projection": test_wproj},
    )
    plot_map(test_W, test_wproj, W=test_W, vmin=0, vmax=1, ax=axes[0])
    plot_map(test_W, test_wproj, vmin=0, vmax=1, ax=axes[1])
    plt.close("all")


def test_plot_map_healpix_gnomview():
    """Smoke test: sparse HEALPix + gnomview (no display)."""
    plt.switch_backend("Agg")
    nside = 8
    pid = np.array([0, 14, 100, 201], dtype=np.int64)
    v = np.linspace(0.0, 1.0, pid.size)
    plot_map(
        v,
        wproj=None,
        pixel_id=pid,
        hp_nside=nside,
        title="hp",
        vmin=0,
        vmax=1,
        cbar_label="K",
    )
    ax = plt.gca()
    assert ax.axison
    assert ax.get_xlabel() == "R.A [deg]"
    assert ax.get_ylabel() == "Dec. [deg]"
    plt.close("all")


def test_plot_map_healpix_frequency_average():
    """(n_pix, n_chan) is averaged along the last axis before plotting."""
    plt.switch_backend("Agg")
    nside = 8
    pid = np.arange(10, dtype=np.int64)
    n_ch = 5
    cube = np.random.default_rng(0).standard_normal((pid.size, n_ch))
    W = np.ones_like(cube)
    plot_map(
        cube,
        wproj=None,
        pixel_id=pid,
        hp_nside=nside,
        W=W,
        have_cbar=False,
    )
    plt.close("all")


def test_plot_map_healpix_cartview_ra_dec_range():
    """healpy.cartview path with explicit lonra/latra-style bounds."""
    plt.switch_backend("Agg")
    nside = 4
    pid = np.array([0, 5, 20], dtype=np.int64)
    v = np.ones(pid.size)
    plot_map(
        v,
        wproj=None,
        pixel_id=pid,
        hp_nside=nside,
        ra_range=(0.0, 90.0),
        dec_range=(-30.0, 30.0),
        xsize=120,
        ysize=80,
        have_cbar=False,
    )
    plt.close("all")


def test_plot_projected_map():
    plt.switch_backend("Agg")
    wcs = create_wcs(ra_cr=0, dec_cr=-30, ngrid=[100, 200], resol=[0.1, 0.1])
    test_arr = np.random.normal(size=(100, 200, 200))
    test_arr[:, :, :20] = np.nan
    test_res, test_A = pca_clean(test_arr, 1, return_A=True, ignore_nan=True)
    plot_projected_map(test_A, test_res, wcs)
    plt.close("all")


def test_plot_patch_split():
    ps = Specification(
        survey="meerklass_2021",
        band="L",
        ra_range=(334, 357),
        dec_range=(-35, -26.5),
    )
    mask_arr = ps.get_jackknife_patches(ra_patch_num=8, dec_patch_num=4, nu_patch_num=2)
    visualise_patch_split(mask_arr, ps.wproj)
    plt.close("all")
