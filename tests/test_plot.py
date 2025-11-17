import matplotlib.pyplot as plt
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
