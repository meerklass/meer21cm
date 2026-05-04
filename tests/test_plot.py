import matplotlib.pyplot as plt
import numpy as np
from meer21cm.plot import *
import meer21cm.plot as plotmod
from meer21cm.util import create_wcs, pca_clean
from meer21cm.dataanalysis import Specification
import pytest


@pytest.mark.parametrize("zero_centre", [True, False])
def test_plt(test_W, test_nu, test_wproj, zero_centre):
    plt.switch_backend("Agg")
    plot_pixels_along_los(test_W, test_W, zaxis=test_nu[:1])
    plot_eigenspectrum(np.array([1, 2]))
    plot_map(
        np.random.normal(size=test_W.shape),
        test_wproj,
        W=test_W,
        vmin=-1,
        vmax=1,
        ZeroCentre=zero_centre,
    )
    plot_map(
        np.random.normal(size=test_W.shape),
        test_wproj,
        vmin=-1,
        vmax=1,
        ZeroCentre=zero_centre,
    )
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


def test_plot_pixels_along_los():
    ps = Specification(
        survey="meerklass_2021",
        band="L",
        ra_range=(334, 357),
        dec_range=(-35, -26.5),
    )
    plot_pixels_along_los(ps.data, ps.w_HI, zaxis=None)
    plt.close("all")


def test_plot_map_hp():
    plt.switch_backend("Agg")
    with pytest.raises(
        ValueError,
        match="plot_map_healpix does not support ax= yet; use plot_map_wcs for subplot grids.",
    ):
        plot_map_healpix(None, None, None, ax=plt.gca())


def test_plot_map_healpix_validation_errors():
    plt.switch_backend("Agg")
    nside = 8
    pid = np.arange(4, dtype=np.int64)
    with pytest.raises(ValueError, match="expects map_plot shape"):
        plot_map_healpix(np.zeros((2, 2, 2)), pid, nside, have_cbar=False)
    with pytest.raises(ValueError, match="does not match pixel_id length"):
        plot_map_healpix(np.zeros(3), pid, nside, have_cbar=False)
    with pytest.raises(ValueError, match="W must match map_plot shape"):
        plot_map_healpix(np.zeros(4), pid, nside, W=np.ones(3), have_cbar=False)
    bad_pid = np.array([0, 1, 2, 9999], dtype=np.int64)
    with pytest.raises(ValueError, match="pixel_id out of range"):
        plot_map_healpix(np.zeros(4), bad_pid, nside, have_cbar=False)


def test_plot_map_healpix_zero_center_branch():
    plt.switch_backend("Agg")
    nside = 8
    pid = np.arange(8, dtype=np.int64)
    vals = np.linspace(-1.0, 1.0, pid.size)
    plot_map_healpix(vals, pid, nside, ZeroCentre=True, have_cbar=False)
    plt.close("all")


def test_plot_map_dispatch_validation():
    with pytest.raises(ValueError, match="Pass either ``wproj`` .* not both"):
        plot_map(
            np.ones(3),
            wproj=create_wcs(0, -30, [4, 4], [1, 1]),
            pixel_id=np.array([0, 1, 2]),
            hp_nside=1,
        )
    with pytest.raises(ValueError, match="hp_nside is required"):
        plot_map(np.ones(3), pixel_id=np.array([0, 1, 2]))
    with pytest.raises(TypeError, match="plot_map requires ``wproj``"):
        plot_map(np.ones((4, 4)))


def test_plot_helpers_edge_ranges():
    assert plotmod._normalize_lonra(10.0, 10.0) == (10.0, 370.0)
    assert plotmod._apply_ra_buffer(0.0, 360.0, 1.0, 0.5) == [0.0, 360.0]
    lo, hi = plotmod._apply_ra_buffer(20.0, 20.0, 0.0, 1.5)
    assert hi > lo
    lo2, hi2 = plotmod._apply_ra_buffer(20.0, 20.0 + 1e-12, 0.0, 1.5)
    assert hi2 > lo2
    lat0, lat1 = plotmod._apply_dec_buffer(90.0, 90.0, 0.0)
    assert lat1 > lat0
    with pytest.raises(ValueError, match="positive span"):
        plotmod._cart_extent_from_ranges([1.0, 1.0], [0.0, 1.0], 0.5)


def test_apply_dec_buffer_second_guard_pathological(monkeypatch):
    # Pathological case: force clip(mid) to +inf so second ``if a >= b`` is taken.
    monkeypatch.setattr(plotmod.np, "clip", lambda x, a, b: float("inf"))
    out = plotmod._apply_dec_buffer(0.0, 0.0, 0.0)
    assert np.isinf(out[0])
    assert np.isinf(out[1])


def test_healpix_tick_formatters_branches():
    class DummyRaMap:
        def get_ylim(self):
            return (0.0, 0.0)

        def get_lonlat(self, x, y):
            if x == -1:
                return None
            if x == -2:
                return (np.nan, 0.0)
            if x == 0:
                return (359.9999999, 0.0)
            if x == 1:
                return (12.3, 0.0)
            return (0.0, 12.3)

    class DummyDecMap:
        def get_xlim(self):
            return (0.0, 0.0)

        def get_lonlat(self, x, y):
            if y == -1:
                return None
            if y == -2:
                return (0.0, np.nan)
            if y == 0:
                return (0.0, 10.0001)
            if y == 1:
                return (0.0, 12.3)
            return (0.0, 0.0)

    ra_fmt = plotmod._healpix_ra_tick_formatter(DummyRaMap())
    dec_fmt = plotmod._healpix_dec_tick_formatter(DummyDecMap())
    assert ra_fmt(np.nan, 0) == ""
    assert ra_fmt(-1, 0) == ""
    assert ra_fmt(-2, 0) == ""
    assert ra_fmt(0, 0) == "0°"
    assert ra_fmt(1, 0) == "12.3°"
    assert dec_fmt(np.nan, 0) == ""
    assert dec_fmt(-1, 0) == ""
    assert dec_fmt(-2, 0) == ""
    assert dec_fmt(0, 0) == "10°"
    assert dec_fmt(1, 0) == "12.3°"


def test_finish_healpix_cartview_figure_colorbar_layout():
    class DummyText:
        def __init__(self):
            self.removed = False

        def remove(self):
            self.removed = True

    class DummyPos:
        def __init__(self):
            self.width = 0.4
            self.x0 = 0.2
            self.y0 = 0.01
            self.height = 0.05

    class DummyAxisObj:
        def set_major_formatter(self, _fmt):
            self.formatter = _fmt

    class DummyAx:
        def __init__(self):
            self.xaxis = DummyAxisObj()
            self.yaxis = DummyAxisObj()
            self.texts = [DummyText()]
            self._pos = DummyPos()
            self.label = None
            self.pos_set = None

        def set_axis_on(self):
            self.axis_on = True

        def set_xlabel(self, text, labelpad=None):
            self.label = (text, labelpad)

        def set_ylabel(self, text):
            self.ylabel = text

        def grid(self, *args, **kwargs):
            self.grid_set = True

        def get_position(self):
            return self._pos

        def set_position(self, pos):
            self.pos_set = pos

        def get_ylim(self):
            return (0.0, 1.0)

        def get_xlim(self):
            return (0.0, 1.0)

        def get_lonlat(self, x, y):
            return (x, y)

    ax_map = DummyAx()
    cb_ax = DummyAx()
    fig = type("Fig", (), {"axes": [object(), ax_map, cb_ax]})()
    plotmod._finish_healpix_cartview_figure(
        fig,
        ax_map,
        have_cbar=True,
        cbar_label="K",
        cbarshrink=0.5,
        cbar_yoffset=-0.5,
        xlabel_pad=6.0,
    )
    assert ax_map.axis_on
    assert cb_ax.texts[0].removed
    assert cb_ax.label == ("K", 2.0)
    assert cb_ax.pos_set is not None
