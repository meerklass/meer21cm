"""Unit tests for MultipolePowerSpectrum and mesh k_in / column APIs."""

import logging
from unittest.mock import patch

import numpy as np
import pytest

from meer21cm.estimator import FieldPowerSpectrum
from meer21cm.grid import fourier_window_for_assignment
from meer21cm.multipole import MultipolePowerSpectrum
from meer21cm.window import (
    accumulate_mesh_window_matrices,
    build_mesh_window_mas_out,
    build_mesh_window_matrix,
    propose_mesh_k_in,
    tapered_theory_beam_kernels,
)


ELLS = (0, 2, 4)
BOX_LEN = (80.0, 80.0, 80.0)
BOX_NDIM = (16, 16, 16)
_FAR_OBS = np.array([0.0, 0.0, 1.0e5], dtype=float)


def _configure_bins(obj, k_hi=0.22):
    k_nyq = np.asarray(obj.k_nyquist, dtype=float)
    from meer21cm.util import get_nd_slicer

    slicer = get_nd_slicer()
    w = np.ones_like(obj.k_mode, dtype=float)
    for i in range(3):
        w = w * (np.abs(obj.k_vec[i])[slicer[i]] <= 0.5 * k_nyq[i])
    w[0, 0, 0] = 0.0
    obj.k1dweights = w
    obj.k1dbins = np.linspace(0.08, k_hi, 6)
    return obj


def _make_mps(**kwargs):
    defaults = dict(
        field_1=np.ones(BOX_NDIM, dtype="f8"),
        box_len=BOX_LEN,
        los="endpoint",
        los_observer=_FAR_OBS,
        kaiser_rsd=False,
        tracer_bias_1=1.0,
        window="mesh",
        include_beam=[False, False],
        n_k_in=24,
    )
    defaults.update(kwargs)
    mps = MultipolePowerSpectrum(**defaults)
    return _configure_bins(mps)


def test_constructor_defaults():
    mps = MultipolePowerSpectrum(
        field_1=np.ones((8, 8, 8)),
        box_len=(40.0, 40.0, 40.0),
    )
    assert mps.los == "endpoint"
    assert list(mps.include_beam) == [False, False]
    assert list(mps.include_sky_sampling) == [False, False]
    assert list(mps.compensate) == [False, False]
    assert mps.window_kind == "mesh"


def test_window_matrix_warns_before_run(caplog):
    mps = _make_mps()
    with caplog.at_level(logging.WARNING, logger="meer21cm.multipole"):
        got = mps.window_matrix
    assert got is None
    assert "run_window_matrix" in caplog.text
    assert mps._window_matrix_obj is None


def test_bias_update_does_not_rebuild_window():
    mps = _make_mps()
    mps.run_window_matrix()
    w0 = mps._window_matrix_obj
    assert w0 is not None
    p0 = mps.model_multipoles["P_ell"][0].copy()
    checksum = float(np.sum(w0.matrix))
    mps.tracer_bias_1 = 1.7
    p1 = mps.model_multipoles["P_ell"][0]
    assert mps._window_matrix_obj is w0
    assert float(np.sum(w0.matrix)) == pytest.approx(checksum)
    assert not np.allclose(p0, p1)
    # P0 scales as b^2 for no-RSD isotropic theory
    assert np.nanmedian(p1 / np.where(p0 == 0, np.nan, p0)) == pytest.approx(
        1.7**2, rel=0.15
    )


def test_beam_setting_marks_window_stale(caplog):
    mps = _make_mps()
    mps.run_window_matrix()
    assert mps._window_stale is False
    mps.beam_n_mu = 8
    assert mps._window_stale is True
    with caplog.at_level(logging.WARNING, logger="meer21cm.multipole"):
        _ = mps.model_multipoles
    assert "stale" in caplog.text.lower()
    assert mps._window_matrix_obj is not None


def test_column_accumulate_matches_serial():
    mps = _make_mps()
    serial = mps._fill_window_columns()
    n = serial.matrix.shape[1]
    mid = n // 2
    a = mps._fill_window_columns(columns=list(range(0, mid)))
    b = mps._fill_window_columns(columns=list(range(mid, n)))
    got = accumulate_mesh_window_matrices([a, b])
    assert np.allclose(got.matrix, serial.matrix, rtol=1e-10, atol=1e-12)


def test_propose_mesh_k_in_spans_grid():
    mps = _make_mps()
    k_in = propose_mesh_k_in(mps, n=24)
    k_max = float(np.max(mps.k_mode))
    assert k_in[-1] >= k_max


def test_truncated_k_in_warns():
    weights = np.ones(BOX_NDIM, dtype=float)
    fps = FieldPowerSpectrum(
        np.ones(BOX_NDIM),
        BOX_LEN,
        weights_1=weights,
        los="endpoint",
        los_observer=_FAR_OBS,
        _skip_specification=True,
    )
    fps.k1dbins = np.linspace(0.08, 0.2, 5)
    fps.k1dweights = np.ones_like(fps.k_mode)
    k_in = np.geomspace(0.05, 0.12, 10)
    with pytest.warns(UserWarning, match="does not span"):
        build_mesh_window_matrix(fps, k_in, weights=weights, ells=(0, 2))


def test_from_power_spectrum_copies_lightcone_state():
    mps = _make_mps()
    mps._pix_coor_in_cartesian = np.ones((12, 3), dtype=float)
    mps.has_resol = True
    mps.sigma_beam_ch = 0.4
    got = MultipolePowerSpectrum.from_power_spectrum(
        mps, window="mesh", kaiser_rsd=False
    )
    assert np.allclose(got._pix_coor_in_cartesian, mps._pix_coor_in_cartesian)
    assert got.has_resol is True
    assert float(np.mean(got.sigma_beam_ch)) == pytest.approx(0.4)
    assert got.kaiser_rsd is False


def test_far_observer_mesh_monopole_matches_get_1d_power():
    mps = _make_mps(kaiser_rsd=False, tracer_bias_1=1.2)
    mps.run_window_matrix()
    model = mps.model_multipoles
    p3d = mps.power_kmu("auto_1", include_mean_amp=True)
    p0_1d, _, _ = mps.get_1d_power(
        p3d, k1dbins=mps.k1dbins, multipole_ell=0, k1dweights=mps.k1dweights
    )
    p0_w = np.asarray(model["P_ell"][0], dtype=float)
    rel = np.abs(p0_w / np.where(np.abs(p0_1d) > 0, p0_1d, np.nan) - 1.0)
    assert np.nanmedian(rel) < 0.15


def _cic_blob(shape):
    """Compact CIC-like mass (linear tent around the box centre)."""
    nx, ny, nz = (int(s) for s in shape)
    x = np.arange(nx) - (nx - 1) / 2.0
    y = np.arange(ny) - (ny - 1) / 2.0
    z = np.arange(nz) - (nz - 1) / 2.0
    wx = np.clip(1.0 - np.abs(x) / max(nx / 4.0, 1.0), 0.0, 1.0)
    wy = np.clip(1.0 - np.abs(y) / max(ny / 3.0, 1.0), 0.0, 1.0)
    wz = np.clip(1.0 - np.abs(z) / max(nz / 3.0, 1.0), 0.0, 1.0)
    return (wx[:, None, None] * wy[None, :, None] * wz[None, None, :]).astype(float)


def _paint_ngp_cic(shape, positions):
    """NGP and periodic CIC of the same off-grid particle positions (grid units)."""
    nx, ny, nz = (int(s) for s in shape)
    ngp = np.zeros((nx, ny, nz), dtype=float)
    cic = np.zeros((nx, ny, nz), dtype=float)
    dims = np.array([nx, ny, nz], dtype=int)
    for p in positions:
        p = np.asarray(p, dtype=float)
        nijk = np.mod(np.rint(p).astype(int), dims)
        ngp[nijk[0], nijk[1], nijk[2]] += 1.0
        i0 = np.floor(p).astype(int)
        f = p - np.floor(p)
        for dx in (0, 1):
            wx = (1.0 - f[0]) if dx == 0 else f[0]
            ix = (i0[0] + dx) % nx
            for dy in (0, 1):
                wy = (1.0 - f[1]) if dy == 0 else f[1]
                iy = (i0[1] + dy) % ny
                for dz in (0, 1):
                    wz = (1.0 - f[2]) if dz == 0 else f[2]
                    iz = (i0[2] + dz) % nz
                    cic[ix, iy, iz] += wx * wy * wz
    return ngp, cic


def _half_cell_positions(shape, axis):
    """Off-grid particles filling the central half of the box (CIC ≠ NGP)."""
    nx, ny, nz = (int(s) for s in shape)
    pos = []
    ranges = []
    for n in (nx, ny, nz):
        lo = max(1, n // 4)
        hi = max(lo + 1, (3 * n) // 4)
        ranges.append(range(lo, hi))
    for i in ranges[0]:
        for j in ranges[1]:
            for k in ranges[2]:
                pos.append((i + 0.5, j + 0.5, k + 0.5))
    return pos


def _bh_cube(shape, axis):
    from scipy.signal.windows import blackmanharris

    t = np.ones(shape, dtype=float)
    w = np.asarray(blackmanharris(int(shape[axis])), dtype=float)
    slicer = [None, None, None]
    slicer[int(axis)] = slice(None)
    return t * w[tuple(slicer)]


def _falling_p0(k_in):
    k = np.asarray(k_in, dtype=float)
    return 1.0 / (1.0 + (k / 0.08) ** 2)


def _taper_operator_p0(box_ndim, box_len, axis, *, apply_taper=True):
    """Inner-mode (T×CIC, MAS in mode_scale) vs MAS-out × (T×NGP)."""
    ngp, cic = _paint_ngp_cic(box_ndim, _half_cell_positions(box_ndim, axis))
    if apply_taper:
        taper = _bh_cube(box_ndim, axis)
        w_cic = cic * taper
        w_ngp = ngp * taper
    else:
        w_cic = cic
        w_ngp = ngp
    fps = FieldPowerSpectrum(
        np.ones(box_ndim, dtype="f8"),
        box_len,
        weights_1=w_cic,
        los="endpoint",
        los_observer=_FAR_OBS,
        _skip_specification=True,
    )
    fps.grid_scheme = "cic"
    fps = _configure_bins(fps, k_hi=0.22)
    k_in = propose_mesh_k_in(fps, n=16)
    w_mas2 = fourier_window_for_assignment(fps.box_ndim, "cic") ** 2
    inner = build_mesh_window_matrix(
        fps, k_in, weights=w_cic, ells=(0,), mode_scale=w_mas2, renorm_weights=w_cic
    )
    wrong = build_mesh_window_mas_out(
        fps,
        k_in,
        renorm_weights=w_cic,
        ells=(0,),
        raw_comb=w_ngp,
        beam_at_theory_mode=False,
    )
    p0 = _falling_p0(inner.k_in)
    p_in = np.asarray(inner.apply({0: p0})[0], dtype=float)
    p_w = np.asarray(wrong.apply({0: p0})[0], dtype=float)
    k = np.asarray(inner.k_out, dtype=float)
    return k, p_in, p_w


def _p0_median_rel(p_a, p_b):
    ratio = np.asarray(p_a, float) / np.where(
        np.abs(p_b) > 0, np.asarray(p_b, float), np.nan
    )
    return float(np.nanmedian(np.abs(ratio - 1.0)))


def test_post_deposit_taper_operators_disagree_on_short_axis():
    """BH on Nx=8 varies on the CIC scale: inner-mode ≠ MAS-out × (T×NGP)."""
    k, p_in, p_w = _taper_operator_p0((8, 16, 16), (40.0, 80.0, 80.0), axis=0)
    med = _p0_median_rel(p_in, p_w)
    k_peak_in = float(k[int(np.nanargmax(k * p_in))])
    k_peak_w = float(k[int(np.nanargmax(k * p_w))])
    assert med >= 0.20 or not np.isclose(k_peak_in, k_peak_w, rtol=0.05, atol=0.005), (
        f"operators agree on short axis (median |ratio−1|={med:.3f}, "
        f"kP0 peak {k_peak_in:.4f} vs {k_peak_w:.4f})"
    )


def test_inner_mode_taper_matches_estimator_weights():
    """MPS inner-mode P0 matches build_mesh_window_matrix on the same T×CIC."""
    box_ndim = (8, 16, 16)
    box_len = (40.0, 80.0, 80.0)
    blob = _cic_blob(box_ndim)
    for axes in ((0,), (0, 1, 2)):
        mps = MultipolePowerSpectrum(
            field_1=np.ones(box_ndim, dtype="f8"),
            box_len=box_len,
            weights_grid_1=blob.copy(),
            los="endpoint",
            los_observer=_FAR_OBS,
            kaiser_rsd=False,
            tracer_bias_1=1.0,
            window="mesh",
            window_ells=(0,),
            window_taper_axes=axes,
            grid_scheme="cic",
            include_beam=[False, False],
            n_k_in=16,
        )
        mps.weights_1 = blob.copy()
        mps.apply_taper_to_field(1, axis=list(axes))
        mps = _configure_bins(mps, k_hi=0.22)
        mps.run_window_matrix()
        p_mps = np.asarray(mps.model_multipoles["P_ell"][0], dtype=float)
        w = np.asarray(mps.weights_1, dtype=float)
        w_mas2 = fourier_window_for_assignment(mps.box_ndim, "cic") ** 2
        k_in = mps._resolve_k_in()
        mat = build_mesh_window_matrix(
            mps, k_in, weights=w, ells=(0,), mode_scale=w_mas2, renorm_weights=w
        )
        theory0 = mps.get_theory_multipoles_kmu(mat.k_in, ells=(0,), nmu=16)["P_ell"][0]
        p_ref = np.asarray(mat.apply({0: theory0})[0], dtype=float)
        nmodes = np.asarray(mat.nmodes, dtype=float)
        good = nmodes > 4
        rel = np.abs(p_mps / np.where(np.abs(p_ref) > 0, p_ref, np.nan) - 1.0)
        assert (
            np.nanmedian(rel[good]) < 0.05
        ), f"axes={axes} median rel={np.nanmedian(rel[good]):.4f}"


def test_post_deposit_taper_operators_closer_on_long_axis():
    """Slow T (Nz=64 BH on z): extra MAS-out×T error vs inner-mode is small.

    Untilted NGP vs CIC already differs; the *additional* error from
    multiplying T is the commutation term.  On Nx=8 that extra is large;
    on Nz=64 T is slow vs CIC and the extra is much smaller.  Documents
    the slow-T limit (why z-taper looked fine); not a pass/fail on 07.
    """
    _, sT_in, sT_w = _taper_operator_p0((8, 16, 16), (40.0, 80.0, 80.0), 0)
    _, s0_in, s0_w = _taper_operator_p0(
        (8, 16, 16), (40.0, 80.0, 80.0), 0, apply_taper=False
    )
    _, lT_in, lT_w = _taper_operator_p0((16, 16, 64), (80.0, 80.0, 320.0), 2)
    _, l0_in, l0_w = _taper_operator_p0(
        (16, 16, 64), (80.0, 80.0, 320.0), 2, apply_taper=False
    )
    extra_short = _p0_median_rel(sT_in, sT_w) - _p0_median_rel(s0_in, s0_w)
    extra_long = _p0_median_rel(lT_in, lT_w) - _p0_median_rel(l0_in, l0_w)
    assert extra_long < 0.5 * extra_short, (
        f"long-axis extra |ratio−1| from T={extra_long:.3f} not << "
        f"short-axis {extra_short:.3f} (slow-T limit)"
    )


def test_model_power_3d_bins_match_window_apply():
    """Bin of ``accumulate_mesh_window_cubes`` matches ``W @ P0`` (same weights)."""
    mps = _make_mps(n_k_in=16)
    mat = mps.run_window_matrix()
    p_w = mps.model_multipoles
    mps.run_model_power_3d()
    p_c = mps._bin_model_power_3d()
    nmodes = np.asarray(mat.nmodes, dtype=float)
    good = nmodes > 4
    scale = float(np.nanmax(np.abs(p_w["P_ell"][0])))
    for ell in ELLS:
        a = np.asarray(p_w["P_ell"][ell], dtype=float)
        b = np.asarray(p_c["P_ell"][ell], dtype=float)
        assert np.allclose(
            a[good], b[good], rtol=1e-10, atol=1e-10 * max(scale, 1.0)
        ), f"ell={ell}"


def test_taper_plus_beam_uses_cic_then_t_kernels():
    """Taper + ``sigma_beam_ch``: CIC-then-T (oms=1, ``W_MAS^2`` on theory), not T×NGP."""
    box_ndim = (8, 8, 8)
    box_len = (40.0, 40.0, 40.0)
    rng = np.random.default_rng(1)
    pix = rng.uniform(0.0, 40.0, size=(24, 3))
    mps = MultipolePowerSpectrum(
        field_1=np.ones(box_ndim, dtype="f8"),
        box_len=box_len,
        los="endpoint",
        los_observer=_FAR_OBS,
        kaiser_rsd=False,
        tracer_bias_1=1.0,
        window="mesh",
        window_ells=(0, 2),
        window_taper_axes=(2,),
        grid_scheme="cic",
        include_beam=[False, False],
        n_k_in=8,
        beam_at_theory_mode=True,
        beam_n_mu=4,
        beam_diag_as_ratio=False,
        nu=np.linspace(900.0e6, 1050.0e6, 8),
        sigma_beam_ch=0.4,
    )
    mps._pix_coor_in_cartesian = np.asarray(pix, dtype=float)
    mps.box_origin = np.zeros(3, dtype=float)
    mps.weights_1 = np.ones(box_ndim, dtype=float)
    mps.apply_taper_to_field(1, axis=[2])
    mps = _configure_bins(mps, k_hi=0.22)
    assert mps._use_tapered_beam()
    assert not mps._use_mas_out()
    n_cell = int(pix.shape[0])
    with patch(
        "meer21cm.multipole_ops.beam_edge_cell_mass",
        return_value=np.ones(n_cell),
    ):
        kw = mps._mesh_operator_kwargs()
        kw_ngp = tapered_theory_beam_kernels(
            mps,
            mps._resolve_k_in(),
            renorm_weights=mps.weights_1,
            mas="ngp",
            taper_axes=(2,),
            ells=(0, 2),
            n_mu=4,
            n_phi=1,
            beam_diag_as_ratio=False,
        )
    oms = np.asarray(kw["out_mode_scale"], dtype=float)
    assert np.allclose(oms, 1.0)
    w_mas2 = fourier_window_for_assignment(mps.box_ndim, "cic") ** 2
    ms = np.asarray(kw["mode_scale"], dtype=float)
    ms_ngp = np.asarray(kw_ngp["mode_scale"], dtype=float)
    assert np.allclose(ms, ms_ngp * w_mas2)
    oms_ngp = np.asarray(kw_ngp["out_mode_scale"], dtype=float)
    assert np.allclose(oms_ngp, w_mas2)
    assert not np.allclose(oms_ngp, 1.0)
