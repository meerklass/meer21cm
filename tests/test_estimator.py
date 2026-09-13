"""Tests for FieldPowerSpectrum (estimator)."""

import numpy as np
import pytest

from meer21cm.estimator import FieldPowerSpectrum
from meer21cm.power_ops import get_fourier_density


def _smooth_gaussian_field(ndim, box_len, seed=42, k_smooth=0.12):
    """Isotropic Gaussian random field (no RSD)."""
    rng = np.random.default_rng(seed)
    ndim = tuple(int(n) for n in ndim)
    box_len = np.asarray(box_len, dtype=float)
    white = rng.normal(size=ndim)
    kx = 2 * np.pi * np.fft.fftfreq(ndim[0], d=box_len[0] / ndim[0])
    ky = 2 * np.pi * np.fft.fftfreq(ndim[1], d=box_len[1] / ndim[1])
    kz = 2 * np.pi * np.fft.fftfreq(ndim[2], d=box_len[2] / ndim[2])
    k2 = sum(np.meshgrid(kx**2, ky**2, kz**2, indexing="ij"))
    filt = np.exp(-0.5 * k2 / k_smooth**2)
    return np.fft.ifftn(np.fft.fftn(white) * filt).real


def _median_rel_diff(a, b):
    a_np = np.asarray(a, dtype=float)
    b_np = np.asarray(b, dtype=float)
    mask = np.isfinite(a_np) & np.isfinite(b_np) & (np.abs(b_np) > 0)
    if mask.sum() == 0:
        return np.nan
    return float(np.median(np.abs(a_np[mask] / b_np[mask] - 1.0)))


def test_k_para_reserved_and_unhandled_los():
    field = np.ones((8, 8, 8))
    box_len = np.array([80.0, 80.0, 80.0])

    fps_g = FieldPowerSpectrum(field, box_len, los="global", _skip_specification=True)
    k_para_g = np.asarray(fps_g.k_para)

    for local in ("endpoint", "firstpoint"):
        fps = FieldPowerSpectrum(
            field,
            box_len,
            los=local,
            los_observer=(0.0, 0.0, 1.0e4),
            _skip_specification=True,
        )
        np.testing.assert_allclose(fps.k_para, k_para_g)
        mu = fps.mu_mode
        assert np.all(np.isfinite(mu))
        assert mu.shape == fps.k_mode.shape

    fps_m = FieldPowerSpectrum(field, box_len, los="midpoint", _skip_specification=True)
    with pytest.raises(NotImplementedError, match="k_para"):
        _ = fps_m.k_para
    with pytest.raises(NotImplementedError, match="mu_mode"):
        _ = fps_m.mu_mode

    fps = FieldPowerSpectrum(field, box_len, los="global", _skip_specification=True)
    fps.los = "bogus"  # bypass constructor validation
    with pytest.raises(ValueError, match="Unknown los"):
        _ = fps.k_para
    with pytest.raises(ValueError, match="Unknown los"):
        _ = fps.mu_mode


def test_multipole_bin_index_map_requires_k1dbins():
    field = np.ones((8, 8, 8))
    box_len = np.array([80.0, 80.0, 80.0])
    fps = FieldPowerSpectrum(field, box_len, los="global", _skip_specification=True)
    assert getattr(fps, "k1dbins", None) is None

    # multipole_bin_index_map: missing / invalid k1dbins, bad k1dweights
    with pytest.raises(ValueError, match="k1dbins is required"):
        fps.multipole_bin_index_map()
    with pytest.raises(ValueError, match="bin edges"):
        fps.multipole_bin_index_map(k1dbins=np.array([0.1]))
    with pytest.raises(ValueError, match="bin edges"):
        fps.multipole_bin_index_map(k1dbins=np.array([[0.1, 0.2], [0.2, 0.3]]))
    k1dbins = np.linspace(0.1, 0.4, 4)
    with pytest.raises(ValueError, match="k1dweights shape"):
        fps.multipole_bin_index_map(k1dbins=k1dbins, k1dweights=np.ones(3))

    # measure_multipoles: missing k1dbins, bad which, no field_2 for auto_2/cross
    with pytest.raises(ValueError, match="k1dbins is required"):
        fps.measure_multipoles()
    with pytest.raises(ValueError, match="which must be"):
        fps.measure_multipoles(which="auto_3", k1dbins=k1dbins)
    with pytest.raises(ValueError, match="field_2 is None"):
        fps.measure_multipoles(which="auto_2", k1dbins=k1dbins)
    with pytest.raises(ValueError, match="field_2 is None"):
        fps.measure_multipoles(which="cross", k1dbins=k1dbins)

    fps_m = FieldPowerSpectrum(field, box_len, los="midpoint", _skip_specification=True)
    with pytest.raises(NotImplementedError, match="measure_multipoles"):
        fps_m.measure_multipoles(k1dbins=k1dbins)
    fps.los = "bogus"
    with pytest.raises(ValueError, match="Unhandled los"):
        fps.measure_multipoles(k1dbins=k1dbins)


def test_local_los_requires_observer():
    field = np.ones((8, 8, 8))
    box_len = np.array([80.0, 80.0, 80.0])
    k1dbins = np.linspace(0.1, 0.4, 4)
    fps = FieldPowerSpectrum(field, box_len, los="endpoint", _skip_specification=True)
    with pytest.raises(ValueError, match="los_observer"):
        fps.measure_multipoles(k1dbins=k1dbins, ells=(2,))
    with pytest.raises(ValueError, match="los_observer"):
        _ = fps.mu_mode
    with pytest.raises(ValueError, match="los_observer"):
        fps.multipole_bin_index_map(k1dbins=k1dbins)


def test_yamamoto_monopole_matches_auto_power_3d():
    field = _smooth_gaussian_field((16, 16, 16), (80.0, 80.0, 80.0))
    box_len = np.array([80.0, 80.0, 80.0])
    fps = FieldPowerSpectrum(
        field,
        box_len,
        los="endpoint",
        los_observer=(0.0, 0.0, 1.0e4),
        _skip_specification=True,
    )
    p0 = fps.multipole_power_3d(0, which="auto_1")
    np.testing.assert_allclose(p0, fps.auto_power_3d_1)


def test_yamamoto_firstpoint_equals_endpoint_auto():
    field = _smooth_gaussian_field((16, 16, 16), (80.0, 80.0, 80.0), seed=7)
    box_len = np.array([80.0, 80.0, 80.0])
    k1dbins = np.linspace(0.1, 0.4, 5)
    obs = (0.0, 0.0, 5.0e3)
    meas = {}
    for los in ("firstpoint", "endpoint"):
        fps = FieldPowerSpectrum(
            field,
            box_len,
            los=los,
            los_observer=obs,
            _skip_specification=True,
        )
        meas[los] = fps.measure_multipoles(k1dbins=k1dbins, ells=(0, 1, 2, 3, 4))
    for ell in (0, 2, 4):
        np.testing.assert_allclose(
            meas["firstpoint"].P_ell[ell], meas["endpoint"].P_ell[ell]
        )
    for ell in (1, 3):
        np.testing.assert_allclose(
            meas["firstpoint"].P_ell[ell],
            -meas["endpoint"].P_ell[ell],
            atol=1e-10 * np.nanmax(np.abs(meas["firstpoint"].P_ell[ell])) + 1e-20,
        )


def test_yamamoto_cross_firstpoint_differs_endpoint():
    """Cross firstpoint ≠ endpoint (Ylm on tracer 1 vs 2). ℓ=0 matches Re(F1 F2*)."""
    from meer21cm.power_ops import bin_3d_to_1d

    ndim = (16, 16, 16)
    box_len = np.array([80.0, 80.0, 80.0])
    field_1 = _smooth_gaussian_field(ndim, box_len, seed=3)
    field_2 = _smooth_gaussian_field(ndim, box_len, seed=9)
    k1dbins = np.linspace(0.1, 0.4, 5)
    obs = (0.0, 0.0, 5.0e3)
    meas = {}
    fps_ref = None
    for los in ("firstpoint", "endpoint"):
        fps = FieldPowerSpectrum(
            field_1,
            box_len,
            field_2=field_2,
            los=los,
            los_observer=obs,
            _skip_specification=True,
        )
        meas[los] = fps.measure_multipoles(
            which="cross", k1dbins=k1dbins, ells=(0, 1, 2, 3, 4)
        )
        fps_ref = fps
    p0_3d, _, _ = bin_3d_to_1d(fps_ref.cross_power_3d, fps_ref.k_mode, k1dbins)
    np.testing.assert_allclose(meas["firstpoint"].P_ell[0], p0_3d)
    np.testing.assert_allclose(meas["endpoint"].P_ell[0], p0_3d)
    # Distinct fields → even ℓ>0 firstpoint ≠ endpoint.
    rel_p2 = _median_rel_diff(meas["firstpoint"].P_ell[2], meas["endpoint"].P_ell[2])
    assert rel_p2 > 1e-3
    rel_p1 = _median_rel_diff(meas["firstpoint"].P_ell[1], meas["endpoint"].P_ell[1])
    assert rel_p1 > 1e-3 or np.nanmax(np.abs(meas["firstpoint"].P_ell[1])) == 0.0


def test_yamamoto_far_observer_matches_global():
    ndim = (24, 24, 24)
    box_len = np.array([120.0, 120.0, 120.0])
    field = _smooth_gaussian_field(ndim, box_len, seed=11)
    k1dbins = np.linspace(0.08, 0.35, 6)
    fps_g = FieldPowerSpectrum(field, box_len, los="global", _skip_specification=True)
    meas_g = fps_g.measure_multipoles(k1dbins=k1dbins, ells=(0, 2, 4))
    R = 1.0e5 * float(np.max(box_len))
    # Centre the box on the far-z axis so n̂ → z-hat uniformly.
    obs = (-0.5 * box_len[0], -0.5 * box_len[1], R)
    fps_e = FieldPowerSpectrum(
        field,
        box_len,
        los="endpoint",
        los_observer=obs,
        _skip_specification=True,
    )
    meas_e = fps_e.measure_multipoles(k1dbins=k1dbins, ells=(0, 1, 2, 3, 4))
    for ell in (0, 2, 4):
        rel = _median_rel_diff(meas_g.P_ell[ell], meas_e.P_ell[ell])
        assert rel < 1e-3, f"ell={ell} median rel diff {rel}"
    # Far observer: odd wide-angle ≈ 0 vs even amplitude.
    even_scale = max(
        float(np.nanmax(np.abs(meas_e.P_ell[0]))),
        float(np.nanmax(np.abs(meas_e.P_ell[2]))),
        1e-30,
    )
    for ell in (1, 3):
        odd_max = float(np.nanmax(np.abs(meas_e.P_ell[ell])))
        assert odd_max < 1e-3 * even_scale, f"ell={ell} odd/even {odd_max / even_scale}"


def test_no_rsd_discrete_mu_yamamoto_matches_global():
    """Isotropic Gaussian field: P2/P4 from discrete μ only; far Yamamoto ≈ global."""
    ndim = (24, 24, 24)
    box_len = np.array([120.0, 120.0, 120.0])
    field = _smooth_gaussian_field(ndim, box_len, seed=21)
    k1dbins = np.linspace(0.08, 0.35, 6)
    fps_g = FieldPowerSpectrum(field, box_len, los="global", _skip_specification=True)
    meas_g = fps_g.measure_multipoles(k1dbins=k1dbins, ells=(0, 2, 4))
    assert np.nanmax(np.abs(meas_g.P_ell[2])) > 0
    R = 1.0e5 * float(np.max(box_len))
    obs = (-0.5 * box_len[0], -0.5 * box_len[1], R)
    fps_f = FieldPowerSpectrum(
        field,
        box_len,
        los="firstpoint",
        los_observer=obs,
        _skip_specification=True,
    )
    meas_f = fps_f.measure_multipoles(k1dbins=k1dbins, ells=(0, 2, 4))
    for ell in (0, 2, 4):
        rel = _median_rel_diff(meas_g.P_ell[ell], meas_f.P_ell[ell])
        assert rel < 1e-3, f"ell={ell} median rel diff {rel}"
def test_fourier_field_uses_frozen_field_mean():
    field = np.arange(27, dtype=float).reshape(3, 3, 3) + 1.0
    frozen = 2.0
    box_len = np.array([3.0, 3.0, 3.0])
    fps = FieldPowerSpectrum(
        field,
        box_len,
        field_2=2.0 * field,
        mean_center_1=True,
        unitless_1=True,
        mean_center_2=True,
        unitless_2=True,
    )
    fps.field_mean_1 = frozen
    fps.field_mean_2 = 2.0 * frozen
    expected_1 = get_fourier_density(
        field, mean_center=True, unitless=True, field_mean=frozen
    )
    expected_2 = get_fourier_density(
        2.0 * field, mean_center=True, unitless=True, field_mean=2.0 * frozen
    )
    default_1 = get_fourier_density(field, mean_center=True, unitless=True)
    np.testing.assert_allclose(fps.fourier_field_1, expected_1)
    np.testing.assert_allclose(fps.fourier_field_2, expected_2)
    assert not np.allclose(fps.fourier_field_1, default_1)
