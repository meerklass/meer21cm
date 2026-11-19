"""Tests for the one-dimensional line-of-sight power spectrum."""

import numpy as np
import pytest

from meer21cm import LineOfSightPowerSpectrum, MockSimulation, PowerSpectrum
from meer21cm.los import (
    LineOfSightMeasurement,
    _weighted_mean_and_error,
    azimuthal_pixel_window_power,
    beam_window_power,
    estimate_1d_power,
    hann_window,
    pixel_window_power,
    resample_sightlines,
    uniform_comoving_grid,
)
from meer21cm.power_ops import bin_3d_to_1d, get_power_spectrum
from meer21cm.transfer import RadialTransferFunction
from meer21cm.util import create_wcs, f_21

KAISER_NSEED = 3


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _kaiser_mock(seed=1):
    """Small LPP-Kaiser mock on a HEALPix patch (cheap enough for unit tests)."""
    mock = MockSimulation(
        nu=np.linspace(789.1e6, 887.8e6, 64),
        hp_nside=64,
        ra_range=(10, 40),
        dec_range=(-15, 15),
        density="lognormal",
        kaiser_rsd=True,
        rsd_from_field=True,
        parallel_plane=False,
        tracer_bias_1=1.0,
        sigma_v_1=0.0,
        mean_amp_1=1.0,
        seed=seed,
        precision=False,
        model_k_from_field=True,
        include_beam=[False, False],
        include_sky_sampling=[False, False],
        compensate=[False, False],
        downres_factor_transverse=0.25,
        downres_factor_radial=0.43,
        batch_number=8,
    )
    mock.get_enclosing_box()
    return mock


def _measured_power_func(mock):
    """``power_kmu_func`` built from the realisation's own real-space power."""
    field = np.asarray(mock.mock_matter_field_r)
    p3d = get_power_spectrum(np.fft.rfftn(field, norm="forward"), mock.box_len)
    p1d, keff, nmodes = bin_3d_to_1d(p3d, mock.kmode, np.geomspace(5e-3, 2.0, 80))
    ok = np.isfinite(p1d) & (nmodes > 0)
    f = float(mock.f_growth_true)
    bias = float(mock.tracer_bias_1)

    def power_kmu(k, mu):
        return np.interp(k, keff[ok], p1d[ok]) * (bias + f * mu**2) ** 2

    return power_kmu


# ---------------------------------------------------------------------------
# low-level helpers
# ---------------------------------------------------------------------------
def test_hann_window():
    w = hann_window(64)
    assert w.shape == (64,)
    assert np.isclose(np.mean(w**2), 3.0 / 8.0, rtol=1e-12)
    # periodic: first sample is zero, and the window is symmetric
    assert np.isclose(w[0], 0.0)
    assert np.allclose(w[1:], w[1:][::-1])


def test_uniform_comoving_grid():
    chi = np.linspace(2272.0, 2868.0, 128)
    chi_u, dchi_u = uniform_comoving_grid(chi, over=2)
    assert np.isclose(dchi_u, np.median(np.diff(chi)) / 2.0)
    assert np.allclose(np.diff(chi_u), dchi_u)
    assert chi_u[0] >= chi[0] and chi_u[-1] <= chi[-1]


def test_resample_sightlines_identity():
    chi = np.linspace(0.0, 10.0, 11)
    cube = np.vstack([chi, 2.0 * chi])
    out = resample_sightlines(cube, chi, chi)
    assert np.allclose(out, cube)


def test_pixel_window_power():
    k = np.array([0.0, 0.05, 0.1])
    assert np.allclose(pixel_window_power(0.0, 0.0, 20.0), 1.0)
    w = pixel_window_power(k, np.zeros_like(k), 20.0)
    assert np.all(w <= 1.0) and np.all(np.diff(w) < 0.0)
    expected = np.sinc(k * 20.0 / 2.0 / np.pi) ** 2
    assert np.allclose(w, expected)


def test_beam_window_power():
    assert np.isclose(beam_window_power(0.0, 15.0), 1.0)
    k = np.array([0.05, 0.1])
    assert np.allclose(beam_window_power(k, 15.0), np.exp(-((k * 15.0) ** 2)))


# ---------------------------------------------------------------------------
# estimator
# ---------------------------------------------------------------------------
def test_estimator_white_noise():
    rng = np.random.default_rng(0)
    sigma, dchi = 2.0, 1.5
    n, nchan = 4000, 128
    data = rng.normal(scale=sigma, size=(n, nchan))
    knyq = np.pi / dchi
    edges = np.linspace(2 * 2 * np.pi / (nchan * dchi), knyq, 21)
    meas = estimate_1d_power(data, dchi, edges, which="auto_1")
    expected = sigma**2 * dchi
    assert np.median(meas.power1d) == pytest.approx(expected, rel=0.05)
    assert meas.n_sightlines == n


def test_estimator_single_mode():
    n, m = 128, 20
    dchi = 1.0
    j = np.arange(n)
    data = np.tile(np.cos(2.0 * np.pi * m * j / n), (4, 1))
    edges = np.linspace(0.5, 1.5, 21)
    meas = estimate_1d_power(data, dchi, edges)
    k_mode = 2.0 * np.pi * m / (n * dchi)
    assert np.argmax(meas.power1d) == np.digitize(k_mode, edges) - 1
    # the power is concentrated in a few bins around the mode
    assert meas.power1d.max() > 10.0 * np.median(meas.power1d)


def test_estimator_cross_matches_auto():
    rng = np.random.default_rng(1)
    data = rng.normal(size=(200, 64))
    edges = np.linspace(0.05, 2.5, 13)
    auto1 = estimate_1d_power(data, 1.0, edges)
    auto2 = estimate_1d_power(data, 1.0, edges, data_2=data, which="auto_2")
    cross = estimate_1d_power(data, 1.0, edges, data_2=data, which="cross")
    assert np.allclose(auto1.power1d, auto2.power1d, rtol=1e-12)
    assert np.allclose(auto1.power1d, cross.power1d, rtol=1e-12)


def test_estimator_requires_data_2():
    data = np.zeros((10, 32))
    edges = np.linspace(0.05, 2.0, 6)
    with pytest.raises(ValueError, match="data_2 is required"):
        estimate_1d_power(data, 1.0, edges, which="cross")
    with pytest.raises(ValueError, match="data_2 is required"):
        estimate_1d_power(data, 1.0, edges, which="auto_2")


def test_measurement_types():
    rng = np.random.default_rng(2)
    meas = estimate_1d_power(rng.normal(size=(20, 64)), 1.0, np.linspace(0.1, 3.0, 8))
    assert isinstance(meas, LineOfSightMeasurement)
    assert meas.power1d.shape == (7,)
    assert meas.k1deff.shape == (7,)
    assert meas.k1dbins.shape == (8,)
    assert meas.nmodes.shape == (7,) and meas.nmodes.dtype == np.int64
    assert meas.power1d.dtype == np.float64


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
def test_azimuthal_pixel_window_power():
    kperp = np.array([0.0, 0.02, 0.05])
    w = azimuthal_pixel_window_power(kperp, 20.0, n_phi=64)
    assert w[0] == pytest.approx(1.0)
    assert np.all(w <= 1.0)
    assert np.all(np.diff(w) < 0.0)
    # at small k_perp the orientation average is close to the aligned square
    aligned = np.sinc(kperp * 20.0 / 2.0 / np.pi) ** 2 * np.ones_like(kperp)
    assert np.allclose(w, aligned, rtol=1e-3)


def test_model_matches_projection():
    """With all windows off and P=1, the k_perp integral is analytic."""
    kperp_edges = np.linspace(0.0, 0.5, 51)
    los = LineOfSightPowerSpectrum(
        nu=np.linspace(789.1e6, 887.8e6, 16),
        hp_nside=8,
        kperp_edges=kperp_edges,
        include_pixel_window=False,
        include_beam_window=False,
        power_kmu_func=lambda k, mu: np.ones_like(k),
    )
    model = los.model_1d_power(kpar=np.array([0.01, 0.02]))
    expected = kperp_edges[-1] ** 2 / (4.0 * np.pi)
    assert np.allclose(model, expected)


def test_model_uses_stored_mesh():
    los = LineOfSightPowerSpectrum(
        nu=np.linspace(789.1e6, 887.8e6, 16),
        hp_nside=8,
        kperp_edges=np.linspace(0.0, 0.3, 33),
        include_pixel_window=False,
        include_beam_window=False,
    )
    kpar = np.array([0.02, 0.05, 0.1])
    los.model_1d_power(kpar=kpar)
    assert los.kmode.shape == (32, 3)
    assert los.mumode.shape == (32, 3)
    expected_mu = kpar[None, :] / np.sqrt(los.kperp[:, None] ** 2 + kpar[None, :] ** 2)
    assert np.allclose(los.mumode, expected_mu)
    assert np.allclose(los.kmode, np.sqrt(los.kperp[:, None] ** 2 + kpar[None, :] ** 2))


def test_taper_option():
    rng = np.random.default_rng(3)
    sigma, dchi = 2.0, 1.5
    n, nchan = 4000, 128
    data = rng.normal(scale=sigma, size=(n, nchan))
    edges = np.linspace(2 * 2 * np.pi / (nchan * dchi), np.pi / dchi, 21)
    t_on = estimate_1d_power(data, dchi, edges, taper=True)
    t_off = estimate_1d_power(data, dchi, edges, taper=False)
    assert np.median(t_off.power1d) == pytest.approx(sigma**2 * dchi, rel=0.05)
    assert not np.allclose(t_on.power1d, t_off.power1d)


def test_model_requires_data_2():
    los = LineOfSightPowerSpectrum(
        nu=np.linspace(789.1e6, 887.8e6, 16),
        hp_nside=8,
        kperp_edges=np.linspace(0.0, 0.3, 33),
        include_pixel_window=False,
        include_beam_window=False,
    )
    with pytest.raises(ValueError, match="data_2 is required"):
        los.model_1d_power(which="auto_2", kpar=np.array([0.01]))


def test_data_aliases():
    data = np.zeros((768, 8))
    los = LineOfSightPowerSpectrum(
        data=data, nu=np.linspace(789.1e6, 887.8e6, 8), hp_nside=8
    )
    assert los.data_1 is los.data
    new = np.ones((768, 8))
    los.data_1 = new
    assert los.data is new
    assert los.data_2 is None


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def kaiser_maps():
    """Three LPP-Kaiser maps and their model inputs."""
    rows = []
    for seed in range(1, KAISER_NSEED + 1):
        mock = _kaiser_mock(seed)
        map_cube = mock.propagate_mock_field_to_data(
            mock.mock_tracer_field_1, beam=False
        )
        rows.append((mock, map_cube, _measured_power_func(mock)))
    return rows


def test_los_ps_kaiser_end_to_end(kaiser_maps):
    """The 05/Kaiser analogue: map vs model, averaged over a few seeds."""
    ratios = []
    for mock, map_cube, power_kmu in kaiser_maps:
        los = LineOfSightPowerSpectrum.from_power_spectrum(
            mock, data=map_cube, power_kmu_func=power_kmu
        )
        meas, model = los.run()
        ratios.append(meas.power1d / model)
    ratio = np.mean(ratios, axis=0)
    k = meas.k1deff
    low = k < 0.04
    assert np.any(low)
    assert np.median(np.abs(ratio[low] - 1.0)) < 0.08


def test_los_ps_analytic_model_end_to_end(kaiser_maps):
    """The default analytic model agrees to the accuracy of the mock's own P(k)."""
    ratios = []
    for mock, map_cube, _ in kaiser_maps:
        los = LineOfSightPowerSpectrum.from_power_spectrum(mock, data=map_cube)
        meas, model = los.run()
        ratios.append(meas.power1d / model)
    ratio = np.mean(ratios, axis=0)
    low = meas.k1deff < 0.04
    assert np.median(np.abs(ratio[low] - 1.0)) < 0.20


def test_los_ps_beam_lowers_power():
    mock = _kaiser_mock(1)
    map_cube = mock.propagate_mock_field_to_data(mock.mock_tracer_field_1, beam=False)
    kpar = np.array([0.05, 0.1])
    los = LineOfSightPowerSpectrum.from_power_spectrum(mock, data=map_cube)
    no_beam = los.model_1d_power(kpar=kpar)
    los_beam = LineOfSightPowerSpectrum.from_power_spectrum(
        mock, data=map_cube, sigma_beam_ch=np.full(mock.nu.size, 1.0)
    )
    with_beam = los_beam.model_1d_power(kpar=kpar)
    assert np.all(with_beam < no_beam)
    assert np.all(with_beam > 0.0)


def test_los_ps_wcs_backend():
    """The estimator must work on a WCS cube as well as HEALPix."""
    wcs = create_wcs(0.0, 0.0, [8, 8], [1.0, 1.0])
    mock = MockSimulation(
        nu=np.linspace(789.1e6, 887.8e6, 16),
        wproj=wcs,
        num_pix_x=8,
        num_pix_y=8,
        density="lognormal",
        tracer_bias_1=1.0,
        mean_amp_1=1.0,
        seed=1,
        precision=False,
        model_k_from_field=True,
        include_beam=[False, False],
        include_sky_sampling=[False, False],
        compensate=[False, False],
        downres_factor_transverse=0.25,
        downres_factor_radial=0.43,
        batch_number=8,
    )
    mock.get_enclosing_box()
    map_cube = mock.propagate_mock_field_to_data(mock.mock_tracer_field_1, beam=False)
    assert map_cube.ndim == 3

    los = LineOfSightPowerSpectrum.from_power_spectrum(mock, data=map_cube)
    meas = los.measure_1d_power()
    assert np.all(np.isfinite(meas.power1d))
    n_unmasked = int(np.count_nonzero(np.any(mock.weights_map_pixel > 0, axis=-1)))
    assert meas.n_sightlines == n_unmasked
    model = los.model_1d_power(kpar=meas.k1deff)
    assert model.shape == meas.power1d.shape
    assert np.all(np.isfinite(model))


def _unit_power(k, mu):
    return np.ones_like(k)


def _small_los(nu=np.linspace(789.1e6, 887.8e6, 16), **kw):
    return LineOfSightPowerSpectrum(nu=nu, hp_nside=8, **kw)


def _assert_los_matches(base, fresh):
    assert np.allclose(base.chi_ch, fresh.chi_ch, rtol=1e-12, atol=0.0)
    assert np.allclose(base.dchi_ch, fresh.dchi_ch, rtol=1e-12, atol=0.0)
    assert np.allclose(base.chi_u, fresh.chi_u, rtol=1e-12, atol=0.0)
    assert base.dchi_u == pytest.approx(fresh.dchi_u, rel=1e-12, abs=0.0)
    assert np.allclose(base.kperp_edges, fresh.kperp_edges, rtol=1e-12, atol=0.0)
    assert np.allclose(
        base.transverse_window(), fresh.transverse_window(), rtol=1e-12, atol=0.0
    )
    kpar = np.array([0.02, 0.05])
    assert np.allclose(
        base.model_1d_power(kpar=kpar),
        fresh.model_1d_power(kpar=kpar),
        rtol=1e-12,
        atol=0.0,
    )


@pytest.mark.parametrize(
    "base_kw, attr, value",
    [
        (dict(sigma_beam_ch=0.5), "sigma_beam_ch", 1.0),
        (dict(sigma_beam_ch=0.5), "include_beam_window", False),
        (dict(sigma_beam_ch=0.5), "include_pixel_window", False),
        (dict(sigma_beam_ch=0.5), "over", 3),
        (dict(sigma_beam_ch=0.5), "taper", False),
        (dict(sigma_beam_ch=0.5), "n_chi", 3),
        (dict(sigma_beam_ch=0.5), "n_kperp", 64),
        (dict(sigma_beam_ch=0.5), "radial_transfer", "auto"),
        (dict(sigma_beam_ch=0.5, radial_transfer="auto"), "taper", False),
    ],
)
def test_los_parameter_changes_take_effect(base_kw, attr, value):
    base = _small_los(power_kmu_func=_unit_power, **base_kw)
    base.model_1d_power(kpar=np.array([0.02, 0.05]))
    _ = base.dchi_ch
    setattr(base, attr, value)
    fresh_kw = dict(base_kw)
    fresh_kw[attr] = value
    fresh = _small_los(power_kmu_func=_unit_power, **fresh_kw)
    _assert_los_matches(base, fresh)


def test_los_parameter_changes_take_effect_geometry():
    kpar = np.array([0.02, 0.05])
    new_nu = np.linspace(700e6, 800e6, 16)
    base = _small_los(power_kmu_func=_unit_power)
    base.model_1d_power(kpar=kpar)
    _ = base.dchi_ch
    base.nu = new_nu
    fresh = _small_los(nu=new_nu, power_kmu_func=_unit_power)
    _assert_los_matches(base, fresh)

    base = _small_los(power_kmu_func=_unit_power)
    base.model_1d_power(kpar=kpar)
    _ = base.dchi_ch
    base.fiducial_cosmology = "WMAP9"
    fresh = _small_los(
        power_kmu_func=_unit_power,
        fiducial_cosmology="WMAP9",
        true_cosmology=base.true_cosmology,
    )
    _assert_los_matches(base, fresh)


def test_los_kperp_max_change_takes_effect():
    base = _small_los(sigma_beam_ch=0.5, power_kmu_func=_unit_power)
    base.model_1d_power(kpar=np.array([0.02, 0.05]))
    _ = base.dchi_ch
    kmax = 1.5 * float(base.kperp_edges[-1])
    base.kperp_max = kmax
    fresh = _small_los(sigma_beam_ch=0.5, power_kmu_func=_unit_power, kperp_max=kmax)
    assert base.kperp_edges[-1] == pytest.approx(kmax)
    _assert_los_matches(base, fresh)


def test_uniform_comoving_grid_errors():
    with pytest.raises(ValueError, match="at least two channels"):
        uniform_comoving_grid(np.array([1.0]))
    with pytest.raises(ValueError, match="strictly increasing"):
        uniform_comoving_grid(np.array([2.0, 2.0, 2.0]))
    with pytest.raises(ValueError, match="over must be >= 1"):
        uniform_comoving_grid(np.array([1.0, 3.0]), over=0)
    descending = np.array([5.0, 3.0, 1.0])
    chi_d, dchi_d = uniform_comoving_grid(descending)
    chi_a, dchi_a = uniform_comoving_grid(descending[::-1])
    assert np.allclose(chi_d, chi_a)
    assert dchi_d == pytest.approx(dchi_a)


def test_resample_sightlines_errors_and_order():
    chi = np.linspace(0.0, 10.0, 6)
    data = np.arange(6, dtype=float)
    with pytest.raises(ValueError, match="last axis of cube"):
        resample_sightlines(data, chi[:-1], chi)
    with pytest.raises(ValueError, match="chi_u must be strictly ascending"):
        resample_sightlines(data, chi, chi[::-1])
    with pytest.raises(ValueError, match="chi_in must be strictly monotonic"):
        resample_sightlines(data, np.array([0.0, 1.0, 1.0, 2.0, 3.0, 4.0]), chi)
    forward = resample_sightlines(data, chi, chi)
    backward = resample_sightlines(data[::-1], chi[::-1], chi)
    assert np.allclose(forward, backward)
    cube = np.arange(2 * 3 * 6, dtype=float).reshape(2, 3, 6)
    out = resample_sightlines(cube, chi, chi)
    assert out.shape == (2, 3, len(chi))


def test_weighted_mean_and_error():
    mean, err = _weighted_mean_and_error(np.array([[4.0]]), None)
    assert mean == pytest.approx(4.0)
    assert err == pytest.approx(0.0)
    vals = np.array([[1.0], [3.0]])
    mean, err = _weighted_mean_and_error(vals, np.array([1.0, 3.0]))
    assert mean[0] == pytest.approx(2.5)
    assert err[0] == pytest.approx(np.sqrt(0.375))
    with pytest.raises(ValueError, match="one entry per sightline"):
        _weighted_mean_and_error(vals, np.array([1.0]))
    with pytest.raises(ValueError, match="sum to a positive value"):
        _weighted_mean_and_error(vals, np.array([0.0, 0.0]))


def test_estimator_validation():
    data = np.ones((4, 16))
    edges = np.linspace(0.1, 1.0, 5)
    with pytest.raises(ValueError, match="which must be"):
        estimate_1d_power(data, 1.0, edges, which="galaxy")
    with pytest.raises(ValueError, match="data_1 must be 2-D"):
        estimate_1d_power(np.ones(16), 1.0, edges)
    with pytest.raises(ValueError, match="data_2 must have the same shape"):
        estimate_1d_power(data, 1.0, edges, data_2=np.ones((3, 16)), which="auto_2")
    with pytest.raises(ValueError, match="kpar_edges must be a 1-D array"):
        estimate_1d_power(data, 1.0, np.ones((2, 2)))
    with pytest.raises(ValueError, match="no sightline with a positive weight"):
        estimate_1d_power(data, 1.0, edges, weights_1=np.zeros(4))
    k = 2.0 * np.pi * np.fft.rfftfreq(16, d=1.0)
    gap = np.array([k[2] + 1e-4, k[3] - 1e-4, k[5]])
    meas = estimate_1d_power(data, 1.0, gap, taper=False)
    assert meas.nmodes[0] == 0
    assert np.isnan(meas.power1d[0])
    assert meas.k1deff[0] == 0.0


def test_estimator_auto_2_uses_data_2_and_weights_2():
    rng = np.random.default_rng(4)
    d1 = rng.normal(size=(12, 32))
    d2 = rng.normal(size=(12, 32))
    w = rng.random(12) + 0.1
    edges = np.linspace(0.05, 2.0, 8)
    auto2 = estimate_1d_power(
        d1, 1.2, edges, data_2=d2, weights_2=w, which="auto_2", taper=False
    )
    auto1 = estimate_1d_power(d2, 1.2, edges, weights_1=w, taper=False)
    assert np.allclose(auto2.power1d, auto1.power1d)
    assert np.allclose(auto2.power1d_err, auto1.power1d_err)


def test_estimator_single_tone_normalisation():
    n = 64
    dchi = 1.5
    mode = 5
    amp = 2.0
    sample = np.arange(n)
    tone = amp * np.cos(2.0 * np.pi * mode * sample / n)
    data = np.broadcast_to(tone, (4, n)).copy()
    k = 2.0 * np.pi * np.fft.rfftfreq(n, d=dchi)
    edges = np.concatenate(([0.0], 0.5 * (k[:-1] + k[1:]), [k[-1] + 1.0]))
    meas = estimate_1d_power(data, dchi, edges, taper=False)
    expected = dchi * n * amp**2 / 4.0
    assert meas.power1d[mode] == pytest.approx(expected, rel=1e-12)
    others = np.abs(meas.power1d)
    others[mode] = 0.0
    assert np.all(others < 1e-20 * expected)


def test_los_constructor_validation():
    with pytest.raises(ValueError, match="n_kperp must be at least 8"):
        _small_los(n_kperp=4)
    with pytest.raises(ValueError, match="kperp_edges must be a 1-D array"):
        _small_los(kperp_edges=np.ones((2, 2)))
    with pytest.raises(ValueError, match="kperp_edges must be a 1-D array"):
        _small_los(kperp_edges=np.array([0.1]))
    with pytest.raises(ValueError, match="radial_transfer must be"):
        _small_los(radial_transfer="none")


def test_los_geometry():
    los = _small_los()
    expected = (
        los.astropy_cosmo_fiducial.comoving_distance(np.asarray(los.z_ch, dtype=float))
        .to("Mpc")
        .value
    )
    assert np.allclose(los.chi_ch, expected)
    nu = np.asarray(los.nu, dtype=float)
    dnu = float(los.freq_resol)
    z_far = f_21 / (nu.min() - 0.5 * dnu) - 1.0
    z_near = f_21 / (nu.max() + 0.5 * dnu) - 1.0
    chi_edge = los.astropy_cosmo_fiducial.comoving_distance([z_far, z_near]).value
    assert los.dchi_ch.sum() == pytest.approx(abs(chi_edge[0] - chi_edge[1]), rel=1e-3)
    assert los.dchi_u == pytest.approx(
        np.median(np.abs(np.diff(los.chi_ch))) / los.over
    )
    reversed_nu = _small_los(nu=np.asarray(los.nu)[::-1].copy())
    assert np.allclose(reversed_nu.chi_u, los.chi_u)
    assert reversed_nu.dchi_u == pytest.approx(los.dchi_u)


def test_los_map_handling():
    los = _small_los(power_kmu_func=_unit_power)
    with pytest.raises(ValueError, match="which must be"):
        los.uniform_map("bogus")
    with pytest.raises(ValueError, match="data_2 is required"):
        los.measure_1d_power(which="auto_2")
    los.data_2 = np.zeros(4)
    with pytest.raises(ValueError, match="at least 2 dimensions"):
        los.uniform_map("auto_2")
    los.data_2 = None
    rng = np.random.default_rng(1)
    los.data = rng.normal(size=los.data.shape)
    flat = los.data.reshape(-1, los.data.shape[-1])
    assert np.allclose(
        los.uniform_map(), resample_sightlines(flat, los.chi_ch, los.chi_u)
    )
    edges = los._default_k1dbins()
    weights = los.weights_map_pixel.reshape(flat.shape[0], -1).mean(axis=-1)
    direct = estimate_1d_power(
        los.uniform_map(), los.dchi_u, edges, weights_1=weights, taper=los.taper
    )
    measured = los.measure_1d_power()
    assert np.allclose(measured.power1d, direct.power1d)
    n_full = measured.n_sightlines
    los.data = np.array(los.data, copy=True)
    los.weights_map_pixel = np.array(los.weights_map_pixel, copy=True)
    los.data[0] = np.nan
    los.weights_map_pixel[1] = 0.0
    masked = los.measure_1d_power()
    assert masked.n_sightlines == n_full - 2
    kept = los.measure_1d_power(per_sightline=True)
    assert kept.per_sightline.shape == (flat.shape[0], kept.power1d.size)
    assert los.measure_1d_power().per_sightline is None


def test_los_auto_2_and_cross_match_auto_1():
    rng = np.random.default_rng(7)
    los = _small_los(
        kaiser_rsd=True,
        tracer_bias_1=1.5,
        tracer_bias_2=1.5,
        mean_amp_1=1.0,
        mean_amp_2=1.0,
        sigma_v_1=0.0,
        sigma_v_2=0.0,
        include_pixel_window=False,
        include_beam_window=False,
        kperp_edges=np.linspace(0.0, 0.3, 33),
    )
    los.data = rng.normal(size=los.data.shape)
    los.data_2 = np.array(los.data, copy=True)
    auto_1 = los.measure_1d_power("auto_1")
    auto_2 = los.measure_1d_power("auto_2")
    cross = los.measure_1d_power("cross")
    assert np.allclose(auto_2.power1d, auto_1.power1d, rtol=1e-12, atol=0.0)
    assert np.allclose(cross.power1d, auto_1.power1d, rtol=1e-12, atol=0.0)
    kpar = auto_1.k1deff
    model_1 = los.model_1d_power("auto_1", kpar=kpar)
    model_2 = los.model_1d_power("auto_2", kpar=kpar)
    model_x = los.model_1d_power("cross", kpar=kpar)
    assert np.allclose(model_2, model_1, rtol=1e-10, atol=0.0)
    assert np.allclose(model_x, model_1, rtol=1e-10, atol=0.0)


def test_los_default_bins():
    los = _small_los(power_kmu_func=_unit_power)
    k = 2.0 * np.pi * np.fft.rfftfreq(los.chi_u.size, d=los.dchi_u)
    edges = k[2 : min(k.size, 39)]
    assert np.allclose(los._default_k1dbins(), edges)
    assert np.allclose(los.model_1d_power(), los.model_1d_power(kpar=edges[:-1]))
    bins = np.linspace(0.02, 0.2, 5)
    binned = _small_los(power_kmu_func=_unit_power, k1dbins=bins)
    centres = 0.5 * (bins[:-1] + bins[1:])
    assert np.allclose(binned.model_1d_power(), binned.model_1d_power(kpar=centres))
    assert np.allclose(binned.measure_1d_power().k1dbins, bins)
    override = np.linspace(0.05, 0.15, 4)
    assert np.allclose(binned.measure_1d_power(k1dbins=override).k1dbins, override)
    short = _small_los(
        nu=np.linspace(789.1e6, 887.8e6, 2), over=1, power_kmu_func=_unit_power
    )
    with pytest.raises(ValueError, match="not enough samples"):
        short.model_1d_power()


def test_los_kperp_grid():
    los = _small_los()
    dperp_min = np.sqrt(float(los.pixel_area)) * np.pi / 180.0 * float(los.chi_u.min())
    assert los.kperp_edges[-1] == pytest.approx(4.0 * np.pi / dperp_min)
    assert los.kperp_edges.size == los.n_kperp + 1
    kmax = 3.0 * np.pi / dperp_min
    capped = _small_los(kperp_max=kmax)
    assert capped.kperp_edges[-1] == pytest.approx(kmax)
    with pytest.raises(ValueError, match="kperp_max is below the first zero"):
        _small_los(kperp_max=0.5 * np.pi / dperp_min).kperp_edges
    decreasing = _small_los(kperp_edges=np.array([0.2, 0.1, 0.0]))
    with pytest.raises(ValueError, match="kperp_edges must be strictly increasing"):
        decreasing.kperp_edges
    edges = los.kperp_edges
    assert np.allclose(los.kperp, 0.5 * (edges[:-1] + edges[1:]))
    assert np.allclose(los.dk_perp, np.diff(edges))


def test_los_chi_nodes():
    los = _small_los()
    _, weights = los._chi_nodes()
    assert weights.sum() == pytest.approx(1.0)
    los.taper = lambda n: np.zeros(n)
    _, flat = los._chi_nodes()
    assert np.allclose(flat, np.full(los.n_chi, 1.0 / los.n_chi))


def test_los_model_validation():
    los = _small_los(power_kmu_func=_unit_power)
    with pytest.raises(ValueError, match="which must be"):
        los.model_1d_power(which="galaxy")
    with pytest.raises(ValueError, match="kpar must be non-empty"):
        los.model_1d_power(kpar=[])
    los.power_kmu_func = lambda k, mu: np.ones(k.shape[0])
    with pytest.raises(ValueError, match="power_kmu returned shape"):
        los.model_1d_power(kpar=np.array([0.02]))


def test_los_beam_window_constant_sigma(monkeypatch):
    monkeypatch.setattr(
        LineOfSightPowerSpectrum,
        "sigma_beam_ch_in_mpc",
        property(lambda self: 2.0),
    )
    los = _small_los(
        sigma_beam_ch=0.5,
        include_pixel_window=False,
        power_kmu_func=_unit_power,
    )
    assert np.allclose(
        los.transverse_window(), np.exp(-((2.0 * los.kperp) ** 2)), rtol=1e-12, atol=0.0
    )
    edges = np.linspace(0.0, 2.0, 2001)
    integral = _small_los(
        sigma_beam_ch=0.5,
        include_pixel_window=False,
        kperp_edges=edges,
        power_kmu_func=_unit_power,
    )
    expected = (1.0 - np.exp(-16.0)) / (16.0 * np.pi)
    assert np.allclose(
        integral.model_1d_power(kpar=np.array([0.01])), expected, rtol=1e-5
    )


def test_los_beam_window_follows_comoving_distance():
    """A channel-dependent beam is sampled on chi, independent of nu order."""
    nu = np.linspace(789.1e6, 887.8e6, 16)
    sigma = np.linspace(0.3, 1.2, nu.size)
    common = dict(
        include_pixel_window=False,
        include_beam_window=True,
        power_kmu_func=_unit_power,
        kperp_edges=np.linspace(0.0, 0.05, 33),
    )
    forward = _small_los(nu=nu, sigma_beam_ch=sigma, **common)
    backward = _small_los(
        nu=nu[::-1].copy(), sigma_beam_ch=sigma[::-1].copy(), **common
    )
    assert np.allclose(
        forward.transverse_window(),
        backward.transverse_window(),
        rtol=1e-10,
        atol=0.0,
    )
    # One channel's beam width does not describe the whole band.
    stuck = beam_window_power(
        forward.kperp, float(np.abs(forward.sigma_beam_ch_in_mpc[-1]))
    )
    assert not np.allclose(forward.transverse_window(), stuck, rtol=1e-3)


def test_los_radial_transfer_paths():
    kpar = np.array([0.02, 0.05])
    los = _small_los(
        include_pixel_window=False,
        include_beam_window=False,
        power_kmu_func=_unit_power,
    )
    bare = los.model_1d_power(kpar=kpar)
    identity = RadialTransferFunction(los.chi_u, los.dchi_u)
    los.radial_transfer = identity
    assert np.allclose(los.model_1d_power(kpar=kpar), bare)
    radial = RadialTransferFunction(
        los.chi_u,
        los.dchi_u,
        chi_ch=los.chi_ch,
        dchi_ch=los.dchi_ch,
        dchi_vox=float(np.median(los.dchi_ch)),
    )
    los.radial_transfer = radial
    assert np.allclose(los.model_1d_power(kpar=kpar), bare * radial.transfer(kpar))
    los.radial_transfer = "auto"
    auto = RadialTransferFunction(
        los.chi_u,
        los.dchi_u,
        chi_ch=los.chi_ch,
        dchi_ch=los.dchi_ch,
        dchi_vox=float(np.median(los.dchi_ch)) / 4.0,
        taper=los.taper,
    )
    assert np.allclose(
        los.model_1d_power(kpar=kpar),
        bare * auto.transfer(kpar),
        rtol=1e-12,
        atol=0.0,
    )


def test_los_run_matches_parts():
    rng = np.random.default_rng(0)
    los = _small_los(power_kmu_func=_unit_power)
    los.data = rng.normal(size=los.data.shape)
    measured, model = los.run()
    again = los.measure_1d_power()
    assert np.allclose(measured.power1d, again.power1d)
    assert np.allclose(model, los.model_1d_power(kpar=measured.k1deff))


def test_los_from_power_spectrum_defaults():
    nu = np.linspace(789.1e6, 887.8e6, 16)
    bins = np.linspace(0.02, 0.3, 8)
    ps = PowerSpectrum(nu=nu, hp_nside=8, k1dbins=bins)
    ps.data = np.arange(np.size(ps.data), dtype=float).reshape(ps.data.shape)
    los = LineOfSightPowerSpectrum.from_power_spectrum(ps)
    assert np.allclose(los.k1dbins, ps.k1dbins)
    assert np.allclose(los.data, ps.data)
    overridden = LineOfSightPowerSpectrum.from_power_spectrum(
        ps, tracer_bias_1=2.0, k1dbins=None
    )
    assert overridden.tracer_bias_1 == 2.0
    assert overridden.k1dbins is None


def test_los_radial_auto_kaiser_end_to_end(kaiser_maps):
    ratios_auto = []
    ratios_none = []
    k = None
    for mock, map_cube, power_kmu in kaiser_maps:
        los = LineOfSightPowerSpectrum.from_power_spectrum(
            mock,
            data=map_cube,
            power_kmu_func=power_kmu,
            radial_transfer="auto",
        )
        measured, model = los.run()
        ratios_auto.append(measured.power1d / model)
        k = measured.k1deff
        plain = LineOfSightPowerSpectrum.from_power_spectrum(
            mock, data=map_cube, power_kmu_func=power_kmu
        )
        ratios_none.append(
            measured.power1d / plain.model_1d_power(kpar=measured.k1deff)
        )
    low = k < 0.2
    assert np.median(np.abs(np.mean(ratios_auto, axis=0)[low] - 1.0)) < 0.03
    assert np.median(np.abs(np.mean(ratios_none, axis=0)[low] - 1.0)) > 0.15
