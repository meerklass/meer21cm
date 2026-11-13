"""Tests for the one-dimensional line-of-sight power spectrum."""

import numpy as np
import pytest

from meer21cm import LineOfSightPowerSpectrum, MockSimulation
from meer21cm.los import (
    LineOfSightMeasurement,
    azimuthal_pixel_window_power,
    beam_window_power,
    estimate_1d_power,
    hann_window,
    pixel_window_power,
    resample_sightlines,
    uniform_comoving_grid,
)
from meer21cm.power_ops import bin_3d_to_1d, get_power_spectrum
from meer21cm.util import create_wcs

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
