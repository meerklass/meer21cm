"""Tests for the shared 1D line-of-sight transverse projection model."""

import numpy as np
import pytest

from meer21cm import LineOfSightPowerSpectrum, OneDimPowerModel, PowerSpectrum, Stacking
from meer21cm.los_model import (
    azimuthal_pixel_window,
    beam_window,
    chi_nodes_and_weights,
    discrete_beam_window,
    pixel_window_power,
)
from meer21cm.util import create_wcs


def test_window_helpers_exponents_and_validation():
    kperp = np.linspace(0.0, 0.3, 32)
    assert np.allclose(pixel_window_power(0.0, 0.0, 15.0), 1.0)
    assert np.allclose(beam_window(kperp, 20.0, power=0.0), 1.0)
    assert np.allclose(
        beam_window(kperp, 20.0, power=2.0),
        beam_window(kperp, 20.0, power=1.0) ** 2,
    )
    with pytest.raises(ValueError, match="power must be non-negative"):
        beam_window(kperp, 20.0, power=-1.0)

    # Discrete beam with beam_power=0, 1, 1.5, 2
    w_bp0 = discrete_beam_window(kperp, 20.0, 25.0, beam_power=0.0)
    w_pix = azimuthal_pixel_window(kperp, 20.0, pixel_power=2.0)
    assert np.allclose(w_bp0, w_pix)
    w_bp1 = discrete_beam_window(kperp, 20.0, 25.0, beam_power=1.0)
    w_bp15 = discrete_beam_window(kperp, 20.0, 25.0, beam_power=1.5)
    w_bp2 = discrete_beam_window(kperp, 20.0, 25.0, beam_power=2.0)
    assert w_bp2[5] < w_bp15[5] < w_bp1[5] < w_bp0[5]
    with pytest.raises(ValueError, match="beam_power must be non-negative"):
        discrete_beam_window(kperp, 20.0, 25.0, beam_power=-0.5)


def test_chi_nodes_and_weights_validation_and_edges():
    with pytest.raises(ValueError, match="n_chi"):
        chi_nodes_and_weights([1000.0, 1100.0], n_chi=0)
    with pytest.raises(ValueError, match="chi_grid"):
        chi_nodes_and_weights([], n_chi=3)
    with pytest.raises(ValueError, match="weights length"):
        chi_nodes_and_weights([1000.0, 1100.0], n_chi=2, weights=[1.0])
    nodes, w = chi_nodes_and_weights([1000.0, 1000.0], n_chi=3)
    assert nodes.shape == (1,) and w[0] == pytest.approx(1.0)


def test_one_dim_power_model_standalone_and_shared_consistency():
    n_pix = 5
    n_ch = 12
    wcs = create_wcs(180.0, 0.0, [n_pix, n_pix], 0.5)
    nu = np.linspace(700e6, 724e6, n_ch)
    ps = PowerSpectrum(
        data=np.zeros((n_pix, n_pix, n_ch)),
        weights_map_pixel=np.ones((n_pix, n_pix, n_ch)),
        nu=nu,
        wproj=wcs,
        num_pix_x=n_pix,
        num_pix_y=n_pix,
        tracer_bias_1=1.4,
        tracer_bias_2=2.0,
        sigma_v_1=200.0,
        sigma_v_2=50.0,
        mean_amp_1="average_hi_temp",
        mean_amp_2=1.0,
        sigma_beam_ch=np.full(n_ch, 0.6),
        filter_map_los=False,
        auto_set_radecnu_bounds=False,
        init_box_from_map_data=False,
    )
    m1d = OneDimPowerModel(ps, n_chi=3, n_kperp=64, beam_power=1.0)
    assert m1d.dchi_ch.shape == (n_ch,)
    assert m1d.dk_perp.shape == (64,)
    kpar = np.array([0.02, 0.08, 0.2])

    # Matches Stacking.model_1d_power identically and preserves ps._kmode/_mumode
    orig_kmode = None if ps._kmode is None else np.array(ps._kmode, copy=True)
    orig_mumode = None if ps._mumode is None else np.array(ps._mumode, copy=True)
    st = Stacking.from_power_spectrum(ps, n_chi=3, n_kperp=64)
    p_m1d = m1d.model_1d_power(kpar, which="cross")
    p_stack = st.model_1d_power(kpar, which="cross")
    assert np.allclose(p_m1d, p_stack, rtol=1e-13, atol=0.0)
    if orig_kmode is None:
        assert ps._kmode is None
    else:
        assert np.array_equal(ps._kmode, orig_kmode)
    if orig_mumode is None:
        assert ps._mumode is None
    else:
        assert np.array_equal(ps._mumode, orig_mumode)

    # Live mutation on bound ps is reflected immediately without state desync
    ps.tracer_bias_2 = 4.0
    p_doubled = m1d.model_1d_power(kpar, which="cross")
    assert np.all(p_doubled > p_m1d)

    # Matches LineOfSightPowerSpectrum with beam_power=2 on same chi_u / taper weights
    los = LineOfSightPowerSpectrum.from_power_spectrum(
        ps, n_chi=3, n_kperp=64, beam_power=2.0, beam_mode="discrete"
    )
    m1d_los = OneDimPowerModel(
        los, n_chi=3, n_kperp=64, beam_power=2.0, beam_mode="discrete"
    )
    chi_u = los.chi_u
    from meer21cm.util import taper_values

    w2 = taper_values(los.taper, chi_u.size) ** 2
    assert np.allclose(
        m1d_los.model_1d_power(
            kpar,
            which="auto_1",
            chi_min=float(chi_u.min()),
            chi_grid=chi_u,
            node_weights=w2,
        ),
        los.model_1d_power(which="auto_1", kpar=kpar),
        rtol=1e-13,
        atol=0.0,
    )

    # Validation errors on OneDimPowerModel constructor
    with pytest.raises(ValueError, match="n_chi"):
        OneDimPowerModel(ps, n_chi=0)
    with pytest.raises(ValueError, match="n_kperp"):
        OneDimPowerModel(ps, n_kperp=4)
    with pytest.raises(ValueError, match="beam_mode"):
        OneDimPowerModel(ps, beam_mode="airy")
    with pytest.raises(ValueError, match="kperp_edges"):
        OneDimPowerModel(ps, kperp_edges=[0.1])
