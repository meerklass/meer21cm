"""Tests for FieldPowerSpectrum (estimator)."""

import numpy as np
import pytest

from meer21cm.estimator import FieldPowerSpectrum
from meer21cm.power_ops import get_fourier_density


def test_k_para_reserved_and_unhandled_los():
    field = np.ones((8, 8, 8))
    box_len = np.array([80.0, 80.0, 80.0])

    for reserved in ("endpoint", "firstpoint", "midpoint"):
        fps = FieldPowerSpectrum(field, box_len, los=reserved)
        with pytest.raises(NotImplementedError, match="k_para"):
            _ = fps.k_para
        with pytest.raises(NotImplementedError, match="mu_mode"):
            _ = fps.mu_mode

    fps = FieldPowerSpectrum(field, box_len, los="global")
    fps.los = "bogus"  # bypass constructor validation
    with pytest.raises(ValueError, match="Unhandled los"):
        _ = fps.k_para
    with pytest.raises(ValueError, match="Unhandled los"):
        _ = fps.mu_mode


def test_multipole_bin_index_map_requires_k1dbins():
    field = np.ones((8, 8, 8))
    box_len = np.array([80.0, 80.0, 80.0])
    fps = FieldPowerSpectrum(field, box_len, los="global")
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

    # measure_multipoles: reserved and unhandled los
    for reserved in ("endpoint", "firstpoint", "midpoint"):
        fps_r = FieldPowerSpectrum(field, box_len, los=reserved)
        with pytest.raises(NotImplementedError, match="measure_multipoles"):
            fps_r.measure_multipoles(k1dbins=k1dbins)
    fps.los = "bogus"
    with pytest.raises(ValueError, match="Unhandled los"):
        fps.measure_multipoles(k1dbins=k1dbins)


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
