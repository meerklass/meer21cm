"""
Analytic tests for the monopole Hankel transforms in :mod:`meer21cm.fftlog`.

Closed-form pair (isotropic 3D Fourier convention used by
:class:`~meer21cm.fftlog.PowerToCorrelation` for ``ell=0``)::

    ξ(s) = 1/(2π²) ∫₀^∞ dk k² P(k) j₀(ks)
         = ∫ d³k/(2π)³ P(k) e^{i k·s}

For a Gaussian power spectrum

    P(k) = exp(-β k²),   β > 0,

the correlation function is also Gaussian:

    ξ(s) = (4π β)^{-3/2} exp(-s² / (4β)).

Equivalently, with the parametrisation used in many FFTLog demos,
``P(k) = exp(-½ a² k²)`` (so ``β = a²/2``):

    ξ(s) = (2π a²)^{-3/2} exp(-s² / (2 a²)).

These identities are standard 3D Fourier transforms of Gaussians; they
provide an absolute check of normalisation and of the spherical-Bessel
kernel in :class:`~meer21cm.fftlog.PowerToCorrelation` /
:class:`~meer21cm.fftlog.CorrelationToPower`.
"""

import numpy as np
import pytest

from meer21cm.fftlog import (
    CorrelationToPower,
    FFTlog,
    PowerToCorrelation,
    _SphericalBesselJKernel,
    _pad,
    _parse_extrap,
    _parse_pad_width,
)


def _xi_gaussian_from_pk_gaussian(s, beta):
    """Analytic ξ(s) for P(k) = exp(-β k²)."""
    return (4.0 * np.pi * beta) ** (-1.5) * np.exp(-(s**2) / (4.0 * beta))


def _pk_gaussian(k, beta):
    return np.exp(-beta * k**2)


def test_power_to_correlation_gaussian_analytic():
    """
    P(k)=exp(-β k²) → ξ(s)=(4πβ)^{-3/2} exp(-s²/(4β)).
    """
    beta = 50.0
    k = np.geomspace(1e-4, 2.0, 1024)
    pk = _pk_gaussian(k, beta)

    s, xi = PowerToCorrelation(k, ell=0, lowring=False, xy=1.0)(pk, extrap="edge")
    xi_ana = _xi_gaussian_from_pk_gaussian(s, beta)

    # Avoid endpoints where finite-range / padding effects dominate
    mask = (s > 0.5) & (s < 30.0) & (xi_ana > 1e-8 * xi_ana.max())
    assert mask.sum() > 50
    rel = np.abs(xi[mask] - xi_ana[mask]) / xi_ana[mask]
    assert np.median(rel) < 1e-6
    assert np.max(rel) < 1e-4


def test_power_to_correlation_gaussian_param_a():
    """
    Same pair with P(k)=exp(-½ a² k²), ξ(s)=(2π a²)^{-3/2} exp(-s²/(2 a²)).

    This is the form sketched in ``misc/yamamoto/test_fftlog.ipynb``; the
    notebook was missing the overall ``(2π a²)^{-3/2}`` prefactor.
    """
    a = 4.0
    beta = 0.5 * a**2
    k = np.geomspace(1e-3, 5.0, 1024)
    pk = np.exp(-0.5 * a**2 * k**2)

    s, xi = PowerToCorrelation(k, ell=0, lowring=False, xy=1.0)(pk, extrap="edge")
    xi_ana = (2.0 * np.pi * a**2) ** (-1.5) * np.exp(-(s**2) / (2.0 * a**2))
    # Cross-check equivalence to the β form
    np.testing.assert_allclose(
        xi_ana, _xi_gaussian_from_pk_gaussian(s, beta), rtol=1e-12
    )

    mask = (s > 1.0) & (s < 20.0) & (xi_ana > 1e-6 * xi_ana.max())
    rel = np.abs(xi[mask] - xi_ana[mask]) / xi_ana[mask]
    assert np.median(rel) < 1e-6


def test_correlation_to_power_gaussian_analytic():
    """
    Inverse Hankel: analytic ξ(s) → recovers P(k)=exp(-β k²).
    """
    beta = 50.0
    # Use the s-grid produced by PowerToCorrelation so padding / xy match
    k_ref = np.geomspace(1e-4, 2.0, 1024)
    s, _ = PowerToCorrelation(k_ref, ell=0, lowring=False, xy=1.0)(
        _pk_gaussian(k_ref, beta), extrap="edge"
    )
    xi = _xi_gaussian_from_pk_gaussian(s, beta)

    k, pk = CorrelationToPower(s, ell=0, lowring=False, xy=1.0)(xi, extrap="edge")
    pk_ana = _pk_gaussian(k, beta)

    # Mid-k: avoid both IR and UV edges of the finite log grid
    mask = (k > 0.05) & (k < 0.3)
    rel = np.abs(pk[mask] - pk_ana[mask]) / pk_ana[mask]
    assert np.median(rel) < 1e-6
    assert np.max(rel) < 1e-3


def test_gaussian_roundtrip_via_analytic_xi():
    """
    P → ξ_num ≈ ξ_ana → P recovers the input Gaussian.
    """
    beta = 40.0
    k = np.geomspace(1e-4, 1.5, 1024)
    pk = _pk_gaussian(k, beta)

    s, xi = PowerToCorrelation(k, ell=0, lowring=False, xy=1.0)(pk, extrap="edge")
    xi_ana = _xi_gaussian_from_pk_gaussian(s, beta)
    # Use analytic ξ for the inverse to isolate CorrelationToPower
    k2, pk2 = CorrelationToPower(s, ell=0, lowring=False, xy=1.0)(xi_ana, extrap="edge")

    mask = (k2 > 0.03) & (k2 < 0.3)
    pk_ana = _pk_gaussian(k2[mask], beta)
    rel = np.abs(pk2[mask] - pk_ana) / pk_ana
    assert np.median(rel) < 1e-6
    assert np.max(rel) < 1e-3

    # Numerical ξ should also match analytic well enough for the same mask in s
    sm = (s > 1.0) & (s < 25.0)
    assert np.median(np.abs(xi[sm] - xi_ana[sm]) / xi_ana[sm]) < 1e-6


def test_parse_pad_width_valid():
    assert _parse_pad_width(3) == (3, 3)
    assert _parse_pad_width((1, 2)) == (1, 2)
    assert _parse_pad_width([0, 4]) == (0, 4)


def test_parse_pad_width_errors():
    with pytest.raises(TypeError, match="pad_width"):
        _parse_pad_width("3")
    with pytest.raises(TypeError, match="pad_width"):
        _parse_pad_width(3.5)
    with pytest.raises(TypeError, match="bool"):
        _parse_pad_width(True)
    with pytest.raises(ValueError, match=">= 0"):
        _parse_pad_width(-1)
    with pytest.raises(ValueError, match="length 2"):
        _parse_pad_width((1, 2, 3))
    with pytest.raises(TypeError, match="pad_width\\[1\\]"):
        _parse_pad_width((1, "x"))


def test_parse_extrap_valid():
    assert _parse_extrap("edge") == ("edge", "edge")
    assert _parse_extrap("log") == ("log", "log")
    assert _parse_extrap(0) == (0, 0)
    assert _parse_extrap(("edge", 0.0)) == ("edge", 0.0)
    assert _parse_extrap(["log", "edge"]) == ("log", "edge")


def test_parse_extrap_errors():
    with pytest.raises(ValueError, match="must be one of"):
        _parse_extrap("zero")
    with pytest.raises(ValueError, match="must be one of"):
        _parse_extrap(("edge", "constant"))
    with pytest.raises(TypeError, match="extrap"):
        _parse_extrap(None)
    with pytest.raises(TypeError, match="bool"):
        _parse_extrap(True)
    with pytest.raises(ValueError, match="finite"):
        _parse_extrap(np.nan)
    with pytest.raises(ValueError, match="length 2"):
        _parse_extrap(("edge",))


def test_pad_edge_and_constant():
    x = np.array([1.0, 2.0, 4.0])
    np.testing.assert_allclose(_pad(x, 1, extrap="edge"), [1.0, 1.0, 2.0, 4.0, 4.0])
    np.testing.assert_allclose(
        _pad(x, (1, 2), extrap=0), [0.0, 1.0, 2.0, 4.0, 0.0, 0.0]
    )
    with pytest.raises(ValueError, match="must be one of"):
        _pad(x, 1, extrap="linear")


def test_pad_zero_width_preserves_content():
    """n_pad=0 returns an empty block; one-sided padding must leave the other side alone."""
    x = np.array([1.0, 2.0, 4.0])
    np.testing.assert_array_equal(_pad(x, 0, extrap="edge"), x)
    np.testing.assert_allclose(
        _pad(x, (0, 2), extrap="edge"), [1.0, 2.0, 4.0, 4.0, 4.0]
    )
    np.testing.assert_allclose(_pad(x, (2, 0), extrap=0.0), [0.0, 0.0, 1.0, 2.0, 4.0])


def test_pad_log_extrapolation_matches_power_law():
    """
    Log-log pad continues the local geometric ratio of the two edge samples.

    For x = [1, 2, 4] the edge ratio is 2, so two samples on each side are
    1/4, 1/2 | 1, 2, 4 | 8, 16.
    """
    x = np.array([1.0, 2.0, 4.0])
    np.testing.assert_allclose(
        _pad(x, 2, extrap="log"), [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    )
    # Asymmetric modes: log on the left, constant fill on the right
    np.testing.assert_allclose(
        _pad(x, (2, 1), extrap=("log", -1.0)),
        [0.25, 0.5, 1.0, 2.0, 4.0, -1.0],
    )


def test_pad_log_requires_two_samples():
    with pytest.raises(ValueError, match="log extrapolation requires at least 2"):
        _pad(np.array([3.0]), 1, extrap="log")


def test_pad_rejects_non_numeric_and_bad_axis():
    with pytest.raises(TypeError, match="array must be numeric"):
        _pad(np.array(["a", "b"]), 1)
    with pytest.raises(TypeError, match="axis must be an int"):
        _pad([1.0, 2.0, 3.0], 1, axis=True)
    with pytest.raises(TypeError, match="axis must be an int"):
        _pad([1.0, 2.0, 3.0], 1, axis=1.5)


def test_pad_along_axis_for_2d():
    """Padding along axis=0 uses column-wise edge values."""
    arr = np.array([[1.0, 10.0], [2.0, 20.0], [4.0, 40.0]])
    out = _pad(arr, 1, axis=0, extrap="edge")
    np.testing.assert_allclose(
        out,
        [[1.0, 10.0], [1.0, 10.0], [2.0, 20.0], [4.0, 40.0], [4.0, 40.0]],
    )


def test_spherical_bessel_kernel_nu_validation():
    with pytest.raises(TypeError, match="nu must be an int, got bool"):
        _SphericalBesselJKernel(True)
    with pytest.raises(TypeError, match="nu must be an int, got float"):
        _SphericalBesselJKernel(1.5)
    with pytest.raises(ValueError, match="nu must be >= 0"):
        _SphericalBesselJKernel(-1)


def test_spherical_bessel_mellin_at_half_integer():
    """
    For ν=0 and z=1.5, U(z)=2^{z-3/2} Γ((ν+z)/2)/Γ((3+ν-z)/2) = 1.
    """
    u = _SphericalBesselJKernel(0)(1.5)
    np.testing.assert_allclose(np.real(u), 1.0, rtol=1e-12)
    np.testing.assert_allclose(np.imag(u), 0.0, atol=1e-12)


def test_fftlog_constructor_validation():
    x = np.geomspace(1e-2, 1.0, 8)
    ker = _SphericalBesselJKernel(0)

    with pytest.raises(TypeError, match="kernel must be callable"):
        FFTlog(x, kernel=None)
    with pytest.raises(ValueError, match="q must be finite"):
        FFTlog(x, ker, q=np.nan)
    with pytest.raises(ValueError, match="x must be 1D"):
        FFTlog(np.ones((2, 4)), ker)
    with pytest.raises(ValueError, match="length >= 2"):
        FFTlog([1.0], ker)
    with pytest.raises(ValueError, match="strictly positive"):
        FFTlog([0.0, 1.0], ker)
    with pytest.raises(ValueError, match="x must be finite"):
        FFTlog([1.0, np.inf], ker)
    with pytest.raises(ValueError, match="minfolds must be >= 1"):
        FFTlog(x, ker, minfolds=0)
    with pytest.raises(ValueError, match="xy must be a positive finite float"):
        FFTlog(x, ker, lowring=False, xy=0.0)
    with pytest.raises(ValueError, match="xy must be a positive finite float"):
        FFTlog(x, ker, lowring=False, xy=np.nan)


def test_minfolds_bool_raises_typeerror():
    k = np.geomspace(1e-3, 1.0, 16)
    with pytest.raises(TypeError, match="minfolds must be an int, got bool"):
        PowerToCorrelation(k, ell=0, minfolds=True)


def test_lowring_shifts_grid_and_gaussian_still_accurate():
    """
    Hamilton low-ringing chooses a different reciprocal product than xy=1,
    so the s-grid moves; the Gaussian Hankel identity must still hold.
    """
    beta = 50.0
    k = np.geomspace(1e-4, 2.0, 1024)
    pk = _pk_gaussian(k, beta)

    tr_lowring = PowerToCorrelation(k, ell=0, lowring=True)
    tr_xy = PowerToCorrelation(k, ell=0, lowring=False, xy=1.0)
    assert not np.allclose(tr_lowring.y, tr_xy.y)

    s, xi = tr_lowring(pk, extrap="edge")
    xi_ana = _xi_gaussian_from_pk_gaussian(s, beta)
    mask = (s > 0.5) & (s < 30.0) & (xi_ana > 1e-8 * xi_ana.max())
    rel = np.abs(xi[mask] - xi_ana[mask]) / xi_ana[mask]
    assert np.median(rel) < 1e-6
    assert np.max(rel) < 1e-4


def test_call_rejects_wrong_fun_length():
    k = np.geomspace(1e-3, 1.0, 32)
    tr = PowerToCorrelation(k, ell=0, lowring=False, xy=1.0)
    with pytest.raises(ValueError, match="fun last dimension"):
        tr(np.ones(16), extrap="edge")


def test_multipole_ell_validation():
    k = np.geomspace(1e-3, 1.0, 16)
    with pytest.raises(TypeError, match="ell must be an int, got bool"):
        PowerToCorrelation(k, ell=True)
    with pytest.raises(ValueError, match="ell must be >= 0"):
        PowerToCorrelation(k, ell=-2)
    with pytest.raises(TypeError, match="ell must be an int, got bool"):
        CorrelationToPower(k, ell=True)
    with pytest.raises(ValueError, match="ell must be >= 0"):
        CorrelationToPower(k, ell=-1)
