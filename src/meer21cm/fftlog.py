r"""
Minimal FFTlog / Hankel transforms for power ↔ correlation multipoles.

This module implements Hamilton's FFTlog algorithm for integrals of the form

.. math::

    G(y) = \int_0^\infty x\,\mathrm{d}x\, F(x)\, K(xy),

evaluated on a logarithmically spaced grid via FFT convolution with the
Mellin transform of the kernel :math:`K`. It is used by
:mod:`meer21cm.smooth_window` to convert window multipoles
:math:`W_\ell(k) \leftrightarrow W_\ell(s)` when building the plane-parallel
survey-window matrix :math:`W_{\ell\ell'}(k,k')`.

Public classes
--------------
- :class:`FFTlog` — generic transform with a user-supplied Mellin kernel.
- :class:`PowerToCorrelation` — :math:`P_\ell(k) \to \xi_\ell(s)`.
- :class:`CorrelationToPower` — :math:`\xi_\ell(s) \to P_\ell(k)`.

Design notes
------------
- Dependency-free inside ``meer21cm`` (numpy + scipy only). The algorithm
  follows the same conventions as `pypower` / `mcfit` / Hamilton's FFTLog.
- Input coordinates **must** be strictly positive and log-spaced.
- For smooth-window round-trips in this package we typically use
  ``lowring=False`` and ``xy=1.0`` so that
  ``CorrelationToPower(PowerToCorrelation(k)[0])`` recovers the same ``k``
  grid (see tests in ``tests/test_smooth_window.py`` / ``tests/test_fftlog.py``).
- Only a single 1D transform is supported per instance (no multi-kernel
  batching). That is enough for the even multipoles used in the first-level
  smooth-window path.

References
----------
- Hamilton, A. J. S., 2000, MNRAS, 312, 257 (FFTLog):
  https://jila.colorado.edu/~ajsh/FFTLog/
- Fang, X., et al. (mcfit): https://github.com/eelregit/mcfit
- cosmodesi/pypower ``fftlog.py`` (API and spherical-Bessel conventions
  used here)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import loggamma

ExtrapMode = Literal["edge", "log"]
ExtrapSide = ExtrapMode | float | int | np.floating | np.integer
ExtrapSpec = ExtrapSide | tuple[ExtrapSide, ExtrapSide]
PadWidth = int | np.integer | tuple[int, int] | Sequence[int]

_EXTRAP_MODES: frozenset[str] = frozenset({"edge", "log"})


def _as_nonneg_int(value: object, name: str) -> int:
    """Cast ``value`` to a non-negative int, or raise ``TypeError`` / ``ValueError``."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an int, got bool")
    if isinstance(value, (int, np.integer)):
        ivalue = int(value)
    else:
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if ivalue < 0:
        raise ValueError(f"{name} must be >= 0, got {ivalue}")
    return ivalue


def _parse_pad_width(pad_width: PadWidth) -> tuple[int, int]:
    """
    Normalize ``pad_width`` to ``(left, right)``.

    Accepts a single non-negative int (both sides) or a length-2 sequence of
    non-negative ints.
    """
    if isinstance(pad_width, (bool, np.bool_)):
        raise TypeError(
            "pad_width must be an int or a length-2 sequence of ints, got bool"
        )
    if isinstance(pad_width, (int, np.integer)):
        n = _as_nonneg_int(pad_width, "pad_width")
        return n, n
    if isinstance(pad_width, Sequence) and not isinstance(pad_width, (str, bytes)):
        if len(pad_width) != 2:
            raise ValueError(
                f"pad_width sequence must have length 2, got length {len(pad_width)}"
            )
        left = _as_nonneg_int(pad_width[0], "pad_width[0]")
        right = _as_nonneg_int(pad_width[1], "pad_width[1]")
        return left, right
    raise TypeError(
        "pad_width must be an int or a length-2 sequence of ints, "
        f"got {type(pad_width).__name__}"
    )


def _parse_extrap_side(extrap: object, name: str) -> ExtrapSide:
    """
    Validate one side of an extrapolation spec.

    Allowed values are ``'edge'``, ``'log'``, or a real numeric fill constant.
    """
    if isinstance(extrap, str):
        if extrap not in _EXTRAP_MODES:
            raise ValueError(
                f"{name} string must be one of {sorted(_EXTRAP_MODES)}, got {extrap!r}"
            )
        return extrap  # type: ignore[return-value]
    if isinstance(extrap, (bool, np.bool_)):
        raise TypeError(f"{name} must be 'edge', 'log', or a real number, got bool")
    if isinstance(extrap, (int, float, np.integer, np.floating)):
        if not np.isfinite(extrap):
            raise ValueError(f"{name} fill value must be finite, got {extrap!r}")
        return extrap
    raise TypeError(
        f"{name} must be 'edge', 'log', or a real number, got {type(extrap).__name__}"
    )


def _parse_extrap(extrap: ExtrapSpec) -> tuple[ExtrapSide, ExtrapSide]:
    """
    Normalize ``extrap`` to ``(left, right)``.

    A single value is applied to both sides; a length-2 sequence/tuple sets
    each side independently.
    """
    if isinstance(extrap, Sequence) and not isinstance(extrap, (str, bytes)):
        if len(extrap) != 2:
            raise ValueError(
                f"extrap sequence must have length 2, got length {len(extrap)}"
            )
        return (
            _parse_extrap_side(extrap[0], "extrap[0]"),
            _parse_extrap_side(extrap[1], "extrap[1]"),
        )
    side = _parse_extrap_side(extrap, "extrap")
    return side, side


def _pad_one_side(
    array: NDArray[np.floating],
    n_pad: int,
    *,
    side: Literal["left", "right"],
    extrap: ExtrapSide,
    axis: int,
    to_axis: list[int],
) -> NDArray[np.floating]:
    """Build the left or right pad block for :func:`_pad`."""
    if n_pad == 0:
        empty_shape = list(array.shape)
        empty_shape[axis] = 0
        return np.empty(empty_shape, dtype=array.dtype)

    if side == "left":
        edge_idx = 0
        next_idx = 1
        exp = np.arange(-n_pad, 0).reshape(to_axis)
    else:
        edge_idx = -1
        next_idx = -2
        exp = np.arange(1, n_pad + 1).reshape(to_axis)

    if extrap == "edge":
        end = np.take(array, [edge_idx], axis=axis)
        return np.repeat(end, n_pad, axis=axis)
    if extrap == "log":
        if array.shape[axis] < 2:
            raise ValueError("log extrapolation requires at least 2 samples along axis")
        end = np.take(array, [edge_idx], axis=axis)
        neighbour = np.take(array, [next_idx], axis=axis)
        ratio = neighbour / end
        # Left: end * ratio**exp with exp < 0 extends to smaller values.
        # Right: pypower uses end / ratio**exp so a rising geometric series
        # continues outward (not mirrored back into the interior).
        if side == "left":
            return end * ratio**exp
        return end / ratio**exp

    fill = float(extrap)
    fill_shape = array.shape[:axis] + (n_pad,) + array.shape[axis + 1 :]
    return np.full(fill_shape, fill, dtype=array.dtype)


def _pad(
    array: ArrayLike,
    pad_width: PadWidth,
    axis: int = -1,
    extrap: ExtrapSpec = 0,
) -> NDArray[np.floating]:
    """
    Pad ``array`` along ``axis`` to the FFTlog integration length.

    The FFTlog discrete convolution is performed on a zero-padded (or
    extrapolated) array whose length is a power of two. This helper builds
    that padded array without changing the physical content of the original
    samples.

    Parameters
    ----------
    array : array_like
        Input array.
    pad_width : int or length-2 sequence of int
        Number of samples to add on each side. A single int pads both sides
        equally; ``(left, right)`` sets each side independently. Values must
        be ``>= 0``.
    axis : int, default -1
        Axis along which to pad.
    extrap : {'edge', 'log'}, float, or length-2 sequence, default 0
        Extrapolation mode for each side (or a common value for both):

        - ``'edge'``: repeat the edge value.
        - ``'log'``: log-log power-law extrapolation using the two edge
          samples (appropriate for log-spaced coordinates / spectra).
        - a finite real number: fill with that constant (default ``0``).

        Pass a length-2 sequence to use different modes on each side.

    Returns
    -------
    padded : ndarray
        Array with ``pad_width`` samples added on each side of ``axis``.

    Raises
    ------
    TypeError
        If ``pad_width``, ``extrap``, or ``axis`` has an unsupported type.
    ValueError
        If ``pad_width`` is negative, ``extrap`` is an unknown string /
        non-finite fill, or a sequence has the wrong length.
    """
    array_np = np.asarray(array)
    if not np.issubdtype(array_np.dtype, np.number):
        raise TypeError(f"array must be numeric, got dtype={array_np.dtype}")
    pad_left_n, pad_right_n = _parse_pad_width(pad_width)
    extrap_left, extrap_right = _parse_extrap(extrap)

    if isinstance(axis, (bool, np.bool_)) or not isinstance(axis, (int, np.integer)):
        raise TypeError(f"axis must be an int, got {type(axis).__name__}")
    axis_i = int(axis) % array_np.ndim
    to_axis = [1] * array_np.ndim
    to_axis[axis_i] = -1

    pad_left = _pad_one_side(
        array_np,
        pad_left_n,
        side="left",
        extrap=extrap_left,
        axis=axis_i,
        to_axis=to_axis,
    )
    pad_right = _pad_one_side(
        array_np,
        pad_right_n,
        side="right",
        extrap=extrap_right,
        axis=axis_i,
        to_axis=to_axis,
    )
    return np.concatenate([pad_left, array_np, pad_right], axis=axis_i)


class _SphericalBesselJKernel:
    r"""
    Mellin transform of the spherical Bessel function :math:`j_\nu(t)`.

    The FFTlog kernel is defined as

    .. math::

        U_K(z) = \int_0^\infty t^{z-1} K(t)\,\mathrm{d}t,

    and for :math:`K(t) = j_\nu(t)` one has (up to the conventions below)

    .. math::

        U(z) = 2^{z-3/2}
        \frac{\Gamma\big(\tfrac{1}{2}(\nu+z)\big)}
             {\Gamma\big(\tfrac{1}{2}(3+\nu-z)\big)}.

    Note
    ----
    Hamilton's FFTLog documentation defines the Mellin transform with
    :math:`t^{z}` rather than :math:`t^{z-1}`. With *this* kernel definition
    the spherical-Bessel power ↔ correlation transforms use an effective
    tilt ``q = 1.5 + q_user`` in :class:`PowerToCorrelation` /
    :class:`CorrelationToPower` (matching pypower / mcfit).
    """

    def __init__(self, nu: int) -> None:
        r"""
        Parameters
        ----------
        nu : int
            Spherical Bessel order :math:`\nu` (multipole :math:`\ell` for
            :math:`j_\ell`).
        """
        if isinstance(nu, (bool, np.bool_)):
            raise TypeError("nu must be an int, got bool")
        if not isinstance(nu, (int, np.integer)):
            raise TypeError(f"nu must be an int, got {type(nu).__name__}")
        if int(nu) < 0:
            raise ValueError(f"nu must be >= 0, got {int(nu)}")
        self.nu = int(nu)

    def __call__(self, z: complex | ArrayLike) -> complex | NDArray[np.complexfloating]:
        r"""
        Evaluate the Mellin kernel at complex frequency ``z``.

        Parameters
        ----------
        z : complex or ndarray
            Mellin frequency (typically ``q + 2\pi i m / (N \Delta)``).

        Returns
        -------
        u : complex or ndarray
            :math:`U_K(z)`.
        """
        return np.exp(
            np.log(2) * (z - 1.5)
            + loggamma(0.5 * (self.nu + z))
            - loggamma(0.5 * (3 + self.nu - z))
        )


class FFTlog:
    r"""
    Generic FFTlog transform

    .. math::

        G(y) = \int_0^\infty x\,\mathrm{d}x\, F(x)\, K(xy).

    The integral is rewritten as a convolution in logarithmic coordinates and
    evaluated with an FFT. A power-law bias :math:`q` regularises the
    integrand:

    .. math::

        F_q(x) = F(x)\, x^{-q},\qquad
        K_q(t) = K(t)\, t^{q},\qquad
        G_q(y) = G(y)\, y^{q},

    so that the same mathematical :math:`G` is recovered after multiplying
    back by :math:`y^{-q}` (implemented via :attr:`padded_prefactor` /
    :attr:`padded_postfactor`).

    Parameters
    ----------
    x : array_like
        Strictly positive, **logarithmically spaced** input coordinates
        (length ``N``). For cosmology these are typically :math:`k` or
        :math:`s`.
    kernel : callable
        Mellin transform :math:`U_K(z)` of the integral kernel. Must accept
        complex ``z`` and return complex ``U_K(z)``.
    q : float, default 0
        Power-law tilt used to regularise the transform. Prefer values that
        make :math:`F(x) x^{1-q}` and :math:`K(t) t^{q}` well behaved at
        the endpoints.
    minfolds : int, default 2
        Controls zero-padding: the FFT length is the smallest power of two
        strictly larger than ``minfolds * N``. Larger padding reduces
        wrap-around (ringing) at the cost of CPU / memory.
    lowring : bool, default True
        If ``True``, choose the output grid from Hamilton's low-ringing
        condition (phase of :math:`U_K` at the Nyquist Mellin mode). If
        ``False``, set the reciprocal product with ``xy`` so that
        ``x[0] * y[-1] ≈ xy`` (useful for invertible round-trips on a fixed
        grid).
    xy : float, default 1.0
        Reciprocal product :math:`x_0 y_{N-1}` used when ``lowring=False``.

    Attributes
    ----------
    x : ndarray
        Input coordinate grid.
    y : ndarray
        Output coordinate grid (same length as ``x``).
    q : float
        Power-law tilt.
    delta : float
        Logarithmic spacing :math:`\Delta = \ln(x_{N-1}/x_0)/(N-1)`.
    padded_size : int
        FFT length after padding.
    kernel : callable
        Mellin kernel :math:`U_K`.

    Notes
    -----
    Discrete implementation (schematic):

    1. Pad :math:`F(x)` and multiply by :math:`x^{-q}`.
    2. ``rfft`` → multiply by :math:`U_K(q + 2\pi i m /(N_\mathrm{pad}\Delta))`
       times a phase that centres the output grid.
    3. ``irfft`` of the conjugate (numpy real-FFT convention matching
       pypower's ``NumpyFFTEngine``) and multiply by :math:`y^{-q}`.
    4. Crop padding to recover ``G`` on :attr:`y`.

    Examples
    --------
    Prefer the specialised subclasses for cosmology. For a custom kernel::

        def U(z):
            ...  # Mellin transform of K

        tr = FFTlog(x, U, q=0.0, lowring=False, xy=1.0)
        y, G = tr(F, extrap='edge')
    """

    def __init__(
        self,
        x: ArrayLike,
        kernel: Callable[[complex | ArrayLike], complex | ArrayLike],
        q: float = 0.0,
        minfolds: int = 2,
        lowring: bool = True,
        xy: float = 1.0,
    ) -> None:
        if not callable(kernel):
            raise TypeError(f"kernel must be callable, got {type(kernel).__name__}")
        self.kernel = kernel
        self.q = float(q)
        if not np.isfinite(self.q):
            raise ValueError(f"q must be finite, got {self.q}")
        self.x = np.asarray(x, dtype=float)
        if self.x.ndim != 1:
            raise ValueError(f"x must be 1D, got ndim={self.x.ndim}")
        if self.x.size < 2:
            raise ValueError(f"x must have length >= 2, got {self.x.size}")
        if np.any(self.x <= 0):
            raise ValueError("x must be strictly positive")
        if not np.all(np.isfinite(self.x)):
            raise ValueError("x must be finite")
        self.minfolds = _as_nonneg_int(minfolds, "minfolds")
        if self.minfolds < 1:
            raise ValueError(f"minfolds must be >= 1, got {self.minfolds}")
        self.lowring = bool(lowring)
        self.xy = float(xy)
        if not np.isfinite(self.xy) or self.xy <= 0:
            raise ValueError(f"xy must be a positive finite float, got {self.xy}")
        self._setup()

    def _setup(self) -> None:
        """
        Precompute padded grids, Mellin frequencies, and FFT multipliers.

        Called once from :meth:`__init__`. Sets :attr:`y`, :attr:`delta`,
        :attr:`padded_u`, :attr:`padded_prefactor`, and :attr:`padded_postfactor`.
        """
        self.size = int(self.x.size)
        self.delta = float(np.log(self.x[-1] / self.x[0]) / (self.size - 1))
        nfolds = (self.size * self.minfolds - 1).bit_length()
        self.padded_size = 2**nfolds
        npad = self.padded_size - self.size
        self.pad_in_left, self.pad_in_right = npad // 2, npad - npad // 2
        self.pad_out_left, self.pad_out_right = npad - npad // 2, npad // 2

        if self.lowring:
            # Hamilton low-ringing: choose ln(x y) from arg U(q + i π/Δ)
            self.lnxy = (
                self.delta
                / np.pi
                * np.angle(self.kernel(self.q + 1j * np.pi / self.delta))
            )
        else:
            self.lnxy = np.log(self.xy) + self.delta

        # Output grid: y_i ∝ 1/x_{N-1-i} with reciprocal product fixed by lnxy
        self.y = np.exp(self.lnxy - self.delta) / self.x[::-1]
        m = np.arange(0, self.padded_size // 2 + 1)
        self.padded_x = _pad(
            self.x, (self.pad_in_left, self.pad_in_right), extrap="log"
        )
        self.padded_y = _pad(
            self.y, (self.pad_out_left, self.pad_out_right), extrap="log"
        )
        self.padded_prefactor = self.padded_x ** (-self.q)
        self.padded_postfactor = self.padded_y ** (-self.q)
        # Mellin frequencies for the real-FFT modes
        u = self.kernel(self.q + 2j * np.pi / self.padded_size / self.delta * m)
        self.padded_u = u * np.exp(
            -2j * np.pi * self.lnxy / self.padded_size / self.delta * m
        )

    def __call__(
        self, fun: ArrayLike, extrap: ExtrapSpec = 0
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        Evaluate the transform of ``fun`` sampled on :attr:`x`.

        Parameters
        ----------
        fun : array_like
            Function values :math:`F(x_i)` with ``fun.shape == (len(x),)``.
        extrap : {'edge', 'log'}, float, or length-2 sequence, default 0
            How to extend ``fun`` into the padded region. See :func:`_pad`.
            For smooth cosmological spectra, ``'edge'`` is usually safer
            than zero-padding.

        Returns
        -------
        y : ndarray
            Output coordinates (same as :attr:`y`).
        result : ndarray
            Transformed function :math:`G(y_i)`, cropped to the original
            length ``len(x)``.
        """
        fun_np = np.asarray(fun, dtype=float)
        if fun_np.shape[-1] != self.size:
            raise ValueError(
                f"fun last dimension {fun_np.shape[-1]} != len(x)={self.size}"
            )
        padded = _pad(fun_np, (self.pad_in_left, self.pad_in_right), extrap=extrap)
        # Match pypower NumpyFFTEngine: irfft(conj(rfft(f)*u))
        fwd = np.fft.rfft(padded * self.padded_prefactor) * self.padded_u
        ffted = np.fft.irfft(np.conj(fwd), n=self.padded_size) * self.padded_postfactor
        return self.y, ffted[self.pad_out_left : self.pad_out_left + self.size]


class PowerToCorrelation(FFTlog):
    r"""
    Hankel transform from power-spectrum multipoles to correlation multipoles.

    .. math::

        \xi_\ell(s)
        = \frac{(-i)^\ell}{2\pi^2}
          \int_0^\infty \mathrm{d}k\, k^2\, P_\ell(k)\, j_\ell(ks).

    Implementation details
    ----------------------
    The integral is cast into the FFTlog form with kernel :math:`j_\ell`
    and an effective tilt ``q_fftlog = 1.5 + q``. Extra factors
    :math:`k^3 / (2\pi)^{3/2}` are absorbed into :attr:`padded_prefactor`
    (together with the base :math:`x^{-q}`), matching the pypower / mcfit
    spherical-Bessel convention.

    For **even** :math:`\ell`, :math:`(-i)^\ell = (-1)^{\ell/2}` is real,
    so this class returns the real multipole. Odd multipoles are supported
    with the same real-phase convention used by pypower when the imaginary
    part of the odd power spectrum is supplied as input
    (``phase = (-1)^{\ell//2}``).

    Parameters
    ----------
    k : array_like
        Log-spaced wavenumbers :math:`k` (must be ``> 0``).
    ell : int, default 0
        Multipole order :math:`\ell`.
    q : float, default 0
        Additional FFTlog tilt on top of the spherical-Bessel default
        ``1.5``. Increase slightly if the integrand is poorly behaved at
        the endpoints.
    **kwargs
        Forwarded to :class:`FFTlog` (``minfolds``, ``lowring``, ``xy``, …).
        For invertible round-trips with :class:`CorrelationToPower`, use
        ``lowring=False, xy=1.0``.

    Examples
    --------
    ::

        k = np.geomspace(1e-3, 1.0, 256)
        pk = np.exp(-(k / 0.1) ** 2)
        s, xi = PowerToCorrelation(k, ell=0, lowring=False, xy=1.0)(
            pk, extrap='edge'
        )
    """

    def __init__(
        self,
        k: ArrayLike,
        ell: int = 0,
        q: float = 0.0,
        **kwargs: object,
    ) -> None:
        if isinstance(ell, (bool, np.bool_)):
            raise TypeError("ell must be an int, got bool")
        ell_i = int(ell)
        if ell_i < 0:
            raise ValueError(f"ell must be >= 0, got {ell_i}")
        self.ell = ell_i
        super().__init__(k, _SphericalBesselJKernel(ell_i), q=1.5 + float(q), **kwargs)
        # k^3 / (2π)^{3/2} from the spherical-Bessel measure / normalisation
        self.padded_prefactor = (
            self.padded_prefactor * self.padded_x**3 / (2 * np.pi) ** 1.5
        )
        phase = (-1) ** (ell_i // 2)
        self.padded_postfactor = self.padded_postfactor * phase


class CorrelationToPower(FFTlog):
    r"""
    Hankel transform from correlation multipoles to power-spectrum multipoles.

    .. math::

        P_\ell(k)
        = 4\pi\, i^\ell
          \int_0^\infty \mathrm{d}s\, s^2\, \xi_\ell(s)\, j_\ell(ks).

    This is the inverse (in the continuum sense) of
    :class:`PowerToCorrelation` for the same :math:`\ell` when both are
    constructed with ``lowring=False`` and the same ``xy``.

    Implementation details
    ----------------------
    Uses the same spherical-Bessel Mellin kernel as
    :class:`PowerToCorrelation`, with effective tilt ``1.5 + q`` and
    prefactor :math:`s^3 (2\pi)^{3/2}`. The real-phase convention
    ``(-1)^{\ell//2}`` matches pypower for even multipoles (and for the
    imaginary part of odd multipoles).

    Parameters
    ----------
    s : array_like
        Log-spaced separations :math:`s` (must be ``> 0``). Typically the
        ``s`` returned by :class:`PowerToCorrelation`.
    ell : int, default 0
        Multipole order :math:`\ell`.
    q : float, default 0
        Additional FFTlog tilt on top of ``1.5``.
    **kwargs
        Forwarded to :class:`FFTlog`.

    Examples
    --------
    Round-trip on a fixed grid::

        k = np.geomspace(1e-3, 1.0, 256)
        pk = np.exp(-(k / 0.1) ** 2)
        s, xi = PowerToCorrelation(k, ell=0, lowring=False, xy=1.0)(
            pk, extrap='edge'
        )
        k2, pk2 = CorrelationToPower(s, ell=0, lowring=False, xy=1.0)(
            xi, extrap='edge'
        )
        # k2 ≈ k and pk2 ≈ pk in the well-sampled mid-k range
    """

    def __init__(
        self,
        s: ArrayLike,
        ell: int = 0,
        q: float = 0.0,
        **kwargs: object,
    ) -> None:
        if isinstance(ell, (bool, np.bool_)):
            raise TypeError("ell must be an int, got bool")
        ell_i = int(ell)
        if ell_i < 0:
            raise ValueError(f"ell must be >= 0, got {ell_i}")
        self.ell = ell_i
        super().__init__(s, _SphericalBesselJKernel(ell_i), q=1.5 + float(q), **kwargs)
        self.padded_prefactor = (
            self.padded_prefactor * self.padded_x**3 * (2 * np.pi) ** 1.5
        )
        phase = (-1) ** (ell_i // 2)
        self.padded_postfactor = self.padded_postfactor * phase
