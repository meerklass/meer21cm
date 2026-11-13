r"""
One-dimensional line-of-sight power spectrum of an intensity map.

This module provides the estimator and the forward model for
:math:`P_{1D}(k_\parallel)`, the power spectrum measured along the line of
sight in each pixel column of a map cube.  It is the one-dimensional analogue
of :class:`meer21cm.power.PowerSpectrum`: the measurement and the model live
in one class.

Estimator
---------
A sightline sampled on a uniform comoving grid ``chi_j = chi_0 + j * dchi``,
``j = 0, ..., N-1``, over a band of length ``L = N * dchi``, is mean-subtracted,
tapered with a periodic window ``w`` and Fourier transformed::

    F_m = sum_j w_j (delta_j - <delta>) exp(-2 pi i j m / N)

(the ``numpy.fft.rfft`` convention, i.e. no ``1/N``).  The power is

.. math::

    \hat P_{1D}(k_m) = \frac{\Delta\chi_u}{N\,\overline{w^2}}\,|F_m|^2,
    \qquad k_m = \frac{2\pi m}{L},

which is an unbiased estimator of the continuous quantity

.. math::

    P_{1D}(k) = \frac{1}{L}\left\langle |\tilde\delta(k)|^2 \right\rangle,
    \qquad \tilde\delta(k) = \int_0^L \mathrm{d}\chi\, e^{ik\chi}\delta(\chi).

The prefactor is the only addition to the numpy convention: ``dchi / N`` turns
the sum into an integral and ``1 / <w^2>`` undoes the taper.  Note that
``<w^2> = 3/8`` for a periodic Hann window, not ``<w>^2 = 1/4``.

Model
-----
The one-dimensional spectrum is the transverse projection of the
three-dimensional power, attenuated by the observation.  For an isotropic
``P_3D`` the azimuthal angle enters only through the window, so the integral
over transverse modes reduces to a single integral over ``k_perp``::

    P_1D(k_par) = 1/(2 pi) int_0^inf k_perp dk_perp P_3D(k, mu)
                  * < W_perp^2(k_perp, chi) W_beam^2(k_perp, chi) >_chi
                  * W_par^2(k_par)

with ``k = sqrt(k_perp^2 + k_par^2)``, ``mu = k_par / k``.  ``P_3D`` comes from
:meth:`meer21cm.model.ModelPowerSpectrum.power_kmu` evaluated on the mesh
stored in ``kmode`` / ``mumode``; ``W_perp^2`` is the azimuthal average of the
pixel top-hat at ``D_perp = sqrt(pixel_area) * chi``; ``W_beam^2`` is a Gaussian
beam; and the angular brackets are an average over the band weighted by the
estimator taper (both windows grow with comoving distance).  ``W_par`` is the
radial window of the channel average and the regridding, supplied by
:class:`meer21cm.transfer.RadialTransferFunction`.

The ``k_perp`` integration grid is a user choice (``kperp_edges``): it is
completely independent of any simulation box or Cartesian grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .model import ModelPowerSpectrum
from .transfer import RadialTransferFunction
from .util import f_21, periodic_hann, taper_values

__all__ = [
    "LineOfSightMeasurement",
    "LineOfSightPowerSpectrum",
    "azimuthal_pixel_window_power",
    "beam_window_power",
    "estimate_1d_power",
    "hann_window",
    "pixel_window_power",
    "resample_sightlines",
    "uniform_comoving_grid",
]

_DEG2RAD = np.pi / 180.0

LOSWhich = str
"""Tracer combination: ``"auto_1"``, ``"auto_2"`` or ``"cross"``."""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def hann_window(n: int) -> NDArray[np.floating]:
    """
    Periodic Hann window.

    Parameters
    ----------
    n : int
        Number of samples.

    Returns
    -------
    NDArray[np.floating]
        ``w_j = 0.5 * (1 - cos(2 pi j / n))``, with ``<w^2> = 3/8``.
    """
    return periodic_hann(n)


def _taper_values(
    taper: bool | Callable[[int], ArrayLike], n: int
) -> NDArray[np.floating]:
    """Resolve a taper specification into ``n`` samples."""
    return taper_values(taper, n)


def uniform_comoving_grid(
    chi_ch: ArrayLike, over: int = 2
) -> tuple[NDArray[np.floating], float]:
    """
    Uniform comoving grid spanning the channel range.

    Parameters
    ----------
    chi_ch : ArrayLike
        Comoving channel centres, in Mpc.
    over : int, default 2
        Oversampling of the uniform grid with respect to the channel spacing.

    Returns
    -------
    chi_u : NDArray[np.floating]
        Ascending uniform grid, in Mpc.
    dchi_u : float
        Spacing of ``chi_u``, in Mpc.
    """
    chi = np.sort(np.asarray(chi_ch, dtype=float).ravel())
    if chi.size < 2:
        raise ValueError("chi_ch must contain at least two channels")
    dchi = float(np.median(np.diff(chi)))
    if not np.isfinite(dchi) or dchi <= 0:
        raise ValueError("chi_ch must be strictly increasing")
    if over < 1:
        raise ValueError("over must be >= 1")
    dchi_u = dchi / float(over)
    n = int(np.floor((chi[-1] - chi[0]) / dchi_u)) + 1
    return chi[0] + np.arange(n) * dchi_u, dchi_u


def resample_sightlines(
    cube: ArrayLike, chi_in: ArrayLike, chi_u: ArrayLike
) -> NDArray[np.floating]:
    """
    Linearly interpolate sightlines onto a uniform comoving grid.

    Parameters
    ----------
    cube : ArrayLike
        Array of shape ``(..., n_chan)``; the last axis is the radial axis.
    chi_in : ArrayLike
        Comoving positions of the input samples, in Mpc, shape ``(n_chan,)``.
    chi_u : ArrayLike
        Target uniform grid, in Mpc.

    Returns
    -------
    NDArray[np.floating]
        Array of shape ``(..., len(chi_u))``.
    """
    data = np.asarray(cube, dtype=float)
    chi_in_arr = np.asarray(chi_in, dtype=float).ravel()
    chi_u_arr = np.asarray(chi_u, dtype=float).ravel()
    if data.shape[-1] != chi_in_arr.size:
        raise ValueError(
            f"last axis of cube ({data.shape[-1]}) must match chi_in ({chi_in_arr.size})"
        )
    if np.any(np.diff(chi_u_arr) <= 0):
        raise ValueError("chi_u must be strictly ascending")
    order = np.argsort(chi_in_arr)
    chi_s = chi_in_arr[order]
    if np.any(np.diff(chi_s) <= 0):
        raise ValueError("chi_in must be strictly monotonic")
    flat = np.asarray(data[..., order], dtype=float).reshape(-1, chi_s.size)
    out = np.empty((flat.shape[0], chi_u_arr.size), dtype=float)
    for i in range(flat.shape[0]):
        out[i] = np.interp(chi_u_arr, chi_s, flat[i])
    return out.reshape(data.shape[:-1] + (chi_u_arr.size,))


def _sinc(x: ArrayLike) -> NDArray[np.floating]:
    r"""``sin(x)/x`` with the ``x -> 0`` limit (``np.sinc(x / pi)``)."""
    return np.sinc(np.asarray(x, dtype=float) / np.pi)


def pixel_window_power(
    kx: ArrayLike, ky: ArrayLike, dperp: float
) -> NDArray[np.floating]:
    r"""
    Power window of a square pixel top-hat on the box Fourier grid.

    .. math::

        |W_\perp|^2 = \mathrm{sinc}^2(k_x D_\perp / 2)\,
                      \mathrm{sinc}^2(k_y D_\perp / 2)

    Parameters
    ----------
    kx, ky : ArrayLike
        Transverse wavenumbers, in Mpc^-1.
    dperp : float
        Comoving side of the pixel, in Mpc.

    Returns
    -------
    NDArray[np.floating]
        The power window, with the broadcast shape of ``kx`` and ``ky``.
    """
    return (
        _sinc(np.asarray(kx, dtype=float) * dperp / 2.0) ** 2
        * _sinc(np.asarray(ky, dtype=float) * dperp / 2.0) ** 2
    )


def beam_window_power(kperp: ArrayLike, sigma_b: float) -> NDArray[np.floating]:
    r"""
    Power window of an achromatic Gaussian beam.

    Parameters
    ----------
    kperp : ArrayLike
        Transverse wavenumber, in Mpc^-1.
    sigma_b : float
        Gaussian beam width in comoving units, in Mpc.

    Returns
    -------
    NDArray[np.floating]
        ``exp(-k_perp^2 sigma_b^2)``.
    """
    return np.exp(-(np.asarray(kperp, dtype=float) ** 2) * float(sigma_b) ** 2)


def azimuthal_pixel_window_power(
    kperp: ArrayLike, dperp: float, n_phi: int = 64
) -> NDArray[np.floating]:
    r"""
    Azimuthally averaged power window of a square pixel top-hat.

    .. math::

        \langle \mathrm{sinc}^2(k_x D_\perp / 2)\,
        \mathrm{sinc}^2(k_y D_\perp / 2) \rangle_\phi ,
        \qquad k_x = k_\perp\cos\phi,\; k_y = k_\perp\sin\phi .

    This is the correct transverse window for the one-dimensional
    :math:`k_\perp` integral: for an isotropic :math:`P_{3D}` the cosine
    depends on :math:`k_\perp` only, so the azimuthal integral acts on the
    window alone.

    Parameters
    ----------
    kperp : ArrayLike
        Transverse wavenumber, in Mpc^-1.
    dperp : float
        Comoving side of the pixel, in Mpc.
    n_phi : int, default 64
        Number of azimuthal samples.

    Returns
    -------
    NDArray[np.floating]
        The averaged power window, shape ``(len(kperp),)``.
    """
    kperp_arr = np.atleast_1d(np.asarray(kperp, dtype=float))
    u = kperp_arr * float(dperp)
    phi = np.linspace(0.0, 2.0 * np.pi, int(n_phi), endpoint=False)
    ux = u[:, None] * np.cos(phi)[None, :]
    uy = u[:, None] * np.sin(phi)[None, :]
    w = _sinc(0.5 * ux) ** 2 * _sinc(0.5 * uy) ** 2
    return w.mean(axis=1)


# ---------------------------------------------------------------------------
# measurement container
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LineOfSightMeasurement:
    """
    Binned one-dimensional power from :meth:`LineOfSightPowerSpectrum.measure_1d_power`.

    Attributes
    ----------
    k1deff : NDArray[np.floating]
        Effective wavenumber centre of each bin, in Mpc^-1.
    power1d : NDArray[np.floating]
        Binned power, averaged over sightlines, in Mpc.
    power1d_err : NDArray[np.floating]
        Standard error of the sightline mean.
    nmodes : NDArray[np.integer]
        Number of FFT modes contributing to each bin.
    k1dbins : NDArray[np.floating]
        Bin edges, in Mpc^-1, shape ``(nk + 1,)``.
    n_sightlines : int
        Number of sightlines used.
    which : str
        Tracer combination that was measured.
    per_sightline : NDArray[np.floating] | None
        Per-sightline binned power, shape ``(n_sightlines, nk)``, if requested.
    """

    k1deff: NDArray[np.floating]
    power1d: NDArray[np.floating]
    power1d_err: NDArray[np.floating]
    nmodes: NDArray[np.integer]
    k1dbins: NDArray[np.floating]
    n_sightlines: int
    which: str = "auto_1"
    per_sightline: NDArray[np.floating] | None = None


def _weighted_mean_and_error(
    values: NDArray[np.floating], weights: NDArray[np.floating] | None
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Weighted mean along axis 0 and the standard error of that mean."""
    vals = np.asarray(values, dtype=float)
    if weights is None:
        n = vals.shape[0]
        mean = vals.mean(axis=0)
        err = vals.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
        return mean, err
    w = np.asarray(weights, dtype=float).ravel()
    if w.size != vals.shape[0]:
        raise ValueError("weights must have one entry per sightline")
    sw = w.sum()
    if sw <= 0:
        raise ValueError("weights must sum to a positive value")
    mean = (w[:, None] * vals).sum(axis=0) / sw
    var = (w[:, None] * (vals - mean[None, :]) ** 2).sum(axis=0) / sw
    n_eff = max(1.0, float(np.count_nonzero(w)))
    return mean, np.sqrt(var / n_eff)


# ---------------------------------------------------------------------------
# estimator
# ---------------------------------------------------------------------------
def estimate_1d_power(
    data_1: ArrayLike,
    dchi_u: float,
    kpar_edges: ArrayLike,
    *,
    data_2: ArrayLike | None = None,
    weights_1: ArrayLike | None = None,
    weights_2: ArrayLike | None = None,
    taper: bool | Callable[[int], ArrayLike] = True,
    which: str = "auto_1",
) -> LineOfSightMeasurement:
    r"""
    Measure :math:`P_{1D}(k_\parallel)` along each axis of a sightline array.

    Parameters
    ----------
    data_1 : ArrayLike
        Samples of the first map on a uniform comoving grid, shape
        ``(n_sightline, n_chan)``.
    dchi_u : float
        Spacing of the uniform comoving grid, in Mpc.
    kpar_edges : ArrayLike
        Bin edges in ``k_parallel``, in Mpc^-1.
    data_2 : ArrayLike, default None
        Second map, needed for ``which="auto_2"`` or ``"cross"``.
    weights_1, weights_2 : ArrayLike, default None
        Per-sightline weights.
    taper : bool or callable, default True
        ``True`` applies a periodic Hann window, ``False`` none, and a
        callable ``f(n) -> (n,) array`` is used as given.
    which : {'auto_1', 'auto_2', 'cross'}, default "auto_1"
        Which spectrum to measure.

    Returns
    -------
    LineOfSightMeasurement
        The binned measurement.

    Notes
    -----
    The normalisation is ``dchi_u / (N * <w^2>)`` per sightline, which makes
    the estimator unbiased for a white field of variance ``sigma^2`` (it
    returns ``sigma^2 * dchi_u``).
    """
    which = str(which).lower()
    if which not in ("auto_1", "auto_2", "cross"):
        raise ValueError("which must be 'auto_1', 'auto_2' or 'cross'")
    d1 = np.asarray(data_1, dtype=float)
    if d1.ndim != 2:
        raise ValueError(f"data_1 must be 2-D (n_sightline, n_chan), got {d1.shape}")
    n = d1.shape[-1]
    if which in ("auto_2", "cross"):
        if data_2 is None:
            raise ValueError(f"data_2 is required for which={which!r}")
        d2 = np.asarray(data_2, dtype=float)
        if d2.shape != d1.shape:
            raise ValueError(
                f"data_2 must have the same shape as data_1 {d1.shape}, got {d2.shape}"
            )
    else:
        d2 = None

    w = _taper_values(taper, n)
    norm = float(dchi_u) / n / float(np.mean(w**2))

    f1 = np.fft.rfft((d1 - d1.mean(axis=-1, keepdims=True)) * w, axis=-1)
    if which == "auto_1":
        spec = np.abs(f1) ** 2
        wts = weights_1
    elif which == "auto_2":
        f2 = np.fft.rfft((d2 - d2.mean(axis=-1, keepdims=True)) * w, axis=-1)
        spec = np.abs(f2) ** 2
        wts = weights_2
    else:
        f2 = np.fft.rfft((d2 - d2.mean(axis=-1, keepdims=True)) * w, axis=-1)
        spec = np.real(f1 * np.conj(f2))
        wts = None if weights_1 is None else weights_1

    k = 2.0 * np.pi * np.fft.rfftfreq(n, d=float(dchi_u))
    pk_all = norm * spec

    edges = np.asarray(kpar_edges, dtype=float)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("kpar_edges must be a 1-D array with at least two edges")
    nk = edges.size - 1
    per_sight = np.full((d1.shape[0], nk), np.nan, dtype=float)
    k1deff = np.zeros(nk, dtype=float)
    nmodes = np.zeros(nk, dtype=np.int64)
    for i in range(nk):
        sel = (k >= edges[i]) & (k < edges[i + 1])
        nmodes[i] = int(np.count_nonzero(sel))
        if nmodes[i] == 0:
            continue
        per_sight[:, i] = pk_all[:, sel].mean(axis=1)
        k1deff[i] = k[sel].mean()

    wts_arr = None if wts is None else np.asarray(wts, dtype=float).ravel()
    if wts_arr is not None:
        good = np.isfinite(wts_arr) & (wts_arr > 0)
    else:
        good = np.ones(d1.shape[0], dtype=bool)
    if not np.any(good):
        raise ValueError("no sightline with a positive weight")
    mean, err = _weighted_mean_and_error(
        per_sight[good], None if wts_arr is None else wts_arr[good]
    )
    return LineOfSightMeasurement(
        k1deff=k1deff,
        power1d=mean,
        power1d_err=err,
        nmodes=nmodes,
        k1dbins=edges,
        n_sightlines=int(np.count_nonzero(good)),
        which=which,
        per_sightline=per_sight,
    )


# ---------------------------------------------------------------------------
# combined estimator + model
# ---------------------------------------------------------------------------
class LineOfSightPowerSpectrum(ModelPowerSpectrum):
    r"""
    Estimator and forward model for the one-dimensional line-of-sight power.

    This is the one-dimensional analogue of
    :class:`meer21cm.power.PowerSpectrum`: the measurement
    (:meth:`measure_1d_power`) and the windowed model (:meth:`model_1d_power`)
    live in one object, so the two can be compared directly.

    The map is held in ``data`` (aliased as ``data_1``) with shape
    ``(n_sightline, n_chan)`` for HEALPix or ``(ny, nx, n_chan)`` for WCS; it
    is flattened internally, so both backends are supported.  A second map may
    be attached as ``data_2`` to measure auto_2 and cross spectra.

    Parameters
    ----------
    data : ArrayLike, default None
        The map cube.  Stored on ``self.data`` / ``self.data_1``.
    data_2 : ArrayLike, default None
        Optional second map cube for ``which="auto_2"`` or ``"cross"``.
    k1dbins : ArrayLike, default None
        Bin edges in ``k_parallel``, in Mpc^-1.  Defaults to one FFT mode per
        bin, from ``m = 2`` (the first mode above the DC hole) upwards.
    taper : bool or callable, default True
        ``True`` applies a periodic Hann window to each sightline, ``False``
        none, and a callable ``f(n) -> (n,) array`` is used as given.
    over : int, default 2
        Oversampling of the uniform comoving grid relative to the channel
        spacing.
    n_chi : int, default 5
        Number of comoving radii at which the transverse window is evaluated
        before averaging over the band.
    kperp_edges : ArrayLike, default None
        Edges of the :math:`k_\\perp` integration bins, in Mpc^-1.  Defaults to
        ``n_kperp`` linear bins from 0 to ``kperp_max``.
    n_kperp : int, default 256
        Number of bins when ``kperp_edges`` is not given.
    kperp_max : float, default None
        Upper edge of the default grid, in Mpc^-1.  Defaults to
        :math:`4\\pi/D_\\perp(\\chi_{\\min})`, i.e. twice the first zero of the
        pixel window at the near end of the band.
    include_pixel_window : bool, default True
        Include the pixel top-hat in the model.
    include_beam_window : bool, default True
        Include the Gaussian beam in the model.  Silently ignored when
        ``sigma_beam_ch`` is not set.
    radial_transfer : RadialTransferFunction or {"auto"}, default None
        Radial window of the channel average and regridding.  ``None``
        corresponds to an ideal radial response; ``"auto"`` builds one from
        this object's own channel/comoving geometry.
    power_kmu_func : callable, default None
        Optional replacement for :meth:`power_kmu`, called as
        ``power_kmu_func(k, mu)``.  Useful when the model power is not the
        analytic one (for example, the measured power of the realisation).
    **params
        Passed to :class:`meer21cm.model.ModelPowerSpectrum` (and through it to
        :class:`meer21cm.dataanalysis.Specification`): ``nu``, ``skymap`` or
        ``hp_nside``, ``sigma_beam_ch``, ``tracer_bias_1``, ``kaiser_rsd``, ...

    Examples
    --------
    >>> los = LineOfSightPowerSpectrum.from_power_spectrum(mock, data=map_cube)
    >>> measurement = los.measure_1d_power()
    >>> model = los.model_1d_power(kpar=measurement.k1deff)
    """

    def __init__(
        self,
        data: ArrayLike | None = None,
        *,
        data_2: ArrayLike | None = None,
        k1dbins: ArrayLike | None = None,
        taper: bool | Callable[[int], ArrayLike] = True,
        over: int = 2,
        n_chi: int = 5,
        kperp_edges: ArrayLike | None = None,
        n_kperp: int = 256,
        kperp_max: float | None = None,
        include_pixel_window: bool = True,
        include_beam_window: bool = True,
        radial_transfer: RadialTransferFunction | str | None = None,
        power_kmu_func: Callable[[NDArray, NDArray], NDArray] | None = None,
        **params,
    ):
        ModelPowerSpectrum.__init__(self, data=data, **params)

        self.k1dbins = None if k1dbins is None else np.asarray(k1dbins, dtype=float)
        self.taper = taper
        self.over = int(over)
        self.n_chi = int(n_chi)
        if int(n_kperp) < 8:
            raise ValueError("n_kperp must be at least 8")
        self.n_kperp = int(n_kperp)
        self.kperp_max = None if kperp_max is None else float(kperp_max)
        self._kperp_edges = (
            None if kperp_edges is None else np.asarray(kperp_edges, dtype=float)
        )
        if self._kperp_edges is not None and (
            self._kperp_edges.ndim != 1 or self._kperp_edges.size < 2
        ):
            raise ValueError("kperp_edges must be a 1-D array with at least two edges")
        self.include_pixel_window = bool(include_pixel_window)
        self.include_beam_window = bool(include_beam_window)
        if isinstance(radial_transfer, str) and radial_transfer != "auto":
            raise ValueError(
                "radial_transfer must be a RadialTransferFunction, None or 'auto'"
            )
        self.radial_transfer = radial_transfer
        self._auto_radial_transfer = None
        self.power_kmu_func = power_kmu_func

        self.data_2 = None if data_2 is None else np.asarray(data_2, dtype=float)

        self._transverse_window_cache = None

    # -- data aliases --------------------------------------------------
    @property
    def data_1(self) -> NDArray[np.floating]:
        """The first map cube; alias of :attr:`data`."""
        return self.data

    @data_1.setter
    def data_1(self, value: ArrayLike) -> None:
        self.data = value

    # -- comoving geometry --------------------------------------------
    @property
    def chi_ch(self) -> NDArray[np.floating]:
        """Comoving distance to each channel centre, in Mpc (fiducial cosmology)."""
        if getattr(self, "_chi_ch_cache", None) is None:
            z = np.asarray(self.z_ch, dtype=float)
            self._chi_ch_cache = (
                self.astropy_cosmo_fiducial.comoving_distance(z).to("Mpc").value
            )
        return self._chi_ch_cache

    @property
    def dchi_ch(self) -> NDArray[np.floating]:
        """Comoving width of each channel, in Mpc."""
        if getattr(self, "_dchi_ch_cache", None) is None:
            nu = np.asarray(self.nu, dtype=float)
            dnu = float(self.freq_resol)
            z_hi = f_21 / (nu - 0.5 * dnu) - 1.0
            z_lo = f_21 / (nu + 0.5 * dnu) - 1.0
            chi = self.astropy_cosmo_fiducial.comoving_distance(
                np.concatenate([z_hi, z_lo])
            ).value
            self._dchi_ch_cache = np.abs(chi[: nu.size] - chi[nu.size :])
        return self._dchi_ch_cache

    @property
    def chi_u(self) -> NDArray[np.floating]:
        """Uniform comoving grid on which the sightlines are analysed, in Mpc."""
        if getattr(self, "_chi_u_cache", None) is None:
            self._chi_u_cache, self._dchi_u_cache = uniform_comoving_grid(
                self.chi_ch, over=self.over
            )
        return self._chi_u_cache

    @property
    def dchi_u(self) -> float:
        """Spacing of :attr:`chi_u`, in Mpc."""
        _ = self.chi_u
        return float(self._dchi_u_cache)

    # -- map handling ---------------------------------------------------
    def _map_for(self, which: str) -> NDArray[np.floating]:
        """Return the map cube for ``which``, flattened to ``(n_sight, n_chan)``."""
        which = str(which).lower()
        if which == "auto_1":
            cube = np.asarray(self.data, dtype=float)
            wcube = getattr(self, "weights_map_pixel", None)
        elif which in ("auto_2", "cross"):
            if self.data_2 is None:
                raise ValueError(
                    f"data_2 is required for which={which!r}; set "
                    "LineOfSightPowerSpectrum.data_2 first."
                )
            cube = np.asarray(self.data_2, dtype=float)
            wcube = None
        else:
            raise ValueError("which must be 'auto_1', 'auto_2' or 'cross'")
        if cube.ndim < 2:
            raise ValueError(
                f"map cube must have at least 2 dimensions, got {cube.shape}"
            )
        return cube, wcube

    def _flatten_map(
        self, which: str
    ) -> tuple[NDArray[np.floating], NDArray[np.floating] | None]:
        """Flatten a map cube to sightlines and build the row mask."""
        cube, wcube = self._map_for(which)
        flat = cube.reshape(-1, cube.shape[-1])
        finite = np.all(np.isfinite(flat), axis=-1)
        if wcube is not None:
            wflat = np.asarray(wcube, dtype=float).reshape(flat.shape[0], -1)
            if wflat.shape[-1] == flat.shape[-1]:
                w_row = wflat.mean(axis=-1)
            else:  # pragma: no cover - defensive
                w_row = wflat.reshape(-1)[: flat.shape[0]]
            good = finite & (w_row > 0)
            return flat, np.where(good, w_row, 0.0)
        return flat, np.where(finite, 1.0, 0.0)

    def uniform_map(self, which: str = "auto_1") -> NDArray[np.floating]:
        """
        The map resampled onto the uniform comoving grid.

        Parameters
        ----------
        which : {'auto_1', 'auto_2'}, default "auto_1"
            Which map to resample.

        Returns
        -------
        NDArray[np.floating]
            Shape ``(n_sightline, len(chi_u))``.
        """
        flat, _ = self._flatten_map(which)
        return resample_sightlines(flat, self.chi_ch, self.chi_u)

    def _default_k1dbins(self) -> NDArray[np.floating]:
        """One FFT mode per bin, starting at ``m = 2``."""
        k = 2.0 * np.pi * np.fft.rfftfreq(self.chi_u.size, d=self.dchi_u)
        if k.size < 3:
            raise ValueError("not enough samples along the line of sight for binning")
        return k[2 : min(k.size, 39)]

    # -- estimator ------------------------------------------------------
    def measure_1d_power(
        self,
        which: str = "auto_1",
        k1dbins: ArrayLike | None = None,
        per_sightline: bool = False,
    ) -> LineOfSightMeasurement:
        """
        Measure :math:`P_{1D}(k_\\parallel)` from the map cube.

        Parameters
        ----------
        which : {'auto_1', 'auto_2', 'cross'}, default "auto_1"
            Spectrum to measure.  ``auto_2`` and ``cross`` require ``data_2``.
        k1dbins : ArrayLike, default None
            Bin edges in Mpc^-1.  Defaults to ``self.k1dbins``, else to one FFT
            mode per bin starting from ``m = 2``.
        per_sightline : bool, default False
            Keep the per-sightline binned spectra in the result.

        Returns
        -------
        LineOfSightMeasurement
            The binned measurement.
        """
        which = str(which).lower()
        d1, w1 = self._flatten_map("auto_1")
        if which in ("auto_2", "cross"):
            d2, w2 = self._flatten_map("auto_2")
        else:
            d2, w2 = None, None

        d1u = resample_sightlines(d1, self.chi_ch, self.chi_u)
        d2u = None if d2 is None else resample_sightlines(d2, self.chi_ch, self.chi_u)
        if which == "auto_2":
            weights = w2
        elif which == "cross":
            weights = np.minimum(w1, w2)
        else:
            weights = w1

        edges = (
            np.asarray(k1dbins, dtype=float)
            if k1dbins is not None
            else (self.k1dbins if self.k1dbins is not None else self._default_k1dbins())
        )
        result = estimate_1d_power(
            d1u,
            self.dchi_u,
            edges,
            data_2=d2u,
            weights_1=weights,
            weights_2=weights,
            taper=self.taper,
            which=which,
        )
        if not per_sightline:
            object.__setattr__(result, "per_sightline", None)
        return result

    # -- model ----------------------------------------------------------
    @property
    def kperp(self) -> NDArray[np.floating]:
        """Centres of the :math:`k_\\perp` integration bins, in Mpc^-1."""
        edges = self.kperp_edges
        return 0.5 * (edges[:-1] + edges[1:])

    @property
    def dk_perp(self) -> NDArray[np.floating]:
        """Widths of the :math:`k_\\perp` integration bins, in Mpc^-1."""
        return np.diff(self.kperp_edges)

    @property
    def kperp_edges(self) -> NDArray[np.floating]:
        """Edges of the :math:`k_\\perp` integration bins, in Mpc^-1."""
        if getattr(self, "_kperp_edges_cache", None) is None:
            edges = self._kperp_edges
            if edges is None:
                dperp_min = (
                    np.sqrt(float(self.pixel_area)) * _DEG2RAD * float(self.chi_u.min())
                )
                kmax = self.kperp_max
                if kmax is None:
                    kmax = 4.0 * np.pi / dperp_min
                else:
                    kmax = float(kmax)
                    if kmax <= 2.0 * np.pi / dperp_min:
                        raise ValueError(
                            "kperp_max is below the first zero of the pixel "
                            f"window ({2 * np.pi / dperp_min:.4f} Mpc^-1); the "
                            "k_perp integral would be truncated."
                        )
                edges = np.linspace(0.0, kmax, self.n_kperp + 1)
            if np.any(np.diff(edges) <= 0):
                raise ValueError("kperp_edges must be strictly increasing")
            self._kperp_edges_cache = np.asarray(edges, dtype=float)
        return self._kperp_edges_cache

    def _chi_nodes(self) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Comoving radii across the band and their taper weights."""
        chi_u = self.chi_u
        n = self.n_chi
        edges = np.linspace(chi_u.min(), chi_u.max(), n + 1)
        nodes = 0.5 * (edges[:-1] + edges[1:])
        w2 = _taper_values(self.taper, chi_u.size) ** 2
        idx = np.clip(np.digitize(chi_u, edges) - 1, 0, n - 1)
        weights = np.bincount(idx, weights=w2, minlength=n)[:n].astype(float)
        if weights.sum() <= 0:
            weights = np.ones(n)
        return nodes, weights / weights.sum()

    def transverse_window(self) -> NDArray[np.floating]:
        r"""
        Band-averaged transverse power window on the :math:`k_\\perp` grid.

        Returns
        -------
        NDArray[np.floating]
            :math:`\langle \overline{W_\perp^2}\, W_{\\rm beam}^2 \rangle_\chi`,
            shape ``(n_kperp,)``.
        """
        if self._transverse_window_cache is not None:
            return self._transverse_window_cache

        kperp = self.kperp
        nodes, weights = self._chi_nodes()

        sigma_b = None
        if (
            self.include_beam_window
            and getattr(self, "sigma_beam_ch", None) is not None
        ):
            sigma_b = np.atleast_1d(np.asarray(self.sigma_beam_ch_in_mpc, dtype=float))
            if sigma_b.size != np.size(self.chi_ch):
                sigma_b = np.full(np.size(self.chi_ch), float(sigma_b.flat[0]))
            sigma_b = np.abs(sigma_b)

        acc = np.zeros_like(kperp)
        for node, wt in zip(nodes, weights):
            w_node = np.ones_like(kperp)
            if self.include_pixel_window:
                dperp = np.sqrt(float(self.pixel_area)) * _DEG2RAD * float(node)
                w_node = w_node * azimuthal_pixel_window_power(kperp, dperp)
            if sigma_b is not None:
                sb = float(np.interp(node, self.chi_ch, sigma_b))
                w_node = w_node * beam_window_power(kperp, sb)
            acc += wt * w_node

        self._transverse_window_cache = acc
        return acc

    def build_model_mesh(
        self, kpar: ArrayLike
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        r"""
        Populate ``kmode`` / ``mumode`` with the ``(k_perp, k_parallel)`` mesh.

        The mesh has shape ``(n_kperp, len(kpar))`` with
        ``k = sqrt(k_perp^2 + k_par^2)`` and ``mu = k_par / k``.  The
        ``kmode`` / ``mumode`` setters clear the dependent model caches, so
        the subsequent :meth:`~meer21cm.model.ModelPowerSpectrum.power_kmu`
        call is evaluated on this mesh.  The mesh is left on the object.

        Parameters
        ----------
        kpar : ArrayLike
            Wavenumbers along the line of sight, in Mpc^-1.

        Returns
        -------
        k_mesh, mu_mesh : NDArray[np.floating]
            Arrays of shape ``(n_kperp, len(kpar))``.
        """
        kpar_arr = np.atleast_1d(np.asarray(kpar, dtype=float)).ravel()
        k_mesh = np.sqrt(self.kperp[:, None] ** 2 + kpar_arr[None, :] ** 2)
        mu_mesh = np.broadcast_to(kpar_arr[None, :], k_mesh.shape) / np.maximum(
            k_mesh, 1e-12
        )
        self.kmode = k_mesh
        self.mumode = mu_mesh
        return self.kmode, self.mumode

    def model_1d_power(
        self, which: str = "auto_1", kpar: ArrayLike | None = None
    ) -> NDArray[np.floating]:
        r"""
        Windowed model :math:`P_{1D}(k_\parallel)`.

        Evaluates the one-dimensional transverse integral

        .. math::

            P_{1D}(k_\parallel) = \sum_i \frac{k_{\perp,i}\Delta k_{\perp,i}}{2\pi}
            P_{3D}(k_i,\mu_i)\,\overline{W^{2}_i}\,|W_\parallel|^{2}

        on the grid given by :attr:`kperp_edges`.

        Parameters
        ----------
        which : {'auto_1', 'auto_2', 'cross'}, default "auto_1"
            Which model spectrum to evaluate.  ``auto_2`` and ``cross``
            require ``data_2`` to be set.
        kpar : ArrayLike, default None
            Wavenumbers at which to evaluate the model, in Mpc^-1.  Defaults to
            the sample points of the default bins, i.e. one FFT mode per bin.

        Returns
        -------
        NDArray[np.floating]
            The model power, shape ``(len(kpar),)``.
        """
        which = str(which).lower()
        if which not in ("auto_1", "auto_2", "cross"):
            raise ValueError("which must be 'auto_1', 'auto_2' or 'cross'")
        if which in ("auto_2", "cross") and self.data_2 is None:
            raise ValueError(
                f"data_2 is required for which={which!r}; set "
                "LineOfSightPowerSpectrum.data_2 first."
            )
        if kpar is None:
            if self.k1dbins is None:
                kpar_arr = self._default_k1dbins()[:-1]
            else:
                edges = np.asarray(self.k1dbins, dtype=float)
                kpar_arr = 0.5 * (edges[:-1] + edges[1:])
        else:
            kpar_arr = np.atleast_1d(np.asarray(kpar, dtype=float)).ravel()
        if kpar_arr.size == 0:
            raise ValueError("kpar must be non-empty")

        k_mesh, mu_mesh = self.build_model_mesh(kpar_arr)
        if self.power_kmu_func is not None:
            power = np.asarray(self.power_kmu_func(k_mesh, mu_mesh), dtype=float)
        else:
            power = self.power_kmu(which=which, include_mean_amp=True)
        if power.shape != k_mesh.shape:
            raise ValueError(
                f"power_kmu returned shape {power.shape}, expected {k_mesh.shape}"
            )

        weight = (self.kperp * self.dk_perp / (2.0 * np.pi))[:, None]
        p1d = np.sum(weight * power * self.transverse_window()[:, None], axis=0)

        radial = self._radial_window()
        if radial is not None:
            p1d = p1d * np.asarray(radial.transfer(kpar_arr), dtype=float)
        return p1d

    def _radial_window(self) -> RadialTransferFunction | None:
        """The radial transfer, building it from the survey geometry for "auto"."""
        if self.radial_transfer is None:
            return None
        if not isinstance(self.radial_transfer, str):
            return self.radial_transfer
        if self._auto_radial_transfer is None:
            self._auto_radial_transfer = RadialTransferFunction(
                self.chi_u,
                self.dchi_u,
                chi_ch=self.chi_ch,
                dchi_ch=self.dchi_ch,
                # a few samples per channel so that the average is converged
                dchi_vox=float(np.median(self.dchi_ch)) / 4.0,
                taper=self.taper,
            )
        return self._auto_radial_transfer

    # -- convenience ----------------------------------------------------
    def run(
        self, which: str = "auto_1", **kwargs
    ) -> tuple[LineOfSightMeasurement, NDArray[np.floating]]:
        """
        Measure the data and evaluate the model at the same wavenumbers.

        Parameters
        ----------
        which : {'auto_1', 'auto_2', 'cross'}, default "auto_1"
            Spectrum to use.
        **kwargs
            Forwarded to :meth:`measure_1d_power`.

        Returns
        -------
        measurement : LineOfSightMeasurement
            The measured spectrum.
        model : NDArray[np.floating]
            The model at ``measurement.k1deff``.
        """
        measurement = self.measure_1d_power(which=which, **kwargs)
        model = self.model_1d_power(which=which, kpar=measurement.k1deff)
        return measurement, model

    @classmethod
    def from_power_spectrum(
        cls,
        ps,
        data: ArrayLike | None = None,
        data_2: ArrayLike | None = None,
        **kwargs,
    ) -> "LineOfSightPowerSpectrum":
        """
        Build from a :class:`~meer21cm.power.PowerSpectrum`-like object.

        Copies the survey geometry, beam, model and bin settings so that the
        line-of-sight analysis matches the object the map was produced with.

        Parameters
        ----------
        ps : object
            A :class:`~meer21cm.power.PowerSpectrum`,
            :class:`~meer21cm.mock.MockSimulation` or compatible object.
        data : ArrayLike, default None
            Map cube to attach.  Defaults to ``ps.data``.
        data_2 : ArrayLike, default None
            Optional second map cube.
        **kwargs
            Constructor overrides.

        Returns
        -------
        LineOfSightPowerSpectrum
        """
        params = {}
        skymap = getattr(ps, "skymap", None)
        if skymap is not None:
            params["skymap"] = skymap
        for key in (
            "nu",
            "sigma_beam_ch",
            "beam_model",
            "beam_unit",
            "ra_range",
            "dec_range",
            "precision",
            "batch_number",
            "mean_amp_1",
            "mean_amp_2",
            "sigma_v_1",
            "sigma_v_2",
            "tracer_bias_1",
            "tracer_bias_2",
            "omega_hi",
            "fog_profile",
            "cross_coeff",
            "kaiser_rsd",
            "fiducial_cosmology",
            "true_cosmology",
            "backend",
        ):
            if key in kwargs:
                continue
            val = getattr(ps, key, None)
            if val is not None:
                params[key] = val
        if "k1dbins" not in kwargs:
            k1dbins = getattr(ps, "k1dbins", None)
            if k1dbins is not None:
                params["k1dbins"] = k1dbins
        if data is None:
            data = getattr(ps, "data", None)
        params.update(kwargs)
        return cls(data=data, data_2=data_2, **params)
