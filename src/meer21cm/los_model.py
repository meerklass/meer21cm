r"""
Shared 1D line-of-sight transverse projection model.

Both :class:`meer21cm.los.LineOfSightPowerSpectrum` (1D power spectrum along
pixel sightlines) and :class:`meer21cm.stack.Stacking` (3D/1D galaxy--intensity
stacking) project a 3D cross- or auto-power spectrum :math:`P_{3D}(k, \mu)` onto
line-of-sight modes :math:`k_\parallel` via

.. math::

    P_{1D}^{(\Delta i, \Delta j)}(k_\parallel)
    = \frac{1}{2\pi} \int_0^{k_{\perp,\max}} k_\perp\,\mathrm{d}k_\perp\;
      \left[P_{3D}\!\left(\sqrt{k_\perp^2 + k_\parallel^2},\;
            \mu = \frac{k_\parallel}{k}\right) + P_{\rm SN}\right]
      \overline{W}_\times^{(\Delta i, \Delta j)}(k_\perp),

where :math:`\overline{W}_\times^{(\Delta i, \Delta j)}(k_\perp)` is the
band-averaged azimuthal projection of the square-pixel window and dish primary
beam at angular pixel lag :math:`(\Delta i, \Delta j)`.

This module provides:

- The unified transverse window functions (:func:`pixel_window_power`,
  :func:`azimuthal_pixel_window`, :func:`beam_window`,
  :func:`discrete_beam_window`) and channel comoving geometry helpers
  (:func:`channel_comoving_distances`, :func:`channel_comoving_widths`,
  :func:`chi_nodes_and_weights`).
- :class:`OneDimPowerModel`, a lightweight composition helper bound to any
  :class:`meer21cm.model.ModelPowerSpectrum`-like object (including
  :class:`~meer21cm.power.PowerSpectrum`,
  :class:`~meer21cm.mock.MockSimulation`,
  :class:`~meer21cm.los.LineOfSightPowerSpectrum`, and
  :class:`~meer21cm.stack.Stacking`) without duplicating cosmology or survey
  state.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .power_ops import gaussian_beam_attenuation, step_window_attenuation
from .util import center_to_edges, freq_to_redshift

__all__ = [
    "OneDimPowerModel",
    "azimuthal_pixel_window",
    "beam_window",
    "channel_comoving_distances",
    "channel_comoving_widths",
    "chi_nodes_and_weights",
    "discrete_beam_window",
    "pixel_window_power",
]

_MODEL_CACHE_ATTRS = (
    "_kmode",
    "_mumode",
    "_auto_power_matter_model_r",
    "_auto_power_matter_model",
    "_auto_power_tracer_1_model_noobs",
    "_auto_power_tracer_2_model_noobs",
    "_cross_power_tracer_model_noobs",
)


def pixel_window_power(
    kx: ArrayLike, ky: ArrayLike, dperp: float
) -> NDArray[np.floating]:
    r"""
    Power window of a square pixel top-hat on a 2D Fourier grid.

    .. math::

        |W_{\rm pix}(k_x, k_y)|^2
        = \operatorname{sinc}^2\!\left(\frac{k_x D_\perp}{2}\right)
          \operatorname{sinc}^2\!\left(\frac{k_y D_\perp}{2}\right).

    Parameters
    ----------
    kx, ky : ArrayLike
        Transverse wavenumbers, in Mpc^-1.
    dperp : float
        Comoving pixel side length :math:`D_\perp`, in Mpc.

    Returns
    -------
    NDArray[np.floating]
        Squared pixel window with the broadcast shape of ``kx`` and ``ky``.
    """
    dperp_f = float(dperp)
    return step_window_attenuation(
        np.asarray(kx, dtype=float), dperp_f, p=2
    ) * step_window_attenuation(np.asarray(ky, dtype=float), dperp_f, p=2)


def azimuthal_pixel_window(
    kperp: ArrayLike,
    dperp: float,
    di: int = 0,
    dj: int = 0,
    n_phi: int = 64,
    pixel_power: float = 2,
) -> NDArray[np.floating]:
    r"""
    Azimuthally averaged square-pixel window at angular lag :math:`(\Delta i, \Delta j)`.

    Evaluates

    .. math::

        \overline{W}_{\rm pix}^{(\Delta i, \Delta j)}(k_\perp; D_\perp)
        = \left\langle
            \cos\!\big(k_\perp D_\perp(\Delta i\cos\phi + \Delta j\sin\phi)\big)\,
            \left[\operatorname{sinc}\!\left(\frac{k_\perp D_\perp\cos\phi}{2}\right)
                  \operatorname{sinc}\!\left(\frac{k_\perp D_\perp\sin\phi}{2}\right)\right]^p
          \right\rangle_\phi ,

    where :math:`\operatorname{sinc}(x) \equiv \sin(x)/x`.
    Both single-pixel LOS auto-power spectra and sky-cube stacking carry
    ``pixel_power=2`` (:math:`|W_{\rm pix}|^2`).

    Parameters
    ----------
    kperp : ArrayLike
        Transverse wavenumbers, in Mpc^-1.
    dperp : float
        Comoving pixel side length :math:`D_\perp`, in Mpc.
    di, dj : int, default 0
        Integer angular lag offsets in map-pixel units.
    n_phi : int, default 64
        Number of azimuthal quadrature angles on :math:`[0, 2\pi)`.
    pixel_power : int or float, default 2
        Exponent :math:`p` of the pixel top-hat window (``2`` for
        :math:`|W_{\rm pix}|^2`, ``1`` for single-leg diagnostics, ``0`` for
        no pixel smoothing).

    Returns
    -------
    NDArray[np.floating]
        Azimuthally averaged pixel window, shape ``(len(kperp),)``.
    """
    dperp_f = float(dperp)
    if dperp_f < 0.0:
        raise ValueError(f"dperp must be non-negative, got {dperp!r}")
    if int(n_phi) < 4:
        raise ValueError(f"n_phi must be >= 4, got {n_phi!r}")
    p_val = float(pixel_power)
    if p_val < 0.0:
        raise ValueError(f"pixel_power must be non-negative, got {pixel_power!r}")
    kperp_arr = np.atleast_1d(np.asarray(kperp, dtype=float))
    phi = np.linspace(0.0, 2.0 * np.pi, int(n_phi), endpoint=False)
    kx = kperp_arr[:, None] * np.cos(phi)[None, :]
    ky = kperp_arr[:, None] * np.sin(phi)[None, :]
    phase = np.cos(dperp_f * (int(di) * kx + int(dj) * ky))
    if p_val == 0.0:
        return phase.mean(axis=1)
    wpix = step_window_attenuation(kx, dperp_f, p=1) * step_window_attenuation(
        ky, dperp_f, p=1
    )
    if p_val == 2.0:
        wpix = wpix * wpix
    elif p_val != 1.0:
        wpix = np.abs(wpix) ** p_val
    return (phase * wpix).mean(axis=1)


def beam_window(
    kperp: ArrayLike, sigma_b: float, power: float = 2.0
) -> NDArray[np.floating]:
    r"""
    Continuous isotropic Gaussian dish-beam window :math:`\exp(-\tfrac{p}{2} k_\perp^2 \sigma_B^2)`.

    Built directly on :func:`meer21cm.power_ops.gaussian_beam_attenuation`.

    Parameters
    ----------
    kperp : ArrayLike
        Transverse wavenumbers, in Mpc^-1.
    sigma_b : float
        Comoving Gaussian beam standard deviation :math:`\sigma_B`, in Mpc.
    power : float, default 2.0
        Exponent :math:`p` of the beam window: ``2`` for HI auto-power
        (:math:`W_{\rm beam}^2 = \exp(-k_\perp^2 \sigma_B^2)`), ``1`` for
        galaxy--HI stacking or cross-spectra where only one leg is beamed
        (:math:`W_{\rm beam} = \exp(-\tfrac12 k_\perp^2 \sigma_B^2)`), or
        ``0`` for no beam attenuation.

    Returns
    -------
    NDArray[np.floating]
        Gaussian beam window with the same shape as ``kperp``.
    """
    p_val = float(power)
    if p_val < 0.0:
        raise ValueError(f"power must be non-negative, got {power!r}")
    k_arr = np.asarray(kperp, dtype=float)
    if p_val == 0.0:
        return np.ones_like(k_arr)
    g = gaussian_beam_attenuation(k_arr, float(sigma_b))
    if p_val == 1.0:
        return g
    if p_val == 2.0:
        return g * g
    return g**p_val


def discrete_beam_window(
    kperp: ArrayLike,
    dperp: float,
    sigma_b: float,
    di: int = 0,
    dj: int = 0,
    n_phi: int = 64,
    pixel_power: float = 2,
    beam_power: float = 1,
    truncate_sigma: float = 5.0,
) -> NDArray[np.floating]:
    r"""
    Azimuthally averaged square-pixel window times discrete pixel-grid Gaussian beam DFT.

    When a separable sum-normalised 2D Gaussian kernel
    :math:`B[\delta i, \delta j] = b[\delta i]\,b[\delta j]` with
    :math:`b[m] \propto \exp(-m^2 / (2\sigma_{\rm pix}^2))` and
    :math:`\sigma_{\rm pix} = \sigma_B / D_\perp` is convolved directly on an
    already-pixelised map grid, its Fourier response is the anisotropic periodic DFT

    .. math::

        \tilde{B}_{\rm disc}(k_x, k_y)
        = \tilde{b}(k_x D_\perp)\,\tilde{b}(k_y D_\perp),
        \qquad
        \tilde{b}(u) = b[0] + 2\sum_{m=1}^{M} b[m]\cos(m u).

    Because :math:`\tilde{B}_{\rm disc}` is anisotropic on the pixel lattice,
    it is averaged jointly with the lag phase and pixel window over azimuth
    :math:`\phi`.

    Parameters
    ----------
    kperp : ArrayLike
        Transverse wavenumbers, in Mpc^-1.
    dperp : float
        Comoving pixel side length :math:`D_\perp`, in Mpc.
    sigma_b : float
        Comoving Gaussian beam standard deviation :math:`\sigma_B`, in Mpc.
    di, dj : int, default 0
        Integer angular lag offsets in map-pixel units.
    n_phi : int, default 64
        Number of azimuthal quadrature angles.
    pixel_power : int or float, default 2
        Exponent of the square-pixel top-hat window.
    beam_power : int or float, default 1
        Exponent of the discrete beam DFT (``1`` for galaxy--HI stacking,
        ``2`` for HI auto-power).
    truncate_sigma : float, default 5.0
        Kernel truncation half-width in units of :math:`\sigma_{\rm pix}`.

    Returns
    -------
    NDArray[np.floating]
        Azimuthally averaged transverse window, shape ``(len(kperp),)``.
    """
    bp = float(beam_power)
    if bp < 0.0:
        raise ValueError(f"beam_power must be non-negative, got {beam_power!r}")
    dperp_f = float(dperp)
    sigma_f = abs(float(sigma_b))
    if dperp_f <= 0.0 or sigma_f == 0.0 or bp == 0.0:
        return azimuthal_pixel_window(
            kperp, dperp_f, di=di, dj=dj, n_phi=n_phi, pixel_power=pixel_power
        )

    sigma_pix = sigma_f / dperp_f
    half = max(1, int(np.ceil(float(truncate_sigma) * sigma_pix)))
    m_vals = np.arange(-half, half + 1, dtype=float)
    b1d = np.exp(-0.5 * (m_vals / sigma_pix) ** 2)
    b1d /= b1d.sum()
    m_pos = np.arange(1, half + 1, dtype=float)
    b0 = float(b1d[half])
    b_pos = b1d[half + 1 :]

    kperp_arr = np.atleast_1d(np.asarray(kperp, dtype=float))
    phi = np.linspace(0.0, 2.0 * np.pi, int(n_phi), endpoint=False)
    kx = kperp_arr[:, None] * np.cos(phi)[None, :]
    ky = kperp_arr[:, None] * np.sin(phi)[None, :]
    ux = kx * dperp_f
    uy = ky * dperp_f
    bx = b0 + 2.0 * np.sum(
        b_pos[None, None, :] * np.cos(ux[:, :, None] * m_pos[None, None, :]),
        axis=-1,
    )
    by = b0 + 2.0 * np.sum(
        b_pos[None, None, :] * np.cos(uy[:, :, None] * m_pos[None, None, :]),
        axis=-1,
    )
    b_resp = bx * by
    if bp == 2.0:
        b_resp = b_resp * b_resp
    elif bp != 1.0:
        b_resp = np.abs(b_resp) ** bp

    phase = np.cos(int(di) * ux + int(dj) * uy)
    pp = float(pixel_power)
    if pp == 0.0:
        wpix: float | NDArray[np.floating] = 1.0
    else:
        wpix = step_window_attenuation(kx, dperp_f, p=1) * step_window_attenuation(
            ky, dperp_f, p=1
        )
        if pp == 2.0:
            wpix = wpix * wpix
        elif pp != 1.0:
            wpix = np.abs(wpix) ** pp
    return (phase * wpix * b_resp).mean(axis=1)


def channel_comoving_distances(spec) -> NDArray[np.floating]:
    """Comoving distance to each frequency channel centre, in Mpc (fiducial cosmology)."""
    z = np.asarray(spec.z_ch, dtype=float)
    return np.asarray(
        spec.astropy_cosmo_fiducial.comoving_distance(z).to("Mpc").value,
        dtype=float,
    )


def channel_comoving_widths(spec) -> NDArray[np.floating]:
    """Comoving width of each frequency channel, in Mpc (fiducial cosmology).

    Channel frequency edges are derived consistently via
    :func:`meer21cm.util.center_to_edges` and converted to redshift via
    :func:`meer21cm.util.freq_to_redshift`.
    """
    nu = np.asarray(spec.nu, dtype=float)
    if nu.size == 1:
        return np.zeros(1, dtype=float)
    nu_edges = np.asarray(center_to_edges(nu), dtype=float)
    z_edges = freq_to_redshift(nu_edges)
    chi_edges = np.asarray(
        spec.astropy_cosmo_fiducial.comoving_distance(z_edges).to("Mpc").value,
        dtype=float,
    )
    return np.abs(np.diff(chi_edges))


def chi_nodes_and_weights(
    chi_grid: ArrayLike,
    n_chi: int = 5,
    weights: ArrayLike | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Partition a comoving distance grid into ``n_chi`` nodes with normalised weights.

    Parameters
    ----------
    chi_grid : ArrayLike
        1D array of comoving distances, in Mpc (need not be pre-sorted).
    n_chi : int, default 5
        Number of distance nodes across ``[min(chi_grid), max(chi_grid)]``.
    weights : ArrayLike, optional
        Per-sample non-negative weights aligned with ``chi_grid`` (before
        sorting). Defaults to uniform weights.

    Returns
    -------
    nodes : NDArray[np.floating]
        Comoving node centres, shape ``(n_nodes,)``.
    node_weights : NDArray[np.floating]
        Normalised weights summing to 1, shape ``(n_nodes,)``.
    """
    n = int(n_chi)
    if n < 1:
        raise ValueError(f"n_chi must be >= 1, got {n_chi!r}")
    chi_arr = np.asarray(chi_grid, dtype=float).ravel()
    if chi_arr.size == 0:
        raise ValueError("chi_grid must be non-empty")
    order = np.argsort(chi_arr)
    chi_s = chi_arr[order]
    if chi_s.size == 1 or n == 1 or float(chi_s[-1] - chi_s[0]) == 0.0:
        return np.array([float(np.mean(chi_s))], dtype=float), np.array(
            [1.0], dtype=float
        )
    edges = np.linspace(chi_s[0], chi_s[-1], n + 1)
    nodes = 0.5 * (edges[:-1] + edges[1:])
    if weights is None:
        w_s = np.ones(chi_s.size, dtype=float)
    else:
        w_arr = np.asarray(weights, dtype=float).ravel()
        if w_arr.size != chi_s.size:
            raise ValueError(
                f"weights length ({w_arr.size}) must match chi_grid ({chi_s.size})"
            )
        w_s = w_arr[order]
    idx = np.clip(np.digitize(chi_s, edges) - 1, 0, n - 1)
    w_nodes = np.bincount(idx, weights=w_s, minlength=n)[:n].astype(float)
    total = float(w_nodes.sum())
    if total <= 0.0:
        w_nodes = np.ones(n, dtype=float) / float(n)
    else:
        w_nodes = w_nodes / total
    return nodes, w_nodes


class OneDimPowerModel:
    r"""
    Standalone 1D transverse-projection forward model bound to a survey/model object.

    Wraps any :class:`meer21cm.model.ModelPowerSpectrum`-like object (such as
    :class:`~meer21cm.power.PowerSpectrum`,
    :class:`~meer21cm.mock.MockSimulation`,
    :class:`~meer21cm.los.LineOfSightPowerSpectrum`, or
    :class:`~meer21cm.stack.Stacking`) by composition, reading live cosmology,
    frequency channels, pixel area, beam widths, and 3D ``power_kmu`` directly
    from ``spec`` without duplicating state.

    Both :class:`~meer21cm.los.LineOfSightPowerSpectrum` (with ``beam_power=2``
    for auto-power spectra) and :class:`~meer21cm.stack.Stacking` (with
    ``beam_power=1`` and arbitrary angular lags :math:`(\Delta i, \Delta j)`)
    delegate their transverse window and 3D-to-1D projection to this class.

    Parameters
    ----------
    spec : object
        A :class:`~meer21cm.model.ModelPowerSpectrum`-compatible object.
    n_chi : int, default 5
        Number of comoving distance nodes across the band.
    kperp_edges : ArrayLike, optional
        Explicit :math:`k_\perp` bin edges in Mpc^-1.
    n_kperp : int, default 256
        Number of uniform :math:`k_\perp` bins when ``kperp_edges`` is ``None``.
    kperp_max : float, optional
        Upper integration limit in Mpc^-1. Defaults to
        :math:`4\pi / D_\perp(\chi_{\min})`.
    include_pixel_window : bool, default True
        Include the square-pixel top-hat window.
    include_beam_window : bool, default True
        Include the dish primary beam when ``spec.sigma_beam_ch`` is set.
    pixel_power : float, default 2.0
        Exponent of the square-pixel window (``2`` for :math:`|W_{\rm pix}|^2`).
    beam_power : float, default 2.0
        Exponent of the dish beam window (``2`` for HI auto-power, ``1`` for
        galaxy--HI stacking/cross-spectra).
    beam_mode : {'continuous', 'discrete'}, default 'continuous'
        ``'continuous'`` uses the isotropic sky Gaussian beam; ``'discrete'``
        uses the anisotropic periodic DFT of a pixel-grid Gaussian kernel.
    power_kmu_func : callable, optional
        Optional replacement for ``spec.power_kmu``, called as
        ``power_kmu_func(k_mesh, mu_mesh)``.
    enforce_first_zero : bool, default False
        If True, raise ``ValueError`` when ``kperp_max`` lies below the first
        pixel-window zero :math:`2\pi / D_\perp(\chi_{\min})`.
    """

    def __init__(
        self,
        spec,
        *,
        n_chi: int = 5,
        kperp_edges: ArrayLike | None = None,
        n_kperp: int = 256,
        kperp_max: float | None = None,
        include_pixel_window: bool = True,
        include_beam_window: bool = True,
        pixel_power: float = 2.0,
        beam_power: float = 2.0,
        beam_mode: str = "continuous",
        power_kmu_func: Callable[[NDArray, NDArray], NDArray] | None = None,
        enforce_first_zero: bool = False,
    ):
        if int(n_chi) < 1:
            raise ValueError(f"n_chi must be >= 1, got {n_chi!r}")
        if int(n_kperp) < 8:
            raise ValueError("n_kperp must be at least 8")
        if str(beam_mode) not in ("continuous", "discrete"):
            raise ValueError(
                f"beam_mode must be 'continuous' or 'discrete', got {beam_mode!r}"
            )
        edges_arr = (
            None if kperp_edges is None else np.asarray(kperp_edges, dtype=float)
        )
        if edges_arr is not None and (edges_arr.ndim != 1 or edges_arr.size < 2):
            raise ValueError("kperp_edges must be a 1-D array with at least two edges")

        self.spec = spec
        self.n_chi = int(n_chi)
        self._kperp_edges = edges_arr
        self.n_kperp = int(n_kperp)
        self.kperp_max = None if kperp_max is None else float(kperp_max)
        self.include_pixel_window = bool(include_pixel_window)
        self.include_beam_window = bool(include_beam_window)
        self.pixel_power = float(pixel_power)
        self.beam_power = float(beam_power)
        self.beam_mode = str(beam_mode)
        self.power_kmu_func = power_kmu_func
        self.enforce_first_zero = bool(enforce_first_zero)

    @property
    def chi_ch(self) -> NDArray[np.floating]:
        """Comoving distance to each channel centre of ``spec``, in Mpc."""
        return channel_comoving_distances(self.spec)

    @property
    def dchi_ch(self) -> NDArray[np.floating]:
        """Comoving width of each channel of ``spec``, in Mpc."""
        return channel_comoving_widths(self.spec)

    def kperp_edges_for(self, chi_min: float | None = None) -> NDArray[np.floating]:
        """Return :math:`k_\\perp` bin edges, optionally anchored to ``chi_min``."""
        edges = self._kperp_edges
        if edges is None:
            chi_ref = float(np.min(self.chi_ch)) if chi_min is None else float(chi_min)
            dperp_min = np.deg2rad(float(self.spec.pix_resol)) * chi_ref
            kmax = self.kperp_max
            if kmax is None:
                kmax = 4.0 * np.pi / dperp_min
            else:
                kmax = float(kmax)
                if not np.isfinite(kmax) or kmax <= 0.0:
                    raise ValueError(f"kperp_max must be positive, got {kmax!r}")
                if self.enforce_first_zero and kmax <= 2.0 * np.pi / dperp_min:
                    raise ValueError(
                        "kperp_max is below the first zero of the pixel "
                        f"window ({2 * np.pi / dperp_min:.4f} Mpc^-1); the "
                        "k_perp integral would be truncated."
                    )
            edges = np.linspace(0.0, kmax, self.n_kperp + 1)
        if np.any(np.diff(edges) <= 0.0):
            raise ValueError("kperp_edges must be strictly increasing")
        return np.asarray(edges, dtype=float)

    @property
    def kperp_edges(self) -> NDArray[np.floating]:
        """Edges of the :math:`k_\\perp` integration bins, in Mpc^-1."""
        return self.kperp_edges_for()

    @property
    def kperp(self) -> NDArray[np.floating]:
        """Centres of the :math:`k_\\perp` integration bins, in Mpc^-1."""
        edges = self.kperp_edges
        return 0.5 * (edges[:-1] + edges[1:])

    @property
    def dk_perp(self) -> NDArray[np.floating]:
        """Widths of the :math:`k_\\perp` integration bins, in Mpc^-1."""
        return np.diff(self.kperp_edges)

    def chi_nodes(
        self,
        chi_grid: ArrayLike | None = None,
        weights: ArrayLike | None = None,
        n_chi: int | None = None,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Comoving distance nodes and normalised weights across ``chi_grid``."""
        grid = self.chi_ch if chi_grid is None else chi_grid
        return chi_nodes_and_weights(
            grid,
            n_chi=self.n_chi if n_chi is None else int(n_chi),
            weights=weights,
        )

    def _sorted_beam_mpc(
        self,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]] | None:
        """Return ``(chi_sorted, sigma_b_mpc_sorted)`` or ``None`` if no beam is set."""
        if getattr(self.spec, "sigma_beam_ch", None) is None:
            return None
        chi_all = self.chi_ch
        sigma_arr = np.atleast_1d(
            np.asarray(self.spec.sigma_beam_ch_in_mpc, dtype=float)
        )
        if sigma_arr.size != chi_all.size:
            sigma_arr = np.full(chi_all.size, float(sigma_arr.flat[0]), dtype=float)
        order = np.argsort(chi_all)
        return chi_all[order], np.abs(sigma_arr[order])

    def transverse_window(
        self,
        di: int = 0,
        dj: int = 0,
        *,
        chi_grid: ArrayLike | None = None,
        node_weights: ArrayLike | None = None,
        n_chi: int | None = None,
        n_phi: int = 64,
        include_pixel_window: bool | None = None,
        include_beam_window: bool | None = None,
        pixel_power: float | None = None,
        beam_power: float | None = None,
        beam_mode: str | None = None,
        kperp: ArrayLike | None = None,
    ) -> NDArray[np.floating]:
        r"""
        Band-averaged transverse window :math:`\overline{W}_\times^{(\Delta i,\Delta j)}(k_\perp)`.

        Parameters
        ----------
        di, dj : int, default 0
            Integer angular lag offsets in map-pixel units.
        chi_grid : ArrayLike, optional
            Comoving distance grid to partition into ``n_chi`` nodes. Defaults
            to :attr:`chi_ch`.
        node_weights : ArrayLike, optional
            Per-sample weights aligned with ``chi_grid`` (e.g. taper squared
            for LOS power or zero-lag channel pair weights for stacking).
        n_chi : int, optional
            Override :attr:`n_chi`.
        n_phi : int, default 64
            Number of azimuthal quadrature angles.
        include_pixel_window, include_beam_window : bool, optional
            Override the instance flags.
        pixel_power, beam_power : float, optional
            Override the pixel and beam exponents.
        beam_mode : {'continuous', 'discrete'}, optional
            Override :attr:`beam_mode`.
        kperp : ArrayLike, optional
            Explicit :math:`k_\perp` grid; defaults to :attr:`kperp`.

        Returns
        -------
        NDArray[np.floating]
            Band-averaged transverse window, shape ``(len(kperp),)``.
        """
        use_pix = (
            self.include_pixel_window
            if include_pixel_window is None
            else bool(include_pixel_window)
        )
        use_beam = (
            self.include_beam_window
            if include_beam_window is None
            else bool(include_beam_window)
        )
        pix_pow = self.pixel_power if pixel_power is None else float(pixel_power)
        b_pow = self.beam_power if beam_power is None else float(beam_power)
        b_mode = self.beam_mode if beam_mode is None else str(beam_mode)
        if b_mode not in ("continuous", "discrete"):
            raise ValueError(
                f"beam_mode must be 'continuous' or 'discrete', got {b_mode!r}"
            )
        eff_pix_pow = pix_pow if use_pix else 0.0

        kperp_arr = (
            self.kperp
            if kperp is None
            else np.atleast_1d(np.asarray(kperp, dtype=float))
        )
        nodes, weights = self.chi_nodes(
            chi_grid=chi_grid, weights=node_weights, n_chi=n_chi
        )
        theta_pix = np.deg2rad(float(self.spec.pix_resol))

        beam_info = self._sorted_beam_mpc() if (use_beam and b_pow > 0.0) else None

        acc = np.zeros_like(kperp_arr, dtype=float)
        for node, wt in zip(nodes, weights):
            dperp = theta_pix * float(node)
            if beam_info is not None and b_mode == "discrete":
                chi_for_beam, sigma_b = beam_info
                sb = float(np.interp(node, chi_for_beam, sigma_b))
                w_node = discrete_beam_window(
                    kperp_arr,
                    dperp,
                    sb,
                    di=di,
                    dj=dj,
                    n_phi=n_phi,
                    pixel_power=eff_pix_pow,
                    beam_power=b_pow,
                )
            else:
                if use_pix or int(di) != 0 or int(dj) != 0:
                    w_node = azimuthal_pixel_window(
                        kperp_arr,
                        dperp,
                        di=di,
                        dj=dj,
                        n_phi=n_phi,
                        pixel_power=eff_pix_pow,
                    )
                else:
                    w_node = np.ones_like(kperp_arr, dtype=float)
                if beam_info is not None:
                    chi_for_beam, sigma_b = beam_info
                    sb = float(np.interp(node, chi_for_beam, sigma_b))
                    w_node = w_node * beam_window(kperp_arr, sb, power=b_pow)
            acc += wt * w_node
        return acc

    def build_mesh(
        self,
        kpar: ArrayLike,
        *,
        kperp: ArrayLike | None = None,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Build the ``(n_kperp, n_kpar)`` ``(k, mu)`` mesh without mutating ``spec``."""
        kpar_arr = np.atleast_1d(np.asarray(kpar, dtype=float)).ravel()
        if kpar_arr.size == 0:
            raise ValueError("kpar must be non-empty")
        kperp_arr = (
            self.kperp
            if kperp is None
            else np.atleast_1d(np.asarray(kperp, dtype=float)).ravel()
        )
        k_mesh = np.sqrt(kperp_arr[:, None] ** 2 + kpar_arr[None, :] ** 2)
        mu_mesh = np.broadcast_to(kpar_arr[None, :], k_mesh.shape) / np.maximum(
            k_mesh, 1e-12
        )
        return k_mesh, mu_mesh

    def eval_power_kmu(
        self,
        kpar: ArrayLike,
        *,
        which: str = "cross",
        include_mean_amp: bool = True,
        cross_shot_noise: float | Callable[[NDArray], ArrayLike] = 0.0,
        preserve_cache: bool = True,
        kperp: ArrayLike | None = None,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
        """
        Evaluate 3D ``P(k, mu)`` on the ``(k_perp, k_par)`` integration mesh.

        Parameters
        ----------
        kpar : ArrayLike
            Line-of-sight wavenumbers in Mpc^-1.
        which : {'cross', 'auto_1', 'auto_2'}, default 'cross'
            Model spectrum passed to ``spec.power_kmu``.
        include_mean_amp : bool, default True
            Multiply by ``mean_amp`` factors.
        cross_shot_noise : float or callable, default 0.0
            Optional shot noise added to ``P(k, mu)``.
        preserve_cache : bool, default True
            When True, save and restore ``spec._kmode``, ``spec._mumode``, and
            dependent cached model arrays so evaluating the 1D projection does
            not clobber a 3D :class:`~meer21cm.power.PowerSpectrum` box cache.
            When False, leave ``k_mesh`` and ``mu_mesh`` stored on ``spec``.
        kperp : ArrayLike, optional
            Explicit :math:`k_\\perp` centres; defaults to :attr:`kperp`.

        Returns
        -------
        k_mesh, mu_mesh, power : tuple of ndarray
            Meshes and evaluated 3D power, each of shape ``(n_kperp, n_kpar)``.
        """
        kpar_arr = np.atleast_1d(np.asarray(kpar, dtype=float)).ravel()
        k_mesh, mu_mesh = self.build_mesh(kpar_arr, kperp=kperp)

        if not preserve_cache:
            self.spec.kmode = k_mesh
            self.spec.mumode = mu_mesh
            if self.power_kmu_func is not None:
                power = np.asarray(self.power_kmu_func(k_mesh, mu_mesh), dtype=float)
            else:
                power = np.asarray(
                    self.spec.power_kmu(which=which, include_mean_amp=include_mean_amp),
                    dtype=float,
                )
        elif self.power_kmu_func is not None:
            power = np.asarray(self.power_kmu_func(k_mesh, mu_mesh), dtype=float)
        else:
            saved = {
                name: getattr(self.spec, name, None) for name in _MODEL_CACHE_ATTRS
            }
            try:
                self.spec.kmode = k_mesh
                self.spec.mumode = mu_mesh
                power = np.asarray(
                    self.spec.power_kmu(which=which, include_mean_amp=include_mean_amp),
                    dtype=float,
                )
            finally:
                for name, val in saved.items():
                    setattr(self.spec, name, val)

        if power.shape != k_mesh.shape:
            src = (
                "power_kmu_func / power_kmu"
                if self.power_kmu_func is not None
                else "power_kmu"
            )
            raise ValueError(
                f"{src} returned shape {power.shape}, expected {k_mesh.shape}"
            )
        if callable(cross_shot_noise):
            power = power + np.asarray(cross_shot_noise(kpar_arr), dtype=float)[None, :]
        elif float(cross_shot_noise) != 0.0:
            power = power + float(cross_shot_noise)
        return k_mesh, mu_mesh, power

    def model_1d_power(
        self,
        kpar: ArrayLike,
        di: int = 0,
        dj: int = 0,
        *,
        which: str = "cross",
        include_mean_amp: bool = True,
        cross_shot_noise: float | Callable[[NDArray], ArrayLike] = 0.0,
        preserve_cache: bool = True,
        chi_min: float | None = None,
        **window_kwargs,
    ) -> NDArray[np.floating]:
        r"""
        Evaluate the transverse-projected 1D power :math:`P_{1D}^{(\Delta i,\Delta j)}(k_\parallel)`.

        Parameters
        ----------
        kpar : ArrayLike
            Line-of-sight wavenumbers in Mpc^-1.
        di, dj : int, default 0
            Integer angular lag offsets in map-pixel units.
        which : {'cross', 'auto_1', 'auto_2'}, default 'cross'
            Model spectrum passed to ``spec.power_kmu``.
        include_mean_amp : bool, default True
            Multiply by ``mean_amp`` factors.
        cross_shot_noise : float or callable, default 0.0
            Optional shot noise added to ``P(k, mu)``.
        preserve_cache : bool, default True
            Save and restore ``spec``'s ``kmode``/``mumode`` caches.
        chi_min : float, optional
            Override minimum comoving distance for the default :math:`k_\perp`
            grid.
        **window_kwargs
            Forwarded to :meth:`transverse_window`.

        Returns
        -------
        NDArray[np.floating]
            Projected 1D power spectrum, shape ``(len(kpar),)``.
        """
        edges = self.kperp_edges_for(chi_min=chi_min)
        kperp = 0.5 * (edges[:-1] + edges[1:])
        dk_perp = np.diff(edges)
        _km, _mm, power = self.eval_power_kmu(
            kpar,
            which=which,
            include_mean_amp=include_mean_amp,
            cross_shot_noise=cross_shot_noise,
            preserve_cache=preserve_cache,
            kperp=kperp,
        )
        w_perp = self.transverse_window(di=di, dj=dj, kperp=kperp, **window_kwargs)
        measure = (kperp * dk_perp / (2.0 * np.pi))[:, None]
        return np.sum(measure * power * w_perp[:, None], axis=0)
