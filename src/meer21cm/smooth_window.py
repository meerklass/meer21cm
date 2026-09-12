r"""
Discrete-shell survey-window matrix for theory multipoles.

Builds a dense matrix that maps continuous theory multipoles
:math:`P_{\ell'}(k_{\mathrm{in}})` to estimator bin multipoles
:math:`P_\ell(k_{\mathrm{out}})`. Each output row averages over discrete
FFT modes in a :math:`k`-shell using the same projector as
:meth:`~meer21cm.estimator.FieldPowerSpectrum.measure_multipoles`:

.. math::

    W_{\ell i,\,\ell' j}
    =
    \sum_{\mathbf{k}_n\in S_i}
    B_{\ell,i}(\mathbf{k}_n)
    \sum_{L}
    \mathcal{L}_L(\mu_n)\,
    W_{L\ell'}\bigl(|\mathbf{k}_n|,\,k'_j\bigr).

Notation
--------
- :math:`i` — estimator output bin index (``k_out``; edges from
  :class:`~meer21cm.estimator.MultipoleShellMap`).
- :math:`j` — theory input node index (``k_in`` / :math:`k'_j`).
- :math:`S_i` — discrete Fourier modes with
  :math:`|\mathbf{k}_n|` in bin :math:`i` (shell membership from
  :attr:`~meer21cm.estimator.MultipoleShellMap.bin_index`).
- :math:`\mu_n` — LOS cosine of mode :math:`\mathbf{k}_n`
  (:attr:`~meer21cm.estimator.MultipoleShellMap.mu`).
- :math:`B_{\ell,i}(\mathbf{k}_n)` — multipole binning weight matching
  :meth:`~meer21cm.estimator.FieldPowerSpectrum.measure_multipoles`:

  .. math::

      B_{\ell,i}(\mathbf{k}_n)
      =
      \frac{w_n}{U_i}\,
      (2\ell+1)\,\mathcal{L}_\ell(\mu_n),

  where :math:`w_n` is the per-mode weight and
  :math:`U_i=\sum_{\mathbf{k}_m\in S_i} w_m` normalises the shell average.
- :math:`L` — continuous multipoles used to rebuild anisotropic power at a
  fixed :math:`|\mathbf{k}|`,
  :math:`P(\mathbf{k})=\sum_L P_L(|k|)\,\mathcal{L}_L(\mu)`.
- :math:`W_{L\ell'}(k,k')` — continuous response kernel (see below).

The continuous kernel :math:`W_{L\ell'}(k,k')` may be:

- **identity** — :math:`\delta_{L\ell'}\delta(k-k')` (no survey convolution;
  only discrete :math:`\mu`-selection on the FFT grid);
- **smooth** — Hankel / Wigner response from measured window multipoles
  :math:`W_L(k)` (selection field or randoms; pypower-style smooth window):

  .. math::

      Q_L(s)
      =
      \frac{i^{L}}{2\pi^{2}}
      \int_0^{\infty}\!\mathrm{d}q\,q^{2}\,
      j_{L}(qs)\,W_L(q),

  .. math::

      W_{L\ell'}(k,k')
      =
      \frac{2}{\pi}
      (-1)^{L/2}(-1)^{\ell'/2}
      \int_0^{\infty}\!\mathrm{d}s\,s^{2}\,
      j_{L}(ks)\,j_{\ell'}(k's)
      \sum_{\mathcal{L}}
      C_{L\ell'\mathcal{L}}\,
      Q_{\mathcal{L}}(s),

  with Wigner–Legendre couplings :math:`C_{L\ell'\mathcal{L}}` from
  :func:`wigner3j_square` (``prefactor=True``). In code the
  :math:`k'^2\mathrm{d}k'` volume element is included when building the
  discrete matrix columns.

Default 3D modelling via :func:`~meer21cm.power_ops.get_modelpk_conv` is
unchanged. Local Yamamoto LOS will reuse the same shell map once
implemented on the estimator.

References
----------
- Beutler et al., 2019, MNRAS, 484, 2233 (arXiv:1810.05051)
- Wilson et al., 2017, MNRAS, 464, 3121
- cosmodesi/pypower smooth-window / FFT-window implementations
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import interpolate
from scipy.special import eval_legendre, spherical_jn

from .estimator import MultipoleShellMap
from .fftlog import CorrelationToPower, PowerToCorrelation
from .util import legendre_polynomial_with_factor

WindowEllMap = Mapping[int, ArrayLike]


# ---------------------------------------------------------------------------
# Legendre / Wigner coupling
# ---------------------------------------------------------------------------


def wigner3j_square(
    ellout: int, ellin: int, prefactor: bool = True
) -> tuple[list[int], list[float]]:
    r"""
    Coefficients for the product of two Legendre polynomials.

    Encodes

    .. math::

        \mathcal{L}_\ell(\mu)\,\mathcal{L}_{\ell'}(\mu)
        = \sum_L C_{\ell\ell'L}\,\mathcal{L}_L(\mu)

    (Wilson et al. / Beutler et al., arXiv:1810.05051).
    """
    ellout_i = int(ellout)
    ellin_i = int(ellin)

    def G(p: int) -> tuple[int, int]:
        toret = 1
        for i in range(1, p + 1):
            toret *= 2 * i - 1
        return toret, math.factorial(p)

    qvals: list[int] = []
    coeffs: list[float] = []
    for p in range(min(ellin_i, ellout_i) + 1):
        numer: list[float] = []
        denom: list[float] = []
        for r in [G(ellout_i - p), G(p), G(ellin_i - p)]:
            numer.append(r[0])
            denom.append(r[1])
        a, b = G(ellin_i + ellout_i - p)
        numer.append(b)
        denom.append(a)
        numer.append(2 * (ellin_i + ellout_i) - 4 * p + 1)
        denom.append(2 * (ellin_i + ellout_i) - 2 * p + 1)
        q = ellin_i + ellout_i - 2 * p
        if prefactor:
            numer.append(2 * ellout_i + 1)
            denom.append(2 * q + 1)
        coeffs.append(float(np.prod(numer, dtype="f8") / np.prod(denom, dtype="f8")))
        qvals.append(q)
    return qvals[::-1], coeffs[::-1]


def _weights_trapz(x: ArrayLike) -> NDArray[np.floating]:
    """Trapezoidal quadrature weights for integrating :math:`f(x)\\,\\mathrm{d}x`."""
    x_np = np.asarray(x, dtype=float)
    w = np.empty_like(x_np)
    w[0] = (x_np[1] - x_np[0]) / 2.0
    w[-1] = (x_np[-1] - x_np[-2]) / 2.0
    w[1:-1] = (x_np[2:] - x_np[:-2]) / 2.0
    return w


def _legendre_plain(ell: int, mu: ArrayLike) -> NDArray[np.floating]:
    """Plain :math:`\\mathcal{L}_\\ell(\\mu)` (no :math:`2\\ell+1` factor)."""
    return np.asarray(eval_legendre(int(ell), np.asarray(mu, dtype=float)), dtype=float)


def _legendre_with_factor(ell: int, mu: ArrayLike) -> NDArray[np.floating]:
    """:math:`(2\\ell+1)\\mathcal{L}_\\ell(\\mu)`, matching the estimator."""
    return np.asarray(
        np.poly1d(legendre_polynomial_with_factor(int(ell)))(
            np.asarray(mu, dtype=float)
        ),
        dtype=float,
    )


# ---------------------------------------------------------------------------
# Hankel helpers
# ---------------------------------------------------------------------------


def power_to_correlation_multipole(
    k: ArrayLike,
    pk: ArrayLike,
    ell: int = 0,
    q: float = 0.0,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Hankel transform :math:`P_\\ell(k) \\to \\xi_\\ell(s)` on a log-spaced ``k`` grid."""
    k_np = np.asarray(k, dtype=float)
    pk_np = np.asarray(pk, dtype=float)
    if k_np.size < 8:
        raise ValueError("Need a fine log-spaced k grid for FFTLog")
    transform = PowerToCorrelation(k_np, ell=ell, q=q, lowring=False, xy=1.0)
    s, xi = transform(pk_np, extrap="edge")
    return s, xi


def correlation_to_power_multipole(
    s: ArrayLike,
    xi: ArrayLike,
    ell: int = 0,
    q: float = 0.0,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Hankel transform :math:`\\xi_\\ell(s) \\to P_\\ell(k)` on a log-spaced ``s`` grid."""
    s_np = np.asarray(s, dtype=float)
    xi_np = np.asarray(xi, dtype=float)
    transform = CorrelationToPower(s_np, ell=ell, q=q, lowring=False, xy=1.0)
    k, pk = transform(xi_np, extrap="edge")
    return k, pk


def interpolate_window_to_log_k(
    k_meas: ArrayLike,
    w_ell: ArrayLike,
    k_log: ArrayLike,
) -> NDArray[np.floating]:
    """Interpolate a measured window multipole onto a log-spaced ``k`` grid."""
    k_meas_np = np.asarray(k_meas, dtype=float)
    w_ell_np = np.asarray(w_ell, dtype=float)
    k_log_np = np.asarray(k_log, dtype=float)
    mask = np.isfinite(k_meas_np) & np.isfinite(w_ell_np) & (k_meas_np > 0)
    if mask.sum() < 2:
        return np.zeros_like(k_log_np)
    order = np.argsort(k_meas_np[mask])
    kk = k_meas_np[mask][order]
    ww = w_ell_np[mask][order]
    uniq, indx = np.unique(kk, return_index=True)
    ww = ww[indx]
    spline = interpolate.interp1d(
        uniq,
        ww,
        kind="linear",
        bounds_error=False,
        fill_value=(ww[0], 0.0),
    )
    return np.asarray(spline(k_log_np), dtype=float)


def build_config_space_window_coupling(
    sep: ArrayLike,
    window_s: Mapping[int, ArrayLike],
    ell_out: int,
    ell_in: int,
) -> NDArray[np.floating]:
    r"""
    Configuration-space window coupling
    :math:`W_{\ell\ell'}(s) = \sum_L C_{\ell\ell'L}\, Q_L(s)`.
    """
    ells_L, coeffs = wigner3j_square(ell_out, ell_in, prefactor=True)
    sep_np = np.asarray(sep)
    block = np.zeros(len(sep_np), dtype=float)
    for L, coeff in zip(ells_L, coeffs):
        if L not in window_s:
            continue
        block += coeff * np.asarray(window_s[L], dtype=float)
    return block


# ---------------------------------------------------------------------------
# Discrete-shell window matrix
# ---------------------------------------------------------------------------


@dataclass
class DiscreteShellWindowMatrix:
    """
    Dense multipole window matrix plus bin metadata.

    Maps concatenated theory multipoles on ``k_in`` to estimator bins
    ``k_out`` (see :func:`build_discrete_shell_window_matrix`).

    Attributes
    ----------
    matrix : ndarray, shape ``(n_ell * n_out, n_ell * n_in)``
        Acts on concatenated ``[P_ell[0](k_in), P_ell[1](k_in), ...]``.
    k_in : ndarray
        Theory wavenumber nodes.
    k_out : ndarray
        Estimator bin centres (effective ``k``).
    nmodes : ndarray
        Modes per output bin (from the shell map).
    ells : tuple of int
        Multipole block order.
    """

    matrix: NDArray[np.floating]
    k_in: NDArray[np.floating]
    k_out: NDArray[np.floating]
    nmodes: NDArray[np.floating]
    ells: tuple[int, ...]

    def apply(
        self, p_ell_in: Mapping[int, ArrayLike] | ArrayLike
    ) -> dict[int, NDArray[np.floating]]:
        """Apply this matrix; see :func:`apply_discrete_shell_window_matrix`."""
        return apply_discrete_shell_window_matrix(p_ell_in, self.matrix, ells=self.ells)


def continuous_window_response_blocks(
    k_window: ArrayLike,
    W_ell: WindowEllMap,
    k_eval: ArrayLike,
    k_in: ArrayLike,
    ells_out: Sequence[int],
    ells_in: Sequence[int],
    n_fftlog: int = 512,
    k_log_min: float | None = None,
    k_log_max: float | None = None,
    q: float = 0.0,
) -> dict[tuple[int, int], NDArray[np.floating]]:
    r"""
    Continuous smooth-window response blocks :math:`W_{L\ell'}(k_{\mathrm{eval}}, k_{\mathrm{in}})`.

    Each block has shape ``(len(k_eval), len(k_in))`` and includes the
    discrete :math:`k_{\mathrm{in}}` volume element
    :math:`\Delta V(k_{\mathrm{in}})=\mathrm{trapz}(k_{\mathrm{in}}^3)/3`
    once (the FFTlog :math:`k` grid is interpolated as a density before
    that volume weight is applied).
    """
    ells_out_t = tuple(int(e) for e in ells_out)
    ells_in_t = tuple(int(e) for e in ells_in)
    k_eval_np = np.asarray(k_eval, dtype=float)
    k_in_np = np.asarray(k_in, dtype=float)
    k_window_np = np.asarray(k_window, dtype=float)

    finite = np.isfinite(k_window_np) & (k_window_np > 0)
    if k_log_min is None:
        k_log_min = max(float(np.min(k_window_np[finite])) * 0.5, 1e-4)
    if k_log_max is None:
        k_log_max = float(np.max(k_window_np[finite])) * 1.5
    k_log = np.geomspace(k_log_min, k_log_max, n_fftlog)

    needed_L: set[int] = set()
    for ell_out in ells_out_t:
        for ell_in in ells_in_t:
            Ls, _ = wigner3j_square(ell_out, ell_in)
            needed_L.update(Ls)
    needed_L.update(ells_out_t)
    needed_L.update(int(L) for L in W_ell.keys())

    W_k: dict[int, NDArray[np.floating]] = {}
    for L in sorted(needed_L):
        if L in W_ell:
            W_k[L] = interpolate_window_to_log_k(k_window_np, W_ell[L], k_log)
        else:
            W_k[L] = np.zeros_like(k_log)

    window_s: dict[int, NDArray[np.floating]] = {}
    sep_ref: NDArray[np.floating] | None = None
    for L, wk in W_k.items():
        sep, xi = power_to_correlation_multipole(k_log, wk, ell=L, q=q)
        if sep_ref is None:
            sep_ref = sep
        window_s[L] = xi
    assert sep_ref is not None
    sep = sep_ref

    dk_vol_in = _weights_trapz(k_in_np**3) / 3.0
    blocks: dict[tuple[int, int], NDArray[np.floating]] = {}

    n_eval = len(k_eval_np)
    n_in = len(k_in_np)
    slab = max(1, min(n_eval, 32))

    for ell_out in ells_out_t:
        for ell_in in ells_in_t:
            coupling_s = build_config_space_window_coupling(
                sep, window_s, ell_out, ell_in
            )
            block = np.zeros((n_eval, n_in), dtype=float)
            for i0 in range(0, n_eval, slab):
                i1 = min(n_eval, i0 + slab)
                k_slice = k_eval_np[i0:i1]
                tmp = spherical_jn(ell_out, k_slice[:, None] * sep) * coupling_s
                fftlog = CorrelationToPower(sep, ell=ell_in, q=q, lowring=False, xy=1.0)
                xin_rows: list[NDArray[np.floating]] = []
                transformed_rows: list[NDArray[np.floating]] = []
                for row in tmp:
                    kk, pk = fftlog(row, extrap="edge")
                    xin_rows.append(kk)
                    transformed_rows.append(pk)
                xin = xin_rows[0]
                transformed = np.asarray(transformed_rows)
                prefactor = 1.0 / (2 * np.pi**2) * ((-1) ** (ell_out // 2))
                # Density W(k_out, k') on the FFTlog grid; apply the k_in
                # volume element once after interpolating onto theory nodes.
                # (Applying trapz(xin^3)/3 here *and* dk_vol_in double-counts.)
                dens = np.real(prefactor * transformed)
                for j, row in enumerate(dens):
                    interp = interpolate.interp1d(
                        xin,
                        row,
                        kind="linear",
                        bounds_error=False,
                        fill_value=0.0,
                    )
                    block[i0 + j, :] = interp(k_in_np) * dk_vol_in
            blocks[(int(ell_out), int(ell_in))] = block
    return blocks


def _linear_interpolation_matrix(
    k_eval: ArrayLike, k_in: ArrayLike
) -> NDArray[np.floating]:
    """
    Rows are linear-interpolation stencils: ``(M @ f)[i] ≈ f(k_eval[i])``.

    Endpoints clamp; non-finite or non-positive ``k_eval`` rows are zero.
    """
    k_eval_np = np.asarray(k_eval, dtype=float)
    k_in_np = np.asarray(k_in, dtype=float)
    n_eval = len(k_eval_np)
    n_in = len(k_in_np)
    mat = np.zeros((n_eval, n_in), dtype=float)
    if n_in == 0:
        return mat
    order = np.argsort(k_in_np)
    k_sorted = k_in_np[order]
    for i, k in enumerate(k_eval_np):
        if not np.isfinite(k) or k <= 0:
            continue
        if k <= k_sorted[0]:
            mat[i, order[0]] = 1.0
            continue
        if k >= k_sorted[-1]:
            mat[i, order[-1]] = 1.0
            continue
        j = int(np.searchsorted(k_sorted, k))
        k0 = k_sorted[j - 1]
        k1 = k_sorted[j]
        t = 0.0 if k1 <= k0 else (k - k0) / (k1 - k0)
        mat[i, order[j - 1]] = 1.0 - t
        mat[i, order[j]] = t
    return mat


def identity_window_response_blocks(
    k_eval: ArrayLike,
    k_in: ArrayLike,
    ells_out: Sequence[int],
    ells_in: Sequence[int],
) -> dict[tuple[int, int], NDArray[np.floating]]:
    r"""
    Identity continuous blocks :math:`W_{L\ell'}=\delta_{L\ell'}\delta(k-k')`.

    Implemented as linear interpolation from ``k_in`` onto ``k_eval`` on the
    diagonal multipole blocks (no :math:`k^2\mathrm{d}k` weight — the stencil
    already maps :math:`P_{\ell'}(k_{\mathrm{in}})\to P_{\ell'}(k_{\mathrm{eval}})`).
    """
    ells_out_t = tuple(int(e) for e in ells_out)
    ells_in_t = tuple(int(e) for e in ells_in)
    interp = _linear_interpolation_matrix(k_eval, k_in)
    zeros = np.zeros_like(interp)
    blocks: dict[tuple[int, int], NDArray[np.floating]] = {}
    for L in ells_out_t:
        for ell_in in ells_in_t:
            blocks[(int(L), int(ell_in))] = (
                interp if int(L) == int(ell_in) else zeros.copy()
            )
    return blocks


def build_discrete_shell_window_matrix(
    shell_map: MultipoleShellMap,
    k_window: ArrayLike | None = None,
    W_ell: WindowEllMap | None = None,
    k_in: ArrayLike | None = None,
    ells: Sequence[int] = (0, 2, 4),
    ells_conv: Sequence[int] | None = None,
    continuous: str = "smooth",
    n_fftlog: int = 512,
    n_k_eval: int = 256,
    k_log_min: float | None = None,
    k_log_max: float | None = None,
    q: float = 0.0,
) -> DiscreteShellWindowMatrix:
    r"""
    Build the discrete-shell multipole window matrix.

    Parameters
    ----------
    shell_map : MultipoleShellMap
        From :meth:`~meer21cm.estimator.FieldPowerSpectrum.multipole_bin_index_map`.
    k_window : array_like, optional
        Wavenumbers where window multipoles were measured (required for
        ``continuous='smooth'``).
    W_ell : mapping, optional
        ``ell -> W_ell(k_window)`` (required for ``continuous='smooth'``).
    k_in : array_like
        Fine theory :math:`k` nodes.
    ells : sequence of int, default (0, 2, 4)
        Multipoles in the matrix block layout (estimator / theory).
    ells_conv : sequence of int, optional
        Continuous convolved multipoles :math:`L` used in
        :math:`P(\mathbf{k})=\sum_L P_L(|k|)\mathcal{L}_L(\mu)`.
        Defaults to ``ells`` for ``continuous='identity'``, else sorted keys
        of ``W_ell`` (or ``ells`` if empty).
    continuous : {'smooth', 'identity'}, default 'smooth'
        Continuous :math:`W_{L\ell'}` kernel. ``'identity'`` uses
        :math:`\delta_{L\ell'}\delta(k-k')` so the discrete shell sum alone
        encodes FFT :math:`\mu`-selection (uniform / no survey window).
        ``'smooth'`` builds the Hankel / Wigner kernel from ``W_ell``
        (pypower-style smooth window).
    n_fftlog : int, default 512
        FFTlog grid size for Hankel transforms (smooth only).
    n_k_eval : int, default 256
        Intermediate :math:`|k|` grid for interpolating continuous
        :math:`W_{L\ell'}` onto discrete modes.
    k_log_min, k_log_max : float, optional
        Ends of the intermediate log ``k`` grid (smooth only).
    q : float, default 0
        Extra FFTlog tilt (smooth only).

    Returns
    -------
    result : DiscreteShellWindowMatrix
        Dense matrix of shape ``(len(ells) * n_out, len(ells) * n_in)``.
    """
    if k_in is None:
        raise TypeError("k_in is required")
    continuous_s = str(continuous).lower()
    if continuous_s not in ("smooth", "identity"):
        raise ValueError(
            "continuous must be 'smooth' or 'identity', got %r" % continuous
        )

    ells_t = tuple(int(e) for e in ells)
    if ells_conv is not None:
        ells_L = tuple(int(e) for e in ells_conv)
    elif continuous_s == "identity":
        ells_L = ells_t
    else:
        keys = [int(L) for L in (W_ell or {}).keys()]
        ells_L = tuple(sorted(keys)) if keys else ells_t

    k_in_np = np.asarray(k_in, dtype=float)
    k_out = np.asarray(shell_map.k_eff, dtype=float)
    n_out = len(k_out)
    n_in = len(k_in_np)
    n_ell = len(ells_t)

    k_mode = np.asarray(shell_map.k, dtype=float)
    k_min_modes = float(np.min(k_mode[np.isfinite(k_mode) & (k_mode > 0)]))
    k_max_modes = float(np.max(k_mode[np.isfinite(k_mode)]))

    if continuous_s == "identity":
        k_eval = np.geomspace(0.5 * k_min_modes, 1.5 * k_max_modes, n_k_eval)
        blocks = identity_window_response_blocks(
            k_eval=k_eval,
            k_in=k_in_np,
            ells_out=ells_L,
            ells_in=ells_t,
        )
    else:
        if k_window is None or W_ell is None:
            raise ValueError("k_window and W_ell are required for continuous='smooth'")
        k_window_np = np.asarray(k_window, dtype=float)
        finite = np.isfinite(k_window_np) & (k_window_np > 0)
        if k_log_min is None:
            k_log_min = max(float(np.min(k_window_np[finite])) * 0.5, 1e-4)
        if k_log_max is None:
            k_log_max = float(np.max(k_window_np[finite])) * 1.5
        k_eval = np.geomspace(
            max(k_log_min, 0.5 * k_min_modes),
            max(k_log_max, 1.5 * k_max_modes),
            n_k_eval,
        )
        blocks = continuous_window_response_blocks(
            k_window,
            W_ell,
            k_eval=k_eval,
            k_in=k_in_np,
            ells_out=ells_L,
            ells_in=ells_t,
            n_fftlog=n_fftlog,
            k_log_min=k_log_min,
            k_log_max=k_log_max,
            q=q,
        )

    interps: dict[tuple[int, int], interpolate.interp1d] = {}
    for key, block in blocks.items():
        interps[key] = interpolate.interp1d(
            k_eval,
            block,
            axis=0,
            kind="linear",
            bounds_error=False,
            fill_value=0.0,
        )

    bin_index = np.asarray(shell_map.bin_index)
    mu = np.asarray(shell_map.mu, dtype=float)
    weights = np.asarray(shell_map.weights, dtype=float)

    matrix = np.zeros((n_ell * n_out, n_ell * n_in), dtype=float)

    L_plain = {L: _legendre_plain(L, mu) for L in ells_L}
    ell_factor = {ell: _legendre_with_factor(ell, mu) for ell in ells_t}

    for i_out, ell_out in enumerate(ells_t):
        for i_bin in range(n_out):
            in_bin = bin_index == i_bin
            w_bin = weights * in_bin
            U = float(np.sum(w_bin))
            if U <= 0 or not np.isfinite(k_out[i_bin]):
                continue
            k_n = k_mode[in_bin]
            w_n = weights[in_bin]
            B_ell = ell_factor[ell_out][in_bin] * (w_n / U)
            L_on_bin = {L: L_plain[L][in_bin] for L in ells_L}

            row_index = i_out * n_out + i_bin
            for i_in, ell_in in enumerate(ells_t):
                acc = np.zeros(n_in, dtype=float)
                for L in ells_L:
                    key = (int(L), int(ell_in))
                    w_rows = np.asarray(interps[key](k_n), dtype=float)
                    coeff = B_ell * L_on_bin[L]
                    acc += np.sum(coeff[:, None] * w_rows, axis=0)
                matrix[
                    row_index,
                    i_in * n_in : (i_in + 1) * n_in,
                ] = acc

    return DiscreteShellWindowMatrix(
        matrix=matrix,
        k_in=k_in_np,
        k_out=k_out,
        nmodes=np.asarray(shell_map.nmodes, dtype=float),
        ells=ells_t,
    )


def apply_discrete_shell_window_matrix(
    p_ell_in: Mapping[int, ArrayLike] | ArrayLike,
    window_matrix: ArrayLike,
    ells: Sequence[int] = (0, 2, 4),
) -> dict[int, NDArray[np.floating]]:
    """
    Apply a discrete-shell window matrix to concatenated or dict multipole theory.

    Parameters
    ----------
    p_ell_in : mapping or array_like
        ``{ell: P_ell(k_in)}`` or a 1D vector of length ``len(ells) * n_in``.
    window_matrix : array_like
        Dense matrix from :func:`build_discrete_shell_window_matrix`.
    ells : sequence of int, default (0, 2, 4)
        Multipole order matching the matrix block layout.

    Returns
    -------
    p_ell_out : dict
        ``{ell: P_ell(k_out)}``.
    """
    ells_t = tuple(int(e) for e in ells)
    if isinstance(p_ell_in, Mapping):
        vec = np.concatenate([np.asarray(p_ell_in[ell], dtype=float) for ell in ells_t])
    else:
        vec = np.asarray(p_ell_in, dtype=float)
    window_matrix_np = np.asarray(window_matrix, dtype=float)
    out = window_matrix_np @ vec
    n_out = window_matrix_np.shape[0] // len(ells_t)
    return {ell: out[i * n_out : (i + 1) * n_out] for i, ell in enumerate(ells_t)}
