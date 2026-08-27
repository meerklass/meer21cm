r"""
Survey-window matrices for theory multipoles: smooth (Hankel/Wigner) and
exact mesh-level (FFT) layers.

Builds dense matrices that map continuous theory multipoles
:math:`P_{\ell'}(k_{\mathrm{in}})` to estimator bin multipoles
:math:`P_\ell(k_{\mathrm{out}})`.  Two continuous layers share the same
discrete-shell row structure (``k_out`` bins from
:class:`~meer21cm.estimator.MultipoleShellMap`):

- **smooth** (``build_discrete_shell_window_matrix`` with
  ``continuous='smooth'`` / ``'identity'``): the scalar survey selection is
  measured as :math:`W_L(k)`, Hankel-transformed to :math:`Q_L(s)`, coupled
  by Wigner-3j to the kernel :math:`W_{L\ell'}(k,k')`, evaluated on each
  FFT mode and sampled the same way as the data estimator (the discrete-
  :math:`\mu` projector; the pypower / Beutler et al. pipeline).
- **mesh** (``build_mesh_window_matrix``): the estimator's own mesh
  response, exact for **any** line of sight.  For a windowed isotropic
  theory with selection :math:`w(x)` and the observer of
  ``ps.los_observer``,

  .. math::

      \langle P_\ell^{3D}(\mathbf k)\rangle
      =
      4\pi R \sum_m Y_{\ell m}(\hat k)\,
      \bigl[\mathrm{FFT}[w\,Y_{\ell m}(\hat n(x))]
      \,\mathrm{FFT}[w]^* \;\circledast\; (t\,P_0)\bigr](\mathbf k),

  evaluated on the mesh with the estimator's own operators.  The varying
  :math:`\hat n(x)` of a true lightcone observer is fully contained in
  :math:`Y_{\ell m}(\hat n(x))`; the discrete-:math:`\mu` projector is its
  :math:`1/d\to 0` limit.

The window matrices are Yamamoto-only (``los='firstpoint'`` /
``'endpoint'``).  ``los='global'`` raises; use
:meth:`~meer21cm.power.PowerSpectrum.get_1d_power` of the 3D cubes instead.

The outer discrete-shell sum applies the discrete-:math:`\mu` projector:
reconstruct :math:`P(k,\mu)=\sum_L P_L(|k|)\,\mathcal{L}_L(\mu)` and bin
with :math:`(2\ell+1)\mathcal{L}_\ell(\mu)`,
:math:`\mu_n=\hat k_n\cdot\hat n_{\mathrm{ref}}` (the local-LOS reference
direction, box centre). This is the **leading-order** binning of the
Yamamoto estimator, whose 3D cube is
:math:`P_\ell^{3D}=(2\ell+1)\mathcal{L}_\ell(\mu)\,P_0^{3D}`; identity
:math:`W` matches :meth:`~meer21cm.power.PowerSpectrum.get_1d_power` of the
3D cube. The *varying* :math:`\hat n(x)` is a higher-order (wide-angle,
:math:`1/d`) correction, not part of this projector.

Notation
--------
- :math:`i` — estimator output bin index (``k_out``; edges from
  :class:`~meer21cm.estimator.MultipoleShellMap`).
- :math:`j` — theory input node index (``k_in`` / :math:`k'_j`).
- :math:`S_i` — discrete Fourier modes with
  :math:`|\mathbf{k}_n|` in bin :math:`i` (shell membership from
  :attr:`~meer21cm.estimator.MultipoleShellMap.bin_index`).
- :math:`w_n` — per-mode binning weight;
  :math:`U_i=\sum_{\mathbf{k}_m\in S_i} w_m` normalises the shell average.
- :math:`t(\mathbf{k}_n)` — optional same-:math:`k` transfer (``mode_scale``;
  MAS / gridding compensation). Does **not** enter :math:`U_i`.
- :math:`\mu_n` — LOS cosine of mode :math:`\mathbf{k}_n`
  (:attr:`~meer21cm.estimator.MultipoleShellMap.mu`;
  :math:`\hat k\cdot\hat n_{\mathrm{ref}}`).
- :math:`W_{L\ell'}(k,k')` — continuous response kernel (see below).

The continuous kernel :math:`W_{L\ell'}(k,k')` may be:

- **identity** — :math:`\delta_{L\ell'}\delta(k-k')` (no survey convolution;
  discrete-shell sampling of continuous :math:`P_{\ell'}` only);
- **smooth** — Hankel / Wigner response from measured window multipoles
  :math:`W_L(k)` (selection field or randoms; pypower-style smooth window):

  .. math::

      Q_L(s)
      =
      \frac{i^{L}}{2\pi^{2}}
      \int_0^{\infty}\!\mathrm{d}q\,q^{2}\,
      j_{L}(qs)\,W_L(q),

  .. math::

      W_{\ell\ell'}(k,k')
      =
      \frac{2}{\pi}
      (-1)^{\ell/2}(-1)^{\ell'/2}
      \int_0^{\infty}\!\mathrm{d}s\,s^{2}\,
      j_{\ell}(ks)\,j_{\ell'}(k's)
      \sum_{\mathcal{L}}
      C_{\ell\ell'\mathcal{L}}\,
      Q_{\mathcal{L}}(s),

  with Wigner–Legendre couplings :math:`C_{\ell\ell'\mathcal{L}}` from
  :func:`wigner3j_square` (``prefactor=True``). In code the
  :math:`k'^2\mathrm{d}k'` volume element is included when building the
  discrete matrix columns.

Continuous theory multipoles still come from the Gauss–Legendre
:math:`\mu` integral
(:meth:`~meer21cm.model.ModelPowerSpectrum.get_theory_multipoles_kmu`).
The discrete FFT sampling of those multipoles is this matrix, not a
second :math:`\mu` integral.

References
----------
- Beutler et al., 2019, MNRAS, 484, 2233 (arXiv:1810.05051)
- Wilson et al., 2017, MNRAS, 464, 3121
- cosmodesi/pypower smooth-window / FFT-window implementations
"""

from __future__ import annotations

import logging
import math
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import interpolate
from scipy.special import eval_legendre, spherical_jn

from .estimator import MultipoleShellMap
from .fftlog import CorrelationToPower, PowerToCorrelation
from .power_ops import power_weights_renorm
from .spherical import get_real_Ylm, unit_khat_from_k_vec
from .util import legendre_polynomial_with_factor
from .wide_angle import power_spectrum_odd_wide_angle_matrix

logger = logging.getLogger(__name__)

WindowEllMap = Mapping[int, ArrayLike]

_WINDOW_LOS = ("firstpoint", "endpoint")
_MESH_K_IN_SPAN_RTOL = 1e-3


def require_yamamoto_los(los: str) -> str:
    """
    Window matrices are Yamamoto-only.

    ``los='global'`` is the legacy 3D path
    (:meth:`~meer21cm.power.PowerSpectrum.get_1d_power` of 3D cubes).
    """
    los_s = str(los)
    if los_s == "global":
        raise ValueError(
            "los='global' is the legacy 3D path (get_1d_power of 3D cubes); "
            "the window matrix is Yamamoto-only (los='firstpoint'/'endpoint')."
        )
    if los_s not in _WINDOW_LOS:
        raise ValueError(
            "los must be 'firstpoint' or 'endpoint' for the window matrix, "
            f"got {los_s!r}"
        )
    return los_s


def propose_mesh_k_in(
    ps,
    n: int = 80,
    *,
    low_factor: float = 0.5,
) -> NDArray[np.floating]:
    """
    Theory :math:`k_{\\mathrm{in}}` spanning every grid :math:`|k|` the mesh
    window can couple in.

    :func:`build_mesh_window_matrix` tiles **all** rFFT modes into Voronoi
    shells of ``k_in``.  A ``k_in`` that stops below the grid's
    ``max |k|`` (as :func:`~meer21cm.multipole_ops.propose_k_in`'s
    ``1.5 k_max`` often does — the grid reaches
    :math:`\\sqrt{3}\,k_{\\mathrm{Nyq}}`) assigns every outer mode the
    constant :math:`P(k_{\\mathrm{in}}[-1])` instead of the falling
    theory.

    Keeps :func:`~meer21cm.multipole_ops.propose_k_in`'s low end and log
    node density and extends the top to the grid maximum.  The smooth
    (Hankel) path integrates :math:`k` continuously and should keep
    :func:`~meer21cm.multipole_ops.propose_k_in`.
    """
    from .multipole_ops import propose_k_in

    k1dbins = getattr(ps, "k1dbins", None)
    if k1dbins is None:
        raise ValueError("ps.k1dbins is required for propose_mesh_k_in")
    base = propose_k_in(k1dbins, n=int(n), low_factor=float(low_factor))
    k_mode = getattr(ps, "k_mode", None)
    if k_mode is None:
        return base
    k_grid_max = float(np.max(np.asarray(k_mode, dtype=float))) * 1.001
    if k_grid_max <= base[-1]:
        return base
    n_wide = int(
        np.ceil(int(n) * np.log(k_grid_max / base[0]) / np.log(base[-1] / base[0]))
    )
    return np.geomspace(float(base[0]), k_grid_max, n_wide)


def _warn_truncated_mesh_k_in(ps, k_in: ArrayLike) -> None:
    """Warn if mesh ``k_in`` does not span the PS Fourier grid."""
    k_mode = getattr(ps, "k_mode", None)
    if k_mode is None:
        return
    k_in_np = np.asarray(k_in, dtype=float)
    if k_in_np.size == 0:
        return
    k_grid_max = float(np.max(np.asarray(k_mode, dtype=float)))
    k_in_max = float(np.max(k_in_np))
    if k_in_max < k_grid_max * (1.0 - _MESH_K_IN_SPAN_RTOL):
        msg = (
            "k_in does not span the PS Fourier grid "
            f"(max k_in={k_in_max:.4g}, max |k|={k_grid_max:.4g}); "
            "outer modes are assigned P(k_in[-1]). "
            "Use meer21cm.window.propose_mesh_k_in."
        )
        logger.warning(msg)
        warnings.warn(msg, UserWarning, stacklevel=3)


def list_mesh_window_columns(
    n_k_in: int,
    *,
    in_group_index: ArrayLike | None = None,
    in_group_scale: Sequence[ArrayLike] | None = None,
    in_bin_weights: Callable[[int, int], ArrayLike | None] | None = None,
    in_shell: Sequence[ArrayLike] | None = None,
    column_empty: Callable[[int, int], bool] | None = None,
) -> list[int] | list[tuple[int, int]]:
    """
    Column ids filled by :func:`build_mesh_window_matrix`.

    Without inner-mode grouping this is ``range(n_k_in)``.  With
    ``in_bin_weights`` it is the active ``(group, k_in)`` pairs.

    Parameters
    ----------
    n_k_in : int
        Number of theory :math:`|k|` nodes.
    in_group_index, in_group_scale, in_bin_weights, in_shell
        Same meaning as in :func:`build_mesh_window_matrix`.
    column_empty : callable, optional
        ``column_empty(j, g) -> True`` to skip a ``(group, k_in)`` pair
        without evaluating ``in_bin_weights`` (used when a CIC mass
        function already knows the intersection is empty).
    """
    n_in = int(n_k_in)
    if in_bin_weights is None and in_group_index is None and in_group_scale is None:
        return list(range(n_in))
    if in_group_scale is not None:
        n_gin = len(in_group_scale)
        gi_flat = None
    elif in_group_index is not None:
        gi_flat = np.asarray(in_group_index, dtype=np.int64).ravel()
        n_gin = int(gi_flat.max()) + 1 if gi_flat.size else 0
    else:
        raise ValueError(
            "in_bin_weights columns require in_group_index or in_group_scale"
        )
    cols: list[tuple[int, int]] = []
    for g in range(n_gin):
        if gi_flat is not None:
            sel_g = gi_flat == g
            if not np.any(sel_g):
                continue
        else:
            sel_g = None
        for j in range(n_in):
            if (
                sel_g is not None
                and in_shell is not None
                and not np.any(np.asarray(in_shell[j]) & sel_g)
            ):
                continue
            if column_empty is not None and column_empty(j, g):
                continue
            if (
                column_empty is None
                and in_bin_weights is not None
                and in_bin_weights(j, g) is None
            ):
                continue
            cols.append((int(g), int(j)))
    return cols


def sum_mesh_window_cubes(
    parts: Sequence[Mapping[int, ArrayLike]],
) -> dict[int, NDArray[np.floating]]:
    r"""
    Sum partial 3D mesh-window accumulators from column chunks.

    Analogue of :func:`accumulate_mesh_window_matrices` for the unbinned
    cubes returned by :func:`accumulate_mesh_window_cubes`.

    Parameters
    ----------
    parts : sequence of dict
        Each mapping is ``{ell: ndarray}`` on the rFFT grid.

    Returns
    -------
    acc : dict
        ``{ell: ndarray}`` summed over ``parts``.
    """
    if not parts:
        raise ValueError("sum_mesh_window_cubes needs at least one partial")
    out = {int(ell): np.asarray(parts[0][ell], dtype=float).copy() for ell in parts[0]}
    for part in parts[1:]:
        for ell in out:
            out[int(ell)] = out[int(ell)] + np.asarray(part[ell], dtype=float)
    return out


def accumulate_mesh_window_matrices(
    parts: Sequence[DiscreteShellWindowMatrix],
) -> DiscreteShellWindowMatrix:
    """
    Sum column-chunk mesh window matrices.

    Each part must share ``k_in``, ``k_out``, ``ells_in`` / ``ells_out``.
    ``offset`` is taken from the first non-None contribution (shot terms
    should be applied only on a full ``columns=None`` build).
    """
    if not parts:
        raise ValueError("accumulate_mesh_window_matrices needs at least one matrix")
    first = parts[0]
    matrix = np.array(first.matrix, dtype=float, copy=True)
    offset = first.offset
    for extra in parts[1:]:
        if extra.matrix.shape != matrix.shape:
            raise ValueError(
                "mesh window chunks have mismatched matrix shapes "
                f"{matrix.shape} vs {extra.matrix.shape}"
            )
        if not np.allclose(extra.k_in, first.k_in):
            raise ValueError("mesh window chunks have mismatched k_in")
        matrix = matrix + np.asarray(extra.matrix, dtype=float)
        if offset is None and extra.offset is not None:
            offset = extra.offset
    return DiscreteShellWindowMatrix(
        matrix=matrix,
        k_in=first.k_in,
        k_out=first.k_out,
        nmodes=first.nmodes,
        ells=first.ells_out,
        ells_in=first.ells_in,
        ells_out=first.ells_out,
        offset=offset,
    )


def run_mesh_window_columns(kwargs: dict, columns) -> DiscreteShellWindowMatrix:
    """
    Pickleable worker: build a subset of mesh-window columns.

    ``kwargs`` are keyword arguments to :func:`build_mesh_window_matrix`
    except ``columns``.  Callers that cannot pickle ``ps`` or
    ``in_bin_weights`` should reconstruct those in an initializer (see
    :class:`~meer21cm.power.MultipolePowerSpectrum`).
    """
    return build_mesh_window_matrix(**kwargs, columns=columns)


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


def _weights_trapz(x: ArrayLike) -> NDArray[np.floating]:
    """Trapezoidal quadrature weights for integrating :math:`f(x)\\,\\mathrm{d}x`."""
    x_np = np.asarray(x, dtype=float)
    w = np.empty_like(x_np)
    w[0] = (x_np[1] - x_np[0]) / 2.0
    w[-1] = (x_np[-1] - x_np[-2]) / 2.0
    w[1:-1] = (x_np[2:] - x_np[:-2]) / 2.0
    return w


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


def window_zero_mode_power(
    selection: ArrayLike,
    box_volume: float,
    weights_grid: ArrayLike | None = None,
) -> float:
    r"""
    :math:`k=0` window power of a selection cube, same convention as :math:`W_L`.

    With a forward FFT, :math:`\tilde w(0)=\langle w\rangle`, so

    .. math::

        W(k=0)
        =
        \langle w\rangle^2\,
        V\,
        \frac{N}{\sum_i w_i^2}.

    The corresponding configuration-space offset is
    :math:`Q_0^{\mathrm{DC}}=W(0)/V`, the pair-count term that FFTLog of
    :math:`k>0` shells cannot see (pypower ``power_zero_nonorm``).
    """
    w = np.asarray(selection, dtype=float)
    if weights_grid is not None:
        w = w * np.asarray(weights_grid, dtype=float)
    n_grid = float(w.size)
    sum_w2 = float(np.sum(w * w))
    if sum_w2 <= 0.0:
        return 0.0
    mean = float(np.mean(w))
    return mean**2 * float(box_volume) * (n_grid / sum_w2)


def discrete_window_power_to_correlation(
    k: ArrayLike,
    W_ell: WindowEllMap,
    sep: ArrayLike,
    nmodes: ArrayLike,
    box_volume: float,
    W_zero: float = 0.0,
) -> dict[int, NDArray[np.floating]]:
    r"""
    Configuration-space window multipoles by a discrete :math:`k`-bin sum.

    Matches pypower ``power_to_correlation_window`` with
    ``volume = (2\pi)^3 n_{\mathrm{modes}} / V``:

    .. math::

        Q_L(s)
        =
        (-1)^{L/2}
        \sum_i
        \frac{n_i}{V}\,
        W_L(k_i)\,
        j_L(k_i s)
        +
        \delta_{L0}\,W(0)/V.

    ``W_zero`` is the :math:`k=0` monopole (see
    :func:`window_zero_mode_power`); it is omitted from the :math:`k>0` sum.
    """
    k_np = np.asarray(k, dtype=float)
    nmodes_np = np.asarray(nmodes, dtype=float)
    sep_np = np.asarray(sep, dtype=float)
    volume = float(box_volume)
    if volume <= 0.0:
        raise ValueError("box_volume must be positive")
    if nmodes_np.shape != k_np.shape:
        raise ValueError("nmodes must match k")
    mask = np.isfinite(k_np) & (k_np > 0) & np.isfinite(nmodes_np) & (nmodes_np > 0)
    k_use = k_np[mask]
    n_use = nmodes_np[mask]
    weight = n_use / volume
    out: dict[int, NDArray[np.floating]] = {}
    if k_use.size == 0:
        q_dc = float(W_zero) / volume
        for L in W_ell:
            out[int(L)] = np.full_like(sep_np, q_dc if int(L) == 0 else 0.0)
        return out
    ks = k_use[:, None] * sep_np[None, :]
    q_dc = float(W_zero) / volume
    for L, wk in W_ell.items():
        ell = int(L)
        w_use = np.asarray(wk, dtype=float)[mask]
        finite = np.isfinite(w_use)
        if not np.any(finite):
            out[ell] = np.full_like(sep_np, q_dc if ell == 0 else 0.0)
            continue
        phase = (-1) ** (ell // 2)
        q = phase * np.sum(
            (weight[finite] * w_use[finite])[:, None] * spherical_jn(ell, ks[finite]),
            axis=0,
        )
        if ell == 0:
            q = q + q_dc
        out[ell] = np.asarray(q, dtype=float)
    return out


def interpolate_window_to_log_k(
    k_meas: ArrayLike,
    w_ell: ArrayLike,
    k_log: ArrayLike,
    *,
    fill_low: float | None = None,
) -> NDArray[np.floating]:
    """
    Interpolate a measured window multipole onto a log-spaced ``k`` grid.

    By default the low-``k`` side clamps to the first measured sample. Pass
    ``fill_low=0`` when a separate :math:`k=0` term (``W_zero``) will supply
    the pair-count, so the FFTLog plateau does not double-count it.
    """
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
    low = float(ww[0]) if fill_low is None else float(fill_low)
    spline = interpolate.interp1d(
        uniq,
        ww,
        kind="linear",
        bounds_error=False,
        fill_value=(low, 0.0),
    )
    return np.asarray(spline(k_log_np), dtype=float)


def _filter_finite_window_multipoles(
    k_window: ArrayLike,
    W_ell: WindowEllMap,
    nmodes: ArrayLike | None = None,
) -> tuple[NDArray[np.floating], WindowEllMap, NDArray[np.floating] | None]:
    """Drop empty / invalid ``k`` shells before Hankel transforms."""
    k_np = np.asarray(k_window, dtype=float)
    mask = np.isfinite(k_np) & (k_np > 0)
    for wk in W_ell.values():
        mask &= np.isfinite(np.asarray(wk, dtype=float))
    if not np.any(mask):
        raise ValueError("No finite window multipole samples")
    k_f = k_np[mask]
    W_f = {int(L): np.asarray(wk, dtype=float)[mask] for L, wk in W_ell.items()}
    nmodes_f = None if nmodes is None else np.asarray(nmodes, dtype=float)[mask]
    return k_f, W_f, nmodes_f


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
    ``k_out`` (see :func:`build_discrete_shell_window_matrix`). Identity
    continuous :math:`W` has no survey convolution: global LOS still
    applies discrete-:math:`\\mu` mixing; local LOS is block-diagonal
    :math:`|k|` rebin.

    Attributes
    ----------
    matrix : ndarray, shape ``(n_ell_out * n_out, n_ell_in * n_in)``
        Acts on concatenated ``[P_ell[0](k_in), P_ell[1](k_in), ...]``.
    k_in : ndarray
        Theory wavenumber nodes.
    k_out : ndarray
        Estimator bin centres (effective ``k``).
    nmodes : ndarray
        Modes per output bin (from the shell map).
    ells : tuple of int
        Output multipole block order (alias of :attr:`ells_out`).
    ells_in : tuple of int, optional
        Theory input multipoles. Defaults to :attr:`ells`.
    ells_out : tuple of int, optional
        Observed output multipoles. Defaults to :attr:`ells`.
    offset : mapping, optional
        ``{ell: P_ell(k_out)}`` additive terms added on top of the matrix
        product in :meth:`apply` (theory-independent; e.g. the map-sampling
        shot diagonal of :func:`build_mesh_window_matrix`, which cannot be a
        linear operator on the theory).
    """

    matrix: NDArray[np.floating]
    k_in: NDArray[np.floating]
    k_out: NDArray[np.floating]
    nmodes: NDArray[np.floating]
    ells: tuple[int, ...]
    ells_in: tuple[int, ...] | None = None
    ells_out: tuple[int, ...] | None = None
    offset: Mapping[int, ArrayLike] | None = None

    def __post_init__(self) -> None:
        self.ells = tuple(int(e) for e in self.ells)
        self.ells_out = (
            tuple(int(e) for e in self.ells)
            if self.ells_out is None
            else tuple(int(e) for e in self.ells_out)
        )
        self.ells = self.ells_out
        self.ells_in = (
            tuple(self.ells_out)
            if self.ells_in is None
            else tuple(int(e) for e in self.ells_in)
        )
        if self.offset is not None:
            self.offset = {
                int(e): np.asarray(o, dtype=float) for e, o in self.offset.items()
            }

    def apply(
        self, p_ell_in: Mapping[int, ArrayLike] | ArrayLike
    ) -> dict[int, NDArray[np.floating]]:
        """Apply this matrix; see :func:`apply_discrete_shell_window_matrix`."""
        out = apply_discrete_shell_window_matrix(
            p_ell_in,
            self.matrix,
            ells=self.ells_out,
            ells_in=self.ells_in,
        )
        if self.offset:
            for ell, off in self.offset.items():
                if ell in out:
                    out[int(ell)] = out[int(ell)] + off
        return out

    def resum_input_odd_wide_angle(
        self,
        los: str = "firstpoint",
        d: float = 1.0,
        ells_even: Sequence[int] | None = None,
    ) -> DiscreteShellWindowMatrix:
        r"""
        Left-multiply by the wa_order=1 matrix so theory input is even only.

        :math:`W \leftarrow W\,M_{\mathrm{WA}}` with :math:`M_{\mathrm{WA}}`
        mapping ``ells_even`` → current :attr:`ells_in` (even + odd).
        """
        if ells_even is None:
            ells_even_t = tuple(e for e in self.ells_in if e % 2 == 0)
        else:
            ells_even_t = tuple(int(e) for e in ells_even)
        if not ells_even_t:
            raise ValueError("ells_even is empty; cannot resum odd wide-angle")
        m_wa = power_spectrum_odd_wide_angle_matrix(
            self.k_in,
            ells_in=ells_even_t,
            ells_out=self.ells_in,
            d=d,
            los=los,
        )
        self.matrix = np.asarray(self.matrix, dtype=float) @ m_wa
        self.ells_in = ells_even_t
        return self


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
    nmodes: ArrayLike | None = None,
    box_volume: float | None = None,
    W_zero: float = 0.0,
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

    use_discrete = nmodes is not None and box_volume is not None
    window_s: dict[int, NDArray[np.floating]] = {}
    sep_ref: NDArray[np.floating] | None = None
    if use_discrete:
        # s-grid matching FFTLog ``xy=1`` so the Q→W(k,k') transform is unchanged.
        sep_ref = 1.0 / k_log[::-1]
        W_for_sum = {
            int(L): np.asarray(W_ell[L], dtype=float) for L in needed_L if L in W_ell
        }
        window_s = discrete_window_power_to_correlation(
            k_window_np,
            W_for_sum,
            sep_ref,
            nmodes=nmodes,
            box_volume=float(box_volume),
            W_zero=0.0,
        )
        for L in needed_L:
            window_s.setdefault(int(L), np.zeros_like(sep_ref))
    else:
        W_k: dict[int, NDArray[np.floating]] = {}
        for L in sorted(needed_L):
            if L in W_ell:
                W_k[L] = interpolate_window_to_log_k(k_window_np, W_ell[L], k_log)
            else:
                W_k[L] = np.zeros_like(k_log)
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
    # k=0 selection power is a constant in Q_0(s). Putting that constant
    # through FFTLog rings; completeness of j_ℓ turns it into a multiple of
    # the identity on each diagonal multipole block (Wigner L=0).
    if box_volume is not None and float(W_zero) != 0.0:
        q_dc = float(W_zero) / float(box_volume)
        id_blocks = identity_window_response_blocks(
            k_eval_np, k_in_np, ells_out_t, ells_in_t
        )
        for ell_out in ells_out_t:
            for ell_in in ells_in_t:
                Ls, coeffs = wigner3j_square(ell_out, ell_in, prefactor=True)
                if 0 not in Ls:
                    continue
                c0 = float(coeffs[Ls.index(0)])
                key = (int(ell_out), int(ell_in))
                blocks[key] = blocks[key] + c0 * q_dc * id_blocks[key]
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
    Identity continuous blocks :math:`W_{\ell\ell'}=\delta_{\ell\ell'}\delta(k-k')`.

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
    ells_in: Sequence[int] | None = None,
    continuous: str = "smooth",
    n_fftlog: int = 512,
    n_k_eval: int = 256,
    k_log_min: float | None = None,
    k_log_max: float | None = None,
    q: float = 0.0,
    nmodes: ArrayLike | None = None,
    box_volume: float | None = None,
    W_zero: float = 0.0,
    mode_scale: ArrayLike | None = None,
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
        Observed / output multipoles (matrix rows).
    ells_in : sequence of int, optional
        Theory input multipoles (matrix columns). Defaults to ``ells``.
    continuous : {'smooth', 'identity'}, default 'smooth'
        Continuous :math:`W_{L\ell'}` kernel. ``'identity'`` uses
        :math:`\delta_{L\ell'}\delta(k-k')` (no survey convolution).
        ``'smooth'`` builds the Hankel / Wigner kernel from ``W_ell``.
        The outer discrete-shell sum then samples that kernel on the FFT
        grid with the discrete-:math:`\mu` projector
        :math:`(2\ell+1)\mathcal{L}_\ell(\mu)`,
        :math:`\mu=\hat k\cdot\hat n_{\mathrm{ref}}`.
        ``shell_map.los='global'`` raises.
    n_fftlog : int, default 512
        FFTlog grid size for Hankel transforms (smooth only).
    n_k_eval : int, default 256
        Intermediate :math:`|k|` grid for interpolating continuous
        :math:`W_{\ell\ell'}` onto discrete modes.
    k_log_min, k_log_max : float, optional
        Ends of the intermediate log ``k`` grid (smooth only).
    q : float, default 0
        Extra FFTlog tilt (smooth only).
    nmodes : array_like, optional
        Modes per measured :math:`W_L` bin. With ``box_volume``, uses a
        pypower-style discrete Hankel (including the :math:`k=0` term
        ``W_zero``) instead of FFTLog of :math:`k>0` shells.
    box_volume : float, optional
        Survey-box volume :math:`V` for the discrete Hankel.
    W_zero : float, default 0
        :math:`k=0` monopole window power (see
        :func:`window_zero_mode_power`).
    mode_scale : array_like, optional
        Same-\(k\) multiplicative transfer on each Cartesian Fourier mode
        (same shape as ``shell_map.k``). Multiplies the theory contribution
        inside the discrete-shell sum but does **not** enter the bin
        normalisation \(U=\sum w_n\). Use for MAS / gridding compensation
        (and any other exact per-mode grid transfer).

    Returns
    -------
    result : DiscreteShellWindowMatrix
        Dense matrix of shape ``(len(ells) * n_out, len(ells_in) * n_in)``.
    """
    if k_in is None:
        raise TypeError("k_in is required")
    continuous_s = str(continuous).lower()
    if continuous_s not in ("smooth", "identity"):
        raise ValueError(
            "continuous must be 'smooth' or 'identity', got %r" % continuous
        )

    ells_out_t = tuple(int(e) for e in ells)
    ells_in_t = (
        tuple(int(e) for e in ells_out_t)
        if ells_in is None
        else tuple(int(e) for e in ells_in)
    )
    require_yamamoto_los(str(getattr(shell_map, "los", "endpoint")))
    # Intermediate L for P(k,μ)=∑_L P_L L_L, then (2ℓ+1) L_ℓ projection.
    ells_L = tuple(sorted(set(ells_out_t) | set(ells_in_t)))

    k_in_np = np.asarray(k_in, dtype=float)
    k_out = np.asarray(shell_map.k_eff, dtype=float)
    n_out = len(k_out)
    n_in = len(k_in_np)
    n_ell_out = len(ells_out_t)
    n_ell_in = len(ells_in_t)

    k_mode = np.asarray(shell_map.k, dtype=float)
    k_min_modes = float(np.min(k_mode[np.isfinite(k_mode) & (k_mode > 0)]))
    k_max_modes = float(np.max(k_mode[np.isfinite(k_mode)]))

    if continuous_s == "identity":
        k_eval = np.geomspace(0.5 * k_min_modes, 1.5 * k_max_modes, n_k_eval)
        blocks = identity_window_response_blocks(
            k_eval=k_eval,
            k_in=k_in_np,
            ells_out=ells_L,
            ells_in=ells_in_t,
        )
    else:
        if k_window is None or W_ell is None:
            raise ValueError("k_window and W_ell are required for continuous='smooth'")
        k_window_np, W_ell_f, nmodes_f = _filter_finite_window_multipoles(
            k_window, W_ell, nmodes=nmodes
        )
        k_window_np = np.asarray(k_window_np, dtype=float)
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
            k_window_np,
            W_ell_f,
            k_eval=k_eval,
            k_in=k_in_np,
            ells_out=ells_L,
            ells_in=ells_in_t,
            n_fftlog=n_fftlog,
            k_log_min=k_log_min,
            k_log_max=k_log_max,
            q=q,
            nmodes=nmodes_f,
            box_volume=box_volume,
            W_zero=W_zero,
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
    weights = np.asarray(shell_map.weights, dtype=float)
    mu = np.asarray(shell_map.mu, dtype=float)
    if mode_scale is None:
        mode_scale_arr = None
    else:
        mode_scale_arr = np.asarray(mode_scale, dtype=float)
        if mode_scale_arr.shape != k_mode.shape:
            raise ValueError(
                "mode_scale must match shell_map.k shape "
                f"(got {mode_scale_arr.shape}, expected {k_mode.shape})"
            )

    L_plain: dict[int, NDArray[np.floating]] = {}
    ell_factor: dict[int, NDArray[np.floating]] = {}
    for L in ells_L:
        L_plain[int(L)] = _legendre_plain(L, mu)
    for ell in ells_out_t:
        ell_factor[int(ell)] = _legendre_with_factor(ell, mu)

    matrix = np.zeros((n_ell_out * n_out, n_ell_in * n_in), dtype=float)

    for i_out, ell_out in enumerate(ells_out_t):
        ell_out_i = int(ell_out)
        for i_bin in range(n_out):
            in_bin = bin_index == i_bin
            w_bin = weights * in_bin
            U = float(np.sum(w_bin))
            if U <= 0 or not np.isfinite(k_out[i_bin]):
                continue
            k_n = k_mode[in_bin]
            w_n = weights[in_bin]
            # mode_scale is a theory transfer at each Cartesian mode; keep U
            # as the estimator's bin weight sum (selection / k1dweights only).
            t_n = 1.0 if mode_scale_arr is None else mode_scale_arr[in_bin]
            wt = (w_n / U) * t_n

            row_index = i_out * n_out + i_bin
            B_ell = ell_factor[ell_out_i][in_bin] * wt
            L_on_bin = {L: L_plain[L][in_bin] for L in ells_L}
            for i_in, ell_in in enumerate(ells_in_t):
                acc = np.zeros(n_in, dtype=float)
                for L in ells_L:
                    w_rows = np.asarray(
                        interps[(int(L), int(ell_in))](k_n), dtype=float
                    )
                    acc += np.sum((B_ell * L_on_bin[L])[:, None] * w_rows, axis=0)
                matrix[
                    row_index,
                    i_in * n_in : (i_in + 1) * n_in,
                ] = acc

    return DiscreteShellWindowMatrix(
        matrix=matrix,
        k_in=k_in_np,
        k_out=k_out,
        nmodes=np.asarray(shell_map.nmodes, dtype=float),
        ells=ells_out_t,
        ells_in=ells_in_t,
        ells_out=ells_out_t,
    )


def apply_discrete_shell_window_matrix(
    p_ell_in: Mapping[int, ArrayLike] | ArrayLike,
    window_matrix: ArrayLike,
    ells: Sequence[int] = (0, 2, 4),
    ells_in: Sequence[int] | None = None,
) -> dict[int, NDArray[np.floating]]:
    """
    Apply a discrete-shell window matrix to concatenated or dict multipole theory.

    Parameters
    ----------
    p_ell_in : mapping or array_like
        ``{ell: P_ell(k_in)}`` or a 1D vector of length ``len(ells_in) * n_in``.
    window_matrix : array_like
        Dense matrix from :func:`build_discrete_shell_window_matrix`.
    ells : sequence of int, default (0, 2, 4)
        Output multipoles (matrix row blocks).
    ells_in : sequence of int, optional
        Theory input multipoles (matrix column blocks). Defaults to ``ells``.

    Returns
    -------
    p_ell_out : dict
        ``{ell: P_ell(k_out)}``.
    """
    ells_out_t = tuple(int(e) for e in ells)
    ells_in_t = (
        tuple(int(e) for e in ells_out_t)
        if ells_in is None
        else tuple(int(e) for e in ells_in)
    )
    if isinstance(p_ell_in, Mapping):
        vec = np.concatenate(
            [np.asarray(p_ell_in[ell], dtype=float) for ell in ells_in_t]
        )
    else:
        vec = np.asarray(p_ell_in, dtype=float)
    window_matrix_np = np.asarray(window_matrix, dtype=float)
    out = window_matrix_np @ vec
    n_out = window_matrix_np.shape[0] // len(ells_out_t)
    return {ell: out[i * n_out : (i + 1) * n_out] for i, ell in enumerate(ells_out_t)}


# ---------------------------------------------------------------------------
# Exact mesh-level (FFT) window
# ---------------------------------------------------------------------------


def _extend_hermitian_z(
    c_rfft: NDArray[np.complexfloating], shape: tuple
) -> NDArray[np.complexfloating]:
    """Hermitian extension of an rfft-grid array to the full FFT grid.

    For the transform of a real field,
    :math:`F(-k_x,-k_y,-k_z)=F^*(k_x,k_y,k_z)`, so the negative-:math:`k_z`
    half is filled from the positive half with
    :math:`(x,y)\\mapsto(-x,-y)` and conjugation.  (Flipping only the
    :math:`z` axis is incorrect for a generic real field.)
    """
    nz = c_rfft.shape[2]
    out = np.zeros(shape, dtype=np.result_type(c_rfft, complex))
    out[..., :nz] = c_rfft
    src = c_rfft[..., 1 : shape[2] - nz + 1]
    src_flipped_z = np.flip(src, axis=-1)
    src_neg_xy = np.roll(np.flip(np.flip(src_flipped_z, 0), 1), (1, 1), (0, 1))
    out[..., nz:] = np.conj(src_neg_xy)
    return out


# ---------------------------------------------------------------------------
# Map-sampling shot diagonal (exact per-cell b=b' term)
# ---------------------------------------------------------------------------
#
# The estimator FFTs ``(field x weights)_j = sum_b m_b W_jb`` (map values at
# the cell positions, MAS-regridded, no interlacing).  The exact b=b'
# diagonal of its P0 cube is
#
#     D_0(k) = (V R / N_box^2) sum_b m_b^2 |W_b(k)|^2,   W_b(k) = sum_j W_jb e^{-ik.x_j},
#
# while the coherent models (mesh window / exact discrete window) carry their
# **own** diagonal
#
#     M_0(k) = (R / N_box^2) sum_q t(q) P(q) sum_b |W_b(k-q)|^2,
#
# whose per-cell variance is the mode_scale-suppressed sigma_t^2, not the map
# variance.  The difference D_0 - M_0 is the additive monopole "shot" floor.
# Both terms are computed exactly from
# the per-cell stencils via per-lag scalar sums:
#
#     |W_b(k)|^2 = sum_d e^{-2pi i d.n/N} P_d(b),   d = box-index lag between
#                                                   two stencil points of b,
#     B_d = sum_b m_b^2 P_d(b),   G_d = sum_b P_d(b)     (P_d real, even in d).
#
# Binned like the estimator (with the k1dweights):
#
#     bin_i[D_0] = (V R / N^2) sum_d B_d C_{i,d},
#     bin_i[M_0^(j)] = (R / N^2) sum_d T_d^(j) G_d C_{i,d},
#
# with C_{i,d} = (1/U_i) sum_{n in bin i} w_n e^{-2pi i d.n/N} and
# T_d^(j) = sum_{q in shell j} t(q) e^{+2pi i d.n_q/N} over the full grid.


def _map_cell_stencils(ps) -> tuple[NDArray[np.floating], NDArray[np.integer]]:
    """Exact MAS stencil of every map cell (no interlacing, shift=0).

    Replicates ``grid.project_particle_to_regular_grid`` for the cell
    positions ``ps.pix_coor_in_box`` with the configured ``ps.grid_scheme``.
    Returns ``(w, idx3)``: ``(n_cell, n_shift)`` weights and ``(n_cell,
    n_shift, 3)`` box indices (``-1`` components where the stencil is
    truncated at the box boundary).
    """
    from .grid import allowed_window_scheme, particle_to_mesh_distance, project_function

    pos = np.asarray(ps.pix_coor_in_box, dtype=float)
    box_len = np.asarray(ps.box_len, dtype=float)
    box_ndim = np.asarray(ps.box_ndim, dtype=int)
    s, indx = particle_to_mesh_distance(pos, box_len, box_ndim)
    indx = np.array(indx).T
    scheme = str(ps.grid_scheme)
    p = allowed_window_scheme.index(scheme)
    shift_limit = int(np.floor(p / 2 + 0.5))
    shifts = np.arange(-shift_limit, shift_limit + 1)
    sm = np.meshgrid(shifts, shifts, shifts, indexing="ij")
    shifts_arr = np.stack([sm[i].ravel() for i in range(3)], axis=1)
    n_sh = shifts_arr.shape[0]
    n_cell = pos.shape[0]
    w = np.zeros((n_cell, n_sh), dtype=np.float64)
    idx3 = np.full((n_cell, n_sh, 3), -1, dtype=np.int64)
    for i in range(n_sh):
        sh = shifts_arr[i]
        gf = (
            project_function(s[:, 0] + sh[0], scheme)
            * project_function(s[:, 1] + sh[1], scheme)
            * project_function(s[:, 2] + sh[2], scheme)
        )
        idx_shift = indx - sh[None, :]
        ok = np.all((idx_shift >= 0) & (idx_shift < box_ndim[None, :]), axis=1) & (
            gf > 0
        )
        w[:, i] = gf
        idx3[:, i] = idx_shift
        idx3[~ok, i] = -1
        w[~ok, i] = 0.0
    keep = w.sum(axis=0) > 0
    return w[:, keep], idx3[:, keep]


def _pair_lag_scalars(
    w: NDArray[np.floating],
    idx3: NDArray[np.integer],
    m2: ArrayLike,
) -> tuple[NDArray[np.integer], NDArray[np.floating]]:
    """Per-lag scalars from the ordered stencil pairs of every map cell.

    ``B_d = sum_b m2_b sum_{ordered pairs j-j'=d} w_j w_j'`` with ``d`` the
    box-index lag between the two stencil points of a cell (pairs with a
    truncated point are dropped).  Returns ``(lags, B)`` with ``lags``
    ``(n_lag, 3)`` in the ``(dx, dy, dz)`` ordering (dx slowest) and lag
    components in ``[-2, 2]`` (the CIC support difference).
    """
    n_cell, n_sh = w.shape
    d_off = np.arange(-2, 3)
    lags = np.array(
        [(dx, dy, dz) for dx in d_off for dy in d_off for dz in d_off], dtype=np.int64
    )
    n_lag = len(lags)
    B = np.zeros(n_lag, dtype=np.float64)
    valid = np.all(idx3 >= 0, axis=2)
    m2_np = np.asarray(m2, dtype=float)
    if m2_np.shape != (n_cell,):
        raise ValueError(f"m2 must have shape ({n_cell},) per map cell")
    for i in range(n_sh):
        wi = w[:, i]
        vi = valid[:, i]
        for j in range(n_sh):
            msk = vi & valid[:, j]
            if not msk.any():
                continue
            d = idx3[msk, i] - idx3[msk, j]
            dd = (d[:, 0] + 2) * 25 + (d[:, 1] + 2) * 5 + (d[:, 2] + 2)
            contrib = m2_np[msk] * wi[msk] * w[msk, j]
            B += np.bincount(dd, weights=contrib, minlength=n_lag)
    return lags, B


def _mode_index_grids(ps) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """rFFT-grid mode indices (n_i = k_i L_i / 2pi) and the full z indices."""
    box_len = np.asarray(ps.box_len, dtype=float)
    kx, ky, kz = ps.k_vec
    Nx, Ny, Nzr = np.shape(kx)[0], np.shape(ky)[0], np.shape(kz)[0]
    nxg = np.broadcast_to(
        (np.asarray(kx, float) * box_len[0] / (2.0 * np.pi)).reshape(Nx, 1, 1),
        (Nx, Ny, Nzr),
    )
    nyg = np.broadcast_to(
        (np.asarray(ky, float) * box_len[1] / (2.0 * np.pi)).reshape(1, Ny, 1),
        (Nx, Ny, Nzr),
    )
    nzg = np.broadcast_to(
        (np.asarray(kz, float) * box_len[2] / (2.0 * np.pi)).reshape(1, 1, Nzr),
        (Nx, Ny, Nzr),
    )
    nzf = np.arange(int(np.asarray(ps.box_ndim, dtype=int)[2]), dtype=float)
    return nxg, nyg, nzg, nzf


def _lag_phases(nxg, nyg, nzg, lags, nz_full) -> NDArray[np.complexfloating]:
    """e^{-2pi i d.n/N} on the rFFT grid for every lag.

    The z-mode indices run over the rFFT half-grid but the phase denominator
    is the FULL z length ``nz_full`` (the FFT lattice is periodic in Nz)."""
    Nx, Ny, Nzr = nxg.shape
    out = np.empty((len(lags), Nx, Ny, Nzr), dtype=np.complex64)
    for i, (dx, dy, dz) in enumerate(lags):
        out[i] = np.exp(
            -2.0j * np.pi * (dx * nxg / Nx + dy * nyg / Ny + dz * nzg / nz_full)
        ).astype(np.complex64, copy=False)
    return out


def _bin_lag_phases(
    ps, lags, bin_idx, valid, k1dweights, w_bin
) -> NDArray[np.complexfloating]:
    """C_{i,d} = (1/U_i) sum_{n in bin i} w_n e^{-2pi i d.n/N}."""
    nxg, nyg, nzg, nzf = _mode_index_grids(ps)
    Nx, Ny, Nzr = nxg.shape
    Nz = int(nzf.size)
    n_out = len(w_bin)
    C = np.zeros((n_out, len(lags)), dtype=np.complex128)
    bin_valid = bin_idx[valid]
    for i_d, (dx, dy, dz) in enumerate(lags):
        ph = np.exp(-2.0j * np.pi * (dx * nxg / Nx + dy * nyg / Ny + dz * nzg / Nz))
        wph = np.asarray(k1dweights, dtype=float) * ph.ravel()
        summed = np.zeros(n_out, dtype=np.complex128)
        np.add.at(summed, bin_valid, wph[valid])
        C[:, i_d] = summed / np.where(np.isfinite(w_bin) & (w_bin > 0), w_bin, 1.0)
    return C


def map_sampling_shot_diagonal(
    ps,
    *,
    weights: ArrayLike,
    mode_scale: ArrayLike | None,
    map_m2: ArrayLike,
    k_in: ArrayLike | None = None,
    stencils: tuple | None = None,
) -> dict:
    r"""
    Exact b=b' (diagonal) map-sampling "shot" terms of the mesh estimator.

    The estimator FFTs the MAS-regridded map; the exact diagonal of its P0
    cube is

    .. math::

        D_0(k) = \frac{V R}{N^2}\sum_b m_b^2 |W_b(k)|^2,

    while the coherent window models carry their own diagonal

    .. math::

        M_0(k) = \frac{R}{N^2}\sum_q t(q) P(q) \sum_b |W_b(k-q)|^2

    (the mode_scale-suppressed variance, not the map variance).  This helper
    returns the binned pieces needed to replace :math:`M_0` by :math:`D_0`
    (the additive monopole shot floor):

    - ``offset``: ``{0: bin[D_0](k_out)}`` — the theory-independent data
      diagonal (per-seed realized ``m_b^2``), added on top of the windowed
      multipoles;
    - ``cols``: ``(n_in, n_out)`` — ``bin_i[M_0^(j)]`` for every theory
      column ``j`` (the matrix subtraction that removes the model's own
      diagonal);
    - ``full``: ``bin[M_0]`` with the full-grid theory
      ``ps.auto_power_matter_model_r`` (when present) — the mirror for the
      exact discrete-window / smooth models, whose diagonal is the same
      operator evaluated on the full theory rather than the shells.

    ``map_m2`` are the per-cell map second moments in the
    ``ps.pix_coor_in_box`` order (the cells that ``grid_data_to_field``
    regrids).  ``stencils`` optionally carries ``(w, idx3)`` from
    :func:`_map_cell_stencils` (the geometry is deterministic, so it can be
    computed once and shared across seeds).  ``k_in`` is only needed for the
    per-column ``cols`` piece (``None`` returns ``cols=None``).
    """
    box_ndim = np.asarray(ps.box_ndim, dtype=int)
    n_grid = int(np.prod(box_ndim))
    V = float(np.prod(np.asarray(ps.box_len, dtype=float)))
    R = float(
        power_weights_renorm(
            np.asarray(weights, dtype=float), np.asarray(weights, dtype=float)
        )
    )
    if stencils is None:
        stencils = _map_cell_stencils(ps)
    w_st, idx3 = stencils
    lags, B = _pair_lag_scalars(w_st, idx3, np.asarray(map_m2, dtype=float))
    _, G = _pair_lag_scalars(w_st, idx3, np.ones(w_st.shape[0], dtype=float))

    # estimator-style binning (same convention as build_mesh_window_matrix)
    k_mode = np.asarray(ps.k_mode, dtype=float).ravel()
    k1dweights = (
        np.ones_like(k_mode)
        if getattr(ps, "k1dweights", None) is None
        else np.asarray(ps.k1dweights, dtype=float).ravel()
    )
    k1dbins = np.asarray(ps.k1dbins, dtype=float)
    n_out = len(k1dbins) - 1
    bin_idx = np.digitize(k_mode, k1dbins) - 1
    valid = (bin_idx >= 0) & (bin_idx < n_out) & (k1dweights > 0)
    w_bin = np.bincount(
        bin_idx[valid], weights=k1dweights[valid], minlength=n_out
    ).astype(float)
    C = _bin_lag_phases(ps, lags, bin_idx, valid, k1dweights, w_bin)

    pref_d = V * R / n_grid**2
    pref_m = R / n_grid**2
    offset0 = pref_d * np.real(C @ B)

    k_in_np = np.asarray(k_in, dtype=float) if k_in is not None else None
    cols = None
    if k_in_np is not None:
        shell_edges = np.concatenate(
            ([0.0], 0.5 * (k_in_np[:-1] + k_in_np[1:]), [np.inf])
        )
        # full-grid shell lag scalars: T_d^(j) = sum_{q in shell j} t(q) e^{2pi i d.n/N}
        nxg, nyg, nzg, nzf = _mode_index_grids(ps)
        Nx, Ny, Nzr = nxg.shape
        Nz = int(nzf.size)
        ms = (
            np.ones(k_mode.shape, dtype=float)
            if mode_scale is None
            else np.asarray(mode_scale, dtype=float).ravel()
        )
        if ms.shape != k_mode.shape:
            raise ValueError("mode_scale must match the rFFT grid shape")
        n_in = len(k_in_np)
        T = np.zeros((n_in, len(lags)), dtype=np.complex128)
        z_edge = (nzg[0, 0] == 0) | (nzg[0, 0] == Nzr - 1)  # (Nzr,) edge mask
        for j in range(n_in):
            shell = (k_mode >= shell_edges[j]) & (k_mode < shell_edges[j + 1])
            if not shell.any():
                continue
            t_s = ms * shell
            for i_d, (dx, dy, dz) in enumerate(lags):
                ph = np.exp(
                    2.0j * np.pi * (dx * nxg / Nx + dy * nyg / Ny + dz * nzg / Nz)
                )
                ph_m = np.exp(
                    2.0j * np.pi * (dx * nxg / Nx + dy * nyg / Ny - dz * nzg / Nz)
                )
                terms = t_s * (ph + np.where(z_edge, 0.0, ph_m)).ravel()
                T[j, i_d] = np.sum(terms)
        cols = pref_m * np.real((T * G[None, :]) @ C.T)  # (n_in, n_out)

    full = None
    if (
        hasattr(ps, "auto_power_matter_model_r")
        and ps.auto_power_matter_model_r is not None
    ):
        ms = (
            np.ones(k_mode.shape, dtype=float)
            if mode_scale is None
            else np.asarray(mode_scale, dtype=float).ravel()
        )
        nxg, nyg, nzg, nzf = _mode_index_grids(ps)
        Nx, Ny, Nzr = nxg.shape
        Nz = int(nzf.size)
        z_edge = (nzg[0, 0] == 0) | (nzg[0, 0] == Nzr - 1)
        pth = np.asarray(ps.auto_power_matter_model_r, dtype=float).ravel().copy()
        pth[0] = 0.0  # DC excluded, matching exact_window_models
        pth = pth * ms
        T_full = np.zeros(len(lags), dtype=np.complex128)
        # full-grid sum with the even z-extension of the (even) t·P grid
        for i_d, (dx, dy, dz) in enumerate(lags):
            ph = np.exp(2.0j * np.pi * (dx * nxg / Nx + dy * nyg / Ny + dz * nzg / Nz))
            ph_m = np.exp(
                2.0j * np.pi * (dx * nxg / Nx + dy * nyg / Ny - dz * nzg / Nz)
            )
            T_full[i_d] = np.sum(pth * (ph + np.where(z_edge, 0.0, ph_m)).ravel())
        full = pref_m * np.real(C @ (T_full * G))

    return {
        "offset": {0: offset0},
        "cols": cols,
        "full": full,
        "lags": lags,
        "stencils": stencils,
        "n_out": n_out,
    }


def _yamamoto_xi_kernels(w, xhat, ells_out, *, deconvolve_mas, wh_safe):
    """Real-space Yamamoto kernels ``xi[ℓ,m] = IFFT[FFT[w Y_ℓm] FFT[w]*]``."""
    w = np.asarray(w, dtype=float)
    shape = tuple(w.shape)
    w_tilde = np.fft.rfftn(w, norm="forward")
    if deconvolve_mas:
        w_tilde = w_tilde / wh_safe
    xi: dict[tuple[int, int], NDArray[np.complexfloating]] = {}
    for ell in ells_out:
        for m in range(-ell, ell + 1):
            ylm = get_real_Ylm(ell, m)
            c_rfft = np.fft.rfftn(w * ylm(*xhat), norm="forward")
            if deconvolve_mas:
                c_rfft = c_rfft / wh_safe
            c_rfft = c_rfft * np.conj(w_tilde)
            xi[(ell, m)] = np.fft.ifftn(_extend_hermitian_z(c_rfft, shape))
    return xi, w_tilde


def _execute_mesh_window(
    ps,
    k_in: ArrayLike,
    *,
    weights: ArrayLike,
    ells: Sequence[int] = (0, 2, 4),
    mode_scale: ArrayLike | None = None,
    out_mode_scale: ArrayLike | None = None,
    deconvolve_mas: bool = False,
    w_mas: ArrayLike | None = None,
    renorm_weights: ArrayLike | None = None,
    map_m2: ArrayLike | None = None,
    out_bin_weights: Sequence[ArrayLike] | None = None,
    out_group_index: ArrayLike | None = None,
    diag_correction: dict | None = None,
    in_bin_weights: Callable[[int, int], ArrayLike | None] | None = None,
    in_group_index: ArrayLike | None = None,
    in_group_scale: Sequence[ArrayLike] | None = None,
    leg_scale: dict | None = None,
    columns: Sequence[int] | Sequence[tuple[int, int]] | None = None,
    column_empty: Callable[[int, int], bool] | None = None,
    theory: ArrayLike | None = None,
    theory_kmu: ArrayLike | None = None,
    want: str = "matrix",
    progress=None,
):
    r"""Shared mesh-window column fill (``want='matrix'`` or ``'cubes'``).

    See :func:`build_mesh_window_matrix` and
    :func:`accumulate_mesh_window_cubes` for equations and parameters.
    Extra arguments: ``theory``, ``theory_kmu``, ``want``, ``progress``.
    """
    require_yamamoto_los(str(getattr(ps, "los", "endpoint")))
    want_s = str(want)
    if want_s not in ("matrix", "cubes"):
        raise ValueError("want must be 'matrix' or 'cubes'")
    ells_out_t = tuple(int(e) for e in ells)
    ells_in_t = (0,)
    k_in_np = np.asarray(k_in, dtype=float)
    _warn_truncated_mesh_k_in(ps, k_in_np)
    w = np.asarray(weights, dtype=float)
    shape = tuple(w.shape)
    n_grid = int(np.prod(shape))
    w_ren = w if renorm_weights is None else np.asarray(renorm_weights, dtype=float)
    if w_ren.shape != shape:
        raise ValueError("renorm_weights must match weights shape")
    R = float(power_weights_renorm(w_ren, w_ren))

    w_tilde = np.fft.rfftn(w, norm="forward")
    wh_safe = None
    if deconvolve_mas:
        if w_mas is None:
            raise ValueError("deconvolve_mas=True requires w_mas on the rFFT grid")
        wh = np.asarray(w_mas, dtype=float)
        if wh.shape != w_tilde.shape:
            raise ValueError(f"w_mas shape {wh.shape} != rFFT shape {w_tilde.shape}")
        wh_safe = np.where(np.abs(wh) > 1e-30, wh, 1.0)
        w_tilde = w_tilde / wh_safe

    khat = unit_khat_from_k_vec(ps.k_vec)
    xhat = ps.los_xhat
    nz = w_tilde.shape[2]

    k_mode = np.asarray(ps.k_mode, dtype=float).ravel()
    k1dweights = (
        np.ones_like(k_mode)
        if getattr(ps, "k1dweights", None) is None
        else np.asarray(ps.k1dweights, dtype=float).ravel()
    )
    k1dbins = np.asarray(ps.k1dbins, dtype=float)
    n_out = len(k1dbins) - 1
    if mode_scale is None:
        ms = np.ones(w_tilde.shape, dtype=float)
    else:
        ms = np.asarray(mode_scale, dtype=float)
        if ms.shape != w_tilde.shape:
            raise ValueError(
                "mode_scale must match the rFFT grid shape "
                f"(got {ms.shape}, expected {w_tilde.shape})"
            )

    if out_mode_scale is None:
        oms = np.ones(w_tilde.shape, dtype=float)
    else:
        oms = np.asarray(out_mode_scale, dtype=float)
        if oms.shape != w_tilde.shape:
            raise ValueError(
                "out_mode_scale must match the rFFT grid shape "
                f"(got {oms.shape}, expected {w_tilde.shape})"
            )

    # |k| shell of each theory node (Voronoi on k_in) and of each output bin
    shell_edges = np.concatenate(([0.0], 0.5 * (k_in_np[:-1] + k_in_np[1:]), [np.inf]))
    in_shell = [
        (k_mode >= shell_edges[j]) & (k_mode < shell_edges[j + 1])
        for j in range(len(k_in_np))
    ]
    pk_mu = None
    if theory_kmu is not None:
        pk_mu = np.asarray(theory_kmu, dtype=float)
        if pk_mu.shape != w_tilde.shape:
            raise ValueError(f"theory_kmu shape {pk_mu.shape} != rFFT {w_tilde.shape}")
    theory_j = None
    if theory is not None:
        theory_j = np.asarray(theory, dtype=float).reshape(-1)
        if theory_j.size != len(k_in_np):
            raise ValueError(f"theory length {theory_j.size} != n_k_in {len(k_in_np)}")
    acc_cubes = (
        {ell: np.zeros(w_tilde.shape, dtype=float) for ell in ells_out_t}
        if want_s == "cubes"
        else None
    )
    bin_idx = np.digitize(k_mode, k1dbins) - 1
    valid = (bin_idx >= 0) & (bin_idx < n_out) & (k1dweights > 0)
    w_bin = np.bincount(
        bin_idx[valid], weights=k1dweights[valid], minlength=n_out
    ).astype(float)
    w_bin[w_bin <= 0] = np.nan
    k_eff = np.bincount(
        bin_idx[valid], weights=(k_mode * k1dweights)[valid], minlength=n_out
    ) / np.where(np.isnan(w_bin), 1.0, w_bin)
    nmodes = np.bincount(bin_idx[valid], minlength=n_out).astype(float)

    matrix = np.zeros((len(ells_out_t) * n_out, len(k_in_np)), dtype=float)
    group_masks = None
    if in_bin_weights is not None:
        if out_bin_weights is not None:
            raise ValueError("in_bin_weights is incompatible with out_bin_weights")
        if map_m2 is not None:
            raise ValueError("in_bin_weights is incompatible with map_m2")
        if in_group_scale is not None and in_group_index is not None:
            raise ValueError("in_group_scale and in_group_index are alternatives")
        if in_group_scale is not None:
            scale_list = []
            for i_s, arr in enumerate(in_group_scale):
                a = np.asarray(arr, dtype=float)
                if a.size != k_mode.size:
                    raise ValueError(
                        f"in_group_scale[{i_s}] size {a.size} != n_mode {k_mode.size}"
                    )
                scale_list.append(a)
            n_gin = len(scale_list)
            gi_flat = None
        elif in_group_index is None:
            raise ValueError("in_bin_weights requires in_group_index or in_group_scale")
        else:
            gi_flat = np.asarray(in_group_index, dtype=np.int64).ravel()
            if gi_flat.size != k_mode.size:
                raise ValueError(
                    f"in_group_index size {gi_flat.size} != n_mode {k_mode.size}"
                )
            n_gin = int(gi_flat.max()) + 1
            scale_list = None
        weight_list = None
        xi = None
    elif out_bin_weights is None:
        weight_list = None
        xi, _ = _yamamoto_xi_kernels(
            w, xhat, ells_out_t, deconvolve_mas=deconvolve_mas, wh_safe=wh_safe
        )
    else:
        if map_m2 is not None:
            raise ValueError("out_bin_weights is incompatible with map_m2")
        if out_group_index is None:
            if len(out_bin_weights) != n_out:
                raise ValueError(
                    f"out_bin_weights length {len(out_bin_weights)} != n_out {n_out}"
                )
            g_flat = np.where(valid, bin_idx, -1)
        else:
            g_flat = np.asarray(out_group_index, dtype=np.int64).ravel()
            if g_flat.size != k_mode.size:
                raise ValueError(
                    f"out_group_index size {g_flat.size} != n_mode {k_mode.size}"
                )
            if int(g_flat.max()) + 1 > len(out_bin_weights):
                raise ValueError(
                    f"out_group_index has {int(g_flat.max()) + 1} groups but "
                    f"out_bin_weights has {len(out_bin_weights)}"
                )
        weight_list = []
        for i, wi in enumerate(out_bin_weights):
            arr = np.asarray(wi, dtype=float)
            if arr.shape != shape:
                raise ValueError(
                    f"out_bin_weights[{i}] shape {arr.shape} != weights {shape}"
                )
            weight_list.append(arr)
        group_masks = [valid & (g_flat == g) for g in range(len(weight_list))]

    leg_s = None
    if leg_scale is not None:
        if diag_correction is not None:
            raise ValueError("leg_scale and diag_correction are alternatives")
        leg_s = {}
        for key, arr in leg_scale.items():
            a = np.asarray(arr, dtype=float)
            if a.shape != w_tilde.shape:
                raise ValueError(
                    f"leg_scale[{key}] shape {a.shape} != rFFT {w_tilde.shape}"
                )
            leg_s[tuple(int(v) for v in key)] = a

    diag_c = None
    if diag_correction is not None:
        diag_c = {}
        for key, arr in diag_correction.items():
            a = np.asarray(arr, dtype=float)
            if a.shape != w_tilde.shape:
                raise ValueError(
                    f"diag_correction[{key}] shape {a.shape} != rFFT {w_tilde.shape}"
                )
            diag_c[tuple(int(v) for v in key)] = a

    def _theory_ifft(j, extra=None):
        # T(q) is real and even, so the Hermitian extension is q -> -q on
        # all three axes.  A z-only flip happens to agree for an isotropic
        # shell but not for a mu group, whose membership is set by the
        # observer LOS rather than the box axes.
        # extra may be a bool partition (mu/phi groups) or a float
        # theory weight (α_LM² on the whole shell).
        t_rfft = ms * in_shell[j].reshape(w_tilde.shape)
        if extra is not None:
            t_rfft = t_rfft * np.asarray(extra, dtype=float).reshape(w_tilde.shape)
        if pk_mu is not None:
            t_rfft = t_rfft * pk_mu
        return np.fft.ifftn(_extend_hermitian_z(t_rfft.astype(complex), shape))

    def _fill_from_xi(xi_use, xi_t, j, mask=None, accumulate=False, in_mask=None):
        sel = valid if mask is None else mask
        t_diag = None
        if diag_c is not None:
            # kappa = 0 means q = k, so an inner-mode group restricts which
            # output modes may take the diagonal — otherwise every group
            # would add it again.
            sel_in = in_shell[j] if in_mask is None else (in_shell[j] & in_mask)
            t_diag = ms * sel_in.reshape(w_tilde.shape)
            if pk_mu is not None:
                t_diag = t_diag * pk_mu
        amp = 1.0 if pk_mu is not None or theory_j is None else float(theory_j[j])
        for i_ell, ell in enumerate(ells_out_t):
            cube = np.zeros(w_tilde.shape, dtype=complex)
            for m in range(-ell, ell + 1):
                ylm = get_real_Ylm(ell, m)
                conv = np.fft.fftn(xi_use[(ell, m)] * xi_t) * n_grid
                term = conv[..., :nz]
                if leg_s is not None and (ell, m) in leg_s:
                    term = term * leg_s[(ell, m)]
                cube = cube + ylm(*khat) * term
                if t_diag is not None and (ell, m) in diag_c:
                    cube = cube + ylm(*khat) * (diag_c[(ell, m)] * t_diag)
            p3d = (4.0 * np.pi) * R * np.real(cube) * oms
            if want_s == "cubes":
                acc_cubes[ell] += amp * p3d
                continue
            binned = (
                np.bincount(bin_idx[sel], weights=p3d.ravel()[sel], minlength=n_out)
                / w_bin
            )
            rows = slice(i_ell * n_out, (i_ell + 1) * n_out)
            if mask is None and not accumulate:
                matrix[rows, j] = np.nan_to_num(binned)
            else:
                matrix[rows, j] += np.nan_to_num(binned)

    col_j = None
    col_gj = None
    if columns is not None:
        col_list = list(columns)
        if col_list and isinstance(col_list[0], (tuple, list, np.ndarray)):
            col_gj = {(int(g), int(j)) for g, j in col_list}
        else:
            col_j = {int(j) for j in col_list}

    if in_bin_weights is not None:
        empty_in = np.zeros(k_mode.size, dtype=bool)
        for g in range(n_gin):
            if gi_flat is not None:
                sel_g = gi_flat == g
                if not np.any(sel_g):
                    continue
                extra_g = sel_g
            else:
                sel_g = None
                extra_g = scale_list[g]
            for j in range(len(k_in_np)):
                if col_gj is not None and (g, j) not in col_gj:
                    continue
                if col_j is not None and j not in col_j:
                    continue
                if sel_g is not None and not np.any(in_shell[j] & sel_g):
                    continue
                if column_empty is not None and column_empty(j, g):
                    continue
                cube_g = in_bin_weights(j, g)
                if cube_g is None:
                    if progress is not None:
                        progress()
                    continue
                cube_g = np.asarray(cube_g, dtype=float)
                if cube_g.shape != shape:
                    raise ValueError(
                        f"in_bin_weights({j},{g}) shape {cube_g.shape} != {shape}"
                    )
                xi_g, _ = _yamamoto_xi_kernels(
                    cube_g,
                    xhat,
                    ells_out_t,
                    deconvolve_mas=deconvolve_mas,
                    wh_safe=wh_safe,
                )
                if sel_g is not None:
                    in_mask = sel_g
                else:
                    # α_LM² weights every shell mode; add κ=0 once.
                    in_mask = in_shell[j] if g == 0 else empty_in
                _fill_from_xi(
                    xi_g,
                    _theory_ifft(j, extra=extra_g),
                    j,
                    accumulate=True,
                    in_mask=in_mask,
                )
                if progress is not None:
                    progress()
    elif weight_list is None:
        for j in range(len(k_in_np)):
            if col_j is not None and j not in col_j:
                continue
            if col_gj is not None and j not in {jj for _, jj in col_gj}:
                continue
            _fill_from_xi(xi, _theory_ifft(j), j)
            if progress is not None:
                progress()
    else:
        j_iter = range(len(k_in_np))
        if col_j is not None:
            j_iter = [j for j in j_iter if j in col_j]
        xi_t_list = {j: _theory_ifft(j) for j in j_iter}
        for g, w_g in enumerate(weight_list):
            if not np.any(group_masks[g]):
                continue
            xi_g, _ = _yamamoto_xi_kernels(
                w_g, xhat, ells_out_t, deconvolve_mas=deconvolve_mas, wh_safe=wh_safe
            )
            for j, xi_t in xi_t_list.items():
                if col_gj is not None and (g, j) not in col_gj:
                    continue
                _fill_from_xi(xi_g, xi_t, j, mask=group_masks[g])

    shot_offset = None
    if want_s == "matrix" and map_m2 is not None and columns is None:
        # exact b=b' diagonal: replace the model's own diagonal (whose
        # per-cell variance is mode_scale-suppressed) with the data's actual
        # diagonal (the map variance).  The subtraction is per column; the
        # data diagonal is a theory-independent monopole offset.
        shot = map_sampling_shot_diagonal(
            ps,
            weights=weights,
            mode_scale=ms,
            map_m2=map_m2,
            k_in=k_in_np,
        )
        if shot["cols"].shape != (len(k_in_np), n_out):
            raise RuntimeError("shot diagonal column shape mismatch")
        matrix[0:n_out, :] -= shot["cols"].T
        shot_offset = shot["offset"]

    if want_s == "cubes":
        return acc_cubes
    return DiscreteShellWindowMatrix(
        matrix=matrix,
        k_in=k_in_np,
        k_out=k_eff,
        nmodes=nmodes,
        ells=ells_out_t,
        ells_in=ells_in_t,
        ells_out=ells_out_t,
        offset=shot_offset,
    )


def build_mesh_window_matrix(
    ps,
    k_in: ArrayLike,
    *,
    weights: ArrayLike,
    ells: Sequence[int] = (0, 2, 4),
    mode_scale: ArrayLike | None = None,
    out_mode_scale: ArrayLike | None = None,
    deconvolve_mas: bool = False,
    w_mas: ArrayLike | None = None,
    renorm_weights: ArrayLike | None = None,
    map_m2: ArrayLike | None = None,
    out_bin_weights: Sequence[ArrayLike] | None = None,
    out_group_index: ArrayLike | None = None,
    diag_correction: dict | None = None,
    in_bin_weights: Callable[[int, int], ArrayLike | None] | None = None,
    in_group_index: ArrayLike | None = None,
    in_group_scale: Sequence[ArrayLike] | None = None,
    leg_scale: dict | None = None,
    columns: Sequence[int] | Sequence[tuple[int, int]] | None = None,
    column_empty: Callable[[int, int], bool] | None = None,
) -> DiscreteShellWindowMatrix:
    r"""
    Exact mesh-level (FFT) window matrix for a local-LOS Yamamoto estimator.

    Replaces the smooth (Hankel/Wigner) continuous layer with the estimator's
    own mesh response.  For a windowed isotropic theory with selection
    :math:`w(x)` and the observer of ``ps.los_observer`` (:math:`\hat n(x)`
    from ``ps.los_xhat``), the estimator's 3D multipole cube has the exact
    ensemble mean

    .. math::

        \langle P_\ell^{3D}(\mathbf k)\rangle
        =
        4\pi R \sum_m Y_{\ell m}(\hat k)\,
        \bigl[\mathrm{FFT}[w\,Y_{\ell m}(\hat n(x))]\,
        \mathrm{FFT}[w]^* \;\circledast\; (t\,P_0)\bigr](\mathbf k),

    with :math:`R` the weight renorm and ``mode_scale`` :math:`t` multiplying
    the theory inside the convolution.  Each matrix column is the estimator's
    response to a **unit theory shell** at ``k_in[j]`` — the kernel convolved
    with the shell indicator — binned with the estimator's :math:`|k|` shells
    and ``k1dweights``, so ``apply({0: P_0(k_in)})`` returns the windowed
    multipoles.

    Exact for **any** LOS: the varying :math:`\hat n(x)` of a true lightcone
    observer is fully contained in :math:`Y_{\ell m}(\hat n(x))`; the
    discrete-:math:`\mu` projector of the smooth path is the
    :math:`1/d\to 0` limit (constant :math:`\hat n` reduces the matrix to the
    discrete-μ binning of :math:`(2\ell+1)L_\ell(\hat k\cdot\hat n)|w̃|^2`).

    Input multipoles: isotropic monopole only (``ells_in = (0,)``) — the
    response to :math:`\ell'>0` input would require an angular mesh injection
    (pypower ``MeshFFTWindow``-style; not implemented).  The matrix therefore
    maps :math:`P_0(k_{\mathrm{in}})\to P_\ell(k_{\mathrm{out}})`, which is
    sufficient for no-RSD theories (test 04).

    For a CIC (or other MAS) deposit of off-grid map cells the exact response
    factors as :math:`|W_{\mathrm{MAS}}(k)|^2` at the **output** mode times a
    convolution against the **raw** cell comb (no MAS).  Pass
    ``out_mode_scale = W_MAS(k)^2``, ``deconvolve_mas=True`` and
    ``w_mas = W_MAS(k)``, and keep only map-sampling (etc.) in
    ``mode_scale``.  Default behaviour (``out_mode_scale=None``,
    ``deconvolve_mas=False``) is unchanged.

    Parameters
    ----------
    ps :
        FieldPowerSpectrum-like object with ``k_vec``, ``k_mode``,
        ``x_vec``, ``los_observer`` (local ``los``) and the estimator's
        ``k1dbins`` / ``k1dweights``.
    k_in : array_like
        Fine theory :math:`|k|` nodes (matrix columns).
    weights : array_like
        The selection cube :math:`w` that multiplies the data in the
        estimator FFT (e.g. ``weights_1`` = lightcone counts).
    ells : sequence of int, default (0, 2, 4)
        Output multipoles (matrix rows).
    mode_scale : array_like, optional
        Same-k transfer on the **inner** (theory) mode ``q`` (e.g. the
        map-sampling kernel).  Do not put ``W_MAS^2`` here when using
        ``out_mode_scale`` / ``deconvolve_mas``.
    out_mode_scale : array_like, optional
        Per-Cartesian-mode factor on the 3D response at the **output**
        mode ``k`` before ``|k|``-shell binning (e.g. ``W_MAS(k)^2``).
    deconvolve_mas : bool, default False
        Divide ``FFT(weights)`` by ``w_mas`` to recover the raw cell comb.
    w_mas : array_like, optional
        MAS window ``W_hat(k)`` on the rFFT grid (not squared).  Required
        when ``deconvolve_mas`` is True.
    renorm_weights : array_like, optional
        Weights used only for ``R = power_weights_renorm`` (defaults to
        ``weights``).  Use the estimator's CIC counts here when
        ``weights`` is a raw/NGP comb.
    map_m2 : array_like, optional
        Per-cell map second moments :math:`m_b^2` (the cells that
        ``grid_data_to_field`` regrids, in the ``ps.pix_coor_in_box`` order).
        When given, the matrix includes the **Poisson-limit** map-sampling
        shot diagonal (see :func:`map_sampling_shot_diagonal`).  This is the
        Poisson estimate of a sub-Poissonian per-cell term and **must not
        be applied** as a lightcone P0 correction — it over-corrects.
        Kept for diagnostics / unit tests only.  Off by default.
    out_bin_weights : sequence of arrays, optional
        Real-space cubes, one per **output mode group** (the beam-in-kernel
        path: per-cell :math:`\tilde B_b(\mathbf k)` on the deposit
        comb).  Without ``out_group_index`` a group is one output
        :math:`|k|` bin and the length must be ``n_out``.
        ``renorm_weights`` still sets :math:`R`.  Incompatible with
        ``map_m2``.
    out_group_index : array_like, optional
        Int array on the rFFT grid assigning each estimator mode to a
        group of ``out_bin_weights`` (``-1`` = excluded).  Lets one output
        :math:`|k|` bin be split into several kernels (e.g. by
        :math:`|\mu|`, so the kernel keeps the beam's :math:`\hat k`
        anisotropy).  Each group contributes additively to its bin's row,
        with the **full** bin weight as denominator, so the shell average
        is unchanged.
    diag_correction : dict, optional
        ``{(ell, m): dS}`` on the rFFT grid, added to the kernel's
        :math:`\boldsymbol\kappa=0` term
        (:func:`~meer21cm.multipole_ops.beam_diagonal_correction`).
        The kernel's zero-lag value is
        :math:`\langle wY_{\ell m}\rangle\langle w\rangle`; a real-space
        cube can only carry a :math:`\hat k`-independent selection, so a
        mode-dependent one (the beam) needs this additive term to get the
        :math:`\ell>0` diagonal right.  Leakage stays on the cube kernel.
    in_bin_weights : callable, optional
        ``f(j, g) -> cube`` giving the selection for theory column ``j``
        and **inner** mode group ``g`` of ``in_group_index`` (``None`` to
        skip an empty intersection).  This is where a beam belongs: it
        multiplies the field at the theory mode :math:`\mathbf q`, before
        the selection, so each :math:`(|q|, \hat q)` group needs its own
        beamed cube (:func:`~meer21cm.multipole_ops.beam_theory_cell_kernels`).
        Groups sum into the same column.  Mutually exclusive with
        ``out_bin_weights`` / ``map_m2``.
    in_group_index : array_like, optional
        Int array on the rFFT grid assigning each theory mode to a group
        of ``in_bin_weights`` (``-1`` = excluded).  A partition: each
        mode belongs to one group.  Alternative to ``in_group_scale``.
    in_group_scale : sequence of arrays, optional
        Per-group theory weights on the rFFT grid (e.g.
        :math:`\alpha_{LM}(\hat q)^2` for a diagonal :math:`Y_{LM}`
        expansion).  Every group sees the **whole** theory shell, scaled
        by this array — not a partition.  Alternative to
        ``in_group_index``.  The :math:`\boldsymbol\kappa=0` diagonal
        correction is added once (on the first group).
    leg_scale : dict, optional
        ``{(ell, m): r}`` on the rFFT grid, multiplying the whole
        :math:`\boldsymbol\kappa` profile of that leg product rather than
        only its :math:`\boldsymbol\kappa=0` value (the ``ratio`` form of
        :func:`~meer21cm.multipole_ops.beam_theory_diagonal_correction`).
        Use instead of ``diag_correction`` when the correction has to
        reach the window leakage too.
    columns : sequence of int or (group, k_in) pairs, optional
        Fill only these matrix columns.  Integers select theory nodes
        ``j`` (no inner grouping).  ``(g, j)`` pairs select inner-mode
        groups of ``in_bin_weights``.  ``None`` fills every column.
        Chunks sum with :func:`accumulate_mesh_window_matrices`.
        ``map_m2`` is applied only on a full (``columns is None``) build.

    Returns
    -------
    mat : DiscreteShellWindowMatrix
        Dense :math:`P_0(k_{\mathrm{in}})\to P_\ell(k_{\mathrm{out}})`
        matrix.  Unbinned theory-baked cubes are
        :func:`accumulate_mesh_window_cubes`.

    References
    ----------
    Hand et al., 2017, AJ, 154, 199 (arXiv:1704.02357)
    """
    return _execute_mesh_window(
        ps,
        k_in,
        weights=weights,
        ells=ells,
        mode_scale=mode_scale,
        out_mode_scale=out_mode_scale,
        deconvolve_mas=deconvolve_mas,
        w_mas=w_mas,
        renorm_weights=renorm_weights,
        map_m2=map_m2,
        out_bin_weights=out_bin_weights,
        out_group_index=out_group_index,
        diag_correction=diag_correction,
        in_bin_weights=in_bin_weights,
        in_group_index=in_group_index,
        in_group_scale=in_group_scale,
        leg_scale=leg_scale,
        columns=columns,
        column_empty=column_empty,
        want="matrix",
    )


def accumulate_mesh_window_cubes(
    ps,
    k_in: ArrayLike,
    *,
    weights: ArrayLike,
    ells: Sequence[int] = (0, 2, 4),
    mode_scale: ArrayLike | None = None,
    out_mode_scale: ArrayLike | None = None,
    deconvolve_mas: bool = False,
    w_mas: ArrayLike | None = None,
    renorm_weights: ArrayLike | None = None,
    out_bin_weights: Sequence[ArrayLike] | None = None,
    out_group_index: ArrayLike | None = None,
    diag_correction: dict | None = None,
    in_bin_weights: Callable[[int, int], ArrayLike | None] | None = None,
    in_group_index: ArrayLike | None = None,
    in_group_scale: Sequence[ArrayLike] | None = None,
    leg_scale: dict | None = None,
    columns: Sequence[int] | Sequence[tuple[int, int]] | None = None,
    column_empty: Callable[[int, int], bool] | None = None,
    theory: ArrayLike | None = None,
    theory_kmu: ArrayLike | None = None,
    progress=None,
) -> dict[int, NDArray[np.floating]]:
    r"""
    Theory-baked 3D Yamamoto multipole cubes from the mesh-window operator.

    Same column loop as :func:`build_mesh_window_matrix`, but each column
    is scaled by theory and **summed unbinned** on the rFFT grid instead
    of being stored as a matrix column.  For isotropic theory

    .. math::

        P_\ell^{3D}(\mathbf k)
        =
        \sum_j P_0(k_{\mathrm{in},j})\,
        \mathcal{R}_\ell(\mathbf k;\,S_j),

    where :math:`\mathcal{R}_\ell(\mathbf k; S_j)` is the unit-shell
    response that becomes column :math:`j` of :math:`W` after
    :math:`|k|`-shell binning.  Consequently, with the same
    ``k1dweights``,

    .. math::

        \mathrm{bin}_{|k|}[P_\ell^{3D}] = (W P_0)_\ell.

    For Kaiser anisotropy pass ``theory_kmu``: the theory IFFT carries
    :math:`P(k,\mu)` on the rFFT grid (``amp = 1``; ``theory`` unused).
    Mesh :math:`W` still has ``ells_in = (0,)`` — anisotropic theory
    lives on these cubes, not in ``mode_scale``.

    Parameters
    ----------
    ps, k_in, weights, ells, mode_scale, out_mode_scale, ...
        Same kernel arguments as :func:`build_mesh_window_matrix`.
    theory : array_like, optional
        Isotropic :math:`P_0(k_{\mathrm{in}})` of length ``len(k_in)``.
        Ignored when ``theory_kmu`` is set.  If both are omitted every
        column contributes with amplitude 1 (sum of unit shells).
    theory_kmu : array_like, optional
        Anisotropic :math:`P(k,\mu)` on the rFFT grid (same shape as
        ``ps.k_mode``).  Typical source:
        :meth:`~meer21cm.model.ModelPowerSpectrum.power_kmu`.
    columns, column_empty, progress
        Column subset, skip-empty callback, and per-column progress hook
        (same as the validation 3D accumulator).

    Returns
    -------
    acc : dict
        ``{ell: ndarray}`` on the rFFT grid.

    References
    ----------
    Hand et al., 2017, AJ, 154, 199 (arXiv:1704.02357)
    """
    return _execute_mesh_window(
        ps,
        k_in,
        weights=weights,
        ells=ells,
        mode_scale=mode_scale,
        out_mode_scale=out_mode_scale,
        deconvolve_mas=deconvolve_mas,
        w_mas=w_mas,
        renorm_weights=renorm_weights,
        out_bin_weights=out_bin_weights,
        out_group_index=out_group_index,
        diag_correction=diag_correction,
        in_bin_weights=in_bin_weights,
        in_group_index=in_group_index,
        in_group_scale=in_group_scale,
        leg_scale=leg_scale,
        columns=columns,
        column_empty=column_empty,
        theory=theory,
        theory_kmu=theory_kmu,
        want="cubes",
        progress=progress,
    )


def ngp_raw_cell_comb(ps, particle_mass=None):
    r"""
    NGP deposit of map cells at ``ps.pix_coor_in_box`` (raw cell comb).

    For a CIC (or other MAS) regrid of off-grid cells the exact mesh-window
    response factors as :math:`|W_{\mathrm{MAS}}(k)|^2` at the **output**
    mode times a convolution against this **raw** comb (no MAS).  Pass the
    result as ``weights`` to :func:`build_mesh_window_matrix` with
    ``out_mode_scale = W_MAS(k)^2`` and ``renorm_weights`` = the
    estimator's CIC counts (see :func:`build_mesh_window_mas_out`).

    Parameters
    ----------
    ps :
        Object with ``pix_coor_in_box``, ``box_len``, ``box_ndim``.
    particle_mass : array_like, optional
        Per-cell masses (e.g. a pre-deposit frequency taper), length
        ``len(pix_coor_in_box)``.  Default: unit masses.

    Returns
    -------
    comb : ndarray
        Real-space NGP cube matching ``box_ndim``.
    """
    from .grid import project_particle_to_regular_grid

    pix = np.asarray(ps.pix_coor_in_box, dtype=float)
    if particle_mass is None:
        mass = np.ones(pix.shape[0], dtype=float)
    else:
        mass = np.asarray(particle_mass, dtype=float)
        if mass.shape != (pix.shape[0],):
            raise ValueError(
                f"particle_mass shape {mass.shape} != n_cell {pix.shape[0]}"
            )
    raw, _w, _c = project_particle_to_regular_grid(
        pix,
        np.asarray(ps.box_len, float),
        np.asarray(ps.box_ndim, int),
        particle_mass=mass,
        grid_scheme="nnb",
        average=False,
    )
    return np.asarray(raw, dtype=float)


def _mas_out_kernel_kwargs(
    ps,
    k_in,
    *,
    renorm_weights,
    ells=(0, 2, 4),
    mode_scale=None,
    particle_mass=None,
    raw_comb=None,
    out_mode_scale_extra=None,
    beam_at_output_mode=False,
    beam_at_theory_mode=False,
    beam_n_mu=4,
    beam_n_phi=1,
    beam_diag_correction=True,
    beam_diag_as_ratio=False,
    beam_l_max=None,
    beam_ylm=False,
    beam_ylm_lmax=2,
) -> dict:
    """Keyword arguments for :func:`build_mesh_window_matrix` (MAS-at-output)."""
    from .grid import fourier_window_for_assignment

    beam_at_output_mode = bool(beam_at_output_mode)
    beam_at_theory_mode = bool(beam_at_theory_mode)
    beam_diag_as_ratio = bool(beam_diag_as_ratio)

    w_mas2 = fourier_window_for_assignment(ps.box_ndim, ps.grid_scheme) ** 2
    out_bin_weights = None
    out_group_index = None
    diag_correction = None
    in_bin_weights = None
    in_group_index = None
    in_group_scale = None
    leg_scale = None
    if beam_ylm and not beam_at_theory_mode:
        raise ValueError("beam_ylm requires beam_at_theory_mode")
    if beam_at_theory_mode:
        if raw_comb is not None:
            raise ValueError("beam_at_theory_mode cannot be combined with raw_comb")
        from .multipole_ops import (
            beam_edge_cell_mass,
            beam_theory_cell_kernels,
            beam_theory_diagonal_correction,
            beam_ylm_alpha,
            beam_ylm_cell_kernels,
            beam_ylm_diagonal_correction,
        )

        edge = beam_edge_cell_mass(ps)
        if particle_mass is not None:
            extra_m = np.asarray(particle_mass, dtype=float)
            if extra_m.shape != edge.shape:
                raise ValueError(
                    f"particle_mass shape {extra_m.shape} != n_cell {edge.shape}"
                )
            edge = edge * extra_m
        use_ylm = bool(beam_ylm) and getattr(ps, "sigma_beam_ch", None) is not None
        if use_ylm:
            if beam_diag_as_ratio:
                raise ValueError("beam_ylm does not support beam_diag_as_ratio")
            labels, in_kernel = beam_ylm_cell_kernels(
                ps, k_in, l_max=int(beam_ylm_lmax), cell_mass=edge
            )
            alpha = beam_ylm_alpha(ps, labels)
            in_group_scale = [alpha[g] ** 2 for g in range(len(labels))]
            in_bin_weights = in_kernel
        else:
            in_group_index, in_kernel = beam_theory_cell_kernels(
                ps,
                k_in,
                n_mu=int(beam_n_mu),
                n_phi=int(beam_n_phi),
                mode_scale=mode_scale,
                cell_mass=edge,
            )
            in_bin_weights = in_kernel
        kernel = ngp_raw_cell_comb(ps, particle_mass=edge)
        if beam_diag_correction and getattr(ps, "sigma_beam_ch", None) is not None:
            if use_ylm:
                diag_correction = beam_ylm_diagonal_correction(
                    ps,
                    k_in,
                    ells=ells,
                    l_max_cube=int(beam_ylm_lmax),
                    l_max_beam=beam_l_max,
                    cell_mass=edge,
                )
            else:
                corr = beam_theory_diagonal_correction(
                    ps,
                    k_in,
                    ells=ells,
                    n_mu=int(beam_n_mu),
                    n_phi=int(beam_n_phi),
                    mode_scale=mode_scale,
                    cell_mass=edge,
                    l_max_beam=beam_l_max,
                    ratio=bool(beam_diag_as_ratio),
                )
                if beam_diag_as_ratio:
                    leg_scale = corr
                else:
                    diag_correction = corr
    elif beam_at_output_mode:
        if raw_comb is not None:
            raise ValueError("beam_at_output_mode cannot be combined with raw_comb")
        from .multipole_ops import (
            beam_diagonal_correction,
            beam_edge_cell_mass,
            beam_output_cell_masses,
        )

        masses, out_group_index = beam_output_cell_masses(ps, n_mu=int(beam_n_mu))
        edge = beam_edge_cell_mass(ps)
        if particle_mass is not None:
            extra_m = np.asarray(particle_mass, dtype=float)
            if extra_m.shape != edge.shape:
                raise ValueError(
                    f"particle_mass shape {extra_m.shape} != n_cell {edge.shape}"
                )
            edge = edge * extra_m
        out_bin_weights = [
            ngp_raw_cell_comb(ps, particle_mass=masses[g] * edge)
            for g in range(masses.shape[0])
        ]
        kernel = out_bin_weights[0]
        if beam_diag_correction:
            diag_correction = beam_diagonal_correction(
                ps,
                ells=ells,
                masses=masses,
                group_index=out_group_index,
                l_max_beam=beam_l_max,
                cell_mass=edge,
            )
    elif raw_comb is None:
        kernel = ngp_raw_cell_comb(ps, particle_mass=particle_mass)
    else:
        kernel = np.asarray(raw_comb, dtype=float)
    if out_mode_scale_extra is not None:
        extra = np.asarray(out_mode_scale_extra, dtype=float)
        if extra.shape != w_mas2.shape:
            raise ValueError(
                "out_mode_scale_extra shape "
                f"{extra.shape} != W_MAS^2 shape {w_mas2.shape}"
            )
        w_mas2 = w_mas2 * extra
    return dict(
        weights=kernel,
        ells=ells,
        mode_scale=mode_scale,
        out_mode_scale=w_mas2,
        renorm_weights=renorm_weights,
        out_bin_weights=out_bin_weights,
        out_group_index=out_group_index,
        diag_correction=diag_correction,
        in_bin_weights=in_bin_weights,
        in_group_index=in_group_index,
        in_group_scale=in_group_scale,
        leg_scale=leg_scale,
    )


def build_mesh_window_mas_out(
    ps,
    k_in: ArrayLike,
    *,
    renorm_weights: ArrayLike,
    ells: Sequence[int] = (0, 2, 4),
    mode_scale: ArrayLike | None = None,
    particle_mass: ArrayLike | None = None,
    raw_comb: ArrayLike | None = None,
    out_mode_scale_extra: ArrayLike | None = None,
    beam_at_output_mode: bool = False,
    beam_at_theory_mode: bool = False,
    beam_n_mu: int = 4,
    beam_n_phi: int = 1,
    beam_diag_correction: bool = True,
    beam_diag_as_ratio: bool = False,
    beam_l_max: int | None = None,
    beam_ylm: bool = False,
    beam_ylm_lmax: int = 2,
    columns: Sequence[int] | Sequence[tuple[int, int]] | None = None,
) -> DiscreteShellWindowMatrix:
    r"""
    Preferred MAS-at-output mesh window for lightcone CIC deposits.

    Builds :func:`build_mesh_window_matrix` with:

    - ``weights`` = NGP raw cell comb at ``pix_coor_in_box``
      (:func:`ngp_raw_cell_comb`), or ``raw_comb`` if given;
    - ``out_mode_scale`` = :math:`W_{\mathrm{MAS}}(k)^2`;
    - ``renorm_weights`` = the estimator's CIC counts (for ``R``);
    - ``mode_scale`` = map-sampling (etc.) only — do **not** put
      :math:`W_{\mathrm{MAS}}^2` here.

    The theory :math:`q` integral runs over the PS Fourier grid only.
    Extending it past the grid is negligible: out-of-zone :math:`q` enters
    weighted by the cell-comb power at large lag, which is
    :math:`\sim 10^{-6}` of the :math:`\kappa = 0` spike.

    Parameters
    ----------
    particle_mass :
        Optional per-cell masses for the NGP comb (e.g. a pre-deposit
        frequency taper or the binary-mask beam edge factor
        :func:`~meer21cm.multipole_ops.beam_edge_cell_mass`).
        Ignored when ``raw_comb`` is provided.
    out_mode_scale_extra :
        Optional extra factor at the **output** Fourier mode, multiplied
        onto :math:`W_{\mathrm{MAS}}(k)^2` (e.g. the scalar beam transfer
        :func:`~meer21cm.multipole_ops.beam_out_mode_scale`).  With
        ``beam_at_output_mode`` the default extra is the residual anisotropy
        :math:`\\bar B^2(k)/\\langle B\\rangle_{\\mathrm{bin}}^2`.
    beam_at_output_mode :
        If True, attach :math:`\tilde B_b(\mathbf k)` to the **output**
        Fourier mode (the deposit comb at the estimator :math:`k`).
        The cube kernel carries the mean-field cell mass
        :math:`\langle\tilde B_b\rangle/n_b` (leakage), and the exact
        :math:`\ell`-dependent **diagonal** is restored additively by
        :func:`~meer21cm.multipole_ops.beam_diagonal_correction`.
        A real-space cube can only hold a :math:`\hat k`-independent
        selection, but the beam is
        :math:`u_{\mathbf k}(x)=w(x)\tilde B_x(\mathbf k)`.
    beam_at_theory_mode :
        Preferred beam model.  Attaches
        :math:`\tilde B_b(\mathbf q)` to the **theory** mode, which is
        where the beam physically acts — it smooths the field before the
        selection multiplies it.  The theory shell is split into
        ``beam_n_mu`` :math:`|\mu|` groups, each with its own beamed cube
        (:func:`~meer21cm.multipole_ops.beam_theory_cell_kernels`), and
        the groups sum into the same column.  Curved sky and chromaticity
        are exact per cell.  The cube still cannot hold the beam's
        azimuthal structure, so the default is an additive
        :math:`\boldsymbol\kappa=0` correction
        (:func:`~meer21cm.multipole_ops.beam_theory_diagonal_correction`).
        Overrides ``beam_at_output_mode``.
    beam_n_mu :
        With ``beam_at_theory_mode``, number of :math:`|\mu|` groups of the
        **theory** mode (production default 4: enough to resolve
        \(k_\perp\)-dependent leakage).  With ``beam_at_output_mode``,
        number of :math:`|\mu|` sub-groups per output :math:`|k|` bin for
        the mean-field cube
        (:func:`~meer21cm.multipole_ops.beam_mode_group_index`); there
        it only affects the leakage and ``1`` is the measured optimum.
    beam_n_phi :
        Extra equal-count azimuth bins around \(\hat n_{\mathrm{ref}}\)
        (default 1: a further azimuth split does not recover the beam's
        azimuthal structure around each cell's own \(\hat n_b\)).
    beam_diag_correction :
        Apply the exact per-mode beam response of
        :func:`~meer21cm.multipole_ops.beam_theory_diagonal_correction`.
        Needed because no real-space cube can hold the beam's azimuthal
        structure: on its own the cube saturates at \(0.40\) of the exact
        \(\ell=2\) zero-lag response, however fine ``beam_n_mu`` is.
    beam_diag_as_ratio :
        If True, apply that correction as a **ratio** on the whole
        :math:`\boldsymbol\kappa` profile.  Production default is False:
        additive at :math:`\boldsymbol\kappa=0` only, so the
        \(n_\mu\)-split leakage is not rescaled.  Both are exact on the
        diagonal; the ratio assumes the beam's directional response is
        slowly varying across the window width and fights the grouping.
    beam_l_max :
        Highest beam multipole :math:`L` in the diagonal expansion.
        :math:`\ell` couples to :math:`L\ge\ell`; default
        ``max(ells) + 4`` for per-mode convergence.
    beam_ylm :
        Opt-in diagonal :math:`Y_{LM}` cubes
        (:func:`~meer21cm.multipole_ops.beam_ylm_cell_kernels`)
        instead of :math:`|\mu|` groups.  Off by default — production
        stays :math:`n_\mu=4` + additive :math:`\kappa=0`.  Requires
        ``beam_at_theory_mode``.  Incompatible with ``beam_diag_as_ratio``.
    beam_ylm_lmax :
        Highest even :math:`L` of the cubes (default 2: 6 cubes).
    """
    kw = _mas_out_kernel_kwargs(
        ps,
        k_in,
        renorm_weights=renorm_weights,
        ells=ells,
        mode_scale=mode_scale,
        particle_mass=particle_mass,
        raw_comb=raw_comb,
        out_mode_scale_extra=out_mode_scale_extra,
        beam_at_output_mode=beam_at_output_mode,
        beam_at_theory_mode=beam_at_theory_mode,
        beam_n_mu=beam_n_mu,
        beam_n_phi=beam_n_phi,
        beam_diag_correction=beam_diag_correction,
        beam_diag_as_ratio=beam_diag_as_ratio,
        beam_l_max=beam_l_max,
        beam_ylm=beam_ylm,
        beam_ylm_lmax=beam_ylm_lmax,
    )
    return build_mesh_window_matrix(ps, k_in, columns=columns, **kw)


def tapered_theory_beam_kernels(
    ps,
    k_in: ArrayLike,
    *,
    renorm_weights: ArrayLike,
    mas: str = "cic",
    taper_axes: Sequence[int] = (),
    ells: Sequence[int] = (0, 2, 4),
    mode_scale: ArrayLike | None = None,
    n_mu: int = 4,
    n_phi: int = 1,
    beam_diag_as_ratio: bool = False,
    taper_func=None,
) -> dict:
    r"""
    Mesh-window kwargs for post-deposit taper plus theory-mode beam.

    Shared cell masses :math:`m_b(\mathbf q)` come from
    :func:`~meer21cm.multipole_ops.beam_theory_cell_masses`
    (``n_mu=4``, additive :math:`\kappa=0`).  ``mas`` selects the
    deposit of those masses:

    - ``"cic"`` (production, model B) — :math:`T\times\mathrm{CIC}[m_b]`,
      ``out_mode_scale = 1``, and :math:`W_{\mathrm{MAS}}^2` stays on
      the **theory** shell together with map sampling (inner-mode pairing:
      CIC is already in the real-space cube).
    - ``"ngp"`` (diagnostic, model 07) — :math:`T\times\mathrm{NGP}[m_b]`,
      then ``out_mode_scale = W_{\mathrm{MAS}}^2`` (MAS at the **output**
      mode).  Wrong when :math:`T` varies on CIC scales.

    The returned dict is passed to :func:`build_mesh_window_matrix` or
    :func:`accumulate_mesh_window_cubes`.  It includes ``column_empty``.

    Parameters
    ----------
    ps :
        Lightcone ``PowerSpectrum``-like object with ``pix_coor_in_box``,
        ``sigma_beam_ch``, and a local LOS.
    k_in : array_like
        Theory :math:`|k|` nodes.
    renorm_weights : array_like
        Estimator weights used for :math:`R=\sum w^2` (typically the
        **already-tapered** CIC counts).
    mas : {'cic', 'ngp'}, default 'cic'
        Deposit of the theory-mode beam masses.
    taper_axes : sequence of int, default ()
        Axes of the Blackman–Harris product :func:`~meer21cm.grid.bh_taper_cube`.
        Empty means :math:`T=1`.
    ells, mode_scale, n_mu, n_phi, beam_diag_as_ratio
        Forwarded to the theory-mode beam kernels / diagonal correction.
    taper_func : callable, optional
        1-D window.  Default Blackman–Harris.

    Returns
    -------
    kwargs : dict
        Arguments for :func:`build_mesh_window_matrix` plus
        ``column_empty(j, g)``.

    Notes
    -----
    See ``misc/rsd_sims/window_formalism.md`` §11.1 for the two operators.
    """
    from .grid import (
        bh_taper_cube,
        fourier_window_for_assignment,
        project_particle_to_regular_grid,
    )
    from .multipole_ops import (
        beam_edge_cell_mass,
        beam_theory_cell_masses,
        beam_theory_diagonal_correction,
        map_sampling_mode_scale,
    )

    mas_s = str(mas).lower()
    if mas_s not in ("ngp", "cic"):
        raise ValueError(f"mas must be 'ngp' or 'cic', got {mas!r}")
    axes = tuple(int(a) for a in taper_axes)
    samp = mode_scale
    if samp is None:
        try:
            samp = map_sampling_mode_scale(ps, z_resolved=True)
        except Exception:
            samp = None
    kw = _mas_out_kernel_kwargs(
        ps,
        k_in,
        renorm_weights=renorm_weights,
        ells=ells,
        mode_scale=samp,
        beam_at_theory_mode=True,
        beam_n_mu=int(n_mu),
        beam_n_phi=int(n_phi),
        beam_diag_correction=True,
        beam_diag_as_ratio=bool(beam_diag_as_ratio),
    )
    shape = tuple(np.asarray(kw["weights"]).shape)
    taper = (
        bh_taper_cube(shape, axes, taper_func=taper_func)
        if axes
        else np.ones(shape, dtype=float)
    )
    edge = beam_edge_cell_mass(ps)
    _idx, _edges, mass_fn = beam_theory_cell_masses(
        ps,
        k_in,
        n_mu=int(n_mu),
        n_phi=int(n_phi),
        mode_scale=samp,
        cell_mass=edge,
    )

    def column_empty(j, g):
        return mass_fn(j, g) is None

    orig_kernel = kw["in_bin_weights"]
    if mas_s == "ngp":

        def in_kernel(j, g):
            cube = orig_kernel(j, g)
            if cube is None:
                return None
            return np.asarray(cube, dtype=float) * taper

        kw["in_bin_weights"] = in_kernel
    else:
        pix = np.asarray(ps.pix_coor_in_box, dtype=float)
        box_len = np.asarray(ps.box_len, float)
        box_ndim = np.asarray(ps.box_ndim, int)

        def in_kernel(j, g):
            mass = mass_fn(j, g)
            if mass is None:
                return None
            cube, _w, _c = project_particle_to_regular_grid(
                pix,
                box_len,
                box_ndim,
                particle_mass=mass,
                grid_scheme="cic",
                average=False,
                compensate=False,
            )
            return np.asarray(cube, dtype=float) * taper

        w_mas2 = fourier_window_for_assignment(ps.box_ndim, ps.grid_scheme) ** 2
        kw["out_mode_scale"] = np.ones_like(
            np.asarray(kw["out_mode_scale"], dtype=float)
        )
        ms = kw["mode_scale"]
        if ms is None:
            kw["mode_scale"] = w_mas2
        else:
            kw["mode_scale"] = np.asarray(ms, dtype=float) * w_mas2
        kw["in_bin_weights"] = in_kernel

    kw["column_empty"] = column_empty
    return kw


def accumulate_mesh_window_mas_out_cubes(
    ps,
    k_in: ArrayLike,
    *,
    renorm_weights: ArrayLike,
    theory: ArrayLike | None = None,
    theory_kmu: ArrayLike | None = None,
    ells: Sequence[int] = (0, 2, 4),
    mode_scale: ArrayLike | None = None,
    particle_mass: ArrayLike | None = None,
    raw_comb: ArrayLike | None = None,
    out_mode_scale_extra: ArrayLike | None = None,
    beam_at_output_mode: bool = False,
    beam_at_theory_mode: bool = False,
    beam_n_mu: int = 4,
    beam_n_phi: int = 1,
    beam_diag_correction: bool = True,
    beam_diag_as_ratio: bool = False,
    beam_l_max: int | None = None,
    beam_ylm: bool = False,
    beam_ylm_lmax: int = 2,
    columns: Sequence[int] | Sequence[tuple[int, int]] | None = None,
    progress=None,
) -> dict[int, NDArray[np.floating]]:
    r"""
    3D Yamamoto cubes for the MAS-at-output (untapered lightcone) operator.

    Same kernels as :func:`build_mesh_window_mas_out`; theory is baked
    with :func:`accumulate_mesh_window_cubes`.
    """
    kw = _mas_out_kernel_kwargs(
        ps,
        k_in,
        renorm_weights=renorm_weights,
        ells=ells,
        mode_scale=mode_scale,
        particle_mass=particle_mass,
        raw_comb=raw_comb,
        out_mode_scale_extra=out_mode_scale_extra,
        beam_at_output_mode=beam_at_output_mode,
        beam_at_theory_mode=beam_at_theory_mode,
        beam_n_mu=beam_n_mu,
        beam_n_phi=beam_n_phi,
        beam_diag_correction=beam_diag_correction,
        beam_diag_as_ratio=beam_diag_as_ratio,
        beam_l_max=beam_l_max,
        beam_ylm=beam_ylm,
        beam_ylm_lmax=beam_ylm_lmax,
    )
    return accumulate_mesh_window_cubes(
        ps,
        k_in,
        theory=theory,
        theory_kmu=theory_kmu,
        columns=columns,
        progress=progress,
        **kw,
    )


def predict_mesh_windowed_multipoles(
    ps,
    k_in: ArrayLike | None = None,
    *,
    renorm_weights: ArrayLike,
    ells: Sequence[int] = (0, 2, 4),
    mode_scale: ArrayLike | None = None,
    particle_mass: ArrayLike | None = None,
    raw_comb: ArrayLike | None = None,
    out_mode_scale_extra: ArrayLike | None = None,
    beam_at_output_mode: bool = False,
    beam_at_theory_mode: bool = False,
    beam_n_mu: int = 4,
    beam_n_phi: int = 1,
    beam_diag_correction: bool = True,
    beam_diag_as_ratio: bool = False,
    beam_ylm: bool = False,
    beam_ylm_lmax: int = 2,
    n_k_in: int = 80,
    nmu: int = 64,
) -> dict[int, NDArray[np.floating]]:
    """
    One-shot MAS-out mesh window applied to the isotropic theory monopole.

    Theory is ``get_theory_multipoles_kmu`` of ``ps`` on the matrix
    ``k_in``.  Returns ``{ell: P_ell(k_out)}``.
    """
    if k_in is None:
        k_in = propose_mesh_k_in(ps, n=int(n_k_in))
    k_in_np = np.asarray(k_in, dtype=float)
    mat = build_mesh_window_mas_out(
        ps,
        k_in_np,
        renorm_weights=renorm_weights,
        ells=ells,
        mode_scale=mode_scale,
        particle_mass=particle_mass,
        raw_comb=raw_comb,
        out_mode_scale_extra=out_mode_scale_extra,
        beam_at_output_mode=beam_at_output_mode,
        beam_at_theory_mode=beam_at_theory_mode,
        beam_n_mu=beam_n_mu,
        beam_n_phi=beam_n_phi,
        beam_diag_correction=beam_diag_correction,
        beam_diag_as_ratio=beam_diag_as_ratio,
        beam_ylm=beam_ylm,
        beam_ylm_lmax=beam_ylm_lmax,
    )
    theory0 = ps.get_theory_multipoles_kmu(mat.k_in, ells=(0,), nmu=int(nmu))["P_ell"][
        0
    ]
    return mat.apply({0: theory0})
