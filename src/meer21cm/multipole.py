r"""
Yamamoto estimator + windowed multipole model on one object.

:class:`MultipolePowerSpectrum` is the analogue of
:class:`~meer21cm.power.PowerSpectrum` for local-LOS multipoles: same
survey I/O and lightcone gridding, data multipoles from
:meth:`~meer21cm.estimator.FieldPowerSpectrum.measure_multipoles`, and a
windowed theory vector from :class:`WindowedMultipoleModel`.

The expensive window operator (selection, beam, map sampling, MAS) is
**not** built on access.  Call :meth:`run_window_matrix` (serial) or
:meth:`get_arg_list_for_window_columns` + :func:`run_window_column` +
:meth:`accumulate_window_columns` (external multiprocessing / MPI).
Changing beam or survey settings flags the cached matrix stale; theory
:math:`P(k,\mu)` updates automatically and is applied as ``W @ P_ell(k_in)``.

Mesh windows map isotropic :math:`P_0(k_{\mathrm{in}})` only
(``ells_in = (0,)``) when applying :math:`W`.  Kaiser :math:`P(k,\mu)`
on the mesh path is the 3D bake :meth:`MultipolePowerSpectrum.run_model_power_3d`
(theory IFFT carries ``power_kmu``).  Fitting Kaiser :math:`P_2,P_4`
with a 1D matrix uses ``window='smooth'``.

Helper functions (k-grids, beam / sampling transfers, selection fields)
live in :mod:`meer21cm.multipole_ops`.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import windows

from .estimator import (
    FieldPowerSpectrum,
    MultipoleMeasurement,
    MultipoleShellMap,
)
from .grid import LightconeGriddingMixin, bh_taper_cube, fourier_window_for_assignment
from .model import ModelPowerSpectrum
from .multipole_ops import (
    Tracer,
    _excess_zero_mode_window_power,
    _resolve_window_k_edges,
    _window_effective_weights,
    accumulate_window_multipoles,
    make_galaxy_poisson_mean_density,
    map_sampling_mode_scale,
    propose_k1dbins_window,
    propose_k_in,
    propose_window_measure_ells,
    run_smooth_window_realization,
)
from .power import _FieldModelGlueMixin
from .power_ops import bin_3d_to_1d, power_weights_renorm
from .util import tagging
from .wide_angle import propose_odd_wa_ells
from .window import (
    DiscreteShellWindowMatrix,
    WindowEllMap,
    _mas_out_kernel_kwargs,
    accumulate_mesh_window_matrices,
    accumulate_mesh_window_cubes,
    apply_discrete_shell_window_matrix,
    build_discrete_shell_window_matrix,
    build_mesh_window_matrix,
    list_mesh_window_columns,
    propose_mesh_k_in,
    require_yamamoto_los,
    run_mesh_window_columns,
    sum_mesh_window_cubes,
    tapered_theory_beam_kernels,
    window_zero_mode_power,
)

logger = logging.getLogger(__name__)

__all__ = [
    "MultipolePowerSpectrum",
    "SmoothWindowEstimator",
    "WindowedMultipoleModel",
    "init_window_column_worker",
    "run_window_column",
    "run_model_cube_column",
    "predict_windowed_multipoles",
    "propose_k_in",
    "propose_k1dbins_window",
    "propose_window_measure_ells",
    "map_sampling_mode_scale",
    "run_smooth_window_realization",
    "accumulate_window_multipoles",
]

WindowKind = Literal["mesh", "smooth"]

_WINDOW_NOT_BUILT = (
    "Window matrix has not been built. Call run_window_matrix() or "
    "get_arg_list_for_window_columns() then accumulate_window_columns()."
)
_WINDOW_STALE = (
    "Cached window matrix is stale (beam or survey settings changed). "
    "Call run_window_matrix() to rebuild."
)
_MODEL_CUBES_NOT_BUILT = (
    "3D model cubes have not been built. Call run_model_power_3d() or "
    "get_arg_list_for_model_cubes() then accumulate_model_cubes()."
)
_MESH_KAISER_NO_CUBES = (
    "window='mesh' with kaiser_rsd=True: W @ P0 has no Kaiser quadrupole. "
    "Call run_model_power_3d() and read model_multipoles (bins those cubes) "
    "or use window='smooth'."
)

_WINDOW_WORKER: MultipolePowerSpectrum | None = None


class WindowedMultipoleModel(ModelPowerSpectrum):
    """
    Continuous multipole theory with an optional discrete-shell window matrix.

    Starts from :meth:`~meer21cm.model.ModelPowerSpectrum.power_kmu` (cosmo +
    RSD only; **no** beam, map sampling, or MAS compensation — those belong in
    the window), forms unconvolved multipoles at fine :math:`k_{\\mathrm{in}}`
    by a continuous :math:`\\mu` integral, then optionally applies a
    :class:`~meer21cm.window.DiscreteShellWindowMatrix` (discrete
    :math:`\\mu` sampling via box-centre :math:`\\hat n_{\\mathrm{ref}}`;
    the leading-order Yamamoto binning, matching 3D→1D for identity
    :math:`W`). ``los='global'`` is not a window path.
    Does **not** use :func:`~meer21cm.power_ops.get_modelpk_conv`.

    Parameters
    ----------
    window_matrix : DiscreteShellWindowMatrix or ndarray, optional
        Pre-built discrete-shell window matrix.
    window_ells : sequence of int, default (0, 2, 4)
        Default multipoles for theory and matrix blocks.
    """

    def __init__(
        self,
        *args: Any,
        window_matrix: DiscreteShellWindowMatrix | ArrayLike | None = None,
        window_ells: Sequence[int] = (0, 2, 4),
        **params: Any,
    ) -> None:
        super().__init__(*args, **params)
        self.window_ells = tuple(int(e) for e in window_ells)
        self._window_matrix_obj: DiscreteShellWindowMatrix | None = None
        self._window_matrix_raw: NDArray[np.floating] | None = None
        if window_matrix is not None:
            self.set_window_matrix(window_matrix)

    @property
    def window_matrix(self) -> NDArray[np.floating] | None:
        """Dense matrix ``(n_ell * n_out, n_ell * n_in)``, or ``None``."""
        if self._window_matrix_obj is not None:
            return self._window_matrix_obj.matrix
        return self._window_matrix_raw

    @property
    def k_out(self) -> NDArray[np.floating] | None:
        """Estimator bin centres from the attached discrete-shell matrix."""
        if self._window_matrix_obj is not None:
            return self._window_matrix_obj.k_out
        return None

    @property
    def k_in(self) -> NDArray[np.floating] | None:
        """Fine theory ``k_in`` nodes from the attached discrete-shell matrix."""
        if self._window_matrix_obj is not None:
            return self._window_matrix_obj.k_in
        return None

    @property
    def k_in_window(self) -> NDArray[np.floating] | None:
        """Alias for :attr:`k_in` (historical name)."""
        return self.k_in

    def set_window_matrix(
        self, window_matrix: DiscreteShellWindowMatrix | ArrayLike
    ) -> None:
        """Attach a discrete-shell window matrix (object or raw ndarray)."""
        if isinstance(window_matrix, DiscreteShellWindowMatrix):
            self._window_matrix_obj = window_matrix
            self._window_matrix_raw = None
            self.window_ells = tuple(window_matrix.ells_out)
        else:
            self._window_matrix_obj = None
            self._window_matrix_raw = np.asarray(window_matrix, dtype=float)

    def build_window_matrix_from_shell(
        self,
        shell_map: MultipoleShellMap,
        k_window: ArrayLike,
        W_ell: WindowEllMap,
        k_in: ArrayLike,
        ells: Sequence[int] | None = None,
        wide_angle: bool = False,
        wa_d: float | None = None,
        wa_los: str | None = None,
        **kwargs: Any,
    ) -> DiscreteShellWindowMatrix:
        """Build and attach a :class:`DiscreteShellWindowMatrix`."""
        if ells is None:
            ells = self.window_ells
        ells_out = tuple(int(e) for e in ells)
        ells_in = kwargs.pop("ells_in", None)
        even = tuple(e for e in ells_out if e % 2 == 0)
        if wide_angle:
            if not even:
                raise ValueError(
                    "wide_angle requires at least one even output multipole"
                )
            odds = propose_odd_wa_ells(even)
            ells_in = tuple(sorted(set(even) | set(odds)))
            if wa_d is None:
                raise ValueError("wa_d is required when wide_angle=True")
            if wa_los is None:
                wa_los = "firstpoint"
        result = build_discrete_shell_window_matrix(
            shell_map,
            k_window,
            W_ell,
            k_in=k_in,
            ells=ells_out,
            ells_in=ells_in,
            **kwargs,
        )
        if wide_angle:
            result.resum_input_odd_wide_angle(los=wa_los, d=float(wa_d), ells_even=even)
        self.set_window_matrix(result)
        return result

    def get_theory_multipoles_kmu(
        self,
        k_in: ArrayLike,
        ells: Sequence[int] | None = None,
        nmu: int = 64,
        which: str = "auto_1",
    ) -> dict[str, Any]:
        """
        Continuous multipoles; defaults ``ells`` to :attr:`window_ells`.

        See :meth:`ModelPowerSpectrum.get_theory_multipoles_kmu`.
        """
        if ells is None:
            ells = self.window_ells
        return super().get_theory_multipoles_kmu(k_in, ells=ells, nmu=nmu, which=which)

    def get_model_multipoles(
        self,
        which: str = "auto_1",
        k_in: ArrayLike | None = None,
        ells: Sequence[int] | None = None,
        nmu: int = 64,
        apply_window: bool = True,
    ) -> dict[str, Any]:
        """
        Theory multipoles, optionally convolved with the window matrix.

        Parameters
        ----------
        which : {'auto_1', 'auto_2', 'cross'}
            Tracer combination.
        k_in : array_like, optional
            Fine theory ``k``. Defaults to ``k_in`` stored on the attached
            :class:`DiscreteShellWindowMatrix`.
        ells : sequence of int, optional
            Multipoles (must match the matrix if ``apply_window``).
        nmu : int, default 64
            Gauss–Legendre nodes for the continuous :math:`\\mu` integral.
        apply_window : bool, default True
            If True and a window matrix is set, apply it.

        Returns
        -------
        result : dict
            ``k``, ``P_ell``, ``ells``, ``window_applied``; when the
            matrix is applied also ``k_in``, ``nmodes``,
            ``P_ell_unconvolved``.

        Notes
        -----
        Continuous theory is a Gauss–Legendre :math:`\\mu` integral of
        :meth:`~meer21cm.model.ModelPowerSpectrum.power_kmu`.  The
        matrix maps :math:`P_{\\ell'}(k_{\\mathrm{in}})\\to
        P_\\ell(k_{\\mathrm{out}})`.  Mesh windows have
        ``ells_in=(0,)`` (isotropic :math:`P_0`); Kaiser fitting on
        mesh uses :meth:`MultipolePowerSpectrum.run_model_power_3d`
        rather than this product.

        References
        ----------
        Beutler, Castorina & Zhang, 2019, JCAP, 03, 040 (arXiv:1810.05051)
        """
        if ells is None:
            ells = self.window_ells
        if k_in is None:
            k_in = self.k_in_window
        if k_in is None:
            raise ValueError(
                "k_in is required when no DiscreteShellWindowMatrix is attached"
            )

        ells_theory = ells
        if apply_window and self._window_matrix_obj is not None:
            ells_theory = self._window_matrix_obj.ells_in
        raw = self.get_theory_multipoles_kmu(
            k_in, ells=ells_theory, nmu=nmu, which=which
        )
        if apply_window and self.window_matrix is not None:
            if self._window_matrix_obj is not None:
                convolved = self._window_matrix_obj.apply(raw["P_ell"])
                k_out = self._window_matrix_obj.k_out
                nmodes = self._window_matrix_obj.nmodes
                ells_out = self._window_matrix_obj.ells_out
            else:
                convolved = apply_discrete_shell_window_matrix(
                    raw["P_ell"], self.window_matrix, ells=ells
                )
                k_out = None
                nmodes = None
                ells_out = tuple(int(e) for e in ells)
            return {
                "k": k_out if k_out is not None else raw["k"],
                "k_in": raw["k"],
                "nmodes": nmodes,
                "P_ell": convolved,
                "P_ell_unconvolved": raw["P_ell"],
                "ells": ells_out,
                "window_applied": True,
            }
        return {
            "k": raw["k"],
            "k_in": raw["k"],
            "nmodes": None,
            "P_ell": raw["P_ell"],
            "ells": raw["ells"],
            "window_applied": False,
        }


class SmoothWindowEstimator:
    """
    Measure survey-window multipoles from the HI selection and/or galaxy randoms.

    Three distinct :math:`k` grids (do not conflate them):

    - ``k1dbins_window`` — fine bin **edges** used only to measure
      :math:`W_L(k)` (stored centres in :attr:`k_window` / :attr:`k` after
      :meth:`accumulate`). Should resolve the low-``k`` window peak.
    - ``k_in`` — fine theory nodes, passed to :meth:`build_window_matrix`
      (matrix columns).
    - ``k1dbins_out`` — coarse estimator bin **edges** (legacy
      ``ps.k1dbins``); shell map / matrix rows (:attr:`k_out`).

    Legacy ``k1dbins`` alone still sets both window and out edges (old
    behaviour). Prefer passing ``k1dbins_window`` and ``k1dbins_out``
    separately.

    For HI, :meth:`run_one` measures multipoles of ``weights_hi`` (the field
    that multiplies the data cube) and rescales them by
    :func:`~meer21cm.power_ops.power_weights_renorm` so :math:`W_L` is
    :math:`O(1)` while the data estimator still applies the same renorm.
    Galaxy / cross use the selection cube the same way; ``tot_num_galaxies``
    only sets optional Poisson sampling density (amplitude restored).
    ``seed`` matters only for those Poisson draws.

    Does **not** open an MPI or multiprocessing pool. Callers map
    :meth:`get_arg_list_for_seeds` (or :func:`run_smooth_window_realization`)
    externally, then :meth:`accumulate`. Use :meth:`build_window_matrix` to
    turn accumulated :math:`W_L(k)` (or an identity :math:`|k|`-rebin
    kernel) into a
    :class:`~meer21cm.window.DiscreteShellWindowMatrix`.

    Parameters
    ----------
    box_len : array_like
        Box lengths in Mpc.
    k1dbins : array_like, optional
        Legacy: if ``k1dbins_window`` / ``k1dbins_out`` are omitted, used for
        **both** window measurement and estimator ``k_out``.
    k1dbins_window : array_like, optional
        Fine bin edges for measuring :math:`W_L(k)`.
    k1dbins_out : array_like, optional
        Coarse estimator bin edges (matrix ``k_out``; usually ``ps.k1dbins``).
    ells : sequence, default (0, 2, 4)
        Multipoles to measure / store.
    los : {'endpoint', 'firstpoint'}, default 'endpoint'
        Yamamoto LOS for
        :class:`~meer21cm.estimator.FieldPowerSpectrum`.
        ``los='global'`` is rejected (use the 3D ``get_1d_power`` path).
    los_observer : array_like, optional
        Observer position for local Yamamoto LOS (Mpc). Defaults to a far
        observer along :math:`+z` so :math:`\\hat n_{\\mathrm{ref}}=\\hat z`
        (plane-parallel Yamamoto; identity :math:`W` matches 3D→1D).
        ``from_power_spectrum`` uses ``ps.los_observer`` or ``ps.box_origin``
        when present.
    tracer : {'hi', 'gal', 'cross'}
        HI selection auto, galaxy randoms auto, or HI×gal cross window.
    weights_hi : ndarray, optional
        IM selection / weight cube whose multipoles define the HI window.
    selection_mask : ndarray, optional
        Galaxy footprint (e.g. ``counts_in_box > 0``).
    dndz_box : ndarray, optional
        Per-voxel dN/dz weight for galaxy randoms.
    tot_num_galaxies : float, optional
        If set, Poisson-sample the galaxy selection at this expected count,
        then divide by ``N/∑sel`` so the FFT field has selection amplitude
        (not number-count amplitude). Omit for a deterministic selection FFT.
    mean_density : float, optional
        Constant mean density on the mask (alternative amplitude for the
        selection cube when ``tot_num_galaxies`` is omitted).
    weights_grid_1, weights_grid_2 : ndarray, optional
        Optional extra FFT grid weights (same role as estimator
        ``weights_1/2``). Leave ``None`` when ``weights_hi`` is already the
        full multiplicative selection (avoid double-counting).
    wide_angle : bool, default False
        If True, measure extra odd :math:`W_L` and resum wa_order=1 into
        the discrete-shell matrix (even theory input only).
    wa_d : float, optional
        Comoving distance to the effective redshift (Mpc) for wide-angle.
    wa_los : {'firstpoint', 'endpoint'}, optional
        Wide-angle LOS convention. Defaults to :attr:`los` when local.
    """

    def __init__(
        self,
        box_len,
        k1dbins=None,
        ells=(0, 2, 4),
        los="endpoint",
        los_observer=None,
        tracer="hi",
        weights_hi=None,
        selection_mask=None,
        dndz_box=None,
        tot_num_galaxies=None,
        mean_density=None,
        weights_grid_1=None,
        weights_grid_2=None,
        mean_center_1=False,
        mean_center_2=False,
        unitless_1=False,
        unitless_2=False,
        k1dbins_window=None,
        k1dbins_out=None,
        wide_angle=False,
        wa_d=None,
        wa_los=None,
    ):
        self.box_len = np.asarray(box_len, dtype=float)
        self.k1dbins_window, self.k1dbins_out = _resolve_window_k_edges(
            k1dbins, k1dbins_window, k1dbins_out
        )
        # Legacy alias: estimator / shell-map edges (not the W_L measure grid).
        self.k1dbins = self.k1dbins_out
        self.ells = tuple(int(e) for e in ells)
        self.los = require_yamamoto_los(los)
        if los_observer is None:
            # Far +z: n_ref = z, so identity W matches get_1d_power of the 3D cube.
            self.los_observer = np.array([0.0, 0.0, 1.0e12], dtype=float)
        else:
            self.los_observer = np.asarray(los_observer, dtype=float)
        self.wide_angle = bool(wide_angle)
        self.wa_d = None if wa_d is None else float(wa_d)
        self.wa_los = None if wa_los is None else str(wa_los).lower()
        self._ells_measure = propose_window_measure_ells(
            self.ells, wide_angle=self.wide_angle
        )
        self.tracer = str(tracer).lower()
        self.weights_hi = weights_hi
        self.selection_mask = selection_mask
        self.dndz_box = dndz_box
        self.tot_num_galaxies = tot_num_galaxies
        self.mean_density = mean_density
        self.weights_grid_1 = weights_grid_1
        self.weights_grid_2 = weights_grid_2
        self.mean_center_1 = mean_center_1
        self.mean_center_2 = mean_center_2
        self.unitless_1 = unitless_1
        self.unitless_2 = unitless_2

        self.k = None
        self.nmodes = None
        self.W_ell = None
        self.W_ell_std = None
        self.n_realizations = 0
        self.window_matrix: DiscreteShellWindowMatrix | None = None
        self.k_in = None
        self.shell_map: MultipoleShellMap | None = None

    @property
    def k_window(self) -> NDArray[np.floating] | None:
        """Measured :math:`W_L` wavenumber centres (after :meth:`accumulate`)."""
        return self.k

    @property
    def k_out(self) -> NDArray[np.floating] | None:
        """Estimator bin centres from the shell map / built window matrix."""
        if self.window_matrix is not None:
            return self.window_matrix.k_out
        if self.shell_map is not None:
            return np.asarray(self.shell_map.k_eff, dtype=float)
        return None

    @classmethod
    def from_power_spectrum(cls, ps, tracer="hi", ells=(0, 2, 4), **kwargs):
        """
        Build from a :class:`~meer21cm.power.PowerSpectrum`-like object.

        Uses ``weights_field_1`` (else ``counts_in_box``) as the HI selection
        field for window multipoles, and ``(selection > 0)`` as the default
        galaxy mask.

        ``k1dbins_out`` defaults to ``ps.k1dbins``. Pass a finer
        ``k1dbins_window`` for measuring :math:`W_L`; if omitted, it falls
        back to the same edges as ``k1dbins_out`` (legacy behaviour).
        """
        # Resolve the HI selection lazily so that objects without survey
        # pixel data (e.g. survey-free simulation boxes) never trigger the
        # ``counts_in_box`` computation when ``weights_hi`` is given.
        weights_hi = kwargs.pop("weights_hi", None)
        if weights_hi is None:
            weights_hi = getattr(ps, "weights_field_1", None)
        if weights_hi is None:
            weights_hi = getattr(ps, "counts_in_box", None)
        if weights_hi is None:
            weights_hi = getattr(ps, "weights_1", None)
        selection_mask = kwargs.pop(
            "selection_mask",
            None if weights_hi is None else (np.asarray(weights_hi) > 0),
        )
        k1dbins = kwargs.pop("k1dbins", None)
        k1dbins_out = kwargs.pop(
            "k1dbins_out", k1dbins if k1dbins is not None else ps.k1dbins
        )
        k1dbins_window = kwargs.pop("k1dbins_window", None)
        k_mode = getattr(ps, "k_mode", None)
        k_fund = None
        if k_mode is not None:
            kpos = np.asarray(k_mode, dtype=float)
            kpos = kpos[np.isfinite(kpos) & (kpos > 0)]
            if kpos.size:
                k_fund = float(np.min(kpos))
        if k1dbins_window is not None and k_fund is not None:
            kw = np.asarray(k1dbins_window, dtype=float).copy()
            kw[0] = max(float(kw[0]), k_fund)
            k1dbins_window = kw
        kwargs.setdefault("los", getattr(ps, "los", "endpoint"))
        if "los_observer" not in kwargs:
            los_observer = getattr(ps, "los_observer", None)
            if los_observer is None:
                los_observer = getattr(ps, "box_origin", None)
            kwargs["los_observer"] = los_observer
        if kwargs.get("wide_angle") and "wa_d" not in kwargs:
            origin = np.asarray(getattr(ps, "box_origin", 0.0), dtype=float).reshape(-1)
            if origin.size == 1:
                origin = np.full(3, float(origin.flat[0]))
            elif origin.size != 3:
                origin = np.pad(origin, (0, max(0, 3 - origin.size)))[:3]
            box_len_ps = np.asarray(ps.box_len, dtype=float).reshape(-1)
            kwargs["wa_d"] = float(np.linalg.norm(origin + 0.5 * box_len_ps))
        return cls(
            box_len=ps.box_len,
            k1dbins=k1dbins,
            k1dbins_window=k1dbins_window,
            k1dbins_out=k1dbins_out,
            ells=ells,
            tracer=tracer,
            weights_hi=weights_hi,
            selection_mask=selection_mask,
            weights_grid_1=kwargs.pop(
                "weights_grid_1", getattr(ps, "weights_grid_1", None)
            ),
            weights_grid_2=kwargs.pop(
                "weights_grid_2", getattr(ps, "weights_grid_2", None)
            ),
            **kwargs,
        )

    def _worker_kwargs(self):
        return dict(
            box_len=self.box_len,
            k1dbins_window=self.k1dbins_window,
            tracer=self.tracer,
            ells=self._ells_measure,
            los=self.los,
            los_observer=self.los_observer,
            weights_hi=self.weights_hi,
            selection_mask=self.selection_mask,
            dndz_box=self.dndz_box,
            tot_num_galaxies=self.tot_num_galaxies,
            mean_density=self.mean_density,
            weights_grid_1=self.weights_grid_1,
            weights_grid_2=self.weights_grid_2,
            mean_center_1=self.mean_center_1,
            mean_center_2=self.mean_center_2,
            unitless_1=self.unitless_1,
            unitless_2=self.unitless_2,
        )

    def get_arg_list_for_seeds(self, seed_list):
        """
        Build pickleable argument tuples for external batch mapping.

        Each tuple is ``(kwargs_dict, seed)`` suitable for
        ``run_smooth_window_realization(**kwargs, seed=seed)``.
        """
        base = self._worker_kwargs()
        return [(dict(base), int(seed)) for seed in seed_list]

    def run_one(self, seed):
        """Run a single realization (convenience wrapper)."""
        return run_smooth_window_realization(seed=seed, **self._worker_kwargs())

    def accumulate(self, results):
        """Average realization results into ``W_ell`` / :attr:`k_window`."""
        acc = accumulate_window_multipoles(results)
        self.k = acc.k
        self.nmodes = acc.nmodes
        self.W_ell = acc.W_ell
        self.W_ell_std = acc.W_ell_std
        self.n_realizations = acc.n_realizations
        return acc

    def zero_mode_window_power(self) -> float:
        """
        :math:`k=0` window power of the selection, same convention as :math:`W_L`.

        Added as a constant :math:`Q_0^{\\mathrm{DC}}=W(0)/V` when building
        the smooth matrix (the measured :math:`W_L` bins start at the
        fundamental and never include this mode).
        """
        volume = float(np.prod(np.asarray(self.box_len, dtype=float)))
        tracer = self.tracer
        w1 = None
        w2 = None
        if tracer in ("hi", "cross"):
            if self.weights_hi is None:
                return 0.0
            w1 = _window_effective_weights(self.weights_hi, self.weights_grid_1)
        if tracer in ("gal", "cross"):
            mask = self.selection_mask
            if mask is None and self.weights_hi is not None:
                mask = np.asarray(self.weights_hi) > 0
            if mask is None:
                return 0.0
            gal_sel = make_galaxy_poisson_mean_density(
                mask,
                dndz_box=self.dndz_box,
                mean_density=self.mean_density,
                tot_num_galaxies=None,
            )
            w2 = _window_effective_weights(gal_sel, self.weights_grid_2)
        if tracer == "hi":
            return window_zero_mode_power(w1, volume)
        if tracer == "gal":
            return window_zero_mode_power(w2, volume)
        return float(np.mean(w1) * np.mean(w2) * volume * power_weights_renorm(w1, w2))

    def make_shell_map(
        self,
        k1dweights=None,
    ) -> MultipoleShellMap:
        """
        Build a :class:`~meer21cm.estimator.MultipoleShellMap` for ``k_out``.

        Uses :attr:`k1dbins_out` (legacy estimator edges), not the fine
        :attr:`k1dbins_window` used to measure :math:`W_L`. The map is
        :math:`|k|`-shell membership for the discrete-shell window.
        :attr:`~meer21cm.estimator.MultipoleShellMap.mu` is
        :math:`\\hat k\\cdot\\hat n_{\\mathrm{ref}}` (the local-LOS reference
        direction, box centre) and is used as the discrete-:math:`\\mu`
        projector.
        """
        shape = None
        for candidate in (
            self.weights_hi,
            self.selection_mask,
            self.weights_grid_1,
            self.weights_grid_2,
        ):
            if candidate is not None:
                shape = np.asarray(candidate).shape
                break
        if shape is None:
            raise ValueError(
                "Cannot infer box_ndim for shell map; provide weights_hi, "
                "selection_mask, or weights_grid_1/2"
            )
        field = np.ones(shape, dtype=float)
        fps = FieldPowerSpectrum(
            field,
            self.box_len,
            los=self.los,
            los_observer=self.los_observer,
            _skip_specification=True,
        )
        self.shell_map = fps.multipole_bin_index_map(
            k1dbins=self.k1dbins_out,
            k1dweights=k1dweights,
            los=self.los,
        )
        return self.shell_map

    def build_window_matrix(
        self,
        k_in,
        shell_map: MultipoleShellMap | None = None,
        n_fftlog=512,
        continuous="smooth",
        wide_angle=None,
        wa_d=None,
        wa_los=None,
        **kwargs,
    ) -> DiscreteShellWindowMatrix:
        """
        Build a discrete-shell window matrix ``W_{ℓℓ'}(k_out, k_in)``.

        Parameters
        ----------
        k_in : array_like
            Fine theory :math:`k` nodes (matrix columns; independent of
            :attr:`k1dbins_window` and :attr:`k1dbins_out`).
        shell_map : MultipoleShellMap, optional
            Estimator shells for ``k_out``. Defaults to
            :meth:`make_shell_map` from :attr:`k1dbins_out`.
        continuous : {'smooth', 'identity'}, default 'smooth'
            ``'identity'`` needs no accumulated ``W_ell`` (no survey
            convolution). ``'smooth'`` requires :meth:`accumulate`
            first (uses measured :attr:`k_window` / :attr:`W_ell`).
            The outer discrete-shell sum always applies the
            discrete-:math:`\\mu` projector
            (:math:`\\mu=\\hat k\\cdot\\hat n_{\\mathrm{ref}}`, the leading-order
            Yamamoto binning / 3D→1D sampling).
        wide_angle : bool, optional
            If True, include wa_order=1 odd theory columns then resum so
            :meth:`~meer21cm.window.DiscreteShellWindowMatrix.apply`
            takes even Kaiser :math:`P_\\ell` only.
        wa_d : float, optional
            Comoving distance to the effective redshift (Mpc). Defaults to
            :attr:`wa_d` stored on this estimator.
        wa_los : {'firstpoint', 'endpoint'}, optional
            Wide-angle LOS. Defaults to :attr:`wa_los` or the estimator
            :attr:`los`.
        """
        continuous_s = str(continuous).lower()
        if continuous_s == "smooth" and (self.W_ell is None or self.k is None):
            raise RuntimeError(
                "Accumulate window multipoles before building a smooth matrix"
            )
        if shell_map is None:
            shell_map = self.shell_map
        if shell_map is None:
            shell_map = self.make_shell_map()
        self.k_in = np.asarray(k_in, dtype=float)
        do_wa = self.wide_angle if wide_angle is None else bool(wide_angle)
        ells_out = self.ells
        ells_in = ells_out
        even = tuple(e for e in ells_out if e % 2 == 0)
        if do_wa:
            if not even:
                raise ValueError(
                    "wide_angle requires at least one even output multipole"
                )
            odds = propose_odd_wa_ells(even)
            ells_in = tuple(sorted(set(even) | set(odds)))
            d_wa = self.wa_d if wa_d is None else float(wa_d)
            if d_wa is None:
                raise ValueError("wa_d is required when wide_angle=True")
            los_wa = wa_los if wa_los is not None else self.wa_los
            if los_wa is None:
                los_wa = (
                    self.los if self.los in ("firstpoint", "endpoint") else "firstpoint"
                )
        self.window_matrix = build_discrete_shell_window_matrix(
            shell_map,
            None if continuous_s == "identity" else self.k_window,
            None if continuous_s == "identity" else self.W_ell,
            k_in=self.k_in,
            ells=ells_out,
            ells_in=ells_in,
            continuous=continuous_s,
            n_fftlog=n_fftlog,
            **kwargs,
        )
        if do_wa:
            self.window_matrix.resum_input_odd_wide_angle(
                los=los_wa, d=d_wa, ells_even=even
            )
        return self.window_matrix


def init_window_column_worker(mps: MultipolePowerSpectrum) -> None:
    """Pool initializer: store the parent object for :func:`run_window_column`."""
    global _WINDOW_WORKER
    _WINDOW_WORKER = mps


def run_window_column(kwargs: dict, columns) -> DiscreteShellWindowMatrix:
    """
    Pickleable worker for one mesh (or smooth) window column chunk.

    ``kwargs`` is one dict from
    :meth:`MultipolePowerSpectrum.get_arg_list_for_window_columns`.
    For the mesh path with beam kernels, pass
    ``initializer=init_window_column_worker, initargs=(mps,)`` so the
    worker can reuse the parent's setup; ``kwargs`` may then omit
    unpickleable callables and ``run_window_column`` uses the stored
    object.
    """
    if kwargs.get("use_worker_object"):
        if _WINDOW_WORKER is None:
            raise RuntimeError(
                "run_window_column needs init_window_column_worker(mps) "
                "when use_worker_object is True"
            )
        if str(kwargs.get("kind", "mesh")).lower() == "cubes":
            return _WINDOW_WORKER._fill_model_cubes(columns)
        return _WINDOW_WORKER._fill_window_columns(columns)
    kind = str(kwargs.get("kind", "mesh")).lower()
    if kind == "mesh":
        return run_mesh_window_columns(kwargs["build_kwargs"], columns)
    if kind == "cubes":
        return accumulate_mesh_window_cubes(**kwargs["build_kwargs"], columns=columns)
    raise ValueError(f"unsupported window kind for column worker: {kind!r}")


def run_model_cube_column(kwargs: dict, columns) -> dict[int, np.ndarray]:
    """
    Pickleable worker for one mesh-window 3D-cube column chunk.

    ``kwargs`` is one dict from
    :meth:`MultipolePowerSpectrum.get_arg_list_for_model_cubes`.
    Beam kernels require
    ``initializer=init_window_column_worker, initargs=(mps,)``.
    """
    return run_window_column(kwargs, columns)


class MultipolePowerSpectrum(
    _FieldModelGlueMixin,
    LightconeGriddingMixin,
    FieldPowerSpectrum,
    WindowedMultipoleModel,
):
    r"""
    Combined Yamamoto field estimator and windowed multipole model.

    Analog of :class:`~meer21cm.power.PowerSpectrum` for local-LOS
    multipoles.  Survey / box / field constructor arguments are the same
    (see :class:`~meer21cm.power.PowerSpectrum`); defaults differ:
    ``los='endpoint'``, and beam / sky sampling / MAS compensation are
    **off** on the legacy 3D ``get_model_power_i`` path so those
    operators are not double-counted.  They live in the window.

    The window is **not** built on access (too expensive).  Call
    :meth:`run_window_matrix` for the 1D matrix :math:`W`, and/or
    :meth:`run_model_power_3d` for unbinned 3D theory cubes (cylindrical
    power, mesh Kaiser).  One operator config drives both.

    Typical usage::

        mps = MultipolePowerSpectrum.from_power_spectrum(
            ps, window="mesh", window_taper_axes=(2,),  # optional
        )
        mps.apply_taper_to_field(1, axis=[2])           # data, if tapering
        mps.run_window_matrix()                         # W (isotropic P0)
        data = mps.measure_multipoles(which="auto_1", ells=(0, 2, 4))
        model_1d = mps.model_multipoles                 # W @ P_ell
        mps.run_model_power_3d()                        # same operator, theory baked
        P0_cy = mps.get_cy_power(mps.model_multipole_power_3d(0))

    Mesh operator (unchanged physics, selected automatically):

    - Untapered CIC lightcone + ``sigma_beam_ch``: MAS-out + theory-mode
      beam (``n_mu=4``, additive :math:`\kappa=0`).
    - Taper and **no** beam: inner-mode CIC-then-T,
      ``mode_scale = W_MAS^2 × sampling``.
    - Taper and beam: CIC-then-T + theory-mode beam
      (:func:`~meer21cm.window.tapered_theory_beam_kernels` with
      ``mas='cic'``): :math:`T\times\mathrm{CIC}[m_b]`,
      ``out_mode_scale=1``, :math:`W_{\mathrm{MAS}}^2` on the theory
      shell.

    Mesh :math:`W` has ``ells_in=(0,)``.  Kaiser anisotropy on mesh is
    :meth:`run_model_power_3d` (``power_kmu`` on the theory IFFT) then
    bin the cubes — not ``W @ P_0``.  Smooth windows keep Hankel / Wigner
    :math:`W` times Kaiser :math:`P_\ell`.

    Do **not** pair Yamamoto data with inherited
    ``auto_power_tracer_*_model`` (those cubes use box-:math:`z` :math:`\mu`).

    Parameters
    ----------
    field_1, box_len, weights_*, k1dbins, tracer_bias_*, ...
        Same as :class:`~meer21cm.power.PowerSpectrum`.
    los : {'endpoint', 'firstpoint', 'global', 'midpoint'}, default 'endpoint'
        Line of sight for :meth:`~meer21cm.estimator.FieldPowerSpectrum.measure_multipoles`.
        Local Yamamoto is ``'endpoint'`` / ``'firstpoint'``.
    los_observer : array_like, optional
        Observer position in Mpc.  Defaults to :attr:`box_origin`.
    window : {'mesh', 'smooth'}, default 'mesh'
        ``'mesh'`` is the lightcone FFT window.  ``'smooth'`` is the
        Hankel / Wigner matrix (Kaiser :math:`P_\ell` in).
    window_ells : sequence of int, default (0, 2, 4)
        Output multipoles for :math:`W` and the 3D cubes.
    window_taper_axes : sequence of int, default ()
        Post-deposit Blackman–Harris axes.  Empty: untapered operator.
        With a dish beam this selects CIC-then-T + beam; without a beam,
        inner-mode CIC-then-T (no theory-mode beam).
    n_k_in : int, default 80
        Number of theory :math:`|k|` nodes
        (:func:`~meer21cm.window.propose_mesh_k_in`).
    beam_at_theory_mode : bool, default True
        Attach the dish beam at the **theory** Fourier mode (production).
    beam_n_mu : int, default 4
        :math:`|\mu|` groups for the theory-mode beam kernel.
    beam_n_phi : int, default 1
        Azimuthal groups for the theory-mode beam kernel.
    beam_diag_correction : bool, default True
        Additive :math:`\kappa=0` diagonal correction for :math:`\ell>0`.
    beam_diag_as_ratio : bool, default False
        If True, use the ratio form of the diagonal correction
        (rescales leakage; not production).
    beam_at_output_mode : bool, default False
        Older output-mode beam kernel (diagnostic).
    beam_ylm, beam_ylm_lmax
        Optional :math:`Y_{LM}` grouping of the theory-mode beam.
    theory_nmu : int, default 64
        Gauss–Legendre nodes for continuous :math:`P_\ell(k_{\mathrm{in}})`.
    wide_angle : bool, default False
        Smooth-window only: resum wa_order=1 odd wide-angle into :math:`W`.
    include_beam, include_sky_sampling, compensate : list, default all False
        Legacy 3D ``get_model_power_i`` flags.  Keep False on this class.
    kaiser_rsd : bool, default True
        RSD in ``power_kmu``.  On mesh, quadrupole/hexadecapole 1D needs
        :meth:`run_model_power_3d`.
    **params
        Forwarded to :class:`~meer21cm.cosmology.CosmologyCalculator`
        (survey, cosmology, …).

    Notes
    -----
    Column-parallel fills:
    :meth:`get_arg_list_for_window_columns` /
    :meth:`accumulate_window_columns` and
    :meth:`get_arg_list_for_model_cubes` /
    :meth:`accumulate_model_cubes`.  There is no in-library MPI.

    References
    ----------
    Hand et al., 2017, AJ, 154, 199 (arXiv:1704.02357)
    Beutler, Castorina & Zhang, 2019, JCAP, 03, 040 (arXiv:1810.05051)
    """

    _window_cache_names = (
        "_window_matrix_obj",
        "_window_matrix_raw",
        "_model_power_3d",
    )

    def __init__(
        self,
        field_1=None,
        box_len=None,
        weights_field_1=None,
        weights_grid_1=None,
        mean_center_1=False,
        unitless_1=False,
        field_2=None,
        weights_field_2=None,
        weights_grid_2=None,
        mean_center_2=False,
        unitless_2=False,
        k1dbins=None,
        kmode=None,
        mumode=None,
        tracer_bias_1=1.0,
        sigma_v_1=0.0,
        tracer_bias_2=None,
        sigma_v_2=0.0,
        include_beam=None,
        fog_profile="lorentz",
        cross_coeff=1.0,
        model_k_from_field=True,
        mean_amp_1=1.0,
        mean_amp_2=1.0,
        sampling_resol=None,
        include_sky_sampling=None,
        downres_factor_transverse=1.2,
        downres_factor_radial=2.0,
        init_box_from_map_data=False,
        box_buffkick=5,
        compensate=None,
        taper_func=windows.blackmanharris,
        kaiser_rsd=True,
        grid_scheme="nnb",
        interlace_shift=0.0,
        num_particle_per_pixel=1,
        seed=None,
        kperpbins=None,
        kparabins=None,
        flat_sky=False,
        flat_sky_padding=[0, 0, 0],
        k1dweights=None,
        los="endpoint",
        los_observer=None,
        window="mesh",
        window_ells=(0, 2, 4),
        beam_at_theory_mode=True,
        beam_n_mu=4,
        beam_n_phi=1,
        beam_diag_correction=True,
        beam_diag_as_ratio=False,
        beam_at_output_mode=False,
        beam_ylm=False,
        beam_ylm_lmax=2,
        theory_nmu=64,
        n_k_in=80,
        window_taper_axes=(),
        wide_angle=False,
        **params,
    ):
        if seed is None:
            seed = np.random.randint(0, 2**32)
        self.seed = seed
        self.num_particle_per_pixel = num_particle_per_pixel
        if field_1 is None:
            if "box_ndim" in params.keys():
                field_1 = np.ones(params["box_ndim"])
            else:
                field_1 = np.ones([1, 1, 1])
        if box_len is None:
            box_len = np.array([1, 1, 1])
        if include_beam is None:
            include_beam = [False, False]
        if include_sky_sampling is None:
            include_sky_sampling = [False, False]
        if compensate is None:
            compensate = [False, False]
        WindowedMultipoleModel.__init__(
            self,
            kmode=kmode,
            mumode=mumode,
            tracer_bias_1=tracer_bias_1,
            sigma_v_1=sigma_v_1,
            tracer_bias_2=tracer_bias_2,
            sigma_v_2=sigma_v_2,
            include_beam=include_beam,
            fog_profile=fog_profile,
            cross_coeff=cross_coeff,
            weights_field_1=weights_field_1,
            weights_field_2=weights_field_2,
            weights_grid_1=weights_grid_1,
            weights_grid_2=weights_grid_2,
            mean_amp_1=mean_amp_1,
            mean_amp_2=mean_amp_2,
            sampling_resol=sampling_resol,
            include_sky_sampling=include_sky_sampling,
            kaiser_rsd=kaiser_rsd,
            compensate=compensate,
            window_ells=window_ells,
            **params,
        )
        self.model_k_from_field = model_k_from_field
        FieldPowerSpectrum.__init__(
            self,
            field_1,
            box_len,
            weights_1=weights_grid_1,
            mean_center_1=mean_center_1,
            unitless_1=unitless_1,
            field_2=field_2,
            weights_2=weights_grid_2,
            mean_center_2=mean_center_2,
            unitless_2=unitless_2,
            los=los,
            los_observer=los_observer,
            _skip_specification=True,
        )
        if model_k_from_field:
            self.propagate_field_k_to_model()
        self.k1dbins = k1dbins
        self.kperpbins = kperpbins
        self.kparabins = kparabins
        self.downres_factor_transverse = downres_factor_transverse
        self.downres_factor_radial = downres_factor_radial
        init_attr = [
            "_rot_mat_sky_to_box",
            "_pix_coor_in_cartesian",
            "_counts_in_box",
            "_flat_sky",
            "_box_origin",
            "_box_voxel_redshift",
        ]
        for attr in init_attr:
            setattr(self, attr, None)
        self.upgrade_sampling_from_gridding = False
        self.box_buffkick = box_buffkick
        self.taper_func = taper_func
        if init_box_from_map_data:
            self.get_enclosing_box()
        self.grid_scheme = grid_scheme
        self.interlace_shift = interlace_shift
        self.flat_sky = flat_sky
        self.flat_sky_padding = flat_sky_padding
        self.k1dweights = k1dweights

        self._window_kind = str(window).lower()
        if self._window_kind not in ("mesh", "smooth"):
            raise ValueError("window must be 'mesh' or 'smooth'")
        self._beam_at_theory_mode = bool(beam_at_theory_mode)
        self._beam_n_mu = int(beam_n_mu)
        self._beam_n_phi = int(beam_n_phi)
        self._beam_diag_correction = bool(beam_diag_correction)
        self._beam_diag_as_ratio = bool(beam_diag_as_ratio)
        self._beam_at_output_mode = bool(beam_at_output_mode)
        self._beam_ylm = bool(beam_ylm)
        self._beam_ylm_lmax = int(beam_ylm_lmax)
        self.theory_nmu = int(theory_nmu)
        self.n_k_in = int(n_k_in)
        self._window_taper_axes = tuple(int(a) for a in window_taper_axes)
        self.wide_angle = bool(wide_angle)
        self._window_k_in: NDArray[np.floating] | None = None
        self._window_stale = False
        self._data_multipoles = None
        self._model_multipoles = None
        self._model_power_3d = None

    def clean_cache(self, attr):
        """Flag the window matrix stale instead of wiping it."""
        names = list(attr)
        kept = []
        hit_window = False
        for name in names:
            if name in self._window_cache_names or name == "_window_matrix":
                hit_window = True
            else:
                kept.append(name)
        if hit_window and getattr(self, "_window_matrix_obj", None) is not None:
            self._window_stale = True
        super().clean_cache(kept)

    def _warn_window_status(self) -> None:
        """Warn if the window matrix is missing or stale; do not rebuild it."""
        if self._window_matrix_obj is None and self._window_matrix_raw is None:
            logger.warning(_WINDOW_NOT_BUILT)
            return
        if self._window_stale:
            logger.warning(_WINDOW_STALE)

    @property
    @tagging("beam", "box", "nu", "window")
    def window_matrix(self) -> NDArray[np.floating] | None:
        """Dense window matrix, or ``None`` if :meth:`run_window_matrix` has not been called."""
        self._warn_window_status()
        return WindowedMultipoleModel.window_matrix.fget(self)

    @property
    def window_kind(self) -> str:
        """``'mesh'`` or ``'smooth'``."""
        return self._window_kind

    @window_kind.setter
    def window_kind(self, value: str) -> None:
        kind = str(value).lower()
        if kind not in ("mesh", "smooth"):
            raise ValueError("window must be 'mesh' or 'smooth'")
        self._window_kind = kind
        if "window_dep_attr" in dir(self):
            self.clean_cache(self.window_dep_attr)

    @property
    def beam_n_mu(self) -> int:
        """Number of :math:`|\\mu|` groups for the theory-mode beam kernel."""
        return self._beam_n_mu

    @beam_n_mu.setter
    def beam_n_mu(self, value: int) -> None:
        self._beam_n_mu = int(value)
        if "window_dep_attr" in dir(self):
            self.clean_cache(self.window_dep_attr)

    @property
    def beam_at_theory_mode(self) -> bool:
        """If True, attach the dish beam at the theory mode :math:`\\mathbf q`."""
        return self._beam_at_theory_mode

    @beam_at_theory_mode.setter
    def beam_at_theory_mode(self, value: bool) -> None:
        self._beam_at_theory_mode = bool(value)
        if "window_dep_attr" in dir(self):
            self.clean_cache(self.window_dep_attr)

    @property
    def beam_diag_as_ratio(self) -> bool:
        """If True, apply the :math:`\\kappa=0` beam correction as a ratio."""
        return self._beam_diag_as_ratio

    @beam_diag_as_ratio.setter
    def beam_diag_as_ratio(self, value: bool) -> None:
        self._beam_diag_as_ratio = bool(value)
        if "window_dep_attr" in dir(self):
            self.clean_cache(self.window_dep_attr)

    @property
    def beam_at_output_mode(self) -> bool:
        """If True, attach the dish beam at the output Fourier mode."""
        return self._beam_at_output_mode

    @beam_at_output_mode.setter
    def beam_at_output_mode(self, value: bool) -> None:
        self._beam_at_output_mode = bool(value)
        if "window_dep_attr" in dir(self):
            self.clean_cache(self.window_dep_attr)

    @property
    def window_taper_axes(self) -> tuple[int, ...]:
        """Box axes that use a post-deposit taper (inner-mode window)."""
        return self._window_taper_axes

    @window_taper_axes.setter
    def window_taper_axes(self, value) -> None:
        self._window_taper_axes = tuple(int(a) for a in value)
        if "window_dep_attr" in dir(self):
            self.clean_cache(self.window_dep_attr)

    @property
    @tagging("box", "field_1")
    def data_multipoles(self) -> MultipoleMeasurement:
        """Yamamoto (or global) data multipoles; built on first access."""
        if self._data_multipoles is None:
            self._data_multipoles = self.measure_multipoles(
                which="auto_1",
                k1dbins=self.k1dbins,
                ells=self.window_ells,
                k1dweights=self.k1dweights,
            )
        return self._data_multipoles

    @property
    @tagging("cosmo_model", "nu", "kmode", "mumode", "tracer_1", "rsd")
    def model_multipoles(self) -> dict[str, Any] | None:
        r"""
        Windowed theory multipoles from the current ``power_kmu``.

        - ``window='smooth'`` or mesh with ``kaiser_rsd=False``: ``W @ P_\ell(k_{\mathrm{in}})``
          after :meth:`run_window_matrix`.
        - ``window='mesh'`` and ``kaiser_rsd=True``: bins the 3D cubes from
          :meth:`run_model_power_3d` (Kaiser lives on the theory IFFT, not
          in mesh ``W``, which has ``ells_in=(0,)``).  If cubes are missing,
          logs a warning and falls back to ``W @ P_0`` when a matrix exists.

        Returns
        -------
        result : dict or None
            Keys ``k``, ``P_ell``, ``nmodes`` (and ``k_in`` / ``window_applied``
            on the matrix path).  ``None`` if neither ``W`` nor 3D cubes
            have been built.
        """
        mesh_kaiser = self.window_kind == "mesh" and bool(self.kaiser_rsd)
        if mesh_kaiser and self._model_power_3d is not None:
            return self._bin_model_power_3d()
        self._warn_window_status()
        if mesh_kaiser and self._model_power_3d is None:
            logger.warning(_MESH_KAISER_NO_CUBES)
        if self._window_matrix_obj is None and self._window_matrix_raw is None:
            return None
        if self._model_multipoles is None:
            self._model_multipoles = self.get_model_multipoles(
                which="auto_1",
                nmu=self.theory_nmu,
                apply_window=True,
            )
        return self._model_multipoles

    def _bin_model_power_3d(self) -> dict[str, Any]:
        """Bin cached 3D model cubes with the estimator ``k1dbins`` / weights."""
        p_ell: dict[int, NDArray[np.floating]] = {}
        k_eff = nmodes = None
        for ell in self.window_ells:
            p1, ke, nm = bin_3d_to_1d(
                self._model_power_3d[int(ell)],
                self.k_mode,
                self.k1dbins,
                weights=self.k1dweights,
                vectorize=False,
            )
            p_ell[int(ell)] = np.asarray(p1, dtype=float)
            k_eff, nmodes = ke, nm
        return {
            "k": np.asarray(k_eff, dtype=float),
            "P_ell": p_ell,
            "nmodes": np.asarray(nmodes, dtype=float),
            "window_applied": True,
            "from_cubes": True,
        }

    def _resolve_k_in(self) -> NDArray[np.floating]:
        """Theory :math:`k_{\\mathrm{in}}` nodes (cached, mesh, or smooth default)."""
        if self._window_k_in is not None:
            return np.asarray(self._window_k_in, dtype=float)
        if self.k1dbins is None:
            raise ValueError("k1dbins is required before building the window matrix")
        if self.window_kind == "mesh":
            return propose_mesh_k_in(self, n=self.n_k_in)
        return propose_k_in(self.k1dbins, n=self.n_k_in)

    def _renorm_weights(self) -> NDArray[np.floating]:
        """Estimator weights used for :math:`R = \\sum w^2`."""
        w = getattr(self, "weights_1", None)
        if w is None:
            w = getattr(self, "weights_grid_1", None)
        if w is None:
            return np.ones(self.box_ndim, dtype=float)
        return np.asarray(w, dtype=float)

    def _selection_weights(self) -> NDArray[np.floating]:
        """Real-space selection for an untapered mesh window."""
        w = getattr(self, "weights_field_1", None)
        if w is None:
            w = self._renorm_weights()
        return np.asarray(w, dtype=float)

    def _taper_cube(self, shape) -> NDArray[np.floating] | None:
        """Product taper over :attr:`window_taper_axes`, or ``None``."""
        axes = self._window_taper_axes
        if not axes:
            return None
        return bh_taper_cube(shape, axes, taper_func=self.taper_func)

    def _window_mode_scale(self) -> NDArray[np.floating] | None:
        """Map-sampling :math:`|S(k)|^2` on the rFFT grid, if available."""
        if not getattr(self, "has_resol", False):
            return None
        try:
            return map_sampling_mode_scale(self, z_resolved=True)
        except Exception:
            logger.debug("map_sampling_mode_scale unavailable; mode_scale=None")
            return None

    def _use_inner_mode_taper(self) -> bool:
        """Post-deposit taper is inner-mode, not MAS-out × (T×NGP)."""
        return bool(self._window_taper_axes)

    def _use_tapered_beam(self) -> bool:
        """CIC-then-T plus theory-mode beam (production taper + dish beam)."""
        return self._use_inner_mode_taper() and self._use_beam()

    def _inner_mode_scale(self) -> NDArray[np.floating]:
        """Inner-mode transfer :math:`W_{\\mathrm{MAS}}(k)^2` times map sampling."""
        w_mas2 = fourier_window_for_assignment(self.box_ndim, self.grid_scheme) ** 2
        samp = self._window_mode_scale()
        if samp is None:
            return w_mas2
        return w_mas2 * samp

    def _inner_mode_taper_weights(self) -> NDArray[np.floating]:
        """Estimator :math:`w(x)`: CIC counts after ``apply_taper_to_field``.

        ``window_taper_axes`` selects this inner-mode branch; it does not
        remultiply :math:`T`.  Call :meth:`apply_taper_to_field` (or set
        ``weights_1`` to already-tapered CIC) before building the window.
        """
        return np.asarray(self._renorm_weights(), dtype=float)

    def _use_mas_out(self) -> bool:
        """True when the untapered lightcone uses MAS-at-output NGP kernels."""
        if self._use_inner_mode_taper():
            return False
        pix = getattr(self, "_pix_coor_in_cartesian", None)
        if pix is None:
            return False
        return str(self.grid_scheme).lower() != "nnb"

    def _use_beam(self) -> bool:
        """True when a theory-mode dish beam is attached to the mesh window."""
        return (
            bool(self._beam_at_theory_mode)
            and getattr(self, "sigma_beam_ch", None) is not None
        )

    def _mesh_operator_kwargs(self) -> dict:
        """Kernel kwargs for :func:`build_mesh_window_matrix` (mesh only)."""
        k_in = self._resolve_k_in()
        ells = self.window_ells
        mode_scale = self._window_mode_scale()
        if self._use_tapered_beam():
            return tapered_theory_beam_kernels(
                self,
                k_in,
                renorm_weights=self._inner_mode_taper_weights(),
                mas="cic",
                taper_axes=self._window_taper_axes,
                ells=ells,
                mode_scale=mode_scale,
                n_mu=self._beam_n_mu,
                n_phi=self._beam_n_phi,
                beam_diag_as_ratio=self._beam_diag_as_ratio,
                taper_func=self.taper_func,
            )
        if self._use_inner_mode_taper():
            weights = self._inner_mode_taper_weights()
            return dict(
                weights=weights,
                ells=ells,
                mode_scale=self._inner_mode_scale(),
                renorm_weights=weights,
            )
        if self._use_mas_out():
            return _mas_out_kernel_kwargs(
                self,
                k_in,
                renorm_weights=self._renorm_weights(),
                ells=ells,
                mode_scale=mode_scale,
                beam_at_theory_mode=self._use_beam(),
                beam_at_output_mode=self._beam_at_output_mode and not self._use_beam(),
                beam_n_mu=self._beam_n_mu,
                beam_n_phi=self._beam_n_phi,
                beam_diag_correction=self._beam_diag_correction,
                beam_diag_as_ratio=self._beam_diag_as_ratio,
                beam_ylm=self._beam_ylm,
                beam_ylm_lmax=self._beam_ylm_lmax,
            )
        weights = self._selection_weights()
        return dict(
            weights=weights,
            ells=ells,
            mode_scale=mode_scale,
            renorm_weights=self._renorm_weights(),
        )

    def _fill_window_columns(self, columns=None) -> DiscreteShellWindowMatrix:
        """Build (a chunk of) the mesh or smooth window matrix."""
        k_in = self._resolve_k_in()
        ells = self.window_ells
        if self.window_kind == "smooth":
            swe = SmoothWindowEstimator.from_power_spectrum(
                self,
                ells=ells,
                wide_angle=self.wide_angle,
            )
            if columns is not None:
                raise ValueError(
                    "smooth window has no mesh columns; use run_window_matrix() "
                    "or SmoothWindowEstimator.get_arg_list_for_seeds"
                )
            swe.accumulate([swe.run_one(0)])
            return swe.build_window_matrix(k_in, continuous="smooth")
        kw = self._mesh_operator_kwargs()
        return build_mesh_window_matrix(self, k_in, columns=columns, **kw)

    def _theory_for_cubes(self):
        """``(theory_p0, theory_kmu)`` for :meth:`run_model_power_3d`."""
        k_in = self._resolve_k_in()
        if self.kaiser_rsd and self.window_kind == "mesh":
            pk = np.asarray(
                self.power_kmu(which="auto_1", include_mean_amp=True), dtype=float
            )
            return None, pk
        theory0 = self.get_theory_multipoles_kmu(k_in, ells=(0,), nmu=self.theory_nmu)[
            "P_ell"
        ][0]
        return np.asarray(theory0, dtype=float), None

    def _fill_model_cubes(self, columns=None) -> dict[int, np.ndarray]:
        """Build (a chunk of) theory-baked 3D mesh-window cubes."""
        if self.window_kind == "smooth":
            raise ValueError(
                "run_model_power_3d is the mesh-window 3D bake; "
                "smooth windows use model_multipoles = W @ P_ell"
            )
        k_in = self._resolve_k_in()
        kw = self._mesh_operator_kwargs()
        theory, theory_kmu = self._theory_for_cubes()
        return accumulate_mesh_window_cubes(
            self,
            k_in,
            theory=theory,
            theory_kmu=theory_kmu,
            columns=columns,
            **kw,
        )

    def run_window_matrix(self, columns=None) -> DiscreteShellWindowMatrix:
        r"""
        Build the window matrix in this process (serial).

        The operator is the same as :meth:`run_model_power_3d` (MAS-out
        + theory-mode beam, inner-mode CIC-then-T, or CIC-then-T + beam).
        Each column is the response to a **unit** theory shell; apply
        current theory with :attr:`model_multipoles` (``W @ P_\ell``).

        The matrix is **not** built on access — this call is required
        and can be expensive.  For column-parallel mesh builds use
        :meth:`get_arg_list_for_window_columns` and
        :meth:`accumulate_window_columns`.

        Parameters
        ----------
        columns : sequence of int or (group, k_in) pairs, optional
            Fill only these columns (see
            :func:`~meer21cm.window.build_mesh_window_matrix`).

        Returns
        -------
        mat : DiscreteShellWindowMatrix
            Attached as :attr:`window_matrix`.
        """
        mat = self._fill_window_columns(columns=columns)
        self.set_window_matrix(mat)
        self._window_k_in = np.asarray(mat.k_in, dtype=float)
        self._window_stale = False
        self._model_multipoles = None
        return mat

    def run_model_power_3d(self, columns=None) -> dict[int, np.ndarray]:
        r"""
        Build theory-baked 3D Yamamoto multipole cubes (serial).

        Same mesh operator as :meth:`run_window_matrix`, but each column
        is scaled by isotropic :math:`P_0(k_{\mathrm{in}})` or, when
        ``kaiser_rsd`` is True, by :meth:`~meer21cm.model.ModelPowerSpectrum.power_kmu`
        on the theory IFFT, and **summed unbinned**.  Use this for
        cylindrical power (:meth:`~meer21cm.power.PowerSpectrum.get_cy_power`)
        and for mesh Kaiser 1D (:attr:`model_multipoles` bins the cubes).

        Parameters
        ----------
        columns : sequence of int or (group, k_in) pairs, optional
            Fill only these columns.

        Returns
        -------
        acc : dict
            ``{ell: ndarray}`` on the rFFT grid.
        """
        acc = self._fill_model_cubes(columns=columns)
        self._model_power_3d = acc
        self._model_multipoles = None
        return acc

    def model_multipole_power_3d(self, ell: int) -> NDArray[np.floating] | None:
        r"""
        Cached 3D model multipole cube for one :math:`\ell`.

        Analogous to :meth:`~meer21cm.estimator.FieldPowerSpectrum.multipole_power_3d`
        for the **data**.  Returns ``None`` (and logs a warning) until
        :meth:`run_model_power_3d` has been called.

        Parameters
        ----------
        ell : int
            Multipole order.

        Returns
        -------
        cube : ndarray or None
            rFFT-shaped :math:`P_\ell^{3D}(\mathbf k)`.
        """
        if self._model_power_3d is None:
            logger.warning(_MODEL_CUBES_NOT_BUILT)
            return None
        return np.asarray(self._model_power_3d[int(ell)], dtype=float)

    def _list_mesh_columns(self):
        """Column indices (or ``(group, j)`` pairs) for a parallel mesh fill."""
        k_in = self._resolve_k_in()
        n_in = int(np.asarray(k_in).size)
        if self.window_kind != "mesh":
            return list(range(n_in))
        kw = self._mesh_operator_kwargs()
        in_bin = kw.get("in_bin_weights")
        if in_bin is None:
            return list(range(n_in))
        k_mode = np.asarray(self.k_mode, dtype=float).ravel()
        k_in_np = np.asarray(k_in, dtype=float)
        edges = np.concatenate(([0.0], 0.5 * (k_in_np[:-1] + k_in_np[1:]), [np.inf]))
        in_shell = [
            (k_mode >= edges[j]) & (k_mode < edges[j + 1]) for j in range(len(k_in_np))
        ]
        return list_mesh_window_columns(
            n_in,
            in_group_index=kw.get("in_group_index"),
            in_group_scale=kw.get("in_group_scale"),
            in_bin_weights=in_bin,
            in_shell=in_shell,
            column_empty=kw.get("column_empty"),
        )

    def _chunk_mesh_columns(self, n_chunks: int | None):
        cols = self._list_mesh_columns()
        n = max(1, int(n_chunks) if n_chunks is not None else len(cols) or 1)
        n = min(n, max(1, len(cols)))
        chunks: list[list] = []
        if cols:
            size, extra = divmod(len(cols), n)
            idx = 0
            for i in range(n):
                take = size + (1 if i < extra else 0)
                if take:
                    chunks.append(list(cols[idx : idx + take]))
                    idx += take
        return chunks

    def get_arg_list_for_window_columns(
        self, n_chunks: int | None = None
    ) -> list[tuple[dict, Any]]:
        """
        Pickleable ``(kwargs, columns)`` tuples for external mapping.

        Mesh without beam kernels: ``kwargs`` is a dict for
        :func:`~meer21cm.window.run_mesh_window_columns`.  Mesh with
        theory-mode beam (or CIC-then-T + beam) kernels sets
        ``use_worker_object=True`` so the pool must be started with
        :func:`init_window_column_worker`.

        Parameters
        ----------
        n_chunks : int, optional
            Number of column chunks.  Default: one chunk per column.

        Returns
        -------
        args : list of (dict, columns)
            Map with :func:`run_window_column`.
        """
        if self.window_kind == "smooth":
            swe = SmoothWindowEstimator.from_power_spectrum(
                self, ells=self.window_ells, wide_angle=self.wide_angle
            )
            return swe.get_arg_list_for_seeds([0])

        chunks = self._chunk_mesh_columns(n_chunks)
        kw = self._mesh_operator_kwargs()
        if kw.get("in_bin_weights") is not None:
            return [({"use_worker_object": True, "kind": "mesh"}, ch) for ch in chunks]
        build_kwargs = dict(ps=self, k_in=self._resolve_k_in(), **kw)
        return [({"kind": "mesh", "build_kwargs": build_kwargs}, ch) for ch in chunks]

    def accumulate_window_columns(
        self, results: Sequence[DiscreteShellWindowMatrix]
    ) -> DiscreteShellWindowMatrix:
        """
        Sum column chunks and attach the window matrix.

        Parameters
        ----------
        results : sequence of DiscreteShellWindowMatrix
            Outputs of :func:`run_window_column` (or
            :func:`~meer21cm.window.build_mesh_window_matrix` with
            ``columns=``).

        Returns
        -------
        mat : DiscreteShellWindowMatrix
            Attached as :attr:`window_matrix`.
        """
        mat = accumulate_mesh_window_matrices(list(results))
        self.set_window_matrix(mat)
        self._window_k_in = np.asarray(mat.k_in, dtype=float)
        self._window_stale = False
        self._model_multipoles = None
        return mat

    def get_arg_list_for_model_cubes(
        self, n_chunks: int | None = None
    ) -> list[tuple[dict, Any]]:
        """
        Pickleable ``(kwargs, columns)`` tuples for 3D cube column chunks.

        Same operator as :meth:`get_arg_list_for_window_columns`.  Map
        with :func:`run_model_cube_column` (or :func:`run_window_column`
        with ``kind='cubes'``).  Beam kernels need
        :func:`init_window_column_worker`.

        Parameters
        ----------
        n_chunks : int, optional
            Number of column chunks.

        Returns
        -------
        args : list of (dict, columns)
        """
        if self.window_kind == "smooth":
            raise ValueError(
                "3D model cubes are the mesh-window bake; "
                "smooth windows use W @ P_ell"
            )
        chunks = self._chunk_mesh_columns(n_chunks)
        kw = self._mesh_operator_kwargs()
        if kw.get("in_bin_weights") is not None:
            return [({"use_worker_object": True, "kind": "cubes"}, ch) for ch in chunks]
        theory, theory_kmu = self._theory_for_cubes()
        build_kwargs = dict(
            ps=self,
            k_in=self._resolve_k_in(),
            theory=theory,
            theory_kmu=theory_kmu,
            **kw,
        )
        return [({"kind": "cubes", "build_kwargs": build_kwargs}, ch) for ch in chunks]

    def accumulate_model_cubes(
        self, results: Sequence[dict[int, np.ndarray]]
    ) -> dict[int, np.ndarray]:
        """
        Sum 3D cube column chunks and cache them.

        Parameters
        ----------
        results : sequence of dict
            ``{ell: ndarray}`` partials from :func:`run_model_cube_column`.

        Returns
        -------
        acc : dict
            ``{ell: ndarray}`` on the rFFT grid.
        """
        acc = sum_mesh_window_cubes(list(results))
        self._model_power_3d = acc
        self._model_multipoles = None
        return acc

    @classmethod
    def from_power_spectrum(cls, ps, **kwargs) -> MultipolePowerSpectrum:
        """
        Copy box, field, weights, bins, LOS, and lightcone state from a
        PowerSpectrum-like object (including :class:`~meer21cm.mock.MockSimulation`).

        Also copies ``kperpbins`` / ``kparabins``, ``sigma_beam_ch`` /
        ``beam_model``, and ``kaiser_rsd`` (overridable via ``kwargs``).
        Window flags (``window``, ``window_taper_axes``, beam knobs,
        ``n_k_in``, …) are constructor kwargs on this call — they are
        not inferred.  Does **not** run :meth:`run_window_matrix` or
        :meth:`run_model_power_3d`.

        Parameters
        ----------
        ps : PowerSpectrum-like
            Source object (must expose ``field_1``, ``box_len``, …).
        **kwargs
            Override copied attributes or set window options.

        Returns
        -------
        mps : MultipolePowerSpectrum
        """
        weights_grid_1 = kwargs.pop(
            "weights_grid_1",
            getattr(ps, "weights_grid_1", getattr(ps, "weights_1", None)),
        )
        weights_grid_2 = kwargs.pop(
            "weights_grid_2",
            getattr(ps, "weights_grid_2", getattr(ps, "weights_2", None)),
        )
        spec: dict[str, Any] = {}
        skymap = getattr(ps, "skymap", None)
        if skymap is not None and "skymap" not in kwargs:
            spec["skymap"] = skymap
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
        ):
            if key in kwargs:
                continue
            try:
                val = getattr(ps, key)
            except Exception:
                continue
            if val is not None:
                spec[key] = val
        obj = cls(
            field_1=ps.field_1,
            box_len=ps.box_len,
            weights_field_1=getattr(ps, "weights_field_1", None),
            weights_grid_1=weights_grid_1,
            mean_center_1=getattr(ps, "mean_center_1", False),
            unitless_1=getattr(ps, "unitless_1", False),
            field_2=getattr(ps, "field_2", None),
            weights_field_2=getattr(ps, "weights_field_2", None),
            weights_grid_2=weights_grid_2,
            mean_center_2=getattr(ps, "mean_center_2", False),
            unitless_2=getattr(ps, "unitless_2", False),
            k1dbins=getattr(ps, "k1dbins", None),
            tracer_bias_1=getattr(ps, "tracer_bias_1", 1.0),
            tracer_bias_2=getattr(ps, "tracer_bias_2", None),
            kaiser_rsd=kwargs.pop("kaiser_rsd", getattr(ps, "kaiser_rsd", True)),
            grid_scheme=kwargs.pop("grid_scheme", getattr(ps, "grid_scheme", "nnb")),
            k1dweights=getattr(ps, "k1dweights", None),
            los=kwargs.pop("los", getattr(ps, "los", "endpoint")),
            los_observer=kwargs.pop("los_observer", getattr(ps, "los_observer", None)),
            **spec,
            **kwargs,
        )
        obj.box_ndim = ps.box_ndim
        for name in (
            "downres_factor_transverse",
            "downres_factor_radial",
            "kperpbins",
            "kparabins",
        ):
            val = getattr(ps, name, None)
            if val is not None:
                setattr(obj, name, val)
        for name in (
            "_pix_coor_in_cartesian",
            "_counts_in_box",
            "_box_origin",
            "_box_voxel_redshift",
            "_rot_mat_sky_to_box",
            "_flat_sky",
        ):
            val = getattr(ps, name, None)
            if val is not None:
                setattr(obj, name, val)
        if getattr(ps, "has_resol", False):
            obj.has_resol = True
        if getattr(ps, "k1dweights", None) is not None:
            obj.k1dweights = ps.k1dweights
        w1 = getattr(ps, "weights_1", None)
        if w1 is not None:
            obj.weights_1 = w1
        return obj


def predict_windowed_multipoles(
    ps,
    *,
    continuous: Literal["identity", "smooth"] = "smooth",
    k_in: ArrayLike | None = None,
    ells: Sequence[int] = (0, 2, 4),
    which: str = "auto_1",
    nmu: int = 64,
    los: str | None = None,
    los_observer: ArrayLike | None = None,
    k1dbins: ArrayLike | None = None,
    k1dbins_window: ArrayLike | None = None,
    n_window_bins: int = 1000,
    k1dweights: ArrayLike | None = None,
    weights_hi: ArrayLike | None = None,
    tracer: Tracer = "hi",
    n_fftlog: int = 512,
    n_k_eval: int = 128,
    W_zero: float | Literal["excess"] | None = None,
    box_volume: float | None = None,
    theory_scale: ArrayLike | None = None,
    mode_scale: ArrayLike | None = None,
    n_k_in: int = 80,
    wide_angle: bool | None = None,
    wa_d: float | None = None,
    wa_los: str | None = None,
) -> dict[str, Any]:
    """
    One-shot continuous :math:`P_\\ell(k_{\\mathrm{in}})` × discrete-shell
    window → estimator multipoles. The outer discrete-shell sum applies the
    discrete-:math:`\\mu` projector
    :math:`(2\\ell+1)\\mathcal{L}_\\ell(\\mu)` with
    :math:`\\mu=\\hat k\\cdot\\hat n_{\\mathrm{ref}}` (the local-LOS reference
    direction, box centre); identity :math:`W` matches 3D→1D.

    Builds (or skips) a
    :class:`SmoothWindowEstimator`, form the shell map, build
    :class:`~meer21cm.window.DiscreteShellWindowMatrix`, evaluate
    continuous theory via
    :meth:`~meer21cm.model.ModelPowerSpectrum.get_theory_multipoles_kmu`,
    and apply the matrix.

    Parameters
    ----------
    ps :
        :class:`~meer21cm.power.PowerSpectrum`-like object with
        ``k1dbins``, ``multipole_bin_index_map``, and
        ``get_theory_multipoles_kmu``.
    continuous : {'identity', 'smooth'}, default 'smooth'
        Discrete-shell continuous layer. ``'identity'`` needs no measured
        :math:`W_L` (no survey convolution). ``'smooth'``
        accumulates one HI selection realisation.
    k_in : array_like, optional
        Fine theory nodes. Defaults to :func:`propose_k_in` on ``k1dbins``.
    W_zero : float or {'excess'}, optional
        Forwarded to
        :meth:`SmoothWindowEstimator.build_window_matrix`. ``'excess'``
        uses :func:`_excess_zero_mode_window_power` (pair-count spike minus
        the first measured :math:`W_0` bin). Ignored for ``continuous=
        'identity'``.
    theory_scale : array_like, optional
        Multiplicative kernel on continuous :math:`P_\\ell(k_{\\mathrm{in}})`
        before the window (same length as ``k_in``). Prefer putting
        same-k grid transfers in ``mode_scale`` instead.
    mode_scale : array_like, optional
        Per-Cartesian-mode theory transfer (same shape as ``ps`` Fourier
        ``k_mode`` / shell map). Forwarded to
        :func:`~meer21cm.window.build_discrete_shell_window_matrix`
        so MAS / gridding compensation multiplies inside the discrete-shell
        sum.
    box_volume, n_fftlog, n_k_eval, wide_angle, wa_d, wa_los :
        Forwarded to the matrix builder.

    Returns
    -------
    dict
        ``k`` (``k_out``), ``P_ell``, ``k_in``, ``P_ell_unconvolved``,
        ``ells``, ``window_matrix``, ``W_zero`` (resolved float or
        ``None``), and ``continuous`` (the raw
        :meth:`~meer21cm.model.ModelPowerSpectrum.get_theory_multipoles_kmu`
        dict).
    """
    continuous_s = str(continuous).lower()
    if continuous_s not in ("identity", "smooth"):
        raise ValueError("continuous must be 'identity' or 'smooth'")

    ells_t = tuple(int(e) for e in ells)
    k1dbins_out = np.asarray(ps.k1dbins if k1dbins is None else k1dbins, dtype=float)
    if k_in is None:
        k_in_arr = propose_k_in(k1dbins_out, n=n_k_in)
    else:
        k_in_arr = np.asarray(k_in, dtype=float)

    los_use = getattr(ps, "los", "endpoint") if los is None else str(los)
    require_yamamoto_los(los_use)
    obs_saved = getattr(ps, "los_observer", None)
    los_saved = getattr(ps, "los", None)
    if los_observer is not None:
        ps.los_observer = np.asarray(los_observer, dtype=float)
    ps.los = los_use

    try:
        mat: DiscreteShellWindowMatrix
        w_zero_resolved: float | None = None
        if continuous_s == "identity":
            ells_out = ells_t
            ells_in = ells_t
            do_wa = bool(wide_angle)
            even = tuple(e for e in ells_out if e % 2 == 0)
            if do_wa:
                if not even:
                    raise ValueError(
                        "wide_angle requires at least one even output multipole"
                    )
                odds = propose_odd_wa_ells(even)
                ells_in = tuple(sorted(set(even) | set(odds)))
            shell = ps.multipole_bin_index_map(
                k1dbins=k1dbins_out,
                k1dweights=k1dweights,
                los=los_use,
            )
            mat = build_discrete_shell_window_matrix(
                shell,
                k_in=k_in_arr,
                ells=ells_out,
                ells_in=ells_in,
                continuous="identity",
                n_k_eval=n_k_eval,
                mode_scale=mode_scale,
            )
            if do_wa:
                d_wa = wa_d
                if d_wa is None:
                    raise ValueError("wa_d is required when wide_angle=True")
                los_wa = (
                    wa_los
                    if wa_los is not None
                    else (
                        los_use
                        if los_use in ("firstpoint", "endpoint")
                        else "firstpoint"
                    )
                )
                mat.resum_input_odd_wide_angle(
                    los=los_wa, d=float(d_wa), ells_even=even
                )
        else:
            k_mode = getattr(ps, "k_mode", None)
            k_fund = None
            if k_mode is not None:
                kpos = np.asarray(k_mode, dtype=float)
                kpos = kpos[np.isfinite(kpos) & (kpos > 0)]
                if kpos.size:
                    k_fund = float(np.min(kpos))
            if k1dbins_window is None:
                k1dbins_window = propose_k1dbins_window(
                    k1dbins_out, k_min=k_fund, n=n_window_bins
                )
            w_hi = weights_hi
            if w_hi is None:
                w_hi = getattr(ps, "weights_1", None)
            swe = SmoothWindowEstimator.from_power_spectrum(
                ps,
                tracer=tracer,
                ells=ells_t,
                weights_hi=w_hi,
                weights_grid_1=None,
                k1dbins_window=k1dbins_window,
                k1dbins_out=k1dbins_out,
                los=los_use,
                los_observer=getattr(ps, "los_observer", None),
                wide_angle=bool(wide_angle) if wide_angle is not None else False,
                wa_d=wa_d,
                wa_los=wa_los,
            )
            swe.accumulate([swe.run_one(0)])
            shell = ps.multipole_bin_index_map(
                k1dbins=k1dbins_out,
                k1dweights=k1dweights,
                los=los_use,
            )
            if W_zero is None:
                w_zero_resolved = None
            elif isinstance(W_zero, str) and str(W_zero).lower() == "excess":
                w_zero_resolved = _excess_zero_mode_window_power(swe)
            else:
                w_zero_resolved = float(W_zero)
            build_kw: dict[str, Any] = dict(
                shell_map=shell,
                continuous="smooth",
                n_fftlog=n_fftlog,
                n_k_eval=n_k_eval,
            )
            if w_zero_resolved is not None:
                build_kw["W_zero"] = w_zero_resolved
            if box_volume is not None:
                build_kw["box_volume"] = float(box_volume)
            elif w_zero_resolved is not None:
                build_kw["box_volume"] = float(
                    np.prod(np.asarray(ps.box_len, dtype=float))
                )
            if wide_angle is not None:
                build_kw["wide_angle"] = bool(wide_angle)
            if wa_d is not None:
                build_kw["wa_d"] = wa_d
            if wa_los is not None:
                build_kw["wa_los"] = wa_los
            if mode_scale is not None:
                build_kw["mode_scale"] = mode_scale
            mat = swe.build_window_matrix(k_in_arr, **build_kw)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            cont = ps.get_theory_multipoles_kmu(
                k_in_arr, ells=ells_t, nmu=nmu, which=which
            )
        P_in = {ell: np.asarray(cont["P_ell"][ell], dtype=float) for ell in ells_t}
        if theory_scale is not None:
            scale = np.asarray(theory_scale, dtype=float)
            if scale.shape != k_in_arr.shape:
                raise ValueError(
                    "theory_scale must have the same shape as k_in "
                    f"(got {scale.shape}, expected {k_in_arr.shape})"
                )
            P_in = {ell: P_in[ell] * scale for ell in ells_t}
        P_win = mat.apply(P_in)
    finally:
        if los_saved is not None:
            ps.los = los_saved
        if los_observer is not None:
            ps.los_observer = obs_saved

    return {
        "k": np.asarray(mat.k_out, dtype=float),
        "P_ell": P_win,
        "k_in": k_in_arr,
        "P_ell_unconvolved": cont["P_ell"],
        "ells": ells_t,
        "window_matrix": mat,
        "W_zero": w_zero_resolved,
        "continuous": cont,
    }
