r"""
Power spectrum estimation from already-gridded 3D fields.

The class :class:`FieldPowerSpectrum` can be used standalone given a
pre-gridded ``field_1`` and ``box_len``. Sky↔box gridding lives on
:class:`meer21cm.grid.LightconeGriddingMixin` /
:class:`meer21cm.power.PowerSpectrum`.

Multipole binning is provided via :meth:`FieldPowerSpectrum.measure_multipoles`:
global plane-parallel (``los='global'``) or local Yamamoto (Hand et al.;
``los='firstpoint'`` / ``'endpoint'``). Both paths form a 3D multipole cube
(:meth:`multipole_power_3d`) then bin with :func:`~meer21cm.power_ops.bin_3d_to_1d`.
``los='midpoint'`` remains reserved. Survey-window matrix construction for
theory multipoles lives in :mod:`meer21cm.window` (discrete-shell matrix).
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .dataanalysis import Specification
from .power_ops import (
    bin_3d_to_1d,
    get_fourier_density,
    get_k_vector,
    get_power_spectrum,
    get_renormed_field,
    get_vec_mode,
    get_x_vector,
    power_weights_renorm,
)
from .spherical import (
    get_real_Ylm,
    unit_khat_from_k_vec,
    unit_los_from_observer,
)
from .util import legendre_polynomial_with_factor, real_dtype_from_array, tagging

logger = logging.getLogger(__name__)

LOSMode = Literal["global", "endpoint", "firstpoint", "midpoint"]
MultipoleWhich = Literal["auto_1", "auto_2", "cross"]

_SUPPORTED_LOS: frozenset[str] = frozenset(
    {"global", "endpoint", "firstpoint", "midpoint"}
)
_IMPLEMENTED_LOS: frozenset[str] = frozenset({"global", "firstpoint", "endpoint"})
_LOCAL_LOS: frozenset[str] = frozenset({"firstpoint", "endpoint"})


@dataclass
class MultipoleShellMap:
    r"""
    Discrete Fourier-mode → 1D-|k| bin assignment for multipole estimation.

    Uses the same bin edges and weighting convention as
    :meth:`FieldPowerSpectrum.measure_multipoles` so that
    :mod:`meer21cm.window` can apply the identical shell average.
    For ``los='global'``, :attr:`mu` is :math:`k_z/|k|`. For local LOS,
    :attr:`mu` is :math:`\hat k\cdot\hat n_{\mathrm{ref}}` at the box
    centre. The window matrix uses this :math:`\mu` as the discrete
    projector (the leading-order binning of the Yamamoto estimator; it
    matches 3D→1D for identity :math:`W`). The Yamamoto *data*
    estimator itself does not use this array.

    Attributes
    ----------
    bin_index : ndarray
        Integer bin index per Fourier mode (same shape as :attr:`k`), or
        ``-1`` if the mode falls outside ``k1dbins``.
    k : ndarray
        Per-mode :math:`|k|` (same shape as the FFT grid ``k_mode``).
    mu : ndarray
        Per-mode :math:`\mu` from :attr:`FieldPowerSpectrum.mu_mode`
        (box-z or box-centre :math:`\hat n`).
    weights : ndarray
        Per-mode binning weights (default ones; same role as ``k1dweights``).
    k1dbins : ndarray
        1D ``k`` bin edges used to build the map.
    k_eff : ndarray
        Effective :math:`k` centre per bin (weight-averaged).
    nmodes : ndarray
        Number of modes with positive weight per bin.
    los : str
        Line-of-sight convention used for :attr:`mu`.
    """

    bin_index: NDArray[np.integer]
    k: NDArray[np.floating]
    mu: NDArray[np.floating]
    weights: NDArray[np.floating]
    k1dbins: NDArray[np.floating]
    k_eff: NDArray[np.floating]
    nmodes: NDArray[np.floating]
    los: LOSMode | str = "global"


@dataclass
class MultipoleMeasurement:
    """
    Binned multipole power from :meth:`FieldPowerSpectrum.measure_multipoles`.

    Attributes
    ----------
    k : ndarray
        Effective wavenumber centres of the 1D bins.
    nmodes : ndarray
        Number of Fourier modes per bin.
    ells : tuple of int
        Multipole orders measured.
    P_ell : dict
        Mapping ``ell ->`` binned multipole power array.
    which : {'auto_1', 'auto_2', 'cross'}
        Which 3D power cube was used.
    los : str
        Line-of-sight convention used for :math:`\\mu`.
    """

    k: NDArray[np.floating]
    nmodes: NDArray[np.floating]
    ells: tuple[int, ...]
    P_ell: dict[int, NDArray[np.floating]]
    which: MultipoleWhich | str = "auto_1"
    los: LOSMode | str = "global"


class FieldPowerSpectrum(Specification):
    r"""
    Compute the power spectrum of a gridded field from LSS data.

    Also supports Legendre multipole binning via
    :meth:`measure_multipoles`, with a selectable line-of-sight convention
    (``los``; ``'global'``, ``'firstpoint'``, and ``'endpoint'``).

    ``field_1_has_random`` and ``field_2_has_random`` default to False. When
    True, that tracer's field is :math:`D - \\alpha R` and its mean-centring
    and unitless flags read back False.

    Parameters
    ----------
    field_1 : array_like
        The density field of the first tracer.
    box_len : array_like
        Comoving box lengths ``(Lx, Ly, Lz)`` in Mpc.
    weights_1 : array_like, optional
        FFT grid weights of the first tracer (uniform if ``None``).
    mean_center_1 : bool, default False
        Whether to mean-center the first tracer field.
    unitless_1 : bool, default False
        Whether to divide the first tracer field by its mean.
    field_2 : array_like, optional
        Second tracer field. If ``None``, auto-power of tracer 2 and
        cross-power are skipped.
    weights_2 : array_like, optional
        FFT grid weights of the second tracer.
    mean_center_2 : bool, default False
        Whether to mean-center the second tracer field.
    unitless_2 : bool, default False
        Whether to divide the second tracer field by its mean.
    los : {'global', 'endpoint', 'firstpoint', 'midpoint'}, default 'global'
        Line-of-sight convention for :attr:`mu_mode` and
        :meth:`measure_multipoles`. ``'global'`` is box :math:`z` (plane
        parallel). ``'firstpoint'`` / ``'endpoint'`` are local Yamamoto
        (Hand et al. / pypower). ``'midpoint'`` is reserved.
    los_observer : array_like, optional
        Observer position in the same Cartesian frame as :attr:`x_vec`
        (Mpc). Local LOS uses :math:`\\hat n(x)=(x+x_{\\mathrm{obs}})/|\\ldots|`.
        Defaults to :attr:`box_origin` when present.
    _skip_specification : bool, default False
        If ``True``, do not call :class:`~meer21cm.dataanalysis.Specification`
        ``__init__`` (used by :class:`~meer21cm.power.PowerSpectrum` after the
        model base has already initialised the survey layer).
    **params
        Forwarded to :class:`~meer21cm.dataanalysis.Specification` when
        ``_skip_specification`` is ``False``.
    """

    def __init__(
        self,
        field_1: ArrayLike,
        box_len: ArrayLike,
        weights_1: ArrayLike | None = None,
        mean_center_1: bool = False,
        unitless_1: bool = False,
        field_2: ArrayLike | None = None,
        weights_2: ArrayLike | None = None,
        mean_center_2: bool = False,
        unitless_2: bool = False,
        los: LOSMode | str = "global",
        los_observer: ArrayLike | None = None,
        _skip_specification: bool = False,
        **params: Any,
    ) -> None:
        if not _skip_specification:
            Specification.__init__(self, **params)
        elif not hasattr(self, "_precision"):
            # Standalone skip path (e.g. window randoms): need dtype for renorm.
            self._precision = True
        # los must be set before box_len / box_ndim: PowerSpectrum setters may
        # sync model k-modes and read k_para / mu_mode.
        self.los: str = self._validate_los(los)
        self._los_observer: NDArray[np.floating] | None = (
            None if los_observer is None else np.asarray(los_observer, dtype=float)
        )
        self._los_xhat_cache: (
            tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]
            | None
        ) = None
        self._cached_khat: (
            tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]
            | None
        ) = None
        self._field_1_has_random = False
        self._field_2_has_random = False
        self._field_1_D = None
        self._field_2_D = None
        self._field_1_R = None
        self._field_2_R = None
        self._field_1_alpha = 0.0
        self._field_2_alpha = 0.0
        self.field_1 = field_1
        self.field_2 = field_2
        self.weights_1: ArrayLike | None = weights_1
        self.weights_2: ArrayLike | None = weights_2
        self.box_len = np.array(box_len)
        self.box_ndim = np.array(np.asarray(field_1).shape)
        self.mean_center_1 = mean_center_1
        self.unitless_1 = unitless_1
        self.mean_center_2 = mean_center_2
        self.unitless_2 = unitless_2
        if field_2 is not None:
            error_message = "field_1 and field_2 must have same dimensions"
            assert np.allclose(
                np.asarray(field_2).shape, np.asarray(field_1).shape
            ), error_message
        self._fourier_field_1: NDArray[np.complexfloating] | None = None
        self._fourier_field_2: NDArray[np.complexfloating] | None = None
        # Populated by measure_multipoles
        self.P_ell: dict[int, NDArray[np.floating]] | None = None
        self.multipole_k: NDArray[np.floating] | None = None
        self.multipole_nmodes: NDArray[np.floating] | None = None
        self.multipole_ells: tuple[int, ...] | None = None

    @staticmethod
    def _validate_los(los: LOSMode | str) -> str:
        """Normalize and validate a LOS convention string."""
        los_s = str(los).lower()
        if los_s not in _SUPPORTED_LOS:
            raise ValueError(
                f"Unknown los={los!r}; expected one of {sorted(_SUPPORTED_LOS)}"
            )
        return los_s

    def _require_implemented_los(self, what: str) -> None:
        """Raise if ``self.los`` is unknown or reserved but not yet implemented."""
        if self.los not in _SUPPORTED_LOS:
            raise ValueError(f"Unhandled los={self.los!r}")
        if self.los not in _IMPLEMENTED_LOS:
            raise NotImplementedError(
                f"{what} for los={self.los!r} is not implemented yet; "
                f"currently supported: {sorted(_IMPLEMENTED_LOS)}"
            )

    @property
    def los_observer(self) -> NDArray[np.floating] | None:
        """
        Observer position (Mpc) for local LOS, or ``None``.

        Explicit ``los_observer`` wins; otherwise :attr:`box_origin` is used
        when present.
        """
        if self._los_observer is not None:
            return np.asarray(self._los_observer, dtype=float)
        origin = getattr(self, "box_origin", None)
        if origin is not None:
            return np.asarray(origin, dtype=float)
        return None

    @los_observer.setter
    def los_observer(self, value: ArrayLike | None) -> None:
        self._los_observer = None if value is None else np.asarray(value, dtype=float)
        self._invalidate_los_xhat()

    def _invalidate_los_xhat(self) -> None:
        """Drop cached per-voxel :math:`\\hat n(x)`."""
        self._los_xhat_cache = None

    @property
    def los_xhat(
        self,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
        r"""
        Cached per-voxel line-of-sight unit vector :math:`\hat n(x)`.

        Components are 3D real-space grids matching :attr:`field_1`. Uses
        :func:`~meer21cm.spherical.unit_los_from_observer` with
        :attr:`los_observer` (default :attr:`box_origin` when present).
        """
        if self._los_xhat_cache is None:
            self._los_xhat_cache = unit_los_from_observer(
                self.x_vec, self._require_los_observer()
            )
        return self._los_xhat_cache

    @property
    def los_xhat_stacked(self) -> NDArray[np.floating]:
        """Stacked :attr:`los_xhat` with shape ``(3,) + box_ndim``."""
        xh, yh, zh = self.los_xhat
        return np.stack([xh, yh, zh], axis=0)

    @property
    def box_len(self) -> NDArray[np.floating]:
        """The length of all sides of the box in Mpc."""
        return self._box_len

    @box_len.setter
    def box_len(self, value: ArrayLike) -> None:
        self._box_len = value
        self._invalidate_los_xhat()
        if hasattr(self, "_cached_khat"):
            self._cached_khat = None
        if "box_dep_attr" in dir(self):
            logger.debug(
                f"cleaning cache of {self.box_dep_attr} due to resetting box_len"
            )
            self.clean_cache(self.box_dep_attr)

    @property
    def box_resol(self) -> NDArray[np.floating]:
        """The grid length of each side of the enclosing box in Mpc."""
        return self.box_len / self.box_ndim

    @property
    def box_ndim(self) -> NDArray[np.integer]:
        """
        The number of grids along each side of the enclosing box.

        To ensure even sampling of +k and -k modes, the number of grids along
        every axis needs to be odd.
        """
        return self._box_ndim

    @box_ndim.setter
    def box_ndim(self, value: ArrayLike) -> None:
        self._box_ndim = value
        self._invalidate_los_xhat()
        if hasattr(self, "_cached_khat"):
            self._cached_khat = None
        if "box_dep_attr" in dir(self):
            logger.debug(
                f"cleaning cache of {self.box_dep_attr} due to resetting box_ndim"
            )
            self.clean_cache(self.box_dep_attr)

    def set_corr_type(self, corr_type: str, tracer_indx: int) -> None:
        """
        Set mean-centering / unitless / amplitude defaults for a tracer.

        Currently only two types are supported, ``"Gal"`` and ``"HI"``
        (case-insensitive). For galaxies the auto power is mean-centred,
        renormalised, and then shot-noise removed; for HI none of the above
        is performed. While ``field_1_has_random`` or ``field_2_has_random``
        is true, that tracer's mean-centring and unitless flags still read
        back false.

        Parameters
        ----------
        corr_type : str
            Tracer type (``'gal...'`` or ``'hi...'``).
        tracer_indx : int
            Either 1 or 2.
        """
        logger.debug("setting corr_type: %s for tracer %s", corr_type, tracer_indx)
        if corr_type[:3].lower() == "gal":
            mean_center = True
            unitless = True
            mean_amp: float | str = 1.0
        elif corr_type[:2].lower() == "hi":
            mean_center = False
            unitless = False
            mean_amp = "average_hi_temp"
        else:
            raise ValueError("unknown corr_type")
        if tracer_indx not in [1, 2]:
            raise ValueError("tracer_indx should be either 1 or 2")
        logger.debug("setting mean_center_%s: %s", tracer_indx, mean_center)
        logger.debug("setting unitless_%s: %s", tracer_indx, unitless)
        logger.debug("setting mean_amp_%s: %s", tracer_indx, mean_amp)
        setattr(self, "mean_center_" + str(tracer_indx), mean_center)
        setattr(self, "unitless_" + str(tracer_indx), unitless)
        setattr(self, "mean_amp_" + str(tracer_indx), mean_amp)

    @property
    def x_vec(self) -> tuple[NDArray[np.floating], ...]:
        """The 3D x-vector of the box."""
        return get_x_vector(
            self.box_ndim,
            self.box_resol,
        )

    @property
    def x_mode(self) -> NDArray[np.floating]:
        """The mode of the 3D x-vector."""
        return get_vec_mode(self.x_vec)

    @property
    def k_vec(self) -> tuple[NDArray[np.floating], ...]:
        """The 3D k-vector of the box."""
        return get_k_vector(
            self.box_ndim,
            self.box_resol,
        )

    @property
    def k_nyquist(self) -> NDArray[np.floating]:
        """The Nyquist frequency of the 3D box along each axis."""
        k_max = np.array([np.abs(self.k_vec[i]).max() for i in range(len(self.k_vec))])
        return k_max

    @property
    def k_perp(self) -> NDArray[np.floating]:
        """The **fiducial** perpendicular k-vector of the 3D box."""
        return get_vec_mode(self.k_vec[:-1])

    @property
    def k_para(self) -> NDArray[np.floating]:
        r"""
        Parallel wavenumber :math:`k_\parallel`.

        Always the last Cartesian k component (box :math:`z` on the rFFT
        grid). Local Yamamoto LOS does not redefine this (cylindrical PS
        still uses box :math:`z`).
        """
        if self.los == "midpoint":
            self._require_implemented_los("k_para")
        self._validate_los(self.los)
        return self.k_vec[-1]

    @property
    def k_mode(self) -> NDArray[np.floating]:
        """The **fiducial** (observed) mode of the 3D k-vector."""
        return get_vec_mode(self.k_vec)

    @property
    def mu_mode(self) -> NDArray[np.floating]:
        r"""
        Angle to the line of sight on the Fourier grid.

        For ``los='global'``:

        .. math::

            \mu = k_\parallel / |k|

        clipped to ``[-1, 1]``. For local LOS this is the diagnostic
        :math:`\hat k \cdot \hat n_{\mathrm{ref}}` at the box centre
        (the Yamamoto estimator does not use this array).
        """
        return self._mu_mode_for_los(self.los)

    def _drop_tracer_cache(self, tracer: int, why: str) -> None:
        """Drop the cached Fourier field of one tracer.

        Parameters
        ----------
        tracer : int
            ``1`` or ``2``.
        why : str
            Name of the attribute that changed, recorded in the debug log.

        Returns
        -------
        None
        """
        name = f"field_{tracer}_dep_attr"
        if name in dir(self):
            logger.debug(
                "cleaning cache of %s due to resetting %s",
                getattr(self, name),
                why,
            )
            self.clean_cache(getattr(self, name))

    def _fkp_field(self, tracer: int) -> NDArray[np.floating]:
        """Return :math:`F = D - \\alpha R` for one tracer.

        Parameters
        ----------
        tracer : int
            ``1`` or ``2``.

        Returns
        -------
        ndarray
            :math:`F`. ``field_*_R`` contributes 0 when it has not been set.

        Raises
        ------
        ValueError
            If ``field_*_has_random`` is True and ``field_*_D`` is missing.
        """
        data = getattr(self, f"_field_{tracer}_D")
        if data is None:
            raise ValueError(
                f"field_{tracer}_D is required when field_{tracer}_has_random is True"
            )
        randoms = getattr(self, f"_field_{tracer}_R")
        if randoms is None:
            randoms = 0.0
        alpha = float(getattr(self, f"_field_{tracer}_alpha"))
        return np.asarray(data, dtype=float) - alpha * np.asarray(randoms, dtype=float)

    @property
    def field_1(self) -> ArrayLike:
        """Density field of tracer 1.

        With ``field_1_has_random`` False this is the array assigned to
        ``field_1``. With the flag True it is :math:`D - \\alpha R`.

        Returns
        -------
        array_like
            The field passed to the Fourier transform.
        """
        if self._field_1_has_random:
            return self._fkp_field(1)
        return self._field_1

    @property
    def field_2(self) -> ArrayLike | None:
        """Density field of tracer 2.

        With ``field_2_has_random`` False this is the array assigned to
        ``field_2``, or None when tracer 2 is absent. With the flag True it
        is :math:`D - \\alpha R`.

        Returns
        -------
        array_like or None
            The field passed to the Fourier transform.
        """
        if self._field_2_has_random:
            return self._fkp_field(2)
        return self._field_2

    @field_1.setter
    def field_1(self, value: ArrayLike) -> None:
        if getattr(self, "_field_1_has_random", False):
            raise ValueError(
                "field_1 is D - alpha R while field_1_has_random is True; "
                "assign field_1_D and field_1_R"
            )
        self._field_1 = value
        self._drop_tracer_cache(1, "field_1")

    @field_2.setter
    def field_2(self, value: ArrayLike | None) -> None:
        if getattr(self, "_field_2_has_random", False):
            raise ValueError(
                "field_2 is D - alpha R while field_2_has_random is True; "
                "assign field_2_D and field_2_R"
            )
        self._field_2 = value
        self._drop_tracer_cache(2, "field_2")

    @property
    def field_1_has_random(self) -> bool:
        """Whether tracer 1 is a data-minus-randoms field.

        Returns
        -------
        bool
            False by default. True makes ``field_1`` return
            :math:`D - \\alpha R` and makes ``mean_center_1`` and
            ``unitless_1`` read back False.
        """
        return self._field_1_has_random

    @field_1_has_random.setter
    def field_1_has_random(self, value: bool) -> None:
        self._field_1_has_random = bool(value)
        self._drop_tracer_cache(1, "field_1_has_random")

    @property
    def field_2_has_random(self) -> bool:
        """Whether tracer 2 is a data-minus-randoms field.

        Returns
        -------
        bool
            False by default. True makes ``field_2`` return
            :math:`D - \\alpha R` and makes ``mean_center_2`` and
            ``unitless_2`` read back False.
        """
        return self._field_2_has_random

    @field_2_has_random.setter
    def field_2_has_random(self, value: bool) -> None:
        self._field_2_has_random = bool(value)
        self._drop_tracer_cache(2, "field_2_has_random")

    @property
    def field_1_D(self) -> ArrayLike | None:
        """Painted data counts :math:`D` of tracer 1.

        Returns
        -------
        array_like or None
            None when no separate data paint is stored. The shot noise then
            uses ``field_1``.
        """
        return self._field_1_D

    @field_1_D.setter
    def field_1_D(self, value: ArrayLike | None) -> None:
        self._field_1_D = value
        self._drop_tracer_cache(1, "field_1_D")

    @property
    def field_2_D(self) -> ArrayLike | None:
        """Painted data counts :math:`D` of tracer 2.

        Returns
        -------
        array_like or None
            None when no separate data paint is stored. The shot noise then
            uses ``field_2``.
        """
        return self._field_2_D

    @field_2_D.setter
    def field_2_D(self, value: ArrayLike | None) -> None:
        self._field_2_D = value
        self._drop_tracer_cache(2, "field_2_D")

    @property
    def field_1_R(self) -> ArrayLike | float:
        """Painted random counts :math:`R` of tracer 1.

        Returns
        -------
        array_like or float
            The stored random paint, or 0 when none has been set.
        """
        if self._field_1_R is None:
            return 0.0
        return self._field_1_R

    @field_1_R.setter
    def field_1_R(self, value: ArrayLike | float | None) -> None:
        self._field_1_R = value
        self._drop_tracer_cache(1, "field_1_R")

    @property
    def field_2_R(self) -> ArrayLike | float:
        """Painted random counts :math:`R` of tracer 2.

        Returns
        -------
        array_like or float
            The stored random paint, or 0 when none has been set.
        """
        if self._field_2_R is None:
            return 0.0
        return self._field_2_R

    @field_2_R.setter
    def field_2_R(self, value: ArrayLike | float | None) -> None:
        self._field_2_R = value
        self._drop_tracer_cache(2, "field_2_R")

    @property
    def field_1_alpha(self) -> float:
        """Ratio :math:`\\alpha = \\sum D / \\sum R` for tracer 1.

        Returns
        -------
        float
            0 when no random catalogue is attached.
        """
        return float(self._field_1_alpha)

    @field_1_alpha.setter
    def field_1_alpha(self, value: float) -> None:
        self._field_1_alpha = float(value)
        self._drop_tracer_cache(1, "field_1_alpha")

    @property
    def field_2_alpha(self) -> float:
        """Ratio :math:`\\alpha = \\sum D / \\sum R` for tracer 2.

        Returns
        -------
        float
            0 when no random catalogue is attached.
        """
        return float(self._field_2_alpha)

    @field_2_alpha.setter
    def field_2_alpha(self, value: float) -> None:
        self._field_2_alpha = float(value)
        self._drop_tracer_cache(2, "field_2_alpha")

    @property
    def mean_center_1(self) -> bool:
        """Whether the Fourier transform subtracts the mean of tracer 1.

        Returns
        -------
        bool
            The stored flag, or False while ``field_1_has_random`` is True.
            The stored flag is kept and applies again if the random-catalogue
            flag is set back to False.
        """
        if self._field_1_has_random:
            return False
        return bool(self._mean_center_1)

    @property
    def mean_center_2(self) -> bool:
        """Whether the Fourier transform subtracts the mean of tracer 2.

        Returns
        -------
        bool
            The stored flag, or False while ``field_2_has_random`` is True.
            The stored flag is kept and applies again if the random-catalogue
            flag is set back to False.
        """
        if self._field_2_has_random:
            return False
        return bool(self._mean_center_2)

    @mean_center_1.setter
    def mean_center_1(self, value: bool) -> None:
        # if weight is updated, clear fourier field
        self._mean_center_1 = value
        if "field_1_dep_attr" in dir(self):
            logger.debug(
                f"cleaning cache of {self.field_1_dep_attr} due to resetting mean_center_1"
            )
            self.clean_cache(self.field_1_dep_attr)

    @mean_center_2.setter
    def mean_center_2(self, value: bool) -> None:
        # if weight is updated, clear fourier field
        self._mean_center_2 = value
        if "field_2_dep_attr" in dir(self):
            logger.debug(
                f"cleaning cache of {self.field_2_dep_attr} due to resetting mean_center_2"
            )
            self.clean_cache(self.field_2_dep_attr)

    @property
    def unitless_1(self) -> bool:
        """Whether tracer 1 is divided by its mean before the Fourier transform.

        Returns
        -------
        bool
            The stored flag, or False while ``field_1_has_random`` is True.
            The stored flag is kept and applies again if the random-catalogue
            flag is set back to False.
        """
        if self._field_1_has_random:
            return False
        return bool(self._unitless_1)

    @property
    def unitless_2(self) -> bool:
        """Whether tracer 2 is divided by its mean before the Fourier transform.

        Returns
        -------
        bool
            The stored flag, or False while ``field_2_has_random`` is True.
            The stored flag is kept and applies again if the random-catalogue
            flag is set back to False.
        """
        if self._field_2_has_random:
            return False
        return bool(self._unitless_2)

    @unitless_1.setter
    def unitless_1(self, value: bool) -> None:
        # if weight is updated, clear fourier field
        self._unitless_1 = value
        if "field_1_dep_attr" in dir(self):
            logger.debug(
                f"cleaning cache of {self.field_1_dep_attr} due to resetting unitless_1"
            )
            self.clean_cache(self.field_1_dep_attr)

    @unitless_2.setter
    def unitless_2(self, value: bool) -> None:
        # if weight is updated, clear fourier field
        self._unitless_2 = value
        if "field_2_dep_attr" in dir(self):
            logger.debug(
                f"cleaning cache of {self.field_2_dep_attr} due to resetting unitless_2"
            )
            self.clean_cache(self.field_2_dep_attr)

    @property
    @tagging("box", "field_1")
    def fourier_field_1(self) -> NDArray[np.complexfloating]:
        """The Fourier transform of the density field of the first tracer."""
        if self._fourier_field_1 is None:
            self.get_fourier_field_1()
        assert self._fourier_field_1 is not None
        return self._fourier_field_1

    def get_fourier_field_1(self) -> None:
        """Calculate the Fourier transform of the density field of the first tracer."""
        result = get_fourier_density(
            self.field_1,
            weights=self.weights_1,
            mean_center=self.mean_center_1,
            unitless=self.unitless_1,
            field_mean=getattr(self, "field_mean_1", None),
        )
        logger.info(
            f"{inspect.currentframe().f_code.co_name}: "
            f"setting self._fourier_field_1"
        )
        self._fourier_field_1 = result

    @property
    @tagging("box", "field_2")
    def fourier_field_2(self) -> NDArray[np.complexfloating] | None:
        """The Fourier transform of the density field of the second tracer."""
        if self._fourier_field_2 is None:
            self.get_fourier_field_2()
        return self._fourier_field_2

    def get_fourier_field_2(self) -> NDArray[np.complexfloating] | None:
        """Calculate the Fourier transform of the density field of the second tracer."""
        if self.field_2 is None:
            logger.info("field_2 is None, returning None")
            return None
        result = get_fourier_density(
            self.field_2,
            weights=self.weights_2,
            mean_center=self.mean_center_2,
            unitless=self.unitless_2,
            field_mean=getattr(self, "field_mean_2", None),
        )
        logger.info(
            f"{inspect.currentframe().f_code.co_name}: "
            f"setting self._fourier_field_2"
        )
        self._fourier_field_2 = result

    # the calculation of this is not heavy, simply on the fly
    @property
    def auto_power_3d_1(self) -> NDArray[np.floating]:
        """The 3D power spectrum of the first tracer."""
        power_spectrum = get_power_spectrum(
            self.fourier_field_1,
            self.box_len,
            weights=self.weights_1,
            renorm=False,
        )
        return power_spectrum * self.renorm_ps_1

    @property
    def auto_power_3d_2(self) -> NDArray[np.floating] | None:
        """The 3D power spectrum of the second tracer."""
        if self.field_2 is None:
            return None
        power_spectrum = get_power_spectrum(
            self.fourier_field_2,
            self.box_len,
            weights=self.weights_2,
            renorm=False,
        )
        return power_spectrum * self.renorm_ps_2

    @property
    def cross_power_3d(self) -> NDArray[np.floating] | None:
        """The 3D cross power spectrum between the two tracers."""
        if self.field_2 is None:
            return None
        weights_2 = self.weights_2
        # if none, the default for get_power_spectrum is
        # to use weights_1, here we want separate weights_2
        if weights_2 is None:
            weights_2 = np.ones(np.asarray(self.field_2).shape)
        power_spectrum = get_power_spectrum(
            self.fourier_field_1,
            self.box_len,
            weights=self.weights_1,
            field_2=self.fourier_field_2,
            weights_2=weights_2,
            renorm=False,
        )
        return power_spectrum * self.renorm_ps_cross

    @property
    def renorm_ps_1(self) -> float | NDArray[np.floating]:
        """The renormalization factor of the power spectrum of the first tracer."""
        grid_w = self.get_weights_none_to_one("weights_1")
        field_w: float | NDArray[np.floating] = 1.0
        mean_renorm = 1.0
        if hasattr(self, "weights_field_1"):
            field_w = self.get_weights_none_to_one("weights_field_1")
            if self.unitless_1:
                mean_renorm = (field_w * grid_w).sum() / (grid_w).sum()
        return (
            power_weights_renorm(grid_w * field_w, grid_w * field_w) * mean_renorm**2
        )

    @property
    def renorm_ps_2(self) -> float | NDArray[np.floating]:
        """The renormalization factor of the power spectrum of the second tracer."""
        grid_w = self.get_weights_none_to_one("weights_2")
        field_w: float | NDArray[np.floating] = 1.0
        mean_renorm = 1.0
        if hasattr(self, "weights_field_2"):
            field_w = self.get_weights_none_to_one("weights_field_2")
            if self.unitless_2:
                mean_renorm = (field_w * grid_w).sum() / (grid_w).sum()
        return (
            power_weights_renorm(grid_w * field_w, grid_w * field_w) * mean_renorm**2
        )

    @property
    def renorm_ps_cross(self) -> float | NDArray[np.floating]:
        """The renormalization factor of the cross power spectrum."""
        grid_w_1 = self.get_weights_none_to_one("weights_1")
        field_w_1: float | NDArray[np.floating] = 1.0
        mean_renorm_1 = 1.0
        if hasattr(self, "weights_field_1"):
            field_w_1 = self.get_weights_none_to_one("weights_field_1")
            if self.unitless_1:
                mean_renorm_1 = (field_w_1 * grid_w_1).sum() / (grid_w_1).sum()
        grid_w_2 = self.get_weights_none_to_one("weights_2")
        field_w_2: float | NDArray[np.floating] = 1.0
        mean_renorm_2 = 1.0
        if hasattr(self, "weights_field_2"):
            field_w_2 = self.get_weights_none_to_one("weights_field_2")
            if self.unitless_2:
                mean_renorm_2 = (field_w_2 * grid_w_2).sum() / (grid_w_2).sum()
        return (
            power_weights_renorm(grid_w_1 * field_w_1, grid_w_2 * field_w_2)
            * mean_renorm_1
            * mean_renorm_2
        )

    def _require_los_observer(self) -> NDArray[np.floating]:
        """Observer position required for local LOS."""
        obs = self.los_observer
        if obs is None:
            raise ValueError(
                "los_observer (or box_origin) is required for local LOS "
                f"(los={self.los!r})"
            )
        return np.asarray(obs, dtype=float)

    def _los_nref(self) -> NDArray[np.floating]:
        """Unit LOS at the box centre (cell-centre frame + observer)."""
        obs = self._require_los_observer()
        center = 0.5 * np.asarray(self.box_len, dtype=float)
        vec = center + obs
        nrm = float(np.linalg.norm(vec))
        if nrm == 0.0:
            return np.array([0.0, 0.0, 1.0], dtype=float)
        return vec / nrm

    def _mu_mode_for_los(self, los: LOSMode | str) -> NDArray[np.floating]:
        """Fourier-grid :math:`\\mu` for a LOS convention (see :attr:`mu_mode`)."""
        los_s = self._validate_los(los)
        if los_s == "global":
            with np.errstate(divide="ignore", invalid="ignore"):
                mu = np.nan_to_num(self.k_para[None, None, :] / self.k_mode)
            return np.clip(mu, -1.0, 1.0)
        if los_s in _LOCAL_LOS:
            nref = self._los_nref()
            khx, khy, khz = self._khat()
            mu = khx * nref[0] + khy * nref[1] + khz * nref[2]
            return np.clip(mu, -1.0, 1.0)
        if los_s == "midpoint":
            self._require_implemented_los("mu_mode")
        raise ValueError(f"Unhandled los={los_s!r}")

    def _khat(
        self,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
        """Cached :math:`\\hat k` on the rFFT lattice."""
        if self._cached_khat is None:
            self._cached_khat = unit_khat_from_k_vec(self.k_vec)
        return self._cached_khat

    def _isotropic_power3d(self, which: MultipoleWhich | str) -> NDArray[np.floating]:
        """Unweighted 3D auto/cross power cube (``|F0|^2`` convention)."""
        if which == "auto_1":
            return self.auto_power_3d_1
        if which == "auto_2":
            power3d = self.auto_power_3d_2
            if power3d is None:
                raise ValueError("field_2 is None; cannot measure auto_2 multipoles")
            return power3d
        if which == "cross":
            power3d = self.cross_power_3d
            if power3d is None:
                raise ValueError("field_2 is None; cannot measure cross multipoles")
            return power3d
        raise ValueError("which must be 'auto_1', 'auto_2', or 'cross'")

    def _weighted_real_field(self, tracer: int) -> NDArray[np.floating]:
        """Real-space ``δ × w`` matching :func:`~meer21cm.power_ops.get_fourier_density`."""
        if tracer == 1:
            field = np.asarray(self.field_1)
            weights = self.weights_1
            mean_center = self.mean_center_1
            unitless = self.unitless_1
        elif tracer == 2:
            if self.field_2 is None:
                raise ValueError("field_2 is None")
            field = np.asarray(self.field_2)
            weights = self.weights_2
            mean_center = self.mean_center_2
            unitless = self.unitless_2
        else:
            raise ValueError("tracer must be 1 or 2")
        field_r = get_renormed_field(
            field, weights=weights, mean_center=mean_center, unitless=unitless
        )
        real_dtype = real_dtype_from_array(field_r)
        if weights is None:
            w = np.ones_like(field_r, dtype=real_dtype)
        else:
            w = np.asarray(weights, dtype=real_dtype)
        return np.asarray(field_r, dtype=real_dtype) * w

    def _fourier_field_tracer(self, tracer: int) -> NDArray[np.complexfloating]:
        if tracer == 1:
            return self.fourier_field_1
        if tracer == 2:
            f2 = self.fourier_field_2
            if f2 is None:
                raise ValueError("field_2 is None")
            return f2
        raise ValueError("tracer must be 1 or 2")

    def _renorm_for_which(
        self, which: MultipoleWhich | str
    ) -> float | NDArray[np.floating]:
        if which == "auto_1":
            return self.renorm_ps_1
        if which == "auto_2":
            return self.renorm_ps_2
        if which == "cross":
            return self.renorm_ps_cross
        raise ValueError("which must be 'auto_1', 'auto_2', or 'cross'")

    def _yamamoto_tracers(self, which: MultipoleWhich | str) -> tuple[int, int]:
        """``(ylm_tracer, f0_tracer)`` for firstpoint; endpoint swaps."""
        if which == "auto_1":
            i_ylm, i_f0 = 1, 1
        elif which == "auto_2":
            if self.field_2 is None:
                raise ValueError("field_2 is None; cannot measure auto_2 multipoles")
            i_ylm, i_f0 = 2, 2
        elif which == "cross":
            if self.field_2 is None:
                raise ValueError("field_2 is None; cannot measure cross multipoles")
            i_ylm, i_f0 = 1, 2
        else:
            raise ValueError("which must be 'auto_1', 'auto_2', or 'cross'")
        if self.los == "endpoint":
            i_ylm, i_f0 = i_f0, i_ylm
        return i_ylm, i_f0

    def _yamamoto_multipole_power_3d(
        self, ell: int, which: MultipoleWhich | str
    ) -> NDArray[np.floating]:
        r"""
        3D Yamamoto multipole cube :math:`P_\ell(\mathbf{k})`.

        .. math::

            A_\ell(\mathbf{k}) = \sum_m Y_{\ell m}(\hat k)\,
            \mathrm{FFT}[\delta_w(\mathbf{x})\,Y_{\ell m}(\hat n(x))]

        then :math:`P_\ell^{3D} = 4\pi\,(A_\ell F_0^*)\,V\,R` with the real
        part for even :math:`\ell` and the imaginary part for odd
        :math:`\ell` (hermitian-antisymmetric; pypower / Hand et al.).
        :math:`\ell=0` is :math:`|F_0|^2 V R` (or the cross analogue).
        Endpoint applies Ylm to the second tracer (pypower swap).
        """
        ell = int(ell)
        i_ylm, i_f0 = self._yamamoto_tracers(which)
        f0 = self._fourier_field_tracer(i_f0)
        box_volume = float(np.prod(np.asarray(self.box_len, dtype=float)))
        renorm = self._renorm_for_which(which)
        if ell == 0:
            f_ylm = self._fourier_field_tracer(i_ylm)
            return np.real(f_ylm * np.conj(f0)) * box_volume * renorm

        xhat = self.los_xhat
        khat = self._khat()
        delta_w = self._weighted_real_field(i_ylm)
        a_ell = np.zeros(f0.shape, dtype=np.result_type(f0.dtype, np.complex128))
        for m in range(-ell, ell + 1):
            ylm = get_real_Ylm(ell, m)
            f_lm = np.fft.rfftn(delta_w * ylm(*xhat), norm="forward")
            a_ell = a_ell + ylm(*khat) * f_lm
        prod = a_ell * np.conj(f0)
        # Odd multipoles are hermitian-antisymmetric → imaginary 3D cube.
        part = np.imag(prod) if (ell % 2) else np.real(prod)
        # pypower endpoint: swap tracers then conj the (complex) power, which
        # flips the odd (imaginary) multipoles. Auto swap is a no-op, so the
        # sign flip is what makes firstpoint odd = -endpoint odd.
        if self.los == "endpoint" and (ell % 2):
            part = -part
        return (4.0 * np.pi) * part * box_volume * renorm

    def multipole_power_3d(
        self,
        ell: int,
        which: MultipoleWhich | str = "auto_1",
    ) -> NDArray[np.floating]:
        r"""
        3D multipole power cube before 1D :math:`|k|` binning.

        * ``los='global'``: isotropic 3D power times
          :math:`(2\ell+1)\mathcal{L}_\ell(\mu)` with :math:`\mu=k_z/|k|`.
        * ``los='firstpoint'`` / ``'endpoint'``: Yamamoto
          :math:`F_\ell F_0` cube (Hand et al. / pypower).

        Parameters
        ----------
        ell : int
            Multipole order.
        which : {'auto_1', 'auto_2', 'cross'}, default 'auto_1'
            Which tracer combination.

        Returns
        -------
        power3d : ndarray
            Same shape as :attr:`k_mode` (rFFT lattice).
        """
        self._require_implemented_los("multipole_power_3d")
        ell = int(ell)
        if self.los == "global":
            power3d = self._isotropic_power3d(which)
            factor = np.poly1d(legendre_polynomial_with_factor(ell))(self.mu_mode)
            return power3d * factor
        if self.los in _LOCAL_LOS:
            return self._yamamoto_multipole_power_3d(ell, which)
        raise ValueError(f"Unhandled los={self.los!r}")

    def multipole_bin_index_map(
        self,
        k1dbins: ArrayLike | None = None,
        k1dweights: ArrayLike | None = None,
        los: LOSMode | str | None = None,
    ) -> MultipoleShellMap:
        r"""
        Map each Fourier mode to a 1D :math:`|k|` multipole bin.

        Uses the same edges, :attr:`k_mode`, and weighting convention as
        :meth:`measure_multipoles`. Intended for the opt-in discrete-shell
        window matrix (:mod:`meer21cm.window`), which averages the
        continuous :math:`W_{\ell\ell'}` kernel over those shells.
        Yamamoto :meth:`measure_multipoles` does not use this map.

        Parameters
        ----------
        k1dbins : array_like, optional
            1D ``k`` bin edges. Defaults to ``self.k1dbins`` when set.
        k1dweights : array_like, optional
            Per-mode weights (same role as in :meth:`measure_multipoles`).
        los : {'global', 'firstpoint', 'endpoint'}, optional
            LOS stored on the shell map (diagnostic :attr:`mu` only).
            Defaults to :attr:`los` on this object.

        Returns
        -------
        shell_map : MultipoleShellMap

        Raises
        ------
        NotImplementedError
            If ``los`` is not yet implemented.
        ValueError
            If ``k1dbins`` is missing.
        """
        self._require_implemented_los("multipole_bin_index_map")
        los_eff = self._validate_los(los) if los is not None else self.los
        if k1dbins is None:
            k1dbins = getattr(self, "k1dbins", None)
        if k1dbins is None:
            raise ValueError("k1dbins is required for multipole_bin_index_map")
        k1dbins_np = np.asarray(k1dbins, dtype=float)
        if k1dbins_np.ndim != 1 or k1dbins_np.size < 2:
            raise ValueError("k1dbins must be a 1D array of bin edges (length >= 2)")

        k_mode = np.asarray(self.k_mode, dtype=float)
        mu = np.asarray(self._mu_mode_for_los(los_eff), dtype=float)
        if k1dweights is None:
            k1dweights = getattr(self, "k1dweights", None)
        if k1dweights is None:
            weights = np.ones_like(k_mode, dtype=float)
        else:
            weights = np.asarray(k1dweights, dtype=float)
            if weights.shape != k_mode.shape:
                raise ValueError(
                    "k1dweights shape %s does not match k_mode shape %s"
                    % (weights.shape, k_mode.shape)
                )

        n_bins = k1dbins_np.size - 1
        bin_index = np.full(k_mode.shape, -1, dtype=np.intp)
        k_flat = k_mode.ravel()
        idx_flat = np.full(k_flat.shape, -1, dtype=np.intp)
        for i in range(n_bins):
            mask = (k_flat >= k1dbins_np[i]) & (k_flat < k1dbins_np[i + 1])
            idx_flat[mask] = i
        bin_index = idx_flat.reshape(k_mode.shape)

        k_eff = np.full(n_bins, np.nan, dtype=float)
        nmodes = np.zeros(n_bins, dtype=float)
        for i in range(n_bins):
            in_bin = bin_index == i
            w_bin = weights * in_bin
            w_sum = np.sum(w_bin)
            nmodes[i] = np.sum(in_bin & (weights > 0))
            if w_sum > 0:
                k_eff[i] = np.sum(k_mode * w_bin) / w_sum

        return MultipoleShellMap(
            bin_index=bin_index,
            k=k_mode,
            mu=mu,
            weights=weights,
            k1dbins=k1dbins_np,
            k_eff=k_eff,
            nmodes=nmodes,
            los=los_eff,
        )

    def measure_multipoles(
        self,
        which: MultipoleWhich | str = "auto_1",
        k1dbins: ArrayLike | None = None,
        ells: Sequence[int] = (0, 2, 4),
        k1dweights: ArrayLike | None = None,
    ) -> MultipoleMeasurement:
        r"""
        Bin 3D multipole power onto 1D :math:`|k|` shells.

        Builds :meth:`multipole_power_3d` for each ``ell`` (global
        :math:`(2\ell+1)\mathcal{L}_\ell(\mu)` or Yamamoto :math:`F_\ell F_0`)
        then applies :func:`~meer21cm.power_ops.bin_3d_to_1d`. Results are
        stored as :attr:`P_ell`, :attr:`multipole_k`, :attr:`multipole_nmodes`,
        and :attr:`multipole_ells`, and returned as a
        :class:`MultipoleMeasurement`.

        Parameters
        ----------
        which : {'auto_1', 'auto_2', 'cross'}, default 'auto_1'
            Which tracer combination.
        k1dbins : array_like
            1D ``k`` bin edges. Required unless ``self.k1dbins`` is already set
            (e.g. on a :class:`~meer21cm.power.PowerSpectrum` instance).
        ells : sequence of int, default (0, 2, 4)
            Multipoles to measure.
        k1dweights : array_like, optional
            Optional per-mode weights for
            :func:`~meer21cm.power_ops.bin_3d_to_1d`. Defaults to ones, or to
            ``self.k1dweights`` when present.

        Returns
        -------
        measurement : MultipoleMeasurement

        Raises
        ------
        NotImplementedError
            If ``los`` is not yet implemented (``midpoint``).
        ValueError
            If ``which`` is unknown, ``k1dbins`` is missing, or local LOS
            has no ``los_observer`` / ``box_origin``.
        """
        self._require_implemented_los("measure_multipoles")
        if which not in ("auto_1", "auto_2", "cross"):
            raise ValueError("which must be 'auto_1', 'auto_2', or 'cross'")
        if which in ("auto_2", "cross") and self.field_2 is None:
            raise ValueError(f"field_2 is None; cannot measure {which} multipoles")

        if k1dbins is None:
            k1dbins = getattr(self, "k1dbins", None)
        if k1dbins is None:
            raise ValueError("k1dbins is required for measure_multipoles")
        k1dbins_np = np.asarray(k1dbins, dtype=float)
        ells_t = tuple(int(e) for e in ells)

        if k1dweights is None:
            k1dweights = getattr(self, "k1dweights", None)
        if k1dweights is None:
            k1dweights = np.ones_like(self.k_mode)

        k_mode = self.k_mode
        P_ell: dict[int, NDArray[np.floating]] = {}
        k_eff: NDArray[np.floating] | None = None
        nmodes: NDArray[np.floating] | None = None
        for ell in ells_t:
            p3d = self.multipole_power_3d(ell, which=which)
            p1d, keff, nm = bin_3d_to_1d(p3d, k_mode, k1dbins_np, weights=k1dweights)
            P_ell[ell] = p1d
            k_eff, nmodes = keff, nm

        assert k_eff is not None and nmodes is not None
        self.P_ell = P_ell
        self.multipole_k = k_eff
        self.multipole_nmodes = nmodes
        self.multipole_ells = ells_t
        return MultipoleMeasurement(
            k=k_eff,
            nmodes=nmodes,
            ells=ells_t,
            P_ell=P_ell,
            which=which,
            los=self.los,
        )
