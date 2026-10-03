r"""
Power spectrum estimation from already-gridded 3D fields.

The class :class:`FieldPowerSpectrum` can be used standalone given a
pre-gridded ``field_1`` and ``box_len``. Sky↔box gridding lives on
:class:`meer21cm.grid.LightconeGriddingMixin` /
:class:`meer21cm.power.PowerSpectrum`.

Multipole binning (global plane-parallel by default; other LOS conventions
reserved) is provided via :meth:`FieldPowerSpectrum.measure_multipoles`.
Survey-window matrix construction for theory multipoles lives in
:mod:`meer21cm.smooth_window` (discrete-shell matrix). Local Yamamoto
(Hand et al.) LOS conventions are reserved on ``los`` for a future update.
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
    get_vec_mode,
    get_x_vector,
    power_weights_renorm,
)
from .util import legendre_polynomial_with_factor, tagging

logger = logging.getLogger(__name__)

LOSMode = Literal["global", "endpoint", "firstpoint", "midpoint"]
MultipoleWhich = Literal["auto_1", "auto_2", "cross"]

_SUPPORTED_LOS: frozenset[str] = frozenset(
    {"global", "endpoint", "firstpoint", "midpoint"}
)
_IMPLEMENTED_LOS: frozenset[str] = frozenset({"global"})


@dataclass
class MultipoleShellMap:
    """
    Discrete Fourier-mode → 1D-|k| bin assignment for multipole estimation.

    Shares the same bin edges and weighting convention as
    :meth:`FieldPowerSpectrum.measure_multipoles` so that
    :mod:`meer21cm.smooth_window` can apply the identical shell projector
    (global plane-parallel today; local LOS later via :attr:`los`).

    Attributes
    ----------
    bin_index : ndarray
        Integer bin index per Fourier mode (same shape as :attr:`k`), or
        ``-1`` if the mode falls outside ``k1dbins``.
    k : ndarray
        Per-mode :math:`|k|` (same shape as the FFT grid ``k_mode``).
    mu : ndarray
        Per-mode :math:`\\mu` from :attr:`FieldPowerSpectrum.mu_mode`.
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
    (``los``; currently only ``'global'`` is implemented).

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
        Line-of-sight convention for :attr:`k_para`, :attr:`mu_mode`, and
        :meth:`measure_multipoles`. Only ``'global'`` (box :math:`z` = LOS)
        is implemented; other values are reserved and raise
        ``NotImplementedError`` when LOS-dependent attributes are used.
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
        """Raise if ``self.los`` is reserved but not yet implemented."""
        if self.los not in _IMPLEMENTED_LOS:
            raise NotImplementedError(
                f"{what} for los={self.los!r} is not implemented yet; "
                f"currently supported: {sorted(_IMPLEMENTED_LOS)}"
            )

    @property
    def box_len(self) -> NDArray[np.floating]:
        """The length of all sides of the box in Mpc."""
        return self._box_len

    @box_len.setter
    def box_len(self, value: ArrayLike) -> None:
        self._box_len = value
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

        For ``los='global'`` this is the last Cartesian k component (box
        :math:`z`). Local LOS conventions will redefine this once implemented.
        """
        if self.los == "global":
            return self.k_vec[-1]
        if self.los in ("endpoint", "firstpoint", "midpoint"):
            self._require_implemented_los("k_para")
        raise ValueError(f"Unhandled los={self.los!r}")

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

        clipped to ``[-1, 1]``. Local LOS estimators will replace this once
        implemented.
        """
        if self.los == "global":
            with np.errstate(divide="ignore", invalid="ignore"):
                mu = np.nan_to_num(self.k_para[None, None, :] / self.k_mode)
            return np.clip(mu, -1.0, 1.0)
        if self.los in ("endpoint", "firstpoint", "midpoint"):
            self._require_implemented_los("mu_mode")
        raise ValueError(f"Unhandled los={self.los!r}")

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

    def multipole_bin_index_map(
        self,
        k1dbins: ArrayLike | None = None,
        k1dweights: ArrayLike | None = None,
    ) -> MultipoleShellMap:
        r"""
        Map each Fourier mode to a 1D :math:`|k|` multipole bin.

        Uses the same edges, :attr:`k_mode`, :attr:`mu_mode`, and weighting
        convention as :meth:`measure_multipoles`. Intended for the opt-in
        discrete-shell window matrix (:mod:`meer21cm.discrete_window`) and for
        future local-LOS estimators that share the same radial binning.

        Parameters
        ----------
        k1dbins : array_like, optional
            1D ``k`` bin edges. Defaults to ``self.k1dbins`` when set.
        k1dweights : array_like, optional
            Per-mode weights (same role as in :meth:`measure_multipoles`).

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
        if k1dbins is None:
            k1dbins = getattr(self, "k1dbins", None)
        if k1dbins is None:
            raise ValueError("k1dbins is required for multipole_bin_index_map")
        k1dbins_np = np.asarray(k1dbins, dtype=float)
        if k1dbins_np.ndim != 1 or k1dbins_np.size < 2:
            raise ValueError("k1dbins must be a 1D array of bin edges (length >= 2)")

        k_mode = np.asarray(self.k_mode, dtype=float)
        mu = np.asarray(self.mu_mode, dtype=float)
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
        w_flat = weights.ravel()
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
            los=self.los,
        )

    def measure_multipoles(
        self,
        which: MultipoleWhich | str = "auto_1",
        k1dbins: ArrayLike | None = None,
        ells: Sequence[int] = (0, 2, 4),
        k1dweights: ArrayLike | None = None,
    ) -> MultipoleMeasurement:
        r"""
        Bin Legendre-weighted multipoles from the 3D field power.

        Uses :attr:`auto_power_3d_1`, :attr:`auto_power_3d_2`, or
        :attr:`cross_power_3d` together with :attr:`mu_mode` (LOS-dependent).
        Results are stored on this instance as :attr:`P_ell`,
        :attr:`multipole_k`, :attr:`multipole_nmodes`, and
        :attr:`multipole_ells`, and returned as a :class:`MultipoleMeasurement`.

        Parameters
        ----------
        which : {'auto_1', 'auto_2', 'cross'}, default 'auto_1'
            Which 3D power cube to multipole-bin.
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
            If ``los`` is not yet implemented.
        ValueError
            If ``which`` is unknown or ``k1dbins`` is missing.
        """
        if self.los == "global":
            if which == "auto_1":
                power3d = self.auto_power_3d_1
            elif which == "auto_2":
                power3d = self.auto_power_3d_2
                if power3d is None:
                    raise ValueError(
                        "field_2 is None; cannot measure auto_2 multipoles"
                    )
            elif which == "cross":
                power3d = self.cross_power_3d
                if power3d is None:
                    raise ValueError("field_2 is None; cannot measure cross multipoles")
            else:
                raise ValueError("which must be 'auto_1', 'auto_2', or 'cross'")

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

            mu = self.mu_mode
            k_mode = self.k_mode
            P_ell: dict[int, NDArray[np.floating]] = {}
            k_eff: NDArray[np.floating] | None = None
            nmodes: NDArray[np.floating] | None = None
            for ell in ells_t:
                factor = np.poly1d(legendre_polynomial_with_factor(ell))(mu)
                p1d, keff, nm = bin_3d_to_1d(
                    power3d * factor, k_mode, k1dbins_np, weights=k1dweights
                )
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

        if self.los in ("endpoint", "firstpoint", "midpoint"):
            self._require_implemented_los("measure_multipoles")
        raise ValueError(f"Unhandled los={self.los!r}")
