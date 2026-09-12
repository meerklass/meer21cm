"""
Model power spectrum from theory, with optional observational window effects.

The class :py:class:`ModelPowerSpectrum` can be used standalone with supplied
``kmode`` / ``mumode``. Observation-dependent windows that need a Cartesian
field grid (map sampling, MAS compensation) are stubbed to ``1.0`` here and
overridden on :class:`meer21cm.power.PowerSpectrum`.
"""

import inspect
import logging
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import eval_legendre

from .cosmology import CosmologyCalculator
from .power_ops import get_modelpk_conv, gaussian_beam_attenuation
from .util import tagging

logger = logging.getLogger(__name__)


class ModelPowerSpectrum(CosmologyCalculator):
    r"""
    The class for computing the model power spectrum of an LSS tracer field.

    Parameters
    ----------
    kmode: np.ndarray, default None
        The **true** mode of k in Mpc-1.
    mumode: np.ndarray, default None
        The **true** mu values of each k-mode so that :math:`k_\parallel = k \times \mu`.
    tracer_bias_1: float, default 1.0
        The linear bias of the first tracer.
    sigma_v_1: float, default 0.0
        The velocity dispersion of the first tracer in km/s.
    tracer_bias_2: float, default None
        The linear bias of the second tracer.
    sigma_v_2: float, default 0.0
        The velocity dispersion of the second tracer in km/s.
    include_beam: list, default [True, False]
        Whether to include the beam attenuation in the model calculation.
        Must be a list of two booleans, the first for the first tracer and the second for the second tracer.
    fog_profile: str, default "lorentz"
        The shape of the finger-of-god profile to be used in the model calculation.
        Either "lorentz" or "gaussian".
    cross_coeff: float, default 1.0
        The cross-correlation coefficient between the two tracers.
    weights_field_1: np.ndarray, default None
        The field-level weights of the first tracer in the density field.
    weights_field_2: np.ndarray, default None
        The field-level weights of the second tracer in the density field.
    weights_grid_1: np.ndarray, default None
        The grid-level weights of the first tracer in the density field.
    weights_grid_2: np.ndarray, default None
        The grid-level weights of the second tracer in the density field.
    mean_amp_1: float, default 1.0
        The mean amplitude of the first tracer.
        Can be used to rescale the power spectrum, for example by the average brightness temperature.
    mean_amp_2: float, default 1.0
        The mean amplitude of the second tracer.
        Can be used to rescale the power spectrum, for example by the average brightness temperature.
    sampling_resol: list, default None
        The sampling resolution of the field in Mpc.
        If ``sampling_resol`` is "auto", the sampling resolution will be set to the pixel size of the map.
    include_sky_sampling: list, default [True, False]
        Whether to include the sky sampling in the model calculation.
        If just a boolean is provided, it will be used for both tracers.
    compensate: list, default [True, True]
        Whether the gridded fields are compensated according to the mass assignment scheme.
        Note that the compensation is applied to the model power spectrum, and **not** to the gridded data fields.
    kaiser_rsd: bool, default True
        Whether to include the RSD effect in the model calculation and mock simulation.
    sigma_z_1: float, default 0.0
        The redshift error of the first tracer.
    sigma_z_2: float, default 0.0
        The redshift error of the second tracer.
    **params: dict
        Additional parameters to be passed to the base class :class:`meer21cm.cosmology.CosmologyCalculator`.
    """

    def __init__(
        self,
        kmode=None,
        mumode=None,
        tracer_bias_1=1.0,
        sigma_v_1=0.0,
        tracer_bias_2=None,
        sigma_v_2=0.0,
        include_beam=[True, False],
        fog_profile="lorentz",
        cross_coeff=1.0,
        weights_field_1=None,
        weights_field_2=None,
        weights_grid_1=None,
        weights_grid_2=None,
        mean_amp_1=1.0,
        mean_amp_2=1.0,
        sampling_resol=None,
        include_sky_sampling=[True, False],
        compensate=[True, True],
        kaiser_rsd=True,
        sigma_z_1=0.0,
        sigma_z_2=0.0,
        **params,
    ):
        super().__init__(**params)
        # for compatibility with FieldPowerSpectrum
        if not hasattr(self, "field_1_dep_attr"):
            self.field_1_dep_attr = []
        if not hasattr(self, "field_2_dep_attr"):
            self.field_2_dep_attr = []
        self.tracer_bias_1 = tracer_bias_1
        self.sigma_v_1 = sigma_v_1
        self.tracer_bias_2 = tracer_bias_2
        self.sigma_v_2 = sigma_v_2
        self.kmode = kmode
        self.mumode = mumode
        if kmode is None:
            self.kmode = np.geomspace(self.kmin, self.kmax, 600).reshape((10, 10, 6))
        if mumode is None:
            self.mumode = np.zeros_like(self.kmode)
        self._include_beam = [None, None]  # for initialization
        self.include_beam = include_beam
        self.cross_coeff = cross_coeff
        self._auto_power_matter_model_r = None
        self._auto_power_matter_model = None
        self._auto_power_tracer_1_model_noobs = None
        self._auto_power_tracer_2_model_noobs = None
        self._cross_power_tracer_model_noobs = None
        self._auto_power_tracer_1_model = None
        self._auto_power_tracer_2_model = None
        self._cross_power_tracer_model = None
        self.weights_field_1 = weights_field_1
        self.weights_field_2 = weights_field_2
        self.weights_grid_1 = weights_grid_1
        self.weights_grid_2 = weights_grid_2
        self.mean_amp_1 = mean_amp_1
        self.mean_amp_2 = mean_amp_2
        self.include_sky_sampling = include_sky_sampling
        self.sampling_resol = sampling_resol
        self.has_resol = True
        if self.sampling_resol is None:
            self.has_resol = False
        # avoid ambiguity problem of == auto
        if isinstance(self.sampling_resol, str):
            if self.sampling_resol == "auto":
                self.sampling_resol = [
                    self.pix_resol_in_mpc,
                    self.pix_resol_in_mpc,
                    self.los_resol_in_mpc,
                ]
        self.fog_profile = fog_profile
        self.kaiser_rsd = kaiser_rsd
        self._compensate = [None, None]  # for initialization
        self.compensate = compensate
        self.sigma_z_1 = sigma_z_1
        self.sigma_z_2 = sigma_z_2

    @property
    def weights_field_1(self):
        """
        The weights of the first tracer in the density field.
        """
        return self._weights_field_1

    @weights_field_1.setter
    def weights_field_1(self, value):
        self._weights_field_1 = value
        if "tracer_1_dep_attr" in dir(self):
            logger.debug(
                f"cleaning cache of {self.tracer_1_dep_attr} due to resetting weights_field_1"
            )
            self.clean_cache(self.tracer_1_dep_attr)

    @property
    def weights_grid_1(self):
        """
        The weights of the first tracer in the rectangular grid.
        """
        return self._weights_grid_1

    @weights_grid_1.setter
    def weights_grid_1(self, value):
        self._weights_grid_1 = value
        if "tracer_1_dep_attr" in dir(self):
            logger.debug(
                f"cleaning cache of {self.tracer_1_dep_attr} and {self.field_1_dep_attr} due to resetting weights_grid_1"
            )
            self.clean_cache(self.tracer_1_dep_attr)
            self.clean_cache(self.field_1_dep_attr)

    @property
    def weights_field_2(self):
        """
        The weights of the second tracer in the density field.
        """
        return self._weights_field_2

    @weights_field_2.setter
    def weights_field_2(self, value):
        self._weights_field_2 = value
        if "tracer_2_dep_attr" in dir(self):
            logger.debug(
                f"cleaning cache of {self.tracer_2_dep_attr} due to resetting weights_field_2"
            )
            self.clean_cache(self.tracer_2_dep_attr)

    @property
    def weights_grid_2(self):
        """
        The weights of the second tracer in the rectangular grid.
        """
        return self._weights_grid_2

    @weights_grid_2.setter
    def weights_grid_2(self, value):
        self._weights_grid_2 = value
        if "tracer_2_dep_attr" in dir(self):
            logger.debug(
                f"cleaning cache of {self.tracer_2_dep_attr} and {self.field_2_dep_attr} due to resetting weights_grid_2"
            )
            self.clean_cache(self.tracer_2_dep_attr)
            self.clean_cache(self.field_2_dep_attr)

    # for compatibility with FieldPowerSpectrum
    weights_1 = weights_grid_1
    weights_2 = weights_grid_2

    @property
    def kaiser_rsd(self):
        """
        Whether RSD is included in the simulation and model calculation.
        If True, uses the linear Kaiser effect and the FoG profile to compute the model power spectrum.
        """
        return self._kaiser_rsd

    @kaiser_rsd.setter
    def kaiser_rsd(self, value):
        self._kaiser_rsd = value
        logger.debug(
            f"cleaning cache of {self.rsd_dep_attr} due to resetting kaiser_rsd"
        )
        self.clean_cache(self.rsd_dep_attr)

    @property
    def fog_profile(self):
        """
        The shape of the finger-of-god profile to be used in the model calculation.
        Either "lorentz" or "gaussian".
        """
        return self._fog_profile

    @fog_profile.setter
    def fog_profile(self, value):
        self._fog_profile = value
        if "tracer_1_dep_attr" in dir(self):
            logger.debug(
                f"cleaning cache of {self.tracer_1_dep_attr} due to resetting fog_profile"
            )
            self.clean_cache(self.tracer_1_dep_attr)
        if "tracer_2_dep_attr" in dir(self):
            logger.debug(
                f"cleaning cache of {self.tracer_2_dep_attr} due to resetting fog_profile"
            )
            self.clean_cache(self.tracer_2_dep_attr)

    @property
    def sigma_v_1(self):
        """
        The velocity dispersion of the first tracer in km/s.
        """
        return self._sigma_v_1

    @sigma_v_1.setter
    def sigma_v_1(self, value):
        self._sigma_v_1 = value
        if "tracer_1_dep_attr" in dir(self):
            logger.debug(
                f"cleaning cache of {self.tracer_1_dep_attr} due to resetting sigma_v_1"
            )
            self.clean_cache(self.tracer_1_dep_attr)

    @property
    def sigma_z_1(self):
        """
        The redshift error of the first tracer.
        """
        return self._sigma_z_1

    @sigma_z_1.setter
    def sigma_z_1(self, value):
        self._sigma_z_1 = value
        if "tracer_1_dep_attr" in dir(self):
            logger.debug(
                f"cleaning cache of {self.tracer_1_dep_attr} due to resetting sigma_z_1"
            )
            self.clean_cache(self.tracer_1_dep_attr)

    @property
    def sigma_v_2(self):
        """
        The velocity dispersion of the second tracer in km/s.
        """
        return self._sigma_v_2

    @sigma_v_2.setter
    def sigma_v_2(self, value):
        self._sigma_v_2 = value
        if "tracer_2_dep_attr" in dir(self):
            logger.debug(
                f"cleaning cache of {self.tracer_2_dep_attr} due to resetting sigma_v_2"
            )
            self.clean_cache(self.tracer_2_dep_attr)

    @property
    def sigma_z_2(self):
        """
        The redshift error of the second tracer.
        """
        return self._sigma_z_2

    @sigma_z_2.setter
    def sigma_z_2(self, value):
        self._sigma_z_2 = value
        if "tracer_2_dep_attr" in dir(self):
            logger.debug(
                f"cleaning cache of {self.tracer_2_dep_attr} due to resetting sigma_z_2"
            )
            self.clean_cache(self.tracer_2_dep_attr)

    @property
    def include_beam(self):
        """
        Whether the beam attenuation is included in the model calculation.
        Must be a list of two booleans, the first for the first tracer and the second for the second tracer.
        If just a boolean is provided, it will be used for both tracers.
        """
        return self._include_beam

    @include_beam.setter
    def include_beam(self, value):
        value_before = self._include_beam
        self._include_beam = value
        if self.sigma_beam_ch is None and (np.array(self.include_beam).sum() > 0):
            logger.debug("no input beam found, setting include_beam to False")
            self._include_beam = [False, False]
        if value_before[0] != value[0]:
            if "tracer_1_dep_attr" in dir(self):
                logger.debug(
                    f"cleaning cache of {self.tracer_1_dep_attr} due to resetting include_beam"
                )
                self.clean_cache(self.tracer_1_dep_attr)
        if value_before[1] != value[1]:
            if "tracer_2_dep_attr" in dir(self):
                logger.debug(
                    f"cleaning cache of {self.tracer_2_dep_attr} due to resetting include_beam"
                )
                self.clean_cache(self.tracer_2_dep_attr)

    @property
    def compensate(self):
        """
        Whether the gridded fields are compensated
        according to the mass assignment scheme.
        Note that the compensation is applied to the model power spectrum,
        and **not** to the gridded data fields.
        """
        return self._compensate

    @compensate.setter
    def compensate(self, value):
        value_before = self._compensate
        if isinstance(value, bool):
            value = (value, value)
        self._compensate = value
        if value_before[0] != value[0]:
            if "tracer_1_dep_attr" in dir(self):
                logger.debug(
                    f"cleaning cache of {self.tracer_1_dep_attr} due to resetting compensate"
                )
                self.clean_cache(self.tracer_1_dep_attr)
        if value_before[1] != value[1]:
            if "tracer_2_dep_attr" in dir(self):
                logger.debug(
                    f"cleaning cache of {self.tracer_2_dep_attr} due to resetting compensate"
                )
                self.clean_cache(self.tracer_2_dep_attr)

    def fog_gaussian(self, sigma_r, kmode=None, mumode=None):
        r"""
        The Gaussian finger-of-god profile.

        .. math::
            {\rm FoG} = {\rm exp}(-(\sigma_r k_\parallel/H)^2/2)

        Note the power spectrum has FoG squared with the two FoG terms that can
        be different for two tracers.

        Parameters
        ----------
        sigma_r: float.
            The velocity dispersion in terms of the comoving distance in Mpc.
        kmode: float, None.
            The mode of 3D k in Mpc-1. If None, self.kmode will be used.
        mumode: float, None.
            The mu values of each 3D k-mode. In None, self.mumode will be used.

        Returns
        -------
        fog: float.
            The FoG term.
        """
        if mumode is None:
            mumode = self.mumode
        if kmode is None:
            kmode = self.kmode
        k_parallel = kmode * mumode
        fog = np.exp(-((sigma_r * k_parallel) ** 2 / 2))
        return fog

    def fog_lorentz(self, sigma_r, kmode=None, mumode=None):
        r"""
        The Lorentzian finger-of-god profile.

        .. math::
            {\rm FoG} = \sqrt{1/(1+(\sigma_r k_\parallel/H)^2)}

        Note the power spectrum has FoG squared with the two FoG terms that can
        be different for two tracers.

        Parameters
        ----------
        sigma_r: float.
            The velocity dispersion in terms of the comoving distance in Mpc.
        kmode: float, None.
            The mode of 3D k in Mpc-1. If None, self.kmode will be used.
        mumode: float, None.
            The mu values of each 3D k-mode. In None, self.mumode will be used.

        Returns
        -------
        fog: float.
            The FoG term.
        """
        if mumode is None:
            mumode = self.mumode
        if kmode is None:
            kmode = self.kmode
        k_parallel = kmode * mumode
        fog = np.sqrt(1 / (1 + (sigma_r * k_parallel) ** 2))
        return fog

    def fog_term(self, sigma_r, kmode=None, mumode=None):
        """
        The FoG term for the model calculation.
        It reads the profile type from the attribute ``fog_profile``.

        Parameters
        ----------
        sigma_r: float.
            The velocity dispersion in terms of the comoving distance in Mpc.
        kmode: float, None.
            The mode of 3D k in Mpc-1. If None, self.kmode will be used.
        mumode: float, None.
            The mu values of each 3D k-mode. In None, self.mumode will be used.

        Returns
        -------
        fog: float.
            The FoG term.
        """
        return getattr(self, "fog_" + self.fog_profile)(sigma_r, kmode, mumode)

    @property
    def tracer_bias_1(self):
        """
        The linear bias of the first tracer.
        """
        return self._tracer_bias_1

    @tracer_bias_1.setter
    def tracer_bias_1(self, value):
        self._tracer_bias_1 = value
        if "tracer_1_dep_attr" in dir(self):
            logger.debug(
                f"cleaning cache of {self.tracer_1_dep_attr} due to resetting tracer_bias_1"
            )
            self.clean_cache(self.tracer_1_dep_attr)

    @property
    def tracer_bias_2(self):
        """
        The linear bias of the second tracer.
        """
        return self._tracer_bias_2

    @tracer_bias_2.setter
    def tracer_bias_2(self, value):
        self._tracer_bias_2 = value
        if "tracer_2_dep_attr" in dir(self):
            logger.debug(
                f"cleaning cache of {self.tracer_2_dep_attr} due to resetting tracer_bias_2"
            )
            self.clean_cache(self.tracer_2_dep_attr)

    @property
    def cross_coeff(self):
        """
        The cross-correlation coefficient between the two tracers.
        """
        return self._cross_coeff

    @cross_coeff.setter
    def cross_coeff(self, value):
        self._cross_coeff = value
        if "cross_coeff_dep_attr" in dir(self):
            logger.debug(
                f"cleaning cache of {self.cross_coeff_dep_attr} due to resetting cross_coeff"
            )
            self.clean_cache(self.cross_coeff_dep_attr)

    @property
    def kmode(self):
        """
        The input kmode for the model calculation.
        """
        return self._kmode

    @kmode.setter
    def kmode(self, value):
        self._kmode = np.asarray(value, dtype=self.real_dtype)
        if "kmode_dep_attr" in dir(self):
            logger.debug(
                f"cleaning cache of {self.kmode_dep_attr} due to resetting kmode"
            )
            self.clean_cache(self.kmode_dep_attr)

    @property
    def mumode(self):
        """
        The mu values of each 3D k-mode.
        """
        return self._mumode

    @mumode.setter
    def mumode(self, value):
        self._mumode = value
        if "mumode_dep_attr" in dir(self):
            logger.debug(
                f"cleaning cache of {self.mumode_dep_attr} due to resetting mumode"
            )
            self.clean_cache(self.mumode_dep_attr)

    @property
    def sampling_resol(self):
        """
        The sampling resolution corresponding to the map-making/gridding
        of the density field.
        """
        return self._sampling_resol

    @sampling_resol.setter
    def sampling_resol(self, value):
        self._sampling_resol = value
        self.has_resol = True
        if self.include_sky_sampling[0]:
            if "tracer_1_dep_attr" in dir(self):
                logger.debug(
                    f"cleaning cache of {self.tracer_1_dep_attr} due to resetting sampling_resol"
                )
                self.clean_cache(self.tracer_1_dep_attr)
        if self.include_sky_sampling[1]:
            if "tracer_2_dep_attr" in dir(self):
                logger.debug(
                    f"cleaning cache of {self.tracer_2_dep_attr} due to resetting sampling_resol"
                )
                self.clean_cache(self.tracer_2_dep_attr)

    @property
    @tagging("cosmo_model", "nu", "kmode")
    def auto_power_matter_model_r(self):
        """
        The model matter power spectrum in real space (without RSD).
        """
        if self._auto_power_matter_model_r is None:
            self.get_model_matter_power_r()
        return self._auto_power_matter_model_r

    def get_model_matter_power_r(self):
        """
        Calculate the model matter power spectrum in real space (without RSD).
        The attribute f"_auto_power_matter_model_r" will be set by the output.
        """
        self._auto_power_matter_model_r = self.matter_power_spectrum_fnc(self.kmode)

    @property
    @tagging("cosmo_model", "nu", "kmode", "mumode", "rsd")
    def auto_power_matter_model(self):
        """
        The model matter power spectrum with RSD effects.
        The 3D k-modes corrospond to the input ``kmode`` and ``mumode``.
        """
        if self._auto_power_matter_model is None:
            self.get_model_matter_power()
        return self._auto_power_matter_model

    @property
    @tagging("cosmo_model", "nu", "kmode", "mumode", "tracer_1", "rsd")
    def auto_power_tracer_1_model_noobs(self):
        """
        The model power spectrum for the first tracer without observational effects.
        *Note that the power is in units of volume, so the mean amplitude is not applied.*
        """
        if self._auto_power_tracer_1_model_noobs is None:
            self.get_model_power_noobs_i(1)
        return self._auto_power_tracer_1_model_noobs

    @property
    @tagging("cosmo_model", "nu", "kmode", "mumode", "tracer_2", "rsd")
    def auto_power_tracer_2_model_noobs(self):
        """
        The model power spectrum for the second tracer without observational effects.
        *Note that the power is in units of volume, so the mean amplitude is not applied.*
        """
        if self._auto_power_tracer_2_model_noobs is None:
            self.get_model_power_noobs_i(2)
        return self._auto_power_tracer_2_model_noobs

    @property
    @tagging(
        "cosmo_model",
        "nu",
        "kmode",
        "mumode",
        "tracer_1",
        "tracer_2",
        "rsd",
        "cross_coeff",
    )
    def cross_power_tracer_model_noobs(self):
        """
        The model power spectrum for the cross correlation between the two tracers without observational effects.
        *Note that the power is in units of volume, so the mean amplitude is not applied.*
        """
        if self._cross_power_tracer_model_noobs is None:
            self.get_model_power_noobs_cross()
        return self._cross_power_tracer_model_noobs

    @property
    @tagging("cosmo_model", "nu", "kmode", "mumode", "tracer_1", "beam", "rsd")
    def auto_power_tracer_1_model(self):
        """
        The 3D model power spectrum for the first tracer.
        The 3D k-modes corrospond to the input ``kmode`` and ``mumode``.
        Unlike ``noobs`` power, the mean amplitude is applied.
        """
        if self._auto_power_tracer_1_model is None:
            self.get_model_power_i(1)
        mean_amp = self.mean_amp_value(1)
        logger.info(
            f"multiplying _auto_power_tracer_1_model with mean_amp_1**2: {mean_amp}**2"
            " to get auto_power_tracer_1_model",
        )
        return self._auto_power_tracer_1_model * mean_amp**2

    @property
    @tagging("cosmo_model", "nu", "kmode", "mumode", "tracer_2", "beam", "rsd")
    def auto_power_tracer_2_model(self):
        """
        The 3D model power spectrum for the second tracer.
        The 3D k-modes corrospond to the input ``kmode`` and ``mumode``.
        Unlike ``noobs`` power, the mean amplitude is applied.
        """
        if self._auto_power_tracer_2_model is None:
            self.get_model_power_i(2)
        mean_amp = self.mean_amp_value(2)
        logger.info(
            f"multiplying _auto_power_tracer_2_model with mean_amp_2**2: {mean_amp}**2"
            " to get auto_power_tracer_2_model",
        )
        return self._auto_power_tracer_2_model * mean_amp**2

    @property
    @tagging(
        "cosmo_model",
        "nu",
        "kmode",
        "mumode",
        "tracer_2",
        "beam",
        "tracer_1",
        "rsd",
        "cross_coeff",
    )
    def cross_power_tracer_model(self):
        """
        The 3D model cross power spectrum between the two tracers.
        The 3D k-modes corrospond to the input ``kmode`` and ``mumode``.
        Unlike ``noobs`` power, the mean amplitude is applied.
        """
        if self._cross_power_tracer_model is None:
            self.get_model_power_cross()
        mean_amp = self.mean_amp_value(1)
        mean_amp2 = self.mean_amp_value(2)
        logger.info(
            f"multiplying _cross_power_tracer_model with mean_amp: {mean_amp} and mean_amp2: {mean_amp2} "
            " to get cross_power_tracer_model",
        )
        return self._cross_power_tracer_model * mean_amp * mean_amp2

    def mean_amp_value(self, i: int) -> float:
        """
        Resolve ``mean_amp_i`` to a float (attribute name lookup if a string).
        """
        mean_amp = getattr(self, f"mean_amp_{i}")
        if isinstance(mean_amp, str):
            logger.info("getting mean_amp_%s from self.%s", i, mean_amp)
            mean_amp = getattr(self, mean_amp)
        if mean_amp is None:
            return 1.0
        return float(mean_amp)

    def power_kmu(
        self, which: str = "auto_1", include_mean_amp: bool = False
    ) -> NDArray[np.floating]:
        r"""
        Anisotropic model :math:`P(k,\mu)` on the current ``kmode`` / ``mumode``.

        Returns the tracer auto or cross spectrum **without** beam, map
        sampling, MAS compensation, or survey-weight convolution — i.e. the
        ``*_model_noobs`` caches. Observational factors are applied later by
        :meth:`get_model_power_i` / :meth:`get_model_power_cross` (3D path) or
        by a multipole window matrix.

        Parameters
        ----------
        which : {'auto_1', 'auto_2', 'cross'}
            Tracer combination.
        include_mean_amp : bool, default False
            If True, multiply by ``mean_amp`` (squared for autos).
        """
        which_s = str(which).lower()
        if which_s == "auto_1":
            power = np.asarray(self.auto_power_tracer_1_model_noobs, dtype=float)
            if include_mean_amp:
                power = power * self.mean_amp_value(1) ** 2
            return power
        if which_s == "auto_2":
            power = np.asarray(self.auto_power_tracer_2_model_noobs, dtype=float)
            if include_mean_amp:
                power = power * self.mean_amp_value(2) ** 2
            return power
        if which_s == "cross":
            power = np.asarray(self.cross_power_tracer_model_noobs, dtype=float)
            if include_mean_amp:
                power = power * self.mean_amp_value(1) * self.mean_amp_value(2)
            return power
        raise ValueError("which must be 'auto_1', 'auto_2', or 'cross'")

    def power_kmu_on_grid(
        self,
        k_in: ArrayLike,
        mu: ArrayLike,
        which: str = "auto_1",
        include_mean_amp: bool = True,
    ) -> NDArray[np.floating]:
        r"""
        Evaluate :meth:`power_kmu` on a broadcast ``(n_k, n_mu)`` mesh.

        Temporarily retargets ``kmode`` / ``mumode`` and clears dependent
        caches; restores them afterward.
        """
        k_in_np = np.asarray(k_in, dtype=float)
        mu_np = np.asarray(mu, dtype=float)
        k_mesh = np.broadcast_to(k_in_np[:, None], (k_in_np.size, mu_np.size)).copy()
        mu_mesh = np.broadcast_to(mu_np[None, :], (k_in_np.size, mu_np.size)).copy()

        old_kmode = self.kmode
        old_mumode = self.mumode
        self._auto_power_matter_model_r = None
        self._auto_power_matter_model = None
        self._auto_power_tracer_1_model_noobs = None
        self._auto_power_tracer_2_model_noobs = None
        self._cross_power_tracer_model_noobs = None
        try:
            self.kmode = k_mesh
            self.mumode = mu_mesh
            return self.power_kmu(which=which, include_mean_amp=include_mean_amp)
        finally:
            self._auto_power_matter_model_r = None
            self._auto_power_matter_model = None
            self._auto_power_tracer_1_model_noobs = None
            self._auto_power_tracer_2_model_noobs = None
            self._cross_power_tracer_model_noobs = None
            self.kmode = old_kmode
            self.mumode = old_mumode

    def _warn_if_global_los_for_continuous_multipoles(self) -> None:
        """
        Prefer :meth:`~meer21cm.power.PowerSpectrum.get_1d_power` under global LOS.

        Continuous multipoles ignore discrete FFT-shell :math:`\\mu` sampling;
        for ``los='global'`` the estimator-matched 1D path is
        ``get_1d_power``.
        """
        if not hasattr(self, "los"):
            return
        if str(getattr(self, "los")).lower() != "global":
            return
        warnings.warn(
            "los='global': preferred 1D power is get_1d_power (discrete FFT "
            "shells), not get_theory_multipoles_kmu (continuous μ integral).",
            UserWarning,
            stacklevel=3,
        )

    def get_theory_multipoles_kmu(
        self,
        k_in: ArrayLike,
        ells: Sequence[int] = (0, 2, 4),
        nmu: int = 64,
        which: str = "auto_1",
    ) -> dict[str, Any]:
        r"""
        Continuous spherical multipoles at ``k_in``.

        .. math::

            P_\ell(k)
            =
            \frac{2\ell+1}{2}
            \int_{-1}^{1}\!\mathrm{d}\mu\,
            P(k,\mu)\,
            \mathcal{L}_\ell(\mu)

        using Gauss–Legendre quadrature (``nmu`` nodes). ``P(k,\mu)`` comes
        from :meth:`power_kmu_on_grid` (noobs + mean amplitude; no beam /
        sampling / MAS).

        If this object has ``los='global'``, a :class:`UserWarning` notes that
        :meth:`~meer21cm.power.PowerSpectrum.get_1d_power` is preferred for
        estimator-matched 1D power.
        """
        self._warn_if_global_los_for_continuous_multipoles()
        ells_t = tuple(int(e) for e in ells)
        k_in_np = np.asarray(k_in, dtype=float)
        mu_nodes, mu_w = np.polynomial.legendre.leggauss(int(nmu))
        power = self.power_kmu_on_grid(
            k_in_np, mu_nodes, which=which, include_mean_amp=True
        )

        P_ell: dict[int, NDArray[np.floating]] = {}
        for ell in ells_t:
            L = eval_legendre(ell, mu_nodes)
            integ = np.sum(power * L[None, :] * mu_w[None, :], axis=1)
            P_ell[ell] = (2 * ell + 1) / 2.0 * integ
        return {"k": k_in_np, "P_ell": P_ell, "ells": ells_t, "nmu": int(nmu)}

    def map_sampling(self):
        """
        The sampling window function from the map cube to be convolved with data.
        Note that the window can only be calculated in Cartesian grids, so it is not used
        in ``ModelPowerSpectrum`` and only in ``PowerSpectrum``.
        """
        return 1.0

    def gridding_compensation(self):
        """
        The sampling window function to be compensated for the gridding mass assignment scheme.
        Note that the window can only be calculated in Cartesian grids, so it is not used
        in ``ModelPowerSpectrum`` and only in ``PowerSpectrum``.
        """
        return 1.0

    # calculate on the fly, no need for tagging
    def beam_attenuation(self):
        """
        The beam attenuation factor.
        """
        if self.sigma_beam_ch is None:
            return 1.0
        # in the future for asymmetric beam this way
        # of writing may be probelmatic
        # Numerical roundoff (more visible in lower-precision runs) can push
        # |mu| slightly above 1 and make 1 - mu^2 marginally negative.
        mu_sq = np.square(self.mumode)
        one_minus_mu_sq = np.clip(1 - mu_sq, 0.0, None)
        k_perp = self.kmode * np.sqrt(one_minus_mu_sq)
        sigma_beam_mpc = self.sigma_beam_in_mpc
        B_beam = gaussian_beam_attenuation(k_perp, sigma_beam_mpc)
        return B_beam

    def cal_rsd_power(
        self,
        power_in_real_space,
        beta1,
        sigma_v_1,
        sigma_z_1,
        beta2=None,
        sigma_v_2=None,
        sigma_z_2=None,
        r=1.0,
        mumode=None,
    ):
        """
        Calculate the redshift space power spectrum.
        If properties of the second tracer are not set,
        they will be set to the same as the first tracer so
        the result will be the auto power spectrum.

        Parameters
        ----------
        power_in_real_space: np.ndarray
            The power spectrum in real space.

        beta1: float
            The growth rate over bias of the first tracer.
        sigma_v_1: float
            The velocity dispersion of the first tracer.
        sigma_z_1: float
            The redshift dispersion of the first tracer.
        beta2: float, default None
            The growth rate over bias of the second tracer.
        sigma_v_2: float, default None
            The velocity dispersion of the second tracer.
        sigma_z_2: float, default None
            The redshift dispersion of the second tracer.
        r: float, default 1.0
            The correlation coefficient between the two tracers.
        mumode: np.ndarray, default None
            The mu values of each 3D k-mode.

        Returns
        -------
        power_in_redshift_space: np.ndarray
            The power spectrum in redshift space.
        """
        if mumode is None:
            mumode = self.mumode
        if beta2 is None:
            beta2 = beta1
        if sigma_v_2 is None:
            sigma_v_2 = sigma_v_1
        if sigma_z_2 is None:
            sigma_z_2 = sigma_z_1
        power_in_redshift_space = (
            power_in_real_space
            * (r + (beta1 + beta2) * mumode**2 + beta1 * beta2 * mumode**4)
            * self.fog_term(self.deltav_to_deltar(sigma_v_1), mumode=mumode)
            * self.fog_term(self.deltav_to_deltar(sigma_v_2), mumode=mumode)
            # fog is exp(-k^2 sigma^2 / 2), whereas redshift error
            # is exp(-k^2 deltar^2)
            * self.fog_gaussian(self.deltaz_to_deltar(sigma_z_1), mumode=mumode)
            * self.fog_gaussian(self.deltaz_to_deltar(sigma_z_2), mumode=mumode)
        )
        return power_in_redshift_space

    def get_model_matter_power(self):
        """
        Calculate the model matter power spectrum.
        The attribute f"_auto_power_matter_model" will be set by the output.
        """
        pk3d_mm_r = self.auto_power_matter_model_r
        if self.kaiser_rsd:
            beta_m = self.f_growth_true
            self._auto_power_matter_model = self.cal_rsd_power(
                pk3d_mm_r,
                beta_m,
                0.0,
                0.0,
            )
        else:
            self._auto_power_matter_model = pk3d_mm_r
        logger.debug(
            "calculated model matter power spectrum, kaiser rsd: %s", self.kaiser_rsd
        )
        logger.debug("model matter power spectrum: %s", self._auto_power_matter_model)

    def get_model_power_noobs_i(self, i):
        """
        Calculate the model power spectrum for the i-th tracer without observational effects.
        The attribute f"_auto_power_tracer_{i}_model_noobs" will be set by the output.
        """
        tracer_bias_i = getattr(self, "tracer_bias_" + str(i))
        if tracer_bias_i is None:
            # For the second tracer, many higher-level APIs (e.g. PowerSpectrum, MockSimulation)
            # rely on the existence of a well-defined theoretical model. If ``tracer_bias_2``
            # is not set, calculating these model power spectra would fail in a non-obvious way
            # later on. Raise an explicit error here instead.
            if i == 2:
                raise ValueError(
                    "tracer_bias_2 is not set, so the theoretical power spectrum for "
                    "tracer_2 (and any cross-correlation involving tracer_2) cannot be "
                    "computed. Please pass tracer_bias_2 when initialising the object "
                    "or set ``obj.tracer_bias_2`` before accessing tracer_2 model power."
                )
        pk3d_mm_r = self.auto_power_matter_model_r
        # tracer in real space is just the matter ps times the bias
        pk3d_tt_r = tracer_bias_i**2 * pk3d_mm_r
        # apply the RSD
        if self.kaiser_rsd:
            beta_i = self.f_growth_true / tracer_bias_i
            power_noobs_i = self.cal_rsd_power(
                pk3d_tt_r,
                beta_i,
                getattr(self, "sigma_v_" + str(i)),
                getattr(self, "sigma_z_" + str(i)),
            )
        else:
            power_noobs_i = pk3d_tt_r
        logger.info(
            f"{inspect.currentframe().f_code.co_name}: setting self._auto_power_tracer_{i}_model_noobs"
        )
        setattr(self, f"_auto_power_tracer_{i}_model_noobs", power_noobs_i)

    def get_model_power_i(self, i):
        """
        Calculate the model power spectrum for the i-th tracer.
        The attribute f"_auto_power_tracer_{i}_model" will be set by the output.

        Starts from :meth:`power_kmu` (cosmo + RSD), then applies beam, map
        sampling, MAS compensation, and survey-weight convolution.

        Parameters
        ----------
        i: int
            The index of the tracer.

        Returns
        -------
        auto_power_model: np.ndarray
            The model power spectrum for the i-th tracer.
        """
        logger.debug(
            "calculating model power for tracer %s with bias %s",
            i,
            getattr(self, "tracer_bias_" + str(i)),
        )
        B_beam = self.beam_attenuation()
        B_sampling = self.map_sampling()
        B_comp = self.gridding_compensation()
        tracer_beam_indx = np.array(self.include_beam).astype("int")[i - 1]
        tracer_samp_indx = np.array(self.include_sky_sampling).astype("int")[i - 1]
        tracer_comp_indx = np.array(self.compensate).astype("int")[i - 1]
        auto_power_model = self.power_kmu(f"auto_{i}", include_mean_amp=False).copy()
        # first apply the beam
        logger.debug("applying beam attenuation?: %s", tracer_beam_indx)
        auto_power_model *= B_beam ** (tracer_beam_indx * 2)
        # then apply the sky-map sampling and gridding compensation
        logger.debug("applying sky-map sampling?: %s", tracer_samp_indx)
        auto_power_model *= B_sampling ** (tracer_samp_indx * 2)
        logger.debug("applying gridding compensation?: %s", tracer_comp_indx)
        auto_power_model *= B_comp ** (tracer_comp_indx * 2)
        # then the weights in the grid space before FFT
        # assume map-making, gridding and field-level weights are commutable
        weights_grid = self.get_weights_none_to_one("weights_grid_" + str(i))
        weights_field = self.get_weights_none_to_one("weights_field_" + str(i))
        weights_tot = weights_field * weights_grid
        logger.debug("applying weights convolution: %s", weights_tot)
        auto_power_model = get_modelpk_conv(
            auto_power_model,
            weights1_in_real=weights_tot,
            weights2=weights_tot,
            renorm=True,
        )
        logger.info(
            f"{inspect.currentframe().f_code.co_name}: "
            f"setting self._auto_power_tracer_{i}_model"
        )
        setattr(self, "_auto_power_tracer_" + str(i) + "_model", auto_power_model)
        return auto_power_model

    def get_model_power_noobs_cross(self):
        """
        Calculate the model cross power spectrum between the two tracers without observational effects.
        The attribute f"_cross_power_tracer_model_noobs" will be set by the output.
        """
        if self.tracer_bias_2 is None:
            raise ValueError(
                "tracer_bias_2 is not set, so the theoretical cross power spectrum for "
                "cross-correlation cannot be computed. "
                "Please pass tracer_bias_2 when initialising the object "
                "or set ``obj.tracer_bias_2`` before accessing cross-power model quantities."
            )
        pk3d_mm_r = self.auto_power_matter_model_r
        pk3d_tt_r = self.tracer_bias_1 * self.tracer_bias_2 * pk3d_mm_r
        if self.kaiser_rsd:
            beta_1 = self.f_growth_true / self.tracer_bias_1
            beta_2 = self.f_growth_true / self.tracer_bias_2
            result = self.cal_rsd_power(
                pk3d_tt_r,
                beta1=beta_1,
                sigma_v_1=self.sigma_v_1,
                sigma_z_1=self.sigma_z_1,
                beta2=beta_2,
                sigma_v_2=self.sigma_v_2,
                sigma_z_2=self.sigma_z_2,
                r=self.cross_coeff,
            )
        else:
            result = pk3d_tt_r * self.cross_coeff
        self._cross_power_tracer_model_noobs = result

    def get_model_power_cross(self):
        """
        Calculate the model cross power spectrum between the two tracers.
        The attribute f"_cross_power_tracer_model" will be set by the output.

        Starts from :meth:`power_kmu` (cosmo + RSD), then applies beam, map
        sampling, MAS compensation, and survey-weight convolution.
        """
        if getattr(self, "tracer_bias_" + str(2)) is None:
            raise ValueError(
                "tracer_bias_2 is not set, so the theoretical cross power spectrum "
                "between tracer_1 and tracer_2 cannot be computed. Please pass "
                "tracer_bias_2 when initialising the object or set "
                "``obj.tracer_bias_2`` before accessing cross-power model quantities."
            )
        B_beam = self.beam_attenuation()
        B_sampling = self.map_sampling()
        B_comp = self.gridding_compensation()
        tracer_beam_indx = np.array(self.include_beam).astype("int")
        tracer_samp_indx = np.array(self.include_sky_sampling).astype("int")
        tracer_comp_indx = np.array(self.compensate).astype("int")
        self._cross_power_tracer_model = self.power_kmu(
            "cross", include_mean_amp=False
        ).copy()
        # then apply the beam, sky-map sampling, and gridding compensation
        logger.debug(
            "applying beam attenuation for tracer 1 and/or 2?: %s", tracer_beam_indx
        )
        self._cross_power_tracer_model *= B_beam ** (
            tracer_beam_indx[0] + tracer_beam_indx[1]
        )
        logger.debug(
            "applying sky-map sampling for tracer 1 and/or 2?: %s", tracer_samp_indx
        )
        self._cross_power_tracer_model *= B_sampling ** (
            tracer_samp_indx[0] + tracer_samp_indx[1]
        )
        logger.debug(
            "applying gridding compensation for tracer 1 and/or 2?: %s",
            tracer_comp_indx,
        )
        self._cross_power_tracer_model *= B_comp ** (
            tracer_comp_indx[0] + tracer_comp_indx[1]
        )
        # then the weights in the grid space before FFT
        weights_grid_1 = self.get_weights_none_to_one("weights_grid_1")
        weights_field_1 = self.get_weights_none_to_one("weights_field_1")
        weights_grid_2 = self.get_weights_none_to_one("weights_grid_2")
        weights_field_2 = self.get_weights_none_to_one("weights_field_2")
        weights_tot_1 = weights_field_1 * weights_grid_1
        weights_tot_2 = weights_field_2 * weights_grid_2
        logger.debug(
            "applying weights convolution: %s and %s", weights_tot_1, weights_tot_2
        )
        logger.info(
            f"{inspect.currentframe().f_code.co_name}: "
            f"setting self._cross_power_tracer_model"
        )
        self._cross_power_tracer_model = get_modelpk_conv(
            self._cross_power_tracer_model,
            weights1_in_real=weights_tot_1,
            weights2=weights_tot_2,
            renorm=True,
        )
        return self._cross_power_tracer_model
