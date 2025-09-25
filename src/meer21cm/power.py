"""
This module handles computation of power spectrum from gridded fields and its corresponding model power spectrum from theory.

The class :py:class:`ModelPowerSpectrum` is the class for computing the model power spectrum of an LSS tracer field.

The class :py:class:`FieldPowerSpectrum` is the class for computing the power spectrum of a gridded field from LSS data.

The class :py:class:`PowerSpectrum` coherently combines the two classes above,
and provides an interface for gridding the intensity mapping data as well as the galaxy catalogue to perform
power spectrum estimation and for auto-correlation and cross-correlation.
"""
import numpy as np
from meer21cm.cosmology import CosmologyCalculator
from meer21cm.grid import (
    minimum_enclosing_box_of_lightcone,
    project_particle_to_regular_grid,
    interlace_two_fields,
    fourier_window_for_assignment,
)
from scipy.signal import windows
from meer21cm.util import (
    tagging,
    radec_to_indx,
    redshift_to_freq,
    freq_to_redshift,
    get_nd_slicer,
    omega_hi_to_average_temp,
)
from meer21cm.dataanalysis import Specification
import healpy as hp
from collections.abc import Iterable
import inspect
import logging

logger = logging.getLogger(__name__)


class ModelPowerSpectrum(CosmologyCalculator):
    r"""
    The class for computing the model power spectrum of an LSS tracer field.

    Parameters
    ----------
    kmode: np.ndarray, default None
        The mode of k in Mpc-1.
    mumode: np.ndarray, default None
        The mu values of each k-mode so that :math:`k_\parallel = k \times \mu`.
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
        self._kmode = value
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
    @tagging("cosmo", "nu", "kmode")
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
    @tagging("cosmo", "nu", "kmode", "mumode", "rsd")
    def auto_power_matter_model(self):
        """
        The model matter power spectrum with RSD effects.
        The 3D k-modes corrospond to the input ``kmode`` and ``mumode``.
        """
        if self._auto_power_matter_model is None:
            self.get_model_matter_power()
        return self._auto_power_matter_model

    @property
    @tagging("cosmo", "nu", "kmode", "mumode", "tracer_1", "rsd")
    def auto_power_tracer_1_model_noobs(self):
        """
        The model power spectrum for the first tracer without observational effects.
        *Note that the power is in units of volume, so the mean amplitude is not applied.*
        """
        if self._auto_power_tracer_1_model_noobs is None:
            self.get_model_power_noobs_i(1)
        return self._auto_power_tracer_1_model_noobs

    @property
    @tagging("cosmo", "nu", "kmode", "mumode", "tracer_2", "rsd")
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
        "cosmo", "nu", "kmode", "mumode", "tracer_1", "tracer_2", "rsd", "cross_coeff"
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
    @tagging("cosmo", "nu", "kmode", "mumode", "tracer_1", "beam", "rsd")
    def auto_power_tracer_1_model(self):
        """
        The 3D model power spectrum for the first tracer.
        The 3D k-modes corrospond to the input ``kmode`` and ``mumode``.
        Unlike ``noobs`` power, the mean amplitude is applied.
        """
        if self._auto_power_tracer_1_model is None:
            self.get_model_power_i(1)
        mean_amp = self.mean_amp_1
        if isinstance(mean_amp, str):
            logger.info(f"getting mean_amp_1 from self.{mean_amp}")
            mean_amp = getattr(self, mean_amp)
        logger.info(
            f"multiplying _auto_power_tracer_1_model with mean_amp_1**2: {mean_amp}**2"
            " to get auto_power_tracer_1_model",
        )
        return self._auto_power_tracer_1_model * mean_amp**2

    @property
    @tagging("cosmo", "nu", "kmode", "mumode", "tracer_2", "beam", "rsd")
    def auto_power_tracer_2_model(self):
        """
        The 3D model power spectrum for the second tracer.
        The 3D k-modes corrospond to the input ``kmode`` and ``mumode``.
        Unlike ``noobs`` power, the mean amplitude is applied.
        """
        if self._auto_power_tracer_2_model is None:
            self.get_model_power_i(2)
        # if still None, means tracer 2 is not set
        if self._auto_power_tracer_2_model is None:
            return None
        mean_amp = self.mean_amp_2
        if isinstance(mean_amp, str):
            logger.info(f"getting mean_amp_2 from self.{mean_amp}")
            mean_amp = getattr(self, mean_amp)
        logger.info(
            f"multiplying _auto_power_tracer_2_model with mean_amp_2**2: {mean_amp}**2"
            " to get auto_power_tracer_2_model",
        )
        return self._auto_power_tracer_2_model * mean_amp**2

    @property
    @tagging(
        "cosmo",
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
        # if still None, means tracer 2 is not set
        if self._cross_power_tracer_model is None:
            return None
        mean_amp2 = self.mean_amp_2
        if isinstance(mean_amp2, str):
            mean_amp2 = getattr(self, mean_amp2)
        mean_amp = self.mean_amp_1
        if isinstance(mean_amp, str):
            mean_amp = getattr(self, mean_amp)
        logger.info(
            f"multiplying _cross_power_tracer_model with mean_amp: {mean_amp} and mean_amp2: {mean_amp2} "
            " to get cross_power_tracer_model",
        )
        return self._cross_power_tracer_model * mean_amp * mean_amp2

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
        k_perp = self.kmode * np.sqrt(1 - self.mumode**2)
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
            beta_m = self.f_growth
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
            return None
        pk3d_mm_r = self.auto_power_matter_model_r
        # tracer in real space is just the matter ps times the bias
        pk3d_tt_r = tracer_bias_i**2 * pk3d_mm_r
        # apply the RSD
        if self.kaiser_rsd:
            beta_i = self.f_growth / tracer_bias_i
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

        Parameters
        ----------
        i: int
            The index of the tracer.

        Returns
        -------
        auto_power_model: np.ndarray
            The model power spectrum for the i-th tracer.
        """
        if getattr(self, "tracer_bias_" + str(i)) is None:
            logger.info("tracer_bias_%s is None, returning None", i)
            return None
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
        auto_power_model = getattr(self, f"auto_power_tracer_{i}_model_noobs").copy()
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
        if self.tracer_bias_1 is None or self.tracer_bias_2 is None:
            return None
        pk3d_mm_r = self.auto_power_matter_model_r
        pk3d_tt_r = self.tracer_bias_1 * self.tracer_bias_2 * pk3d_mm_r
        if self.kaiser_rsd:
            beta_1 = self.f_growth / self.tracer_bias_1
            beta_2 = self.f_growth / self.tracer_bias_2
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
        """
        if getattr(self, "tracer_bias_" + str(2)) is None:
            logger.info("tracer bias 2 is None, returning None")
        if self.tracer_bias_1 is None or self.tracer_bias_2 is None:
            return None
        B_beam = self.beam_attenuation()
        B_sampling = self.map_sampling()
        B_comp = self.gridding_compensation()
        tracer_beam_indx = np.array(self.include_beam).astype("int")
        tracer_samp_indx = np.array(self.include_sky_sampling).astype("int")
        tracer_comp_indx = np.array(self.compensate).astype("int")
        self._cross_power_tracer_model = self.cross_power_tracer_model_noobs.copy()
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


class FieldPowerSpectrum(Specification):
    """
    The class for computing the power spectrum of a gridded field from LSS data.

    Parameters
    ----------
    field_1: np.ndarray.
        The density field of the first tracer.
    field_2: np.ndarray, default None
        The density field of the second tracer.
        If None, calculation of the second tracer and the cross-correlation will be skipped.
    box_len: list of 3 floats.
        The length of the box along each axis.
    weights_1: np.ndarray, default None
        The weights of the first tracer. Default is uniform weights.
    mean_center_1: bool, default False
        Whether to mean-center the first tracer field.
    unitless_1: bool, default False
        Whether to divide the first tracer field by its mean.
    weights_2: np.ndarray, default None
        The weights of the second tracer. Default is uniform weights.
    mean_center_2: bool, default False
        Whether to mean-center the second tracer field.
    unitless_2: bool, default False
        Whether to divide the second tracer field by its mean.
    **params: dict
        Additional parameters to be passed to the base class :class:`meer21cm.dataanalysis.Specification`.
    """

    def __init__(
        self,
        field_1,
        box_len,
        weights_1=None,
        mean_center_1=False,
        unitless_1=False,
        field_2=None,
        weights_2=None,
        mean_center_2=False,
        unitless_2=False,
        **params,
    ):
        Specification.__init__(self, **params)
        self.field_1 = field_1
        self.field_2 = field_2
        self.weights_1 = weights_1
        self.weights_2 = weights_2
        self.box_len = np.array(box_len)
        self.box_ndim = np.array(field_1.shape)
        self.mean_center_1 = mean_center_1
        self.unitless_1 = unitless_1
        self.mean_center_2 = mean_center_2
        self.unitless_2 = unitless_2
        if field_2 is not None:
            error_message = "field_1 and field_2 must have same dimensions"
            assert np.allclose(field_2.shape, field_1.shape), error_message
        self._fourier_field_1 = None
        self._fourier_field_2 = None

    @property
    def box_len(self):
        """
        The length of all sides of the box in Mpc.
        """
        return self._box_len

    @box_len.setter
    def box_len(self, value):
        self._box_len = value
        if "box_dep_attr" in dir(self):
            logger.debug(
                f"cleaning cache of {self.box_dep_attr} due to resetting box_len"
            )
            self.clean_cache(self.box_dep_attr)

    @property
    def box_resol(self):
        """
        The grid length of each side of the enclosing box in Mpc.
        """
        return self.box_len / self.box_ndim

    @property
    def box_ndim(self):
        """
        The number of grids along each side of the enclosing box.
        To ensure even sampling of +k and -k modes, the number of grids along every axis needs to be odd.
        """
        return self._box_ndim

    @box_ndim.setter
    def box_ndim(self, value):
        self._box_ndim = value
        if "box_dep_attr" in dir(self):
            logger.debug(
                f"cleaning cache of {self.box_dep_attr} due to resetting box_ndim"
            )
            self.clean_cache(self.box_dep_attr)

    def set_corr_type(self, corr_type, tracer_indx):
        """
        A utility function to help decide whether a tracer field
        needs to be mean centred, renormalised by its mean, and shot noise removed.
        Currently only two types are supported, "Gal" and "HI" (case-insensitive).
        If the tracer is galaxy (number counts),
        the auto power spectrum is mean centred, renormalised, and then
        shot noise removed. If HI, none of the above will be performed.

        Parameters
        ----------
        corr_type: str
            The tracer type.
        tracer_indx: int
            Either 1 or 2.
        """
        logger.debug("setting corr_type: %s for tracer %s", corr_type, tracer_indx)
        if corr_type[:3].lower() == "gal":
            mean_center = True
            unitless = True
            mean_amp = 1.0
        elif corr_type[:2].lower() == "hi":
            mean_center = False
            unitless = False
            mean_amp = "average_hi_temp"
        else:
            raise ValueError("unknown corr_type")
        if not tracer_indx in [1, 2]:
            raise ValueError("tracer_indx should be either 1 or 2")
        logger.debug("setting mean_center_%s: %s", tracer_indx, mean_center)
        logger.debug("setting unitless_%s: %s", tracer_indx, unitless)
        logger.debug("setting mean_amp_%s: %s", tracer_indx, mean_amp)
        setattr(self, "mean_center_" + str(tracer_indx), mean_center)
        setattr(self, "unitless_" + str(tracer_indx), unitless)
        setattr(self, "mean_amp_" + str(tracer_indx), mean_amp)

    @property
    def x_vec(self):
        """
        The 3D x-vector of the box.
        """
        return get_x_vector(
            self.box_ndim,
            self.box_resol,
        )

    @property
    def x_mode(self):
        """
        The mode of the 3D x-vector.
        """
        return get_vec_mode(self.x_vec)

    @property
    def k_vec(self):
        """
        The 3D k-vector of the box.
        """
        return get_k_vector(
            self.box_ndim,
            self.box_resol,
        )

    @property
    def k_nyquist(self):
        """
        The Nyquist frequency of the 3D box along each axis.
        """
        k_max = np.array([np.abs(self.k_vec[i]).max() for i in range(len(self.k_vec))])
        return k_max

    @property
    def k_perp(self):
        """
        The perpendicular k-vector of the 3D box.
        """
        return get_vec_mode(self.k_vec[:-1])

    @property
    def k_para(self):
        """
        The parallel k-mode of the 3D box.
        """
        return self.k_vec[-1]

    @property
    def k_mode(self):
        """
        The mode of the 3D k-vector.
        """
        return get_vec_mode(self.k_vec)

    @property
    def field_1(self):
        """
        The density field of the first tracer.
        """
        return self._field_1

    @property
    def field_2(self):
        """
        The density field of the second tracer.
        """
        return self._field_2

    @field_1.setter
    def field_1(self, value):
        # if field is updated, clear fourier field
        self._field_1 = value
        if "field_1_dep_attr" in dir(self):
            logger.debug(
                f"cleaning cache of {self.field_1_dep_attr} due to resetting field_1"
            )
            self.clean_cache(self.field_1_dep_attr)

    @field_2.setter
    def field_2(self, value):
        # if field is updated, clear fourier field
        self._field_2 = value
        if "field_2_dep_attr" in dir(self):
            logger.debug(
                f"cleaning cache of {self.field_2_dep_attr} due to resetting field_2"
            )
            self.clean_cache(self.field_2_dep_attr)

    @property
    def mean_center_1(self):
        """
        Whether field_1 needs to be mean centered
        """
        return self._mean_center_1

    @property
    def mean_center_2(self):
        """
        Whether field_2 needs to be mean centered
        """
        return self._mean_center_2

    @mean_center_1.setter
    def mean_center_1(self, value):
        # if weight is updated, clear fourier field
        self._mean_center_1 = value
        if "field_1_dep_attr" in dir(self):
            logger.debug(
                f"cleaning cache of {self.field_1_dep_attr} due to resetting mean_center_1"
            )
            self.clean_cache(self.field_1_dep_attr)

    @mean_center_2.setter
    def mean_center_2(self, value):
        # if weight is updated, clear fourier field
        self._mean_center_2 = value
        if "field_2_dep_attr" in dir(self):
            logger.debug(
                f"cleaning cache of {self.field_2_dep_attr} due to resetting mean_center_2"
            )
            self.clean_cache(self.field_2_dep_attr)

    @property
    def unitless_1(self):
        """
        Whether field_1 needs to be divided by its mean
        """
        return self._unitless_1

    @property
    def unitless_2(self):
        """
        Whether field_2 needs to be divided by its mean
        """
        return self._unitless_2

    @unitless_1.setter
    def unitless_1(self, value):
        # if weight is updated, clear fourier field
        self._unitless_1 = value
        if "field_1_dep_attr" in dir(self):
            logger.debug(
                f"cleaning cache of {self.field_1_dep_attr} due to resetting unitless_1"
            )
            self.clean_cache(self.field_1_dep_attr)

    @unitless_2.setter
    def unitless_2(self, value):
        # if weight is updated, clear fourier field
        self._unitless_2 = value
        if "field_2_dep_attr" in dir(self):
            logger.debug(
                f"cleaning cache of {self.field_2_dep_attr} due to resetting unitless_2"
            )
            self.clean_cache(self.field_2_dep_attr)

    @property
    @tagging("box", "field_1")
    def fourier_field_1(self):
        """
        The Fourier transform of the density field of the first tracer.
        """
        if self._fourier_field_1 is None:
            self.get_fourier_field_1()
        return self._fourier_field_1

    def get_fourier_field_1(self):
        """
        Calculate the Fourier transform of the density field of the first tracer.
        """
        result = get_fourier_density(
            self.field_1,
            weights=self.weights_1,
            mean_center=self.mean_center_1,
            unitless=self.unitless_1,
        )
        logger.info(
            f"{inspect.currentframe().f_code.co_name}: "
            f"setting self._fourier_field_1"
        )
        self._fourier_field_1 = result

    @property
    @tagging("box", "field_2")
    def fourier_field_2(self):
        """
        The Fourier transform of the density field of the second tracer.
        """
        if self._fourier_field_2 is None:
            self.get_fourier_field_2()
        return self._fourier_field_2

    def get_fourier_field_2(self):
        """
        Calculate the Fourier transform of the density field of the second tracer.
        """
        if self.field_2 is None:
            logger.info("field_2 is None, returning None")
            return None
        result = get_fourier_density(
            self.field_2,
            weights=self.weights_2,
            mean_center=self.mean_center_2,
            unitless=self.unitless_2,
        )
        logger.info(
            f"{inspect.currentframe().f_code.co_name}: "
            f"setting self._fourier_field_2"
        )
        self._fourier_field_2 = result

    # the calculation of this is not heavy, simply on the fly
    @property
    def auto_power_3d_1(self):
        """
        The 3D power spectrum of the first tracer.
        """
        power_spectrum = get_power_spectrum(
            self.fourier_field_1,
            self.box_len,
            weights=self.weights_1,
            renorm=False,
        )
        return power_spectrum * self.renorm_ps_1

    @property
    def auto_power_3d_2(self):
        """
        The 3D power spectrum of the second tracer.
        """
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
    def cross_power_3d(self):
        """
        The 3D cross power spectrum between the two tracers.
        """
        if self.field_2 is None:
            return None
        weights_2 = self.weights_2
        # if none, the default for get_power_spectrum is
        # to use weights_1, here we want separate weights_2
        if weights_2 is None:
            weights_2 = np.ones(self.field_2.shape)
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
    def renorm_ps_1(self):
        """
        The renormalization factor of the power spectrum of the first tracer.
        """
        grid_w = self.get_weights_none_to_one("weights_1")
        field_w = 1.0
        mean_renorm = 1.0
        if hasattr(self, "weights_field_1"):
            field_w = self.get_weights_none_to_one("weights_field_1")
            if self.unitless_1:
                mean_renorm = (field_w * grid_w).sum() / (grid_w).sum()
        return (
            power_weights_renorm(grid_w * field_w, grid_w * field_w) * mean_renorm**2
        )

    @property
    def renorm_ps_2(self):
        """
        The renormalization factor of the power spectrum of the second tracer.
        """
        grid_w = self.get_weights_none_to_one("weights_2")
        field_w = 1.0
        mean_renorm = 1.0
        if hasattr(self, "weights_field_2"):
            field_w = self.get_weights_none_to_one("weights_field_2")
            if self.unitless_2:
                mean_renorm = (field_w * grid_w).sum() / (grid_w).sum()
        return (
            power_weights_renorm(grid_w * field_w, grid_w * field_w) * mean_renorm**2
        )

    @property
    def renorm_ps_cross(self):
        """
        The renormalization factor of the cross power spectrum.
        """
        grid_w_1 = self.get_weights_none_to_one("weights_1")
        field_w_1 = 1.0
        mean_renorm_1 = 1.0
        if hasattr(self, "weights_field_1"):
            field_w_1 = self.get_weights_none_to_one("weights_field_1")
            if self.unitless_1:
                mean_renorm_1 = (field_w_1 * grid_w_1).sum() / (grid_w_1).sum()
        grid_w_2 = self.get_weights_none_to_one("weights_2")
        field_w_2 = 1.0
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


def get_renormed_field(
    real_field,
    weights=None,
    mean_center=False,
    unitless=False,
):
    """
    Mean center the field and renormalise it by dividing the mean.

    Parameters
    ----------
    real_field: np.ndarray
        The real-space field.
    weights: np.ndarray, default None
        The weights of the field.
    mean_center: bool, default False
        Whether to mean center the field.
    unitless: bool, default False
        Whether to make the field unitless.

    Returns
    -------
    field: np.ndarray
        The renormalized field.
    """
    field = np.array(real_field)
    if weights is None:
        weights = np.ones_like(field)
    weights = np.array(weights)
    if mean_center or unitless:
        field_mean = np.sum(weights * field) / np.sum(weights)
    else:
        return real_field
    if mean_center:
        field -= field_mean
    if unitless:
        field /= field_mean
    return field


def get_fourier_density(
    real_field,
    weights=None,
    mean_center=False,
    unitless=False,
    norm="forward",
):
    """
    Perform Fourier transform of a density field in real space. Note that
    this is deliberately written in a way that is not dimension specific.
    It can be used to calculate power spectrum of arbitrary dimension.

    Note that, the field is multiplied by the weights
    and then Fourier-transformed, and is **not weight normalised**.

    Parameters
    ----------
    real_field: np.ndarray
        The real-space field.
    weights: np.ndarray, default None
        The weights of the field.
    mean_center: bool, default False
        Whether to mean center the field.
    unitless: bool, default False
        Whether to make the field unitless.
    norm: str, default "forward"
        The normalization of the Fourier transform. Naming is the same as np.fft.

    Returns
    -------
    fourier_field: np.ndarray
        The Fourier transform of the field.
    """
    field = get_renormed_field(
        real_field,
        weights=weights,
        mean_center=mean_center,
        unitless=unitless,
    )
    if weights is None:
        weights = np.ones_like(field)
    weights = np.array(weights)
    fourier_field = np.fft.rfftn(field * weights, norm=norm)
    return fourier_field


def get_x_vector(box_ndim, box_resol):
    """
    Get the position vector along each direction for a given box.

    Parameters
    ----------
    box_ndim: int
        The number of dimensions of the box.
    box_resol: float
        The resolution of the box.

    Returns
    -------
    xvecarr: tuple
        The position vector along each direction.
    """
    xvecarr = tuple(
        box_resol[i] * (np.arange(box_ndim[i]) + 0.5) for i in range(len(box_ndim))
    )
    return xvecarr


def get_k_vector(box_ndim, box_resol):
    """
    Get the wavenumber vector along each direction
    for a given box.

    Parameters
    ----------
    box_ndim: int
        The number of dimensions of the box.
    box_resol: float
        The resolution of the box.

    Returns
    -------
    kvecarr: tuple
        The wavenumber vector along each direction.
    """
    kvecarr = [
        2
        * np.pi
        * np.fft.fftfreq(
            box_ndim[i],
            d=box_resol[i],
        )
        for i in range(len(box_ndim))
    ]
    kvecarr[-1] = np.abs(kvecarr[-1][: box_ndim[-1] // 2 + 1])
    return kvecarr


def get_vec_mode(vecarr):
    """
    Calculate the mode of the n-dimensional vectors on the grids

    Parameters
    ----------
    vecarr: tuple
        The vectors.

    Returns
    -------
    mode: np.ndarray
        The mode of the vectors.
    """
    result = np.sqrt(
        np.sum(
            (np.meshgrid(*([(vec) ** 2 for vec in vecarr]), indexing="ij")),
            0,
        )
    )
    return result


def get_shot_noise_galaxy(
    gal_count,
    box_len,
    weights=None,
):
    """
    Calculate the shot noise of a galaxy number count field.
    """
    if weights is None:
        weights = np.ones(gal_count.shape)
    shot_noise = (
        np.prod(box_len)
        / np.sum(gal_count)
        * power_weights_renorm(weights, weights)
        * weights.mean() ** 2
    )
    return shot_noise


def get_shot_noise(
    real_field,
    box_len,
    weights=None,
):
    """
    Calculate the shot noise of a field.

    Parameters
    ----------
    real_field: np.ndarray
        The real-space field.
    box_len: tuple
        The length of the box along each direction.
    weights: np.ndarray, default None
        The weights of the field.

    Returns
    -------
    shot_noise: float
        The shot noise of the field.
    """
    box_len = np.array(box_len)
    box_volume = np.prod(box_len)
    if weights is None:
        weights = np.ones(real_field.shape)
    weights = np.array(weights)
    weights_renorm = power_weights_renorm(weights, weights)
    shot_noise = (
        box_volume
        * np.sum((weights * real_field) ** 2)
        / np.sum(weights * real_field) ** 2
        * weights_renorm
        * np.mean(weights) ** 2
    )
    return shot_noise


def get_modelpk_conv(psmod, weights1_in_real=None, weights2=None, renorm=True):
    """
    Convolve a model power spectrum with real-space weights.

    Parameters
    ----------
    psmod: np.ndarray
        The model power spectrum.
    weights1_in_real: np.ndarray, default None
        The real-space weights for the first field. Default is None, which means no weights.
    weights2: np.ndarray, default None
        The real-space weights for the second field. Default is None, which means no weights.
    renorm: bool, default True
        Whether to renormalize the power spectrum.

    Returns
    -------
    power_conv: np.ndarray
        The convolved power spectrum.
    """
    if weights1_in_real is None and weights2 is None:
        return psmod
    if weights1_in_real is None:
        weights1_in_real = np.ones_like(weights2)
    if weights2 is None:
        weights2 = np.ones_like(weights1_in_real)
    assert np.allclose(weights1_in_real.shape, weights2.shape)
    if np.allclose(weights1_in_real, 1) and np.allclose(weights2, 1):
        return psmod
    weights_fourier = np.fft.rfftn(weights1_in_real)
    weights_fourier *= np.conj(np.fft.rfftn(weights2))
    # using fft instead of rfft somehow is wrong, I have no idea why
    # the behaviour of convolving with uniform weights is incorrect at k_xyz=0 due to rfft
    power_conv = (
        np.fft.rfftn(
            np.fft.irfftn(psmod, s=weights2.shape)
            * np.fft.irfftn(weights_fourier, s=weights2.shape)
        )
        / weights1_in_real.size
    ) * (
        (weights2.shape[-1] // 2 * 2) / weights2.shape[-1]
    )  # rfft correction for odd number?
    if renorm:
        weights_renorm = power_weights_renorm(weights1_in_real, weights2=weights2)
        power_conv *= weights_renorm
    return power_conv.real


def power_weights_renorm(weights1_in_real=None, weights2=None):
    r"""
    Calculate the renormalization coefficient based on the weights
    on the density field when calculating power spectrum.
    The renormalization is defined as

    .. math::
        \frac{{N_{\rm grid}}} {\sum_{i} w_1(x_i) w_2(x_i)},

    where :math:`N_{\rm grid}` is the number of grids in the box and
    :math:`i` loops over all the grids.

    Note that this renormaliszation corresponds to the diagonal
    renormalisation matrix that does not change the window function convolution,
    but only renormalises the sum of each row of the window function matrix.
    See Chen (2025) [1] for more details.

    Parameters
    ----------
        weights1_in_real: array, default None.
            The weights of the density field in real space.
            Must be in the shape of the regular grid field.
        weights2: array, default None.
            If cross-correlation, the weights for the second field.

    Returns
    -------
        weights_norm: float.
           The renormalization coefficient.

    References
    ----------
        [1] Chen, Z., 2025, "A quadratic estimator view of the transfer function correction in intensity mapping surveys",
        https://ui.adsabs.harvard.edu/abs/2025MNRAS.542L...1C/abstract.
    """
    if weights1_in_real is None and weights2 is None:
        return 1.0
    if weights1_in_real is None:
        weights1_in_real = np.ones_like(weights2)
    if weights2 is None:
        weights2 = np.ones_like(weights1_in_real)
    weights_norm = weights1_in_real.size / np.sum(weights1_in_real * weights2)
    return weights_norm


def get_power_spectrum(
    fourier_field,
    box_len,
    weights=None,
    field_2=None,
    weights_2=None,
    renorm=True,
):
    """
    Calculate the power spectrum for one/two given Fourier fields.

    Parameters
    ----------
    fourier_field: np.ndarray
        The Fourier field of the first tracer.
    box_len: tuple
        The length of the box along each direction.
    weights: np.ndarray, default None
        The weights of the first tracer **in real space**.
    field_2: np.ndarray, default None
        The Fourier field of the second tracer. If None, it is set to be the same as the first field.
    weights_2: np.ndarray, default None
        The weights of the second tracer **in real space**. **Must be provided if field_2 is provided.**
    renorm: bool, default True
        Whether to renormalize the power spectrum by the weights.

    Returns
    -------
    power: np.ndarray
        The power spectrum.
    """
    box_len = np.array(box_len)
    box_volume = np.prod(box_len)
    if field_2 is None:
        field_2 = fourier_field
    fourier_field = np.array(fourier_field)
    field_2 = np.array(field_2)
    power = np.real(fourier_field * np.conj(field_2))
    if weights is None and weights_2 is None:
        return power * box_volume
    if weights is None:
        weights = np.ones(weights_2.shape)
    if weights_2 is None:
        weights_2 = weights
    # if weights_2 is None, the renormalisation sets it to weights
    weights_norm = power_weights_renorm(weights, weights_2)
    if renorm:
        power *= weights_norm
    return power * box_volume


def get_gaussian_noise_floor(
    sigma_n,
    box_ndim,
    box_volume=1.0,
    counts=None,
):
    """
    Calculate the Gaussian noise floor for a given field.

    Parameters
    ----------
    sigma_n: float
        The standard deviation of the noise before being averaged down by the sampling.
    box_ndim: tuple
        The number of grids of the box along each direction.
    box_volume: float, default 1.0
        The volume of the box.
    counts: np.ndarray, default None
        The number of sampling in the box. If None, it is set to be 1.0.

    Returns
    -------
    noise_floor: float
        The noise floor.
    """
    box_ndim = np.array(box_ndim)
    if counts is None:
        counts = np.ones(box_ndim.tolist())
    counts = np.array(counts)
    box_std = sigma_n / np.sqrt(counts)
    fourier_var = np.sum(box_std**2) / np.prod(box_ndim) ** 2
    return fourier_var * box_volume


def bin_3d_to_1d(
    ps3d,
    kfield,
    k1dedges,
    weights=None,
    error=False,
    vectorize=False,
):
    r"""
    Bin a 3d distribution, e.g. power spectrum :math:`P_{3D}(\vec{k})`, into 1D average.

    Note that, the distribution is unraveled to a 1D array, so essentially an array of any
    dimension would do, as long as ``ps3d``, ``kfield``, and ``weights`` have the same size.

    The mean of the 1D average is calculated as

    .. math::
        \hat{P}_{\rm 1D}^{i} = \big(\sum_j P_{\rm 3D}^{ j} w_{ j} \big)/\big(\sum_j w_{ j}\big),

    where j loops over all the modess that fall into the :math:`i^{\rm th}` bin
    and :math:`w_{ j}` is the weights.

    If ``error`` is set to ``True``, a sampling error is calculated and returned so that

    .. math::
        (\Delta P_{\rm 1D}^{\rm i})^2 = \big(\sum_j (P_{\rm 3D}^{\rm j}-\hat{P}_{\rm 1D}^{\rm i})^2 w_{\rm j}^2 \big) \Big/ \big(\sum_j w_{\rm j}\big)^2.

    Parameters
    ----------
    ps3d: np.ndarray
        The 3D distribution to be binned.
    kfield: np.ndarray
        The k-field of the 3D distribution.
    k1dedges: np.ndarray
        The bin edges for the 1D power spectrum.
    weights: np.ndarray, default None
        The weights for each 3D k-mode of the power spectrum.
    error: bool, default False
        Whether to calculate the sampling error.
    vectorize: bool, default False
        Whether to vectorize the calculation, assuming the first axis is independent realisations.

    Returns
    -------
    ps1d: np.ndarray
        The 1D power spectrum.
    ps1derr: np.ndarray
        The sampling error for the 1D power spectrum. Returned only if ``error`` is ``True``.
    k1deff: np.ndarray
        The effective k-mode for each bin.
    nmodes: np.ndarray
        The number of modes in each bin.
    """
    if not vectorize:
        ps3d = np.array(ps3d)[None, ...]
    if weights is None:
        weights = np.ones_like(ps3d[0])
    ps3d = np.array(ps3d).reshape(len(ps3d), -1)
    kfield = np.array(kfield).ravel()
    weights = np.array(weights).ravel()

    indx = (kfield[:, None] >= k1dedges[None, :-1]) * (
        kfield[:, None] < k1dedges[None, 1:]
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        ps1d = np.sum(
            ps3d[:, :, None] * indx[None, :, :] * weights[None, :, None], 1
        ) / np.sum(indx[None, :, :] * weights[None, :, None], 1)
        k1deff = np.sum(kfield[:, None] * indx * weights[:, None], 0) / np.sum(
            indx * weights[:, None], 0
        )
    if error is True:
        with np.errstate(divide="ignore", invalid="ignore"):
            ps1derr = np.sqrt(
                np.sum(
                    (ps3d[:, :, None] - ps1d[:, None, :]) ** 2
                    * (indx[None, :, :] * weights[None, :, None]) ** 2,
                    1,
                )
                / np.sum((indx[None, :, :] * weights[None, :, None]), 1) ** 2
            )
        if not vectorize:
            ps1derr = ps1derr[0]
    if not vectorize:
        ps1d = ps1d[0]
    nmodes = np.sum(indx * (weights[:, None] > 0), 0)

    if error is True:
        return ps1d, ps1derr, k1deff, nmodes
    else:
        return ps1d, k1deff, nmodes


def bin_3d_to_cy(
    ps3d,
    kperp_i,
    kperpedges,
    weights=None,
    average=True,
    vectorize=False,
):
    """
    Function to bin a 3D distribution (e.g. power spectrum) into cylindrical average.
    The arrays are unravelled to 2D before binning by keeping the last axis.
    The 2D array is then binned along the first axis.

    The output is flipped so that the first axis is the original last axis.
    Therefore, to bin a 3D power spectrum to a cylindrical average,
    one can simply run ``bin_3d_to_cy`` twice
    (see ``PowerSpectrum.get_cy_power``).

    Parameters
    ----------
    ps3d: array.
        The 3D distribution to be binned.
    kperp_i: array.
        The perpendicular k-mode corresponding to the first axis.
    kperpedges: array.
        The bin edges for the perpendicular k-mode.
    weights: array, None.
        The weights for each 3D k-mode of the power spectrum.
    average: bool, default True.
        If ``True``, calculate the weighted average of the power spectrum
        in each bin. Else, calculate the weighted sum.
    vectorize: bool, default False
        Whether to vectorize the calculation, assuming the first axis is independent realisations.

    Returns
    -------
    pscy: np.ndarray
        The cylindrical average of the 3D distribution.
    """
    ps3d = np.array(ps3d)
    if not vectorize:
        ps3d = ps3d[None, ...]
    kperpedges = np.array(kperpedges)
    kperp_i = np.array(kperp_i).ravel()
    ps3d = ps3d.reshape((len(ps3d), len(kperp_i), -1))
    if weights is None:
        weights = np.ones_like(ps3d[0])
    weights = np.array(weights).reshape((len(kperp_i), -1))
    indx = (kperp_i[:, None] >= kperpedges[None, :-1]) * (
        kperp_i[:, None] < kperpedges[None, 1:]
    )
    weights = indx[:, None, :] * weights[:, :, None]
    pscy = np.sum(ps3d[:, :, :, None] * weights[None], 1)
    if average:
        pscy = pscy / np.sum(weights, 0)[None]
    if not vectorize:
        pscy = pscy[0]
    return pscy


# def get_independent_fourier_modes(box_dim):
#    r"""
#    Return a boolean array on whether the k-mode is independent.
#    For real-valued signal, a specific k-mode :math:`\vec{k}` and it's opposite
#    :math:`-\vec{k}` are conjugate to each other. This functions finds all the
#    pairs and only assign one of them with ``True``.
#
#    The indexing of the output array is consistent with the ``np.fft.fftfreq``
#    convention.
#
#    Parameters
#    ----------
#    box_dim: array.
#        The shape of the signal.
#
#    Returns
#    -------
#    unique: boolean array.
#        Whether the k-mode is indendent.
#
#    """
#    kvec = get_k_vector(box_dim, np.ones(len(box_dim)))
#    kvecmin = [(np.abs(kvec[i])[kvec[i] != 0]).min() for i in range(len(box_dim))]
#    kvec = [kvec[i] / kvecmin[i] for i in range(len(box_dim))]
#    kvecmax = [(np.abs(kvec[i])).max() for i in range(len(box_dim))]
#    kvecmax = np.max(kvecmax)
#    base = 2 * kvecmax + 1
#    kvec = [kvec[i] * (base**i) for i in range(len(box_dim))]
#    k_indx = np.sum(
#        (np.meshgrid(*([(vec) for vec in kvec]), indexing="ij")),
#        0,
#    )
#    _, indx = np.unique(np.abs(k_indx), return_index=True)
#    unique = np.zeros(np.prod(box_dim))
#    unique[indx] += 1
#    unique = unique.reshape(box_dim) > 0
#    return unique


def gaussian_beam_attenuation(k_perp, beam_sigma_in_mpc):
    """
    The beam attenuation term to be multiplied to model power
    spectrum assuming a Gaussian beam.

    Parameter
    ---------
    k_perp: np.ndarray.
        The transverse k-scale in Mpc^-1
    beam_sigma_in_mpc: float.
        The sigma of the Gaussian beam in Mpc.

    Returns
    -------
    beam_attenuation: np.ndarray.
        The beam attenuation factor.
    """
    return np.exp(-(k_perp**2) * beam_sigma_in_mpc**2 / 2)


def step_window_attenuation(k_dir, step_size_in_mpc, p=1):
    """
    The beam attenuation term to be multiplied to model power
    spectrum assuming a Gaussian beam.

    Parameter
    ---------
    k_perp: float.
        The transverse k-scale in Mpc^-1
    beam_sigma_in_mpc: float.
        The sigma of the Gaussian beam in Mpc.
    p: int, default 1
        The index of assignment scheme.

    Returns
    -------
    window_attenuation: np.ndarray.
        The window attenuation factor.
    """
    # note np.sinc is sin(pi x)/(pi x)
    return np.sinc(k_dir * step_size_in_mpc / np.pi / 2) ** p


class PowerSpectrum(FieldPowerSpectrum, ModelPowerSpectrum):
    """
    The class for coherently combining the :class:`FieldPowerSpectrum` and :class:`ModelPowerSpectrum` classes, and
    providing an interface for gridding the intensity mapping data as well as the galaxy catalogue to perform
    power spectrum estimation and for auto-correlation and cross-correlation.

    Note that, while you can manually set most of the attributes such as the box length, the density field, the weights, etc.,
    the class is mainly used for first gridding the intensity mapping data and the galaxy catalogue to a rectangular grid field,
    which then set these attributes automatically. The usual input parameters are those of :class:`meer21cm.dataanalysis.Specification`.

    For usage, check the tutorial notebooks in the examples and cookbook sections.

    Parameters
    ----------
    field_1: np.ndarray, default None
        The density field of the first tracer.
    box_len: list of 3 floats.
        The length of the box along each axis.
    weights_field_1: np.ndarray, default None
        The field-level weights of the first tracer. Default is uniform weights.
    weights_grid_1: np.ndarray, default None
        The grid-level weights of the first tracer. Default is uniform weights.
    mean_center_1: bool, default False
        Whether to mean-center the first tracer field.
    unitless_1: bool, default False
        Whether to divide the first tracer field by its mean.
    field_2: np.ndarray, default None
        The density field of the second tracer.
        If None, calculation of the second tracer and the cross-correlation will be skipped.
    weights_field_2: np.ndarray, default None
        The field-level weights of the second tracer. Default is uniform weights.
    weights_grid_2: np.ndarray, default None
        The grid-level weights of the second tracer. Default is uniform weights.
    mean_center_2: bool, default False
        Whether to mean-center the second tracer field.
    unitless_2: bool, default False
        Whether to divide the second tracer field by its mean.
    k1dbins: list of floats, default None
        The bin edges of k in Mpc-1 for the 1D power spectrum.
    kmode: float, default None
        The mode of 3D k in Mpc-1 for the model calculation.
    mumode: float, default None
        The mu mode of each 3D k-mode for the model calculation.
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
    fog_profile: str, default "lorentz"
        The shape of the finger-of-god profile to be used in the model calculation.
    cross_coeff: float, default 1.0
        The cross-correlation coefficient between the two tracers.
    model_k_from_field: bool, default True
        Whether to calculate the model k-mode ``self.kmode`` from the field k-mode ``self.k_mode``.
    mean_amp_1: float, default 1.0
        The mean amplitude of the first tracer.
    mean_amp_2: float, default 1.0
        The mean amplitude of the second tracer.
    sampling_resol: list, default None
        The sampling resolution of the field in Mpc.
        If ``sampling_resol`` is "auto", the sampling resolution will be set to the pixel size of the map.
    include_sky_sampling: list, default [True, False]
        Whether to include the sky sampling in the model calculation.
        If just a boolean is provided, it will be used for both tracers.
    downres_factor_transverse: float, default None
        The down-sampling factor for the transverse direction of the rectangular box for gridding.
    downres_factor_radial: float, default None
        The down-sampling factor for the radial direction of the rectangular box for gridding.
    init_box_from_map_data: bool, default False
        If True, the box dimensions will be calculated from the input data cube during initialization.
        You can always manually set the box dimensions later by ``self.get_enclosing_box()``.
    box_buffkick: float, default 5
        The buffer kick for the box on each side when gridding. In the unit of Mpc.
    compensate: list, default [False, False]
        Whether the gridded fields are compensated according to the mass assignment scheme.
        Note that the compensation is applied to the model power spectrum, and **not** to the gridded data fields.
    taper_func: function, default windows.blackmanharris
        The taper function to be applied to the gridded field.
        Note that the taper function is not automatically applied, and is only used when calling
        :meth:`PowerSpectrum.apply_taper_to_field`.
    kaiser_rsd: bool, default True
        Whether to include the RSD effect in the model calculation and mock simulation.
    grid_scheme: str, default "nnb"
        The grid scheme to be used for gridding.
        Can be "nnb", "cic", "tsc" or "pcs".
    interlace_shift: float, default 0.0
        The shift in the unit of grid cell size for interlacing.
    num_particle_per_pixel: int, default 1
        The number of random sampling particles for each sky map pixel.
    seed: int, default None
        The seed for the random number generator.
    kperpbins: list of floats, default None
        The bin edges of k_perp in Mpc-1 for the cylindrical average power spectrum.
    kparabins: list of floats, default None
        The bin edges of k_para in Mpc-1 for the cylindrical average power spectrum.
    flat_sky: bool, default False
        Whether to use the flat sky approximation.
    flat_sky_padding: list, default [0, 0, 0]
        The padding for the flat sky box.
    **params: dict
        Additional parameters to be passed to the base class :class:`meer21cm.cosmology.CosmologyCalculator`.
    """

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
        include_beam=[True, False],
        fog_profile="lorentz",
        cross_coeff=1.0,
        model_k_from_field=True,
        mean_amp_1=1.0,
        mean_amp_2=1.0,
        sampling_resol=None,
        include_sky_sampling=[True, False],
        downres_factor_transverse=1.2,
        downres_factor_radial=2.0,
        init_box_from_map_data=False,
        box_buffkick=5,
        compensate=[False, False],
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
        )
        self.kmode = kmode
        self.mumode = mumode
        if model_k_from_field:
            self.propagate_field_k_to_model()
        self.model_k_from_field = model_k_from_field
        ModelPowerSpectrum.__init__(
            self,
            kmode=self.kmode,
            mumode=self.mumode,
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
            **params,
        )
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

    @property
    def box_buffkick(self):
        """
        The buffer kick for the box on each side when gridding. In the unit of Mpc.
        """
        return self._box_buffkick

    @box_buffkick.setter
    def box_buffkick(self, value):
        if not isinstance(value, Iterable):
            self._box_buffkick = np.array([value, value, value])
        else:
            self._box_buffkick = np.array(value)
        init_attr = [
            "_box_origin",
            "_counts_in_box",
        ]
        logger.debug(f"cleaning cache of {init_attr} due to resetting box_buffkick")
        for attr in init_attr:
            setattr(self, attr, None)

    @property
    def num_particle_per_pixel(self):
        """
        The number of random sampling particles for each sky map pixel.
        """
        return self._num_particle_per_pixel

    @num_particle_per_pixel.setter
    def num_particle_per_pixel(self, value):
        self._num_particle_per_pixel = int(value)
        init_attr = [
            "_box_origin",
            "_counts_in_box",
        ]
        logger.debug(
            f"cleaning cache of {init_attr} due to resetting num_particle_per_pixel"
        )
        for attr in init_attr:
            setattr(self, attr, None)

    @property
    def interlace_shift(self):
        """
        The length in the unit of grid cell size for
        shifting the gridded field for interlacing.
        0 corresponds to no interlacing.
        """
        return self._interlace_shift

    @interlace_shift.setter
    def interlace_shift(self, value):
        self._interlace_shift = value

    @property
    def downres_factor_transverse(self):
        """
        The down-sampling factor for the transverse direction of the rectangular box for gridding.
        The box resolution is then multiplied by this factor from the resolution of the sky map pixel
        specified by ``pix_resol_in_mpc``.
        For example, if ``pix_resol_in_mpc`` is 0.1 Mpc, and ``downres_factor_transverse`` is 2.0,
        the box resolution will be 0.2 Mpc.
        """
        return self._downres_factor_transverse

    @downres_factor_transverse.setter
    def downres_factor_transverse(self, value):
        self._downres_factor_transverse = value
        # clean cache
        init_attr = [
            "_box_origin",
            "_counts_in_box",
        ]
        logger.debug(
            f"cleaning cache of {init_attr} due to resetting downres_factor_transverse"
        )
        for attr in init_attr:
            setattr(self, attr, None)

    @property
    def downres_factor_radial(self):
        """
        The down-sampling factor for the radial direction of the rectangular box for gridding.
        The box resolution is then multiplied by this factor from the resolution of the frequency channel
        specified by ``los_resol_in_mpc``.
        For example, if ``los_resol_in_mpc`` is 0.1 Mpc, and ``downres_factor_radial`` is 2.0,
        the box resolution will be 0.2 Mpc.
        """
        return self._downres_factor_radial

    @downres_factor_radial.setter
    def downres_factor_radial(self, value):
        self._downres_factor_radial = value
        # clean cache
        init_attr = [
            "_box_origin",
            "_counts_in_box",
        ]
        logger.debug(
            f"cleaning cache of {init_attr} due to resetting downres_factor_radial"
        )
        for attr in init_attr:
            setattr(self, attr, None)

    @property
    def counts_in_box(self):
        """
        The counts of the map cube voxels in the rectangular box.
        """
        if self._counts_in_box is None:
            self._counts_in_box = self.get_counts_in_box()
        return self._counts_in_box

    @property
    def flat_sky(self):
        """
        Whether to use flat sky approximation.
        If True, no proper projection and sky rotation is considered.
        Instead, the sky map cube is assumed to be a rectangular grid
        with equal voxel size specified by ``pix_resol_in_mpc`` and
        ``los_resol_in_mpc``.
        """
        return self._flat_sky

    @flat_sky.setter
    def flat_sky(self, value):
        self._flat_sky = bool(value)
        # clean cache
        init_attr = [
            "_box_origin",
            "_counts_in_box",
        ]
        logger.debug(f"cleaning cache of {init_attr} due to resetting flat_sky")
        for attr in init_attr:
            setattr(self, attr, None)

    @property
    def flat_sky_padding(self):
        """
        Pad the rectangular box in the flat sky approximation.

        The input should be a list of 3 integers, corresponding to number of padding cells along
        each dimension in both directions.
        For example, [1,1,1] will pad 2x2x2 cells.
        """
        return self._flat_sky_padding

    @flat_sky_padding.setter
    def flat_sky_padding(self, value):
        self._flat_sky_padding = value
        # clean cache
        init_attr = [
            "_box_origin",
            "_counts_in_box",
        ]
        logger.debug(f"cleaning cache of {init_attr} due to resetting flat_sky_padding")
        for attr in init_attr:
            setattr(self, attr, None)

    def propagate_field_k_to_model(self):
        """
        Use field k-modes for the model
        """
        # use field kmode to propagate into model
        kmode = self.k_mode
        mumode = self.k_para
        slice_indx = (None,) * (len(self.box_len.shape) - 1)
        slice_indx += (slice(None, None, None),)
        with np.errstate(divide="ignore", invalid="ignore"):
            mumode = np.nan_to_num(self.k_para[slice_indx] / kmode)
        self.kmode = kmode
        self.mumode = mumode

    def get_1d_power(
        self,
        power3d,
        k1dbins=None,
        k1dweights=None,
        k_xyz_min=None,
        k_xyz_max=None,
        k_perppara_min=None,
        k_perppara_max=None,
    ):
        """
        Bin the 3D power spectrum into 1D power spectrum.
        If the input ``power3d`` is a string, it is assumed to be an attribute of the class,
        for example ``auto_power_3d_1``.
        Also see :meth:`meer21cm.power.bin_3d_to_1d` for more details.

        Parameters
        ----------
        power3d: np.ndarray or str
            The 3D power spectrum.
        k1dbins: np.ndarray, default None
            The bins for the 1D power spectrum. Default is the same as the class attribute.
        k1dweights: np.ndarray, default None
            The weights for the 3D power spectrum. Default is equal weights for every k-mode.
        k_xyz_min: list of size 3, default None
            The minimum k-mode for the 1D power spectrum in x, y, z directions.
        k_xyz_max: list of size 3, default None
            The maximum k-mode for the 1D power spectrum in x, y, z directions.
        k_perppara_min: list of size 2, default None
            The minimum k_perp and k_para for the 1D power spectrum.
        k_perppara_max: list of size 2, default None
            The maximum k_perp and k_para for the 1D power spectrum.

        Returns
        -------
        power1d: np.ndarray
            The 1D power spectrum.
        k1deff: np.ndarray
            The effective k-mode for each bin.
        nmodes: np.ndarray
            The number of modes in each bin.
        """
        if k1dbins is None:
            k1dbins = self.k1dbins
        if k1dweights is None:
            k1dweights = self.k1dweights
        # if still None, use equal weights
        if k1dweights is None:
            k1dweights = np.ones_like(self.k_mode)
        if isinstance(power3d, str):
            power3d = getattr(self, power3d)
        # filter k-modes
        slicer = get_nd_slicer()
        k_3d_sel_min = 1.0
        if k_xyz_min is not None:
            k_3d_sel_min = [
                ((np.abs(self.k_vec[i]) >= k_xyz_min[i]))[slicer[i]]
                for i in range(len(self.k_vec))
            ]
            k_3d_sel_min = k_3d_sel_min[0] * k_3d_sel_min[1] * k_3d_sel_min[2]
        k_3d_sel_max = 1.0
        if k_xyz_max is not None:
            k_3d_sel_max = [
                ((np.abs(self.k_vec[i]) <= k_xyz_max[i]))[slicer[i]]
                for i in range(len(self.k_vec))
            ]
            k_3d_sel_max = k_3d_sel_max[0] * k_3d_sel_max[1] * k_3d_sel_max[2]
        k_cy_sel_min = 1.0
        if k_perppara_min is not None:
            k_cy_sel_min = ((np.abs(self.k_perp) >= k_perppara_min[0]))[:, :, None] * (
                (np.abs(self.k_para) >= k_perppara_min[1])
            )[None, None, :]
        k_cy_sel_max = 1.0
        if k_perppara_max is not None:
            k_cy_sel_max = ((np.abs(self.k_perp) <= k_perppara_max[0]))[:, :, None] * (
                (np.abs(self.k_para) <= k_perppara_max[1])
            )[None, None, :]
        k1dweights = (
            k1dweights * k_3d_sel_min * k_3d_sel_max * k_cy_sel_min * k_cy_sel_max
        )
        k1dweights[0, 0, 0] = 0.0
        power1d, k1deff, nmodes = bin_3d_to_1d(
            power3d,
            self.k_mode,
            k1dbins,
            weights=k1dweights,
        )
        return power1d, k1deff, nmodes

    def get_cy_power(
        self,
        power3d,
        kperpbins=None,
        kparabins=None,
        kcyweights=None,
    ):
        """
        Bin the 3D power spectrum into cylindrical k_perp-k_para power spectrum.
        If the input ``power3d`` is a string, it is assumed to be an attribute of the class,
        for example ``auto_power_3d_1``.
        Also see :meth:`meer21cm.power.bin_3d_to_cy` for more details.

        Parameters
        ----------
        power3d: np.ndarray or str
            The 3D power spectrum.
        kperpbins: np.ndarray, default None
            The k_perp bins for the cylindrical ps. Default is the same as the class attribute.
        kparabins: np.ndarray, default None
            The k_para bins for the cylindrical ps. Default is the same as the class attribute.
        kcyweights: np.ndarray, default None
            The weights for the 3D power spectrum. Default is equal weights for every k-mode.

        Returns
        -------
        powercy: np.ndarray
            The cylindrical power spectrum.
        weightscy: np.ndarray
            The weights for the cylindrical k-modes.
        """
        if kperpbins is None:
            kperpbins = self.kperpbins
        if kparabins is None:
            kparabins = self.kparabins
        if kcyweights is None:
            kcyweights = np.ones_like(self.k_mode)
        if isinstance(power3d, str):
            power3d = getattr(self, power3d)
        kcyweights[0, 0, 0] = 0.0
        powercy = bin_3d_to_cy(
            power3d,
            self.k_perp,
            kperpbins,
            weights=kcyweights,
        )
        weightscy = bin_3d_to_cy(
            kcyweights,
            self.k_perp,
            kperpbins,
            weights=kcyweights,
            average=False,
        )
        powercy = bin_3d_to_cy(
            powercy,
            np.abs(self.k_para),
            kparabins,
            weights=weightscy,
        )
        weightscy = bin_3d_to_cy(
            weightscy,
            np.abs(self.k_para),
            kparabins,
            weights=weightscy,
            average=False,
        )
        return powercy, weightscy

    # calculate on-the-fly, no cache
    def map_sampling(self, sampling_resol=None, p=1):
        """
        The sampling window function from the map cube to be convolved with model power spectrum.
        This should correspond to the resolution of map-making on the sky and the frequency channel,
        **not** the resolution of the gridded field.

        Parameters
        ----------
        sampling_resol: list, default None
            The sampling resolution of the field in Mpc.
            Default is the class attribute ``sampling_resol``.
        p: int, default 1
            The index of assignment scheme.

        Returns
        -------
        B_sampling: np.ndarray.
            The sampling window function in 3D k-space.
        """
        if not self.has_resol:
            return 1.0
        k_x = self.k_vec[0][:, None, None]
        k_y = self.k_vec[1][None, :, None]
        k_para = self.k_mode * self.mumode
        if sampling_resol is None:
            sampling_resol = self.sampling_resol
        B_sampling = np.nan_to_num(
            step_window_attenuation(k_x, sampling_resol[0], p)
            * step_window_attenuation(k_y, sampling_resol[1], p)
            * step_window_attenuation(k_para, sampling_resol[2], p)
        )
        return B_sampling

    def gridding_compensation(self):
        """
        The sampling window function to be compensated for the gridding mass assignment scheme.
        """
        return fourier_window_for_assignment(self.box_ndim, self.grid_scheme)

    @property
    def box_origin(self):
        """
        The coordinate of the origin of the box in Mpc.
        See :func:`meer21cm.grid.minimum_enclosing_box_of_lightcone`
        for definition.
        """
        return self._box_origin

    @box_origin.setter
    def box_origin(self, value):
        self._box_origin = np.array(value)

    @property
    def rot_mat_sky_to_box(self):
        """
        The rotational matrix from spheircal cooridnate to regular box.

        See :func:`meer21cm.grid.minimum_enclosing_box_of_lightcone`
        for definition.
        """
        return self._rot_mat_sky_to_box

    @property
    def pix_coor_in_cartesian(self):
        """
        The cartesian coordinate of the pixels in Mpc.
        """
        return self._pix_coor_in_cartesian

    @property
    def pix_coor_in_box(self):
        """
        The cartesian coordinate of the pixels in Mpc,
        shifted so that the origin is the origin of the enclosing box.
        """
        return self.pix_coor_in_cartesian - self.box_origin[None, :]

    def use_flat_sky_box(self, flat_sky_padding=None):
        """
        Use flat sky approximation to calculate the box dimensions.

        Parameters
        ----------
        flat_sky_padding: list, default None
            The padding for the flat sky box.
            If None, use the class attribute ``flat_sky_padding``.
        """
        if flat_sky_padding is None:
            flat_sky_padding = self.flat_sky_padding
        logger.debug(f"using flat sky box with padding {flat_sky_padding}")
        logger.info(
            f"{inspect.currentframe().f_code.co_name}: setting self.box_ndim, self.box_len, self.box_origin"
        )
        self.box_ndim = np.array(self.data.shape) + 2 * np.array(flat_sky_padding)
        self.box_len = np.array(self.box_ndim) * np.array(
            [
                self.pix_resol_in_mpc,
                self.pix_resol_in_mpc,
                self.los_resol_in_mpc,
            ]
        )
        # flat sky does not have rotation so there is no box_origin
        self.box_origin = np.array([0, 0, 0])
        if self.model_k_from_field:
            logger.info(
                f"{inspect.currentframe().f_code.co_name}: "
                "setting the model self.kmode and self.mumode to correspond to the field k-modes"
            )
            self.propagate_field_k_to_model()
        self._counts_in_box = None
        nu_ext = np.linspace(
            self.nu.min() - self.freq_resol * flat_sky_padding[2],
            self.nu.max() + self.freq_resol * flat_sky_padding[2],
            len(self.nu) + 2 * flat_sky_padding[2],
        )
        self._box_voxel_redshift = (
            np.ones(self.box_ndim) * freq_to_redshift(nu_ext)[None, None, :]
        )

    def get_enclosing_box(self, rot_mat=None):
        """
        invoke to calculate the box dimensions for enclosing all
        the map pixels.

        Parameters
        ----------
        rot_mat: np.ndarray, default None
            The rotational matrix from the sky to the box.
            If None, calculates the suitable rotation matrix automatically.
        """
        if self.flat_sky:
            self.use_flat_sky_box()
            if self.model_k_from_field:
                logger.info(
                    f"{inspect.currentframe().f_code.co_name}: "
                    "setting the model self.kmode and self.mumode to correspond to the field k-modes"
                )
                self.propagate_field_k_to_model()
            return 1
        ra = self.ra_map.copy()[self.W_HI.sum(-1) > 0]
        dec = self.dec_map.copy()[self.W_HI.sum(-1) > 0]
        logger.debug(f"calculating enclosing box for {len(ra)} particles")
        (
            _x_start,
            _y_start,
            _z_start,
            _x_len,
            _y_len,
            _z_len,
            rot_back,
            pos_arr,
        ) = minimum_enclosing_box_of_lightcone(
            ra,
            dec,
            self.nu,
            cosmo=self.cosmo,
            return_coord=True,
            buffkick=self.box_buffkick,
            rot_mat=rot_mat,
        )
        logger.debug(
            f"{inspect.currentframe().f_code.co_name}: calculated enclosing box with size {_x_len} x {_y_len} x {_z_len}"
        )
        logger.info(
            f"{inspect.currentframe().f_code.co_name}: setting self.box_len, self.box_origin, self.box_ndim"
        )
        self._box_origin = np.array([_x_start, _y_start, _z_start])
        self._box_len = np.array(
            [
                _x_len,
                _y_len,
                _z_len,
            ]
        )
        self._rot_mat_sky_to_box = np.linalg.inv(rot_back)
        # random sample
        num_p = self.num_particle_per_pixel
        ra_sample = [
            ra,
        ] * num_p
        dec_sample = [
            dec,
        ] * num_p
        nu_sample = [
            self.nu,
        ] * num_p
        ra_sample = np.array(ra_sample)
        dec_sample = np.array(dec_sample)
        nu_sample = np.array(nu_sample)
        logger.debug(f"randomly sampled {num_p} particles in each pixel")
        rng = np.random.default_rng(seed=self.seed)
        rand_angle = rng.uniform(
            -self.pix_resol / 2, self.pix_resol / 2, size=(2,) + ra_sample[1:].shape
        )
        rand_nu = rng.uniform(
            -self.freq_resol / 2, self.freq_resol / 2, size=(1,) + nu_sample[1:].shape
        )
        ra_sample[1:] += rand_angle[0]
        dec_sample[1:] += rand_angle[1]
        nu_sample[1:] += rand_nu[0]
        pos_arr = [
            pos_arr,
        ]
        for i in range(1, num_p):
            (_, _, _, _, _, _, _, pos_arr_i) = minimum_enclosing_box_of_lightcone(
                ra_sample[i],
                dec_sample[i],
                nu_sample[i],
                cosmo=self.cosmo,
                return_coord=True,
                buffkick=self.box_buffkick,
                rot_mat=self.rot_mat_sky_to_box,
            )
            pos_arr.append(pos_arr_i)
        pos_arr = np.array(pos_arr)
        pos_arr = pos_arr.reshape((-1, 3))

        self._pix_coor_in_cartesian = pos_arr
        downres = np.array(
            [
                self.downres_factor_transverse,
                self.downres_factor_transverse,
                self.downres_factor_radial,
            ]
        )
        pix_resol_in_mpc = self.pix_resol_in_mpc
        los_resol_in_mpc = self.los_resol_in_mpc
        box_resol = (
            np.array([pix_resol_in_mpc, pix_resol_in_mpc, los_resol_in_mpc]) * downres
        )
        ndim_rg = self.box_len / box_resol
        ndim_rg = ndim_rg.astype("int")
        for i in range(3):
            if ndim_rg[i] % 2 == 0:
                ndim_rg[i] += 1
        box_resol = self.box_len / ndim_rg
        self.box_ndim = ndim_rg
        logger.debug(
            f"calculated box resolution due to downres factor: {box_resol}, {downres}"
        )
        self._counts_in_box = None
        slicer = get_nd_slicer()
        vec = [(self.x_vec[i] + self.box_origin[i])[slicer[i]] for i in range(3)]
        vec_len = np.sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)
        self._box_voxel_redshift = self.z_as_func_of_comov_dist(vec_len)
        if self.model_k_from_field:
            logger.info(
                f"{inspect.currentframe().f_code.co_name}: "
                "setting the model self.kmode and self.mumode to correspond to the field k-modes"
            )
            self.propagate_field_k_to_model()

    @property
    def average_model_hi_temp(self):
        """
        Calculate the average HI brightness temperature in the map cube, taking care of redshift evolution and map sampling.
        """
        t_bar = omega_hi_to_average_temp(self.omega_hi, z=self.z_ch, cosmo=self.cosmo)
        t_bar = (t_bar * self.w_HI.sum((0, 1))).sum() / self.w_HI.sum()
        return t_bar

    @property
    def model_hi_temp_in_box(self):
        """
        Based on the redshift dependence of Omega_HI, calculate
        the average HI brightness temperature for each grid in the rectangular box.
        This can be used in a way that the weighted average of the ``model_hi_temp_in_box``
        is used as the average t_bar in the model power spectrum (by passing it to ``mean_amp_1``),
        whereas the 3D ``model_hi_temp_in_box`` is used as the field weight to account for the effect of
        the redshift evolution of Omega_HI in the power spectrum.
        """
        z_grid = self._box_voxel_redshift
        omega_hi_grid = self.omega_hi_z_func(z_grid)
        t_bar_grid = omega_hi_to_average_temp(omega_hi_grid, z=z_grid, cosmo=self.cosmo)
        return t_bar_grid

    def get_counts_in_box(self):
        """
        Grid the counts of the map cube voxels into the rectangular box, and return the
        effective counts per rectangular grid.
        """
        if self.flat_sky:
            counts_in_grids = self.w_HI
        else:
            pix_coor_orig = self.pix_coor_in_box.reshape(
                (self.num_particle_per_pixel, -1)
            )[0].reshape((-1, 3))
            counts_in_grids, _, _ = project_particle_to_regular_grid(
                pix_coor_orig,
                self.box_len,
                self.box_ndim,
                grid_scheme=self.grid_scheme,
                particle_mass=self.w_HI[self.W_HI.sum(-1) > 0].ravel(),
                compensate=False,  # compensate should be at model level
                average=False,
            )
        return counts_in_grids

    def grid_data_to_field(self, flat_sky=None):
        """
        Grid the stored data map to a rectangular grid field.

        If flat_sky is True, no gridding is performed. Instead, the map cube
        dimensions are taken to be a rectangular grid, with the grid length
        corresponding to the pixel resolution on x/y and los frequency resolution
        as z.

        If flat_sky is False, the data is gridded onto a regular grid using the
        input grid scheme and performing the proper curved sky projection.

        The gridded field is stored as field_1 and the weights are stored as weights_1.
        """
        if flat_sky is None:
            flat_sky = self.flat_sky
        if flat_sky:
            self.field_1 = self.data
            self.weights_1 = self.w_HI.astype(float)
            self.use_flat_sky_box(flat_sky_padding=[0, 0, 0])
            self.mean_center_1 = False
            self.unitless_1 = False
            self.include_sky_sampling = [True, False]
            self.compensate = False
            self.include_beam = [True, False]
            return self.field_1, self.weights_1, (self.weights_1 > 0).astype(float)
        if self.box_origin is None:
            self.get_enclosing_box()
        data_particle = self.data[self.W_HI.sum(-1) > 0].ravel()
        weights_particle = self.w_HI[self.W_HI.sum(-1) > 0].ravel()
        num_p = self.num_particle_per_pixel
        data_particle = [
            data_particle,
        ] * num_p
        weights_particle = [
            weights_particle,
        ] * num_p
        data_particle = np.array(data_particle).ravel()
        weights_particle = np.array(weights_particle).ravel()
        hi_map_rg, hi_weights_rg, pixel_counts_hi_rg = project_particle_to_regular_grid(
            self.pix_coor_in_box,
            self.box_len,
            self.box_ndim,
            grid_scheme=self.grid_scheme,
            particle_mass=data_particle,
            particle_weights=weights_particle,
            compensate=False,  # compensate should be at model level
        )
        hi_map_rg2, _, _ = project_particle_to_regular_grid(
            self.pix_coor_in_box,
            self.box_len,
            self.box_ndim,
            grid_scheme=self.grid_scheme,
            particle_mass=data_particle,
            particle_weights=weights_particle,
            compensate=False,  # compensate should be at model level
            shift=self.interlace_shift,
        )
        hi_map_rg = interlace_two_fields(
            hi_map_rg,
            hi_map_rg2,
            self.interlace_shift,
        )
        hi_map_rg = np.array(hi_map_rg)
        hi_weights_rg = np.array(hi_weights_rg)
        # pixel_counts_hi_rg = np.array(pixel_counts_hi_rg)
        # self.pixel_counts_hi_rg = pixel_counts_hi_rg
        self.field_1 = hi_map_rg
        self.weights_1 = (self.counts_in_box).astype(float)
        # self.apply_taper_to_field(1)
        self.unitless_1 = False
        include_beam = np.array(self.include_beam)
        include_beam[0] = True
        self.include_beam = include_beam
        include_sky_sampling = np.array(self.include_sky_sampling)
        include_sky_sampling[0] = True
        self.include_sky_sampling = include_sky_sampling
        return hi_map_rg, hi_weights_rg, pixel_counts_hi_rg

    def grid_gal_to_field(self, radecfreq=None, flat_sky=None):
        """
        Grid the galaxy catalogue to a rectangular grid field.

        If flat_sky is True, no gridding is performed. Instead, the map cube
        dimensions are taken to be a rectangular grid, with the grid length
        corresponding to the pixel resolution on x/y and los frequency resolution
        as z.

        """
        if self.box_origin is None:
            self.get_enclosing_box()
        if flat_sky is None:
            flat_sky = self.flat_sky
        if radecfreq is None:
            ra_gal = self.ra_gal
            dec_gal = self.dec_gal
            freq_gal = self.freq_gal
        else:
            ra_gal, dec_gal, freq_gal = radecfreq
        if flat_sky:
            self.compensate = False
            z_gal = freq_to_redshift(freq_gal)
            self.use_flat_sky_box(flat_sky_padding=[0, 0, 0])
            pos_indx_1, pos_indx_2 = radec_to_indx(
                ra_gal, dec_gal, self.wproj, to_int=False
            )
            gal_pos_in_box = np.zeros((ra_gal.size, 3))
            gal_pos_in_box[:, 0] = pos_indx_1 / self.num_pix_x * self.box_len[0]
            gal_pos_in_box[:, 1] = pos_indx_2 / self.num_pix_y * self.box_len[1]
            gal_pos_in_box[:, 2] = (
                self.comoving_distance(z_gal).value
                - self.comoving_distance(self.z_ch.min()).value
            )
        else:
            (_, _, _, _, _, _, _, gal_pos_arr) = minimum_enclosing_box_of_lightcone(
                ra_gal,
                dec_gal,
                freq_gal,
                cosmo=self.cosmo,
                return_coord=True,
                tile=False,
                rot_mat=self.rot_mat_sky_to_box,
            )
            gal_pos_in_box = gal_pos_arr - self.box_origin[None, :]
        (
            gal_map_rg,
            gal_weights_rg,
            pixel_counts_gal_rg,
        ) = project_particle_to_regular_grid(
            gal_pos_in_box,
            self.box_len,
            self.box_ndim,
            grid_scheme=self.grid_scheme,
            compensate=False,  # compensate should be at model level
            average=False,
        )
        (gal_map_rg2, _, _,) = project_particle_to_regular_grid(
            gal_pos_in_box,
            self.box_len,
            self.box_ndim,
            grid_scheme=self.grid_scheme,
            compensate=False,  # compensate should be at model level
            average=False,
            shift=self.interlace_shift,
        )
        gal_map_rg = interlace_two_fields(
            gal_map_rg,
            gal_map_rg2,
            self.interlace_shift,
        )
        gal_map_rg = np.array(gal_map_rg)
        gal_weights_rg = np.array(gal_weights_rg)
        pixel_counts_gal_rg = np.array(pixel_counts_gal_rg)
        self.field_2 = gal_map_rg
        # only pixels sampled by the lightcone is used
        weights_g = (self.counts_in_box > 0).astype(float)
        self.weights_2 = weights_g
        self.mean_center_2 = True
        self.unitless_2 = True
        include_beam = np.array(self.include_beam)
        include_beam[1] = False
        self.include_beam = include_beam
        include_sky_sampling = np.array(self.include_sky_sampling)
        include_sky_sampling[1] = False
        self.include_sky_sampling = include_sky_sampling

        return gal_map_rg, gal_weights_rg, pixel_counts_gal_rg

    def get_n_bar_correction(self):
        """
        Calculate the number density correction for the galaxy catalogue.
        """
        n_bar = self.ra_gal.size / self.survey_volume
        n_bar2 = (
            (self.field_2 * self.weights_2).sum()
            / self.weights_2.sum()
            / np.prod(self.box_resol)
        )
        return n_bar2 / n_bar

    def ra_dec_z_for_coord_in_box(self, pos_in_box):
        """
        Convert the coordinates in the box to ra, dec, z,
        and also return the comoving distance to the observer for each point.

        Parameters
        ----------
        pos_in_box: array.
            The coordinates in the box.

        Returns
        -------
        pos_ra: array.
            The ra of the points.
        pos_dec: array.
            The dec of the points.
        pos_z: array.
            The redshift of the points.
        pos_comov_dist: array.
            The comoving distance to the observer for each point.
        """
        pos_arr = pos_in_box + self.box_origin
        rot_back = np.linalg.inv(self.rot_mat_sky_to_box)
        pos_arr = np.einsum("ij,aj->ai", rot_back, pos_arr)
        pos_comov_dist = np.sqrt(np.sum(pos_arr**2, axis=-1))
        pos_z = self.z_as_func_of_comov_dist(pos_comov_dist)
        pos_ra, pos_dec = hp.vec2ang(pos_arr / pos_comov_dist[:, None], lonlat=True)
        return pos_ra, pos_dec, pos_z, pos_comov_dist

    def grid_field_to_sky_map(
        self,
        field,
        average=True,
        mask=True,
        wproj=None,
        num_pix_x=None,
        num_pix_y=None,
    ):
        """
        Grid a field in the rectangular box onto the sky.

        Parameters
        ----------
        field: array.
            The field in the box to be gridded.

        average: bool, default True.
            Whether the field grids are averaged or summed into sky pixels.

        mask: bool, default True.
            If True, the sky map is then masked by the survey selection function.

        wproj: :class:`astropy.wcs.WCS` object, default None.
            The wcs object for the output sky map. Default uses the stored ``self.wproj``.

        num_pix_x: int, default None.
            The number of pixels along the first axis for the sky map. Defulat uses the stored ``self.num_pix_x``.

        num_pix_y: int, default None.
            The number of pixels along the seconds axis for the sky map. Defulat uses the stored ``self.num_pix_y``.

        Returns
        -------
        map_bin: array.
            The output sky map.
        count_bin: array.
            The number of grids in each sky map pixel.

        """
        if wproj is None:
            wproj = self.wproj
        if num_pix_x is None:
            num_pix_x = self.num_pix_x
        if num_pix_y is None:
            num_pix_y = self.num_pix_y
        pos_xyz = np.array(np.meshgrid(*self.x_vec, indexing="ij")).reshape((3, -1)).T
        pos_ra, pos_dec, pos_z, _ = self.ra_dec_z_for_coord_in_box(pos_xyz)
        pos_indx_1, pos_indx_2 = radec_to_indx(pos_ra, pos_dec, wproj, to_int=False)
        pos_indx_z = redshift_to_freq(pos_z) - self.nu.min()
        pos_indx_array = np.array([pos_indx_1, pos_indx_2, pos_indx_z]).T
        map_bin, _, count_bin = project_particle_to_regular_grid(
            pos_indx_array,
            np.array([num_pix_x, num_pix_y, self.nu.max() - self.nu.min()]),
            np.array([num_pix_x, num_pix_y, self.nu.size]),
            particle_mass=field.ravel(),
            average=average,
            compensate=False,
            grid_scheme="nnb",
        )
        if mask:
            map_bin *= self.W_HI
        return map_bin, count_bin

    def gen_random_poisson_galaxy(self, sel=None, num_g_rand=None, seed=None):
        """
        Generate a random galaxy catalogue from the map cube following the Poisson distribution.

        Note that, by default, the random seed is fixed to the class attribute ``self.seed``.
        If you want to generate multiple random catalogues, you need to set a different seed manually for each catalogue.

        Parameters
        ----------
        sel: array, default None
            The selection function of the galaxy catalogue.
            If None, use the class attribute ``self.W_HI``.
        num_g_rand: int, default None
            The number of galaxies to generate. Default uses the number of galaxies stored in the data in `self.ra_gal`.
        seed: int, default None
            The seed for the random number generator. Default uses the class attribute ``self.seed``.

        Returns
        -------
        ra_rand: np.ndarray.
            The ra of the random galaxies.
        dec_rand: np.ndarray.
            The dec of the random galaxies.
        freq_rand: np.ndarray.
            The ``frequency`` of the random galaxies. The redshift of the random galaxies can
            be calculated by ``meer21cm.util.redshift_to_freq(z_rand)``.
        """
        if sel is None:
            sel = self.W_HI[:, :, 0]
        if num_g_rand is None:
            num_g_rand = self.ra_gal.size
        rng = np.random.default_rng(seed=seed)
        ra_rand = self.ra_map[sel]
        dec_rand = self.dec_map[sel]
        ra_rand = rng.choice(ra_rand, size=num_g_rand, replace=True)
        dec_rand = rng.choice(dec_rand, size=num_g_rand, replace=True)
        rand_disp = rng.uniform(
            -self.pix_resol / 2, self.pix_resol / 2, size=num_g_rand * 2
        )
        ra_rand += rand_disp[:num_g_rand]
        dec_rand += rand_disp[num_g_rand:]
        # in future this should be a dNdz
        cov_dist_limit = [
            self.comoving_distance(self.z_ch.min()).to("Mpc").value,
            self.comoving_distance(self.z_ch.max()).to("Mpc").value,
        ]
        cov_dist_rand = rng.uniform(
            cov_dist_limit[0], cov_dist_limit[1], size=num_g_rand
        )
        z_rand = self.z_as_func_of_comov_dist(cov_dist_rand)
        return ra_rand, dec_rand, redshift_to_freq(z_rand)

    def apply_taper_to_field(
        self,
        field,
        taper_func=None,
        axis=[
            2,
        ],
    ):
        """
        Apply a taper to the field, by multiplying the taper function to the
        corresponding weights of the field.

        Parameters
        ----------
        field: int.
            The index of the field to be tapered, either 1 or 2.
        taper_func: function, default None.
            The taper function. Default uses the stored ``self.taper_func``.
        axis: list, default [2,].
            The axis to apply the taper to. Default is the z-axis which is approximately the los.
        """
        if taper_func is None:
            taper_func = self.taper_func
        taper_i = [taper_func(self.box_ndim[i]) for i in range(3)]
        taper = 1
        for i in axis:
            slice_list_i = [None, None, None]
            slice_list_i[i] = slice(None, None, None)
            slice_list_i = tuple(slice_list_i)
            taper = taper * taper_i[i][slice_list_i]
        setattr(self, f"weights_{field}", getattr(self, f"weights_{field}") * taper)
