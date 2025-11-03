"""
This module contains the class for storing the cosmological model used for calculation.

The class :py:class:`CosmologyParameters` is the class for storing the cosmological parameters, and settings for computing matter power spectrum.

The class :py:class:`CosmologyCalculator` is the base class for storing the cosmological model used for calculation.
It is typically used as a base class for other classes that inherit from it, and not used directly.

Note that, there are always two sets of cosmological parameters defined in the class:
- the **fiducial** cosmology, which is the cosmology that is used to transform sky coordinates to comoving coordinates.
- the **true** cosmology, which is the cosmology that is used to compute the model power spectra.

The Alcock–Paczynski effect is then always included in the model power spectrum calculation as well as converting
the field k-modes to model k-modes (see :py:class:`PowerSpectrum`).

"""
import numpy as np
import camb
import astropy
from meer21cm import Specification
from scipy.interpolate import interp1d
from meer21cm.util import (
    omega_hi_to_average_temp,
    tagging,
    HiddenPrints,
    center_to_edges,
    freq_to_redshift,
)
from astropy.cosmology import Planck18, w0waCDM
from copy import deepcopy
import inspect
import logging
from typing import Callable
import warnings

logger = logging.getLogger(__name__)

As_set = {
    "Planck18": np.exp(3.047) / 1e10,
    "Planck15": np.exp(3.064) / 1e10,
    "Planck13": np.exp(3.091) / 1e10,
    "WMAP9": 2.464 / 1e9,
    "WMAP7": 2.42 / 1e9,
    "WMAP5": 2.41 / 1e9,
    "WMAP3": 2.105 / 1e9,
    "WMAP1": 1.893 / 1e9,
}
get_ns_from_astropy = lambda x: getattr(astropy.cosmology, x).meta["n"]


def get_cosmo_dict(cosmo: str or astropy.cosmology.Cosmology):
    """
    Get the cosmology dictionary from the input cosmology.
    """
    if isinstance(cosmo, str):
        cosmo = getattr(astropy.cosmology, cosmo)
    return dict(
        tau=0.0561,
        Neff=3.046,
        omega_cold=cosmo.Om0,
        As=As_set[cosmo.name],
        omega_baryon=cosmo.Ob0,
        ns=cosmo.meta["n"],
        h=cosmo.h,
        neutrino_mass=cosmo.m_nu.sum().value,
        w0=cosmo.w0 if "w0" in cosmo.__dir__() else -1.0,
        wa=cosmo.wa if "wa" in cosmo.__dir__() else 0.0,
    )


fiducial_dict = get_cosmo_dict("Planck18")


class CosmologyParameters:
    r"""
    The class for storing cosmological parameters, and settings for computing matter power
    spectrum. The naming of the input arguments for
    cosmological parameters follow
    `baccoemu <https://baccoemu.readthedocs.io/en/latest/>`_ .
    It either uses `camb` or `baccoemu` to compute the matter power spectrum.

    Note that everything is **not in h unit** unless explicitly specified in name
    (of course except sigma_8 which follows the definition of 8 Mpc/h).

    Further note that, baccoemu is trained on `CLASS <https://github.com/lesgourg/class_public>`_ .
    Therefore, in the usual range of parameters in the LCDM,
    you should see the <1% difference between these two
    backends as differences between the Boltzmann solver codes (although this
    is not well tested on our end). Use it with precaution if you want to do
    precision cosmology type of forecasts and sims.

    Parameters
    ----------
    ps_type: str, default "linear"
        The type of the matter power spectrum.
    kmin: float, default 1e-3
        The minimum k in Mpc^-1 for calculating matter power. k below kmin will be extrapolated.
    kmax: float, default 3.0
        The maximum k in Mpc^-1 for calculating matter power. k above kmax will be extrapolated.
    omega_cold: float, default :py:data:`astropy.cosmology.Planck18.Om0`
        The density fraction of CDM+Baryon at z=0.
    As: float, default :py:data:`astropy.cosmology.Planck18.As`
        The amplitude of the initial power spectrum.
    omega_baryon: float, default :py:data:`astropy.cosmology.Planck18.Ob0`
        The density fraction of baryons at z=0.
    ns: float, default :py:data:`astropy.cosmology.Planck18.meta["n"]`
        The spectral index of the initial power spectrum.
    h: float, default :py:data:`astropy.cosmology.Planck18.h`
        The Hubble parameter over 100km/s/Mpc.
    neutrino_mass: float, default :py:data:`astropy.cosmology.Planck18.m_nu.sum().value`
        The sum of the neutrino mass in eV.
    w0: float, default -1.0
        The dark energy equation of state at a=1 (z=0).
    wa: float, default 0.0
        The redshift-dependent part of the dark energy equation of state.
        :math:`w(a) = w_0 + w_a (1 - a)`.
    expfactor: float, default 1.0
        The expansion factor which is calculated as :math:`a = 1 / (1 + z)`.
    cold: bool, default True
        Whether to use the cold matter power spectrum.
        If True, the matter power spectrum refers to CDM+Baryon.
        If False, the matter power spectrum refers to all matter including massive neutrinos.
    num_kpoints: int, default 200
        The number of k points to compute the interpolation of the matter power spectrum.
    omega_de: float, default None
        The density fraction of dark energy at z=0. If None, it will be calculated using camb from
        the rest of the input parameters.
    tau: float, default 0.0561
        The optical depth of the reionization.
        Note that it does not affect the matter and tracer power spectrum calculation.
    Neff: float, default 3.046
        The effective number of neutrino species.
        Note that it does not affect the matter and tracer power spectrum calculation.
    """

    def __init__(
        self,
        ps_type="linear",
        kmin=1e-3,
        kmax=3.0,
        num_kpoints=200,
        expfactor=1.0,
        cold=True,
        omega_de=None,
        tau=0.0561,
        Neff=3.046,
        omega_cold=Planck18.Om0,
        As=np.exp(3.047) / 1e10,
        omega_baryon=Planck18.Ob0,
        ns=Planck18.meta["n"],
        h=Planck18.h,
        neutrino_mass=Planck18.m_nu.sum().value,
        w0=-1.0,
        wa=0.0,
    ):
        self.ps_type = ps_type
        self.kmin = kmin
        self.kmax = kmax
        self.omega_cold = omega_cold
        self.As = As
        self.omega_baryon = omega_baryon
        self.ns = ns
        self.h = h
        self.neutrino_mass = neutrino_mass
        self.w0 = w0
        self.wa = wa
        self.expfactor = expfactor
        self.cold = cold
        # hard coded no curvature for now
        self.Ok0 = 0
        # CMB related, not needed
        self.Neff = Neff
        self.Tcmb0 = Planck18.Tcmb0
        self.tau = tau
        self.camb_dark_energy_model = "ppf"
        self.num_kpoints = num_kpoints
        self.karr_in_h = np.geomspace(
            self.kmin / self.h, self.kmax / self.h, self.num_kpoints
        )
        self.omega_de = omega_de

    @property
    def omega_de(self):
        """
        The dark energy density fraction at z=0.
        """
        return self._omega_de

    @omega_de.setter
    def omega_de(self, value):
        if value is None:
            value = self.get_derived_Ode()
        self._omega_de = value

    def set_astropy_cosmo(self, name="new"):
        """
        Generate a :class:`astropy.cosmology.w0waCDM` set by the input cosmology.
        Note that the dark energy density ``Ode0`` is a derived quantity which is
        calculated using ``camb`` if not put in.
        """
        # there is some strange overriding issue from astropy
        w0 = deepcopy(self.w0)
        wa = deepcopy(self.wa)
        h = deepcopy(self.h)
        omega_cold = deepcopy(self.omega_cold)
        omega_de = deepcopy(self.omega_de)
        m_nu = deepcopy(self.neutrino_mass)
        omega_baryon = deepcopy(self.omega_baryon)
        cosmo = w0waCDM(
            H0=h * 100,
            Om0=omega_cold,
            Ode0=omega_de,
            Tcmb0=self.Tcmb0.value,
            Neff=self.Neff,
            m_nu=[0, 0, m_nu],
            Ob0=omega_baryon,
            w0=w0,
            wa=wa,
            name=name,
        )
        return cosmo

    def get_camb_pars(self):
        """
        Generate a :class:`camb.model.CAMBparams` set by the input cosmology.
        """
        pars = camb.CAMBparams()
        pars.set_cosmology(
            H0=self.h * 100,
            ombh2=self.omega_baryon * self.h**2,
            omch2=(self.omega_cold - self.omega_baryon) * self.h**2,
            omk=self.Ok0,
            mnu=self.neutrino_mass,
            # these should not affect matter ps?
            nnu=self.Neff,
            TCMB=self.Tcmb0.value,
            tau=self.tau,
        )
        pars.InitPower.set_params(As=self.As, ns=self.ns)

        if self.ps_type == "linear":
            instr = "none"
        else:
            instr = "both"
        pars.NonLinear = getattr(camb.model, "NonLinear_" + instr)
        pars.set_dark_energy(
            w=self.w0, wa=self.wa, dark_energy_model=self.camb_dark_energy_model
        )
        # suppress the output of camb
        with HiddenPrints():
            pars.set_matter_power(
                redshifts=np.unique([0.0, 1 / self.expfactor - 1]),
                kmax=self.kmax / self.h,
            )
        return pars

    def get_derived_Ode(self):
        """
        Use camb to calculate the Ode0 given input parameters.
        """
        # if self._omega_de is None:
        camb_pars = self.get_camb_pars()
        results = camb.get_background(camb_pars)
        tot, de = results.get_background_densities(1.0, ["tot", "de"]).values()
        self.omega_de = (de / tot)[0]
        self._omega_de = (de / tot)[0]

        return self.omega_de

    def get_bacco_pars(self):
        """
        Generate a dictionary that can be used as input for the
        ``bacco`` emulator. Currently only support non-baryonic
        matter power.
        """
        params = {
            "omega_cold": self.omega_cold,
            #'sigma8_cold'   :  self.sigma8_cold,
            "A_s": self.As,
            "omega_baryon": self.omega_baryon,
            "ns": self.ns,
            "hubble": self.h,
            "neutrino_mass": self.neutrino_mass,
            "w0": self.w0,
            "wa": self.wa,
            "expfactor": self.expfactor,
        }
        return params

    def get_matter_power_spectrum_camb(self):
        """
        Compute the CDM power spectrum using camb.
        """
        camb_pars = self.get_camb_pars()
        results = camb.get_results(camb_pars)
        # get sigma8
        s8_fid = results.get_sigma8_0()
        self.sigma_8_0 = s8_fid
        self.f_growth = results.get_fsigma8()[0] / results.get_sigma8()[0]
        self.sigma_8_z = results.get_sigma8()[0]
        kh, z, pk_camb = results.get_matter_power_spectrum(
            minkh=self.kmin / self.h,
            maxkh=self.kmax / self.h,
            npoints=self.num_kpoints,
            var1=7 - 5 * int(self.cold),
            var2=7 - 5 * int(self.cold),
        )
        return pk_camb[np.argmax(z)]

    def get_matter_power_spectrum_bacco(self):
        """
        Emulate the CDM power spectrum using bacco.
        """
        import baccoemu

        emulator = baccoemu.Matter_powerspectrum()
        bacco_pars = self.get_bacco_pars()
        _, baccopk = getattr(emulator, f"get_{self.ps_type}_pk")(
            k=self.karr_in_h, cold=self.cold, **bacco_pars
        )
        self.sigma_8_0 = emulator.get_sigma8(cold=True, **self.get_bacco_pars())
        # an approximate fitting formulae for growth
        wz1 = self.w0 + 0.5 * self.wa
        gamma = 0.55 + (1 + wz1) * (0.05 * float(wz1 >= -1) + 0.02 * float(wz1 < -1))
        self.get_derived_Ode()
        cosmo = self.set_astropy_cosmo()
        self.f_growth = cosmo.Om(1 / self.expfactor - 1) ** gamma
        return baccopk


class CosmologyCalculator(Specification):
    """
    The class for storing the cosmological model used for calculation.

    The underlying cosmological model is defined via :class:`astropy.cosmology.w0waCDM` with all the background
    properties calculated via ``astropy``.

    The matter density fluctuation is calculated using ``camb`` or ``baccoemu`` based on the input `backend`.

    Parameters
    ----------
    backend: str, default "camb"
        The backend to use for computing the matter power spectrum.
        Either "camb" or "bacco".
    omega_hi: float or np.ndarray, default 5e-4
        The HI density as a function of redshift,
        over the critical density of the Universe at z=0.
        If a float is provided, it will be used as the HI density at all redshifts.
        If an array is provided, it will be used as the HI density at each frequency channel.
    fiducial_cosmology: dict, default Planck18
        The fiducial cosmology parameters.
        Fiducial cosmology should be used to perform the power spectrum
        estimation, such as transforming sky coordinates to comoving coordinates.
    true_cosmology: dict, default Planck18
        The true cosmology parameters.
        True cosmology should be varied during parameter inference.
    ps_type: str, default "linear"
        The type of the matter power spectrum.
    kmin: float, default 1e-3
        The minimum k in Mpc^-1 for calculating matter power. k below kmin will be extrapolated.
    kmax: float, default 3.0
        The maximum k in Mpc^-1 for calculating matter power. k above kmax will be extrapolated.
    num_kpoints: int, default 200
        The number of k points to compute the interpolation of the matter power spectrum.
    cold: bool, default True
        Whether to use the cold matter power spectrum.
        If True, the matter power spectrum refers to CDM+Baryon.
        If False, the matter power spectrum refers to all matter including massive neutrinos.
    **params: dict
        Additional parameters to be passed to the base class :class:`Specification`
    """

    def __init__(
        self,
        backend: str = "camb",
        omega_hi: np.ndarray | float = 5e-4,
        fiducial_cosmology: str | dict = fiducial_dict,
        true_cosmology: str | dict | None = None,
        ps_type: str = "linear",
        kmin: float = 1e-3,
        kmax: float = 3.0,
        num_kpoints: int = 200,
        cold: bool = True,
        **params,
    ):
        Specification.__init__(self, **params)
        self._expfactor = None
        self.backend = backend
        self.ps_type = ps_type
        self.kmin = kmin
        self.kmax = kmax
        self.num_kpoints = num_kpoints
        self.cold = cold
        self.fiducial_cosmology = fiducial_cosmology
        if true_cosmology is None:
            true_cosmology = fiducial_cosmology
        self.true_cosmology = true_cosmology
        self._matter_power_spectrum_fnc = None
        self.omega_hi = omega_hi
        self._sound_horizon_drag_true = None
        self._sound_horizon_drag_fiducial = None
        self._alpha_parallel = None
        self._alpha_perp = None

    @property
    def omega_cold(self):
        """
        The cold matter density parameter for the true (fitted) cosmology.
        """
        return self.cospar_true.omega_cold

    @omega_cold.setter
    def omega_cold(self, value: float):
        true_cosmology = self.true_cosmology.copy()
        true_cosmology["omega_cold"] = value
        self.true_cosmology = true_cosmology

    @property
    def As(self):
        """
        The amplitude of the primordial power spectrum for the true (fitted) cosmology.
        Note it does not update fiducial cosmology for power spectrum estimation,
        and only used for model fitting.
        """
        return self.cospar_true.As

    @As.setter
    def As(self, value: float):
        true_cosmology = self.true_cosmology.copy()
        true_cosmology["As"] = value
        self.true_cosmology = true_cosmology

    @property
    def omega_baryon(self):
        """
        The baryon matter density parameter for the true (fitted) cosmology.
        """
        return self.cospar_true.omega_baryon

    @omega_baryon.setter
    def omega_baryon(self, value: float):
        true_cosmology = self.true_cosmology.copy()
        true_cosmology["omega_baryon"] = value
        self.true_cosmology = true_cosmology

    @property
    def h(self):
        """
        The Hubble constant for the true (fitted) cosmology.
        """
        return self.cospar_true.h

    @h.setter
    def h(self, value: float):
        true_cosmology = self.true_cosmology.copy()
        true_cosmology["h"] = value
        self.true_cosmology = true_cosmology

    @property
    def neutrino_mass(self):
        """
        The neutrino mass for the true (fitted) cosmology.
        """
        return self.cospar_true.neutrino_mass

    @neutrino_mass.setter
    def neutrino_mass(self, value: float):
        true_cosmology = self.true_cosmology.copy()
        true_cosmology["neutrino_mass"] = value
        self.true_cosmology = true_cosmology

    @property
    def w0(self):
        """
        The dark energy equation of state parameter for the true (fitted) cosmology.
        """
        return self.cospar_true.w0

    @w0.setter
    def w0(self, value: float):
        true_cosmology = self.true_cosmology.copy()
        true_cosmology["w0"] = value
        self.true_cosmology = true_cosmology

    @property
    def wa(self):
        """
        The dark energy equation of state parameter for the true (fitted) cosmology.
        """
        return self.cospar_true.wa

    @wa.setter
    def wa(self, value: float):
        true_cosmology = self.true_cosmology.copy()
        true_cosmology["wa"] = value
        self.true_cosmology = true_cosmology

    @property
    def ns(self):
        """
        The primordial power spectrum spectral index for the true (fitted) cosmology.
        """
        return self.cospar_true.ns

    @ns.setter
    def ns(self, value: float):
        true_cosmology = self.true_cosmology.copy()
        true_cosmology["ns"] = value
        self.true_cosmology = true_cosmology

    def get_cospar(self, cosmology: dict):
        """
        Generate a :class:`CosmologyParameters` object from the input cosmology.

        Parameters
        ----------
        cosmology: dict
            The cosmology parameters.

        Returns
        -------
        cospar: :class:`CosmologyParameters`
            The cosmology parameters object.
        """
        return CosmologyParameters(
            ps_type=self.ps_type,
            kmin=self.kmin,
            kmax=self.kmax,
            num_kpoints=self.num_kpoints,
            expfactor=self.expfactor,
            cold=self.cold,
            **cosmology,
        )

    @property
    def fiducial_cosmology(self):
        return self._fiducial_cosmology

    @fiducial_cosmology.setter
    def fiducial_cosmology(self, value: dict | str):
        if isinstance(value, str):
            value = get_cosmo_dict(value)
        self._fiducial_cosmology = value
        self._cospar_fiducial = self.get_cospar(value)
        self.astropy_cosmo_fiducial = self.cospar_fiducial.set_astropy_cosmo()
        logger.debug(
            f"cleaning cache of {self.cosmo_fid_dep_attr} due to resetting fiducial_cosmology"
        )
        self.clean_cache(self.cosmo_fid_dep_attr)

    @property
    def true_cosmology(self):
        return self._true_cosmology

    @true_cosmology.setter
    def true_cosmology(self, value: dict | str):
        if isinstance(value, str):
            value = get_cosmo_dict(value)
        self._true_cosmology = value
        self._cospar_true = self.get_cospar(value)
        self.astropy_cosmo_true = self.cospar_true.set_astropy_cosmo()
        logger.debug(
            f"cleaning cache of {self.cosmo_model_dep_attr} due to resetting true_cosmology"
        )
        self.clean_cache(self.cosmo_model_dep_attr)

    @property
    @tagging("nu")
    def cospar_fiducial(self):
        if self._cospar_fiducial is None:
            self._cospar_fiducial = self.get_cospar(self.fiducial_cosmology)
        return self._cospar_fiducial

    @property
    @tagging("nu")
    def cospar_true(self):
        if self._cospar_true is None:
            self._cospar_true = self.get_cospar(self.true_cosmology)
        return self._cospar_true

    @property
    def f_growth_true(self):
        """
        The growth factor at ``self.expfactor`` for the true (fitted) cosmology.
        """
        if "f_growth" not in self.cospar_true.__dict__:
            getattr(self.cospar_true, f"get_matter_power_spectrum_{self.backend}")()
        return self.cospar_true.f_growth

    @property
    def f_growth_fiducial(self):
        """
        The growth factor at ``self.expfactor`` for the fiducial cosmology.
        """
        if "f_growth" not in self.cospar_fiducial.__dict__:
            getattr(self.cospar_fiducial, f"get_matter_power_spectrum_{self.backend}")()
        return self.cospar_fiducial.f_growth

    @property
    @tagging("nu")
    def expfactor(self):
        """
        The expansion factor
        """
        if self._expfactor is None:
            self._expfactor = 1 / (1 + self.z)
        return self._expfactor

    @property
    def ps_type(self):
        """
        linear or nonlinear for the matter power.
        """
        return self._ps_type

    @ps_type.setter
    def ps_type(self, value):
        self._ps_type = value
        logger.debug(
            f"cleaning cache of {self.cosmo_fid_dep_attr} due to resetting ps_type"
        )
        self.clean_cache(self.cosmo_fid_dep_attr)
        logger.debug(
            f"cleaning cache of {self.cosmo_model_dep_attr} due to resetting ps_type"
        )
        self.clean_cache(self.cosmo_model_dep_attr)

    @property
    def kmin(self):
        """
        The minimum k in Mpc^-1 for calculating matter power. k below kmin will be extrapolated.
        """
        return self._kmin

    @kmin.setter
    def kmin(self, value):
        self._kmin = value
        logger.debug(
            f"cleaning cache of {self.cosmo_fid_dep_attr} due to resetting kmin"
        )
        self.clean_cache(self.cosmo_fid_dep_attr)
        logger.debug(
            f"cleaning cache of {self.cosmo_model_dep_attr} due to resetting kmin"
        )
        self.clean_cache(self.cosmo_model_dep_attr)

    @property
    def kmax(self):
        """
        The maximum k in Mpc^-1 for calculating matter power. k above kmax will be extrapolated.
        """
        return self._kmax

    @kmax.setter
    def kmax(self, value):
        self._kmax = value
        logger.debug(
            f"cleaning cache of {self.cosmo_fid_dep_attr} due to resetting kmax"
        )
        self.clean_cache(self.cosmo_fid_dep_attr)
        logger.debug(
            f"cleaning cache of {self.cosmo_model_dep_attr} due to resetting kmax"
        )
        self.clean_cache(self.cosmo_model_dep_attr)

    @property
    def num_kpoints(self):
        """
        The number of k points for calculating matter power.
        """
        return self._num_kpoints

    @num_kpoints.setter
    def num_kpoints(self, value: int):
        self._num_kpoints = value
        logger.debug(
            f"cleaning cache of {self.cosmo_fid_dep_attr} due to resetting num_kpoints"
        )
        self.clean_cache(self.cosmo_fid_dep_attr)
        logger.debug(
            f"cleaning cache of {self.cosmo_model_dep_attr} due to resetting num_kpoints"
        )
        self.clean_cache(self.cosmo_model_dep_attr)

    @property
    def cold(self):
        """
        If True (recommended), the matter power spectrum is the CDM ps.
        If False, it will include massive neutrino for bacco and total matter
        for camb.
        """
        return self._cold

    @cold.setter
    def cold(self, value):
        self._cold = value
        logger.debug(
            f"cleaning cache of {self.cosmo_fid_dep_attr} due to resetting self.cold"
        )
        self.clean_cache(self.cosmo_fid_dep_attr)
        logger.debug(
            f"cleaning cache of {self.cosmo_model_dep_attr} due to resetting self.cold"
        )
        self.clean_cache(self.cosmo_model_dep_attr)

    @property
    def backend(self):
        """
        Which backend to use for computing the matter power.
        Either camb or bacco.
        """
        return self._backend

    @backend.setter
    def backend(self, value):
        self._backend = value
        logger.debug(
            f"cleaning cache of {self.cosmo_fid_dep_attr} due to resetting self.backend"
        )
        self.clean_cache(self.cosmo_fid_dep_attr)
        logger.debug(
            f"cleaning cache of {self.cosmo_model_dep_attr} due to resetting self.backend"
        )
        self.clean_cache(self.cosmo_model_dep_attr)

    @property
    def omega_hi_z_func(self):
        """
        Interpolate the input ``self.omega_hi`` at each frequency channel ``self.z_ch`` to a function of redshift.
        """
        if np.all(self.omega_hi == self.omega_hi[0]):
            return lambda z: self.omega_hi[0] * np.ones_like(z)
        func = interp1d(
            self.z_ch, self.omega_hi, bounds_error=False, fill_value="extrapolate"
        )
        return func

    @property
    @tagging("nu")
    def omega_hi_z_mean(self):
        """
        The mean HI density at the central redshift ``self.z``, over the critical density of the Universe at z=0.
        Interpolated from the input ``self.omega_hi`` at each frequency channel ``self.z_ch``.
        """
        if self._omega_hi_z_mean is None:
            self._omega_hi_z_mean = self.omega_hi_z_func(self.z)
        return self._omega_hi_z_mean

    @property
    def average_hi_temp(self):
        """
        The average HI brightness temperature in Kelvin at central redshift ``self.z``.
        Calculation is based on the true (fitted) cosmology.
        """
        logger.debug(
            f"invoking {inspect.currentframe().f_code.co_name} to calculate the average HI brightness temperature"
        )
        logger.debug(
            f"omega_hi: {self.omega_hi_z_mean}, z: {self.z}, cosmo: {self.astropy_cosmo_true}"
        )
        tbar = omega_hi_to_average_temp(
            self.omega_hi_z_mean, z=self.z, cosmo=self.astropy_cosmo_true
        )
        return tbar

    @property
    def omega_hi(self):
        """
        The HI density as a function of redshift, over the critical density of the Universe at z=0,
        defined at each frequency channel so that ``self.omega_hi`` corresponds to ``self.z_ch``.
        Interpolation will be used to get the HI density at other redshifts.
        If a float is provided, it will be used as the HI density at all redshifts.
        """
        return self._omega_hi

    @omega_hi.setter
    def omega_hi(self, value):
        if isinstance(value, float):
            result = value * np.ones_like(self.z_ch)
        else:
            assert len(value) == len(
                self.z_ch
            ), "omega_hi must be defined at each frequency channel if an array"
            result = value
        self._omega_hi_z_mean = None
        self._omega_hi = result

    @property
    @tagging("cosmo_model", "nu")
    def matter_power_spectrum_fnc(self):
        """
        Interpolation function for the real-space isotropic matter power spectrum.
        The matter power spectrum is calculated for the true (fitted) cosmology.
        """
        if self._matter_power_spectrum_fnc is None:
            self.get_matter_power_spectrum()
        return self._matter_power_spectrum_fnc

    def get_matter_power_spectrum(self):
        """
        Calculate the matter power spectrum, interpolate it, and save it into the class attribute `matter_power_spectrum_fnc`.

        Note that, Alcock–Paczynski effect is included here in the power spectrum,
        and therefore for all model power spectra that depends on the matter power spectrum.
        """
        cosmo = self.astropy_cosmo_true
        kh = self.cospar_true.karr_in_h
        pk = getattr(self.cospar_true, f"get_matter_power_spectrum_{self.backend}")()
        karr = kh * cosmo.h
        pkarr = pk / cosmo.h**3
        AP_amp = 1 / (self.alpha_iso**3)
        matter_power_func = interp1d(
            karr,
            pkarr * AP_amp,
            bounds_error=False,
            fill_value="extrapolate",
        )
        logger.info(
            f"{inspect.currentframe().f_code.co_name}_{self.backend}: "
            "setting self._matter_power_spectrum_fnc"
        )
        self._matter_power_spectrum_fnc = matter_power_func

    def deltaz_to_deltar(self, delta_z):
        """
        Convert a redshift interval delta_z to a comoving distance interval delta_r.
        The conversion is based on the true (fitted) cosmology.

        Note that, the usual redshift error defined in galaxy survey is usually delta_z / (1+z).

        Parameters
        ----------
        delta_z: float.
            The redshift interval.

        Returns
        -------
        delta_r: float.
            The comoving distance interval in Mpc.
        """
        cosmo = self.astropy_cosmo_true
        H_z = cosmo.H(self.z)
        delta_r = (delta_z * astropy.constants.c / H_z).to("Mpc").value
        return delta_r

    def deltav_to_deltar(self, delta_v):
        """
        Convert a velocity interval delta_v to a comoving distance interval delta_r.
        The conversion is based on the true (fitted) cosmology.

        Parameters
        ----------
        delta_v: float.
            The velocity interval in km/s.

        Returns
        -------
        delta_r: float.
            The comoving distance interval in Mpc.
        """
        cosmo = self.astropy_cosmo_true
        H_z = cosmo.H(self.z)
        delta_r = (1 + self.z) * delta_v / H_z.to("km s^-1 Mpc^-1").value
        return delta_r

    @property
    @tagging("cosmo_model", "beam")
    def sigma_beam_ch_in_mpc(self):
        """
        The input beam size parameter in Mpc in each channel.
        The comoving distance is calculated for the true (fitted) cosmology.
        """
        if self._sigma_beam_ch_in_mpc is None and self.sigma_beam_ch is not None:
            self._sigma_beam_ch_in_mpc = (
                self.astropy_cosmo_true.comoving_distance(self.z_ch).to("Mpc").value
                * (self.sigma_beam_ch * self.beam_unit).to("rad").value
            )
        return self._sigma_beam_ch_in_mpc

    @property
    def pix_resol_in_mpc(self):
        """
        angular resolution of the map pixel in Mpc for the fiducial cosmology.
        """
        return (
            np.sqrt(self.pixel_area)
            * np.pi
            / 180
            * self.astropy_cosmo_fiducial.comoving_distance(self.z).to("Mpc").value
        )

    @property
    def los_resol_in_mpc(self):
        """
        effective frequency resolution in Mpc for the fiducial cosmology.
        """
        comov_dist = self.astropy_cosmo_fiducial.comoving_distance(self.z_ch).value
        los_resol_in_mpc = (comov_dist.max() - comov_dist.min()) / len(self.nu)
        return los_resol_in_mpc

    @property
    @tagging("cosmo_fid")
    def z_as_func_of_comov_dist(self):
        """
        Returns a function that returns the redshift
        for input comoving distance.
        The comoving distance is calculated for the fiducial cosmology.
        """
        if self._z_as_func_of_comov_dist is None:
            self.get_z_as_func_of_comov_dist()
        return self._z_as_func_of_comov_dist

    def get_z_as_func_of_comov_dist(self):
        """
        Calculate an array of comoving distances with redshifts,
        and construct a function that returns the redshift for input comoving distance.
        The function is saved into the class attribute `z_as_func_of_comov_dist`.
        The comoving distance is calculated for the fiducial cosmology.
        """
        zarr = np.linspace(0, self.z_interp_max, 20001)
        xarr = self.astropy_cosmo_fiducial.comoving_distance(zarr).value
        func = interp1d(xarr, zarr)
        self._z_as_func_of_comov_dist = func

    @property
    def survey_volume(self, i=None):
        """
        Total survey volume in Mpc^3.
        The volume is calculated for the fiducial cosmology.

        Note that, the sampling along the sky map is assumed to be the same for all frequency channels,
        and the code by default uses the maximum sampling channel to calculate the area.
        This is desired, as the survey lightcone can contain holes inside, which is considered part of the volume.

        Parameters
        ----------
        i: int, default None
            The index of the frequency channel to calculate the survey volume.
            Default is None, which uses the maximum sampling channel.
        """
        cosmo = self.astropy_cosmo_fiducial
        if i is None:
            i = self.maximum_sampling_channel
        nu_ext = center_to_edges(self.nu)
        z_ext = freq_to_redshift(nu_ext)
        volume = (
            (self.W_HI[:, :, i].sum() * self.pixel_area * (np.pi / 180) ** 2)
            / 3
            * (
                cosmo.comoving_distance(z_ext.max()) ** 3
                - cosmo.comoving_distance(z_ext.min()) ** 3
            ).value
        )
        return volume

    def get_sound_horizon_drag(
        self,
        Omh2: float,
        Obh2: float,
        Onuh2: float,
    ):
        """
        Calculate the sound horizon at drag epoch for a set of cosmological parameters.

        Uses the fitting formula from [1411.1074](https://arxiv.org/abs/1411.1074)
        """
        rs_d = (
            55.154
            * np.exp(-72.3 * (Onuh2 + 0.0006) ** 2)
            / (Obh2**0.12807 * (Omh2 - Onuh2) ** 0.25351)
        )
        return rs_d

    @property
    @tagging("cosmo_model")
    def sound_horizon_drag_true(self):
        """
        The sound horizon at drag epoch for the true (fitted) cosmology.
        """
        if self._sound_horizon_drag_true is None:
            self._sound_horizon_drag_true = self.get_sound_horizon_drag(
                Omh2=self.astropy_cosmo_true.Om0 * self.astropy_cosmo_true.h**2,
                Obh2=self.astropy_cosmo_true.Ob0 * self.astropy_cosmo_true.h**2,
                Onuh2=self.astropy_cosmo_true.Onu0 * self.astropy_cosmo_true.h**2,
            )
        return self._sound_horizon_drag_true

    @property
    @tagging("cosmo_fid")
    def sound_horizon_drag_fiducial(self):
        """
        The sound horizon at drag epoch for the fiducial cosmology.
        """
        if self._sound_horizon_drag_fiducial is None:
            self._sound_horizon_drag_fiducial = self.get_sound_horizon_drag(
                Omh2=self.astropy_cosmo_fiducial.Om0
                * self.astropy_cosmo_fiducial.h**2,
                Obh2=self.astropy_cosmo_fiducial.Ob0
                * self.astropy_cosmo_fiducial.h**2,
                Onuh2=self.astropy_cosmo_fiducial.Onu0
                * self.astropy_cosmo_fiducial.h**2,
            )
        return self._sound_horizon_drag_fiducial

    @property
    @tagging("cosmo_model", "cosmo_fid", "nu")
    def alpha_parallel(self):
        """
        The line of sight Alcock–Paczynski effect parameter.
        """
        if self._alpha_parallel is None:
            self._alpha_parallel = self.get_alpha_parallel()
        return self._alpha_parallel

    def get_alpha_parallel(self):
        """
        Calculate the line of sight Alcock–Paczynski effect parameter.
        """
        # actual DH has a factor of c which is cancelled out here
        DH_over_rd_fid = (
            1
            / self.astropy_cosmo_fiducial.H(self.z).value
            / self.sound_horizon_drag_fiducial
        )
        DH_over_rd_true = (
            1 / self.astropy_cosmo_true.H(self.z).value / self.sound_horizon_drag_true
        )
        return DH_over_rd_true / DH_over_rd_fid

    @property
    @tagging("cosmo_model", "cosmo_fid", "nu")
    def alpha_perp(self):
        if self._alpha_perp is None:
            self._alpha_perp = self.get_alpha_perp()
        return self._alpha_perp

    def get_alpha_perp(self):
        """
        Calculate the transverse Alcock–Paczynski effect parameter.
        """
        Dm_over_rd_fid = (
            self.astropy_cosmo_fiducial.comoving_transverse_distance(self.z).value
            / self.sound_horizon_drag_fiducial
        )
        Dm_over_rd_true = (
            self.astropy_cosmo_true.comoving_transverse_distance(self.z).value
            / self.sound_horizon_drag_true
        )
        return Dm_over_rd_true / Dm_over_rd_fid

    @property
    def alpha_iso(self):
        r"""
        The isotropic Alcock–Paczynski effect parameter,
        which is :math:`\alpha_{\parallel}^{1/3} \alpha_{\perp}^{2/3}`.
        """
        return self.alpha_parallel ** (1 / 3) * self.alpha_perp ** (2 / 3)

    @property
    def alpha_AP(self):
        r"""
        The anisotropic Alcock–Paczynski effect parameter,
        which is :math:`\alpha_{\parallel} / \alpha_{\perp}`.
        """
        return self.alpha_parallel / self.alpha_perp

    @property
    def cosmo(self):
        """
        A shortcut to the :py:class:`astropy.cosmology.Cosmology` object for the true cosmology.

        Should only be used when true and fiducial cosmology are the same, and returns a warning if not.

        If you set `cosmo` to a new value, it will set `true_cosmology` and `fiducial_cosmology` to the same value.
        If `true_cosmology` and `fiducial_cosmology` are different, an error will be raised.
        """
        if self.true_cosmology != self.fiducial_cosmology:
            warnings.warn(
                "true and fiducial cosmology are different, this shortcut is for the true cosmology:"
                f"{self.true_cosmology}"
            )
        return self.astropy_cosmo_true

    @cosmo.setter
    def cosmo(self, value):
        if self.true_cosmology != self.fiducial_cosmology:
            raise ValueError(
                "true and fiducial cosmology are different, cannot set cosmo to a new value"
                "Please set `true_cosmology` and `fiducial_cosmology` respectively"
            )
        self.true_cosmology = value
        self.fiducial_cosmology = value
