"""
This module contains the class for storing the cosmological model used for calculation.

The class :py:class:`CosmologyParameters` is the class for storing the cosmological parameters, and settings for computing matter power spectrum.

The class :py:class:`CosmologyCalculator` is the base class for storing the cosmological model used for calculation.
It is typically used as a base class for other classes that inherit from it, and not used directly.

"""
import numpy as np
import camb
import astropy
from meer21cm import Specification
from scipy.interpolate import interp1d
from meer21cm.util import omega_hi_to_average_temp, tagging, HiddenPrints
from astropy.cosmology import Planck18, w0waCDM
from copy import deepcopy
import inspect
import logging
from typing import Callable

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


def extract_astropy_cosmo_set(value):
    """
    Extract predefined astropy :py:class:`astropy.cosmology.FlatLambdaCDM` cosmology and turn it into a :py:class:`astropy.cosmology.w0waCDM` object.

    Parameters
    ----------
    value: str
        The name of the cosmology, e.g. Planck18

    Returns
    -------
    cosmo: :class:`astropy.cosmology.w0waCDM`
        The output cosmology.
    """
    self = getattr(astropy.cosmology, value)
    cosmo = w0waCDM(
        H0=self.H0,
        Om0=self.Om0,
        Ode0=self.Ode0,
        w0=-1.0,
        wa=0.0,
        Tcmb0=self.Tcmb0,
        Neff=self.Neff,
        m_nu=self.m_nu,
        Ob0=self.Ob0,
        name=value,
    )
    return cosmo


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
        omega_cold=Planck18.Om0,
        As=np.exp(3.047) / 1e10,
        # sigma8_cold=Planck18.meta['sigma8'],
        omega_baryon=Planck18.Ob0,
        ns=Planck18.meta["n"],
        h=Planck18.h,
        neutrino_mass=Planck18.m_nu.sum().value,
        w0=-1.0,
        wa=0.0,
        expfactor=1.0,
        cold=True,
        num_kpoints=200,
        omega_de=None,
        tau=0.0561,
        Neff=3.046,
        **params,
    ):
        self.ps_type = ps_type
        self.kmin = kmin
        self.kmax = kmax
        self._omega_cold = omega_cold
        # self.sigma8_cold = sigma8_cold
        self.As = As
        self._omega_baryon = omega_baryon
        self.ns = ns
        self._h = h
        self._neutrino_mass = neutrino_mass
        self._w0 = w0
        self._wa = wa
        self._expfactor = expfactor
        self.cold = cold
        # hard coded no curvature for now
        self.Ok0 = 0
        # CMB related, not needed
        self.Neff = Neff
        # self.Neff = 2.0
        self.Tcmb0 = Planck18.Tcmb0
        self.tau = tau
        self.camb_dark_energy_model = "ppf"
        self.num_kpoints = num_kpoints
        self.karr_in_h = np.geomspace(
            self.kmin / self._h, self.kmax / self._h, self.num_kpoints
        )
        self._omega_de = omega_de
        self.omega_de = omega_de

    @property
    def expfactor(self):
        r"""
        The expansion factor which is calculated as :math:`a = 1 / (1 + z)`.
        """
        return self._expfactor

    def set_astropy_cosmo(self, name="new"):
        """
        Generate a :class:`astropy.cosmology.w0waCDM` set by the input cosmology.
        Note that the dark energy density ``Ode0`` is a derived quantity which is
        calculated using ``camb`` if not put in.
        """
        # there is some strange overriding issue from astropy
        w0 = deepcopy(self._w0)
        wa = deepcopy(self._wa)
        h = deepcopy(self._h)
        omega_cold = deepcopy(self._omega_cold)
        omega_de = deepcopy(self.omega_de)
        m_nu = deepcopy(self._neutrino_mass)
        omega_baryon = deepcopy(self._omega_baryon)
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
            H0=self._h * 100,
            ombh2=self._omega_baryon * self._h**2,
            omch2=(self._omega_cold - self._omega_baryon) * self._h**2,
            omk=self.Ok0,
            mnu=self._neutrino_mass,
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
            w=self._w0, wa=self._wa, dark_energy_model=self.camb_dark_energy_model
        )
        # suppress the output of camb
        with HiddenPrints():
            pars.set_matter_power(
                redshifts=np.unique([0.0, 1 / self.expfactor - 1]),
                kmax=self.kmax / self._h,
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
            "omega_cold": self._omega_cold,
            #'sigma8_cold'   :  self.sigma8_cold,
            "A_s": self.As,
            "omega_baryon": self._omega_baryon,
            "ns": self.ns,
            "hubble": self._h,
            "neutrino_mass": self._neutrino_mass,
            "w0": self._w0,
            "wa": self._wa,
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
            minkh=self.kmin / self._h,
            maxkh=self.kmax / self._h,
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
        wz1 = self._w0 + 0.5 * self._wa
        gamma = 0.55 + (1 + wz1) * (0.05 * float(wz1 >= -1) + 0.02 * float(wz1 < -1))
        self.get_derived_Ode()
        cosmo = self.set_astropy_cosmo()
        self.f_growth = cosmo.Om(1 / self.expfactor - 1) ** gamma
        return baccopk


class CosmologyCalculator(Specification, CosmologyParameters):
    """
    The class for storing the cosmological model used for calculation.

    The underlying cosmological model is defined via :class:`astropy.cosmology.LambdaCDM` with all the background
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
    **params: dict
        Additional parameters to be passed to the base class :class:`CosmologyParameters`
        and :class:`meer21cm.dataanalysis.Specification`.
    """

    def __init__(
        self,
        backend: str = "camb",
        omega_hi: np.ndarray | float = 5e-4,
        **params: dict,
    ):
        # super().__init__(**params)
        self._omega_de = None
        self._expfactor = None
        Specification.__init__(self, **params)

        # override redshift and cosmology
        CosmologyParameters.__init__(self, expfactor=self.expfactor, **params)
        self.backend = backend
        # reset background cosmology based on input
        if "cosmo" not in params.keys():
            self.cosmo = self.set_astropy_cosmo()
        else:
            self.cosmo = params["cosmo"]
        self._matter_power_spectrum_fnc = None
        self.omega_hi = omega_hi

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
            f"cleaning cache of {self.cosmo_dep_attr} due to resetting ps_type"
        )
        self.clean_cache(self.cosmo_dep_attr)

    @property
    def kmin(self):
        """
        The minimum k in Mpc^-1 for calculating matter power. k below kmin will be extrapolated.
        """
        return self._kmin

    @kmin.setter
    def kmin(self, value):
        self._kmin = value
        logger.debug(f"cleaning cache of {self.cosmo_dep_attr} due to resetting kmin")
        self.clean_cache(self.cosmo_dep_attr)

    @property
    def kmax(self):
        """
        The maximum k in Mpc^-1 for calculating matter power. k above kmax will be extrapolated.
        """
        return self._kmax

    @kmax.setter
    def kmax(self, value):
        self._kmax = value
        logger.debug(f"cleaning cache of {self.cosmo_dep_attr} due to resetting kmax")
        self.clean_cache(self.cosmo_dep_attr)

    @property
    def omega_cold(self):
        """
        The density fraction of CDM+Baryon at z=0.
        """
        return self._omega_cold

    @omega_cold.setter
    def omega_cold(self, value):
        self._omega_cold = value
        # update background cosmology, clear cache triggered automatically
        logger.debug(
            f"recalculating Ode and update background cosmology due to resetting omega_cold"
        )
        self.get_derived_Ode()
        cosmo = self.set_astropy_cosmo()
        self.cosmo = cosmo.clone(Om0=value)

    @property
    def As(self):
        """
        The amplitude of the initial power spectrum
        """
        return self._As

    @As.setter
    def As(self, value):
        self._As = value
        logger.debug(f"cleaning cache of {self.cosmo_dep_attr} due to resetting As")
        self.clean_cache(self.cosmo_dep_attr)

    @property
    def omega_baryon(self):
        """
        The energy fraction of the baryons at current z=0.
        """
        return self._omega_baryon

    @omega_baryon.setter
    def omega_baryon(self, value):
        self._omega_baryon = value
        logger.debug(
            f"recalculating Ode and update background cosmology due to resetting omega_baryon"
        )
        self.get_derived_Ode()
        cosmo = self.set_astropy_cosmo()
        self.cosmo = cosmo.clone(Ob0=value)

    @property
    def ns(self):
        """
        Running index of the initial power spectrum
        """
        return self._ns

    @ns.setter
    def ns(self, value):
        self._ns = value
        logger.debug(f"cleaning cache of {self.cosmo_dep_attr} due to resetting ns")
        self.clean_cache(self.cosmo_dep_attr)

    @property
    def h(self):
        """
        The Hubble parameter over 100km/s/Mpc.
        """
        return self._h

    @h.setter
    def h(self, value):
        self._h = value
        logger.debug(
            f"recalculating Ode and update background cosmology due to resetting h"
        )
        self.get_derived_Ode()
        cosmo = self.set_astropy_cosmo()
        self.cosmo = cosmo.clone(H0=value * 100)

    @property
    def neutrino_mass(self):
        """
        sum of the neutrino mass in eV.
        """
        return self._neutrino_mass

    @neutrino_mass.setter
    def neutrino_mass(self, value):
        self._neutrino_mass = value
        logger.debug(
            f"recalculating Ode and update background cosmology due to resetting neutrino_mass"
        )
        self.get_derived_Ode()
        cosmo = self.set_astropy_cosmo()
        self.cosmo = cosmo.clone(m_nu=[0, 0, value])

    @property
    def w0(self):
        """
        The dark energy equation of state at a=1 (z=0).
        """
        return self._w0

    @w0.setter
    def w0(self, value):
        self._w0 = value
        logger.debug(
            f"recalculating Ode and update background cosmology due to resetting w0"
        )
        self.get_derived_Ode()
        cosmo = self.set_astropy_cosmo()
        self.cosmo = cosmo.clone(w0=value)

    @property
    def wa(self):
        r"""
        The redshift-dependent part of the dark energy equation of state.
        :math:`w(a) = w_0 + w_a (1 - a)`.
        """
        return self._wa

    @wa.setter
    def wa(self, value):
        self._wa = value
        logger.debug(
            f"recalculating Ode and update background cosmology due to resetting wa"
        )
        self.get_derived_Ode()
        cosmo = self.set_astropy_cosmo()
        self.cosmo = cosmo.clone(wa=value)

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
            f"cleaning cache of {self.cosmo_dep_attr} due to resetting self.cold"
        )
        self.clean_cache(self.cosmo_dep_attr)

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
            f"cleaning cache of {self.cosmo_dep_attr} due to resetting self.backend"
        )
        self.clean_cache(self.cosmo_dep_attr)

    @property
    def omega_de(self):
        """
        The dark energy density fraction at z=0.
        Default is to calculate using camb based on input cosmology,
        but can be manually set (make sure you check consistency yourself).
        """
        if self._omega_de is None:
            self._omega_de = self.get_derived_Ode()
        return self._omega_de

    @omega_de.setter
    def omega_de(self, value):
        if value is None:
            return None
        self._omega_de = value
        cosmo = self.cosmo
        self.cosmo = cosmo.clone(Ode0=value)

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
        """
        logger.debug(
            f"invoking {inspect.currentframe().f_code.co_name} to calculate the average HI brightness temperature"
        )
        logger.debug(
            f"omega_hi: {self.omega_hi_z_mean}, z: {self.z}, cosmo: {self.cosmo}"
        )
        tbar = omega_hi_to_average_temp(self.omega_hi_z_mean, z=self.z, cosmo=self)
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

    @Specification.cosmo.setter
    def cosmo(self, value):
        cosmo = value
        if isinstance(value, str):
            cosmo = extract_astropy_cosmo_set(value)
            As = As_set[value]
            ns = get_ns_from_astropy(value)
            self._As = As
            self._ns = ns

        # update background cosmological parameters
        self._h = cosmo.h
        self._omega_cold = cosmo.Om0
        self._omega_de = cosmo.Ode0
        self._w0 = cosmo.w0
        self._wa = cosmo.wa
        self._neutrino_mass = cosmo.m_nu.value.sum()
        self._omega_baryon = cosmo.Ob0
        self._cosmo = cosmo
        # there is probably a more elegant way of doing this, but I dont know how
        # maybe just inheriting astropy cosmology class?
        for key in cosmo.__dir__():
            if key[0] != "_":
                self.__dict__.update({key: getattr(cosmo, key)})
        # cosmology changed, clear cache
        self.clean_cache(self.cosmo_dep_attr)

    @property
    @tagging("cosmo", "nu")
    def matter_power_spectrum_fnc(self):
        """
        Interpolation function for the real-space isotropic matter power spectrum.
        """
        if self._matter_power_spectrum_fnc is None:
            self.get_matter_power_spectrum()
        return self._matter_power_spectrum_fnc

    def get_matter_power_spectrum(self):
        """
        Calculate the matter power spectrum, interpolate it, and save it into the class attribute `matter_power_spectrum_fnc`.
        """
        kh = self.karr_in_h
        pk = getattr(self, f"get_matter_power_spectrum_{self.backend}")()
        karr = kh * self.h
        pkarr = pk / self.h**3
        matter_power_func = interp1d(
            karr,
            pkarr,
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
        H_z = self.H(self.z)
        delta_r = (delta_z * astropy.constants.c / H_z).to("Mpc").value
        return delta_r

    def deltav_to_deltar(self, delta_v):
        """
        Convert a velocity interval delta_v to a comoving distance interval delta_r.

        Parameters
        ----------
        delta_v: float.
            The velocity interval in km/s.

        Returns
        -------
        delta_r: float.
            The comoving distance interval in Mpc.
        """
        H_z = self.H(self.z)
        delta_r = (1 + self.z) * delta_v / H_z.to("km s^-1 Mpc^-1").value
        return delta_r
