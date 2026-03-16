"""
This module contains the sampler classes for performing model fitting.

Similar to :mod:`meer21cm.transfer`, sampler class takes in the attributes of a :class:`meer21cm.power.PowerSpectrum` instance,
and perform the model fitting that includes the observational effects, window functions etc defined in the PS instance.
The difference is that, the sampler class takes the dictionary of attributes instead of the PS itself,
so you should always use :func:`meer21cm.inference.extract_model_fitting_inputs` to extract the attributes from the PS instance,
and use the dictionary to initialize the sampler class. For example:

.. code-block:: python

    >>> from meer21cm.inference import extract_model_fitting_inputs, SamplerEmcee
    >>> from meer21cm.power import PowerSpectrum
    >>> # the ps instance is usually something you have already defined and used to read data
    >>> # or it can be a MockSimulation instance you have used to generate the mock
    >>> ps = PowerSpectrum(band="L", survey="meerklass_2021", sigma_v_1=100, tracer_bias_1=1.5, tracer_bias_2=2.0)
    >>> ps_dict = extract_model_fitting_inputs(ps)
    >>> sampler = SamplerEmcee(ps_dict, ...)

similarily, you can also use :class:`meer21cm.inference.SamplerNautilus` to perform model fitting.

Caution:

1. ``data_vector``, ``data_covariance`` and ``observables`` must be in the same order. For example, if ``observables`` is ["cross", "hi-auto"], then the first half of ``data_vector`` should be the cross-power and second half should be the hi-auto power. Similarly, the [:half, :half] of ``data_covariance`` should be the cross-power covariance and the [half:, half:] should be the hi-auto power covariance (the rest of the matrix should be the cross-correlation between cross and auto powers).
2. ``params_name`` must be in the same order as the ``params_prior``.
"""

import numpy as np
from meer21cm.power import PowerSpectrum
from meer21cm.transfer import required_attrs
import emcee
import os
import nautilus
from multiprocessing import Pool
from scipy.stats import norm


def extract_model_fitting_inputs(
    ps: PowerSpectrum,
):
    """
    Extract a dictionary of model fitting inputs from a
    :class:`meer21cm.power.PowerSpectrum` instance.

    Parameters
    ----------
    ps : :class:`meer21cm.power.PowerSpectrum`
        A :class:`meer21cm.power.PowerSpectrum` instance.

    Returns
    -------
    ps_dict : dict
        A dictionary of model fitting inputs.
    """
    attr_dict = {}
    for attr in required_attrs:
        attr_dict[attr] = getattr(ps, attr)
    attr_dict["box_len"] = ps.box_len
    attr_dict["box_ndim"] = ps.box_ndim
    return attr_dict


class SamplerBase:
    """
    Base class for all samplers.

    Parameters
    ----------
    ps_dict : dict
        A dictionary of model fitting inputs.
    data_vector : np.ndarray
        The data vector.
    data_covariance : np.ndarray
        The data covariance matrix.
    params_name : list[str]
        The names of the parameters.
        Must match the attribute names of the :class:`meer21cm.power.PowerSpectrum` instance.
    params_prior : list[tuple[str, float, float]]
        The prior distribution of the parameters.
        The first element of the tuple is the distribution type, which can be "uniform" or "gaussian".
        The second and third elements are the parameters of the distribution.
        For uniform distribution, the second and third elements are the lower and upper bounds of the distribution.
        For gaussian distribution, the second and third elements are the mean and standard deviation of the distribution.
    observables : list[str], default ["cross"]
        The observables to fit.
        Must be a list of "cross", "hi-auto", or "gg-auto".
    save : bool, default False
        Whether to save the fitting results while running the sampler.
    save_filename : str | None, default None
        The filename to save the fitting results when ``save`` is True.
        If not provided, the results will not be saved.
    save_model_blobs : bool, default False
        Whether to save the model blobs.
        The model blobs are the model vectors listed in the ``observables`` argument.
    do_hartlap_correction : bool, default False
        Whether to apply the Hartlap correction to the data covariance matrix.
    num_mocks : int, default None
        The number of mock data vectors to use for the Hartlap correction.
        If not provided, the Hartlap correction will not be applied.
    do_percival_correction : bool, default False
        Whether to apply the Percival correction to the data covariance matrix.
    """

    def __init__(
        self,
        ps_dict: dict,
        data_vector: np.ndarray,
        data_covariance: np.ndarray,
        params_name: list[str],
        params_prior: list[tuple[str, float, float]],
        observables: list[str] = ["cross"],
        save: bool = False,
        save_filename: str | None = None,
        save_model_blobs: bool = False,
        do_hartlap_correction: bool = False,
        num_mocks: int | None = None,
        do_percival_correction: bool = False,
    ):
        self.ps_dict = ps_dict
        self.params_name = params_name
        self.params_prior = params_prior
        self.save = save
        self.save_filename = save_filename
        self.observables = observables
        self.data_vector = data_vector
        self.data_covariance = data_covariance
        self._inverse_covariance = None
        self.save_model_blobs = save_model_blobs
        self.do_hartlap_correction = do_hartlap_correction
        self.num_mocks = num_mocks
        self.do_percival_correction = do_percival_correction
        self.validate_input()

    @property
    def do_hartlap_correction(self) -> bool:
        """
        Whether to apply the Hartlap correction to the data covariance matrix.
        """
        return self._do_hartlap_correction

    @do_hartlap_correction.setter
    def do_hartlap_correction(self, value: bool):
        self._do_hartlap_correction = value
        self._inverse_covariance = None

    @property
    def do_percival_correction(self) -> bool:
        """
        Whether to apply the Percival correction to the data covariance matrix.
        """
        return self._do_percival_correction

    @do_percival_correction.setter
    def do_percival_correction(self, value: bool):
        self._do_percival_correction = value
        self._inverse_covariance = None

    @property
    def num_mocks(self) -> int | None:
        """
        The number of mock data vectors to use for the Hartlap and Percival corrections.
        """
        return self._num_mocks

    @num_mocks.setter
    def num_mocks(self, value: int | None):
        self._num_mocks = value
        self._inverse_covariance = None

    @property
    def hartlap_factor(self) -> float:
        r"""
        The Hartlap factor.
        The Hartlap correction is given by [1]_ :

        .. math::
            \hat{\Sigma^{-1}} = \frac{N_m - N_d - 2}{N_m - 1} \Sigma^{-1}
        where :math:`N_m` is the number of mock realisations for calculating the covariance, :math:`N_d` is the number of data points,
        and :math:`\Sigma` is the mock-based covariance matrix.

        References
        ----------
        .. [1] Hartlap, J., et al. 2007, A&A, astro-ph/0608064
        """
        if self.do_hartlap_correction and self.num_mocks is not None:
            return (self.num_mocks - self.data_vector.size - 2) / (self.num_mocks - 1)
        return 1.0

    @property
    def percival_factor(self) -> float:
        r"""
        The Percival factor.
        The Percival correction is given by [1]_ :

        .. math::
            \hat{\Sigma^{-1}} = \frac{1 + B(N_d - N_p)}{1 + A + B(N_p + 1)} \Sigma^{-1}

        .. math::
            A = \frac{2}{(N_m - N_d - 1) (N_m - N_d - 4)}

        .. math::
            B = \frac{N_m - N_d - 2}{(N_m - N_d - 1)(N_m - N_d - 4)}

        where :math:`N_d` is the number of data points,
        :math:`N_p` is the number of parameters,
        :math:`N_m` is the number of mock realisations for calculating the covariance,
        and :math:`\Sigma` is the mock-based covariance matrix.

        References
        ----------
        .. [1] Percival, W. J. et al. 2014, MNRAS, 1312.4841
        """
        if self.do_percival_correction and self.num_mocks is not None:
            N_d = self.data_vector.size
            N_p = len(self.params_name)
            N_m = self.num_mocks
            A = 2 / ((N_m - N_d - 1) * (N_m - N_d - 4))
            B = (N_m - N_d - 2) / ((N_m - N_d - 1) * (N_m - N_d - 4))
            return (1 + B * (N_d - N_p)) / (1 + A + B * (N_p + 1))
        return 1.0

    @property
    def inverse_covariance(self) -> np.ndarray:
        """
        The matrix inverse of the data covariance.
        """
        if self._inverse_covariance is None:
            self._inverse_covariance = np.linalg.inv(self.data_covariance)
            self._inverse_covariance = (
                self.hartlap_factor * self.percival_factor * self._inverse_covariance
            )
        return self._inverse_covariance

    @property
    def data_covariance(self) -> np.ndarray:
        """
        The data covariance matrix.
        """
        return self._data_covariance

    @data_covariance.setter
    def data_covariance(self, data_covariance: np.ndarray):
        self._data_covariance = data_covariance
        self._inverse_covariance = None

    def validate_input(self):
        """
        Validate the input parameters.
        """
        if not len(self.params_name) == len(self.params_prior):
            raise ValueError("params_names and params_prior must have the same length")
        data_len = self.data_vector.size
        if self.data_covariance.shape != (data_len, data_len):
            raise ValueError(
                "data_covariance must be a square matrix with the same length as data_vector"
            )
        if not all(obs in ["cross", "hi-auto", "gg-auto"] for obs in self.observables):
            raise ValueError(
                "observables must be a list of 'cross', 'hi-auto', or 'gg-auto'"
            )

    def get_model_instance(self, params: list[float] | np.ndarray) -> PowerSpectrum:
        """
        Get the model instance from the input ``ps_dict`` and the updated parameters ``params``.

        Parameters
        ----------
        params : list[float] | np.ndarray
            The updated values of the parameters defined in the ``params_name`` argument.

        Returns
        -------
        model_instance : :class:`meer21cm.power.PowerSpectrum`
            The model PS instance to calculate the model vector.
        """
        model_instance = PowerSpectrum(**self.ps_dict)
        model_instance._box_len = self.ps_dict["box_len"]
        model_instance._box_ndim = self.ps_dict["box_ndim"]
        model_instance.propagate_field_k_to_model()
        for param_name, param_value in zip(self.params_name, params):
            setattr(model_instance, param_name, param_value)
        return model_instance

    def get_model_vector(self, params: list[float] | np.ndarray) -> np.ndarray:
        """
        Calculate the model vector from the input parameter values.

        Parameters
        ----------
        params : list[float] | np.ndarray
            The updated values of the parameters defined in the ``params_name`` argument.

        Returns
        -------
        model_vector : np.ndarray
            The model vector.
            The shape is (len(observables), len(k1dbins) - 1).
        """
        model_instance = self.get_model_instance(params)
        model_vector = np.zeros(
            (len(self.observables), len(self.ps_dict["k1dbins"]) - 1)
        )
        i = 0
        for obs in self.observables:
            if obs == "cross":
                model_vector[i], _, _ = model_instance.get_1d_power(
                    model_instance.cross_power_tracer_model,
                    k1dweights=model_instance.k1dweights,
                )
            elif obs == "hi-auto":
                model_vector[i], _, _ = model_instance.get_1d_power(
                    model_instance.auto_power_tracer_1_model,
                    k1dweights=model_instance.k1dweights,
                )
            elif obs == "gg-auto":
                model_vector[i], _, _ = model_instance.get_1d_power(
                    model_instance.auto_power_tracer_2_model,
                    k1dweights=model_instance.k1dweights,
                )
            i += 1
        return model_vector

    def compute_log_likelihood(
        self, params: list[float] | np.ndarray
    ) -> float | tuple[float, np.ndarray]:
        """
        Calculate the log likelihood from the input parameter values.

        Parameters
        ----------
        params : list[float] | np.ndarray
            The updated values of the parameters defined in the ``params_name`` argument.

        Returns
        -------
        log_likelihood : float
            The log likelihood.
        """
        model_vector = self.get_model_vector(params)
        model_minus_data = model_vector.ravel() - self.data_vector.ravel()
        log_likelihood = (
            -0.5 * model_minus_data @ self.inverse_covariance @ model_minus_data
        )
        if self.save_model_blobs:
            return log_likelihood, model_vector
        return log_likelihood


class SamplerEmcee(SamplerBase):
    """
    A sampler class to perform model fitting using the ``emcee`` sampler.

    Parameters
    ----------
    ps_dict : dict
        A dictionary of model fitting inputs.
    data_vector : np.ndarray
        The data vector.
    data_covariance : np.ndarray
        The data covariance matrix.
    params_name : list[str]
        The names of the parameters.
        Must match the attribute names of the :class:`meer21cm.power.PowerSpectrum` instance.
    params_prior : list[tuple[str, float, float]]
        The prior distribution of the parameters.
        The first element of the tuple is the distribution type, which can be "uniform" or "gaussian".
        The second and third elements are the parameters of the distribution.
        For uniform distribution, the second and third elements are the lower and upper bounds of the distribution.
        For gaussian distribution, the second and third elements are the mean and standard deviation of the distribution.
    nwalkers : int
        The number of random walkers.
    nsteps : int
        The number of sampling steps.
    nthreads : int
        The number of parallel threads.
    observables : list[str], default ["cross"]
        The observables to fit.
        Must be a list of "cross", "hi-auto", or "gg-auto".
    mp_backend : str, default "multiprocessing"
        The backend to use for the multiprocessing.
        Can be "multiprocessing" or "mpi".
    save : bool, default False
        Whether to save the fitting results while running the sampler.
    save_filename : str | None, default None
        The filename to save the fitting results when ``save`` is True.
        If not provided, the results will not be saved.
    save_model_blobs : bool, default False
        Whether to save the model blobs.
        The model blobs are the model vectors listed in the ``observables`` argument.
    do_hartlap_correction : bool, default False
        Whether to apply the Hartlap correction to the data covariance matrix.
    num_mocks : int, default None
        The number of mock data vectors to use for the Hartlap correction.
        If not provided, the Hartlap correction will not be applied.
    do_percival_correction : bool, default False
        Whether to apply the Percival correction to the data covariance matrix.
    """

    def __init__(
        self,
        ps_dict: dict,
        data_vector: np.ndarray,
        data_covariance: np.ndarray,
        params_name: list[str],
        params_prior: list[tuple[str, float, float]],
        nwalkers: int,
        nsteps: int,
        nthreads: int,
        observables: list[str] = ["cross"],
        mp_backend: str = "multiprocessing",
        save: bool = False,
        save_filename: str | None = None,
        save_model_blobs: bool = False,
        do_hartlap_correction: bool = False,
        num_mocks: int | None = None,
        do_percival_correction: bool = False,
    ):
        self.nwalkers = nwalkers
        self.nsteps = nsteps
        self.nthreads = nthreads
        self.mp_backend = mp_backend
        super().__init__(
            ps_dict,
            data_vector,
            data_covariance,
            params_name,
            params_prior,
            observables,
            save,
            save_filename,
            save_model_blobs,
            do_hartlap_correction,
            num_mocks,
            do_percival_correction,
        )

    def log_prior_gaussian(self, value, mean, sigma):
        """
        Calculate the log prior for a gaussian distribution.

        Parameters
        ----------
        value : float
            The value to calculate the log prior for.
        mean : float
            The mean of the distribution.
        sigma : float
            The standard deviation of the distribution.

        Returns
        -------
        log_prior : float
            The log prior.
        """
        return -0.5 * (value - mean) ** 2 / sigma**2

    def log_prior_uniform(self, value, low, high):
        """
        Calculate the log prior for a flat (uniform) distribution.

        Parameters
        ----------
        value : float
            The value to calculate the log prior for.
        low : float
            The lower bound of the distribution.
        high : float
            The upper bound of the distribution.

        Returns
        -------
        log_prior : float
            The log prior.
        """
        if value < low or value > high:
            return -np.inf
        else:
            return 0.0

    def log_prior(self, params_values):
        """
        Calculate the log prior from the input parameter values.

        Parameters
        ----------
        params_values : list[float] | np.ndarray
            The updated values of the parameters defined in the ``params_name`` argument.

        Returns
        -------
        log_prior : float
            The log prior.
        """
        log_prior = 0.0
        for i, param_name in enumerate(self.params_name):
            prior_func = getattr(self, f"log_prior_{self.params_prior[i][0]}")
            log_prior_i = prior_func(
                params_values[i],
                self.params_prior[i][1],
                self.params_prior[i][2],
            )
            log_prior += log_prior_i
        return log_prior

    def log_likelihood(self, params_values):
        """
        Calculate the log likelihood plus the log prior from the input parameter values.

        Parameters
        ----------
        params_values : list[float] | np.ndarray
            The updated values of the parameters defined in the ``params_name`` argument.

        Returns
        -------
        log_likelihood : float
            The log likelihood plus the log prior.
        model_vector : np.ndarray, optional
            The model vector.
            Only returned when ``save_model_blobs`` is True.
        """
        lp = self.log_prior(params_values)
        if not np.isfinite(lp):
            if self.save_model_blobs:
                return lp, np.zeros(
                    (len(self.observables), (len(self.ps_dict["k1dbins"]) - 1))
                )
            return lp
        else:
            if self.save_model_blobs:
                ll, model_vector = self.compute_log_likelihood(params_values)
                return lp + ll, model_vector
            return lp + self.compute_log_likelihood(params_values)

    @property
    def ndim(self):
        """
        The number of parameters to fit.
        """
        return len(self.params_name)

    def run(
        self,
        resume: bool,
        progress: bool = True,
        start_coord: np.ndarray | None = None,
        run_sampler: bool = True,
    ):
        """
        Run the sampler.

        Parameters
        ----------
        resume : bool
            Whether to resume the sampler from the last saved state.
            If ``resume`` is False and the save file already exists,
            **the sampler will reset the save file**.
        progress : bool, default True
            Whether to show the progress bar.
        start_coord : np.ndarray | None, default None
            The initial coordinates of the walkers.
            If not provided, the sampler will randomly initialize the walkers.
        run_sampler : bool, default True
            Whether to run the sampler.
            If False, the sampler will only initialize the walkers and return the sampler object.

        Returns
        -------
        sampler : emcee.EnsembleSampler
            The sampler object.
        """
        if resume and self.save:
            start_coord = None
        else:
            # TODO: need smarter auto init position
            if start_coord is None:
                init_pos = np.array(
                    [self.ps_dict[param_name] for param_name in self.params_name]
                )
                # TODO: need smarter auto init position
                start_coord = (
                    1 + np.random.uniform(-1e-2, 1e-2, size=(self.nwalkers, self.ndim))
                ) * init_pos[None, :] + np.random.uniform(
                    0, 1e-2, size=(self.nwalkers, self.ndim)
                ) * (
                    np.abs(init_pos[None, :]) < 1e-2
                )
        nsteps = self.nsteps
        if self.save:
            backend = emcee.backends.HDFBackend(self.save_filename)
            if os.path.isfile(self.save_filename):
                if not resume and backend.iteration > 0:
                    backend.reset(self.nwalkers, self.ndim)
                nsteps = nsteps - backend.iteration
        else:
            backend = None
        pool = None
        if self.mp_backend == "multiprocessing":
            pool_func = Pool
        elif self.mp_backend == "mpi":
            from mpi4py.futures import MPIPoolExecutor

            pool_func = MPIPoolExecutor
        with pool_func(self.nthreads) as pool:
            sampler = emcee.EnsembleSampler(
                self.nwalkers,
                self.ndim,
                self.log_likelihood,
                backend=backend,
                pool=pool,
            )
            if run_sampler:
                sampler.run_mcmc(start_coord, nsteps, progress=progress)
        return sampler

    def get_backend(self) -> emcee.backends.HDFBackend:
        """
        Get the backend of the sampler.
        """
        if self.save:
            return emcee.backends.HDFBackend(self.save_filename)
        else:
            raise ValueError("No save filename provided")

    def get_points(self, sampler: emcee.EnsembleSampler | None = None) -> np.ndarray:
        """
        Get the chain from the sampler.

        Parameters
        ----------
        sampler : emcee.EnsembleSampler | None, default None
            The sampler object.
            If not provided, the sampler will use the backend to get the chain
            (only works if ``save`` is True).

        Returns
        -------
        chain : np.ndarray
            The chain.
            The shape is (nwalkers, nsteps, ndim).
        """
        if sampler is None:
            return self.get_backend().get_chain()
        return sampler.get_chain()

    def get_blobs(self, sampler: emcee.EnsembleSampler | None = None) -> np.ndarray:
        """
        Get the blobs from the sampler.

        Parameters
        ----------
        sampler : emcee.EnsembleSampler | None, default None
            The sampler object.
            If not provided, the sampler will use the backend to get the blobs
            (only works if ``save`` is True).

        Returns
        -------
        blobs : np.ndarray
            The model vectorblobs.
        """
        if sampler is None:
            return self.get_backend().get_blobs()
        return sampler.get_blobs()

    def get_log_prob(self, sampler: emcee.EnsembleSampler | None = None) -> np.ndarray:
        """
        Get the log probability from the sampler.

        Parameters
        ----------
        sampler : emcee.EnsembleSampler | None, default None
            The sampler object.
            If not provided, the sampler will use the backend to get the log probability
            (only works if ``save`` is True).

        Returns
        -------
        log_prob : np.ndarray
            The log probability.
            The shape is (nwalkers, nsteps).
        """
        if sampler is None:
            return self.get_backend().get_log_prob()
        return sampler.get_log_prob()


class SamplerNautilus(SamplerBase):
    """
    A sampler class to perform model fitting using the ``nautilus`` sampler.

    Parameters
    ----------
    ps_dict : dict
        A dictionary of model fitting inputs.
    data_vector : np.ndarray
        The data vector.
    data_covariance : np.ndarray
        The data covariance matrix.
    params_name : list[str]
        The names of the parameters.
        Must match the attribute names of the :class:`meer21cm.power.PowerSpectrum` instance.
    params_prior : list[tuple[str, float, float]]
        The prior distribution of the parameters.
        The first element of the tuple is the distribution type, which can be "uniform" or "gaussian".
        The second and third elements are the parameters of the distribution.
        For uniform distribution, the second and third elements are the lower and upper bounds of the distribution.
        For gaussian distribution, the second and third elements are the mean and standard deviation of the distribution.
    observables : list[str], default ["cross"]
        The observables to fit.
        Must be a list of "cross", "hi-auto", or "gg-auto".
    n_live_points : int, default 2000
        The number of live points.
    f_live : float, default 0.01
        The fraction of live points to keep.
    n_shell : int, default 1
        The number of shells to use.
    n_eff : int, default 10000
        The effective number of samples.
    save : bool, default False
        Whether to save the fitting results while running the sampler.
    save_filename : str | None, default None
        The filename to save the fitting results when ``save`` is True.
        If not provided, the results will not be saved.
    mp_backend : str, default "multiprocessing"
        The backend to use for the multiprocessing.
        Can be "multiprocessing" or "mpi".
    nthreads : int, default 1
        The number of parallel threads.
    timeout : float, default np.inf
        The timeout for the sampler.
        If the sampler runs longer than the timeout, it will be terminated.
    save_model_blobs : bool, default False
        Whether to save the model blobs.
        The model blobs are the model vectors listed in the ``observables`` argument.
    hartlap_correction : bool, default False
        Whether to apply the Hartlap correction to the data covariance matrix.
    num_mocks : int, default None
        The number of mock data vectors to use for the Hartlap correction.
        If not provided, the Hartlap correction will not be applied.
    percival_correction : bool, default False
        Whether to apply the Percival correction to the data covariance matrix.
    """

    def __init__(
        self,
        ps_dict: dict,
        data_vector: np.ndarray,
        data_covariance: np.ndarray,
        params_name: list[str],
        params_prior: list[tuple[str, float, float]],
        observables: list[str] = ["cross"],
        n_live_points: int = 2000,
        f_live: float = 0.01,
        n_shell: int = 1,
        n_eff: int = 10000,
        save: bool = False,
        save_filename: str | None = None,
        mp_backend: str = "multiprocessing",
        nthreads: int = 1,
        timeout: float = np.inf,
        save_model_blobs: bool = False,
        do_hartlap_correction: bool = False,
        num_mocks: int | None = None,
        do_percival_correction: bool = False,
    ):
        self.n_live_points = n_live_points
        self.f_live = f_live
        self.n_shell = n_shell
        self.n_eff = n_eff
        self.mp_backend = mp_backend
        self.nthreads = nthreads
        self.timeout = timeout
        super().__init__(
            ps_dict,
            data_vector,
            data_covariance,
            params_name,
            params_prior,
            observables,
            save,
            save_filename,
            save_model_blobs,
            do_hartlap_correction,
            num_mocks,
            do_percival_correction,
        )

    def get_nautilus_prior(self):
        """
        Generate the ``nautilus.Prior`` objects from the input parameters.
        """
        prior = nautilus.Prior()
        for i, param_name in enumerate(self.params_name):
            if self.params_prior[i][0] == "uniform":
                dist = (self.params_prior[i][1], self.params_prior[i][2])
            elif self.params_prior[i][0] == "gaussian":
                dist = norm(loc=self.params_prior[i][1], scale=self.params_prior[i][2])
            prior.add_parameter(param_name, dist=dist)
        return prior

    def run(self, resume: bool, progress: bool = True, run_sampler: bool = True):
        """
        Run the sampler.

        Parameters
        ----------
        resume : bool
            Whether to resume the sampler from the last saved state.
        progress : bool, default True
            Whether to show the progress bar.
        run_sampler : bool, default True
            Whether to run the sampler.
            If False, the sampler will only initialize the walkers and return the sampler object.

        Returns
        -------
        sampler : nautilus.Sampler
            The sampler object.
        """
        if not self.save:
            resume = False
        if self.mp_backend == "multiprocessing":
            pool_func = Pool
        elif self.mp_backend == "mpi":
            from mpi4py.futures import MPIPoolExecutor

            pool_func = MPIPoolExecutor
        with pool_func(self.nthreads) as pool:
            sampler = nautilus.Sampler(
                self.get_nautilus_prior(),
                self.compute_log_likelihood,
                pass_dict=False,
                pool=pool,
                n_live=self.n_live_points,
                resume=resume,
                filepath=self.save_filename,
            )
            # nautilus does not timeout properly, so no test coverage for this
            if run_sampler:  # pragma: no cover
                sampler.run(
                    f_live=self.f_live,
                    n_shell=self.n_shell,
                    n_eff=self.n_eff,
                    discard_exploration=True,
                    verbose=progress,
                    timeout=self.timeout,
                )
        return sampler

    def get_posterior(self, sampler: nautilus.Sampler | None = None):
        """
        Get the posterior from the sampler.

        Parameters
        ----------
        sampler : nautilus.Sampler | None, default None
            The sampler object.
            If not provided, the sampler will use the backend to get the posterior
            (only works if ``save`` is True).

        Returns
        -------
        posterior : tuple
            The posterior.
            The first element is the points.
            The second element is the log weights.
            The third element is the log likelihoods.
            If ``save_model_blobs`` is True, the fourth element is the model blobs.
        """
        if sampler is None:
            sampler = nautilus.Sampler(
                self.get_nautilus_prior(),
                self.compute_log_likelihood,
                pass_dict=False,
                filepath=self.save_filename,
            )
        return sampler.posterior(return_blobs=self.save_model_blobs)
