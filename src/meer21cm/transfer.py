"""
This module contains the transfer function analysis class and related functions.

In data analysis, you should have already used :class:`meer21cm.power.PowerSpectrum`
to calculate the power spectrum of the data,
or :class:`meer21cm.mock.MockSimulation` to generate the mock data for power spectrum estimation.
In that case, you can pass the :class:`meer21cm.power.PowerSpectrum` or :class:`meer21cm.mock.MockSimulation`
instance as an input to :class:`meer21cm.transfer.TransferFunction`.
The transfer function class then takes into account of the settings of the power spectrum instance,
and perform mock simulations to estimate the numerical transfer function,
keeping the consistency of the gridding, beam, and other settings.

Finally note that, the transfer function calculation supports parallelisation using MPI,
by using :class:`mpi4py.futures.MPIPoolExecutor`.
The default installation of meer21cm does not include ``mpi4py``,
and you need to configure MPI first (usually with openmpi or mpich), and then install ``mpi4py`` manually.
"""
import numpy as np
from meer21cm.util import pca_clean, dft_matrix, inv_dft_matrix
from meer21cm.mock import MockSimulation
from multiprocessing import Pool
from meer21cm.power import PowerSpectrum
from typing import Callable


def fft_matrix(mat, norm="backward"):
    r"""
    Perform the Fourier transform of a matrix.

    .. math::
        \tilde{M} = \mathcal{F} M \mathcal{F}^{-1}

    where :math:`\mathcal{F}` is the Fourier transform matrix.
    See also :func:`meer21cm.util.dft_matrix`.

    Parameters
    ----------
        mat: np.ndarray
            The matrix to be transformed.
        norm: str, default "backward"
            The normalization of the Fourier transform.

    Returns
    -------
        mat_fft: np.ndarray
            The Fourier transformed matrix.
    """

    return (
        dft_matrix(mat.shape[0], norm=norm)
        @ mat
        @ inv_dft_matrix(mat.shape[0], norm=norm)
    )


def get_pca_matrix(map, N_fg, weights, mean_center_map):
    r"""
    Get the PCA matrix that operates on the map data.

    The eigendecomposition of the map data gives the eigenvectors :math:`v_{i}`,
    which can be used to form the source mixing matrix A so that

    .. math::
        A = \{v_1, v_2, ..., v_{N_fg}\}

    The PCA matrix is then

    .. math::
        R = I - A A^T

    The residual data vector after cleaning is then

    .. math::
        r_{ija} = \sum_{b} R_{ab} m_{ijb}

    where :math:`R_{ab}` is the PCA matrix, and :math:`m_{ijb}` is the uncleaned map data.

    Parameters
    ----------
        map: np.ndarray
            The map data.
        N_fg: int
            The number of foreground components.
        weights: np.ndarray
            The weights of the map data.
        mean_center_map: bool
            Whether to mean center the map data.

    Returns
    -------

    """
    _, A_mat = pca_clean(
        map,
        N_fg,
        weights=weights,
        mean_center=mean_center_map,
        return_A=True,
        ignore_nan=True,
    )
    R_mat = np.eye(map.shape[-1]) - A_mat @ A_mat.T
    return np.nan_to_num(R_mat)


def analytic_transfer_function(clean_mat_1, clean_mat_2=None):
    r"""
    Calculate the analytic transfer function of a clean matrix.
    See Chen 2025 [1] for derivations.

    For a foreground cleaning matrix :math:`R_{ab}`,
    the residual data vector for power spectrum estimation is :math:`r_{ija} = \sum_b R_{ab} m_{ijb}`,
    where i,j are the pixel indices, and a,b are the frequency indices.

    Under the flat sky approximation, the signal loss, as well as the mode-mixing,
    is along the line-of-sight (k_para) direction.
    The unnormalised window function matrix is

    .. math::
        H_{ab} = |\tilde{R}^1_{ab} (\tilde{R}^2_{ab})^*|_{\rm Re}

    The corresponding signal loss is :math:`\sum_b H_{ab}`, and the analytical
    transfer function is the inverse of the signal loss.

    After normalisation, the window function matrix is

    .. math::
        W = {\rm diag}\Big(\sum_b H_{ab}\Big)^{-1} H

    Parameters
    ----------
        clean_mat_1: np.ndarray
            The clean matrix that applies to the data vector.
        clean_mat_2: np.ndarray, optional
            The clean matrix that applies to the second data vector for cross-correlation.
            If not provided, it is assumed to be the same as clean_mat_1 for auto-correlation.

    Returns
    -------
        transfer_func: np.ndarray
            The analytical transfer function.
        Wab: np.ndarray
            The normalised window function matrix.

    References
    ----------
    .. [1] Chen, Z.,, "A quadratic estimator view of the transfer function correction in intensity mapping surveys", https://ui.adsabs.harvard.edu/abs/2025MNRAS.542L...1C/abstract.
    """
    assert (clean_mat_1.ndim == 2) and (clean_mat_1.shape[0] == clean_mat_1.shape[1])
    if clean_mat_2 is None:
        clean_mat_2 = clean_mat_1
    assert np.allclose(clean_mat_1.shape, clean_mat_2.shape)
    num_k = clean_mat_1.shape[0] // 2 + 1
    R_mat_fourier_1 = fft_matrix(clean_mat_1)
    R_mat_fourier_2 = fft_matrix(clean_mat_2)
    Hab = (np.conj(R_mat_fourier_1) * R_mat_fourier_2).real
    Hab = Hab[:num_k, :num_k]
    signal_loss = Hab.sum(1)
    renorm_mat = np.diag(1 / signal_loss)
    Wab = renorm_mat @ Hab
    return 1 / signal_loss, Wab


required_attrs = [
    "wproj",
    "num_pix_x",
    "num_pix_y",
    "nu",
    "data",
    "map_has_sampling",
    "weights_map_pixel",
    "true_cosmology",
    "fiducial_cosmology",
    "sigma_beam_ch",
    "ps_type",
    "cold",
    "backend",
    "omega_hi",
    "tracer_bias_1",
    "tracer_bias_2",
    "sigma_v_1",
    "sigma_v_2",
    "include_beam",
    "fog_profile",
    "weights_field_1",
    "weights_field_2",
    "weights_grid_1",
    "weights_grid_2",
    "mean_amp_1",
    "mean_amp_2",
    "kaiser_rsd",
    "sigma_z_1",
    "sigma_z_2",
    "sampling_resol",
    "box_buffkick",
    "num_particle_per_pixel",
    "mean_center_1",
    "mean_center_2",
    "unitless_1",
    "unitless_2",
    "compensate",
    "taper_func",
    "grid_scheme",
    "interlace_shift",
    "flat_sky",
    "flat_sky_padding",
    "kperpbins",
    "kparabins",
    "k1dbins",
    "include_sky_sampling",
    "k1dweights",
]


class TransferFunction:
    """
    A class to perform transfer function analysis.

    By default, the transfer function is calculated by generating mock HI signal,
    and perform PCA cleaning by injecting the mock HI signal into the data (unless
    ``R_mat`` is provided, in which case no PCA is performed and R_mat is used to clean the injected data).
    The transfer function is then calculated by dividing the 1D power spectrum of the cleaned data
    by the 1D power spectrum of the mock data. A minimum example running 10 realizations:

    .. code-block:: python

        >>> tf = TransferFunction(ps, N_fg=3)
        >>> tf1d_arr = tf.run(range(10), type="auto") # gives you the 1D transfer function with 10 realizations

    If ``type`` is ``"auto"``, the numerator is the **cross-correlation between cleaned HI mock and
    the original HI mock** (and therefore the result is based on HI "auto" power),
    and the denominator is the auto power of the original HI mock.
    If ``type`` is ``"cross"``, the numerator is the cross-correlation between cleaned HI mock
    and galaxy mock, and the denominator is the cross power of the original HI mock and galaxy mock.

    Additionally, if ``type`` is ``"null"``, instead of running transfer function calculation,
    no HI mock is generated, and only mock galaxy data is used to cross-correlate with the map
    data as a null test.

    Parameters
    ----------
        ps: :class:`meer21cm.power.PowerSpectrum`
            The power spectrum instance.
        N_fg: int
            The number of eigenmodes for PCA cleaning.
        R_mat: np.ndarray, default None
            The PCA matrix. If not provided, it will be calculated from the data.
            If provided, no map injection is performed and the cleaning uses this matrix.
        uncleaned_data: np.ndarray, default None
            The uncleaned data, which the mock HI signal is to be injected into.
            If not provided, the mock HI signal is injected into ``ps.data``.
            Note that this can be **wrong** if ``ps.data`` is already overridden by the residual map.
            In that case, you should provide the original map.
        highres_sim: int, default Nones
            The mock field will be gridded to a high resolution sky map,
            with the resolution specified by this integer times the pixel resolution, and then
            down-sampled to the pixel resolution.
        upres_transverse: float, default 4
            The mock field is first generated in a rectangular box.
            This is the up-sampling factor of the box comparing to the pixel resolution in the transverse direction.
        upres_radial: float, default 4
            The up-sampling factor in the radial direction comparing to the frequency resolution.
        mean_center_map: bool, default True
            Whether to mean center the map data for PCA cleaning.
        pca_map_weights: np.ndarray, default None
            If mean centering is performed, this is the weights for the mean calculation.
            Also used to weight the data for covariance calculation for PCA eigendecomposition.
        parallel_plane: bool, default True
            Whether to simulate the mock field in the parallel-plane limit (i.e. k_z = k_parallel).
            Only used if ``rsd_from_field`` is True, otherwise parallel-plane is hard-coded to True.
        rsd_from_field: bool, default False
            Whether to generate the RSD effect at field level. If False, the RSD effect is generated at power spectrum level.
        discrete_source_dndz: function, default np.ones_like
            The redshift distribution of the discrete tracer sources.
            Must be a function of redshift.
            Note that the overall number of discrete sources is set to ``ps.ra_gal.size``, so
            only the shape of the dndz is used, not the normalization.
        pool: str, default "multiprocessing"
            The pool to use for parallelisation. Can be "multiprocessing" or "mpi".
        num_process: int, default None
            The number of processes to use for parallelisation.
            If not provided, the number of processes is set to the number of cores available.
    """

    def __init__(
        self,
        ps: PowerSpectrum,
        N_fg: int,
        R_mat: np.ndarray | None = None,
        uncleaned_data: np.ndarray | None = None,
        highres_sim: int | None = None,
        upres_transverse: float = 4,
        upres_radial: float = 4,
        mean_center_map: bool = True,
        pca_map_weights: np.ndarray | None = None,
        parallel_plane: bool = True,
        rsd_from_field: bool = False,
        discrete_source_dndz: Callable = np.ones_like,
        pool: str = "multiprocessing",
        num_process: int | None = None,
    ):
        self.ps = ps
        self.N_fg = N_fg
        self.highres_sim = highres_sim
        self.upres_transverse = upres_transverse
        self.upres_radial = upres_radial
        self.mean_center_map = mean_center_map
        if pca_map_weights is None:
            pca_map_weights = self.ps.weights_map_pixel
        self.pca_map_weights = pca_map_weights
        self.parallel_plane = parallel_plane
        self.rsd_from_field = rsd_from_field
        self.discrete_source_dndz = discrete_source_dndz
        if uncleaned_data is None:
            uncleaned_data = ps.data
        self.uncleaned_data = uncleaned_data
        self.pool = pool
        self.num_process = num_process
        self.R_mat = R_mat

    def get_mock_instance_attr_dict(self, seed):
        """
        Generate the attribute dictionary for the mock instance.
        It reads the attributes from the input power spectrum instance,
        and then sets a few mock-specific attributes specified in the input parameters
        of :class:`meer21cm.transfer.TransferFunction`.
        The seed attribute is given at the input of this function.

        Parameters
        ----------
            seed: int
                The seed for the mock instance.

        Returns
        -------
            attr_dict: dict
                The attribute dictionary for the mock instance.
        """
        attr_dict = {}
        for attr in required_attrs:
            attr_dict[attr] = getattr(self.ps, attr)
        attr_dict["seed"] = seed
        if hasattr(self.ps, "ra_gal"):
            attr_dict["num_discrete_source"] = self.ps.ra_gal.size
        attr_dict["highres_sim"] = self.highres_sim
        attr_dict["parallel_plane"] = self.parallel_plane
        attr_dict["rsd_from_field"] = self.rsd_from_field
        attr_dict["discrete_source_dndz"] = self.discrete_source_dndz
        attr_dict["downres_factor_radial"] = 1 / self.upres_radial
        attr_dict["downres_factor_transverse"] = 1 / self.upres_transverse
        highres_sim = self.highres_sim if self.highres_sim is not None else 1
        attr_dict["kmax"] = self.ps.kmax * highres_sim
        attr_dict["num_kpoints"] = self.ps.num_kpoints * highres_sim
        return attr_dict

    def get_arg_list_for_parallel_null(
        self,
        seed_list,
        return_power_3d=False,
        return_power_1d=False,
    ):
        """
        Generate a list of arguments for parallelisation of the null test runs.
        This list is then used for ``pool.starmap``.

        Parameters
        ----------
            seed_list: list
                The list of seeds for the mock instances.
            return_power_3d: bool, default False
                Whether to return the 3D power spectrum of the null test.
            return_power_1d: bool, default False
                Whether to return the 1D power spectrum of the null test.
                Note that for null test, the result itself is the 1D power, so this
                is a dummy argument to keep consistency with
                ``cross`` and ``auto`` runs.

        Returns
        -------
            arg_list: list
                The list of arguments for the parallelisation.
        """
        arg_list = []
        for seed in seed_list:
            mock_attr_dict = self.get_mock_instance_attr_dict(seed)
            arg_list.append(
                (
                    mock_attr_dict,
                    self.ps.field_1,
                    self.ps.downres_factor_radial,
                    self.ps.downres_factor_transverse,
                    self.ps.weights_field_1,
                    self.ps.weights_grid_1,
                    self.ps.weights_field_2,
                    self.ps.weights_grid_2,
                    self.ps.k1dweights,
                    return_power_3d,
                )
            )
        return arg_list

    def get_arg_list_for_parallel_cross(
        self, seed_list, return_power_3d=False, return_power_1d=False
    ):
        """
        Generate a list of arguments for parallelisation of the cross-correlation runs
        (i.e. the transfer function is calculated between mock HI and mock galaxy).
        This list is then used for ``pool.starmap``.

        Parameters
        ----------
            seed_list: list
                The list of seeds for the mock instances.
            return_power_3d: bool, default False
                Whether to return the 3D power spectrum of the cross-correlation.
            return_power_1d: bool, default False
                Whether to return the 1D power spectrum of the cross-correlation.

        Returns
        -------
            arg_list: list
                The list of arguments for the parallelisation.
        """
        arg_list = []
        for seed in seed_list:
            mock_attr_dict = self.get_mock_instance_attr_dict(seed)
            arg_list.append(
                (
                    mock_attr_dict,
                    self.ps.downres_factor_radial,
                    self.ps.downres_factor_transverse,
                    self.ps.weights_field_1,
                    self.ps.weights_grid_1,
                    self.ps.weights_field_2,
                    self.ps.weights_grid_2,
                    self.ps.k1dweights,
                    self.N_fg,
                    self.pca_map_weights,
                    self.mean_center_map,
                    self.R_mat,
                    self.uncleaned_data,
                    return_power_3d,
                    return_power_1d,
                )
            )
        return arg_list

    def get_arg_list_for_parallel_auto(
        self, seed_list, return_power_3d=False, return_power_1d=False
    ):
        """
        Generate a list of arguments for parallelisation of the auto-correlation runs
        (i.e. the transfer function is calculated between cleaned mock HI and original mock HI).
        This list is then used for ``pool.starmap``.

        Parameters
        ----------
            seed_list: list
                The list of seeds for the mock instances.
            return_power_3d: bool, default False
                Whether to return the 3D power spectrum of the auto-correlation.
            return_power_1d: bool, default False
                Whether to return the 1D power spectrum of the auto-correlation.

        Returns
        -------
            arg_list: list
                The list of arguments for the parallelisation.
        """
        arg_list = []
        for seed in seed_list:
            mock_attr_dict = self.get_mock_instance_attr_dict(seed)
            arg_list.append(
                (
                    mock_attr_dict,
                    self.ps.downres_factor_radial,
                    self.ps.downres_factor_transverse,
                    self.ps.weights_field_1,
                    self.ps.weights_grid_1,
                    self.ps.k1dweights,
                    self.N_fg,
                    self.pca_map_weights,
                    self.mean_center_map,
                    self.R_mat,
                    self.uncleaned_data,
                    return_power_3d,
                    return_power_1d,
                )
            )
        return arg_list

    def run(
        self, seed_list, type="cross", return_power_3d=False, return_power_1d=False
    ):
        """
        Run the transfer function calculation.

        Note that, ``run`` automatically uses a parallel pool to loop over the ``seed_list``.
        If you believe the parallel behaviour is not as expected,
        you can manually extract the argument list and map the function yourself.
        For example:
        .. code-block:: python

        >>> tf = TransferFunction(ps, N_fg=3)
        >>> tf1d_arr = tf.run(range(10), type="auto")

        is the same as:
        .. code-block:: python

        >>> tf = TransferFunction(ps, N_fg=3)
        >>> arg_list = tf.get_arg_list_for_parallel_auto(range(10))
        >>> tf1d_arr = []
        >>> with Pool(tf.num_process) as pool:
        >>>     for result_i in pool.starmap(run_tf_calculation_auto, arg_list):
        >>>         tf1d_arr.append(result_i)



        Parameters
        ----------
            seed_list: list
                The list of seeds for the mock instances.
            type: str, default "cross"
                The type of transfer function calculation to run.
                Can be "cross", "auto", or "null".
            return_power_3d: bool, default False
                Whether to return the 3D power spectrum of the transfer function.
            return_power_1d: bool, default False
                Whether to return the 1D power spectrum of the transfer function.

        Returns
        -------
            results_arr: list
                The list of results for the transfer function calculation, with
                each element being a sublist for the result of one seed.
                The first element of each sublist is the transfer function.
                If ``return_power_3d`` is True, the second element is the 3D power spectrum of the original mock data,
                and the third element is the 3D power spectrum of the cleaned mock data.
                If ``return_power_1d`` is True, the next element is the 1D power spectrum of the original mock data,
                and the next element after that is the 1D power spectrum of the cleaned mock data.
        """
        if type == "cross":
            run_func = run_tf_calculation_cross
        elif type == "auto":
            run_func = run_tf_calculation_auto
        elif type == "null":
            run_func = run_null_test
        else:
            raise ValueError(f"Invalid type: {type}")
        arg_func = getattr(self, f"get_arg_list_for_parallel_{type}")
        arg_list = arg_func(seed_list, return_power_3d, return_power_1d)
        if self.pool == "multiprocessing":
            pool_func = Pool
        elif self.pool == "mpi":
            from mpi4py.futures import MPIPoolExecutor

            pool_func = MPIPoolExecutor
        else:
            raise ValueError(f"Invalid pool: {self.pool}")
        results_arr = []
        with pool_func(self.num_process) as pool:
            for result_i in pool.starmap(run_func, arg_list):
                results_arr.append(result_i)
        return results_arr


# this must be pickleable inputs for multiprocessing
def run_tf_calculation_cross(
    mock_attr_dict,
    downres_factor_radial,
    downres_factor_transverse,
    weights_field_1,
    weights_grid_1,
    weights_field_2,
    weights_grid_2,
    k_sel_3d_to_1d,
    N_fg,
    pca_map_weights,
    mean_center_map,
    R_mat=None,
    uncleaned_data=0.0,
    return_power_3d=False,
    return_power_1d=False,
):
    """
    Run the transfer function calculation by calculating the ratio of the 1D cross-power spectrum of
    the cleaned mock HI data x mock galaxy data to the 1D cross-power spectrum of the original mock HI data x mock galaxy data.

    Parameters
    ----------
        mock_attr_dict: dict
            The attribute dictionary to initialize the mock instance.
        downres_factor_radial: float
            The downres factor for the radial direction for gridding the map data.
        downres_factor_transverse: float
            The downres factor for the transverse direction for gridding the map data.
        weights_field_1: np.ndarray
            The field-level weights for the field 1.
        weights_grid_1: np.ndarray
            The grid-level weights for the field 1.
        weights_field_2: np.ndarray
            The field-level weights for the field 2.
        weights_grid_2: np.ndarray
            The grid-level weights for the field 2.
        k_sel_3d_to_1d: np.ndarray
            The weights for averaging the 3D power spectrum k-modes to 1D power spectrum.
        N_fg: int
            The number of foreground components to clean.
        pca_map_weights: np.ndarray
            The weights for the mean centering and covariance matrix calculation during PCA.
        mean_center_map: bool
            Whether to mean-center the map before PCA cleaning.
        R_mat: np.ndarray, default None
            The PCA matrix to clean the map.
            If not provided, it will be calculated by injecting the original mock HI data into the data map.
            If provided, it will be used to clean the map, and no injection + PCA is performed.
        uncleaned_data: np.ndarray, default 0.0
            The uncleaned data map.
        return_power_3d: bool, default False
            Whether to return the 3D power spectrum of the mock results, including the uncleaned and cleaned data.
        return_power_1d: bool, default False
            Whether to return the 1D power spectrum of the mock results, including the uncleaned and cleaned data.

    Returns
    -------
        result: list
            The list of results, including the transfer function, the 3D power spectrum of the uncleaned and cleaned data, and the 1D power spectrum of the uncleaned and cleaned data.
    """
    mock = MockSimulation(**mock_attr_dict)
    mock.data = mock.propagate_mock_field_to_data(mock.mock_tracer_field_1)
    mock.propagate_mock_tracer_to_gal_cat()
    mock.downres_factor_radial = downres_factor_radial
    mock.downres_factor_transverse = downres_factor_transverse
    mock.get_enclosing_box()
    mock_map_rg, _, _ = mock.grid_data_to_field()
    mockgal_map_rg, _, _ = mock.grid_gal_to_field()
    mock.field_1 = mock_map_rg
    mock.weights_field_1 = weights_field_1
    mock.weights_grid_1 = weights_grid_1
    mock.field_2 = mockgal_map_rg
    mock.weights_field_2 = weights_field_2
    mock.weights_grid_2 = weights_grid_2
    # get 1d power
    pmock_1d_cross, _, _ = mock.get_1d_power(
        mock.cross_power_3d, k1dweights=k_sel_3d_to_1d
    )
    if return_power_3d:
        power_3d_uncleaned = mock.cross_power_3d
    # perform PCA cleaning
    map_to_clean = uncleaned_data + mock.data
    R_mat_mock = R_mat
    if R_mat is None:
        R_mat_mock = get_pca_matrix(
            map_to_clean, N_fg, pca_map_weights, mean_center_map
        )
    mock_map_cleaned = np.einsum("ij,abj->abi", R_mat_mock, mock.data)
    mock.data = mock_map_cleaned
    # regrid the cleaned data
    mock_map_rg, _, _ = mock.grid_data_to_field()
    mock.field_1 = mock_map_rg
    mock.weights_field_1 = weights_field_1
    mock.weights_grid_1 = weights_grid_1
    # get 1d power again
    pmock_1d_cross_cleaned, _, _ = mock.get_1d_power(
        mock.cross_power_3d, k1dweights=k_sel_3d_to_1d
    )
    tf_1d_i = pmock_1d_cross_cleaned / pmock_1d_cross
    result = [tf_1d_i]
    if return_power_3d:
        result.append(power_3d_uncleaned)
        result.append(mock.cross_power_3d)
    if return_power_1d:
        result.append(pmock_1d_cross)
        result.append(pmock_1d_cross_cleaned)
    return result


def run_tf_calculation_auto(
    mock_attr_dict,
    downres_factor_radial,
    downres_factor_transverse,
    weights_field_1,
    weights_grid_1,
    k_sel_3d_to_1d,
    N_fg,
    pca_map_weights,
    mean_center_map,
    R_mat=None,
    uncleaned_data=0.0,
    return_power_3d=False,
    return_power_1d=False,
):
    """
    Run the transfer function calculation by calculating the ratio between the 1D power spectrum of
    cleaned mock HI data x original mock HI data to the 1D power spectrum of the original mock HI data x original mock HI data.

    Parameters
    ----------
        mock_attr_dict: dict
            The attribute dictionary to initialize the mock instance.
        downres_factor_radial: float
            The downres factor for the radial direction for gridding the map data.
        downres_factor_transverse: float
            The downres factor for the transverse direction for gridding the map data.
        weights_field_1: np.ndarray
            The field-level weights for the field 1.
        weights_grid_1: np.ndarray
            The grid-level weights for the field 1.
        k_sel_3d_to_1d: np.ndarray
            The weights for averaging the 3D power spectrum k-modes to 1D power spectrum.
        N_fg: int
            The number of foreground components to clean.
        pca_map_weights: np.ndarray
            The weights for the mean centering and covariance matrix calculation during PCA.
        mean_center_map: bool
            Whether to mean-center the map before PCA cleaning.
        R_mat: np.ndarray, default None
            The PCA matrix to clean the map.
            If not provided, it will be calculated by injecting the original mock HI data into the data map.
            If provided, it will be used to clean the map, and no injection + PCA is performed.
        uncleaned_data: np.ndarray, default 0.0
            The uncleaned data map.
        return_power_3d: bool, default False
            Whether to return the 3D power spectrum of the mock results, including the uncleaned and cleaned data.
        return_power_1d: bool, default False
            Whether to return the 1D power spectrum of the mock results, including the uncleaned and cleaned data.

    Returns
    -------
        result: list
            The list of results, including the transfer function, the 3D power spectrum of the uncleaned and cleaned data, and the 1D power spectrum of the uncleaned and cleaned data.
    """
    mock = MockSimulation(**mock_attr_dict)
    mock.data = mock.propagate_mock_field_to_data(mock.mock_tracer_field_1)
    mock.downres_factor_radial = downres_factor_radial
    mock.downres_factor_transverse = downres_factor_transverse
    mock.get_enclosing_box()
    mock_map_rg_before, _, _ = mock.grid_data_to_field()
    mock.field_1 = mock_map_rg_before
    mock.weights_field_1 = weights_field_1
    mock.weights_grid_1 = weights_grid_1
    # get 1d power
    pmock_1d_before, _, _ = mock.get_1d_power(
        mock.auto_power_3d_1, k1dweights=k_sel_3d_to_1d
    )
    if return_power_3d:
        power_3d_uncleaned = mock.auto_power_3d_1
    # perform PCA cleaning
    map_to_clean = uncleaned_data + mock.data
    R_mat_mock = R_mat
    if R_mat is None:
        R_mat_mock = get_pca_matrix(
            map_to_clean, N_fg, pca_map_weights, mean_center_map
        )
    mock_map_cleaned = np.einsum("ij,abj->abi", R_mat_mock, mock.data)
    mock.data = mock_map_cleaned
    # regrid the cleaned data
    mock_map_rg_cleaned, _, _ = mock.grid_data_to_field()
    mock.field_1 = mock_map_rg_before
    mock.weights_field_1 = weights_field_1
    mock.weights_grid_1 = weights_grid_1
    mock.field_2 = mock_map_rg_cleaned
    mock.weights_field_2 = weights_field_1
    mock.weights_grid_2 = weights_grid_1
    mock.unitless_2 = False
    mock.mean_center_2 = False

    # get 1d power again
    pmock_1d_cross_cleaned, _, _ = mock.get_1d_power(
        mock.cross_power_3d, k1dweights=k_sel_3d_to_1d
    )
    tf_1d_i = pmock_1d_cross_cleaned / pmock_1d_before
    result = [tf_1d_i]
    if return_power_3d:
        result.append(power_3d_uncleaned)
        result.append(mock.cross_power_3d)
    if return_power_1d:
        result.append(pmock_1d_before)
        result.append(pmock_1d_cross_cleaned)
    return result


def run_null_test(
    mock_attr_dict,
    hi_map_rg,
    downres_factor_radial,
    downres_factor_transverse,
    weights_field_1,
    weights_grid_1,
    weights_field_2,
    weights_grid_2,
    k_sel_3d_to_1d,
    return_power_3d=False,
):
    """
    Run null test realisations by calculating the 1D cross-power spectrum of the mock galaxy x data map.

    Parameters
    ----------
        mock_attr_dict: dict
            The attribute dictionary to initialize the mock instance.
        hi_map_rg: np.ndarray
            The HI data map.
        downres_factor_radial: float
            The downres factor for the radial direction for gridding the map data.
        downres_factor_transverse: float
            The downres factor for the transverse direction for gridding the map data.
        weights_field_1: np.ndarray
            The field-level weights for the field 1.
        weights_grid_1: np.ndarray
            The grid-level weights for the field 1.
        weights_field_2: np.ndarray
            The field-level weights for the field 2.
        weights_grid_2: np.ndarray
            The grid-level weights for the field 2.
        k_sel_3d_to_1d: np.ndarray
            The weights for averaging the 3D power spectrum k-modes to 1D power spectrum.
        return_power_3d: bool, default False
            Whether to return the 3D power spectrum of the mock results.

    Returns
    -------
        result: list
            The list of results, including the 1D cross-power spectrum of the mock galaxy x data map.
            If ``return_power_3d`` is True, the second element is the 3D power spectrum of the mock results.
    """
    mock = MockSimulation(**mock_attr_dict)
    mock.propagate_mock_tracer_to_gal_cat()
    mock.downres_factor_radial = downres_factor_radial
    mock.downres_factor_transverse = downres_factor_transverse
    mock.get_enclosing_box()
    mockgal_map_rg, _, _ = mock.grid_gal_to_field()
    mock.field_1 = hi_map_rg
    mock.weights_field_1 = weights_field_1
    mock.weights_grid_1 = weights_grid_1
    mock.field_2 = mockgal_map_rg
    mock.weights_field_2 = weights_field_2
    mock.weights_grid_2 = weights_grid_2
    pmock_1d_cross, _, _ = mock.get_1d_power(
        mock.cross_power_3d, k1dweights=k_sel_3d_to_1d
    )
    result = [pmock_1d_cross]
    if return_power_3d:
        result.append(mock.cross_power_3d)
    return result
