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
from .util import pca_clean, dft_matrix, inv_dft_matrix
from .mock import MockSimulation
from multiprocessing import Pool
from .power import PowerSpectrum
from typing import Callable
from scipy.interpolate import interp1d
import pickle


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
    "batch_number",
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
        discrete_source_dndz: list or function, default np.ones_like
            The redshift distribution of the discrete tracer sources.
            Must be a function of redshift.
            Note that the overall number of discrete sources is set to ``ps.ra_gal.size``
            or ``ps.num_discrete_source``, so
            only the shape of the dndz is used, not the normalization.
            If a list is provided, the first element is the redshift array, and the second element is the dndz array,
            and a 1D linear interpolation is used to get the dndz function.
        pool: str, default "multiprocessing"
            The pool to use for parallelisation. Can be "multiprocessing" or "mpi".
        num_process: int, default None
            The number of processes to use for parallelisation.
            If not provided, the number of processes is set to the number of cores available.
        unmask_during_mock: bool, default False
            Whether to unmask the wcs during the mock simulation.
            If True, the survey area will be unmasked during the mock simulation, and then masked again after mock HI is generated.
            This is useful for mitigating edge effects and when box buffer is small.
            For a large patch with a wide z-range, it is recommended to set this to True,
            otherwise the mock HI map may contain nan and cause errors.
        mock_directory: string, default None
            Path prefix for precomputed mocks. The workers concatenate
            ``{hi_stem}{seed}.pkl`` and ``{gal_stem}{seed}.fits`` onto this
            string. A directory path must end with the path separator.
            ``None`` generates each mock inside the worker.
        project_combined: bool, default True
            If True, the PCA matrix (found from ``uncleaned_data + mock``) is
            applied to ``uncleaned_data + mock``. If False it is applied to the
            mock only. The two differ by the cross-power between the mock tracer
            and ``uncleaned_data`` cleaned with that matrix. Because the matrix
            depends on the injected mock, this term does not average to zero:
            it is extra signal loss that the real measurement also contains, so
            ``False`` underestimates the signal loss. ``False`` is kept for
            comparison and is expected to be removed. With mean-centering on,
            the weighted frequency mean is removed from the cube that is
            projected.
        subtract_cleaned_data: bool, default True
            Has no effect with ``project_combined=False``. If True, ``uncleaned_data``
            cleaned with its own PCA matrix (found from ``uncleaned_data``
            alone, or ``R_mat`` when that is given) is subtracted from each
            cleaned ``uncleaned_data + mock`` map before the power spectrum is
            taken. That map does not depend on the mock, so the mean transfer
            function is unchanged and its mock-to-mock scatter is reduced. The
            scatter then no longer contains the data's own contribution, so it
            is not an error estimate for the data; set this to False to use the
            scatter as an error. ``mock_dilution`` is divided out of the
            difference.
        convolve_loaded_mock: bool, default False
            If True, convolve a cube loaded from ``mock_directory`` with a
            Gaussian primary beam using ``sigma_beam_ch``. Has no effect when
            the mock is generated inside the worker.
        construct_fkp_mock: bool, default False
            If True, the ``type="cross"`` worker builds an FKP field for the
            mock galaxies instead of painting raw counts: ``F_mock = D_mock -
            alpha_mock * R``, with ``alpha_mock = sum(D_mock) / sum(R)``
            recomputed per mock from that mock's own painted counts, and
            ``R`` taken from ``ps.field_2_R`` (the real survey's randoms,
            already painted once by ``ps.grid_gal_to_field(construct_fkp=
            True)`` -- not repainted per mock). Requires
            ``ps.field_2_has_random`` to be True (i.e. the real data side was
            itself built with ``construct_fkp=True``), otherwise
            ``ps.field_2_R`` is undefined and this raises. ``ps.weights_grid_2``
            is read as a pure taper/footprint template with no FKP amplitude
            in it -- true for the standard real-data sequence, where
            ``construct_fkp=True`` sets it to all-ones and any taper applied
            afterwards multiplies it directly, never by ``alpha * R``. The
            mock's weights are then ``weights_field_2 = alpha_mock * R *
            template``, ``weights_grid_2 = template``. Without this, the
            worker always stamps the parent's ``weights_field_2``/
            ``weights_grid_2`` onto the mock's raw painted counts, which is
            not an FKP field for that mock (see the ``tf_wmocks`` handoff
            notes).
        mock_dilution: np.ndarray, default None
            Optional multiplicative array, same shape as the gridded HI
            field, applied to the mock HI signal *after* beam convolution
            (``convolve_loaded_mock``) but before it is injected into
            ``uncleaned_data``. Intended for ``hit / hit_nomask`` (<=1):
            this applies that same dilution to the mock signal,
            so the injection step sees what a real HI signal would
            have looked like after block combination. The transfer function
            denominator (the mock's own, "original" power) is taken
            *before* this dilution, so the TF still measures recovery of
            the true, undiluted signal. After PCA cleaning, the cleaned
            mock is divided back by ``mock_dilution`` (i.e. multiplied by
            ``hit_nomask / hit``), mirroring the rescale applied to the
            real cleaned data. ``None`` (default) leaves the mock
            undiluted, unchanged from prior behaviour.
        gal_cat: str, default None
            Tracer name, for example ``"LRG"``. The galaxy file is then
            ``{mock_directory}gal_{gal_cat}_mock_{seed}.fits``. ``None`` uses
            ``gal_stem``. Has no effect when the mock is generated inside the
            worker (``mock_directory=None``).
        hi_stem: str, default "HI_mock_"
            Filename text between ``mock_directory`` and the seed for the HI pickle.
        gal_stem: str, default "gal_mock_"
            Filename text between ``mock_directory`` and the seed for the galaxy
            FITS table when ``gal_cat`` is ``None``. The auto-power worker does
            not read that table.

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
        discrete_source_dndz: list | Callable = np.ones_like,
        pool: str = "multiprocessing",
        num_process: int | None = None,
        unmask_during_mock: bool = False,
        mock_directory: str | None = None,
        project_combined: bool = True,
        convolve_loaded_mock: bool = False,
        construct_fkp_mock: bool = False,
        mock_dilution: np.ndarray | None = None,
        gal_cat: str | None = None,
        hi_stem: str = "HI_mock_",
        gal_stem: str = "gal_mock_",
        subtract_cleaned_data: bool = True,
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
            uncleaned_data = np.zeros_like(ps.data)
        self.uncleaned_data = uncleaned_data
        self.pool = pool
        self.num_process = num_process
        self.R_mat = R_mat
        self.unmask_during_mock = unmask_during_mock
        self.mock_directory = mock_directory
        self.project_combined = project_combined
        self.convolve_loaded_mock = convolve_loaded_mock
        self.construct_fkp_mock = construct_fkp_mock
        self.mock_dilution = mock_dilution
        self.gal_cat = gal_cat
        self.hi_stem = hi_stem
        self.gal_stem = gal_stem
        self.subtract_cleaned_data = subtract_cleaned_data

    def get_cleaned_data(self):
        """
        ``uncleaned_data`` cleaned with its own PCA matrix, for
        ``subtract_cleaned_data``.

        The matrix is ``R_mat`` when that is given, otherwise it is found from
        ``uncleaned_data`` alone with the same ``N_fg``, weights and
        mean-centering as the transfer function. No rescaling by
        ``mock_dilution`` is applied here.

        Returns
        -------
            cleaned_data: np.ndarray or None
                The cleaned cube. ``None`` when nothing is to be subtracted:
                ``subtract_cleaned_data`` or ``project_combined`` is False, or
                ``uncleaned_data`` is zero everywhere.
        """
        if not (self.subtract_cleaned_data and self.project_combined):
            return None
        if not np.any(self.uncleaned_data):
            return None
        cleaning_matrix = self.R_mat
        if cleaning_matrix is None:
            cleaning_matrix = get_pca_matrix(
                self.uncleaned_data,
                self.N_fg,
                self.pca_map_weights,
                self.mean_center_map,
            )
        return apply_cleaning_matrix(
            0.0,
            self.uncleaned_data,
            cleaning_matrix,
            self.pca_map_weights,
            self.mean_center_map,
            True,
        )

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
        attr_dict["nu_min"] = self.ps.nu_min
        attr_dict["nu_max"] = self.ps.nu_max
        if hasattr(self.ps, "ra_gal"):
            attr_dict["num_discrete_source"] = self.ps.ra_gal.size
        # if it is a mock itself then overrides with its own attribute
        if hasattr(self.ps, "num_discrete_source"):
            attr_dict["num_discrete_source"] = self.ps.num_discrete_source
        attr_dict["highres_sim"] = self.highres_sim
        attr_dict["parallel_plane"] = self.parallel_plane
        attr_dict["rsd_from_field"] = self.rsd_from_field
        # attr_dict["discrete_source_dndz"] = self.discrete_source_dndz
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
        multipole_ells=None,
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
            multipole_ells: list, default None
                Accepted for the same call signature as the auto and cross
                argument lists. The null test does not use it.

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
                    self.unmask_during_mock,
                    self.ps.ra_range,
                    self.ps.dec_range,
                    self.discrete_source_dndz,
                )
            )
        return arg_list

    def get_arg_list_for_parallel_cross(
        self,
        seed_list,
        return_power_3d=False,
        return_power_1d=False,
        multipole_ells=None,
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
            multipole_ells: list, default None
                Multipoles passed through to the worker. ``None`` is the monopole.

        Returns
        -------
            arg_list: list
                The list of arguments for the parallelisation.
        """
        if self.construct_fkp_mock and not getattr(
            self.ps, "field_2_has_random", False
        ):
            raise ValueError(
                "construct_fkp_mock=True requires ps.field_2_has_random (i.e. "
                "ps.grid_gal_to_field(construct_fkp=True) was already called "
                "on the real data) so ps.field_2_R is defined."
            )
        cleaned_data = self.get_cleaned_data()
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
                    self.unmask_during_mock,
                    self.ps.ra_range,
                    self.ps.dec_range,
                    self.discrete_source_dndz,
                    self.project_combined,
                    multipole_ells,
                    self.mock_directory,
                    self.convolve_loaded_mock,
                    self.construct_fkp_mock,
                    (
                        getattr(self.ps, "field_2_R", None)
                        if self.construct_fkp_mock
                        else None
                    ),
                    self.mock_dilution,
                    self.gal_cat,
                    self.hi_stem,
                    self.gal_stem,
                    cleaned_data,
                )
            )
        return arg_list

    def get_arg_list_for_parallel_auto(
        self,
        seed_list,
        return_power_3d=False,
        return_power_1d=False,
        multipole_ells=None,
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
            multipole_ells: list, default None
                Multipoles passed through to the worker. ``None`` is the monopole.

        Returns
        -------
            arg_list: list
                The list of arguments for the parallelisation.
        """
        cleaned_data = self.get_cleaned_data()
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
                    self.unmask_during_mock,
                    self.ps.ra_range,
                    self.ps.dec_range,
                    self.project_combined,
                    multipole_ells,
                    self.mock_directory,
                    self.convolve_loaded_mock,
                    self.hi_stem,
                    self.gal_stem,
                    cleaned_data,
                )
            )
        return arg_list

    def run(
        self,
        seed_list,
        type="cross",
        return_power_3d=False,
        return_power_1d=False,
        multipole_ells=None,
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
            multipole_ells: list, default None
                Multipoles to compute. ``None`` is the monopole. A single
                multipole is returned as a 1D array. Several multipoles are
                stacked on the first axis.

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
        arg_list = arg_func(seed_list, return_power_3d, return_power_1d, multipole_ells)
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


def _multipole_ells(multipole_ells):
    """Return the multipoles to bin. ``None`` is the monopole."""
    if multipole_ells is None:
        return (0,)
    ells = tuple(multipole_ells)
    if len(ells) == 0:
        raise ValueError("multipole_ells must contain at least one multipole")
    return ells


def multipole_power_1d(mock, power3d, k1dweights, multipole_ells):
    """
    Bin a 3D power spectrum into one or more multipoles.

    A single multipole is the 1D array returned by ``get_1d_power``.
    Several multipoles are stacked on the first axis.
    """
    ells = _multipole_ells(multipole_ells)
    spectra = []
    for ell in ells:
        power_1d, _, _ = mock.get_1d_power(
            power3d, k1dweights=k1dweights, multipole_ell=ell
        )
        spectra.append(power_1d)
    if len(spectra) == 1:
        return spectra[0]
    return np.stack(spectra, axis=0)


def apply_cleaning_matrix(
    mock_data,
    uncleaned_data,
    cleaning_matrix,
    map_weights,
    mean_center_map,
    project_combined,
):
    """
    Apply a frequency cleaning matrix to a sky cube.

    ``cleaning_matrix`` is applied to the mock cube, or to
    ``uncleaned_data + mock_data`` when ``project_combined`` is True.
    When ``mean_center_map`` is True, the frequency mean of that chosen cube,
    weighted by ``map_weights``, is removed before the matrix is applied.

    Parameters
    ----------
    mock_data: np.ndarray
        Mock sky cube, shape ``(nx, ny, n_freq)``.
    uncleaned_data: np.ndarray
        Data cube added to the mock when ``project_combined`` is True.
    cleaning_matrix: np.ndarray
        Frequency matrix with shape ``(n_freq, n_freq)``.
    map_weights: np.ndarray
        Weights for the frequency mean, same sky shape as the cube.
    mean_center_map: bool
        Whether to subtract the weighted frequency mean.
    project_combined: bool
        Whether to project the sum of the data and the mock.

    Returns
    -------
    np.ndarray
        Cleaned cube, shape ``(nx, ny, n_freq)``.
    """
    combined = np.asarray(uncleaned_data) + mock_data
    array_to_clean = combined if project_combined else np.asarray(mock_data)
    if mean_center_map:
        data_mean = np.sum(map_weights * array_to_clean, axis=(0, 1)) / np.sum(
            map_weights, axis=(0, 1)
        )
        array_to_clean = array_to_clean - data_mean[None, None, :]
    return np.einsum("ij,abj->abi", cleaning_matrix, array_to_clean)


def _align_loaded_cube_masks(mock):
    """Broadcast the spatial mask and pixel weights onto the loaded cube."""
    shape = mock.data.shape
    sampling = np.asarray(mock.map_has_sampling)
    if sampling.shape != shape:
        if sampling.ndim == 3 and sampling.shape[:2] == shape[:2]:
            spatial = np.any(sampling, axis=-1)
        elif sampling.shape == shape[:2]:
            spatial = np.asarray(sampling, dtype=bool)
        else:
            spatial = np.ones(shape[:2], dtype=bool)
        mock.map_has_sampling = np.broadcast_to(spatial[..., None], shape).copy()
    weights = np.asarray(mock.weights_map_pixel)
    if weights.shape != shape:
        if weights.ndim == 3 and weights.shape[:2] == shape[:2]:
            spatial_weights = np.mean(weights, axis=-1)
        else:
            spatial_weights = np.ones(shape[:2], dtype=float)
        mock.weights_map_pixel = np.broadcast_to(
            spatial_weights[..., None], shape
        ).copy()


def _convolve_loaded_cube(mock):
    """Convolve a loaded HI cube with a per-channel Gaussian primary beam."""
    from .telescope import gaussian_beam, isotropic_beam_profile, weighted_convolution

    sigma = None if mock.sigma_beam_ch is None else np.asarray(mock.sigma_beam_ch)
    n_freq = mock.data.shape[-1]
    if sigma is None or sigma.shape != (n_freq,):
        raise ValueError(
            "sigma_beam_ch must have one value for each loaded frequency channel"
        )
    beam = np.zeros(mock.data.shape, dtype=float)
    for channel, sigma_channel in enumerate(sigma):
        beam[:, :, channel] = isotropic_beam_profile(
            mock.num_pix_x,
            mock.num_pix_y,
            mock.wproj,
            gaussian_beam(float(sigma_channel)),
        )
    weights = np.ones(mock.data.shape, dtype=float)
    convolved, _ = weighted_convolution(mock.data, beam, weights)
    mock.data = convolved


def _galaxy_file_stem(gal_stem="gal_mock_", gal_cat=None):
    """Galaxy filename text between the path prefix and the seed.

    ``gal_cat`` selects ``gal_{gal_cat}_mock_``. With ``gal_cat`` left as
    ``None``, the text is ``gal_stem``. Supplying both with different values
    raises ``ValueError``.
    """
    if gal_cat is None:
        return gal_stem
    tracer_stem = f"gal_{gal_cat}_mock_"
    if gal_stem not in ("gal_mock_", tracer_stem):
        raise ValueError(
            f"gal_cat={gal_cat!r} selects galaxy stem {tracer_stem!r}, "
            f"but gal_stem is {gal_stem!r}"
        )
    return tracer_stem


def load_precomputed_mock(
    mock,
    prefix,
    read_galaxies=True,
    convolve_loaded_mock=False,
    gal_cat=None,
    hi_stem="HI_mock_",
    gal_stem="gal_mock_",
):
    """
    Replace a mock's sky cube with a saved HI map and, optionally, its galaxies.

    The pickle is ``{prefix}{hi_stem}{seed}.pkl`` with ``HImap`` and ``freq``
    in MHz. Channels outside ``mock.nu_min < freq < mock.nu_max`` are dropped,
    ``mock.nu`` is set to the kept frequencies, and the sky mask is rebuilt to
    the cube shape. When ``read_galaxies`` is True, the galaxy file is
    ``{prefix}{stem}{seed}.fits``, where ``stem`` is ``gal_{gal_cat}_mock_``
    when ``gal_cat`` is set and ``gal_stem`` otherwise.
    ``convolve_loaded_mock`` applies the Gaussian primary beam.

    Parameters
    ----------
    mock: MockSimulation
        Mock whose ``seed``, frequency window, and beam width are already set.
    prefix: str
        Path prefix concatenated onto the file names.
    read_galaxies: bool, default True
        Whether to read the galaxy FITS table.
    convolve_loaded_mock: bool, default False
        Whether to convolve the loaded cube with ``sigma_beam_ch``.
    gal_cat: str, default None
        Tracer name. The galaxy file stem is then ``gal_{gal_cat}_mock_``.
    hi_stem: str, default "HI_mock_"
        Filename text between ``prefix`` and the seed for the HI pickle.
    gal_stem: str, default "gal_mock_"
        Filename text between ``prefix`` and the seed for the galaxy FITS table
        when ``gal_cat`` is ``None``.

    Returns
    -------
    MockSimulation
        The same mock, updated in place.
    """
    hi_path = f"{prefix}{hi_stem}{mock.seed}.pkl"
    with open(hi_path, "rb") as handle:
        payload = pickle.load(handle)
    hi_map = np.asarray(payload["HImap"])
    freq_hz = np.asarray(payload["freq"], dtype=float) * 1e6
    keep = (freq_hz > mock.nu_min) & (freq_hz < mock.nu_max)
    if not np.any(keep):
        raise ValueError(
            f"no channels in {hi_path} lie in " f"{mock.nu_min} < freq < {mock.nu_max}"
        )
    mock.data = hi_map[:, :, keep]
    mock.nu = freq_hz[keep]
    _align_loaded_cube_masks(mock)
    if convolve_loaded_mock:
        _convolve_loaded_cube(mock)
    if read_galaxies:
        stem = _galaxy_file_stem(gal_stem, gal_cat)
        mock.gal_file = f"{prefix}{stem}{mock.seed}.fits"
        mock.read_gal_cat()
    return mock


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
    unmask_during_mock=False,
    ra_range=None,
    dec_range=None,
    discrete_source_dndz=None,
    project_combined=True,
    multipole_ells=None,
    mock_directory=None,
    convolve_loaded_mock=False,
    construct_fkp_mock=False,
    random_counts_2=None,
    mock_dilution=None,
    gal_cat=None,
    hi_stem="HI_mock_",
    gal_stem="gal_mock_",
    cleaned_data=None,
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
        unmask_during_mock: bool, default False
            Whether to unmask the wcs during the mock simulation.
            If True, the survey area will be unmasked during the mock simulation, and then masked again after mock HI is generated.
            This is useful for mitigating edge effects and when box buffer is small.
            For a large patch with a wide z-range, it is recommended to set this to True,
            otherwise the mock HI map may contain nan and cause errors.
        ra_range: tuple, default None
            The range of the RA in the sky map. Map and galaxy will be trimmed to this range.
        dec_range: tuple, default None
            The range of the Dec in the sky map. Map and galaxy will be trimmed to this range.
        discrete_source_dndz: list or function, default None
            The redshift distribution of the discrete tracer sources.
            Must be a function of redshift.
            Note that the overall number of discrete sources is set to ``ps.ra_gal.size``
            or ``ps.num_discrete_source``, so
            only the shape of the dndz is used, not the normalization.
            If a list is provided, the first element is the redshift array, and the second element is the dndz array,
            and a 1D linear interpolation is used to get the dndz function.
        project_combined: bool, default True
            If True, apply the cleaning matrix to ``uncleaned_data + mock``.
            See :class:`meer21cm.transfer.TransferFunction`.
        multipole_ells: sequence, default None
            Multipoles to bin. ``None`` is the monopole. One multipole is a 1D
            array. Several are stacked on the first axis.
        mock_directory: str, default None
            Prefix for ``{hi_stem}{seed}.pkl`` and ``{gal_stem}{seed}.fits``.
            ``None`` generates the mock in this function.
        convolve_loaded_mock: bool, default False
            Convolve a loaded HI cube with the Gaussian primary beam.
        construct_fkp_mock: bool, default False
            Build ``F_mock = D_mock - alpha_mock * R`` for the mock galaxies
            instead of stamping ``weights_field_2``/``weights_grid_2``
            unchanged onto the mock's raw painted counts. See
            :class:`meer21cm.transfer.TransferFunction`.
        random_counts_2: np.ndarray, default None
            Painted random counts ``R`` (``ps.field_2_R``), reused for every
            mock. Required when ``construct_fkp_mock`` is True.
        mock_dilution: np.ndarray, default None
            See :class:`meer21cm.transfer.TransferFunction`. Applied to the
            mock HI field after beam convolution, before injection into
            ``uncleaned_data``; divided back out of the cleaned mock before
            the transfer function ratio is taken. ``None`` is a no-op.
        gal_cat: str, default None
            Tracer name. The galaxy file stem is then ``gal_{gal_cat}_mock_``.
            See :class:`meer21cm.transfer.TransferFunction`.
        hi_stem: str, default "HI_mock_"
            Filename text between ``mock_directory`` and the seed for the HI pickle.
        gal_stem: str, default "gal_mock_"
            Filename text between ``mock_directory`` and the seed for the galaxy
            table when ``gal_cat`` is ``None``.
        cleaned_data: np.ndarray, default None
            ``uncleaned_data`` cleaned with its own PCA matrix. When given, it
            is subtracted from the cleaned ``uncleaned_data + mock`` cube.
            See ``subtract_cleaned_data`` in
            :class:`meer21cm.transfer.TransferFunction`.
    Returns
    -------
        result: list
            The list of results, including the transfer function, the 3D power spectrum of the uncleaned and cleaned data, and the 1D power spectrum of the uncleaned and cleaned data.
    """
    mock = MockSimulation(**mock_attr_dict)
    if mock_directory is None:
        if not isinstance(discrete_source_dndz, Callable):
            discrete_source_dndz = interp1d(
                discrete_source_dndz[0],
                discrete_source_dndz[1],
                kind="linear",
                bounds_error=False,
                fill_value=0,
            )
        mock.discrete_source_dndz = discrete_source_dndz
        if unmask_during_mock:
            W_HI_mock = mock.W_HI.copy()
            w_HI_mock = mock.w_HI.copy()
            mock.W_HI = np.ones_like(mock.W_HI)
            mock.w_HI = np.ones_like(mock.w_HI)
        mock.data = mock.propagate_mock_field_to_data(mock.mock_tracer_field_1)
        mock.propagate_mock_tracer_to_gal_cat()
        if unmask_during_mock:
            mock.W_HI = W_HI_mock
            mock.w_HI = w_HI_mock
            mock.data[mock.W_HI == 0] = 0

    else:
        load_precomputed_mock(
            mock,
            mock_directory,
            read_galaxies=True,
            convolve_loaded_mock=convolve_loaded_mock,
            gal_cat=gal_cat,
            hi_stem=hi_stem,
            gal_stem=gal_stem,
        )

    mock.ra_range = ra_range
    mock.dec_range = dec_range
    mock.trim_map_to_range()
    mock.trim_gal_to_range()
    mock.downres_factor_radial = downres_factor_radial
    mock.downres_factor_transverse = downres_factor_transverse
    mock.get_enclosing_box()
    mock_map_rg, _, _ = mock.grid_data_to_field()
    mockgal_map_rg, _, _ = mock.grid_gal_to_field()
    mock.field_1 = mock_map_rg
    mock.weights_field_1 = weights_field_1
    mock.weights_grid_1 = weights_grid_1
    if construct_fkp_mock:
        # F_mock = D_mock - alpha_mock*R via the same assembly
        # grid_gal_to_field(construct_fkp=True) uses, but with R reused
        # unchanged from the real survey (random_counts_2 = ps.field_2_R)
        # instead of painted fresh -- only D_mock, and hence alpha_mock,
        # differ seed to seed. weights_grid_2 is the parent's taper/
        # footprint template (not the FKP amplitude), baked in directly.
        mock.set_fkp_field_from_counts(
            mockgal_map_rg, random_counts_2, weights_grid_2_template=weights_grid_2
        )
    else:
        mock.field_2 = mockgal_map_rg
        mock.weights_field_2 = weights_field_2
        mock.weights_grid_2 = weights_grid_2

    pmock_1d_cross = multipole_power_1d(
        mock, mock.cross_power_3d, k_sel_3d_to_1d, multipole_ells
    )
    if return_power_3d:
        power_3d_uncleaned = mock.cross_power_3d
    # mock_dilution applied here, after the mock's own (undiluted) reference
    # power above -- the transfer function denominator stays the true, clean
    # HI signal. The dilution only affects the copy injected into
    # uncleaned_data and cleaned below, mimicking how a real HI signal would
    # have looked after block combination. See TransferFunction's docstring.
    if mock_dilution is not None:
        mock.data = mock.data * mock_dilution
    map_to_clean = np.asarray(uncleaned_data) + mock.data
    cleaning_matrix = R_mat
    if cleaning_matrix is None:
        cleaning_matrix = get_pca_matrix(
            map_to_clean, N_fg, pca_map_weights, mean_center_map
        )
    mock.data = apply_cleaning_matrix(
        mock.data,
        uncleaned_data,
        cleaning_matrix,
        pca_map_weights,
        mean_center_map,
        project_combined,
    )
    if cleaned_data is not None:
        mock.data = mock.data - cleaned_data
    if mock_dilution is not None:
        # Undo the dilution on the cleaned output, mirroring the hit_nomask/
        # hit rescale applied to the real cleaned data.
        mock.data = mock.data / mock_dilution
    mock_map_rg, _, _ = mock.grid_data_to_field()
    mock.field_1 = mock_map_rg
    mock.weights_field_1 = weights_field_1
    mock.weights_grid_1 = weights_grid_1
    pmock_1d_cross_cleaned = multipole_power_1d(
        mock, mock.cross_power_3d, k_sel_3d_to_1d, multipole_ells
    )
    tf_1d_i = np.asarray(pmock_1d_cross_cleaned) / np.asarray(pmock_1d_cross)
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
    unmask_during_mock=False,
    ra_range=None,
    dec_range=None,
    project_combined=True,
    multipole_ells=None,
    mock_directory=None,
    convolve_loaded_mock=False,
    hi_stem="HI_mock_",
    gal_stem="gal_mock_",
    cleaned_data=None,
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
        unmask_during_mock: bool, default False
            Whether to unmask the wcs during the mock simulation.
            If True, the survey area will be unmasked during the mock simulation, and then masked again after mock HI is generated.
            This is useful for mitigating edge effects and when box buffer is small.
            For a large patch with a wide z-range, it is recommended to set this to True,
            otherwise the mock HI map may contain nan and cause errors.
        ra_range: tuple, default None
            The range of the RA in the sky map. Map and galaxy will be trimmed to this range.
        dec_range: tuple, default None
            The range of the Dec in the sky map. Map and galaxy will be trimmed to this range.
        project_combined: bool, default True
            If True, apply the cleaning matrix to ``uncleaned_data + mock``.
            See :class:`meer21cm.transfer.TransferFunction`.
        multipole_ells: sequence, default None
            Multipoles to bin. ``None`` is the monopole. One multipole is a 1D
            array. Several are stacked on the first axis.
        mock_directory: str, default None
            Prefix for ``{hi_stem}{seed}.pkl``. ``None`` generates the mock here.
            The galaxy file is not read on this path.
        convolve_loaded_mock: bool, default False
            Convolve a loaded HI cube with the Gaussian primary beam.
        hi_stem: str, default "HI_mock_"
            Filename text between ``mock_directory`` and the seed for the HI pickle.
        gal_stem: str, default "gal_mock_"
            Filename text between ``mock_directory`` and the seed for the galaxy
            table. The galaxy file is not read on this path.
        cleaned_data: np.ndarray, default None
            ``uncleaned_data`` cleaned with its own PCA matrix. When given, it
            is subtracted from the cleaned ``uncleaned_data + mock`` cube.
            See ``subtract_cleaned_data`` in
            :class:`meer21cm.transfer.TransferFunction`.
    Returns
    -------
        result: list
            The list of results, including the transfer function, the 3D power spectrum of the uncleaned and cleaned data, and the 1D power spectrum of the uncleaned and cleaned data.
    """
    mock = MockSimulation(**mock_attr_dict)
    if mock_directory is None:
        if unmask_during_mock:
            W_HI_mock = mock.W_HI.copy()
            w_HI_mock = mock.w_HI.copy()
            mock.W_HI = np.ones_like(mock.W_HI)
            mock.w_HI = np.ones_like(mock.w_HI)
        mock.data = mock.propagate_mock_field_to_data(mock.mock_tracer_field_1)
        if unmask_during_mock:
            mock.W_HI = W_HI_mock
            mock.w_HI = w_HI_mock
            mock.data[mock.W_HI == 0] = 0
    else:
        load_precomputed_mock(
            mock,
            mock_directory,
            read_galaxies=False,
            convolve_loaded_mock=convolve_loaded_mock,
            hi_stem=hi_stem,
            gal_stem=gal_stem,
        )
    mock.ra_range = ra_range
    mock.dec_range = dec_range
    mock.trim_map_to_range()
    mock.downres_factor_radial = downres_factor_radial
    mock.downres_factor_transverse = downres_factor_transverse
    mock.get_enclosing_box()
    mock_map_rg_before, _, _ = mock.grid_data_to_field()
    mock.field_1 = mock_map_rg_before
    mock.weights_field_1 = weights_field_1
    mock.weights_grid_1 = weights_grid_1
    pmock_1d_before = multipole_power_1d(
        mock, mock.auto_power_3d_1, k_sel_3d_to_1d, multipole_ells
    )
    if return_power_3d:
        power_3d_uncleaned = mock.auto_power_3d_1
    map_to_clean = np.asarray(uncleaned_data) + mock.data
    cleaning_matrix = R_mat
    if cleaning_matrix is None:
        cleaning_matrix = get_pca_matrix(
            map_to_clean, N_fg, pca_map_weights, mean_center_map
        )
    if mean_center_map and not project_combined:
        # Flat-sky gridding aliases field_1 with this map. Subtract in place so
        # the cross against the cleaned cube uses the mean-centered mock.
        data_mean = np.sum(pca_map_weights * mock.data, axis=(0, 1)) / np.sum(
            pca_map_weights, axis=(0, 1)
        )
        mock.data -= data_mean[None, None, :]
        cleaned = np.einsum("ij,abj->abi", cleaning_matrix, mock.data)
    else:
        cleaned = apply_cleaning_matrix(
            mock.data,
            uncleaned_data,
            cleaning_matrix,
            pca_map_weights,
            mean_center_map,
            project_combined,
        )
        if cleaned_data is not None:
            cleaned = cleaned - cleaned_data
    mock.data = cleaned
    mock_map_rg_cleaned, _, _ = mock.grid_data_to_field()
    mock.field_1 = mock_map_rg_before
    mock.weights_field_1 = weights_field_1
    mock.weights_grid_1 = weights_grid_1
    mock.field_2 = mock_map_rg_cleaned
    mock.weights_field_2 = weights_field_1
    mock.weights_grid_2 = weights_grid_1
    mock.unitless_2 = False
    mock.mean_center_2 = False
    pmock_1d_cross_cleaned = multipole_power_1d(
        mock, mock.cross_power_3d, k_sel_3d_to_1d, multipole_ells
    )
    tf_1d_i = np.asarray(pmock_1d_cross_cleaned) / np.asarray(pmock_1d_before)
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
    unmask_during_mock=False,
    ra_range=None,
    dec_range=None,
    discrete_source_dndz=None,
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
        unmask_during_mock: bool, default False
            Whether to unmask the wcs during the mock simulation.
            If True, the survey area will be unmasked during the mock simulation, and then masked again after mock HI is generated.
            This is useful for mitigating edge effects and when box buffer is small.
            For a large patch with a wide z-range, it is recommended to set this to True,
            otherwise the mock HI map may contain nan and cause errors.
        ra_range: tuple, default None
            The range of the RA in the sky map. Map and galaxy will be trimmed to this range.
        dec_range: tuple, default None
            The range of the Dec in the sky map. Map and galaxy will be trimmed to this range.
        discrete_source_dndz: list or function, default None
            The redshift distribution of the discrete tracer sources.
            Must be a function of redshift.
            Note that the overall number of discrete sources is set to ``ps.ra_gal.size``
            or ``ps.num_discrete_source``, so
            only the shape of the dndz is used, not the normalization.
            If a list is provided, the first element is the redshift array, and the second element is the dndz array,
            and a 1D linear interpolation is used to get the dndz function.
    Returns
    -------
        result: list
            The list of results, including the 1D cross-power spectrum of the mock galaxy x data map.
            If ``return_power_3d`` is True, the second element is the 3D power spectrum of the mock results.
    """
    mock = MockSimulation(**mock_attr_dict)
    if not isinstance(discrete_source_dndz, Callable):
        discrete_source_dndz = interp1d(
            discrete_source_dndz[0],
            discrete_source_dndz[1],
            kind="linear",
            bounds_error=False,
            fill_value=0,
        )
    mock.discrete_source_dndz = discrete_source_dndz
    if unmask_during_mock:
        W_HI_mock = mock.W_HI.copy()
        w_HI_mock = mock.w_HI.copy()
        mock.W_HI = np.ones_like(mock.W_HI)
        mock.w_HI = np.ones_like(mock.w_HI)
    mock.propagate_mock_tracer_to_gal_cat()
    if unmask_during_mock:
        mock.W_HI = W_HI_mock
        mock.w_HI = w_HI_mock
    mock.ra_range = ra_range
    mock.dec_range = dec_range
    mock.trim_map_to_range()
    mock.trim_gal_to_range()
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
