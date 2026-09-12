"""
This module handles computation of power spectrum from gridded fields and its corresponding model power spectrum from theory.

The class :py:class:`~meer21cm.model.ModelPowerSpectrum` is the class for computing the model power spectrum of an LSS tracer field.

The class :py:class:`~meer21cm.estimator.FieldPowerSpectrum` is the class for computing the power spectrum of a gridded field from LSS data.

The class :py:class:`PowerSpectrum` coherently combines the two classes above,
and provides an interface for gridding the intensity mapping data as well as the galaxy catalogue to perform
power spectrum estimation and for auto-correlation and cross-correlation.

For a modular layout, see also :mod:`meer21cm.power_ops`, :mod:`meer21cm.grid`,
:mod:`meer21cm.estimator`, and :mod:`meer21cm.model`.
"""

import logging

import numpy as np
from scipy.signal import windows

from .estimator import FieldPowerSpectrum
from .grid import (
    LightconeGriddingMixin,
    fourier_window_for_assignment,
    interlace_two_fields,
    minimum_enclosing_box_of_lightcone,
    project_particle_to_regular_grid,
)
from .model import ModelPowerSpectrum
from .power_ops import (
    bin_3d_to_1d,
    bin_3d_to_cy,
    gaussian_beam_attenuation,
    get_fourier_density,
    get_gaussian_noise_floor,
    get_k_vector,
    get_modelpk_conv,
    get_power_spectrum,
    get_renormed_field,
    get_shot_noise,
    get_shot_noise_galaxy,
    get_vec_mode,
    get_x_vector,
    power_weights_renorm,
    step_window_attenuation,
)
from .util import (
    find_ch_id,
    freq_to_redshift,
    get_nd_slicer,
    legendre_polynomial_with_factor,
    omega_hi_to_average_temp,
    radec_to_indx,
    real_dtype_from_array,
    redshift_to_freq,
    tagging,
)

# Re-exports for backward compatibility (tests, docs, and ``from meer21cm.power import *``).
from .dataanalysis import Specification  # noqa: F401

logger = logging.getLogger(__name__)

__all__ = [
    "FieldPowerSpectrum",
    "ModelPowerSpectrum",
    "PowerSpectrum",
    "LightconeGriddingMixin",
    "Specification",
    "bin_3d_to_1d",
    "bin_3d_to_cy",
    "find_ch_id",
    "fourier_window_for_assignment",
    "freq_to_redshift",
    "gaussian_beam_attenuation",
    "get_fourier_density",
    "get_gaussian_noise_floor",
    "get_k_vector",
    "get_modelpk_conv",
    "get_nd_slicer",
    "get_power_spectrum",
    "get_renormed_field",
    "get_shot_noise",
    "get_shot_noise_galaxy",
    "get_vec_mode",
    "get_x_vector",
    "interlace_two_fields",
    "legendre_polynomial_with_factor",
    "minimum_enclosing_box_of_lightcone",
    "omega_hi_to_average_temp",
    "power_weights_renorm",
    "project_particle_to_regular_grid",
    "radec_to_indx",
    "real_dtype_from_array",
    "redshift_to_freq",
    "step_window_attenuation",
    "tagging",
]


class PowerSpectrum(LightconeGriddingMixin, FieldPowerSpectrum, ModelPowerSpectrum):
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
    los : {'global', 'endpoint', 'firstpoint', 'midpoint'}, default 'global'
        Line-of-sight convention for field multipoles (see
        :class:`~meer21cm.estimator.FieldPowerSpectrum`). Only ``'global'``
        is implemented.
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
        los="global",
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
        # Initialise survey/cosmo/model first so Specification runs once with **params.
        ModelPowerSpectrum.__init__(
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

    def _sync_model_k_from_field(self):
        """Propagate field k-modes to the model when both box geometry attrs exist."""
        if not getattr(self, "model_k_from_field", False):
            return
        box_len = getattr(self, "_box_len", None)
        box_ndim = getattr(self, "_box_ndim", None)
        if box_len is None or box_ndim is None:
            return
        self.propagate_field_k_to_model()

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
        self._sync_model_k_from_field()

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
        self._sync_model_k_from_field()

    @property
    def grid_scheme(self):
        """
        The grid scheme to be used for gridding.
        Can be "nnb", "cic", "tsc" or "pcs".
        """
        return self._grid_scheme

    @grid_scheme.setter
    def grid_scheme(self, value):
        self._grid_scheme = value
        for dep_name in ("tracer_1_dep_attr", "tracer_2_dep_attr"):
            if dep_name in dir(self):
                dep_attr = getattr(self, dep_name)
                logger.debug(
                    f"cleaning cache of {dep_attr} due to resetting grid_scheme"
                )
                self.clean_cache(dep_attr)

    def propagate_field_k_to_model(self):
        r"""
        Use field k-modes for the model, taking into account the Alcock–Paczynski effect.

        .. math::
            k_\perp^\text{fiducial} = k_\perp^\text{true} \times \alpha_\perp

            k_\parallel^\text{fiducial} = k_\parallel^\text{true} \times \alpha_\parallel

        """
        # use field kmode to propagate into model
        kperp = self.k_perp / self.alpha_perp
        kpara = self.k_para / self.alpha_parallel
        self.kmode = np.sqrt(kperp[:, :, None] ** 2 + kpara[None, None, :] ** 2)
        with np.errstate(divide="ignore", invalid="ignore"):
            mu = np.nan_to_num(kpara[None, None, :] / self.kmode)
        self.mumode = np.clip(mu, -1.0, 1.0)

    def get_1d_power(
        self,
        power3d,
        k1dbins=None,
        k1dweights=None,
        k_xyz_min=None,
        k_xyz_max=None,
        k_perppara_min=None,
        k_perppara_max=None,
        multipole_ell=0,
        mu_model=None,
    ):
        """
        Bin the 3D power spectrum into 1D power spectrum.
        If the input ``power3d`` is a string, it is assumed to be an attribute of the class,
        for example ``auto_power_3d_1``.
        Also see :meth:`meer21cm.power.bin_3d_to_1d` for more details.

        By default the 1D power spectrum is calculated for the monopole.
        Passing ``multipole_ell`` will calculate the 1D power spectrum for the given multipole.

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
        multipole_ell: int, default 0
            The multipole order for the 1D power spectrum.
            By default the 1D power spectrum is calculated for the monopole.
        mu_model: np.ndarray, default None
            The mu-modes for the legendre polynomial.
            If None, use the class attribute ``mumode``.

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
        if mu_model is None:
            mu_model = self.mumode
        multipole_factor = np.poly1d(legendre_polynomial_with_factor(multipole_ell))(
            mu_model
        )
        power1d, k1deff, nmodes = bin_3d_to_1d(
            power3d * multipole_factor,
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
        multipole_ell=0,
        mu_model=None,
    ):
        """
        Bin the 3D power spectrum into cylindrical k_perp-k_para power spectrum.
        If the input ``power3d`` is a string, it is assumed to be an attribute of the class,
        for example ``auto_power_3d_1``.
        Also see :meth:`meer21cm.power.bin_3d_to_cy` for more details.

        Passing ``multipole_ell`` will calculate the cylindrical power spectrum multiplied by the Legendre polynomial.

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
        multipole_ell: int, default 0
            The multipole order for the cylindrical power spectrum.
            By default the cylindrical power spectrum is calculated for the monopole.
        mu_model: np.ndarray, default None
            The mu-modes for the legendre polynomial.
            If None, use the class attribute ``mumode``.

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
        if mu_model is None:
            mu_model = self.mumode
        multipole_factor = np.poly1d(legendre_polynomial_with_factor(multipole_ell))(
            mu_model
        )
        powercy = bin_3d_to_cy(
            power3d * multipole_factor,
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
    def average_model_hi_temp(self):
        """
        Calculate the average HI brightness temperature in the map cube, taking care of redshift evolution and map sampling.
        Calculation is based on the true (fitted) cosmology.
        """
        t_bar = omega_hi_to_average_temp(
            self.omega_hi, z=self.z_ch, cosmo=self.astropy_cosmo_true
        )
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
        Calculation is based on the true (fitted) cosmology.
        """
        z_grid = self._box_voxel_redshift
        omega_hi_grid = self.omega_hi_z_func(z_grid)
        t_bar_grid = omega_hi_to_average_temp(
            omega_hi_grid, z=z_grid, cosmo=self.astropy_cosmo_true
        )
        return t_bar_grid

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
