"""
This module contains the classes for generating mock intensity mapping data, galaxy catalogues for cross-correlation,
and HI galaxy catalogues for HI emission line observations.
"""

import numpy as np
from scipy.interpolate import interp1d
from astropy.cosmology import Planck18
from numpy.random import default_rng
from astropy import constants
from .util import (
    get_wcs_coor,
    radec_to_indx,
    freq_to_redshift,
    tagging,
    busy_function_simple,
    himf_pars_jones18,
    sample_from_dist,
    center_to_edges,
    tully_fisher,
    redshift_to_freq,
    find_ch_id,
    mass_intflux_coeff,
    Obuljen18,
    create_udres_wproj,
    sample_map_from_highres,
    angle_in_range,
    get_nd_slicer,
    real_dtype_from_array,
)
from .power import PowerSpectrum, Specification, power_weights_renorm
from .telescope import weighted_convolution
from halomod import TracerHaloModel as THM
import warnings
import inspect
import logging

logger = logging.getLogger(__name__)


class MockSimulation(PowerSpectrum):
    """
    The class for generating mock intensity mapping data cube and galaxy catalogues for cross-correlation.

    Parameters
    ----------
    density: str, default "lognormal"
        The density distribution of the mock field. Can be "lognormal" or "gaussian".
    num_discrete_source: int, default 100
        The number of discrete tracer sources.
    discrete_base_field: int, default 2
        The tracer (1 or 2) field to sample the discrete tracer positions.
    highres_sim: int, default None
        For WCS maps, this factor is **deprecated** in :meth:`propagate_mock_field_to_data`
        (a warning is issued and native resolution is used). It still controls the optional
        high-resolution path in :meth:`HIGalaxySimulation.propagate_hi_profile_to_map`.
    parallel_plane: bool, default True
        Whether to simulate the mock field in the parallel-plane limit (i.e. k_z = k_parallel).
        Only used if ``rsd_from_field`` is True, otherwise parallel-plane is hard-coded to True.
    rsd_from_field: bool, default False
        Whether to generate the RSD effect at field level. If False, the RSD effect is generated at power spectrum level.
    discrete_source_dndz: function, default np.ones_like
        The redshift distribution of the discrete tracer sources.
        Must be a function of redshift.
        Note that the overall number of discrete sources is given by ``self.num_discrete_source``, so
        only the shape of the dndz is used, not the normalization.
    **params: dict
        Additional parameters to be passed to the base class :class:`meer21cm.power.PowerSpectrum`.
    """

    def __init__(
        self,
        density="lognormal",
        num_discrete_source=100,
        discrete_base_field=2,
        highres_sim=None,
        parallel_plane=True,
        rsd_from_field=False,
        discrete_source_dndz=np.ones_like,
        mock_amp_1=None,
        mock_amp_2=None,
        **params,
    ):
        super().__init__(**params)
        self.density = density.lower()
        init_attr = [
            "_x_start",
            "_y_start",
            "_z_start",
            "_x_len",
            "_y_len",
            "_z_len",
            "_rot_mat_sky_to_box",
            "_pix_coor_in_cartesian",
            "_box_len",
            "_box_resol",
            "_box_ndim",
            "_mock_matter_field_r",
            "_mock_matter_field",
            "_mock_tracer_position_in_box",
            "_mock_tracer_position_in_radecz",
            "_mock_tracer_field_1",
            "_mock_tracer_field_2",
            "_mock_tracer_field_1_r",
            "_mock_tracer_field_2_r",
            "_mock_kaiser_field_k_matter",
            "_mock_kaiser_field_k_tracer_1",
            "_mock_kaiser_field_k_tracer_2",
            "_mock_velocity_u_matter",
            "_mock_velocity_u_tracer_1",
            "_mock_velocity_u_tracer_2",
            "_mock_amp_1",
            "_mock_amp_2",
        ]
        for attr in init_attr:
            setattr(self, attr, None)
        self.num_discrete_source = num_discrete_source
        self.discrete_base_field = discrete_base_field
        self.highres_sim = highres_sim
        self.parallel_plane = parallel_plane
        self.rsd_from_field = rsd_from_field
        if not rsd_from_field and not parallel_plane:
            warnings.warn("rsd_from_field is False, parallel_plane will be ignored")
        self.discrete_source_dndz = discrete_source_dndz
        self.mock_amp_1 = mock_amp_1
        self.mock_amp_2 = mock_amp_2
        if self.true_cosmology != self.fiducial_cosmology:
            warnings.warn(
                "true_cosmology != fiducial_cosmology, "
                "box dimensions and model power will be inconsistent"
            )

    def _iter_last_axis_batches(self, axis_size):
        """
        Return stable last-axis batches for chunked processing.

        Always returns at least one non-empty batch so `batch_number=1`
        exercises the same loop path as `batch_number>1`.
        """
        all_indx = np.arange(axis_size)
        return [
            sel for sel in np.array_split(all_indx, self.batch_number) if sel.size > 0
        ]

    def _iter_field_los_chunks(self, field):
        """
        Iterate over LOS chunks of a 3D field.
        """
        for sel in self._iter_last_axis_batches(field.shape[-1]):
            yield sel, field[..., sel]

    def _apply_weighted_convolution_in_batches(
        self,
        map_data,
        map_counts,
        beam_image=None,
        wproj=None,
        num_pix_x=None,
        num_pix_y=None,
    ):
        """
        Apply beam convolution in channel batches.

        This keeps the same behavior for `batch_number=1` while allowing
        conservative channel-wise execution for larger batch counts.
        """
        map_out = np.zeros_like(map_data)
        lazy_beam = beam_image is None
        for ch_sel in self._iter_last_axis_batches(map_data.shape[-1]):
            if lazy_beam:
                beam_chunk = self.get_beam_image(
                    wproj=wproj,
                    num_pix_x=num_pix_x,
                    num_pix_y=num_pix_y,
                    cache=False,
                    ch_sel=ch_sel,
                )
            else:
                beam_chunk = beam_image[..., ch_sel]
            map_i, _ = weighted_convolution(
                map_data[..., ch_sel],
                beam_chunk,
                (map_counts[..., ch_sel] > 0).astype(float),
            )
            map_out[..., ch_sel] = map_i
        return map_out

    def _project_field_chunk_to_sky_map(
        self,
        field_chunk,
        los_sel,
        average=True,
        wproj=None,
        num_pix_x=None,
        num_pix_y=None,
    ):
        """
        Project one LOS chunk of a box field to the sky map grid.

        This helper provides a chunk-level boundary for external orchestration.
        """
        return self.grid_field_to_sky_map(
            field_chunk,
            average=average,
            mask=False,
            wproj=wproj,
            num_pix_x=num_pix_x,
            num_pix_y=num_pix_y,
            los_sel=los_sel,
        )

    def _sample_tracer_positions_from_density_chunk(
        self, density_chunk, dndz_prob_chunk, z_sel, rng
    ):
        """
        Sample tracer positions from one LOS chunk of density and dN/dz weights.
        """
        weighted_density = density_chunk * dndz_prob_chunk
        n_per_cell = rng.poisson(weighted_density)
        x_chunk = np.meshgrid(
            self.x_vec[0], self.x_vec[1], self.x_vec[2][z_sel], indexing="ij"
        )
        tracer_positions = np.array([x.flatten() for x in x_chunk]).T
        tracer_positions = tracer_positions.repeat(n_per_cell.flatten(), axis=0)
        if tracer_positions.size == 0:
            return tracer_positions
        tracer_positions += (
            rng.uniform(-0.5, 0.5, size=(tracer_positions.shape[0], len(self.box_ndim)))
            * self.box_resol[None, :]
        )
        if self.flat_sky:
            tracer_positions -= (
                np.array(self.flat_sky_padding) * np.array(self.box_resol)
            )[None, :]
        return tracer_positions

    @property
    def tot_num_source_in_box(self):
        """
        The total number of mock sources in the box needed to achieve
        ``self.num_discrete_source`` number of sources in the survey volume.
        Only used internally, no physical meaning.
        Note that if you change the simulation settings such as ``self.num_discrete_source``,
        ``self.discrete_source_dndz``, ``self.z_ch``, ``self.W_HI``, etc,
        this property will be automatically updated but the mock catalog is not.
        """
        if self.flat_sky:
            dndz_arr = self.discrete_source_dndz(self.box_voxel_redshift)
            z_sel = (self.box_voxel_redshift >= self.z_ch.min()) & (
                self.box_voxel_redshift <= self.z_ch.max()
            )
            dndz_arr = dndz_arr[z_sel]
            dndz_arr /= dndz_arr.max()
            ratio_dndz = 1 / dndz_arr.mean()
            self._dndz_renorm = (
                lambda z: self.discrete_source_dndz(z) / dndz_arr.max() * ratio_dndz
            )
            ratio = (
                np.prod(np.array(self.data.shape) + 2 * np.array(self.flat_sky_padding))
                / self.W_HI.sum()
            )
            return self.num_discrete_source * ratio
        else:
            nu_ext = center_to_edges(self.nu)
            z_ext = freq_to_redshift(nu_ext)
            volume_per_channel = (
                (self.W_HI[:, :, 0].sum() * self.pixel_area * (np.pi / 180) ** 2)
                / 3
                * (
                    self.astropy_cosmo_true.comoving_distance(z_ext[:-1]) ** 3
                    - self.astropy_cosmo_true.comoving_distance(z_ext[1:]) ** 3
                ).value
            )
            dn_channel = self.discrete_source_dndz(self.z_ch)
            renorm = self.num_discrete_source / (dn_channel * volume_per_channel).sum()
            ratio = np.prod(self.box_len) / self.survey_volume
            renorm *= np.prod(self.box_len) / (self.num_discrete_source * ratio)
            self._dndz_renorm = lambda z: self.discrete_source_dndz(z) * renorm
            return self.num_discrete_source * ratio

    @property
    def highres_sim(self):
        """
        Optional integer upsampling factor for :meth:`HIGalaxySimulation.propagate_hi_profile_to_map`.

        For WCS, :meth:`propagate_mock_field_to_data` ignores this value and emits
        a ``DeprecationWarning`` when it is not ``None``.
        """
        return self._highres_sim

    @highres_sim.setter
    def highres_sim(self, value):
        self._highres_sim = value

    @property
    def parallel_plane(self):
        """
        Whether the mock field is generated in the parallel-plane limit.
        """
        return self._parallel_plane

    @parallel_plane.setter
    def parallel_plane(self, value):
        self._parallel_plane = value
        logger.debug(
            f"cleaning cache of {self.rsd_dep_attr} due to resetting parallel_plane"
        )
        self.clean_cache(self.rsd_dep_attr)

    @property
    def rsd_from_field(self):
        """
        If True, the kaiser rsd effect is generated at field level by
        calculating the corresponding peculiar velocity field.
        This allows the lognormal mock to go beyond the parallel-plane limit.
        If False, the kaiser rsd effect is generated at power spectrum level assuming parallel-plane,
        and then the field is generated from the anistropic power spectrum.
        """
        return self._rsd_from_field

    @rsd_from_field.setter
    def rsd_from_field(self, value):
        self._rsd_from_field = value
        logger.debug(
            f"cleaning cache of {self.rsd_dep_attr} due to resetting rsd_from_field"
        )
        self.clean_cache(self.rsd_dep_attr)

    @property
    def num_discrete_source(self):
        """
        The total number of discrete tracer sources.
        Note that the final mock catalogue is not exactly this number,
        due to Poisson sampling errors.
        """
        return self._num_discrete_source

    @num_discrete_source.setter
    def num_discrete_source(self, value):
        self._num_discrete_source = value
        logger.debug(
            f"cleaning cache of {self.discrete_dep_attr} due to resetting num_discrete_source"
        )
        self.clean_cache(self.discrete_dep_attr)

    @property
    def discrete_source_dndz(self):
        """
        The redshift kernel of the discrete tracer sources.
        Must be a function of redshift.
        ** Only the shape of the dndz is used, not the normalization. **
        The overall number of discrete sources is given by ``self.num_discrete_source``.
        Must be in the unit of per volume instead of unitless
        , i.e. the integral of the dndz over redshift must be a number density instead of number of sources.
        """
        return self._discrete_source_dndz

    @discrete_source_dndz.setter
    def discrete_source_dndz(self, value):
        self._discrete_source_dndz = value
        logger.debug(
            f"cleaning cache of {self.discrete_dep_attr} due to resetting discrete_source_dndz"
        )
        self.clean_cache(self.discrete_dep_attr)

    @property
    def discrete_base_field(self):
        """
        Which tracer (1 or 2) field to sample the
        discrete tracer positions.
        """
        return self._discrete_base_field

    @discrete_base_field.setter
    def discrete_base_field(self, value):
        if isinstance(value, str):
            value = int(value)
        if not value in [1, 2]:
            raise ValueError("discrete_base_field must be 1 or 2")
        self._discrete_base_field = value
        logger.debug(
            f"cleaning cache of {self.discrete_dep_attr} due to resetting discrete_base_field"
        )
        self.clean_cache(self.discrete_dep_attr)

    @property
    @tagging("cosmo_model", "nu", "mock", "box")
    def mock_matter_field_r(self):
        """
        The simulated dark matter density field in real space.
        """
        if self._mock_matter_field_r is None:
            self.get_mock_matter_field_r()
        return self._mock_matter_field_r

    @property
    @tagging("cosmo_model", "nu", "mock", "box", "rsd")
    def mock_matter_field(self):
        """
        The simulated dark matter density field in redshift space.
        If ``self.kaiser_rsd`` is False, it is simply set to the real space field.
        """
        if self._mock_matter_field is None:
            self.get_mock_matter_field()
        return self._mock_matter_field

    def get_mock_matter_field_r(self):
        """
        Generate the mock matter field in real space.
        """
        logger.info(
            f"invoking {inspect.currentframe().f_code.co_name} to set __mock_matter_field_r"
        )
        self._mock_matter_field_r = self.get_mock_field_r(bias=1)

    def get_mock_field_r(self, bias=1):
        """
        Generate a mock field in real space that follows the input matter power
        spectrum ``self.matter_power_spectrum_fnc`` * bias**2.
        """
        logger.info(
            f"invoking {inspect.currentframe().f_code.co_name} with bias={bias}"
        )
        if self.box_ndim is None:
            self.get_enclosing_box()
        self.propagate_field_k_to_model()
        power_array = self.auto_power_matter_model_r * bias**2
        power_array = np.asarray(power_array, dtype=self.real_dtype)
        power_array[0, 0, 0] = 0.0
        delta_x = self.get_mock_field_from_power(power_array)
        return delta_x

    def get_mock_field_from_power(self, power_array):
        """
        Generate a mock field that follows the input matter power
        spectrum ``power_array``.
        """
        logger.info(
            f"invoking {inspect.currentframe().f_code.co_name} assuming {self.density} distribution"
        )
        if self.density == "lognormal":
            backend = generate_lognormal_field
        elif self.density == "gaussian":
            backend = generate_gaussian_field
        else:
            raise ValueError(
                f"density must be 'lognormal' or 'gaussian', got {self.density}"
            )
        power_array = np.asarray(power_array, dtype=self.real_dtype)
        delta_x = backend(self.box_ndim, self.box_len, power_array, self.seed)
        return delta_x

    @property
    @tagging("cosmo_model", "nu", "mock", "box", "rsd")
    def mock_velocity_u_matter(self):
        r"""
        The normalised peculiar velocity field in real space, defined as

        .. math::
            u_i = -\frac{v_i}{\mathcal{H} f}

        where :math:`v_i` is the peculiar velocity, :math:`\mathcal{H}` is the
        conformal Hubble parameter, and :math:`f` is the growth rate.
        """
        if self._mock_velocity_u_matter is None:
            self._mock_velocity_u_matter = self.get_mock_velocity_u_field(
                mock_field=self.mock_matter_field_r
            )
        return self._mock_velocity_u_matter

    @property
    @tagging("cosmo_model", "nu", "mock", "box", "rsd", "tracer_1")
    def mock_velocity_u_tracer_1(self):
        """
        The normalised peculiar velocity field used for the first tracer.
        While the peculiar velocity field is only dependent on the matter field,
        note that if you simulate lognormal tracer fields, the two fields are not exactly correlated.
        One simple way to think about it is that field_1/bias_1 is not equal to field_2/bias_2, because both fields range
        from -1 to +max, and similarly the mock matter field is also not field_1/bias_1 or field_2/bias_2.
        This is an intrinsic weakness of the lognormal mock that you should be aware of.

        As a result, for each tracer the velocity field should be recalculated based on field_i/tracer_bias_i.
        """
        if self._mock_velocity_u_tracer_1 is None:
            self._mock_velocity_u_tracer_1 = self.get_mock_velocity_u_field(
                mock_field=self.mock_tracer_field_1_r / self.tracer_bias_1
            )
        return self._mock_velocity_u_tracer_1

    @property
    @tagging("cosmo_model", "nu", "mock", "box", "rsd", "tracer_2")
    def mock_velocity_u_tracer_2(self):
        """
        The normalised peculiar velocity field used for the second tracer.

        See the docstring of :meth:`mock_velocity_u_tracer_1` for more details.
        """
        if self._mock_velocity_u_tracer_2 is None:
            self._mock_velocity_u_tracer_2 = self.get_mock_velocity_u_field(
                mock_field=self.mock_tracer_field_2_r / self.tracer_bias_2
            )
        return self._mock_velocity_u_tracer_2

    def get_mock_velocity_u_field(self, mock_field):
        r"""
        Generate the normalised peculiar velocity field in real space

        Parameters
        ----------
        mock_field: np.ndarray
            The mock field in real space.

        Returns
        -------
        u_r: np.ndarray
            The normalised peculiar velocity field in real space.
        """
        logger.info(f"invoking {inspect.currentframe().f_code.co_name}")
        real_dtype = real_dtype_from_array(mock_field)
        delta_k = np.fft.rfftn(mock_field, norm="forward")
        complex_dtype = delta_k.dtype
        slicer = get_nd_slicer()
        with np.errstate(divide="ignore", invalid="ignore"):
            kvecoverk2 = np.array(
                [self.k_vec[i][slicer[i]] / self.kmode**2 for i in range(3)]
            ).astype(real_dtype, copy=False)
        kvecoverk2[:, self.kmode == 0] = 0
        u_k = np.asarray(-1j, dtype=complex_dtype) * kvecoverk2 * delta_k[None]
        u_r = np.array(
            [
                np.fft.irfftn(u_k[i], s=mock_field.shape, norm="forward")
                for i in range(3)
            ]
        ).astype(real_dtype, copy=False)
        return u_r

    @property
    @tagging("cosmo_model", "nu", "mock", "box", "rsd")
    def mock_kaiser_field_k_matter(self):
        """
        The Kaiser rsd effect correction for the mock matter field in k-space.
        """
        if self._mock_kaiser_field_k_matter is None:
            self.get_mock_kaiser_field_k(field="matter")
        return self._mock_kaiser_field_k_matter

    @property
    @tagging("cosmo_model", "nu", "mock", "box", "rsd", "tracer_1")
    def mock_kaiser_field_k_tracer_1(self):
        """
        The Kaiser rsd effect correction for the mock tracer field 1 in k-space.

        See the docstring of :meth:`get_mock_kaiser_field_k` for more details.
        """
        if self._mock_kaiser_field_k_tracer_1 is None:
            self.get_mock_kaiser_field_k(field="tracer_1")
        return self._mock_kaiser_field_k_tracer_1

    @property
    @tagging("cosmo_model", "nu", "mock", "box", "rsd", "tracer_2")
    def mock_kaiser_field_k_tracer_2(self):
        """
        The Kaiser rsd effect correction for the mock tracer field 2 in k-space.

        See the docstring of :meth:`get_mock_kaiser_field_k` for more details.
        """
        if self._mock_kaiser_field_k_tracer_2 is None:
            self.get_mock_kaiser_field_k(field="tracer_2")
        return self._mock_kaiser_field_k_tracer_2

    def get_mock_kaiser_field_k(self, field):
        r"""
        Generate the Kaiser rsd effect for the mock matter field in k-space.

        In the parrallel-plane limit, the Kaiser rsd effect is given by:

        .. math::

            \delta_{\rm rsd} = f \mu^2 \delta_k

        where :math:`f` is the growth rate, :math:`\mu` is the cosine of the angle
        between the wave vector and the line of sight, and :math:`\delta_k` is the Fourier
        transform of the real-space matter density field.

        This function returns :math:`\delta_{\rm rsd}` in k-space so that for any mock tracer field,
        the Kaiser effect can be applied by adding :math:`\delta_{\rm rsd}` to the real-space tracer field in k-space.
        """
        u_r = getattr(self, f"mock_velocity_u_{field}")
        slicer = get_nd_slicer()
        real_dtype = real_dtype_from_array(u_r)
        if self.parallel_plane:
            box_coord_l = np.zeros((3,) + tuple(self.box_ndim), dtype=real_dtype)
            box_coord_l[-1] = np.asarray(1.0, dtype=real_dtype)
        else:
            box_coord = np.meshgrid(*self.x_vec, indexing="ij")
            box_coord = np.array(box_coord, dtype=real_dtype) + np.asarray(
                self.box_origin[:, None, None, None], dtype=real_dtype
            )
            box_coord_l = box_coord / np.sqrt((box_coord**2).sum(axis=0))[None]
        self._box_coord_l = box_coord_l
        u_in_xhat = (u_r * box_coord_l).sum(axis=0)
        y_k = np.array(
            [np.fft.rfftn(u_in_xhat * box_coord_l[i], norm="forward") for i in range(3)]
        )
        y_k_dot_k = np.array(
            [(y_k[i] * self.k_vec[i][slicer[i]]) for i in range(3)]
        ).sum(axis=0)
        delta_rsd_k = (
            np.asarray(1j * self.f_growth_true, dtype=y_k_dot_k.dtype) * y_k_dot_k
        )
        logger.info(
            f"{inspect.currentframe().f_code.co_name}: "
            f"setting _mock_kaiser_field_k_{field} "
            f"with parallel_plane={self.parallel_plane}"
        )
        setattr(self, f"_mock_kaiser_field_k_{field}", delta_rsd_k)
        return delta_rsd_k

    def get_mock_matter_field(self):
        """
        Generate the mock matter field in redshift space.
        """
        logger.info(f"invoking {inspect.currentframe().f_code.co_name}, ")
        self._mock_matter_field = self.get_mock_field_in_redshift_space(
            delta_x=self.mock_matter_field_r, field="matter", sigma_v=0
        )

    def get_mock_field_in_redshift_space(self, delta_x, field, sigma_v):
        """
        Generate the mock field in redshift space.

        Parameters
        ----------
        delta_x: np.ndarray
            The mock field in real space.
        field: str
            The field to be simulated.
            Can be "matter" or "tracer_1" or "tracer_2".
        sigma_v: float
            The velocity dispersion of the tracer in km/s.

        Returns
        -------
        delta_x: np.ndarray
            The mock field in redshift space.
        """
        if self.kaiser_rsd:
            logger.info(f"invoking {inspect.currentframe().f_code.co_name}")
            delta_k = np.fft.rfftn(delta_x, norm="forward")
            delta_k += getattr(self, f"mock_kaiser_field_k_{field}")
            mumode = self.mumode
            fog = self.fog_term(
                self.deltav_to_deltar(sigma_v), kmode=self.kmode, mumode=mumode
            )
            delta_k *= fog
            delta_x = np.fft.irfftn(delta_k, s=delta_x.shape, norm="forward")
        return np.asarray(delta_x, dtype=self.real_dtype)

    @property
    def mock_amp_1(self):
        """
        The overall amplitude of the mock tracer field 1 in each grid. This can be T_HI(z) at each grid for example,
        which is multiplied to the simulated density field to get the final mock field 1 ``self.mock_tracer_field_1``.
        If not given, default is to use ``self.mean_amp_1`` as the amplitude.
        """
        if self._mock_amp_1 is None:
            return self.mean_amp_1
        return self._mock_amp_1

    @mock_amp_1.setter
    def mock_amp_1(self, value):
        self._mock_amp_1 = value
        return self._mock_amp_1

    @property
    def mock_amp_2(self):
        """
        The overall amplitude of the mock tracer field 2 in each grid. This can be T_HI(z) at each grid for example,
        which is multiplied to the simulated density field to get the final mock field 2 ``self.mock_tracer_field_2``.
        If not given, default is to use ``self.mean_amp_2`` as the amplitude.
        """
        if self._mock_amp_2 is None:
            return self.mean_amp_2
        return self._mock_amp_2

    @mock_amp_2.setter
    def mock_amp_2(self, value):
        self._mock_amp_2 = value
        return self._mock_amp_2

    @property
    @tagging("cosmo_model", "nu", "mock", "box", "tracer_1", "rsd")
    def mock_tracer_field_1(self):
        """
        The simulated tracer field 1 in redshift space with unit if ``mock_amp_1`` or ``mean_amp_1`` is given.
        """
        if self._mock_tracer_field_1 is None:
            self.get_mock_tracer_field(1)
        mean_amp = self.mock_amp_1
        if isinstance(mean_amp, str):
            mean_amp = getattr(self, mean_amp)
        amp = np.asarray(mean_amp, dtype=self._mock_tracer_field_1.dtype)
        return self._mock_tracer_field_1 * amp

    @property
    @tagging("cosmo_model", "nu", "mock", "box", "tracer_2", "rsd")
    def mock_tracer_field_2(self):
        """
        The simulated tracer field 2 in redshift space with unit if ``mock_amp_2`` or ``mean_amp_2`` is given.
        """
        if self._mock_tracer_field_2 is None:
            self.get_mock_tracer_field(2)
        mean_amp = self.mock_amp_2
        if isinstance(mean_amp, str):
            mean_amp = getattr(self, mean_amp)
        amp = np.asarray(mean_amp, dtype=self._mock_tracer_field_2.dtype)
        return self._mock_tracer_field_2 * amp

    @property
    @tagging("cosmo_model", "nu", "mock", "box", "tracer_1")
    def mock_tracer_field_1_r(self):
        """
        The simulated tracer field 1 **unitsless density contrast** in real space.
        """
        if self._mock_tracer_field_1_r is None:
            self.get_mock_tracer_field_r(1)
        return self._mock_tracer_field_1_r

    @property
    @tagging("cosmo_model", "nu", "mock", "box", "tracer_2")
    def mock_tracer_field_2_r(self):
        """
        The simulated tracer field 2 **unitsless density contrast** in real space.
        """
        if self._mock_tracer_field_2_r is None:
            self.get_mock_tracer_field_r(2)
        return self._mock_tracer_field_2_r

    def get_mock_tracer_field_r(self, tracer_i):
        """
        Generate the mock tracer field in real space.

        Parameters
        ----------
        tracer_i: int
            The index of the tracer. Can be 1 or 2.

        Returns
        -------
        delta_x: np.ndarray
            The mock tracer field in real space.
        """
        delta_x = self.get_mock_field_r(bias=getattr(self, f"tracer_bias_{tracer_i}"))
        logger.info(
            f"{inspect.currentframe().f_code.co_name}: "
            f"seeting _mock_tracer_field_{tracer_i}_r"
        )
        setattr(self, f"_mock_tracer_field_{tracer_i}_r", delta_x)
        return delta_x

    def get_mock_tracer_field(self, tracer_i):
        """
        Generate the mock tracer field in redshift space.

        Parameters
        ----------
        tracer_i: int
            The index of the tracer. Can be 1 or 2.

        Returns
        -------
        delta_x: np.ndarray
            The mock tracer field in redshift space.
        """
        if self.rsd_from_field and self.kaiser_rsd:
            delta_x = self.get_mock_field_in_redshift_space(
                delta_x=getattr(self, f"mock_tracer_field_{tracer_i}_r"),
                field=f"tracer_{tracer_i}",
                sigma_v=getattr(self, f"sigma_v_{tracer_i}"),
            )
        elif self.kaiser_rsd:
            if self.box_ndim is None:
                self.get_enclosing_box()
            power_array = getattr(self, f"auto_power_tracer_{tracer_i}_model_noobs")
            delta_x = self.get_mock_field_from_power(power_array)
        else:
            delta_x = getattr(self, f"mock_tracer_field_{tracer_i}_r")
        logger.info(
            f"{inspect.currentframe().f_code.co_name}: setting _mock_tracer_field_{tracer_i}"
        )
        setattr(self, f"_mock_tracer_field_{tracer_i}", delta_x)
        return delta_x

    @property
    @tagging(
        "cosmo_model", "nu", "mock", "box", "tracer_1", "tracer_2", "discrete", "rsd"
    )
    def mock_tracer_position_in_box(self):
        """
        The simulated tracer positions in the box in real space.
        """
        if self._mock_tracer_position_in_box is None:
            self.get_mock_tracer_position_in_box(self.discrete_base_field)
        return self._mock_tracer_position_in_box

    def get_mock_tracer_position_in_box(self, tracer_i, density_field=None):
        """
        Function to retrieve the tracer positions in redshift space.
        Modified from ``powerbox``.

        Parameters
        ----------
        tracer_i: int
            The index of the tracer. Can be 1 or 2.
        density_field: np.ndarray, default None
            The density field of the tracer. If None, the simulated mock tracer field will be used.

        Returns
        -------
        tracer_positions: np.ndarray
            The tracer positions in the rectangular box.
        """
        rng = default_rng(self.seed)
        # note that `_mock...` does not have mean amplitude so this is what
        # we should be using instead of `mock...`, this is just to invoke
        # the simulation
        getattr(self, "mock_tracer_field_" + str(tracer_i))
        # now actually getting the underlying overdensity
        # if self.rsd_from_field:
        # density_field = getattr(self, "_mock_tracer_field_" + str(tracer_i) + "_r") + 1
        # else:
        #    density_field = getattr(self, "_mock_tracer_field_" + str(tracer_i)) + 1
        if density_field is None:
            density_field = getattr(self, "_mock_tracer_field_" + str(tracer_i))
        density_field = np.array(density_field, copy=True)
        density_field += np.asarray(1.0, dtype=density_field.dtype)
        num_g = self.tot_num_source_in_box
        norm_before = density_field.sum()
        mask = (density_field > 0).astype(density_field.dtype)
        mask_renorm = np.sqrt(power_weights_renorm(mask, mask))
        density_field[density_field < 0] = 0
        # some cheats to renorm the density field
        density_field_new = density_field ** (mask_renorm)
        norm = density_field_new.sum()
        density_field = density_field ** (norm / norm_before)
        norm = density_field.sum()
        if norm > 0:
            density_field *= np.asarray(num_g / norm, dtype=density_field.dtype)
        else:
            density_field[:] = 0
        tracer_positions_list = []
        for z_sel, density_chunk in self._iter_field_los_chunks(density_field):
            dndz_prob_chunk = self._dndz_renorm(self.box_voxel_redshift[..., z_sel])
            tracer_positions_i = self._sample_tracer_positions_from_density_chunk(
                density_chunk=density_chunk,
                dndz_prob_chunk=dndz_prob_chunk,
                z_sel=z_sel,
                rng=rng,
            )
            if tracer_positions_i.size > 0:
                tracer_positions_list.append(tracer_positions_i)
        if len(tracer_positions_list) == 0:
            tracer_positions = np.zeros((0, len(self.box_ndim)), dtype=self.real_dtype)
        else:
            tracer_positions = np.concatenate(tracer_positions_list, axis=0)
            tracer_positions = np.asarray(tracer_positions, dtype=self.real_dtype)
        # for some reason I can not get this to work
        # if self.rsd_from_field and self.kaiser_rsd:
        #    box_coord_l = self._box_coord_l
        #    distance_shift = (
        #        -self.f_growth *
        #        box_coord_l *
        #        (box_coord_l *
        #        getattr(self, f"mock_velocity_u_tracer_{tracer_i}")
        #        ).sum(axis=0)[None]
        #    ).reshape((3,-1)).T
        #    tracer_positions += distance_shift[tracer_which_cell]
        logger.info(
            f"{inspect.currentframe().f_code.co_name}: "
            "setting _mock_tracer_position_in_box"
        )
        self._mock_tracer_position_in_box = tracer_positions

    @property
    @tagging(
        "cosmo_model", "nu", "mock", "box", "tracer_1", "tracer_2", "discrete", "rsd"
    )
    def mock_tracer_position_in_radecz(self):
        """
        The simulated tracer positions projected onto the grid. The tracers outside
        the binary selection window ``map_has_sampling`` or outside the frequency range
        can be trimmed off. Furthermore, excess tracers are also excluded. The selection
        function is stored at ``inside_range``.

        Returns a list in the order of (ra,dec,z,inside_range).
        """
        if self._mock_tracer_position_in_radecz is None:
            self.get_mock_tracer_position_in_radecz()
        return self._mock_tracer_position_in_radecz

    @property
    def ra_mock_tracer(self):
        """
        The RA coordinate of mock galaxies on the sky
        """
        return self.mock_tracer_position_in_radecz[0]

    @property
    def dec_mock_tracer(self):
        """
        The Dec coordinate of mock galaxies on the sky
        """
        return self.mock_tracer_position_in_radecz[1]

    @property
    def z_mock_tracer(self):
        """
        The redshift of mock galaxies
        """
        return self.mock_tracer_position_in_radecz[2]

    @property
    def mock_inside_range(self):
        """
        Whether the mock galaxies are inside the survey area and frequency range
        """
        return self.mock_tracer_position_in_radecz[3]

    def get_mock_tracer_position_in_radecz(self):
        """
        Project the mock tracer positions in the rectangular box to the sky coordinates and redshifts.
        """
        if self.flat_sky:
            self._mock_tracer_comov_dist = (
                self.mock_tracer_position_in_box[:, -1]
                + self.astropy_cosmo_true.comoving_distance(self.z_ch.min()).value
            )
            z_mock_tracer = self.z_as_func_of_comov_dist(self._mock_tracer_comov_dist)
            pos_indx_1 = (
                self.mock_tracer_position_in_box[:, 0]
                / self.box_len[0]
                * self.num_pix_x
            )
            pos_indx_2 = (
                self.mock_tracer_position_in_box[:, 1]
                / self.box_len[1]
                * self.num_pix_y
            )
            ra_mock_tracer, dec_mock_tracer = get_wcs_coor(
                self.wproj, pos_indx_1, pos_indx_2
            )
        else:
            (
                ra_mock_tracer,
                dec_mock_tracer,
                z_mock_tracer,
                tracer_comov_dist,
            ) = self.ra_dec_z_for_coord_in_box(self.mock_tracer_position_in_box)
            self._mock_tracer_comov_dist = tracer_comov_dist
        freq_tracer = redshift_to_freq(z_mock_tracer)
        tracer_ch_id = find_ch_id(freq_tracer, self.nu)
        # num_ch id is for tracer outside the frequency range
        z_sel = tracer_ch_id < len(self.nu)
        radec_sel = (
            angle_in_range(ra_mock_tracer, self.ra_range[0], self.ra_range[1])
            * (dec_mock_tracer > self.dec_range[0])
            * (dec_mock_tracer < self.dec_range[1])
        )
        inside_range = z_sel * radec_sel
        mock_inside_range = inside_range
        logger.info(
            f"invoking {inspect.currentframe().f_code.co_name} with flat_sky={self.flat_sky}: "
            "setting _mock_tracer_position_in_radecz"
        )
        self._mock_tracer_position_in_radecz = (
            ra_mock_tracer,
            dec_mock_tracer,
            z_mock_tracer,
            mock_inside_range,
        )

    def propagate_mock_tracer_to_gal_cat(self, trim=True):
        """
        Propagate the mock tracer positions to the galaxy data catalogue.

        If trim, only tracers inside the ra-dec-z range will be propagated.

        Parameters
        ----------
        trim: bool, default True
            If True, only the mock tracer positions inside the ra-dec-z range will be propagated.
        """
        ra, dec, z, inside_range = self.mock_tracer_position_in_radecz
        inside_range = inside_range.copy()
        # if False, inside_range is all True
        inside_range = (inside_range + 1 - trim) > 0
        self._ra_gal = ra[inside_range]
        self._dec_gal = dec[inside_range]
        self._z_gal = z[inside_range]
        logger.info(
            f"{inspect.currentframe().f_code.co_name}: "
            f"setting ra_gal, dec_gal, z_gal with trim={trim}"
        )

    def _propagate_mock_field_to_data_wcs(self, field, beam=True, average=True):
        """
        Project a box mock field onto the native WCS map grid (optionally beam-smooth).
        """
        if self.sigma_beam_ch is None:
            beam = False
        wproj = self.wproj
        num_pix_x = self.num_pix_x
        num_pix_y = self.num_pix_y
        if self.flat_sky:
            field_dtype = real_dtype_from_array(field)
            map_out = np.zeros(
                (num_pix_x, num_pix_y, self.nu.size),
                dtype=field_dtype,
            )
            for ch_sel, field_chunk in self._iter_field_los_chunks(field):
                map_out[..., ch_sel] = field_chunk
            map_counts = np.ones_like(map_out, dtype=field_dtype)
        else:
            map_out = None
            map_counts = None
            for ch_sel, field_chunk in self._iter_field_los_chunks(field):
                map_i, counts_i = self._project_field_chunk_to_sky_map(
                    field_chunk,
                    ch_sel,
                    average=average,
                    wproj=wproj,
                    num_pix_x=num_pix_x,
                    num_pix_y=num_pix_y,
                )
                if map_out is None:
                    map_out = np.zeros_like(map_i)
                    map_counts = np.zeros_like(counts_i)
                if average:
                    map_out += map_i * counts_i
                else:
                    map_out += map_i
                map_counts += counts_i
                del map_i, counts_i
            if average:
                with np.errstate(divide="ignore", invalid="ignore"):
                    map_out = np.nan_to_num(map_out / map_counts)
        if beam:
            map_out = self._apply_weighted_convolution_in_batches(
                map_out,
                map_counts,
                beam_image=None,
                wproj=wproj,
                num_pix_x=num_pix_x,
                num_pix_y=num_pix_y,
            )
        return map_out

    def _propagate_mock_field_to_data_healpix(self, field, beam=True, average=True):
        """Placeholder until HEALPix mock field propagation is implemented."""
        raise NotImplementedError(
            "propagate_mock_field_to_data for HEALPix sky maps is not implemented yet."
        )

    def propagate_mock_field_to_data(
        self, field, beam=True, highres_sim=None, average=True
    ):
        """
        Grid the mock tracer field onto the sky map cube.

        Parameters
        ----------
        field: np.ndarray
            The mock tracer field to be gridded.
        beam: bool, default True
            Whether to convolve the beam model to the sky map.
        highres_sim: int, default None
            **Deprecated for WCS.** If this argument or ``self.highres_sim`` is not
            ``None`` while the sky map uses WCS, a ``DeprecationWarning`` is emitted
            and the factor is ignored; gridding always uses the native map resolution.
        average: bool, default True
            If True, then the map pixel value will be the average of the rectangular box values (i.e. the mock field is in temperature units).
            If False, then the map pixel value will be the sum of the rectangular box values (i.e. the mock field is in flux density units).

        Returns
        -------
        np.ndarray
            The output sky map cube.
        """
        fmt = self.skymap.format
        if fmt == "wcs":
            resolved = highres_sim if highres_sim is not None else self.highres_sim
            if resolved is not None:
                warnings.warn(
                    "highres_sim is deprecated for WCS sky maps and is ignored in "
                    "propagate_mock_field_to_data; use native-resolution gridding.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            return self._propagate_mock_field_to_data_wcs(
                field, beam=beam, average=average
            )
        if fmt == "healpix":
            return self._propagate_mock_field_to_data_healpix(
                field, beam=beam, average=average
            )


class HIGalaxySimulation(MockSimulation):
    """
    The class for generating mock HI galaxy catalogues for HI emission line observations.

    Parameters
    ----------
    no_vel: bool, default True
        If True, HI sources will have no velocity width, and the total flux will be
        allocated to the central frequency channel.
    tf_slope: float, default None
        The slope of the Tully-Fisher relation when ``no_vel`` is False.
    tf_zero: float, default None
        The intercept of the Tully-Fisher relation when ``no_vel`` is False.
    halo_model: :class:`halomod.TracerHaloModel`, default None
        The halo model to be used. Only used if ``hi_mass_from`` is "hod".
    hi_mass_from: str, default "hod"
        The method for calculating the HI mass of the mock tracers.
        Can be "hod" or "himf".
    himf_pars: dict, default None
        The parameters for the HI mass function. Only used if ``hi_mass_from`` is "himf".
    num_ch_ext_on_each_side: int, default 5
        The number of frequency channels to be extended on each side of the central frequency channel
        when generating the HI profile. A good rule of thumb is 200km/s over ``self.vel_resol``.
    **params: dict
        Additional parameters to be passed to the base class :class:`meer21cm.mock.MockSimulation`.
    """

    def __init__(
        self,
        no_vel=True,
        tf_slope=None,
        tf_zero=None,
        halo_model=None,
        hi_mass_from="hod",
        himf_pars=None,
        num_ch_ext_on_each_side=5,
        **params,
    ):
        super().__init__(**params)
        # has to have tracer 2 for sim
        if self.tracer_bias_2 is None:
            self.tracer_bias_2 = 1.0
        self.no_vel = no_vel
        self.tf_slope = tf_slope
        self.tf_zero = tf_zero
        if halo_model is None:
            halo_model = THM(
                cosmo_model=self.astropy_cosmo_true,
                z=self.z,
                hod_model=Obuljen18,
            )
        self.halo_model = halo_model
        self.hi_mass_from = hi_mass_from.lower()
        if himf_pars is None:
            himf_pars = himf_pars_jones18(self.h / 0.7)
        self.himf_pars = himf_pars
        self.num_ch_ext_on_each_side = num_ch_ext_on_each_side
        init_attr = [
            "_halo_mass_mock_tracer",
            "_hi_mass_mock_tracer",
            "_hi_profile_mock_tracer",
        ]
        for attr in init_attr:
            setattr(self, attr, None)

    @property
    def hi_mass_from(self):
        """
        Methods for calculating HI mass.
        Can either be 'hod' or 'himf'.
        """
        return self._hi_mass_from

    @hi_mass_from.setter
    def hi_mass_from(self, value):
        self._hi_mass_from = value
        if "himass_dep_attr" in dir(self):
            self.clean_cache(self.himass_dep_attr)

    @property
    def no_vel(self):
        """
        If True, HI sources will have no velocity width.
        """
        return self._no_vel

    @no_vel.setter
    def no_vel(self, value):
        self._no_vel = value
        if "hivel_dep_attr" in dir(self):
            self.clean_cache(self.hivel_dep_attr)

    @property
    def tf_slope(self):
        """
        Slope of Tully-Fisher relation.
        See :func:`meer21cm.util.tully_fisher`.
        """
        return self._tf_slope

    @tf_slope.setter
    def tf_slope(self, value):
        self._tf_slope = value
        if "hivel_dep_attr" in dir(self):
            self.clean_cache(self.hivel_dep_attr)

    @property
    def tf_zero(self):
        """
        The intercept of the T-F relation. See :func:`meer21cm.util.tully_fisher`.
        """
        return self._tf_zero

    @tf_zero.setter
    def tf_zero(self, value):
        self._tf_zero = value
        if "hivel_dep_attr" in dir(self):
            self.clean_cache(self.hivel_dep_attr)

    @property
    def halo_model(self):
        """
        A :class:`halomod.TracerHaloModel` object for storing the halo model.
        """
        return self._halo_model

    @halo_model.setter
    def halo_model(self, value):
        self._halo_model = value
        if "hm_dep_attr" in dir(self):
            self.clean_cache(self.hm_dep_attr)

    @property
    @tagging(
        "hm",
        "cosmo_model",
        "nu",
        "mock",
        "box",
        "tracer_1",
        "tracer_2",
        "discrete",
        "rsd",
    )
    def halo_mass_mock_tracer(self):
        """
        The halo mass of the mock tracers in log10 M_sun/h.
        """
        if self._halo_mass_mock_tracer is None:
            self.get_halo_mass_mock_tracer()
        return self._halo_mass_mock_tracer

    def get_halo_mass_mock_tracer(self):
        """
        Calculate the halo mass of the mock tracers.

        In this case, the discrete tracer is considered to be the halos instead of galaxies,
        so corroespondingly you should set ``self.num_discrete_source`` to the number of halos,
        and ``self.tracer_bias_2`` to the effective bias of the halo number density field.

        The sampling is then done by using the halo mass function stored in ``self.halo_model``.
        See ``hmf`` package for more details about the halo mass function.
        """
        # propagate mock catalogue to galaxy catalogue
        self.propagate_mock_tracer_to_gal_cat()
        num_g_tot_in_mockrange = self.ra_gal.size
        num_g_tot_in_map = self.ra_mock_tracer.size
        hm = self.halo_model
        dlog10m = hm.dlog10m
        m_arr_inv = hm.m[::-1]
        # in (Mpc/h)^-3
        n_halo = np.cumsum((hm.dndlog10m * dlog10m)[::-1])
        # in Mpc^-3
        n_halo *= self.h**3
        dV_pix = self.pix_resol_in_mpc**2 * self.los_resol_in_mpc
        m_indx_mock = np.where(
            (n_halo * dV_pix * self.W_HI.sum()) < num_g_tot_in_mockrange
        )[0].max()
        logm_halo_min = np.log10(m_arr_inv)[m_indx_mock]
        dndlog10_fnc = interp1d(np.log10(hm.m), hm.dndlog10m)
        self._halo_mass_mock_tracer = sample_from_dist(
            dndlog10_fnc,
            logm_halo_min,
            np.log10(hm.m.max()),
            size=num_g_tot_in_map,
            seed=self.seed,
        )

    @property
    @tagging(
        "hm",
        "cosmo_model",
        "nu",
        "mock",
        "box",
        "tracer_1",
        "tracer_2",
        "discrete",
        "rsd",
        "himass",
    )
    def hi_mass_mock_tracer(self):
        """
        The HI mass of the mock tracers in log10 M_sun.
        """
        if self._hi_mass_mock_tracer is None:
            getattr(self, "get_hi_mass_mock_tracer" + "_" + self.hi_mass_from)()
        return self._hi_mass_mock_tracer

    def get_hi_mass_mock_tracer_hod(self):
        """
        Calculate the HI mass of the mock tracers using the halo occupation distribution
        based on the HOD model stored in ``self.halo_model``.

        See ``halomod`` package for more details about the HOD model.
        """
        himass_g = (
            self.halo_model.hod.total_occupation(10**self.halo_mass_mock_tracer)
            / self.h
        )  # no h
        self._hi_mass_mock_tracer = np.log10(himass_g)

    @property
    @tagging(
        "hm",
        "cosmo_model",
        "nu",
        "mock",
        "box",
        "tracer_1",
        "tracer_2",
        "discrete",
        "rsd",
        "himass",
        "hivel",
    )
    def hi_profile_mock_tracer(self):
        """
        The emission line profiles of the mock tracers in Jansky
        """
        if self._hi_profile_mock_tracer is None:
            self.get_hi_profile_mock_tracer()
        return self._hi_profile_mock_tracer

    def get_hi_profile_mock_tracer(self):
        """
        Calculate the emission line profiles of the mock tracers.

        The profiles are calculated using the HI mass of the mock tracers,
        and the Tully-Fisher relation stored in ``self.tf_slope`` and ``self.tf_zero``.
        """
        hifluxd_ch = hi_mass_to_flux_profile(
            self.hi_mass_mock_tracer,
            self.z_mock_tracer,
            self.nu,
            self.tf_slope,
            self.tf_zero,
            cosmo=self.astropy_cosmo_true,
            seed=self.seed,
            no_vel=self.no_vel,
            num_ch_ext_on_each_side=self.num_ch_ext_on_each_side,
        )
        self._hi_profile_mock_tracer = hifluxd_ch

    def propagate_hi_profile_to_map(
        self, return_highres=False, beam=True, beam_image=None
    ):
        """
        Project the ``hi_profile_mock_tracer`` onto sky maps and convolve with the beam (if ``beam``).
        If ``return_highres``, the returned map is the higher resolution map
        specified by ``highres_sim``. If not, the map will be downsampled to the original resolution.
        """
        if self.sigma_beam_ch is None and beam_image is None:
            beam = False
        highres = self.highres_sim
        num_pix_x = self.num_pix_x
        num_pix_y = self.num_pix_y
        hifluxd_ch = self.hi_profile_mock_tracer
        if highres is None:
            # no need for downsampling
            return_highres = True
            wproj = self.wproj
        else:
            wproj = create_udres_wproj(self.wproj, highres)
            num_pix_x *= highres
            num_pix_y *= highres

        indx_0, indx_1 = radec_to_indx(
            self.ra_mock_tracer, self.dec_mock_tracer, wproj, to_int=False
        )
        nu_ext = self.nu.copy()
        for i in range(self.num_ch_ext_on_each_side * 2):
            nu_ext = center_to_edges(nu_ext)
        indx_z = find_ch_id(redshift_to_freq(self.z_mock_tracer), nu_ext)
        num_ch_vel = hifluxd_ch.shape[0] // 2
        hi_map_ext_in_jy = np.zeros(
            (num_pix_x, num_pix_y, len(nu_ext)), dtype=hifluxd_ch.dtype
        )
        for i, indx_diff in enumerate(
            np.linspace(-num_ch_vel, num_ch_vel, 2 * num_ch_vel + 1).astype("int")
        ):
            hiflux_i = hifluxd_ch[i]
            indx_2 = indx_z + indx_diff
            indx_bins = [
                np.arange(hi_map_ext_in_jy.shape[i] + 1) - 0.5 for i in range(3)
            ]
            map_i, _ = np.histogramdd(
                np.array([indx_0, indx_1, indx_2]).T,
                bins=indx_bins,
                weights=hiflux_i,
            )
            hi_map_ext_in_jy += map_i
        # remove the excess channels used for taking into account of galaxies
        # whose centres are outside the frequency range but the profile tails are inside
        hi_map_in_jy = hi_map_ext_in_jy[
            :, :, self.num_ch_ext_on_each_side : -self.num_ch_ext_on_each_side
        ]
        if beam:
            if beam_image is None:
                map_counts = np.ones_like(hi_map_in_jy, dtype=hi_map_in_jy.dtype)
                hi_map_in_jy = self._apply_weighted_convolution_in_batches(
                    hi_map_in_jy,
                    map_counts,
                    beam_image=None,
                    wproj=wproj,
                    num_pix_x=num_pix_x,
                    num_pix_y=num_pix_y,
                )
            else:
                map_counts = np.ones_like(hi_map_in_jy, dtype=hi_map_in_jy.dtype)
                hi_map_in_jy = self._apply_weighted_convolution_in_batches(
                    hi_map_in_jy,
                    map_counts,
                    beam_image=beam_image,
                )
        if return_highres:
            return hi_map_in_jy
        spec = Specification(
            wproj=wproj,
            num_pix_x=num_pix_x,
            num_pix_y=num_pix_y,
        )
        ra_map = spec.ra_map
        dec_map = spec.dec_map
        hi_map_in_jy = sample_map_from_highres(
            hi_map_in_jy,
            ra_map,
            dec_map,
            self.wproj,
            self.num_pix_x,
            self.num_pix_y,
            average=False,
        )
        return hi_map_in_jy


def hi_mass_to_flux_profile(
    loghimass,
    z_g,
    nu,
    tf_slope=None,
    tf_zero=None,
    cosmo=Planck18,
    seed=None,
    num_ch_ext_on_each_side=5,
    internal_step=1001,
    no_vel=True,
):
    r"""
    Convert HI mass to emission line profile.

    The relation between mass and flux (flux density integrated) of a HI source can be written as [1]

    .. math::
        M_{\rm HI} = \frac{16\pi m_H}{3 h f_{21} A_{10}}\, D_L^2\, S,

    where :math:`M_{\rm HI}` is the HI mass, :math:`m_H` is the mass of neutral hydrogen atom,
    :math:`h` is the Planck constant, :math:`f_{21}` is the rest frequency of 21cm line,
    :math:`A_{10}` is the spontaneous emission rate of 21cm line,
    :math:`D_L` is the luminosity distance of the source, and
    :math:`S` is the flux.

    If ``no_vel`` is set to ``False``, the w50 parameters of the HI sources are calculated using
    a Tully-Fisher relation. Random inclinations and busy function parameters are assigned to the sources.
    The busy functions are then used as the emission profiles. The profiles are gridded into frequency channels.
    The gridded profile, ``hifluxd_ch`` has the shape of (ch_offset,num_source), with the zeroth axis corresponding
    to the channel offset, and varies from (-N,-N+1,...,0,...,N-1,N).

    Parameters
    ----------
    loghimass: array
        The HI mass of sources in log10 solmar mass (**no h**).
    z_g: array
        The redshifts of the sources
    nu: array
        The frequency channels in Hz.
    tf_slope: float, default None.
        The slope of the T-F relation. See :func:`meer21cm.util.tully_fisher`.
    tf_zero: float, default None.
        The intercept of the T-F relation. See :func:`meer21cm.util.tully_fisher`.
    cosmo: optional, default Planck18.
        The cosmology used.
    seed: optional, default None.
        The seed number for rng.
    num_ch_ext_on_each_side: optional, default 5.
        Internal parameter, no need to change.
    internal_step: optional, default 1001.
        Internal parameter, decrease for less accuracy.
        Can be lower when velocity resolution is low.
    no_vel: optional, default True.
        If True, source will have no emission line profile and just a delta peak.

    Returns
    -------
    hifluxd_ch: array
        The flux density of each source in Jy.

    References
    ----------
    .. [1] Meyer et al., "Tracing HI Beyond the Local Universe", https://arxiv.org/abs/1705.04210
    """
    rng = np.random.default_rng(seed=seed)
    # internally allowing some extension so galaxies
    # outside the frequency range can be accurately calculated
    # so the tail of profile can be correctly distributed into the range
    nu_ext = nu.copy()
    for i in range(num_ch_ext_on_each_side * 2):
        nu_ext = center_to_edges(nu_ext)
    num_g = loghimass.size
    lumi_dist_g = cosmo.luminosity_distance(z_g).to("Mpc").value
    # in Jy Hz
    hiintflux_g = 10**loghimass / mass_intflux_coeff / lumi_dist_g**2
    freq_resol = np.diff(nu).mean()
    # no velocity, just one channel
    if no_vel:
        hifluxd_ch = hiintflux_g / freq_resol
        return hifluxd_ch[None, :]
    # get w_50
    hivel_g = tully_fisher(10**loghimass / 1.4, tf_slope, tf_zero, inv=True)
    incli_g = np.abs(np.sin(rng.uniform(0, 2 * np.pi, size=num_g)))
    # get width
    hiwidth_g = incli_g * hivel_g
    # in km/s/freq
    dvdf = (constants.c / nu).to("km/s").value.mean()
    vel_resol = dvdf * np.diff(nu).mean()
    # get the number of channels needed to plot the profile
    num_ch_vel = (int(hiwidth_g.max() / vel_resol)) // 2 + 2
    # in terms of frequency offset
    freq_ch_arr = np.linspace(-num_ch_vel, num_ch_vel, 2 * num_ch_vel + 1) * freq_resol
    # busy function parameters
    busy_c = 10 ** (rng.uniform(-3, -2, size=num_g))
    busy_b = 10 ** (rng.uniform(-2, 0, size=num_g))
    # fine resolution velocity offset points for calculating the profile
    vel_int_arr = np.linspace(-hiwidth_g.max(), hiwidth_g.max(), num=internal_step)
    hiprofile_g = busy_function_simple(
        vel_int_arr[:, None],
        1,
        busy_b,
        (busy_c / hiwidth_g)[None, :] * 2,
        hiwidth_g[None, :] / 2,
    )
    # the sum over profile should give the integrated flux
    hiprofile_g = (
        hiprofile_g / (np.sum(hiprofile_g, axis=0))[None, :] * hiintflux_g[None, :]
    )
    gal_freq = redshift_to_freq(z_g)
    gal_which_ch = find_ch_id(gal_freq, nu_ext)
    # the fine resolution point in Hz
    freq_int_arr = (
        vel_int_arr[None, :] * gal_freq[:, None] / constants.c.to("km/s").value
    )
    hicumflux_g = np.cumsum(hiprofile_g, axis=0)
    # central freq of a galaxy relative to the frequency channel
    freq_start_pos = np.zeros_like(gal_freq)
    inrange_sel = gal_which_ch < nu_ext.size
    freq_start_pos[inrange_sel] = (
        nu_ext[gal_which_ch[inrange_sel]] - gal_freq[inrange_sel] - freq_resol / 2
    )
    freq_gal_arr = freq_ch_arr[:, None] - freq_start_pos[None, :]
    freq_indx = np.argmin(
        np.abs(freq_gal_arr[:, :, None] - freq_int_arr[None, :, :]).reshape(
            (-1, freq_int_arr.shape[-1])
        ),
        axis=1,
    )
    freq_indx = freq_indx.reshape(freq_gal_arr.shape)
    hifluxd_ch = np.zeros(freq_indx.shape)
    for i in range(num_g):
        hifluxd_ch[:, i] = hicumflux_g[:, i][freq_indx[:, i]]
    hifluxd_ch = np.diff(hifluxd_ch, axis=0)
    hifluxd_ch = np.concatenate((np.zeros(num_g)[None, :], hifluxd_ch), axis=0)
    # from Jy Hz to Jy
    hifluxd_ch /= freq_resol
    return hifluxd_ch


def generate_gaussian_field(
    box_ndim, box_len, power_spectrum, seed, ps_has_volume=True
):
    r"""
    Generate a Gaussian field with the given power spectrum.
    If ``ps_has_volume`` is ``True``, the power spectrum is assumed to have the volume unit,
    usually in the unit of :math:`(Mpc)^{-3}`.

    The length unit of the box is arbitrary, as long as the unit in ``box_len`` and
    ``power_spectrum`` are consistent.

    Parameters
    ----------
    box_ndim: array
        The number of grid points in each dimension.
    box_len: array
        The length of the box in each dimension.
    power_spectrum: array
        The input power spectrum.
    seed: int
        The seed for the random number generator.
    ps_has_volume: bool, default True
        If ``True``, the power spectrum is assumed to have the volume unit.

    Returns
    -------
    delta_x: array
        The generated Gaussian field.
    """
    ps = np.asarray(power_spectrum)
    real_dtype = real_dtype_from_array(ps)
    rng = np.random.default_rng(seed)
    noise_real = rng.normal(0, 1, box_ndim).astype(real_dtype, copy=False)
    noise_k = np.fft.rfftn(noise_real)
    ps = ps.astype(real_dtype, copy=True)
    if ps_has_volume:
        scale = np.asarray(np.prod(box_ndim) / np.prod(box_len), dtype=real_dtype)
        ps *= scale
    scaling = np.sqrt(ps)
    delta_k = noise_k * scaling

    # Inverse FFT to get real-valued Gaussian field
    delta_x = np.fft.irfftn(delta_k, axes=range(len(box_ndim)), s=box_ndim)
    return delta_x.astype(real_dtype, copy=False)


def generate_lognormal_field(
    box_ndim, box_len, power_spectrum, seed, ps_has_volume=True
):
    r"""
    Generate a lognormal field with the given power spectrum.
    If ``ps_has_volume`` is ``True``, the power spectrum is assumed to have the volume unit,
    usually in the unit of :math:`(Mpc)^{-3}`.

    The length unit of the box is arbitrary, as long as the unit in ``box_len`` and
    ``power_spectrum`` are consistent.

    Parameters
    ----------
    box_ndim: array
        The number of grid points in each dimension.
    box_len: array
        The length of the box in each dimension.
    power_spectrum: array
        The input power spectrum.
    seed: int
        The seed for the random number generator.
    ps_has_volume: bool, default True
        If ``True``, the power spectrum is assumed to have the volume unit.

    Returns
    -------
    delta_x: array
        The generated lognormal field.
    """
    ps = np.asarray(power_spectrum)
    real_dtype = real_dtype_from_array(ps)
    ps = ps.astype(real_dtype, copy=True)
    if ps_has_volume:
        scale = np.asarray(np.prod(box_ndim) / np.prod(box_len), dtype=real_dtype)
        ps *= scale
    # Compute correlation function ξ_δ(r) via inverse FFT
    xi_delta = np.fft.irfftn(ps, axes=(0, 1, 2), s=box_ndim).real.astype(
        real_dtype, copy=False
    )
    # Compute Gaussian correlation function ξ_G(r) = ln(1 + ξ_δ(r))
    eps = np.asarray(1e-10, dtype=real_dtype)
    xi_G = np.log(np.asarray(1.0, dtype=real_dtype) + xi_delta + eps)  # Avoid log(0)
    # Compute Gaussian power spectrum Delta_G(k)
    Delta_G = np.abs(np.fft.rfftn(xi_G))
    delta_x_g = generate_gaussian_field(
        box_ndim, box_len, Delta_G, seed, ps_has_volume=False
    )
    # Apply lognormal transformation
    sigma_sq = np.asarray(np.mean(Delta_G.ravel()[1:]), dtype=real_dtype)
    half = np.asarray(0.5, dtype=real_dtype)
    one = np.asarray(1.0, dtype=real_dtype)
    delta_x = np.exp(delta_x_g - sigma_sq * half) - one
    return delta_x.astype(real_dtype, copy=False)


def generate_colored_noise(x_size, x_len, power_spectrum, seed=None):
    """
    Generate random 1D gaussian fluctuations following a specific spectrum.
    This is similar to ``colorednoise`` package for generating colored noise.
    It is simply wrapping the ``generate_gaussian_field`` function under the hood.
    Note that the Fourier convention used should be consistent with :py:mod:`np.fft`,
    and the power spectrum is dimensionless.

    Parameters
    ----------
        x_size: int
            The number of sampling.
        x_len: float
            The **total length** of the sampling.
        power_spectrum: array
            The power spectrum of the random noise in Fourier space.
        seed: int, default None
            The seed number for random generator for sampling. If None, a random seed is used.

    Returns
    -------
        rand_arr: float array.
            The random noise.
    """

    rand_arr = generate_gaussian_field(
        x_size, x_len, power_spectrum, seed, ps_has_volume=False
    )
    rand_arr -= rand_arr.mean()
    return rand_arr
