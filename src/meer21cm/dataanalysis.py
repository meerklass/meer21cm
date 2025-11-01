"""
This module contains the base class for reading and visualizing the map data cube.

Note that, the defined class, :py:class:`Specification`, is the base class for reading and visualizing the map data cube.
It is typically used as a base class for other classes that inherit from it, and not used directly.
"""


import numpy as np
from astropy.io import fits
from meer21cm.util import (
    check_unit_equiv,
    get_wcs_coor,
    freq_to_redshift,
    f_21,
    center_to_edges,
    find_ch_id,
    tagging,
    find_property_with_tags,
    angle_in_range,
    create_wcs,
)
from astropy import constants, units
from meer21cm.io import (
    cal_freq,
    read_map,
    filter_incomplete_los,
    read_pickle,
)
from astropy.wcs.utils import proj_plane_pixel_area
from itertools import chain
import meer21cm
from meer21cm.telescope import *
import meer21cm.telescope as telescope
import logging
import numbers
import inspect

logger = logging.getLogger(__name__)

default_data_dir = meer21cm.__file__.rsplit("/", 1)[0] + "/data/"

default_nu = {
    "meerkat_L": cal_freq(np.arange(4096) + 1, band="L"),
    "meerkat_UHF": cal_freq(np.arange(4096) + 1, band="UHF"),
    "meerklass_2021_L": cal_freq(np.arange(4096) + 1, band="L"),
    "meerklass_2019_L": cal_freq(np.arange(4096) + 1, band="L"),
    "meerklass_UHF": cal_freq(np.arange(4096) + 1, band="UHF"),
}


class Specification:
    """
    Base class for reading and visualizing the map data cube.

    Parameters
    ----------
    nu: np.ndarray, default None
        The frequencies of the survey in Hz.
    wproj: :py:class:`astropy.wcs.WCS`, default None
        The WCS object for the map.
    num_pix_x: int, default None
        The number of pixels in the first axis of the map data.
    num_pix_y: int, default None
        The number of pixels in the second axis of the map data.
    map_has_sampling: np.ndarray, default None
        A binary window for whether a pixel has been sampled.
    sigma_beam_ch: np.ndarray, default None
        The beam size parameter for each frequency channel.
    beam_unit: :py:class:`astropy.units.Unit`, default :py:class:`astropy.units.deg`
        The unit of the beam size parameter.
    map_unit: :py:class:`astropy.units.Unit`, default :py:class:`astropy.units.K`
        The unit of the map data.
    map_file: str, default None
        The file path of the map data. Supports automatic reading of the MeerKLASS L-band data.
        For UHF data use `pickle_file` for the file path of the pickle file.
    counts_file: str, default None
        The file path of the hit counts data. Supports automatic reading of the MeerKLASS L-band data.
        For UHF data use `pickle_file` for the file path of the pickle file.
    pickle_file: str, default None
        The file path of the pickle file. Supports automatic reading of the MeerKLASS UHF data.
    los_axis: int, default -1
        The axis of the map data that corresponds to the line of sight.
        **Warning**: Tranposing the data to align the los axis is not properly taken care of in the code.
        If your map los axis is not the last axis, it is recommended to manually transpose the data so that
        the los axis is the last axis.
    nu_min: float, default None,
        The minimum frequency of the survey in Hz. Data below this frequency will be clipped.
    nu_max: float, default None,
        The maximum frequency of the survey in Hz. Data above this frequency will be clipped.
    filter_map_los: bool, default True
        Whether to filter the map data along the line of sight. See :meth:`meer21cm.io.filter_incomplete_los`
    gal_file: str, default None,
        The file path of the galaxy catalogue.
    weighting: str, default "counts"
        The weighting scheme for the map data.
    ra_range: tuple, default (0, 360)
        The range of the right ascension of the map data in degrees. Data outside this range will be masked.
    dec_range: tuple, default (-90, 90)
        The range of the declination of the map data in degrees. Data outside this range will be masked.
    beam_model: str, default "gaussian"
        The shape of the beam.
    data: np.ndarray, default None
        The map data.
    weights_map_pixel: np.ndarray, default None
        The weights per map pixel.
    counts: np.ndarray, default None
        The number of hits per pixel for the map data.
    survey: str, default ""
        The survey name.
    band: str, default ""
        The band of the survey.
    z_interp_max: float, default 6.0
        The maximum redshift to interpolate the redshift as a function of comoving distance.
        See :meth:`meer21cm.dataanalysis.Specification.get_z_as_func_of_comov_dist`.
    soft_filter_los: bool, default True
        If `filter_map_los` is True, whether to use a soft criterion.
        If False, any line of sight that is not 100% sampled will be removed.
        If True, the maximum sampling fraction of the map cube is calculated and used as the criterion.
        See :meth:`meer21cm.io.filter_incomplete_los`.
    """

    def __init__(
        self,
        nu=None,
        wproj=None,
        num_pix_x=None,
        num_pix_y=None,
        map_has_sampling=None,
        sigma_beam_ch=None,
        beam_unit=units.deg,
        map_unit=units.K,
        map_file=None,
        counts_file=None,
        pickle_file=None,
        los_axis=-1,
        nu_min=None,
        nu_max=None,
        filter_map_los=True,
        gal_file=None,
        weighting="counts",
        ra_range=(0, 360),
        dec_range=(-90, 90),
        beam_model="gaussian",
        data=None,
        weights_map_pixel=None,
        counts=None,
        survey="",
        band="",
        z_interp_max=6.0,
        soft_filter_los=True,
        **kwparams,
    ):
        self.survey = survey
        self.band = band
        spec_key = survey + "_" + band
        if spec_key in default_nu.keys():
            logger.info(
                f"found {spec_key} in predefined settings, using default settings"
                " and override the following parameters:"
                " nu, nu_min, nu_max, num_pix_x, num_pix_y, wproj",
            )
            nu = default_nu[spec_key]
            nu_min = default_nu_min[spec_key]
            nu_max = default_nu_max[spec_key]
            num_pix_x = default_num_pix_x[spec_key]
            num_pix_y = default_num_pix_y[spec_key]
            wproj = default_wproj[spec_key]
        self.dependency_dict = find_property_with_tags(self)
        funcs = list(chain.from_iterable(list(self.dependency_dict.values())))
        for func_i in np.unique(np.array(funcs)):
            setattr(self, func_i + "_dep_attr", [])
        for dep_attr, inp_func in self.dependency_dict.items():
            for func in inp_func:
                old_dict = getattr(self, func + "_dep_attr")
                setattr(
                    self,
                    func + "_dep_attr",
                    old_dict
                    + [
                        "_" + dep_attr,
                    ],
                )
        self.map_file = map_file
        self.counts_file = counts_file
        self.pickle_file = pickle_file
        self.los_axis = los_axis
        sel_nu = True
        if nu is None:
            nu = np.array([f_21 - 1, f_21])
            sel_nu = False
        if nu_min is None:
            nu_min = -np.inf
        if nu_max is None:
            nu_max = np.inf
        nu_sel = (nu > nu_min) * (nu < nu_max)
        if sel_nu:
            if nu_sel.sum() == 0:
                raise ValueError("input nu is not in the range of nu_min and nu_max")
            self.nu = nu[nu_sel]
        else:
            self.nu = nu
        self.nu_min = nu_min
        self.nu_max = nu_max
        if num_pix_x is None:
            num_pix_x = 3
        if num_pix_y is None:
            num_pix_y = 3
        if wproj is None:
            wproj = create_wcs(0.0, 0.0, [num_pix_x, num_pix_y], 1.0)

        self.wproj = wproj
        self.num_pix_x = num_pix_x
        self.num_pix_y = num_pix_y
        self.sigma_beam_ch = sigma_beam_ch
        self.beam_unit = beam_unit
        if map_has_sampling is None:
            map_has_sampling = np.ones(
                (num_pix_x, num_pix_y, len(self.nu)), dtype="bool"
            )
            map_has_sampling[0] = False
            map_has_sampling[-1] = False
            map_has_sampling[:, 0] = False
            map_has_sampling[:, -1] = False
        self.map_has_sampling = map_has_sampling
        self.map_unit = map_unit
        self.map_unit_type
        xx, yy = np.meshgrid(np.arange(num_pix_x), np.arange(num_pix_y), indexing="ij")
        # the coordinates of each pixel in the map
        self._ra_map, self._dec_map = get_wcs_coor(wproj, xx, yy)
        self.__dict__.update(kwparams)
        self.filter_map_los = filter_map_los
        self.soft_filter_los = soft_filter_los
        self.gal_file = gal_file
        self.weighting = weighting
        self.ra_range = ra_range
        self.dec_range = dec_range
        self._sigma_beam_ch_in_mpc = None
        if data is None:
            data = np.zeros(self.map_has_sampling.shape)
        self.data = data
        if weights_map_pixel is None:
            weights_map_pixel = np.ones(self.map_has_sampling.shape)
            weights_map_pixel[0] = 0.0
            weights_map_pixel[-1] = 0.0
            weights_map_pixel[:, 0] = 0.0
            weights_map_pixel[:, -1] = 0.0
        self.weights_map_pixel = weights_map_pixel
        if counts is None:
            counts = np.ones(self.map_has_sampling.shape)
        self.counts = counts
        self.trim_map_to_range()
        self.beam_type = None
        self.beam_model = beam_model
        self._beam_image = None
        self._z_as_func_of_comov_dist = None
        self.z_interp_max = z_interp_max

    @property
    def map_unit_type(self):
        """
        The type of the map unit. If the map unit is temperature, return "T".
        If the map unit is flux density, return "F".
        If the map unit is not temperature or flux density, raise an error.
        """
        map_unit = self.map_unit
        if not check_unit_equiv(map_unit, units.K):
            if not check_unit_equiv(map_unit, units.Jy):
                raise (
                    ValueError,
                    "map unit has be to either temperature or flux density.",
                )
            else:
                map_unit_type = "F"
        else:
            map_unit_type = "T"
        return map_unit_type

    def clean_cache(self, attr):
        """
        Set the attributes to None.
        This is used to clear the cache of the attributes.
        """
        for att in attr:
            if att in self.__dict__.keys():
                setattr(self, att, None)

    @property
    def beam_type(self):
        """
        The beam type that can be either be
        isotropic or anisotropic.
        """
        return self._beam_type

    @beam_type.setter
    def beam_type(self, value):
        self._beam_type = value
        if "beam_dep_attr" in dir(self):
            self.clean_cache(self.beam_dep_attr)

    @property
    def beam_model(self):
        """
        The name of the beam function.
        """
        return self._beam_model

    @beam_model.setter
    def beam_model(self, value):
        beam_func = value + "_beam"
        if beam_func not in telescope.__dict__.keys():
            raise ValueError(f"{value} is not a beam model")
        self._beam_model = value
        self.beam_type = getattr(telescope, value + "_beam").tags[0]
        if "beam_dep_attr" in dir(self):
            self.clean_cache(self.beam_dep_attr)

    @property
    def beam_unit(self):
        """
        The unit of input beam size parameter sigma
        """
        return self._beam_unit

    @beam_unit.setter
    def beam_unit(self, value):
        self._beam_unit = value
        if "beam_dep_attr" in dir(self):
            self.clean_cache(self.beam_dep_attr)

    @property
    def sigma_beam_ch(self):
        """
        The input beam size parameter sigma for each channel.
        If one number is provided, it will be used for all channels.
        """
        return self._sigma_beam_ch

    @sigma_beam_ch.setter
    def sigma_beam_ch(self, value):
        if isinstance(value, numbers.Number):
            value = np.ones(self.nu.size) * float(value)
        self._sigma_beam_ch = value
        if "beam_dep_attr" in dir(self):
            self.clean_cache(self.beam_dep_attr)

    @property
    def sigma_beam_in_mpc(self):
        """
        The channel averaged beam size in Mpc
        """
        if self.sigma_beam_ch_in_mpc is None:
            return None
        return self.sigma_beam_ch_in_mpc.mean()

    @property
    def nu(self):
        """
        The input frequencies of the survey
        """
        return self._nu

    @nu.setter
    def nu(self, value):
        self._nu = np.array(value)
        if "nu_dep_attr" in dir(self):
            self.clean_cache(self.nu_dep_attr)

    # nu dependent, but it calculates on the fly
    # so no need for tags
    @property
    def z_ch(self):
        """
        The redshift of each frequency channel
        """
        return freq_to_redshift(self.nu)

    @property
    def z(self):
        """
        The effective centre redshift of the frequency range
        """
        return self.z_ch.mean()

    @property
    def dvdf_ch(self):
        """
        velocity resolution per unit frequency in each channel, in km/s/Hz
        """
        return (constants.c / self.nu).to("km/s").value

    @property
    def vel_resol_ch(self):
        """
        velocity resolution of each channel in km/s
        """
        return self.dvdf_ch * self.freq_resol

    @property
    def dvdf(self):
        """
        velocity resolution per unit frequency on average, in km/s/Hz
        """
        return self.dvdf_ch.mean()

    @property
    def vel_resol(self):
        """
        velocity resolution on average in km/s
        """
        return self.vel_resol_ch.mean()

    @property
    def freq_resol(self):
        """
        frequency resolution in Hz
        """
        return np.diff(self.nu).mean()

    @property
    def pixel_area(self):
        """
        angular area of the map pixel in deg^2
        """
        return proj_plane_pixel_area(self.wproj)

    @property
    def pix_resol(self):
        """
        angular resolution of the map pixel in deg
        """
        return np.sqrt(self.pixel_area)

    @property
    def data(self):
        """
        The map data
        """
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    @property
    def counts(self):
        """
        The number of hits per pixel for the map data
        """
        return self._counts

    @counts.setter
    def counts(self, value):
        self._counts = value

    @property
    def map_has_sampling(self):
        """
        A binary window for whether a pixel has been sampled
        """
        return self._map_has_sampling

    @map_has_sampling.setter
    def map_has_sampling(self, value):
        self._map_has_sampling = value

    W_HI = map_has_sampling

    @property
    def ra_map(self):
        """
        The right ascension of each pixel in the map.
        """
        return self._ra_map

    @property
    def dec_map(self):
        """
        The declination of each pixel in the map.
        """
        return self._dec_map

    @property
    def weights_map_pixel(self):
        """
        The weights per map pixel.
        """
        return self._weights_map_pixel

    @weights_map_pixel.setter
    def weights_map_pixel(self, value):
        self._weights_map_pixel = value

    w_HI = weights_map_pixel

    @property
    def ra_gal(self):
        """
        The right ascension of each galaxy in the catalogue for cross-correlation.
        """
        return self._ra_gal

    @property
    def dec_gal(self):
        """
        The declination of each galaxy in the catalogue for cross-correlation.
        """
        return self._dec_gal

    @property
    def z_gal(self):
        """
        The redshifts of each galaxy in the catalogue for cross-correlation.
        """
        return self._z_gal

    @property
    def freq_gal(self):
        """
        The 21cm line frequency for each galaxy in Hz.
        """
        return f_21 / (1 + self.z_gal)

    @property
    def ch_id_gal(self):
        """
        The channel id (0-indexed) of each galaxy in the catalogue
        for cross-correlation.
        Galaxies out of the frequency range will be given len(self.nu) as indices.
        """
        return find_ch_id(self.freq_gal, self.nu)

    def read_gal_cat(
        self,
        ra_col="RA",
        dec_col="DEC",
        z_col="Z",
        trim=True,
    ):
        """
        Read in a galaxy catalogue for cross-correlation
        and save the data into the class attributes.
        The data is read from the `gal_file`, which has to be a FITS file.

        Parameters
        ----------
        ra_col: str, default "RA"
            The column name of the right ascension in the galaxy catalogue.
        dec_col: str, default "DEC"
            The column name of the declination in the galaxy catalogue.
        z_col: str, default "Z"
            The column name of the redshift in the galaxy catalogue.
        trim: bool, default True
            Whether to trim the galaxy catalogue to the ra,dec,z range of the map.
            See :meth:`meer21cm.dataanalysis.Specification.trim_gal_to_range`.
        """
        if self.gal_file is None:
            print("no gal_file specified")
            return None
        hdu = fits.open(self.gal_file)
        ra_g = hdu[1].data[ra_col]  # Right ascension (J2000) [deg]
        dec_g = hdu[1].data[dec_col]  # Declination (J2000) [deg]
        z_g = hdu[1].data[z_col]  # Spectroscopic redshift, -1 for none attempted
        self._ra_gal = ra_g
        self._dec_gal = dec_g
        self._z_gal = z_g
        if trim:
            self.trim_gal_to_range()

    def read_from_pickle(self):
        """
        Read in a pickle file for cross-correlation
        and save the data into the class attributes.
        See :meth:`meer21cm.io.read_pickle` for more details.
        """
        if self.pickle_file is None:
            print("no pickle_file specified")
            return None
        (
            self.data,
            self.counts,
            self.map_has_sampling,
            self._ra_map,
            self._dec_map,
            self.nu,
            self.wproj,
        ) = read_pickle(
            self.pickle_file,
            nu_min=self.nu_min,
            nu_max=self.nu_max,
            los_axis=self.los_axis,
        )
        self.num_pix_x, self.num_pix_y = self._ra_map.shape
        if self.filter_map_los:
            print("filtering map los")
            (self.data, self.map_has_sampling, _, self.counts,) = filter_incomplete_los(
                self.data,
                self.map_has_sampling,
                self.counts,
                self.counts,
                soft_mask=self.soft_filter_los,
            )

        if self.weighting.lower()[:5] == "count":
            self.weights_map_pixel = self.counts
        elif self.weighting.lower()[:7] == "uniform":
            self.weights_map_pixel = (self.counts > 0).astype("float")
        self.trim_map_to_range()

    def read_from_fits(self):
        """
        Read in a FITS file for the map data and hit counts.
        The FITS file need to follow the format of the MeerKLASS L-band data.
        See :meth:`meer21cm.io.read_map` for more details.

        After reading the data, the map data and hit counts are filtered along the frequency direction
        (see :meth:`meer21cm.io.filter_incomplete_los`), and trimmed to the specified range
        (see :meth:`meer21cm.dataanalysis.Specification.trim_map_to_range`).
        The weights per pixel are set to the hit counts if `self.weighting` is "counts",
        or set to 1 if `self.weighting` is "uniform".
        """
        if self.map_file is None:
            print("no map_file specified")
            return None
        (
            self.data,
            self.counts,
            self.map_has_sampling,
            self._ra_map,
            self._dec_map,
            self.nu,
            self.wproj,
        ) = read_map(
            self.map_file,
            counts_file=self.counts_file,
            nu_min=self.nu_min,
            nu_max=self.nu_max,
            los_axis=self.los_axis,
            band=self.band,
        )
        self.num_pix_x, self.num_pix_y = self._ra_map.shape
        if self.filter_map_los:
            (self.data, self.map_has_sampling, _, self.counts,) = filter_incomplete_los(
                self.data,
                self.map_has_sampling,
                self.counts,
                self.counts,
                soft_mask=self.soft_filter_los,
            )

        if self.weighting.lower()[:5] == "count":
            self.weights_map_pixel = self.counts
        elif self.weighting.lower()[:7] == "uniform":
            self.weights_map_pixel = (self.counts > 0).astype("float")
        self.trim_map_to_range()

    def trim_map_to_range(self):
        """
        Trim the map to the specified range.
        The map data and counts outside the range will be set to zero.
        The map_has_sampling and weights_map_pixel will be set to False outside the range.
        """
        ra_range = np.array(self.ra_range)
        dec_range = np.array(self.dec_range)
        logger.debug(
            f"flagging map and weights outside ra_range: {ra_range}, dec_range: {dec_range}"
        )
        ra_sel = angle_in_range(self.ra_map, ra_range[0], ra_range[1])
        dec_sel = (self.dec_map > dec_range[0]) * (self.dec_map < dec_range[1])
        map_sel = (ra_sel * dec_sel)[:, :, None]
        self.data = self.data * map_sel
        self.counts = self.counts * map_sel
        self.map_has_sampling = self.map_has_sampling * map_sel
        self.weights_map_pixel = self.weights_map_pixel * map_sel

    def trim_gal_to_range(self):
        """
        Trim the galaxy catalogue to the specified range.
        The galaxy catalogue outside the ra-dec-z range will be removed.

        Note that, a small buffer corresponding to half of the frequency channel bandwidth
        is added to the redshift range.
        """
        ra_range = np.array(self.ra_range)
        dec_range = np.array(self.dec_range)
        freq_edges = center_to_edges(self.nu)
        z_edges = freq_to_redshift(freq_edges)
        logger.debug(
            f"flagging galaxy catalogue outside ra_range: {ra_range}, dec_range: {dec_range} and "
            f"z_range: [{z_edges.min()}, {z_edges.max()}]"
        )
        gal_sel = (
            angle_in_range(self.ra_gal, ra_range[0], ra_range[1])
            * (self.dec_gal > dec_range[0])
            * (self.dec_gal < dec_range[1])
        )
        z_sel = (self.z_gal > z_edges.min()) * (self.z_gal < z_edges.max())
        gal_sel *= z_sel
        self._ra_gal = self.ra_gal[gal_sel]
        self._dec_gal = self.dec_gal[gal_sel]
        self._z_gal = self.z_gal[gal_sel]
        return gal_sel

    @property
    @tagging("beam", "nu")
    def beam_image(self):
        """
        Returns the beam image projected onto the sky map for the input beam model.
        """
        if self._beam_image is None:
            self.get_beam_image()
        return self._beam_image

    def get_beam_image(
        self,
        wproj=None,
        num_pix_x=None,
        num_pix_y=None,
        cache=True,
    ):
        """
        Calculate the beam image projected onto the sky map for the input beam model.

        Parameters
        ----------
        wproj: :py:class:`astropy.wcs.WCS`, default None
            The WCS object for the map. Default uses `self.wproj`.
        num_pix_x: int, default None
            The number of pixels in the first axis of the map data. Default uses `self.num_pix_x`.
        num_pix_y: int, default None
            The number of pixels in the second axis of the map data. Default uses `self.num_pix_y`.
        cache: bool, default True
            Whether to cache the beam image. Default is True.
            If True, the beam image will be cached and returned directly if it is already computed.
            If False, the beam image will be computed and returned.
            The cache is saved into the class attribute `beam_image`.
        """
        if self.sigma_beam_ch is None:
            logger.info(
                f"sigma_beam_ch is None, returning None for {inspect.currentframe().f_code.co_name}"
            )
            return None
        logger.info(
            f"invoking {inspect.currentframe().f_code.co_name} to get the beam image"
        )
        logger.info(f"beam_type: {self.beam_type}, sigma_beam_ch: {self.sigma_beam_ch}")
        if wproj is None:
            wproj = self.wproj
        if num_pix_x is None:
            num_pix_x = self.num_pix_x
        if num_pix_y is None:
            num_pix_y = self.num_pix_y
        pix_resol = np.sqrt(proj_plane_pixel_area(wproj))
        beam_image = np.zeros((num_pix_x, num_pix_y, len(self.nu)))
        beam_model = getattr(telescope, self.beam_model + "_beam")
        if self.beam_type == "isotropic":
            for i in range(len(self.nu)):
                beam_image[:, :, i] = telescope.isotropic_beam_profile(
                    num_pix_x,
                    num_pix_y,
                    wproj,
                    beam_model(self.sigma_beam_ch[i]),
                )
        else:
            beam_image = beam_model(
                self.nu,
                wproj,
                num_pix_x,
                num_pix_y,
                band=self.band,
            )
            sigma_beam_from_image = (
                np.sqrt(beam_image.sum(axis=(0, 1)) / 2 / np.pi) * pix_resol
            )
            self.sigma_beam_ch = sigma_beam_from_image
        if cache:
            self._beam_image = beam_image
        return beam_image

    def convolve_data(self, kernel):
        """
        convolve data with an input kernel, and
        update the corresponding weights.
        """
        logger.info(
            f"invoking {inspect.currentframe().f_code.co_name} to convolve map data with kernel: {kernel}"
        )
        data, w_HI = telescope.weighted_convolution(
            self.data,
            kernel,
            self.w_HI,
        )
        self.data = data
        self.w_HI = w_HI

    @property
    def maximum_sampling_channel(self):
        """
        Returns the index of the frequency channel with the maximum sampling on the sky map.
        """
        return np.argmax(self.map_has_sampling.sum(axis=(0, 1)))

    def get_weights_none_to_one(self, attr_name):
        """
        Get the weights, and if it is None, convert it to 1.0 of size of kmode.
        Only used for power spectrum calculation.
        Defined here for inheritance.
        """
        weights = getattr(self, attr_name)
        if weights is None:
            if hasattr(self, "box_ndim"):
                weights = np.ones(self.box_ndim)
            else:
                shape = np.array(self.kmode.shape)
                shape[-1] = 2 * shape[-1] - 2
                weights = np.ones(shape)
        return weights

    def get_jackknife_patches(
        self,
        ra_patch_num,
        dec_patch_num,
        nu_patch_num,
        ra_range=None,
        dec_range=None,
        nu_range=None,
    ):
        """
        Split the map into roughly equal patches. Each patch can then be
        masked, which can be used for jackknife resampling for covariance estimation.
        Note that the masks=True is where the pixels **should be masked**.
        So for example, if you want to exclude a patch, the correct survey window
        is then ``self.W_HI * (1-mask_arr[i])`` and the weights
        ``self.w_HI * (1-mask_arr[i])``.

        If you want to examine the patch splits, you can visualise the mask array
        by using :func:`meer21cm.plot.visualise_patch_split`.

        Parameters
        ----------
        ra_patch_num: int
            The number of patche grids in the right ascension direction.
        dec_patch_num: int
            The number of patche grids in the declination direction.
        nu_patch_num: int
            The number of patche grids in the frequency direction.
        ra_range: tuple, default None
            The range of the right ascension of the map data in degrees.
            Default uses ``self.ra_range``.
        dec_range: tuple, default None
            The range of the declination of the map data in degrees.
            Default uses ``self.dec_range``.
        nu_range: tuple, default None
            The range of the frequency of the map data in Hz.
            Default uses ``[self.nu.min() - self.freq_resol/2, self.nu.max() + self.freq_resol/2]``.
        """
        if ra_range is None:
            ra_range = self.ra_range
        if dec_range is None:
            dec_range = self.dec_range
        assert (
            dec_range[0] < dec_range[1]
        ), "dec_range[0] must be less than dec_range[1]"
        assert dec_range[0] >= -90, "dec must be between -90 and 90"
        assert dec_range[1] <= 90, "dec must be between -90 and 90"
        if nu_range is None:
            nu_range = [
                self.nu.min() - self.freq_resol / 2,
                self.nu.max() + self.freq_resol / 2,
            ]
        assert nu_range[0] < nu_range[1], "nu_range[0] must be less than nu_range[1]"
        assert not (
            ra_range[0] == 0 and ra_range[1] == 360
        ), "ra_range is whole sky 0-360, check if you have passed a value to it"
        ra_delta_map = (self.ra_map - ra_range[0]) % 360
        ra_delta_bins = np.linspace(
            0, (ra_range[1] - ra_range[0]) % 360, ra_patch_num + 1
        )
        dec_bins = np.linspace(dec_range[0], dec_range[1], dec_patch_num + 1)
        nu_bins = np.linspace(nu_range[0], nu_range[1], nu_patch_num + 1)
        ra_indx = np.digitize(ra_delta_map, ra_delta_bins)
        ra_indx[ra_indx == 0] = len(ra_delta_bins)
        dec_indx = np.digitize(self.dec_map, dec_bins)
        dec_indx[dec_indx == 0] = len(dec_bins)
        nu_indx = np.digitize(self.nu, nu_bins)
        nu_indx[nu_indx == 0] = len(nu_bins)
        ra_indx -= 1
        dec_indx -= 1
        nu_indx -= 1
        mask_arr = np.zeros(
            (ra_patch_num, dec_patch_num, nu_patch_num) + self.W_HI.shape
        )
        for i in range(len(ra_delta_bins) - 1):
            for j in range(len(dec_bins) - 1):
                for k in range(len(nu_bins) - 1):
                    W_ijk = ((ra_indx == i) * (dec_indx == j))[:, :, None] * (
                        nu_indx == k
                    )[None, None, :]
                    mask_arr[i, j, k] = W_ijk
        return mask_arr

    def create_white_noise_map(self, sigma_N, counts=None, seed=None, inf_to_zero=True):
        """
        Create a white noise map with the given standard deviation.
        The sigma in each pixel is then scaled by the counts 1/sqrt(counts).

        Note that, the default seed is **fixed** to the class attribute ``self.seed``.
        If you want to generate multiple random catalogues, you need to set a different seed manually for each catalogue.

        If you want to use different noise level per pixel, you can either pass a 3D
        array of sigma_N, or a single number and a 3D array of counts.
        You can usually pass ``self.counts`` as the counts array, but do check the counts
        are set up correctly by ``plot_map(self.counts, self.wproj)``.

        Finally, note that the noise map is not masked by the survey selection function.
        You can mask the noise map manually by ``noise_map *= self.W_HI``.

        Parameters
        ----------
        sigma_N: float or array.
            The standard deviation of the white noise.
        counts: array, default None.
            The counts in each pixel. If None, the counts will be one across the cube.
        seed: int, default None.
            The seed for the random number generator. Default uses the class attribute ``self.seed``.
        inf_to_zero: bool, default True.
            If True, the inf values in the noise map will be set to zero.
        Returns
        -------
        noise_map: array.
            The white noise map.
        """
        if counts is None:
            counts = np.ones(self.data.shape)
        rng = np.random.default_rng(seed=seed)
        noise_map = rng.normal(scale=sigma_N / np.sqrt(counts), size=self.data.shape)
        if inf_to_zero:
            noise_map[np.isinf(noise_map)] = 0.0
        return noise_map

    def check_is_map_noiselike_using_pca(self, A_mat, data=None, sigma_N=1.0):
        """
        Use the source mixing matrix from eigendecomposition of the covariance matrix,
        project out the map data with more and more modes,
        and check if the variance of the residual map
        behaves like white noise.

        You can use :func:`meer21cm.util.pca_clean` to retrieve the source mixing matrix:
        .. code-block:: python

            N_fg = 15 # check 15 modes removed
            res_map, A_mat = pca_clean(ps.data, N_fg, weights=ps.W_HI, return_A=True)
            res, noise = ps.check_is_map_noiselike_using_pca(A_mat)
            plt.plot(res / noise)

        If the residual map is noise-like, the plot should decrease and
        eventually reach a plateau.

        If you know the expected std of the map (per hit), you can pass it to
        ``sigma_N`` to scale the noise variance, and the plateau should
        be close to 1.

        Note that, the input data should be the mean-centered data.
        You can use :func:`meer21cm.util.mean_center_signal` to mean-center the data if needed.


        Parameters
        ----------
        A_mat: array.
            The source mixing matrix.
        data: array, default None.
            The data to be projected out. If None, the class attribute ``self.data`` will be used.
        sigma_N: float.
        """
        res_var = []
        noise_var = []
        if data is None:
            data = self.data
        for i in range(A_mat.shape[1]):
            R_mat = np.eye(self.nu.size) - np.dot(
                A_mat[:, : i + 1], A_mat[:, : i + 1].T
            )
            var_attenuation = np.trace(R_mat.T @ R_mat) / self.nu.size
            data_res = np.einsum("ij, abj -> abi", R_mat, data)
            res_var.append((data_res * np.sqrt(self.counts))[self.W_HI > 0].var())
            noise_var.append(var_attenuation)
        res_var = np.array(res_var)
        noise_var = np.array(noise_var) * sigma_N**2
        return res_var, noise_var
