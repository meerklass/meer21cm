import healpy as hp
import numpy as np
import meer21cm
from .util import healpix_to_wcs, read_healpix_fits, check_unit_equiv
from astropy import units
from healpy.rotator import Rotator

default_data_dir = meer21cm.__file__.rsplit("/", 1)[0] + "/data/"


class ForegroundSimulation:
    """
    Foreground simulation class.
    All outputs are in units of K_RJ (Kelvin).

    Parameters
    ----------
    hp_nside: int
        HEALPix nside.
    wproj: tuple
        WCS projection parameters.
    num_pix_x: int
        Number of pixels in x direction.
    num_pix_y: int
        Number of pixels in y direction.
    backend: str, default 'haslam'
        Backend to use for foreground simulation.
        Options: 'gdsm' (Global Sky Model), 'pysm' (PySM3), 'haslam' (Haslam 408 MHz map).
    pysm_preset_strings: list, default ['d1', 's1', 'f1', 'a1', 'c1']
        List of PySM3 preset strings for included components.
        Default: ['d1', 's1', 'f1', 'a1', 'c1'].
    sp_indx_for_haslam_backend: float, default -2.0
        Index of the spectral index for the Haslam 408 MHz map.
        Only used for 'haslam' backend.
    coord_system: str, default 'C'
        Coordinate system to use for the foreground simulation.
        Options: 'G' (Galactic), 'C' (Celestial), 'E' (Ecliptic).
    """

    def __init__(
        self,
        hp_nside=256,
        wproj=None,
        num_pix_x=None,
        num_pix_y=None,
        backend="haslam",
        pysm_preset_strings=["d1", "s1", "f1", "a1", "c1"],
        sp_indx_for_haslam_backend=-2.0,
        coord_system="C",
    ):
        self.backend = backend
        assert backend in [
            "gdsm",
            "pysm",
            "haslam",
        ], "backend must be either 'gdsm', 'pysm' or 'haslam'"
        self.hp_nside = hp_nside
        self.wproj = wproj
        self.num_pix_x = num_pix_x
        self.num_pix_y = num_pix_y
        assert hp.isnsideok(hp_nside), "hp_nside must be a valid HEALPix nside"
        self.pysm_preset_strings = pysm_preset_strings
        self.sp_indx_for_haslam_backend = sp_indx_for_haslam_backend
        self.coord_system = coord_system

    def healpix_gen_haslam(self, freq):
        """
        Generate HEALPix map of foregrounds using Haslam 408 MHz map.

        Parameters
        ----------
        freq: float or array-like
            Frequency in Hz.

        Returns
        -------
        cube: numpy.ndarray
            HEALPix map of foregrounds in units of K_RJ.
        """
        sp_indx = self.sp_indx_for_haslam_backend
        freq = np.atleast_1d(freq)
        haslam_map, hp_nside, map_unit, map_freq = read_healpix_fits(
            default_data_dir + "haslam408_dsds_Remazeilles2014.fits"
        )
        assert check_unit_equiv(map_unit, units.K), "map unit must be temperature"
        haslam_map = (haslam_map * map_unit).to(units.K).value
        haslam_map = hp.ud_grade(haslam_map, self.hp_nside)
        r = Rotator(coord=["G", self.coord_system])
        haslam_map = r.rotate_map_pixel(haslam_map)
        cube = haslam_map[None, :] * ((freq / map_freq) ** sp_indx)[:, None]
        return cube

    def healpix_gen_gdsm(self, freq):
        """
        Generate HEALPix map of foregrounds using Global Sky Model.

        Parameters
        ----------
        freq: float or array-like
            Frequency in Hz.

        Returns
        -------
        cube: numpy.ndarray
            HEALPix map of foregrounds in units of K_RJ.
        """
        from pygdsm import GlobalSkyModel16

        gsm = GlobalSkyModel16(
            freq_unit="MHz",
            data_unit="TRJ",
            resolution="hi" if self.hp_nside > 64 else "low",
        )
        cube = gsm.generate(freq / 1e6)
        cube = hp.ud_grade(cube, self.hp_nside)
        r = Rotator(coord=["G", self.coord_system])
        if len(freq) == 1:
            cube = cube[None, :]
        for i in range(cube.shape[0]):
            cube[i] = r.rotate_map_pixel(cube[i])
        return cube

    def healpix_gen_pysm(self, freq):
        """
        Generate HEALPix map of foregrounds using PySM3.

        Parameters
        ----------
        freq: float or array-like
            Frequency in Hz.

        Returns
        -------
        cube: numpy.ndarray
            HEALPix map of foregrounds in units of K_RJ.
        """
        import pysm3

        pysm = pysm3.Sky(nside=self.hp_nside, preset_strings=self.pysm_preset_strings)
        cube = []
        for f in freq:
            cube_i = pysm.get_emission(f * units.Hz)[0].value / 1e6
            cube_i = hp.ud_grade(cube_i, self.hp_nside)
            r = Rotator(coord=["G", self.coord_system])
            cube_i = r.rotate_map_pixel(cube_i)
            cube.append(cube_i)
        cube = np.array(cube)
        return cube

    def fg_wcs_cube(
        self,
        freq,
    ):
        """
        Generate WCS cube of foregrounds.
        The function will call the appropriate healpix_gen_* function based on the backend,
        and then project the map to the WCS coordinates if provided.

        Parameters
        ----------
        freq: float or array-like
            Frequency in Hz.

        Returns
        -------
        out_map: numpy.ndarray
            WCS cube of foregrounds in units of K_RJ.
        """
        freq = np.atleast_1d(freq)
        wproj = self.wproj
        xdim = self.num_pix_x
        ydim = self.num_pix_y
        out_map = getattr(self, f"healpix_gen_{self.backend}")(freq)
        if wproj is not None:
            xx, yy = np.meshgrid(
                np.arange(xdim),
                np.arange(ydim),
                indexing="ij",
            )
            out_map_proj = np.zeros(xx.shape + (out_map.shape[0],))
            for ch_id in range(out_map.shape[0]):
                out_map_proj[:, :, ch_id] = healpix_to_wcs(
                    out_map[ch_id], xx, yy, wproj
                )
            out_map = out_map_proj
        return out_map
