"""
Sky map geometry backends used by ``Specification``.

Supports WCS (``WcsSkyMap``) and partial-sky HEALPix (``HealpixSkyMap``) with an
explicit or auto-derived sparse ``pixel_id`` list (no full-sphere map storage).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import healpy as hp
import numpy as np
from astropy.wcs.utils import proj_plane_pixel_area

from .util import angle_in_range, get_wcs_coor


def _pixel_ids_in_ra_dec_range(nside, ra_range, dec_range):
    """Return sorted unique HEALPix pixel indices inside ``(ra_range, dec_range)`` (degrees)."""
    ra_lo = float(ra_range[0])
    ra_hi = float(ra_range[1])
    dec_lo = min(float(dec_range[0]), float(dec_range[1]))
    dec_hi = max(float(dec_range[0]), float(dec_range[1]))
    theta1_rad = np.radians(90.0 - dec_hi)
    theta2_rad = np.radians(90.0 - dec_lo)
    cand = hp.query_strip(
        int(nside), theta1_rad, theta2_rad, inclusive=True, nest=False
    )
    ra, dec = hp.pix2ang(int(nside), cand.astype(np.int64), nest=False, lonlat=True)
    sel = angle_in_range(ra, ra_lo, ra_hi) * (dec > dec_lo) * (dec < dec_hi)
    return np.sort(np.unique(cand.astype(np.int64)[sel]))


class SkyMap(ABC):
    """Abstract geometry interface for map-like sky coordinates."""

    format: str = "unknown"

    @property
    @abstractmethod
    def ra_map(self):
        """RA coordinates of map pixels in degrees."""

    @property
    @abstractmethod
    def dec_map(self):
        """Dec coordinates of map pixels in degrees."""

    @property
    @abstractmethod
    def pix_resol(self):
        """Characteristic linear pixel scale in degrees (``sqrt(pixel_area)`` for a square pixel)."""

    @property
    @abstractmethod
    def pixel_area(self):
        """Pixel solid angle expressed as square degrees (numeric :math:`\\Delta\\alpha \\Delta\\delta` style).

        Matches :func:`astropy.wcs.utils.proj_plane_pixel_area` for degree-valued CDELT maps
        (product of pixel spans in degrees, not healpy steradians unless converted).
        """

    @property
    @abstractmethod
    def map_shape_template(self):
        """Angular shape prefix for map cubes (without LOS axis)."""

    @abstractmethod
    def trim_selector(self, ra_range, dec_range):
        """Return boolean selector over stored sky pixels inside requested ranges."""


class WcsSkyMap(SkyMap):
    """WCS-based sky map geometry."""

    format = "wcs"

    def __init__(self, wproj, num_pix_x, num_pix_y):
        self._wproj = wproj
        self._num_pix_x = int(num_pix_x)
        self._num_pix_y = int(num_pix_y)
        xx, yy = np.meshgrid(
            np.arange(self._num_pix_x),
            np.arange(self._num_pix_y),
            indexing="ij",
        )
        self._ra_map, self._dec_map = get_wcs_coor(self._wproj, xx, yy)

    @property
    def wproj(self):
        return self._wproj

    @property
    def num_pix_x(self):
        return self._num_pix_x

    @property
    def num_pix_y(self):
        return self._num_pix_y

    @property
    def ra_map(self):
        return self._ra_map

    @property
    def dec_map(self):
        return self._dec_map

    @property
    def pixel_area(self):
        # For MeerKLASS-style WCS with CDELT in degrees, this agrees with |cdelt1*cdelt2| in sq deg,
        # which is what cosmology.volume and pix_resol_in_mpc assume.
        return proj_plane_pixel_area(self.wproj)

    @property
    def pix_resol(self):
        return np.sqrt(self.pixel_area)

    @property
    def map_shape_template(self):
        return (self.num_pix_x, self.num_pix_y)

    def trim_selector(self, ra_range, dec_range):
        ra_range = np.array(ra_range)
        dec_range = np.array(dec_range)
        ra_sel = angle_in_range(self.ra_map, ra_range[0], ra_range[1])
        dec_sel = (self.dec_map > dec_range[0]) * (self.dec_map < dec_range[1])
        return ra_sel * dec_sel


class HealpixSkyMap(SkyMap):
    """
    Sparse HEALPix sky geometry: pixels are listed by ``pixel_id`` only.

    ``pixel_id`` may be supplied explicitly or derived from ``ra_range``
    and ``dec_range`` together with ``hp_nside`` (survey footprint only).
    """

    format = "healpix"

    def __init__(
        self,
        hp_nside,
        *,
        pixel_id=None,
        ra_range=None,
        dec_range=None,
    ):
        nside_i = int(hp_nside)
        self._hp_nside = nside_i
        npix_sphere = hp.nside2npix(nside_i)

        if pixel_id is None:
            if ra_range is None or dec_range is None:
                raise ValueError(
                    "HealpixSkyMap: pass pixel_id, or give both ra_range and dec_range "
                    "to derive it."
                )
            pid = _pixel_ids_in_ra_dec_range(nside_i, ra_range, dec_range)
        else:
            pid = np.asarray(pixel_id, dtype=np.int64).ravel()
            if pid.size == 0:
                raise ValueError("HealpixSkyMap requires at least one pixel_id.")
            pid = np.unique(pid)
            invalid = np.where((pid < 0) | (pid >= npix_sphere))[0]
            if invalid.size > 0:
                raise ValueError("HealpixSkyMap: pixel_id out of range for this nside.")

        self._pixel_id = pid
        lon, lat = hp.pix2ang(nside_i, self._pixel_id, nest=False, lonlat=True)
        self._ra_map = np.asarray(lon, dtype=np.float64)
        self._dec_map = np.asarray(lat, dtype=np.float64)
        # Match WCS convention: pix area as square degrees (not healpy sr default).
        self._pixel_area = float(hp.nside2pixarea(self._hp_nside, degrees=True))

    @property
    def hp_nside(self):
        return self._hp_nside

    @property
    def pixel_id(self):
        return self._pixel_id

    @property
    def ra_map(self):
        return self._ra_map

    @property
    def dec_map(self):
        return self._dec_map

    @property
    def pixel_area(self):
        return self._pixel_area

    @property
    def pix_resol(self):
        return float(np.sqrt(self._pixel_area))

    @property
    def map_shape_template(self):
        return (self._pixel_id.size,)

    def trim_selector(self, ra_range, dec_range):
        ra_range = np.array(ra_range, dtype=float)
        dec_range = np.array(dec_range, dtype=float)
        ra_sel = angle_in_range(self._ra_map, ra_range[0], ra_range[1])
        dec_sel = (self._dec_map > dec_range[0]) * (self._dec_map < dec_range[1])
        out = np.asarray(ra_sel * dec_sel, dtype=bool)
        return out

        #    def subset_pixels(self, keep_mask):
        """
        Return a new map geometry keeping only pixels where ``keep_mask`` is True.
        """
        keep = np.asarray(keep_mask, dtype=bool)
        if keep.shape != (self._pixel_id.size,):
            raise ValueError(
                f"keep_mask shape {keep.shape} does not match n_pix={self._pixel_id.size}."
            )
        if keep.sum() == 0:
            raise ValueError("subset_pixels requires at least one pixel to remain.")
        return HealpixSkyMap(self._hp_nside, pixel_id=self._pixel_id[keep])


__all__ = ["SkyMap", "WcsSkyMap", "HealpixSkyMap", "_pixel_ids_in_ra_dec_range"]
