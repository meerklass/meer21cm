"""
Sky map geometry backends used by ``Specification``.

This module intentionally starts with WCS-only support so we can refactor
geometry handling without changing behavior.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from astropy.wcs.utils import proj_plane_pixel_area

from .util import angle_in_range, get_wcs_coor


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
        """Characteristic angular pixel size in degrees."""

    @property
    @abstractmethod
    def pixel_area(self):
        """Angular pixel area in deg^2."""

    @property
    @abstractmethod
    def map_shape_template(self):
        """Angular shape prefix for map cubes (without LOS axis)."""

    @abstractmethod
    def trim_selector(self, ra_range, dec_range):
        """Return boolean selector for pixels inside requested ranges."""


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
