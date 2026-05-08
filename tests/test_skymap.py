from meer21cm.skymap import WcsSkyMap, HealpixSkyMap
import pytest
import numpy as np
import healpy as hp


def test_healpix_skymap():
    with pytest.raises(
        ValueError,
        match="HealpixSkyMap: pass pixel_id, or give both ra_range and dec_range to derive it.",
    ):
        HealpixSkyMap(2)
    with pytest.raises(
        ValueError, match="HealpixSkyMap requires at least one pixel_id."
    ):
        HealpixSkyMap(2, pixel_id=np.array([]))
    with pytest.raises(
        ValueError, match="HealpixSkyMap: pixel_id out of range for this nside."
    ):
        HealpixSkyMap(2, pixel_id=np.array([hp.nside2npix(2) + 1]))
