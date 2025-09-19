import numpy as np
from meer21cm import Specification
from meer21cm.fg import ForegroundSimulation as FgSim
import pytest
import healpy as hp


@pytest.mark.parametrize("backend", ["haslam", "pysm", "gdsm"])
def test_fg_wcs_cube(backend):
    sp = Specification(
        band="L",
        survey="meerklass_2021",
    )
    fg = FgSim(
        hp_nside=256,
        wproj=sp.wproj,
        num_pix_x=sp.num_pix_x,
        num_pix_y=sp.num_pix_y,
        backend=backend,
    )
    test = fg.fg_wcs_cube([1e9, 1.1e9])
    assert test.shape == (sp.num_pix_x, sp.num_pix_y, 2)
    assert np.abs(test.mean()) < 10  # temp should be a few K
    # test healpix output
    fg = FgSim(
        hp_nside=256,
        backend=backend,
    )
    test = fg.fg_wcs_cube([1e9, 1.1e9])
    assert test.shape == (2, hp.nside2npix(256))
    assert np.abs(test.mean()) < 10  # temp should be a few K
    # test wcs output
    fg = FgSim(
        hp_nside=256,
        backend=backend,
    )
    test = fg.fg_wcs_cube([1e9, 1.1e9])
    if backend == "gdsm":
        test = fg.fg_wcs_cube(1e9)
        # gdsm has no consistent shape for single freq
        assert test.shape == (1, hp.nside2npix(256))
