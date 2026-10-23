import pytest
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import os
import meer21cm
from meer21cm import MockSimulation
from meer21cm.util import create_wcs

data_dir = meer21cm.__file__.rsplit("/", 1)[0] + "/data/"


@pytest.fixture
def test_haslam_map():
    map_file = data_dir + "haslam408_dsds_Remazeilles2014.fits"
    return map_file


@pytest.fixture
def test_gsm_1ghz_jy():
    map_file = data_dir + "gdsm16_1ghz_jy.fits"
    return map_file


@pytest.fixture
def test_gsm_1ghz():
    map_file = data_dir + "gdsm16_1ghz.fits"
    return map_file


@pytest.fixture
def test_fits():
    map_file = data_dir + "test_fits.fits"
    return map_file


@pytest.fixture
def test_gal_fits():
    map_file = data_dir + "test_gal.fits"
    return map_file


@pytest.fixture
def test_wcs():
    map_file = data_dir + "test_fits.fits"
    wcs = WCS(map_file)
    return wcs


@pytest.fixture
def test_wproj():
    map_file = data_dir + "test_fits.fits"
    wcs = WCS(map_file)
    wproj = wcs.dropaxis(-1)
    return wproj


@pytest.fixture
def test_pickle():
    map_file = data_dir + "test.pickle"
    return map_file


@pytest.fixture
def test_W():
    map_file = data_dir + "test_W.npy"
    return np.load(map_file)[:, :, 0][:, :, None]


@pytest.fixture
def test_nu():
    file = data_dir + "test_nu.npy"
    return np.load(file) * 1e6


@pytest.fixture
def test_GAMA_range():
    raminGAMA, ramaxGAMA = 339, 351
    decminGAMA, decmaxGAMA = -35, -30
    return (raminGAMA, ramaxGAMA), (decminGAMA, decmaxGAMA)


@pytest.fixture
def saved_mock_prefix(tmp_path):
    """
    Prefix for one tiny HI pickle and galaxy FITS table.

    ``TransferFunction.mock_directory`` is this prefix: the loader concatenates
    ``HI_mock_{seed}.pkl`` and ``gal_mock_{seed}.fits`` onto it. The seed is 0.
    """
    nu = np.linspace(900e6, 1000e6, 5)
    mock = MockSimulation(
        nu=nu,
        wproj=create_wcs(0.0, 0.0, [4, 4], [1.0, 1.0]),
        num_pix_x=4,
        num_pix_y=4,
        flat_sky=True,
        seed=0,
    )
    mock.data = np.arange(4 * 4 * nu.size, dtype=float).reshape(4, 4, nu.size)
    mock._ra_gal = np.array([10.0, 11.5])
    mock._dec_gal = np.array([-2.0, 1.25])
    mock._z_gal = np.array([0.42, 0.55])
    mock.weights_gal = np.array([1.0, 0.5])
    prefix = str(tmp_path) + os.sep
    mock.save_mock_products(prefix)
    return prefix
