from meer21cm import CosmologyCalculator, Specification
from meer21cm.cosmology import CosmologyParameters
from astropy.cosmology import Planck18, Planck15
import numpy as np
import camb
from meer21cm.util import f_21
import pytest


def test_set_background():
    pars = CosmologyParameters()
    pars.get_derived_Ode()
    pars.cosmo = pars.set_astropy_cosmo()
    assert np.allclose(pars.cosmo.Onu0, Planck18.Onu0)
    assert np.allclose(pars.cosmo.Ogamma0, Planck18.Ogamma0)
    assert np.abs(pars.cosmo.Ok0) < 1e-5
    # test w0wa
    pars = CosmologyParameters(
        w0=-0.85,
        wa=0.1,
    )
    pars.get_derived_Ode()
    pars.cosmo = pars.set_astropy_cosmo()
    assert np.allclose(pars.cosmo.Onu0, Planck18.Onu0)
    assert np.allclose(pars.cosmo.Ogamma0, Planck18.Ogamma0)
    assert np.abs(pars.cosmo.Ok0) < 1e-5


@pytest.mark.parametrize("ps_type, accuracy", [("linear", 0.01), ("nonlinear", 0.05)])
def test_compare_matter_power(ps_type, accuracy):
    pars = CosmologyParameters(ps_type=ps_type)
    pkcamb = pars.get_matter_power_spectrum_camb()
    pkbacco = pars.get_matter_power_spectrum_bacco()
    # accuracy not great for nonlinear
    assert np.allclose(np.abs(pkcamb / pkbacco - 1) < accuracy, True)
    # test a different cosmology
    pars = CosmologyParameters(w0=-0.85, wa=-0.1, ps_type=ps_type)
    pkcamb = pars.get_matter_power_spectrum_camb()
    pkbacco = pars.get_matter_power_spectrum_bacco()
    # accuracy not great for nonlinear
    assert np.allclose(np.abs(pkcamb / pkbacco - 1) < accuracy, True)


@pytest.mark.parametrize(
    "flag1, flag2", [(True, False), (False, False), (True, True), (False, True)]
)
def test_f_growth(flag1, flag2):
    pars = CosmologyParameters()
    if flag1:
        pars._expfactor = 0.5
    if flag2:
        pars._w0 = -0.9
        pars._wa = -0.1
    pars.get_matter_power_spectrum_bacco()
    f_growth_bacco = pars.f_growth
    pars.get_matter_power_spectrum_camb()
    f_growth_camb = pars.f_growth
    assert np.abs(f_growth_camb / f_growth_bacco - 1) < 2e-2


def test_cosmo():
    coscal = CosmologyCalculator()
    assert coscal.h == Planck18.h
    assert np.abs(np.log(1e10 * coscal.As) - 3.047) < 1e-3
    # only test invoking, the function itself is tested in util
    t1, ohi1 = coscal.average_hi_temp, coscal.omega_hi.mean()
    # test update omega_hi, scales correctly
    coscal.omega_hi = 5.5e-4
    np.allclose(coscal.average_hi_temp / t1, (coscal.omega_hi.mean() / ohi1))
    # test omega_hi_z_func, proper z func
    coscal.nu = np.array([8e8, 9e8])
    coscal.omega_hi = np.array([6e-4, 5e-4])
    assert np.allclose(coscal.omega_hi_z_mean, 5.5e-4)
    coscal.omega_hi = np.array([5e-4, 5e-4])
    coscal.nu = np.array([8e8, 9e8])
    assert np.allclose(coscal.omega_hi_z_mean, 5e-4)


def test_update_pars():
    coscal = CosmologyCalculator()
    As = coscal.As
    coscal.cosmo = "Planck15"
    coscal.cosmo
    assert coscal.h == Planck15.h
    # As has been updated
    assert coscal.As != As


def test_cache():
    coscal = CosmologyCalculator(
        survey="meerklass_2021",
        band="L",
    )
    test1 = coscal.matter_power_spectrum_fnc(1)
    coscal.nu = [f_21, f_21]
    coscal.nu
    test2 = coscal.matter_power_spectrum_fnc(1)
    assert test1 != test2
    coscal.cosmo = "WMAP1"
    test3 = coscal.matter_power_spectrum_fnc(1)
    assert test3 != test2


@pytest.mark.parametrize("backend", [("camb"), ("bacco")])
def test_mps_fnc(backend):
    coscal = CosmologyCalculator(
        cosmo="WMAP1",
        backend=backend,
    )
    matterps = (
        coscal.matter_power_spectrum_fnc(coscal.karr_in_h * coscal.h) * coscal.h**3
    )
    pkbackend = getattr(coscal, f"get_matter_power_spectrum_{backend}")()
    assert np.allclose(pkbackend, matterps)


@pytest.mark.parametrize(
    "par",
    [
        ("omega_cold"),
        ("omega_baryon"),
        ("h"),
        ("neutrino_mass"),
        ("w0"),
        ("wa"),
    ],
)
def test_update_parameter(par):
    # bacco is faster
    coscal = CosmologyCalculator(backend="bacco", wa=-0.05, w0=-0.95)
    comov = coscal.comoving_distance(1).value
    matterps = (
        coscal.matter_power_spectrum_fnc(coscal.karr_in_h * coscal.h) * coscal.h**3
    )
    # change parameter
    setattr(coscal, par, getattr(coscal, par) * 1.01)
    comov2 = coscal.comoving_distance(1).value
    matterps2 = (
        coscal.matter_power_spectrum_fnc(coscal.karr_in_h * coscal.h) * coscal.h**3
    )
    # baryon and neutrino change to background is negligible
    if par != "neutrino_mass" and par != "omega_baryon":
        assert not np.allclose(comov, comov2)
    assert not np.allclose(matterps, matterps2)


def test_sigmar():
    cosmo = CosmologyCalculator()
    sigma_z = 0.001
    z_arr = np.random.normal(0, sigma_z, 100000) + cosmo.z
    sigma_r = cosmo.comoving_distance(z_arr).std().value
    delta_r = cosmo.deltaz_to_deltar(sigma_z)
    assert np.abs(sigma_r - delta_r) < 3e-2
    sigma_v = 100
    delta_r = cosmo.deltav_to_deltar(sigma_v)
    assert delta_r < 5
