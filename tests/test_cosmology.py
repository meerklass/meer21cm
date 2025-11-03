from meer21cm import CosmologyCalculator, Specification
from meer21cm.cosmology import CosmologyParameters
from astropy.cosmology import Planck18, Planck15
import numpy as np
import camb
from meer21cm.util import f_21
import pytest
from astropy import units
from meer21cm.util import freq_to_redshift


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
    As = coscal.cospar_fiducial.As
    coscal.fiducial_cosmology = "Planck15"
    assert coscal.astropy_cosmo_fiducial.h == Planck15.h
    # As has been updated
    assert coscal.cospar_fiducial.As != As
    # test update true cosmology
    As = coscal.cospar_true.As
    coscal.true_cosmology = "Planck15"
    assert coscal.astropy_cosmo_true.h == Planck15.h
    # As has been updated
    # for true cosmology, As can be directly read
    assert coscal.As != As


def test_cache():
    coscal = CosmologyCalculator(
        survey="meerklass_2021",
        band="L",
    )
    # trigger f_growth
    assert np.allclose(coscal.f_growth_fiducial, coscal.f_growth_true)
    test1 = coscal.matter_power_spectrum_fnc(1)
    coscal.nu = [f_21, f_21]
    coscal.nu
    # test reset
    assert coscal._cospar_fiducial is None
    # retrigger
    coscal.cospar_fiducial
    assert np.allclose(coscal.cospar_fiducial.expfactor, 1)
    test2 = coscal.matter_power_spectrum_fnc(1)
    assert test1 != test2
    coscal.true_cosmology = "WMAP1"
    test3 = coscal.matter_power_spectrum_fnc(1)
    assert test3 != test2
    # fiducial should remain the same
    assert not np.allclose(coscal.f_growth_fiducial, coscal.f_growth_true)


@pytest.mark.parametrize("backend", [("camb"), ("bacco")])
def test_mps_fnc(backend):
    coscal = CosmologyCalculator(
        true_cosmology="WMAP1",
        fiducial_cosmology="WMAP1",
        backend=backend,
    )
    matterps = (
        coscal.matter_power_spectrum_fnc(
            coscal.cospar_true.karr_in_h * coscal.cospar_true.h
        )
        * coscal.cospar_true.h**3
    )
    pkbackend = getattr(coscal.cospar_true, f"get_matter_power_spectrum_{backend}")()
    assert np.allclose(pkbackend, matterps)
    coscal.true_cosmology = "Planck15"
    matterps2 = (
        coscal.matter_power_spectrum_fnc(
            coscal.cospar_true.karr_in_h * coscal.cospar_true.h
        )
        * coscal.cospar_true.h**3
    )
    assert not np.allclose(matterps, matterps2)


@pytest.mark.parametrize(
    "par",
    [
        ("omega_cold"),
        ("omega_baryon"),
        ("h"),
        ("neutrino_mass"),
        ("w0"),
        ("wa"),
        ("As"),
        ("ns"),
    ],
)
def test_update_parameter(par):
    # bacco is faster
    coscal = CosmologyCalculator(backend="bacco")
    comov = coscal.astropy_cosmo_true.comoving_distance(1).value
    matterps = (
        coscal.matter_power_spectrum_fnc(
            coscal.cospar_true.karr_in_h * coscal.cospar_true.h
        )
        * coscal.cospar_true.h**3
    )
    # change parameter
    if par != "wa":
        setattr(coscal, par, getattr(coscal, par) * 1.1)
    else:
        setattr(coscal, par, getattr(coscal, par) + 0.1)
    comov2 = coscal.astropy_cosmo_true.comoving_distance(1).value
    matterps2 = (
        coscal.matter_power_spectrum_fnc(
            coscal.cospar_true.karr_in_h * coscal.cospar_true.h
        )
        * coscal.cospar_true.h**3
    )
    # baryon and neutrino change to background is negligible
    if par not in ["neutrino_mass", "omega_baryon", "As", "ns"]:
        assert not np.allclose(comov, comov2)
    assert not np.allclose(matterps, matterps2)


def test_sigmar():
    coscal = CosmologyCalculator()
    sigma_z = 0.001
    z_arr = np.random.normal(0, sigma_z, 100000) + coscal.z
    sigma_r = coscal.astropy_cosmo_true.comoving_distance(z_arr).std().value
    delta_r = coscal.deltaz_to_deltar(sigma_z)
    assert np.abs(sigma_r - delta_r) < 3e-2
    sigma_v = 100
    delta_r = coscal.deltav_to_deltar(sigma_v)
    assert delta_r < 5


def test_beam_update():
    ps = CosmologyCalculator(
        survey="meerklass_2021",
        band="L",
    )
    assert ps.sigma_beam_ch_in_mpc is None
    assert ps.sigma_beam_in_mpc is None
    ps.sigma_beam_ch = np.ones(ps.nu.size)
    assert ps._sigma_beam_ch_in_mpc is None
    s1 = ps.sigma_beam_ch_in_mpc
    assert np.allclose(s1.mean(), ps.sigma_beam_in_mpc)
    ps.sigma_beam_ch = np.ones(ps.nu.size) * 2
    assert ps._sigma_beam_ch_in_mpc is None
    s2 = ps.sigma_beam_ch_in_mpc
    assert np.allclose(2 * s1, s2)
    # test single number input
    ps.sigma_beam_ch = 2
    assert np.allclose(ps.sigma_beam_ch_in_mpc, s2)
    ps.beam_unit = units.rad
    s3 = ps.sigma_beam_ch_in_mpc
    assert np.allclose(np.pi * s3 / 180, s2)
    # test update cosmo, then beam in mpc also change
    ps.true_cosmology = "Planck15"
    s4 = ps.sigma_beam_ch_in_mpc
    assert not np.allclose(s4, s3)


def test_z_interp():
    ps = CosmologyCalculator(
        survey="meerklass_2021",
        band="L",
    )
    func = ps.z_as_func_of_comov_dist
    z_rand = np.random.uniform(ps.z_ch.min(), ps.z_ch.max(), size=100)
    assert np.allclose(
        func(ps.astropy_cosmo_true.comoving_distance(z_rand).value), z_rand
    )


def test_defaults(test_nu, test_W):
    spec = CosmologyCalculator(
        survey="meerklass_2021",
        band="L",
    )
    assert np.allclose(spec.nu, test_nu)
    assert np.allclose(
        spec.map_has_sampling[1:-1, 1:-1], np.ones(test_W[1:-1, 1:-1].shape)
    )
    assert np.allclose(spec.z_ch, freq_to_redshift(test_nu))
    assert np.allclose(spec.z, freq_to_redshift(test_nu).mean())
    x_res = 0.3 * np.pi / 180 * Planck18.comoving_distance(spec.z).value
    assert np.allclose(spec.pix_resol_in_mpc, x_res)
    los = Planck18.comoving_distance(spec.z_ch).value
    z_res = (los[0] - los[-1]) / len(spec.nu)
    assert np.allclose(z_res, spec.los_resol_in_mpc)


def test_volume():
    spec = CosmologyCalculator(
        survey="meerklass_2021",
        band="L",
    )
    diff = np.abs(
        spec.pix_resol_in_mpc**2
        * spec.los_resol_in_mpc
        * spec.W_HI.sum()
        / spec.survey_volume
        - 1
    )
    assert diff < 1e-2


def test_ap_effect():
    coscal = CosmologyCalculator(
        survey="meerklass_2021",
        band="L",
    )
    assert np.allclose(coscal.alpha_parallel, 1)
    assert np.allclose(coscal.alpha_perp, 1)
    assert np.allclose(coscal.alpha_iso, 1)
    assert np.allclose(coscal.alpha_AP, 1)
    coscal.true_cosmology = "Planck15"
    coscal.fiducial_cosmology = "WMAP1"
    # test warning
    with pytest.warns(UserWarning):
        coscal.cosmo
    assert not np.allclose(coscal.alpha_parallel, 1)
    assert not np.allclose(coscal.alpha_perp, 1)
    assert not np.allclose(coscal.alpha_iso, 1)
    assert not np.allclose(coscal.alpha_AP, 1)
    # z~0, anisotropic should be 1 again
    coscal.nu = np.array([f_21 * 0.999, f_21 * 0.999])
    assert np.abs(coscal.alpha_AP - 1) < 1e-3
    assert not np.allclose(coscal.alpha_iso, 1)


def test_cosmo_shortcut():
    coscal = CosmologyCalculator(
        survey="meerklass_2021",
        band="L",
    )
    coscal.cosmo = "Planck15"
    assert np.allclose(coscal.cosmo.h, Planck15.h)
    coscal.fiducial_cosmology = "WMAP1"
    coscal.true_cosmology = "Planck15"
    with pytest.raises(ValueError):
        coscal.cosmo = "Planck18"
