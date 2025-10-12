import numpy as np
from meer21cm.power import *
import pytest
from scipy.signal import windows
from meer21cm import PowerSpectrum, Specification
from meer21cm.util import center_to_edges, f_21
from meer21cm.mock import generate_gaussian_field


def test_nyquist_k():
    box_len = np.array([100, 200, 60])
    box_dim = np.array([10, 20, 6])
    delta_x = np.zeros(box_dim)
    ps = FieldPowerSpectrum(
        delta_x,
        box_len,
        unitless_1=True,
        mean_center_1=True,
    )
    k_1 = [np.abs(ps.k_vec[i]).max() for i in range(3)]
    assert np.allclose(k_1, ps.k_nyquist)


def test_get_x_vector():
    box_len = np.array([100, 200, 60])
    box_dim = np.array([10, 20, 6])
    box_resol = box_len / box_dim
    xvec = get_x_vector(box_dim, box_resol)
    for i in range(3):
        xbins_i = center_to_edges(xvec[i])
        assert xbins_i[0] == 0.0
        assert xbins_i[-1] == box_len[i]
        assert np.diff(xbins_i).mean() == box_resol[i]


def test_get_k_vector():
    box_len = np.array([100, 200, 60])
    box_dim = np.array([10, 20, 6])
    box_resol = box_len / box_dim
    kvec = get_k_vector(box_dim, box_resol)
    for i in range(3):
        kvec_i = kvec[i]
        assert np.abs(kvec_i).max() == np.pi / box_resol[i]
        assert np.abs(kvec_i[kvec_i != 0]).min() == 2 * np.pi / box_len[i]


def test_get_vec_mode():
    vec1 = np.linspace(0, 5, 6)
    vec2 = np.linspace(8, 10, 3)
    vec3 = np.linspace(15, 19, 5)
    mode1 = get_vec_mode((vec1, vec2, vec3))
    mode2 = np.sqrt(
        vec1[:, None, None] ** 2 + vec2[None, :, None] ** 2 + vec3[None, None, :] ** 2
    )
    assert np.allclose(mode1, mode2)


def test_get_fourier_density():
    rand_arr = np.random.normal(size=(50, 50, 50))
    rand_fourier = get_fourier_density(rand_arr, norm="ortho")
    assert np.abs(np.std(rand_fourier) - 1.0) < 5e-2
    assert np.abs(np.mean(rand_fourier)) < 5e-2
    assert np.abs((np.abs(rand_fourier) ** 2).mean() - 1) < 5e-2
    rand_arr += 1
    rand_fourier = get_fourier_density(
        rand_arr,
        mean_center=True,
        unitless=True,
        norm="ortho",
    )
    assert np.abs(np.std(rand_fourier) - 1.0) < 5e-2
    assert np.abs(np.mean(rand_fourier)) < 5e-2
    assert np.abs((np.abs(rand_fourier) ** 2).mean() - 1) < 5e-2


def test_get_shot_noise_galaxy():
    gal_count = np.ones(100000)
    box_len = [1, 1, 1]
    shot_noise = get_shot_noise_galaxy(gal_count, box_len)
    assert np.allclose(shot_noise, 1e-5)


def test_get_power_spectrum():
    complex_rand = np.random.normal(size=100000) + 1j * np.random.normal(size=100000)
    complex_rand /= np.sqrt(2)
    power_3d = get_power_spectrum(complex_rand, [1, 1, 1], weights=np.ones(100000))
    assert np.abs(power_3d.mean() - 1) < 2e-2
    assert np.abs(power_3d.std() - 1) < 2e-2
    spindx = np.random.uniform(-3, 0)
    box_len = np.array([100, 50, 100])
    box_dim = np.array([100, 200, 60])
    box_resol = box_len / box_dim
    kvec = get_k_vector(box_dim, box_resol)
    kmode = get_vec_mode(kvec)
    ps = kmode ** (spindx)
    ps[0, 0, 0] = 0.0
    delta_x = generate_gaussian_field(box_dim, box_len, ps, None)
    # note that, generate_gaussian_field deliberately does not use get_fourier_density
    delta_fourier = get_fourier_density(delta_x)
    mean_power = (
        (get_power_spectrum(delta_fourier, box_len) / kmode ** (spindx))[kmode > 0]
    ).mean()
    assert np.abs(mean_power - 1) < 0.1
    # test poisson galaxies
    delta_x = np.zeros(100**3)
    num_g = 1000
    rand_choice = np.random.choice(np.arange(100**3), num_g, replace=False)
    delta_x[rand_choice] += 1.0
    delta_x = delta_x.reshape((100, 100, 100))
    delta_fourier = get_fourier_density(delta_x, mean_center=True, unitless=True)
    power_3d = get_power_spectrum(delta_fourier, np.array([100, 100, 100]))
    power_sn = 1e6 / num_g
    assert np.abs(power_3d.mean() / power_sn - 1) < 5e-2
    assert np.abs(power_3d.std() / power_sn - 1) < 5e-2


def test_FieldPowerSpectrum():
    box_len = np.array([100, 50, 100])
    box_dim = np.array([100, 200, 60])
    box_resol = box_len / box_dim
    kvec = get_k_vector(box_dim, box_resol)
    kmode = get_vec_mode(kvec)
    delta_x = np.zeros(box_dim).ravel()
    num_g = 10000
    rand_choice = np.random.choice(np.arange(np.prod(box_dim)), num_g, replace=False)
    delta_x[rand_choice] += 1.0
    delta_x = delta_x.reshape(box_dim)
    sn = get_shot_noise(
        delta_x,
        box_len,
    )
    ps = FieldPowerSpectrum(
        delta_x,
        box_len,
        unitless_1=True,
        mean_center_1=True,
    )
    ps.box_len
    ps.box_resol
    ps.box_ndim
    ps.x_vec
    ps.x_mode
    for i in range(3):
        assert np.allclose(ps.k_vec[i], kvec[i])
    assert np.allclose(ps.k_mode, kmode)
    ps.k_perp
    ps.k_para
    ps.fourier_field_2
    ps.auto_power_3d_2
    ps.cross_power_3d
    ps.get_fourier_field_2()
    power = ps.auto_power_3d_1
    assert np.abs((power - sn).mean() / sn) < 2e-2

    ps = FieldPowerSpectrum(
        delta_x,
        box_len,
        unitless_1=True,
        mean_center_1=True,
        field_2=delta_x,
        mean_center_2=True,
        unitless_2=True,
    )
    power = ps.auto_power_3d_2
    assert np.abs((power.mean() - sn) / sn) < 2e-2
    power = ps.cross_power_3d
    assert np.abs((power.mean() - sn) / sn) < 2e-2

    ps = PowerSpectrum(
        delta_x,
        box_len,
        model_k_from_field=True,
        unitless_1=True,
        mean_center_1=True,
        field_2=delta_x,
        mean_center_2=True,
        unitless_2=True,
        k1dbins=np.linspace(0.1, 0.5, 5),
    )
    power = ps.auto_power_3d_2
    assert np.abs((power.mean() - sn) / sn) < 2e-2
    power = ps.cross_power_3d
    assert np.abs((power.mean() - sn) / sn) < 2e-2
    p1d, k1d, nmodes = ps.get_1d_power("cross_power_3d")


def test_get_shot_noise():
    # test poisson galaxies
    delta_x = np.zeros(100**3)
    num_g = 10000
    rand_choice = np.random.choice(np.arange(100**3), num_g, replace=False)
    delta_x[rand_choice] += 1.0
    delta_x = delta_x.reshape((100, 100, 100))
    box_len = np.array([100, 100, 100])
    power = get_shot_noise(
        delta_x,
        box_len,
    )
    power_sn = 1e6 / num_g
    assert power == power_sn
    # give weights
    weights = np.random.uniform(0, 1, size=delta_x.shape)
    power = get_shot_noise(
        delta_x,
        box_len,
        weights=weights,
    )
    # accuracy depends of num_g
    assert np.abs(power - power_sn) / power_sn < 2e-2


def test_raise_error():
    delta_x = np.ones([100, 100, 100])
    box_len = [1, 1, 1]
    delta_2 = np.ones([2, 2, 2])
    with pytest.raises(AssertionError):
        ps = PowerSpectrum(delta_x, box_len, field_2=delta_2)


def test_bin_functions():
    power_3d = np.ones([10, 10, 5])
    k_perp = np.linspace(0, 99, 100).reshape((10, 10))
    kperp_edges = np.linspace(0, 100, 11)
    power_cy = bin_3d_to_cy(
        power_3d,
        k_perp,
        kperp_edges,
    )
    assert np.allclose(power_cy.shape, [5, 10])
    assert np.allclose(power_cy, np.ones_like(power_cy))
    power_3d = np.linspace(0, 9, 10)
    k_mode = np.linspace(90, 99, 10)
    k1d_edges = np.linspace(90, 100, 11)
    ps1d, ps1derr, k1deff, nmodes = bin_3d_to_1d(
        power_3d,
        k_mode,
        k1d_edges,
        error=True,
    )
    assert np.allclose(ps1d, np.linspace(0, 9, 10))
    assert np.allclose(k1deff, k_mode)
    assert np.allclose(nmodes, np.ones_like(nmodes))
    assert np.allclose(ps1derr, np.zeros_like(nmodes))
    ps1d, k1deff, _ = bin_3d_to_1d(
        power_3d,
        k_mode,
        k1d_edges,
        error=False,
    )


def test_cy_power_in_ps():
    ps = PowerSpectrum(
        kaiser_rsd=False,
    )
    ps.box_len = np.array([1000, 1000, 1000])
    ps.box_ndim = np.array([20, 20, 20])
    ps.propagate_field_k_to_model()
    ps.kperpbins = np.linspace(0, 0.09, 10)
    ps.kparabins = np.linspace(0, 0.06, 7)
    pscy, _ = ps.get_cy_power(
        "auto_power_matter_model",
    )
    kperpcen = (ps.kperpbins[1:] + ps.kperpbins[:-1]) / 2
    kparacen = (ps.kparabins[1:] + ps.kparabins[:-1]) / 2
    k_1d = np.sqrt(kperpcen[:, None] ** 2 + kparacen[None, :] ** 2)
    assert np.all(np.abs(ps.matter_power_spectrum_fnc(k_1d) / pscy - 1) < 1e-1)


def test_power_weights_renorm():
    # uniform weights should give 1
    weights = np.ones([10, 10, 10])
    assert np.allclose(power_weights_renorm(weights, weights), 1)
    assert np.allclose(power_weights_renorm(None, weights), 1)
    assert np.allclose(power_weights_renorm(weights, None), 1)
    assert np.allclose(power_weights_renorm(None, None), 1)

    # random noise, with random taper
    field_1 = np.random.normal(size=[100, 100, 101])
    power_1 = np.abs(np.fft.rfft(field_1)) ** 2
    taper_1 = np.random.uniform(0, 1, size=[100, 100, 101])
    taper_2 = np.random.uniform(0, 1, size=[100, 100, 101])
    assert np.allclose(
        get_modelpk_conv(power_1, taper_1, None).mean(), power_1.mean(), rtol=1e-2
    )
    assert np.allclose(
        get_modelpk_conv(power_1, None, taper_2).mean(), power_1.mean(), rtol=1e-2
    )
    assert np.allclose(
        get_modelpk_conv(power_1, None, None).mean(), power_1.mean(), rtol=1e-2
    )
    assert np.allclose(
        get_modelpk_conv(power_1, taper_1, taper_1).mean(), power_1.mean(), rtol=1e-2
    )
    assert np.allclose(
        get_modelpk_conv(power_1, taper_2, taper_2).mean(), power_1.mean(), rtol=1e-2
    )
    assert np.allclose(
        get_modelpk_conv(power_1, taper_1, taper_2).mean(), power_1.mean(), rtol=1e-2
    )

    # try a random thermal noise
    box_len = np.array([80, 50, 100])
    box_dim = np.array([100, 200, 41])
    rand_noise = np.random.normal(size=box_dim)
    ps = PowerSpectrum(
        rand_noise,
        box_len,
        unitless_1=False,
        mean_center_1=False,
    )
    # without taper
    power1 = ps.auto_power_3d_1
    floor1 = power1.mean()
    # with taper
    taper = windows.blackmanharris(box_dim[1])
    taper = taper[None, :, None] * np.ones(box_dim)
    ps = PowerSpectrum(
        rand_noise,
        box_len,
        unitless_1=False,
        mean_center_1=False,
        weights_grid_1=taper,
    )
    power2 = ps.auto_power_3d_1
    floor2 = power2.mean()
    assert np.abs((floor2 - floor1) / floor1) < 1e-2
    # test with one weight and no second weight
    ps = PowerSpectrum(
        rand_noise,
        box_len,
        unitless_1=False,
        mean_center_1=False,
        weights_grid_1=taper,
        field_2=rand_noise,
        mean_center_2=False,
        unitless_2=False,
    )
    power3 = ps.cross_power_3d
    floor3 = power3.mean()
    assert np.abs((floor3 - floor1) / floor1) < 1e-2
    # test clear cache for fields
    ps.field_1 = rand_noise
    # an update should clean fourier field
    assert ps._fourier_field_1 is None
    ps.field_2 = rand_noise
    assert ps._fourier_field_2 is None
    ps = PowerSpectrum(
        rand_noise,
        box_len,
        unitless_1=False,
        mean_center_1=False,
        weights_grid_1=taper,
        field_2=rand_noise,
        mean_center_2=False,
        unitless_2=False,
    )
    power3 = ps.cross_power_3d
    ps.weights_grid_1 = taper
    ps.weights_grid_2 = taper
    # an update should clean fourier field
    assert ps._fourier_field_1 is None
    assert ps._fourier_field_2 is None
    # test invoking
    ps.box_len
    ps.box_resol
    ps.box_ndim
    assert power_weights_renorm(None) == 1.0


def test_get_modelpk_conv():
    box_dim = np.array([100, 200, 40])
    test_ps = np.ones([100, 200, 21])
    # uniform weights
    test_ps_conv = get_modelpk_conv(
        test_ps, weights1_in_real=np.ones(box_dim), weights2=None, renorm=True
    )
    assert np.allclose(test_ps, test_ps_conv)
    # test a taper with ps with a spectral index
    field = PowerSpectrum(
        field_1=np.ones(box_dim),
        box_len=box_dim,
    )
    kmode = field.k_mode
    test_ps = np.zeros_like(kmode)
    # ns = 2
    test_ps[kmode != 0] = kmode[kmode != 0] ** (-2)
    # any direction would do
    taper = windows.blackmanharris(box_dim[0])[:, None, None] + np.zeros(box_dim)
    test_ps_conv = get_modelpk_conv(test_ps, weights1_in_real=taper, weights2=taper)
    # p * k^2 should be one
    assert np.abs((test_ps_conv * kmode**2).mean() - 1) < 1e-3
    # mode-mixing is small, so every k-mode should also be p * k^2 about 1, litte var
    assert (test_ps_conv * kmode**2).std() < 1e-2


def test_get_gaussian_noise_floor():
    box_len = np.array([80, 50, 100])
    box_dim = np.array([100, 200, 41])
    box_resol = box_len / box_dim
    rand_noise = np.random.normal(size=box_dim)
    ps = PowerSpectrum(
        rand_noise,
        box_len,
        unitless_1=False,
        mean_center_1=False,
    )
    power = ps.auto_power_3d_1
    floor1 = ps.auto_power_3d_1.mean()
    floor2 = get_gaussian_noise_floor(
        1,
        box_dim,
        box_volume=np.prod(ps.box_len),
    )
    assert np.abs((floor1 - floor2) / floor1) < 2e-2
    # test weights
    counts = np.random.randint(1, 100, size=rand_noise.shape)
    rand_noise = rand_noise / np.sqrt(counts)
    ps = PowerSpectrum(
        rand_noise,
        box_len,
        unitless_1=False,
        mean_center_1=False,
    )
    power = ps.auto_power_3d_1
    floor1 = ps.auto_power_3d_1.mean()
    floor2 = get_gaussian_noise_floor(
        1,
        box_dim,
        box_volume=np.prod(ps.box_len),
        counts=counts,
    )
    assert np.abs((floor1 - floor2) / floor1) < 2e-2


# def test_get_independent_fourier_modes():
#    rand_int = np.random.randint(2, 50)
#    box_dim = np.array([2 * rand_int, 2 * rand_int, 2 * rand_int])
#    indep_modes = get_independent_fourier_modes(box_dim)
#    assert indep_modes.sum() == np.prod(box_dim) - (np.prod(box_dim - 1) // 2)
#    box_dim += 1
#    indep_modes = get_independent_fourier_modes(box_dim)
#    assert indep_modes.sum() == np.prod(box_dim) // 2 + 1


def test_model_in_real_space():
    model = ModelPowerSpectrum(kaiser_rsd=False)
    # have mu=1, but no rsd
    model.mumode = np.ones_like(model.kmode)
    model.tracer_bias_1 = 2.0
    matter_ps_real = model.matter_power_spectrum_fnc(model.kmode)
    assert np.allclose(model.auto_power_tracer_1_model, matter_ps_real * 4)
    model.tracer_bias_2 = 3.0
    assert np.allclose(model.auto_power_tracer_2_model, matter_ps_real * 9)
    assert np.allclose(model.cross_power_tracer_model, matter_ps_real * 6)


@pytest.mark.parametrize("fog_profile", ["gaussian", "lorentz"])
def test_ModelPowerSpectrum(fog_profile):
    # test fog
    model = ModelPowerSpectrum(fog_profile=fog_profile)
    assert np.allclose(model.fog_term(1), np.ones(model.kmode.shape))
    model.mumode = np.ones_like(model.kmode)
    assert np.allclose(model.fog_term(np.inf), np.zeros(model.kmode.shape))

    # test matter power with no rsd
    model = ModelPowerSpectrum(fog_profile=fog_profile)
    matter_ps_real = model.matter_power_spectrum_fnc(model.kmode)
    assert np.allclose(model.auto_power_matter_model, matter_ps_real)

    # add rsd, test kaiser term
    model.mumode = np.ones_like(model.kmode)
    matter_ps_rsd = model.auto_power_matter_model
    assert np.allclose(matter_ps_rsd / matter_ps_real, (1 + model.f_growth) ** 2)

    # has mumode, but turn off rsd
    model.kaiser_rsd = False
    assert np.allclose(model.auto_power_matter_model, matter_ps_real)

    # test tracer with no rsd but with bias
    model = ModelPowerSpectrum(tracer_bias_1=2.0, fog_profile=fog_profile)
    assert model.auto_power_tracer_2_model is None
    assert model.cross_power_tracer_model is None
    tracer_ps_rsd = model.auto_power_tracer_1_model
    assert np.allclose(tracer_ps_rsd, matter_ps_real * 4)

    # add rsd
    model.mumode = np.ones_like(model.kmode)
    tracer_ps_rsd = model.auto_power_tracer_1_model
    assert np.allclose(
        tracer_ps_rsd / matter_ps_real, (1 + model.f_growth / 2.0) ** 2 * 4
    )

    # test 2 tracers with no rsd but with bias
    model = ModelPowerSpectrum(
        tracer_bias_1=2.0,
        tracer_bias_2=2.0,
        cross_coeff=0.5,
        fog_profile=fog_profile,
    )
    # test tracer 2 dep
    model.auto_power_tracer_2_model
    model.cross_power_tracer_model
    # update
    model.tracer_bias_2 = 3.0
    tracer_ps_rsd = model.auto_power_tracer_1_model
    assert np.allclose(tracer_ps_rsd, matter_ps_real * 4)
    tracer_ps_rsd = model.auto_power_tracer_2_model
    assert np.allclose(tracer_ps_rsd, matter_ps_real * 9)

    # add rsd
    model.mumode = np.ones_like(model.kmode)
    tracer_ps_rsd = model.auto_power_tracer_2_model
    cross_ps_rsd = model.cross_power_tracer_model
    assert np.allclose(
        tracer_ps_rsd / matter_ps_real, (1 + model.f_growth / 3.0) ** 2 * 9
    )
    assert np.allclose(
        cross_ps_rsd / matter_ps_real,
        (1 + model.f_growth / 2.0) * (1 + model.f_growth / 3.0) * 6 - 6 + 6 * 0.5,
    )
    # test change r
    model.cross_coeff = 0.9
    cross_ps_rsd = model.cross_power_tracer_model
    assert np.allclose(
        cross_ps_rsd / matter_ps_real,
        (1 + model.f_growth / 2.0) * (1 + model.f_growth / 3.0) * 6 - 6 + 6 * 0.9,
    )
    # test change v
    model.sigma_v_1 = 1e20
    assert np.allclose(model.auto_power_tracer_1_model, np.zeros_like(model.kmode))
    model.sigma_v_2 = 1e20
    assert np.allclose(model.auto_power_tracer_2_model, np.zeros_like(model.kmode))
    model.weights_grid_1 = 1
    assert model._auto_power_tracer_1_model is None
    model.weights_grid_2 = 1
    assert model._auto_power_tracer_2_model is None
    model = ModelPowerSpectrum()
    model.compensate = False
    assert model.gridding_compensation() == 1.0


def test_sigma_z_model():
    model = ModelPowerSpectrum(tracer_bias_2=1.0)
    model.mumode = np.ones_like(model.kmode)
    matter_ps = model.auto_power_matter_model
    model.sigma_z_1 = 1e4
    matter_ps_2 = model.auto_power_matter_model
    tracer_ps_1_2 = model.auto_power_tracer_1_model
    # should not affect the matter power
    assert np.allclose(matter_ps_2, matter_ps)
    # tracer power should be almost 0
    assert np.allclose(tracer_ps_1_2, 0)
    model.sigma_z_2 = 1e4
    model.sigma_z_1 = 0
    assert np.all(model.auto_power_tracer_1_model > 0)
    assert np.allclose(model.auto_power_tracer_2_model, 0)
    assert np.allclose(model.cross_power_tracer_model, 0)


def test_gaussian_beam_attenuation():
    # small scale goes to almost zero
    Bbeam_test = gaussian_beam_attenuation(100, 1)
    assert Bbeam_test < 1e-3
    # large scale is not affected
    Bbeam_test = gaussian_beam_attenuation(1e-4, 1)
    assert np.abs(1 - Bbeam_test) < 1e-3
    # FWHM/2
    Bbeam_test = gaussian_beam_attenuation(np.sqrt(2 * np.log(2)), 1)
    assert np.allclose(Bbeam_test, 0.5)
    # without beam
    model = ModelPowerSpectrum(
        tracer_bias_1=2.0,
        tracer_bias_2=3.0,
        cross_coeff=0.5,
        # sigma_beam_ch=np.ones(100)
    )
    model.mumode = np.ones_like(model.kmode)
    tracer_ps_rsd_1 = model.auto_power_tracer_1_model
    tracer_ps_rsd_2 = model.auto_power_tracer_2_model
    cross_ps_rsd = model.cross_power_tracer_model
    # with beam
    model = ModelPowerSpectrum(
        tracer_bias_1=2.0,
        tracer_bias_2=3.0,
        cross_coeff=0.5,
        sigma_beam_ch=np.ones(model.nu.size),
    )
    model.mumode = np.ones_like(model.kmode)
    tracer_ps_rsd_1_b = model.auto_power_tracer_1_model
    tracer_ps_rsd_2_b = model.auto_power_tracer_2_model
    cross_ps_rsd_b = model.cross_power_tracer_model
    assert np.allclose(tracer_ps_rsd_1_b, tracer_ps_rsd_1)
    assert np.allclose(tracer_ps_rsd_2_b, tracer_ps_rsd_2)
    assert np.allclose(cross_ps_rsd_b, cross_ps_rsd)
    sigma_beam = model.sigma_beam_in_mpc
    fwhm_beam = sigma_beam / (np.sqrt(2 * np.log(2)))
    model = ModelPowerSpectrum(
        kmode=np.array([1 / fwhm_beam, 1 / fwhm_beam]),
        mumode=np.array([0, 0]),
        tracer_bias_1=2.0,
        tracer_bias_2=3.0,
        cross_coeff=0.5,
        sigma_beam_ch=np.ones(model.nu.size),
    )
    tracer_ps_rsd_1_b = model.auto_power_tracer_1_model
    tracer_ps_rsd_c_b = model.cross_power_tracer_model

    # test if tracer_2 can be updated when beam included
    model.include_beam = [True, True]
    tracer_ps_rsd_c_b2 = model.cross_power_tracer_model

    model = ModelPowerSpectrum(
        kmode=np.array([1 / fwhm_beam, 1 / fwhm_beam]),
        mumode=np.array([0, 0]),
        tracer_bias_1=2.0,
        tracer_bias_2=3.0,
        cross_coeff=0.5,
        # sigma_beam_ch=np.ones(100)
    )

    tracer_ps_rsd_1 = model.auto_power_tracer_1_model
    tracer_ps_rsd_c = model.cross_power_tracer_model
    assert np.allclose(tracer_ps_rsd_1 / tracer_ps_rsd_1_b, [4, 4])
    # tracer_2 does not have beam
    assert np.allclose(tracer_ps_rsd_c / tracer_ps_rsd_c_b, [2, 2])
    # tracer_2 has beam
    assert np.allclose(tracer_ps_rsd_c / tracer_ps_rsd_c_b2, [4, 4])


def test_set_corrtype():
    box_len = np.array([80, 50, 100])
    box_dim = np.array([100, 200, 41])
    box_resol = box_len / box_dim
    rand_noise = np.random.normal(size=box_dim)
    ps = PowerSpectrum(
        rand_noise,
        box_len,
        unitless_1=False,
        mean_center_1=False,
    )
    # simply test invoking
    ps.box_origin
    ps.box_len
    ps.box_resol
    ps.box_ndim
    ps.set_corr_type("gal", 1)
    assert ps.mean_center_1 == True
    assert ps.unitless_1 == True
    ps.set_corr_type("HI", 1)
    assert ps.mean_center_1 == False
    assert ps.unitless_1 == False
    power = ps.auto_power_3d_1
    floor1 = ps.auto_power_3d_1.mean()
    floor2 = get_gaussian_noise_floor(
        1,
        box_dim,
        box_volume=np.prod(ps.box_len),
    )
    assert np.abs((floor1 - floor2) / floor1) < 2e-2
    with pytest.raises(ValueError):
        ps.set_corr_type("gal", 3)
    with pytest.raises(ValueError):
        ps.set_corr_type("something", 1)


def test_step_window_attenuation():
    assert step_window_attenuation(1, 1) == np.sinc(1 / np.pi / 2)
    assert step_window_attenuation(0, 0) == 1.0


def test_temp_amp():
    box_len = np.array([80, 50, 100])
    box_dim = np.array([10, 20, 40])
    box_resol = box_len / box_dim
    rand_noise = np.random.normal(size=box_dim)
    ps = ModelPowerSpectrum()
    assert ps.map_sampling() == 1
    ps = PowerSpectrum(
        rand_noise,
        box_len,
        unitless_1=False,
        mean_center_1=False,
        field_2=rand_noise,
        mean_center_2=False,
        unitless_2=False,
        mean_amp_1="average_hi_temp",
        mean_amp_2="one",
        tracer_bias_2=1.0,
        # sampling_resol=[0.1, 0.1, 0.1],
        model_k_from_field=True,
        tracer_bias_1=1.0,
    )
    # test a custom avg
    ps.one = 1.0
    assert np.allclose(ps.auto_power_tracer_2_model, ps.auto_power_matter_model)
    assert np.allclose(
        ps.auto_power_tracer_1_model,
        ps.auto_power_matter_model * ps.average_hi_temp**2,
    )
    assert np.allclose(
        ps.cross_power_tracer_model, ps.auto_power_matter_model * ps.average_hi_temp
    )


def test_noise_power_from_map(test_W):
    sp = PowerSpectrum(
        sampling_resol="auto",
        survey="meerklass_2021",
        band="L",
    )
    sp.data = np.random.normal(size=sp.map_has_sampling.shape) * sp.map_has_sampling
    ps = sp
    ps.downres_factor_transverse = 1.5
    ps.downres_factor_radial = 2.0
    ps.box_buffkick = 10
    ps.compensate = False
    noise_map, noise_weights, pix_counts = ps.grid_data_to_field()
    ps.field_1 = noise_map
    ps.weights_field_1 = (ps.counts_in_box > 0).astype("float")
    ps.weights_grid_1 = None
    renorm = power_weights_renorm(ps.weights_field_1, ps.weights_field_1)
    sigma_n = np.zeros_like(ps.field_1)
    sigma_n[ps.counts_in_box > 0] = np.sqrt(1 / ps.counts_in_box[ps.counts_in_box > 0])
    ptn = (sigma_n**2).mean() * np.prod(ps.box_resol) * renorm
    assert np.abs((ps.auto_power_3d_1.mean() - ptn) / ptn) < 1e-2
    gaussian_ness = np.abs(
        (ps.auto_power_3d_1.mean() - ps.auto_power_3d_1.std())
        / ps.auto_power_3d_1.mean()
    )
    assert gaussian_ness < 0.05


def test_renorm_ps():
    # need better test later
    ps = PowerSpectrum()
    ps._box_len = np.array([10, 10, 10])
    ps._box_ndim = np.array([10, 10, 10])
    ps.weights_field_1 = np.ones_like(ps.kmode)
    ps.weights_field_2 = np.ones_like(ps.kmode)
    ps.unitless_1 = False
    ps.unitless_2 = False
    assert np.allclose(ps.renorm_ps_1, 1.0)
    assert np.allclose(ps.renorm_ps_2, 1.0)
    assert np.allclose(ps.renorm_ps_cross, 1.0)
    ps.unitless_1 = True
    ps.unitless_2 = True
    assert np.allclose(ps.renorm_ps_1, 1.0)
    assert np.allclose(ps.renorm_ps_2, 1.0)
    assert np.allclose(ps.renorm_ps_cross, 1.0)


def test_cache():
    ps = PowerSpectrum()
    ps.auto_power_matter_model
    ps.mumode = np.ones_like(ps.kmode)
    assert ps._auto_power_matter_model is None
    assert ps.beam_attenuation() == 1.0
    box_len = np.array([80, 50, 100])
    box_dim = np.array([100, 200, 41])
    box_resol = box_len / box_dim
    rand_noise = np.random.normal(size=box_dim)
    ps = FieldPowerSpectrum(
        rand_noise,
        box_len,
        unitless_1=False,
        mean_center_1=False,
        field_2=rand_noise,
        mean_center_2=False,
        unitless_2=False,
    )
    power1 = ps.auto_power_3d_1
    floor1 = power1.mean()
    power3 = ps.cross_power_3d
    floor3 = power3.mean()
    assert np.abs((floor3 - floor1) / floor1) < 1e-2
    ps.unitless_2 = False
    assert ps._fourier_field_2 is None
    ps.fourier_field_2
    ps.mean_center_2 = False
    assert ps._fourier_field_2 is None
    ps.fourier_field_2


def test_grid_gal(test_gal_fits, test_W):
    ps = PowerSpectrum(
        gal_file=test_gal_fits,
        survey="meerklass_2021",
        band="L",
    )
    ps.W_HI = (test_W * ps.nu[None, None, :]) > 0
    ps.data = ps.W_HI
    ps.w_HI = ps.W_HI
    ps = PowerSpectrum(
        gal_file=test_gal_fits,
        data=ps.data,
        map_has_sampling=ps.W_HI,
        weights_map_pixel=ps.w_HI,
        init_box_from_map_data=True,
        include_sky_sampling=[True, True],
        survey="meerklass_2021",
        band="L",
        tracer_bias_2=1.0,  # just for invoking some tests
    )
    ps.read_gal_cat()
    ps.grid_gal_to_field()
    ps.apply_taper_to_field(2)


def test_shot_noise_tapering():
    ps = PowerSpectrum(
        nu=[f_21, f_21],
        include_sky_sampling=[False, False],
        box_len=[200, 400, 600],
        box_ndim=[40, 80, 120],
        survey="meerklass_2021",
        band="L",
    )
    num_g = 100000
    pos = [
        np.random.uniform(0, ps.box_len[i], size=num_g) for i in range(len(ps.box_ndim))
    ]
    pos = np.array(pos)
    box_edges = [
        np.linspace(0, ps.box_len[i], ps.box_ndim[i] + 1)
        for i in range(len(ps.box_ndim))
    ]
    field, _ = np.histogramdd(pos.T, bins=box_edges)
    ps.field_1 = field
    ps.mean_center_1 = True
    ps.unitless_1 = True
    psn = np.prod(ps.box_len) / num_g
    num_g = 100000
    pos = [
        np.random.uniform(0, ps.box_len[i], size=num_g) for i in range(len(ps.box_ndim))
    ]
    pos = np.array(pos)
    box_edges = [
        np.linspace(0, ps.box_len[i], ps.box_ndim[i] + 1)
        for i in range(len(ps.box_ndim))
    ]
    field, _ = np.histogramdd(pos.T, bins=box_edges)
    ps.field_1 = field
    ps.mean_center_1 = True
    ps.unitless_1 = True
    assert (np.abs(ps.auto_power_3d_1.mean() - psn) / psn) < 2e-2
    taper = [ps.taper_func(ps.box_ndim[i]) for i in range(3)]
    taper = taper[0][:, None, None] * taper[1][None, :, None] * taper[2][None, None, :]
    ps.weights_1 = taper
    assert (np.abs(ps.auto_power_3d_1.mean() - psn) / psn) < 5e-2


def test_rot_back():
    ps = PowerSpectrum(
        include_sky_sampling=[False, False],
        survey="meerklass_2021",
        band="L",
    )
    ps.W_HI = np.ones_like(ps.W_HI)
    ps.get_enclosing_box()
    ra_test, dec_test, z_test, _ = ps.ra_dec_z_for_coord_in_box(ps.pix_coor_in_box)
    z_test = z_test.reshape((-1, len(ps.nu)))
    ra_test = ra_test.reshape((-1, len(ps.nu)))
    dec_test = dec_test.reshape((-1, len(ps.nu)))
    # map pixels should just be map pixels
    assert np.allclose(z_test[0], ps.z_ch)
    assert np.allclose(ps.ra_map.ravel(), ra_test[:, 0])
    assert np.allclose(ps.dec_map.ravel(), dec_test[:, 0])


def test_poisson_gal_gen():
    raminMK, ramaxMK = 334, 357
    decminMK, decmaxMK = -35, -26.5
    ra_range = (raminMK, ramaxMK)
    dec_range = (decminMK, decmaxMK)
    ps = PowerSpectrum(
        ra_range=ra_range,
        dec_range=dec_range,
        omega_hi=5.4e-4,
        mean_amp_1="average_hi_temp",
        tracer_bias_1=1.5,
        tracer_bias_2=1.9,
        survey="meerklass_2021",
        band="L",
        # seed=42,
        kmax=10.0,
        num_particle_per_pixel=2,
        box_buffkick=[5, 5, 5],
    )
    ps._ra_gal = np.ones(40000)
    ps._dec_gal = np.ones(40000)
    ps._z_gal = np.ones(40000)
    radecfreq = ps.gen_random_poisson_galaxy()
    ps.compensate = False
    ps.grid_gal_to_field(radecfreq)
    volume = (
        (ps.W_HI[:, :, 0].sum() * ps.pixel_area * (np.pi / 180) ** 2)
        / 3
        * (
            ps.comoving_distance(ps.z_ch.max()) ** 3
            - ps.comoving_distance(ps.z_ch.min()) ** 3
        ).value
    )
    k1dedges = np.geomspace(0.05, 1, 21)
    ps.k1dbins = k1dedges
    psn = volume / ps.ra_gal.size
    psn1d, _, _ = ps.get_1d_power(
        "auto_power_3d_2",
    )
    plateau = psn1d[-5:].mean()
    assert np.abs(plateau - psn) / psn < 2.5e-1


def test_1d_k_cut():
    ps = PowerSpectrum()
    ps._box_len = np.array([10, 10, 10]) * np.pi
    ps._box_ndim = np.array([10, 10, 10])
    ps.k1dbins = np.linspace(0, 10, 2)
    _, _, nmodes = ps.get_1d_power(np.ones_like(ps.k_mode), k_xyz_min=[0.1, 0.1, 0.1])
    assert nmodes[0] == 9**2 * 5
    _, _, nmodes = ps.get_1d_power(np.ones_like(ps.k_mode), k_xyz_max=[0.1, 100, 100])
    assert nmodes[0] == 1 * 10 * 6 - 1
    _, _, nmodes = ps.get_1d_power(np.ones_like(ps.k_mode), k_perppara_min=[0.1, 0.1])
    assert nmodes[0] == 99 * 5
    _, _, nmodes = ps.get_1d_power(np.ones_like(ps.k_mode), k_perppara_max=[0.1, 100])
    assert nmodes[0] == 1 * 6 - 1


def test_no_init_power():
    ps = PowerSpectrum()
    assert ps.auto_power_tracer_2_model_noobs is None
    assert ps.cross_power_tracer_model_noobs is None


def test_map_sampling():
    ps = PowerSpectrum(
        np.ones((10, 10, 10)),
        np.array([10, 10, 10]),
        model_k_from_field=True,
    )
    ps.sampling_resol = ps.box_resol
    # if sampling resol is the same as box resol, then compensate and step sampling is the same
    assert np.allclose(ps.gridding_compensation(), ps.map_sampling())


def test_average_model_hi_temp():
    raminMK, ramaxMK = 334, 357
    decminMK, decmaxMK = -35, -26.5
    ra_range = (raminMK, ramaxMK)
    dec_range = (decminMK, decmaxMK)
    # all the other settings follow the default, which corresponds to the MeerKLASS L-band deep-field survey
    ps = PowerSpectrum(
        survey="meerklass_2021",
        band="L",
        mean_amp_1="average_hi_temp",
        ra_range=ra_range,
        dec_range=dec_range,
    )
    assert np.allclose(ps.average_model_hi_temp, ps.average_hi_temp, rtol=1e-3)
    ps.get_enclosing_box()
    temp_box_avg = (
        ps.model_hi_temp_in_box * ps.counts_in_box
    ).sum() / ps.counts_in_box.sum()
    assert np.allclose(temp_box_avg, ps.average_model_hi_temp, rtol=1e-3)
