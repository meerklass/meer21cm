from meer21cm import Specification
from astropy.cosmology import Planck18, Planck15
import numpy as np
from astropy import units, constants
import pytest
from astropy.wcs.utils import proj_plane_pixel_area
from meer21cm.util import freq_to_redshift, center_to_edges, f_21, pca_clean
from meer21cm.telescope import dish_beam_sigma


def test_nu_range():
    spec = Specification()
    assert spec.nu_min == -np.inf
    assert spec.nu_max == np.inf
    assert np.allclose(spec.nu, [f_21 - 1, f_21])
    with pytest.raises(ValueError):
        spec = Specification(
            nu=np.array([1e7, 1e7]),
            nu_min=809 * 1e6,
            nu_max=910 * 1e6,
        )


def test_volume():
    spec = Specification(
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


def test_cosmo():
    spec = Specification()
    assert spec.cosmo.h == Planck18.h
    # an update should properly update the internal functions as well
    spec.cosmo = "Planck15"
    assert spec.cosmo is Planck15
    assert spec.cosmo.h == Planck15.h


def test_update_pars():
    spec = Specification()
    # test string input
    spec.cosmo = "Planck15"
    assert spec.cosmo is Planck15
    assert spec.cosmo.h == Planck15.h
    # test direct input
    spec.cosmo = Planck15
    # test nu
    spec.nu = [f_21, f_21]
    assert np.allclose(spec.z, 0)


def test_defaults(test_nu, test_W):
    spec = Specification(
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


def test_unit_conversion():
    spec = Specification(map_unit=units.mK)
    assert spec.map_unit_type == "T"
    spec = Specification(map_unit=units.Jy)
    assert spec.map_unit_type == "F"
    with pytest.raises(Exception):
        spec = Specification(map_unit=units.m)


def test_velocity(test_nu, test_wproj):
    spec = Specification(
        survey="meerklass_2021",
        band="L",
    )
    assert np.allclose(spec.dvdf_ch, (constants.c / test_nu).to("km/s").value)
    assert np.allclose(
        spec.vel_resol_ch,
        (constants.c / test_nu).to("km/s").value * np.diff(test_nu).mean(),
    )
    assert np.allclose(spec.vel_resol, spec.vel_resol_ch.mean())
    assert np.allclose(spec.dvdf, spec.dvdf_ch.mean())
    assert np.allclose(spec.freq_resol, np.diff(test_nu).mean())
    assert np.allclose(spec.pixel_area, proj_plane_pixel_area(test_wproj))
    assert np.allclose(spec.pix_resol, np.sqrt(spec.pixel_area))


def test_read_pickle(test_pickle):
    spec = Specification()
    # should be None
    spec.read_from_pickle()
    # set pickle file
    spec.pickle_file = test_pickle
    spec.nu_min = -np.inf
    spec.nu_max = np.inf
    spec.read_from_pickle()
    spec.weighting = "uniform"
    spec.read_from_pickle()


def test_read_fits(test_fits):
    sp = Specification(
        nu_min=-np.inf,
        nu_max=np.inf,
    )
    # should be None
    sp.read_from_fits()
    sp.read_gal_cat()
    # set map file
    sp.map_file = test_fits
    # set wrong dimensions, see if they get updated correctly
    sp.num_pix_x = 1
    sp.num_pix_y = 1
    sp.read_from_fits()
    assert np.allclose(sp.data.shape, (133, 73, 2))
    assert np.allclose(sp.counts.shape, (133, 73, 2))
    assert np.allclose(sp.ra_map.shape, (133, 73))
    assert np.allclose(sp.dec_map.shape, (133, 73))

    assert np.allclose(sp.map_has_sampling.shape, (133, 73, 2))
    assert sp.num_pix_x == 133
    assert sp.num_pix_y == 73
    assert len(sp.nu) == 2
    sp.W_HI
    # if weights are correctly updated
    assert np.allclose(sp.w_HI, sp.counts)
    # uniform weighting should be just binary
    sp.weighting = "uniform"
    sp.read_from_fits()
    assert np.allclose(sp.w_HI, sp.counts > 0)


def test_gal_readin(test_gal_fits):
    sp = Specification(
        survey="meerklass_2021",
        band="L",
    )
    sp.gal_file = test_gal_fits
    sp.read_gal_cat()
    nu_edges = center_to_edges(sp.nu)
    # see if trimming within the frequency range worked
    assert np.mean(sp.freq_gal >= nu_edges[sp.ch_id_gal]) == 1
    assert np.mean(sp.freq_gal <= nu_edges[sp.ch_id_gal + 1]) == 1
    assert len(sp.ra_gal) == len(sp.z_gal)
    assert len(sp.dec_gal) == len(sp.z_gal)


def test_beam_update():
    ps = Specification(
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
    ps.cosmo = "Planck15"
    s4 = ps.sigma_beam_ch_in_mpc
    assert not np.allclose(s4, s3)


def test_beam_image():
    sp = Specification(
        survey="meerklass_2021",
        band="L",
    )
    # test None
    assert sp.beam_image is None
    D_dish = 13.5
    sigma_exp = dish_beam_sigma(
        D_dish,
        sp.nu,
    )
    sp.sigma_beam_ch = sigma_exp
    beam_image = sp.beam_image
    sigma_beam_from_image = (
        np.sqrt(beam_image.sum(axis=(0, 1)) / 2 / np.pi) * sp.pix_resol
    )
    assert np.allclose(sigma_beam_from_image, sp.sigma_beam_ch, rtol=1e-3, atol=1e-3)
    sp.beam_model = "cos"
    beam_image = sp.beam_image
    sigma_beam_from_image = (
        np.sqrt(beam_image.sum(axis=(0, 1)) / 2 / np.pi) * sp.pix_resol
    )
    # for cos beam image sigma will be different from input since it is not
    # an exact match
    assert np.allclose(sigma_beam_from_image, sp.sigma_beam_ch, rtol=1e-1, atol=5e-2)
    # no parameter, just an input model
    sp.beam_model = "kat"
    beam_image = sp.beam_image
    sigma_beam_from_image = (
        np.sqrt(beam_image.sum(axis=(0, 1)) / 2 / np.pi) * sp.pix_resol
    )
    # sigma_beam_ch updated by the input model
    assert np.allclose(sigma_beam_from_image, sp.sigma_beam_ch)


def test_convolve_data():
    sp = Specification(
        survey="meerklass_2021",
        band="L",
    )
    D_dish = 13.5
    sp.sigma_beam_ch = dish_beam_sigma(
        D_dish,
        sp.nu,
    )
    sp.data = np.zeros(sp.W_HI.shape)
    sp.data[sp.num_pix_x // 2, sp.num_pix_y // 2] = 1.0
    sp.w_HI = sp.W_HI
    sp.convolve_data(sp.beam_image)
    # test renorm
    sum_test = sp.data.sum(axis=(0, 1))
    assert np.allclose(sum_test, np.ones_like(sp.nu))


def test_update_beam_type():
    sp = Specification(survey="meerklass_2021", band="L", beam_model="kat")
    assert sp.beam_type == "anisotropic"
    sp = Specification()
    assert sp.beam_type == "isotropic"
    with pytest.raises(ValueError):
        sp.beam_model = "something"


def test_z_interp():
    ps = Specification(
        survey="meerklass_2021",
        band="L",
    )
    func = ps.z_as_func_of_comov_dist
    z_rand = np.random.uniform(ps.z_ch.min(), ps.z_ch.max(), size=100)
    assert np.allclose(func(ps.cosmo.comoving_distance(z_rand).value), z_rand)


def test_trim_gal():
    sp = Specification(
        survey="meerklass_2021",
        band="L",
        ra_range=[-1, 1],
        dec_range=[-1, 1],
    )
    sp._ra_gal = np.array([0, 0, 2])
    sp._dec_gal = np.array([0, -2, 0])
    sp._z_gal = np.array([0.42, 0.42, 0.42])
    assert np.allclose(sp.trim_gal_to_range(), [1, 0, 0])


def test_get_jackknife_patches():
    ps = Specification(
        survey="meerklass_2021",
        band="L",
    )
    with pytest.raises(AssertionError):
        ps.get_jackknife_patches(
            ra_patch_num=4,
            dec_patch_num=4,
            nu_patch_num=2,
        )
    ps.ra_range = (334, 357)
    ps.dec_range = (-35, -26.5)
    mask_arr = ps.get_jackknife_patches(ra_patch_num=8, dec_patch_num=4, nu_patch_num=2)
    assert mask_arr.shape == (8, 4, 2, ps.num_pix_x, ps.num_pix_y, ps.nu.size)
    # check ra dec makes sense
    mask_pixel = mask_arr.sum((2, 3, 4, 5))
    mask_pixel_mean = mask_pixel.mean()
    mask_pixel_std = mask_pixel.std()
    assert mask_pixel_std / mask_pixel_mean < 5e-2
    # check nu makes sense
    nu_pixel = mask_arr.sum((0, 1, 2, 3, 4))
    assert nu_pixel.std() / nu_pixel.mean() < 1e-2


def test_create_white_noise_map():
    ra_range_MK = (334, 357)
    dec_range_MK = (-35, -26.5)
    ps = Specification(
        band="L",  # band and survey will produce some pre-defined cuts to select
        survey="meerklass_2021",  # the clean frequency sub-band
        ra_range=ra_range_MK,
        dec_range=dec_range_MK,
    )
    noise_map = ps.create_white_noise_map(
        0.1,
    )
    std = ((noise_map * np.sqrt(ps.counts))[ps.counts > 0]).std()
    assert np.allclose(std, 0.1, rtol=5e-3)
    ps.counts = np.random.uniform(1, 100, size=ps.data.shape) * ps.W_HI
    noise_map = ps.create_white_noise_map(0.1, counts=ps.counts)
    std = ((noise_map * np.sqrt(ps.counts))[ps.counts > 0]).std()
    assert np.allclose(std, 0.1, rtol=5e-3)


def test_check_is_map_noiselike_using_pca():
    ra_range_MK = (334, 357)
    dec_range_MK = (-35, -26.5)
    ps = Specification(
        band="L",  # band and survey will produce some pre-defined cuts to select
        survey="meerklass_2021",  # the clean frequency sub-band
        ra_range=ra_range_MK,
        dec_range=dec_range_MK,
    )
    noise_map = ps.create_white_noise_map(
        0.1,
    )
    N_fg = 15
    res_map, A_mat = pca_clean(noise_map, N_fg, weights=ps.W_HI, return_A=True)
    ps.data = noise_map
    res_var, noise_var = ps.check_is_map_noiselike_using_pca(A_mat, sigma_N=0.1)
    assert np.allclose(
        res_var,
        noise_var,
        rtol=1e-1,
    )
