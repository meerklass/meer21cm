from meer21cm import Specification
from astropy.cosmology import Planck18
import numpy as np
from astropy import units, constants
import pytest
from astropy.wcs.utils import proj_plane_pixel_area
from meer21cm.util import (
    center_to_edges,
    f_21,
    pca_clean,
    create_wcs,
    get_ang_between_coord,
)
from meer21cm.telescope import dish_beam_sigma
from meer21cm.skymap import HealpixSkyMap
import healpy as hp
from meer21cm.telescope import weighted_smoothing_healpix


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


def test_update_nu():
    spec = Specification()
    # test nu
    spec.nu = [f_21, f_21]
    assert np.allclose(spec.z, 0)


@pytest.mark.parametrize("precision,dtype", [(True, np.float64), (False, np.float32)])
def test_precision_dtype_casting(precision, dtype):
    spec = Specification(precision=precision)
    assert spec.real_dtype == dtype
    assert spec.nu.dtype == dtype
    assert spec.data.dtype == dtype
    assert spec.counts.dtype == dtype
    assert spec.weights_map_pixel.dtype == dtype
    spec.sigma_beam_ch = 0.01
    assert spec.sigma_beam_ch.dtype == dtype


def test_precision_is_init_only():
    spec = Specification(precision=False)
    with pytest.raises(AttributeError):
        spec.precision = True


@pytest.mark.parametrize("bad_precision", [1, 0, 1.0, "true", None, np.bool_(True)])
def test_precision_must_be_python_bool(bad_precision):
    with pytest.raises(TypeError, match="precision must be bool"):
        Specification(precision=bad_precision)


def test_batch_number_init_and_validation():
    spec = Specification(batch_number=3)
    assert spec.batch_number == 3
    for bad_batch in [0, -1, 1.5, "2", True]:
        with pytest.raises(TypeError, match="batch_number must be a positive integer"):
            Specification(batch_number=bad_batch)
    with pytest.raises(AttributeError):
        spec.batch_number = 2


def test_wcs_geometry_is_init_only(test_wproj):
    spec = Specification(wproj=test_wproj, num_pix_x=5, num_pix_y=5)
    assert spec.wproj is test_wproj
    assert spec.num_pix_x == 5
    assert spec.num_pix_y == 5
    for name, new_value in (
        ("num_pix_x", 11),
        ("num_pix_y", 13),
    ):
        with pytest.raises(AttributeError):
            setattr(spec, name, new_value)
    wproj_alt = create_wcs(10.0, 20.0, [5, 5], 1.0)
    with pytest.raises(AttributeError):
        spec.wproj = wproj_alt


def test_healpix_spec_hp_nside_and_pixel_id():
    spec = Specification(hp_nside=128, ra_range=(40, 50), dec_range=(0, 5))
    assert spec.skymap.format == "healpix"
    assert spec.hp_nside == 128
    assert spec.pixel_id.size > 0
    assert np.all(spec.pixel_id >= 0)
    assert np.all(spec.pixel_id < hp.nside2npix(spec.hp_nside))
    nu_n = len(spec.nu)
    assert spec.data.shape == (spec.pixel_id.size, nu_n)
    ra, dec = hp.pix2ang(spec.hp_nside, spec.pixel_id, lonlat=True)
    assert np.all(ra > 40) and np.all(ra < 50)
    assert np.all(dec > 0) and np.all(dec < 5)


def test_healpix_skymap_via_ctor():
    geom = HealpixSkyMap(2, pixel_id=np.array([0, 13], dtype=np.int64))
    spec = Specification(skymap=geom)
    assert spec.hp_nside == 2
    assert np.array_equal(spec.pixel_id, geom.pixel_id)


def test_healpix_spec_rejects_predictable_survey_with_hp():
    with pytest.raises(ValueError, match="WCS-only"):
        Specification(survey="meerklass_2021", band="L", hp_nside=32)


def test_healpix_spec_mutually_exclusive_skymap_and_hp():
    g = HealpixSkyMap(1, pixel_id=np.array([0], dtype=np.int64))
    with pytest.raises(ValueError, match="only one of skymap or hp_nside"):
        Specification(skymap=g, hp_nside=1)


def test_wcs_spec_has_no_hp_nside_or_pixel_id(test_wproj):
    spec = Specification(wproj=test_wproj, num_pix_x=4, num_pix_y=4)
    with pytest.raises(KeyError):
        _ = spec.hp_nside
    with pytest.raises(KeyError):
        _ = spec.pixel_id


def test_healpix_spec_has_no_wproj():
    spec = Specification(hp_nside=8, ra_range=(0, 10), dec_range=(-5, 5))
    with pytest.raises(KeyError):
        _ = spec.wproj


def test_predefined_spec_maximum_sampling_channel_axis():
    spec = Specification(survey="meerklass_2021", band="L")
    spec.map_has_sampling = np.zeros_like(spec.map_has_sampling)
    spec.map_has_sampling[:, :, 1] = True
    assert spec.maximum_sampling_channel == np.argmax(
        spec.map_has_sampling.sum(axis=(0, 1))
    )
    spec = Specification(hp_nside=32, ra_range=(0, 10), dec_range=(-5, 5))
    spec.map_has_sampling = np.zeros_like(spec.map_has_sampling)
    spec.map_has_sampling[:, 1] = True
    assert spec.maximum_sampling_channel == np.argmax(
        spec.map_has_sampling.sum(axis=(0))
    )


def test_healpix_pixel_area_square_degrees():
    """HEALPix pixel_area must use sq deg like WCS (cdelt product), not healpy steradians."""
    nside = 256
    geom = HealpixSkyMap(nside, pixel_id=np.array([0], dtype=np.int64))
    expected = float(hp.nside2pixarea(nside, degrees=True))
    assert geom.pixel_area == expected
    assert geom.pix_resol == pytest.approx(np.sqrt(expected))


def test_healpix_get_beam_image_not_implemented():
    nu = np.linspace(1.2e9, 1.21e9, 3)
    spec = Specification(
        hp_nside=1,
        ra_range=(0, 10),
        dec_range=(-5, 5),
        nu=nu,
        sigma_beam_ch=np.full(len(nu), 0.01, dtype=float),
    )
    with pytest.raises(NotImplementedError):
        spec.get_beam_image()


def test_healpix_get_beam_window_ch_gaussian():
    nu = np.linspace(975e6, 1025e6, 5)
    lmax_expect = min(3 * 4 - 1, 8192)
    spec = Specification(
        hp_nside=4,
        ra_range=(40.0, 50.0),
        dec_range=(0.0, 5.0),
        nu=nu,
        sigma_beam_ch=dish_beam_sigma(13.5, nu),
    )
    bw = spec.get_beam_window_ch(cache=False)
    assert bw.shape == (len(nu), lmax_expect + 1)
    assert np.all(np.isfinite(bw))


def test_wcs_get_beam_window_ch_raises():
    sp = Specification(survey="meerklass_2021", band="L")
    sp.sigma_beam_ch = dish_beam_sigma(13.5, sp.nu)
    with pytest.raises(ValueError, match="healpix"):
        sp.get_beam_window_ch(cache=False)


def test_healpix_kat_beam_window_notimplemented():
    nu = np.linspace(1.15e9, 1.2e9, 3)
    spec = Specification(
        hp_nside=2,
        ra_range=(0.0, 10.0),
        dec_range=(-5.0, 5.0),
        nu=nu,
        beam_model="kat",
        sigma_beam_ch=np.full(len(nu), 1.0 / 60.0, dtype=float),
    )
    assert spec.beam_type == "anisotropic"
    with pytest.raises(NotImplementedError):
        spec.get_beam_window_ch(cache=False)


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
    sp = Specification(
        nu_min=-np.inf,
        nu_max=np.inf,
        num_pix_x=1,
        num_pix_y=1,
    )
    sp.map_file = test_fits
    sp.read_from_fits()
    assert np.isfinite(sp.nu_min) and np.isfinite(sp.nu_max)
    assert sp.nu_min == pytest.approx(float(np.min(sp.nu)))
    assert sp.nu_max == pytest.approx(float(np.max(sp.nu)))
    assert sp.dec_range[0] == pytest.approx(float(np.min(sp.dec_map)))
    assert sp.dec_range[1] == pytest.approx(float(np.max(sp.dec_map)))
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


def test_read_fits_auto_set_radecnu_bounds_off(test_fits):
    sp = Specification(
        nu_min=-np.inf,
        nu_max=np.inf,
        auto_set_radecnu_bounds=False,
    )
    sp.map_file = test_fits
    sp.read_from_fits()
    assert sp.nu_min == -np.inf and sp.nu_max == np.inf
    assert sp.dec_range == (-90, 90)
    assert sp.ra_range == (0, 360)
    sp.set_radecnu_bounds_from_map()
    assert sp.nu_min == pytest.approx(float(np.min(sp.nu)))
    assert sp.nu_max == pytest.approx(float(np.max(sp.nu)))
    # test manually override
    sp.ra_range = (350, 5)
    sp.set_radecnu_bounds_from_map()
    assert np.allclose(sp.ra_range, (350, 5))


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


def test_get_beam_image_returns_cached_duplicate_call():
    """Second ``get_beam_image(cache=True)`` reuses `_beam_image` when shape matches."""
    sp = Specification(survey="meerklass_2021", band="L")
    D_dish = 13.5
    sp.sigma_beam_ch = dish_beam_sigma(D_dish, sp.nu)
    _ = sp.beam_image
    assert sp._beam_image is not None
    b_cached = sp.get_beam_image(cache=True)
    assert b_cached is sp._beam_image


def test_convolve_data_wcs():
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

    with pytest.raises(ValueError, match="requires ``kernel``"):
        sp.convolve_data(kernel=None)


@pytest.mark.parametrize("beam_model", ["gaussian", "cos"])
def test_convolve_data_healpix(beam_model):
    sigma_beam_ch = 0.4
    spec = Specification(
        hp_nside=128,
        ra_range=(190, 230),
        dec_range=(-5, 15),
        sigma_beam_ch=sigma_beam_ch,
        beam_model=beam_model,
    )
    spec.data[1850] = 1.0
    data_conv, _ = spec.convolve_data(assign_to_self=False)
    # test renorm
    sum_test = data_conv.sum(axis=(0))[0]
    assert np.abs(sum_test - 1) < 1e-2
    if beam_model == "gaussian":
        pixel_sort = np.argsort(data_conv[..., 0])[::-1]
        ra1, dec1 = hp.pix2ang(spec.hp_nside, spec.pixel_id[pixel_sort[0]], lonlat=True)
        ra2, dec2 = hp.pix2ang(spec.hp_nside, spec.pixel_id[pixel_sort[1]], lonlat=True)
        ang_dist = get_ang_between_coord(
            np.atleast_1d(ra1),
            np.atleast_1d(dec1),
            np.atleast_1d(ra2),
            np.atleast_1d(dec2),
        )
        theo_beam = np.exp(-(ang_dist**2) / 2 / sigma_beam_ch**2)
        data_max2 = data_conv[pixel_sort[1], 0]
        data_max = data_conv[pixel_sort[0], 0]
        assert np.abs((data_max2 / data_max - theo_beam) / theo_beam) < 0.1


def test_beam_convolve_input_sigma():
    sigma_beam_ch = 0.4
    spec = Specification(
        hp_nside=128,
        ra_range=(190, 230),
        dec_range=(-5, 15),
        sigma_beam_ch=sigma_beam_ch,
    )
    beam_window_ch = spec.get_beam_window_ch(cache=False)[0]
    weighted_smoothing_healpix(
        spec.data,
        spec.w_HI,
        beam_window_ch,
        spec.hp_nside,
        spec.pixel_id,
        nside_out=64,
        pixel_id_out=np.array([0, 13], dtype=np.int64),
    )


def test_convolve_data_healpix_harmonic():
    """Harmonic smoothing path: ``convolve_data(None)``, no raster kernel."""
    nu = np.linspace(975e6, 1025e6, 6)
    geom = HealpixSkyMap(8, ra_range=(350.0, 355.0), dec_range=(-32.0, -28.0))
    sp = Specification(
        skymap=geom,
        nu=nu,
        sigma_beam_ch=dish_beam_sigma(13.5, nu),
        ra_range=(350.0, 355.0),
        dec_range=(-32.0, -28.0),
    )
    assert sp.pixel_id.size > 0
    mid = np.zeros_like(sp.data, dtype=float)
    mid[mid.shape[0] // 2, :] = 1.0
    sp.data = mid
    sp.w_HI = sp.W_HI.astype(float)

    before = mid.sum(axis=0)
    out_data, out_w = sp.convolve_data(None)
    assert out_data.shape == sp.data.shape == (sp.pixel_id.size, len(sp.nu))
    assert np.all(np.isfinite(out_data))
    assert np.all(np.isfinite(out_w))
    after = out_data.sum(axis=0)
    assert np.allclose(after, before, rtol=0.03)

    with pytest.raises(ValueError, match="kernel"):
        sp.convolve_data(np.ones((sp.pixel_id.size, len(sp.nu))))


def test_update_beam_type():
    sp = Specification(survey="meerklass_2021", band="L", beam_model="kat")
    assert sp.beam_type == "anisotropic"
    sp = Specification()
    assert sp.beam_type == "isotropic"
    with pytest.raises(ValueError):
        sp.beam_model = "something"


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


def test_init_skymap():
    skymap = HealpixSkyMap(128, pixel_id=np.array([0, 13], dtype=np.int64))
    pixel_id = np.array([0, 1])
    with pytest.raises(
        ValueError,
        match="healpix_pixel_id is invalid when passing skymap; set pixel_id on HealpixSkyMap instead.",
    ):
        Specification(skymap=skymap, healpix_pixel_id=pixel_id)
    with pytest.raises(
        ValueError,
        match="healpix_pixel_id requires hp_nside or pass skymap=HealpixSkyMap(...).",
    ):
        Specification(healpix_pixel_id=pixel_id)
    with pytest.raises(KeyError, match="num_pix_x is only defined for WCS sky maps."):
        sp = Specification(hp_nside=128, num_pix_x=100)
        sp.num_pix_x
    with pytest.raises(KeyError, match="num_pix_y is only defined for WCS sky maps."):
        sp = Specification(hp_nside=128, num_pix_y=100)
        sp.num_pix_y
    with pytest.raises(
        KeyError, match="hp_nside is only defined for HEALPix sky maps."
    ):
        sp = Specification(num_pix_x=100, num_pix_y=100)
        sp.hp_nside
    with pytest.raises(
        KeyError, match="pixel_id is only defined for HEALPix sky maps."
    ):
        sp = Specification(num_pix_x=100, num_pix_y=100)
        sp.pixel_id


def test_get_beam_window_ch():
    sp = Specification(hp_nside=32, ra_range=(40, 50), dec_range=(0, 5))
    assert sp.get_beam_window_ch() is None
    sp.sigma_beam_ch = 0.5
    bw = sp.get_beam_window_ch(lmax=100)
    sp.get_beam_window_ch(cache=True)
    previous_beam = sp.beam_window_ch
    # hack a wrong beam window
    sp._beam_window_ch = np.zeros_like(previous_beam)
    assert not np.allclose(sp.get_beam_window_ch(cache=True), previous_beam)
    sp = Specification(
        hp_nside=32, ra_range=(40, 50), dec_range=(0, 5), sigma_beam_ch=0.5
    )
    with pytest.raises(
        ValueError, match="data shape .* does not match weights shape .*"
    ):
        sp._convolve_data_healpix_harmonic(
            [0],
            [0, 1],
        )
    with pytest.raises(
        ValueError,
        match=r"HEALPix map cubes must have shape \(n_pix, n_ch\); got .*. Use self.data ordering for the LOS axis",
    ):
        sp._convolve_data_healpix_harmonic(
            [0, 1],
            [0, 1],
        )
    with pytest.raises(
        ValueError, match=r"data axis 0 .* must equal len\(self.pixel_id\) .*."
    ):
        sp._convolve_data_healpix_harmonic(
            [[0, 1]],
            [[0, 1]],
        )


def test_partial_beam_ch():
    sp = Specification(
        hp_nside=32, ra_range=(40, 50), dec_range=(0, 5), sigma_beam_ch=0.5
    )
    bw = sp.beam_window_ch[0]
    bw2 = sp.get_beam_window_ch(cache=False, ch_sel=[0])
    sp = Specification(band="L", survey="meerklass_2021", sigma_beam_ch=0.5)
    bw3 = sp.beam_image[:, :, 0]
    bw4 = sp.get_beam_image(cache=False, ch_sel=[0])[:, :, 0]
    assert np.allclose(bw3, bw4)
