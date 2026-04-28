import numpy as np
import pytest
from astropy.cosmology import Planck18, WMAP1
from meer21cm.util import *
from meer21cm.util import _ra_range_is_subset_of
import sys
from scipy.special import erf
from halomod import TracerHaloModel
from meer21cm import PowerSpectrum


def test_legendre_polynomial_with_factor():
    coeff = legendre_polynomial_with_factor(2)
    assert np.allclose(coeff, np.array([3 / 2, 0, -1 / 2]) * 5)
    poly = legendre_polynomial_with_factor(2, return_coeff=False)
    assert np.allclose(poly.c, np.array([3 / 2, 0, -1 / 2]) * 5)


def test_get_nd_slicer():
    slicer = get_nd_slicer(3)
    assert len(slicer) == 3
    assert slicer[0] == (slice(None), None, None)
    assert slicer[1] == (None, slice(None), None)
    assert slicer[2] == (None, None, slice(None))


def test_real_dtype_from_array_complex_branch():
    assert real_dtype_from_array(np.array([1 + 1j], dtype=np.complex64)) == np.float32
    assert real_dtype_from_array(np.array([1 + 1j], dtype=np.complex128)) == np.float64


def test_real_dtype_from_array_fallback_default_branch():
    # integer dtype should fall through to the final return(default)
    assert real_dtype_from_array(np.array([1, 2, 3], dtype=np.int32)) == np.dtype(
        np.float64
    )
    # object without a dtype attribute should also return default
    assert real_dtype_from_array([1, 2, 3], default=np.float32) == np.dtype(np.float32)


def test_dft_matrix():
    fft_mat = dft_matrix(10)
    assert np.allclose(fft_mat, np.fft.fft(np.eye(10)))
    assert np.allclose(
        dft_matrix(10, norm="forward"), np.fft.fft(np.eye(10), norm="forward")
    )
    rand = np.random.normal(size=10)
    assert np.allclose(np.fft.fft(rand), fft_mat @ rand)


def test_inv_dft_matrix():
    inv_fft_mat = inv_dft_matrix(10)
    assert np.allclose(inv_fft_mat, np.fft.ifft(np.eye(10)))
    assert np.allclose(
        inv_dft_matrix(10, norm="forward"), np.fft.ifft(np.eye(10), norm="forward")
    )
    rand = np.random.normal(size=10)
    assert np.allclose(np.fft.ifft(rand), inv_fft_mat @ rand)


def test_create_wcs_with_range():
    ra_range = [315, 80]
    dec_range = [-70, 5]
    w_test, num_pix_x, num_pix_y = create_wcs_with_range(
        ra_range,
        dec_range,
        buffer=[1.2, 1.4],
    )
    ra_xx, dec_yy = np.meshgrid(ra_range, dec_range)
    x_indx, y_indx = radec_to_indx(ra_xx, dec_yy, w_test)
    flag = (x_indx >= 0) * (x_indx < num_pix_x) * (y_indx >= 0) * (y_indx < num_pix_y)
    assert flag.mean() == 1


def test_create_wcs():
    wproj = create_wcs(0, 0, 21, 0.3)
    assert np.allclose(wproj.wcs.crpix, [10, 10])
    assert np.allclose(wproj.wcs.cdelt, [0.3, 0.3])
    assert np.allclose(wproj.wcs.crval, [0, 0])
    assert wproj.wcs.ctype[0] == "RA---ZEA"
    assert wproj.wcs.ctype[1] == "DEC--ZEA"


def test_angle_in_range():
    assert angle_in_range(-10, 0, 360)
    assert angle_in_range(350, 340, 10)
    assert angle_in_range(5, 0, 10)


def test_extreme_cases_ra_range_is_subset_of():
    assert not _ra_range_is_subset_of(0, 360 - 1e-10, 0, 350)


def test_ra_array_crosses_zero():
    assert ra_array_crosses_zero([355.0, 357.0, 2.0, 5.0])
    assert not ra_array_crosses_zero([350.0, 10.0])
    assert not ra_array_crosses_zero([10.0, 20.0, 30.0])


def test_tightest_ra_interval():
    lo, hi = tightest_ra_interval([355.0, 357.0, 2.0, 5.0])
    assert lo == 355.0 and hi == 5.0
    lo2, hi2 = tightest_ra_interval([10.0, 20.0, 30.0])
    assert lo2 == 10.0 and hi2 == 30.0
    assert tightest_ra_interval([42.0]) == (42.0, 42.0)
    with pytest.raises(ValueError):
        tightest_ra_interval([])


def test_tightest_ra_interval_full_sky_branch():
    # gaps sum to 360°; the wrap gap is >= 360 - 1e-12 only when all unique RAs lie in
    # an arc of length <= 1e-12° (complement almost the whole circle).
    base = 123.456789
    lo, hi = tightest_ra_interval([base, base + 5e-13])
    assert (lo, hi) == (0.0, 360.0)


def test_which_ra_range_is_tighter():
    assert which_ra_range_is_tighter((10.0, 30.0), (5.0, 40.0)) == -1
    assert which_ra_range_is_tighter((5.0, 40.0), (10.0, 30.0)) == 1
    assert which_ra_range_is_tighter((10.0, 30.0), (10.0, 30.0)) == 0
    assert which_ra_range_is_tighter((355.0, 5.0), (350.0, 10.0)) == -1
    assert which_ra_range_is_tighter((350.0, 10.0), (355.0, 5.0)) == 1
    assert which_ra_range_is_tighter((0.0, 360.0), (15.0, 20.0)) == 1
    assert which_ra_range_is_tighter((15.0, 20.0), (0.0, 360.0)) == -1
    with pytest.raises(ValueError):
        which_ra_range_is_tighter((10.0, 25.0), (20.0, 40.0))
    # Endpoints of each arc lie in the other, but neither arc (angle_in_range) nests
    with pytest.raises(ValueError):
        which_ra_range_is_tighter((190.0, 10.0), (0.0, 200.0))


def test_sample_map_from_highres():
    mock = PowerSpectrum(
        survey="meerklass_2021",
        band="L",
    )
    w = create_udres_wproj(mock.wproj, 3)
    mock2 = PowerSpectrum(
        nu=mock.nu,
        wproj=w,
        num_pix_x=mock.num_pix_x * 3,
        num_pix_y=mock.num_pix_y * 3,
    )
    map_hires = np.ones((mock.num_pix_x * 3, mock.num_pix_y * 3, mock.nu.size))
    map_lowres = sample_map_from_highres(
        map_hires,
        mock2.ra_map,
        mock2.dec_map,
        mock.wproj,
        mock.num_pix_x,
        mock.num_pix_y,
        average=True,
    )
    # get rid of nan
    map_lowres = map_lowres[map_lowres == map_lowres]
    assert np.allclose(map_lowres, np.ones_like(map_lowres))


def test_create_udres_wproj():
    mock = PowerSpectrum(
        survey="meerklass_2021",
        band="L",
    )
    w = create_udres_wproj(mock.wproj, 3)
    mock2 = PowerSpectrum(
        survey="meerklass_2021",
        band="L",
        wproj=w,
        num_pix_x=mock.num_pix_x * 3,
        num_pix_y=mock.num_pix_y * 3,
    )
    assert mock2.ra_map[0, 0] == mock.ra_map[0, 0]
    assert mock2.dec_map[0, 0] == mock.dec_map[0, 0]
    mock.W_HI = np.ones_like(mock.W_HI)
    mock2.W_HI = np.ones_like(mock2.W_HI)
    assert np.allclose(mock.survey_volume, mock2.survey_volume)


def test_super_sample_array():
    arr_in = np.random.normal(size=[100, 50, 20])
    super_factor = [3, 3, 3]
    arr_out = super_sample_array(arr_in, super_factor)
    assert np.allclose(arr_out[::3, ::3, ::3], arr_in)
    assert np.allclose(arr_out[1::3, 1::3, 1::3], arr_in)
    assert np.allclose(arr_out[2::3, 2::3, 2::3], arr_in)
    assert super_sample_array(None, super_factor) is None


def test_tagging():
    @tagging("test")
    def foo():
        pass

    assert foo.tags == ("test",)


def test_find_property_with_tags():
    class Foo:
        def __init__(
            self,
            xinit=1,
        ):
            self.x = xinit
            self.dependency_dict = find_property_with_tags(self)

        @property
        @tagging("test")
        def x(self):
            return self._x

        @x.setter
        def x(self, value):
            self._x = value

        @property
        def y(self):
            return self._y

        @y.setter
        def y(self, value):
            self._y = value

    foo = Foo(1)
    foo.x = 2
    assert foo.dependency_dict == {"x": ("test",)}


def test_center_to_edges():
    outarr = center_to_edges(np.linspace(0.5, 9.5, 10))
    assert np.allclose(outarr, np.linspace(0, 10, 11))


def test_find_ch_id():
    ch_id = find_ch_id(np.array([0.1, 0.6, 1.7, 3.5]), np.array([0, 1, 2]))
    assert np.allclose(ch_id, np.array([0, 1, 2, 3]))


def test_omega_hi_to_average_temp():
    # just for tests
    omega_hi = 1e-4
    z = 0
    HzoverH0 = (Planck18.H(z) / Planck18.H0).to("").value
    tbar_old = 0.18 * omega_hi * Planck18.h * (1 + z) ** 2 / HzoverH0
    tbar = omega_hi_to_average_temp(omega_hi, z=z, cosmo=Planck18)
    assert (np.abs(tbar - tbar_old) / tbar) < 1e-1
    # test another cosmology
    HzoverH0 = (WMAP1.H(z) / WMAP1.H0).to("").value
    tbar_old = 0.18 * omega_hi * WMAP1.h * (1 + z) ** 2 / HzoverH0
    tbar = omega_hi_to_average_temp(omega_hi, z=z, cosmo=WMAP1)
    assert (np.abs(tbar - tbar_old) / tbar) < 1e-1


def test_freq_redshift():
    assert freq_to_redshift(f_21) == 0.0
    assert freq_to_redshift(f_21 / 2) == 1.0
    assert redshift_to_freq(0.0) == f_21
    assert redshift_to_freq(1.0) == f_21 / 2


def test_get_ang_between_coord():
    ra1 = np.zeros(11)
    ra2 = np.array([0])
    dec1 = np.linspace(-30, -40, 11)
    dec2 = np.array([80])
    ang = get_ang_between_coord(ra1, dec1, ra2, dec2)
    assert np.allclose(ang.ravel(), dec2 - dec1)


def test_get_default_args():
    def test_func(x, arg1=1):
        return 1

    defaults = get_default_args(test_func)
    assert len(defaults) == 1
    for k, v in defaults.items():
        assert k == "arg1"
        assert v == 1


def test_get_wcs_coor(test_wproj, test_wcs):
    with pytest.raises(Exception) as e_info:
        get_wcs_coor(test_wcs, np.arange(10), np.arange(10))
    get_wcs_coor(test_wproj, np.arange(10), np.arange(10))


def test_pcaclean():
    test_arr = np.random.normal(size=(10))
    with pytest.raises(Exception) as e_info:
        pca_clean(test_arr, 1, return_analysis=True)
    test_arr = np.random.normal(size=(200, 200, 10))
    C, eignumb, eigenval, V = pca_clean(test_arr, 1, return_analysis=True)
    assert (np.abs(C - np.eye((test_arr.shape[-1]))) < 0.1).mean() == 1
    test_arr = np.random.normal(size=(10, 200, 200))
    # test renorm
    C, eignumb, eigenval, V = pca_clean(
        test_arr,
        1,
        weights=2 * np.ones_like(test_arr),
        return_analysis=True,
        los_axis=0,
        mean_center=True,
    )
    assert C.shape == (10, 10)
    assert V.shape == (10, 10)
    assert (np.abs(C - np.eye((test_arr.shape[0]))) < 0.1).mean() == 1
    assert np.allclose(eignumb, np.linspace(1, 10, 10))
    assert np.std(eigenval) < 0.1
    res_arr = pca_clean(
        test_arr,
        1,
        return_analysis=False,
        los_axis=0,
        weights=2 * np.ones_like(test_arr),
        mean_center=True,
    )
    assert res_arr.shape == test_arr.shape
    assert np.abs((res_arr).mean()) < 1e-3
    res_arr, A_mat = pca_clean(
        test_arr,
        1,
        return_analysis=False,
        los_axis=0,
        mean_center=True,
        return_A=True,
    )
    res_arr = pca_clean(
        test_arr,
        1,
        return_analysis=False,
        los_axis=0,
        weights=2 * np.ones_like(test_arr),
        mean_center=True,
        mean_center_weights=np.ones_like(test_arr),
    )
    assert res_arr.shape == test_arr.shape
    assert np.abs((res_arr).mean()) < 1e-3
    test_arr[:, :, :20] = np.nan
    test_res, test_A = pca_clean(test_arr, 1, return_A=True, ignore_nan=True)
    # first 20 channels are nan
    assert np.isnan(test_A).sum() == 20
    assert np.allclose(test_res[:, :, :20], 0.0)
    assert np.abs((test_res[:, :, 20:]).mean()) < 5e-3
    # after 1 mode std barely changed
    assert np.abs(test_res[:, :, 20:].std() - 1) < 3e-2


def test_radec_to_indx(test_wproj):
    indx_i, indx_j = radec_to_indx(0, -30, test_wproj, to_int=True)
    indx_1, indx_2 = radec_to_indx(0, -30, test_wproj, to_int=False)
    assert np.round(indx_1) == indx_i
    assert np.round(indx_2) == indx_j


def test_rebin_spectrum():
    test_spectrum = np.zeros(503)
    test_spectrum[503 // 2] = 1.0
    test_rebin = rebin_spectrum(test_spectrum, rebin_width=3)
    assert test_rebin.sum() == 1 / 3
    assert test_rebin.size == 503 // 3
    test_rebin = rebin_spectrum(test_spectrum, rebin_width=3, mode="sum")
    assert test_rebin.sum() == 1
    test_rebin = rebin_spectrum(test_spectrum, rebin_width=13, mode="sum")
    assert test_rebin.sum() == 1


def test_hod_obuljen18():
    mass_1 = np.log10(hod_obuljen18(11.27))
    mass_2 = np.log10(10**9.52 * np.exp(-1) / Planck18.h)
    assert np.allclose(mass_1, mass_2)


def test_check_unit_equiv():
    assert check_unit_equiv(units.m, units.cm)
    assert not check_unit_equiv(units.m, units.K)


def test_jy_to_kelvin():
    omega = np.random.uniform(0.01, 1)
    freq = np.random.uniform(0.01, 1) * 1e9
    test = jy_to_kelvin(1, omega, freq)
    test2 = (
        (
            1
            * units.Jy
            / omega
            * (constants.c / freq / units.Hz) ** 2
            / 2
            / constants.k_B
        )
        .to("K")
        .value
    )
    assert np.allclose(test, test2)


def test_busy_function_simple():
    xarr = np.linspace(-10, 10, 101)
    assert np.allclose(busy_function_simple(xarr, 2, 1, 0, 0), erf(-(xarr**2)) + 1)


def test_find_indx_for_subarr():
    arr1 = np.arange(100)
    arr2 = np.arange(1000)
    assert np.allclose(find_indx_for_subarr(arr1, arr2), np.arange(100))
    arr2[0] = 1
    with pytest.raises(AssertionError):
        find_indx_for_subarr(arr1, arr2)


def test_himf():
    h_70 = Planck18.h / 0.7
    mmin = 6
    himf_pars = himf_pars_jones18(h_70)
    nhi, omegahi, psn = cal_himf(himf_pars, mmin, Planck18)
    assert np.allclose(nhi, 0.13980687586146462)
    assert np.allclose(omegahi, 0.00036495842278914405)
    assert np.allclose(psn, 150.93927719814297)
    minput = np.linspace(mmin, 11, 500)
    nhi_cumu = cumu_nhi_from_himf(minput, mmin, himf_pars)
    assert np.allclose(nhi_cumu[0], 0.0)
    assert np.allclose(nhi_cumu[-1], nhi)


def uniform_pdf(x):
    return np.ones_like(x)


def uniform_cdf(x):
    return x


def test_sample_from_dist():
    test_sample = sample_from_dist(uniform_pdf, 0, 1, size=1000000, cdf=False, seed=42)
    count, _ = np.histogram(test_sample, bins=10)
    assert (
        np.abs((count - len(test_sample) / 10) / (len(test_sample) / 10)) > 0.01
    ).sum() == 0
    test_sample = sample_from_dist(uniform_cdf, 0, 1, size=1000000, cdf=True, seed=42)
    count, _ = np.histogram(test_sample, bins=10)
    assert (
        np.abs((count - len(test_sample) / 10) / (len(test_sample) / 10)) > 0.01
    ).sum() == 0


def test_tully_fisher():
    assert np.allclose(tully_fisher(np.ones(100), 0, 2), 1e2)
    assert np.allclose(tully_fisher(np.ones(100), 1, 2, inv=True), 1e-2)


def test_random_sample_indx():
    tot_num = 1000
    sub_num = 200
    sub_indx = np.sort(random_sample_indx(tot_num, sub_num))
    assert np.allclose(sub_indx, np.unique(sub_indx))


def test_Obuljen18():
    hm = TracerHaloModel(hod_model=Obuljen18)
    assert np.allclose(
        hod_obuljen18(10, output_has_h=True), hm.hod.total_occupation(1e10)
    )
    assert hm.hod.sigma_satellite(1e10) == 0


def test_find_id():
    """test `find_id` function"""
    file_arr = np.random.randint(0, 9999999999, 10)
    file_arr = file_arr.astype("str")
    file_arr = np.char.zfill(file_arr, 10)
    assert (vfind_id(file_arr) != file_arr).sum() == 0
    file_arr = np.random.randint(0, 9999999999, 2)
    file_arr = file_arr.astype("str")
    file_arr = np.char.zfill(file_arr, 10)
    file_test = file_arr[0] + "/" + file_arr[1]
    with pytest.raises(ValueError):
        vfind_id(file_test)


def test_convert_hpmap_in_jy_to_temp():
    # just test invoke
    hpmap = np.zeros(hp.nside2npix(128))
    convert_hpmap_in_jy_to_temp(hpmap, 1e8)
