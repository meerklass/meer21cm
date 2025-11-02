import numpy as np
import pytest

from meer21cm.mock import (
    MockSimulation,
    HIGalaxySimulation,
    hi_mass_to_flux_profile,
    generate_gaussian_field,
    generate_colored_noise,
)
from meer21cm import Specification
from meer21cm.util import create_udres_wproj
from meer21cm.util import center_to_edges
from meer21cm.util import radec_to_indx, find_ch_id, redshift_to_freq
from scipy.signal import windows
from scipy.interpolate import interp1d


@pytest.mark.parametrize("parallel_plane", [True, False])
def test_rsd_from_field(parallel_plane):
    mock = MockSimulation(
        survey="meerklass_2021",
        band="L",
        kaiser_rsd=True,
        parallel_plane=parallel_plane,
        density="lognormal",
        tracer_bias_2=1.5,
        num_discrete_source=1e7,
        rsd_from_field=True,
        kmax=20,
    )
    mock.field_2 = mock.mock_tracer_field_2
    ratio1 = mock.auto_power_3d_2 / mock.auto_power_tracer_2_model
    assert np.abs(ratio1.mean() - 1) < 2e-1
    tracer_positions = mock.mock_tracer_position_in_box
    grid_bins = [center_to_edges(mock.x_vec[i]) for i in range(3)]
    count, _ = np.histogramdd(tracer_positions, bins=grid_bins)
    mock.field_2 = count.astype(np.float32)
    mock.mean_center_2 = True
    mock.unitless_2 = True
    shot_noise = np.prod(mock.box_len) / count.sum()
    ratio2 = (mock.auto_power_3d_2 - shot_noise) / mock.auto_power_tracer_2_model
    # curved sky the galaxy mock at small scales is not accurate
    assert np.abs(ratio2.mean() - 1) < 2e-1
    # trigger the warning
    if not parallel_plane:
        with pytest.warns(UserWarning):
            mock = MockSimulation(
                survey="meerklass_2021",
                band="L",
                kaiser_rsd=True,
                parallel_plane=parallel_plane,
                rsd_from_field=False,
            )


@pytest.mark.parametrize("density", [("lognormal"), ("gaussian"), ("test")])
def test_matter_mock(test_W, density):
    # default is Planck18, so use WMAP1 to test
    # if cosmo is properly updated throughout
    k1dedges = np.geomspace(0.05, 1.5, 20)

    mock = MockSimulation(
        survey="meerklass_2021",
        band="L",
        density=density,
        true_cosmology="WMAP1",
        k1dbins=k1dedges,
        model_k_from_field=True,
        upgrade_sampling_from_gridding=True,
        kaiser_rsd=False,
    )
    mock.map_has_sampling = (test_W * np.ones_like(mock.nu)[None, None, :]) > 0
    mock.get_enclosing_box()
    # underlying code has been tested in grid
    # simply test invoking
    mock.box_origin
    mock.box_len
    mock.box_resol
    mock.box_ndim
    mock.rot_mat_sky_to_box
    mock.pix_coor_in_cartesian
    mock.pix_coor_in_box
    # test input and output power consistency
    mock.k1dbins = k1dedges
    if density != "test":
        mock.field_1 = mock.mock_matter_field
        pfield_i, keff, nmodes = mock.get_1d_power(
            "auto_power_3d_1",
        )

        pmatter3d = mock.matter_power_spectrum_fnc(mock.kmode)
        pm1d, _, _ = mock.get_1d_power(
            pmatter3d,
        )
        avg_deviation = np.sqrt(
            ((np.abs((pfield_i - pm1d) / pm1d)) ** 2 * nmodes).sum() / nmodes.sum()
        )
        assert avg_deviation < 2e-1
        # test RSD
        mock.kaiser_rsd = True
        # mock.get_mock_matter_field()
        mock.field_1 = mock.mock_matter_field
        pfield_i_rsd, keff, nmodes = mock.get_1d_power(
            "auto_power_3d_1",
        )
        pm1d_rsd, _, _ = mock.get_1d_power(mock.auto_power_matter_model)
        avg_deviation = np.sqrt(
            ((np.abs((pfield_i_rsd - pm1d_rsd) / pm1d_rsd)) ** 2 * nmodes).sum()
            / nmodes.sum()
        )
        assert avg_deviation < 2e-1
    else:
        with pytest.raises(ValueError):
            mock.field_1 = mock.mock_matter_field


@pytest.mark.parametrize(
    "tracer_i, kaiser_rsd, parallel_plane",
    [(1, True, True), (2, True, False), (1, False, False)],
)
def test_tracer_mock(tracer_i, kaiser_rsd, parallel_plane):
    mock = MockSimulation(
        survey="meerklass_2021",
        band="L",
        kaiser_rsd=kaiser_rsd,
        parallel_plane=parallel_plane,
        tracer_bias_1=1.5,
        tracer_bias_2=1.5,
        rsd_from_field=True,
    )
    mock.ones_func = 1
    setattr(mock, f"mean_amp_{tracer_i}", "ones_func")
    mock.field_2 = getattr(mock, f"mock_tracer_field_{tracer_i}")
    ratio = mock.auto_power_3d_2 / mock.auto_power_tracer_2_model
    # mumode will be slightly smaller if not using parallel plane
    assert np.abs(ratio.mean() - 1) < 2e-1


def test_tracer_position():
    raminGAMA, ramaxGAMA = 339, 351
    decminGAMA, decmaxGAMA = -35, -30
    ra_range = (raminGAMA, ramaxGAMA)
    dec_range = (decminGAMA, decmaxGAMA)
    with pytest.raises(ValueError):
        mock = MockSimulation(
            survey="meerklass_2021",
            band="L",
            ra_range=ra_range,
            dec_range=dec_range,
            kaiser_rsd=True,
            seed=42,
            discrete_base_field="3",
        )
    # now do a proper mock based on tracer 2
    mock = MockSimulation(
        survey="meerklass_2021",
        band="L",
        ra_range=ra_range,
        dec_range=dec_range,
        kaiser_rsd=True,
        discrete_base_field=2,
    )
    mock.data = np.ones(mock.W_HI.shape)
    mock.w_HI = np.ones(mock.W_HI.shape)
    mock.counts = np.ones(mock.W_HI.shape)
    mock.trim_map_to_range()
    mock.downres_factor_radial = 1 / 2.0
    mock.downres_factor_transverse = 1 / 2.0
    mock.get_enclosing_box()
    mock.tracer_bias_2 = 1.9
    mock.num_discrete_source = 10000
    mock.propagate_mock_tracer_to_gal_cat()
    # ensure that galaxies are there
    assert len(mock.ra_mock_tracer) > mock.num_discrete_source
    assert len(mock.dec_mock_tracer) > mock.num_discrete_source
    assert len(mock.z_mock_tracer) > mock.num_discrete_source
    assert np.abs((mock.mock_inside_range).sum() - mock.num_discrete_source) < 2000
    assert np.abs(len(mock.ra_gal) - mock.num_discrete_source) < 2000
    # test of power spectrum is performed in pipeline tests


def test_hi_mass_to_flux():
    raminMK, ramaxMK = 334, 357
    decminMK, decmaxMK = -35, -26.5
    ra_range_MK = (raminMK, ramaxMK)
    dec_range_MK = (decminMK, decmaxMK)
    num_g = 10000
    mock = MockSimulation(
        survey="meerklass_2021",
        band="L",
        ra_range=ra_range_MK,
        dec_range=dec_range_MK,
        tracer_bias_1=1.5,
        tracer_bias_2=1.9,
        num_discrete_source=num_g,
    )
    mock.propagate_mock_tracer_to_gal_cat()
    # some random galaxies
    himass_g = np.random.uniform(9, 11, mock.z_mock_tracer.size)
    hifluxd_ch = hi_mass_to_flux_profile(
        himass_g,
        mock.z_mock_tracer,
        mock.nu,
        cosmo=mock.astropy_cosmo_true,
        seed=mock.seed,
    )
    # approximate from 1705.04210
    approx_mass = (
        hifluxd_ch.sum(0)
        * mock.freq_resol
        * mock.astropy_cosmo_true.luminosity_distance(mock.z_mock_tracer).value ** 2
        * 49.7
    )
    ratio = approx_mass / 10**himass_g
    # scaling should be exact
    assert np.allclose(ratio.std(), 0)
    # mean is about 1
    assert np.abs(1 - ratio.mean()) < 5e-2
    # now with velocity
    hifluxd_ch = hi_mass_to_flux_profile(
        himass_g,
        mock.z_mock_tracer,
        mock.nu,
        cosmo=mock.astropy_cosmo_true,
        seed=mock.seed,
        tf_slope=3.66,
        tf_zero=1.6,
        no_vel=False,
    )
    # approximate from 1705.04210
    approx_mass = (
        hifluxd_ch.sum(0)
        * mock.freq_resol
        * mock.astropy_cosmo_true.luminosity_distance(mock.z_mock_tracer).value ** 2
        * 49.7
    )
    ratio = approx_mass / 10**himass_g
    # scaling should be exact
    assert np.allclose(ratio.std(), 0)
    # mean is about 1
    assert np.abs(1 - ratio.mean()) < 5e-2
    # with enough galaxy should be symmetric
    average_profile = hifluxd_ch.mean(-1)
    # mid point should be peak
    assert np.argmax(average_profile) == average_profile.size // 2
    # symmetric
    assert np.max(np.abs(average_profile[::-1] - average_profile)) < 3e-5


def test_mock_hi_profile():
    raminMK, ramaxMK = 334, 357
    decminMK, decmaxMK = -35, -26.5
    ra_range_MK = (raminMK, ramaxMK)
    dec_range_MK = (decminMK, decmaxMK)
    num_g = 10000
    hisim = HIGalaxySimulation(
        survey="meerklass_2021",
        band="L",
        ra_range=ra_range_MK,
        dec_range=dec_range_MK,
        tracer_bias_1=1.5,
        # tracer_bias_2=1.9,
        num_discrete_source=num_g,
        num_ch_ext_on_each_side=3,
        flat_sky=True,
    )
    # test initialization
    assert hisim.tracer_bias_2 == 1.0
    hisim.tracer_bias_2 = 1.9
    hifluxd_ch = hisim.hi_profile_mock_tracer
    approx_mass = (
        hifluxd_ch.sum(0)
        * hisim.freq_resol
        * hisim.astropy_cosmo_true.luminosity_distance(hisim.z_mock_tracer).value ** 2
        * 49.7
    )
    ratio = approx_mass / 10**hisim.hi_mass_mock_tracer
    # scaling should be exact
    assert np.allclose(ratio.std(), 0)
    # mean is about 1
    assert np.abs(1 - ratio.mean()) < 5e-2
    # add velocity
    hisim.tf_slope = 3.66
    hisim.tf_zero = 1.6
    hisim.no_vel = False
    # test cache
    assert hisim._hi_profile_mock_tracer is None
    assert hisim._halo_mass_mock_tracer is not None
    assert hisim._hi_mass_mock_tracer is not None
    hisim.hi_mass_from = "hod"
    assert hisim._hi_mass_mock_tracer is None
    assert hisim._halo_mass_mock_tracer is not None
    hifluxd_ch = hisim.hi_profile_mock_tracer
    assert hifluxd_ch.shape[0] > 1
    approx_mass = (
        hifluxd_ch.sum(0)
        * hisim.freq_resol
        * hisim.astropy_cosmo_true.luminosity_distance(hisim.z_mock_tracer).value ** 2
        * 49.7
    )
    # scaling should be exact
    assert np.allclose(ratio.std(), 0)
    # mean is about 1
    assert np.abs(1 - ratio.mean()) < 5e-2
    average_profile = hifluxd_ch.mean(-1)
    # mid point should be peak
    assert np.argmax(average_profile) == average_profile.size // 2
    # symmetric
    assert np.max(np.abs(average_profile[::-1] - average_profile)) < 5e-5


@pytest.mark.parametrize("highres", [(None), (3)])
def test_project_hi_profile(highres):
    raminMK, ramaxMK = 334, 357
    decminMK, decmaxMK = -35, -26.5
    ra_range_MK = (raminMK, ramaxMK)
    dec_range_MK = (decminMK, decmaxMK)
    # tests
    hisim = HIGalaxySimulation(
        survey="meerklass_2021",
        band="L",
        # ra_range=ra_range_MK,
        # dec_range=dec_range_MK,
        tracer_bias_1=1.5,
        tracer_bias_2=1.9,
        num_discrete_source=5,
        downres_factor_radial=1 / 3,
        downres_factor_transverse=1 / 3,
        kmax=20,
        nonlinear="both",
        tf_slope=3.66,
        tf_zero=1.6,
        no_vel=False,
        highres_sim=highres,
        num_ch_ext_on_each_side=3,
        flat_sky=True,
    )
    if highres is not None:
        wproj_hires = create_udres_wproj(hisim.wproj, highres)
    else:
        wproj_hires = hisim.wproj
    hifluxd_ch = hisim.hi_profile_mock_tracer
    hi_map_in_jy = hisim.propagate_hi_profile_to_map(return_highres=False, beam=True)
    # extremely rarely, the galaxies are exactly at the edge of the grid so highres
    # down to lowres and directly from lowres will have a misplacement of one pixel
    # this is to avoid that
    if highres is None:
        indx_0, indx_1 = radec_to_indx(
            hisim.ra_mock_tracer, hisim.dec_mock_tracer, hisim.wproj
        )
    else:
        sp_hires = Specification(
            wproj=wproj_hires,
            num_pix_x=hisim.num_pix_x * highres + 100,
            num_pix_y=hisim.num_pix_y * highres + 100,
        )
        indx_test = radec_to_indx(
            hisim.ra_mock_tracer,
            hisim.dec_mock_tracer,
            wproj_hires,
        )
        ratest, dectest = sp_hires.ra_map[indx_test], sp_hires.dec_map[indx_test]
        indx_0, indx_1 = radec_to_indx(ratest, dectest, hisim.wproj)
    indx_z = find_ch_id(redshift_to_freq(hisim.z_mock_tracer), hisim.nu)
    num_ch_vel = hifluxd_ch.shape[0] // 2
    sel = (
        ((indx_z - num_ch_vel) >= 0)
        * ((indx_z + num_ch_vel + 1) < hisim.nu.size)
        * (indx_0 >= 0)
        * (indx_0 < hi_map_in_jy.shape[0])
        * (indx_1 >= 0)
        * (indx_1 < hi_map_in_jy.shape[1])
    )
    if sel.sum() == 0:
        return 1.0
    profile_in_map = hi_map_in_jy[
        indx_0[sel],
        indx_1[sel],
    ]
    for i in range(len(profile_in_map)):
        test1 = profile_in_map[i][
            indx_z[sel][i] - num_ch_vel : indx_z[sel][i] + num_ch_vel + 1
        ]
        test2 = hifluxd_ch[:, sel][:, i]
        assert np.allclose(test1, test2)
    # add in an extremely small beam just to trigger the tests
    hisim.sigma_beam_ch = np.zeros(hisim.nu.size) + 1e-5
    hi_map_in_jy = hisim.propagate_hi_profile_to_map(return_highres=False, beam=True)
    profile_in_map = hi_map_in_jy[
        indx_0[sel],
        indx_1[sel],
    ]
    for i in range(len(profile_in_map)):
        if sel[i]:
            test1 = profile_in_map[i][
                indx_z[sel][i] - num_ch_vel : indx_z[sel][i] + num_ch_vel + 1
            ]
            test2 = hifluxd_ch[:, sel][:, i]
            assert np.allclose(test1, test2)


def test_generate_colored_noise():
    rand_arr = [generate_colored_noise([100], [100], np.ones(51)) for i in range(1000)]
    rand_arr = np.array(rand_arr)
    assert np.allclose(rand_arr.mean(), 0.0)
    assert np.abs(rand_arr.std() - 1.0) < 0.1


def test_flat_sky():
    mock = MockSimulation(
        survey="meerklass_2021",
        band="L",
        highres_sim=None,
        num_discrete_source=1000000,
        tracer_bias_2=1.0,
        kmax=10.0,
        flat_sky=True,
        mean_amp_1="average_hi_temp",
    )
    mock.data = mock.propagate_mock_field_to_data(mock.mock_tracer_field_1)
    mock.grid_data_to_field()
    mock.weights_1 = np.ones_like(mock.data)
    mock.include_sky_sampling = [False, False]
    mock.compensate = False
    ratio = mock.auto_power_3d_1 / mock.auto_power_tracer_1_model
    assert np.abs(ratio.mean() - 1) < 2e-1
    mock.propagate_mock_tracer_to_gal_cat()
    mock.grid_gal_to_field()
    mock.weights_2 = np.ones_like(mock.field_2)
    mock.compensate = False
    shot_noise = np.prod(mock.box_len) / mock.field_2.sum()
    ratio = (mock.auto_power_3d_2 - shot_noise) / mock.auto_power_tracer_2_model
    assert np.abs(ratio.mean() - 1) < 2e-1


def test_dndz():
    mock = MockSimulation(
        survey="meerklass_2021",
        band="L",
    )
    dndz = interp1d(
        np.linspace(mock.z_ch.min(), mock.z_ch.max(), 100),
        windows.blackmanharris(100),
        bounds_error=False,
        fill_value=0,
    )
    raminMK, ramaxMK = 334, 357
    decminMK, decmaxMK = -35, -26.5
    ra_range = (raminMK, ramaxMK)
    dec_range = (decminMK, decmaxMK)
    # all the other settings follow the default, which corresponds to the MeerKLASS L-band deep-field survey
    mock = MockSimulation(
        survey="meerklass_2021",
        band="L",
        flat_sky=True,
        num_discrete_source=1e5,
        tracer_bias_2=1.0,
        ra_range=ra_range,
        dec_range=dec_range,
        discrete_source_dndz=dndz,
        seed=12345,
    )
    # generate the mock tracer
    mock.propagate_mock_tracer_to_gal_cat()
    # grid to galaxy number count
    gal_count, _, _ = mock.grid_gal_to_field()
    kernel = gal_count.sum(axis=(0, 1))
    kernel /= kernel.max()
    input_dndz = dndz(mock.z_ch)
    input_dndz /= input_dndz.max()
    assert np.abs(input_dndz - kernel).mean() < 1e-1
