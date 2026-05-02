import numpy as np
from meer21cm import PowerSpectrum, MockSimulation
from meer21cm.telescope import dish_beam_sigma
import pytest


def test_gaussian_field_map_grid():
    """
    Test generating a Gaussian random field,
    grid it onto the sky map,
    resample it to a coarser grid as if it was data,
    and the output power still matches the input.
    """
    raminGAMA, ramaxGAMA = 339, 351
    decminGAMA, decmaxGAMA = -35, -30
    ps = PowerSpectrum(
        survey="meerklass_2021",
        band="L",
        ra_range=(raminGAMA, ramaxGAMA),
        dec_range=(decminGAMA, decmaxGAMA),
        kaiser_rsd=False,
    )
    ps.data = np.ones(ps.W_HI.shape)
    ps.w_HI = np.ones(ps.W_HI.shape)
    ps.counts = np.ones(ps.W_HI.shape)
    # ps.trim_map_to_range()
    ps.downres_factor_radial = 1 / 2.0
    ps.downres_factor_transverse = 1 / 2.0
    ps.get_enclosing_box()
    pos_value = np.random.normal(size=ps.box_ndim)
    k1dedges = np.geomspace(0.05, 1.5, 20)
    ps.k1dbins = k1dedges
    # ps.propagate_field_k_to_model()
    ps.field_1 = pos_value
    pfield, keff, nmodes = ps.get_1d_power(
        "auto_power_3d_1",
    )
    ps_mod = np.prod(ps.box_resol)
    avg_deviation = np.sqrt(
        ((np.abs((pfield - ps_mod) / ps_mod)) ** 2 * nmodes).sum() / nmodes.sum()
    )
    assert avg_deviation < 5e-2
    # grid the field to sky map
    map_bin, _ = ps.grid_field_to_sky_map(pos_value, average=True)
    # pass it to the object
    ps.data = map_bin
    # regrid the sky map to a regular grid field
    ps.downres_factor_radial = 2.0
    ps.downres_factor_transverse = 1.5
    # ps.get_enclosing_box()
    ps.compensate = False
    hi_map_rg, hi_weights_rg, pix_counts_hi_rg = ps.grid_data_to_field()
    taper_hi = ps.taper_func(ps.box_ndim[-1])
    weights_hi = hi_weights_rg * taper_hi[None, None, :]
    ps.field_1 = hi_map_rg
    ps.weights_1 = weights_hi
    # ps.propagate_field_k_to_model()
    ps.sampling_resol = [
        ps.pix_resol_in_mpc,
        ps.pix_resol_in_mpc,
        ps.los_resol_in_mpc,
    ]
    pmap_1, keff, nmodes = ps.get_1d_power(
        "auto_power_3d_1",
    )
    avg_deviation = np.sqrt(
        ((np.abs((pmap_1 - ps_mod) / ps_mod)) ** 2 * nmodes).sum() / nmodes.sum()
    )
    assert avg_deviation < 1.5e-1


def test_poisson_field_map_grid():
    """
    Test generating a Poisson galaxy sample on the field-level,
    grid it onto the sky map,
    resample it to a coarser grid as if it was data,
    and the output power still matches the input.
    """
    raminGAMA, ramaxGAMA = 339, 351
    decminGAMA, decmaxGAMA = -35, -30
    ps = PowerSpectrum(
        survey="meerklass_2021",
        band="L",
        ra_range=(raminGAMA, ramaxGAMA),
        dec_range=(decminGAMA, decmaxGAMA),
        kaiser_rsd=False,
    )
    ps.data = np.ones(ps.W_HI.shape)
    ps.w_HI = np.ones(ps.W_HI.shape)
    ps.counts = np.ones(ps.W_HI.shape)
    ps.downres_factor_radial = 1 / 2.0
    ps.downres_factor_transverse = 1 / 2.0
    ps.get_enclosing_box()
    pos_value = np.zeros(ps.box_ndim).ravel()
    k1dedges = np.geomspace(0.05, 1.5, 20)
    ps.k1dbins = k1dedges
    num_g = 10000
    gal_pix_indx = np.random.choice(
        np.arange(pos_value.size), size=num_g, replace=False
    )
    pos_value[gal_pix_indx] += 1
    pos_value = pos_value.reshape(ps.box_ndim)
    ps.field_1 = pos_value
    ps.mean_center_1 = True
    ps.unitless_1 = True
    pfield, keff, nmodes = ps.get_1d_power(
        "auto_power_3d_1",
    )
    psn = np.prod(ps.box_len) / num_g
    avg_deviation = np.sqrt(
        ((np.abs((pfield - psn) / psn)) ** 2 * nmodes).sum() / nmodes.sum()
    )
    assert avg_deviation < 5e-2
    map_bin, _ = ps.grid_field_to_sky_map(pos_value, average=False)
    ps.data = map_bin
    ps.downres_factor_radial = 2.0
    ps.downres_factor_transverse = 1.5
    ps.compensate = False
    hi_map_rg, hi_weights_rg, pix_counts_hi_rg = ps.grid_data_to_field()
    # galaxy counts are total not average
    hi_map_rg *= hi_weights_rg
    taper_hi = ps.taper_func(ps.box_ndim[-1])
    weights_hi = hi_weights_rg * taper_hi[None, None, :]
    ps.field_1 = hi_map_rg
    ps.weights_1 = weights_hi
    ps.unitless_1 = True
    ps.sampling_resol = [
        ps.pix_resol_in_mpc,
        ps.pix_resol_in_mpc,
        ps.los_resol_in_mpc,
    ]
    pmap_1, keff, nmodes = ps.get_1d_power(
        "auto_power_3d_1",
    )
    avg_deviation = np.sqrt(
        ((np.abs((pmap_1 - psn) / psn)) ** 2 * nmodes).sum() / nmodes.sum()
    )
    # maybe this accuracy is too low?
    assert avg_deviation < 2e-1


# num_p hasn't worked yet
# @pytest.mark.parametrize("num_p", [(1),])
@pytest.mark.parametrize("beam", [True, False])
def test_mock_field_map_grid(beam):
    """
    Generate a mock HI temp field, project it to sky map,
    grid it onto regular grids, and test input/output matching.
    """
    raminMK, ramaxMK = 334, 357
    decminMK, decmaxMK = -35, -26.5
    ra_range = (raminMK, ramaxMK)
    dec_range = (decminMK, decmaxMK)
    downres_sim_r = 1 / 1.0
    downres_sim_t = 1 / 1.0
    k1dmin = 0.05
    k1dmax = 0.4
    k1dnumbin = 10
    k1dedges = np.geomspace(k1dmin, k1dmax, k1dnumbin + 1)
    mock = MockSimulation(
        survey="meerklass_2021",
        band="L",
        ra_range=ra_range,
        dec_range=dec_range,
        mock_amp_1="average_hi_temp",
        mean_amp_1="average_hi_temp",
        mock_amp_2=1.0,
        tracer_bias_1=1.5,
        downres_factor_radial=downres_sim_r,
        downres_factor_transverse=downres_sim_t,
        k1dbins=k1dedges,
        compensate=False,
        # make sure kmax is higher than your simulation resolution
        kmax=10.0,
        box_buffkick=40,
    )
    if beam:
        D_dish = 13.5
        mock.sigma_beam_ch = dish_beam_sigma(D_dish, mock.nu)
    mock.data = mock.propagate_mock_field_to_data(mock.mock_tracer_field_1)
    mock.grid_data_to_field(flat_sky=True)
    ratio = mock.auto_power_3d_1 / mock.auto_power_tracer_1_model
    if beam:
        k_sel = mock.beam_attenuation() ** 2 > 0.1
    else:
        k_sel = np.ones_like(mock.k_mode)
    p1d, k1d, _ = mock.get_1d_power(
        ratio,
        k1dweights=k_sel.astype(float),
    )
    assert np.abs(np.median(p1d) - 1) < 0.5
    assert p1d.std() < 0.3


def test_mock_tracer_grid():
    """
    Generate a mock galaxy caralogue,
    grid it onto regular grids, and test input/output matching.
    """
    raminGAMA, ramaxGAMA = 339, 351
    decminGAMA, decmaxGAMA = -35, -30
    ra_range = (raminGAMA, ramaxGAMA)
    dec_range = (decminGAMA, decmaxGAMA)
    k1dedges = np.geomspace(0.05, 1.5, 20)
    pmap_1d = []
    pmod_1d = []
    # run 10 realizations
    for i in range(10):
        mock = MockSimulation(
            survey="meerklass_2021",
            band="L",
            ra_range=ra_range,
            dec_range=dec_range,
            kaiser_rsd=True,
            discrete_base_field=2,
            k1dbins=k1dedges,
            target_relative_to_num_g=2.0,
        )
        mock.data = np.ones(mock.W_HI.shape)
        mock.w_HI = np.ones(mock.W_HI.shape)
        mock.counts = np.ones(mock.W_HI.shape)
        mock.downres_factor_radial = 1 / 2.0
        mock.downres_factor_transverse = 1 / 2.0
        mock.get_enclosing_box()
        mock.tracer_bias_2 = 1.9
        mock.num_discrete_source = 2700
        # galaxy catalogue
        mock.propagate_mock_tracer_to_gal_cat()
        mock.downres_factor_radial = 1.5
        mock.downres_factor_transverse = 1.5
        mock.compensate = False
        gal_map_rg, gal_weights_rg, pixel_counts_gal_rg = mock.grid_gal_to_field()
        _, _, pixel_counts_hi_rg = mock.grid_data_to_field()
        mock.get_n_bar_correction()
        taper = mock.taper_func(mock.box_ndim[-1])
        mock.weights_2 = (pixel_counts_hi_rg > 0) * taper[None, None, :]
        shot_noise_g = (
            np.prod(mock.box_len) * (pixel_counts_hi_rg > 0).mean() / mock.ra_gal.size
        )
        mock.sampling_resol = None
        mock.has_resol = False
        pmod_1d_gg, keff, _ = mock.get_1d_power("auto_power_tracer_2_model")
        pdata_1d_gg, keff, nmodes = mock.get_1d_power(
            "auto_power_3d_2",
        )
        pdata_1d_gg -= shot_noise_g
        pmap_1d += [
            pdata_1d_gg,
        ]
        pmod_1d += [
            pmod_1d_gg,
        ]
    pmap_1d = np.array(pmap_1d)
    pmod_1d = np.array(pmod_1d)
    avg_deviation = ((pmap_1d.mean(0) - pmod_1d.mean(0)) / pmap_1d.std(0)).mean()
    # 3 sigma
    assert np.abs(avg_deviation) < 3
