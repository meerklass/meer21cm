import numpy as np
from meer21cm.stack import sum_3d_stack, stack
from meer21cm import Specification
import pytest


@pytest.mark.parametrize("sym", [(False), (True)])
def test_stack(sym):
    raminMK, ramaxMK = 334, 357
    decminMK, decmaxMK = -35, -26.5
    ra_range_MK = (raminMK, ramaxMK)
    dec_range_MK = (decminMK, decmaxMK)
    sp = Specification(
        ra_range=ra_range_MK,
        dec_range=dec_range_MK,
        survey="meerklass_2021",
        band="L",
    )
    data = sp.data.copy()
    source_1 = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])
    source_2 = np.array([1, 2, 3, 4, 3, 2, 1])
    source_avg = source_1.astype("float").copy()
    source_avg[1:-1] += source_2
    source_avg /= 2.0
    data[80, 30, 80 - 4 : 80 + 5] = source_1
    data[50, 40, 140 - 3 : 140 + 4] = source_2
    ra_g = np.array([sp.ra_map[80, 30], sp.ra_map[50, 40]])
    dec_g = np.array([sp.dec_map[80, 30], sp.dec_map[50, 40]])
    z_g = np.array([sp.z_ch[80], sp.z_ch[140]])
    sp._ra_gal = ra_g
    sp._dec_gal = dec_g
    sp._z_gal = z_g
    sp._data = data
    stack_3D_map, stack_3D_weight = stack(sp, symmetrize=sym)
    indx = np.where(stack_3D_map > 0)
    assert np.allclose(np.unique(indx[0]), [10])
    assert np.allclose(np.unique(indx[1]), [10])
    peak_point = stack_3D_map.shape[-1] // 2
    average_profile = stack_3D_map[10, 10, peak_point - 4 : peak_point + 5]
    assert np.allclose(average_profile, [0.5, 1.5, 2.5, 3.5, 4.5, 3.5, 2.5, 1.5, 0.5])
    # change weights
    w_HI = sp.w_HI.copy()
    # source 1 no weights
    w_HI[80, 30, 80 - 4 : 80 + 5] = 0.0
    sp.w_HI = w_HI
    stack_3D_map, stack_3D_weight = stack(sp, symmetrize=sym)
    indx = np.where(stack_3D_map > 0)
    assert np.allclose(np.unique(indx[0]), [10])
    assert np.allclose(np.unique(indx[1]), [10])
    average_profile = stack_3D_map[10, 10, peak_point - 4 : peak_point + 5]
    assert np.allclose(average_profile, [0, 1, 2, 3, 4, 3, 2, 1, 0])
    # test two sources on top of each other
    sp = Specification(
        ra_range=ra_range_MK,
        dec_range=dec_range_MK,
        survey="meerklass_2021",
        band="L",
    )
    data = sp.data.copy()
    source_1 = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])
    source_avg = source_1.astype("float").copy()
    data[80, 30, 80 - 4 : 80 + 5] += source_1
    data[80, 30, 80 - 4 : 80 + 5] += source_1
    ra_g = np.array([sp.ra_map[80, 30], sp.ra_map[80, 30]])
    dec_g = np.array([sp.dec_map[80, 30], sp.dec_map[80, 30]])
    z_g = np.array([sp.z_ch[80], sp.z_ch[80]])
    sp._ra_gal = ra_g
    sp._dec_gal = dec_g
    sp._z_gal = z_g
    sp._data = data
    stack_3D_map, stack_3D_weight = stack(sp, symmetrize=sym)
    average_profile = stack_3D_map[10, 10, peak_point - 4 : peak_point + 5]
    # due to double counting
    assert np.allclose(average_profile, source_avg * 2)


def test_warn_out_of_range_galaxies_filtered():
    """Galaxies outside the map or frequency range emit a warning and are dropped."""
    raminMK, ramaxMK = 334, 357
    decminMK, decmaxMK = -35, -26.5
    ra_range_MK = (raminMK, ramaxMK)
    dec_range_MK = (decminMK, decmaxMK)
    sp = Specification(
        ra_range=ra_range_MK,
        dec_range=dec_range_MK,
        survey="meerklass_2021",
        band="L",
    )
    z_g = np.array([sp.z_ch[80], sp.z_ch[140]])

    ra_g = np.array([180, 181])
    dec_g = np.array([sp.dec_map[80, 30], sp.dec_map[50, 40]])
    sp._ra_gal = ra_g
    sp._dec_gal = dec_g
    sp._z_gal = z_g
    with pytest.warns(
        UserWarning, match="galaxies are outside survey area or frequency range"
    ):
        stack_3D_map, stack_3D_weight = stack(sp)
    assert stack_3D_weight.sum() == 0
    assert stack_3D_map.sum() == 0

    ra_g = np.array([sp.ra_map[80, 30], sp.ra_map[50, 40]])
    dec_g = np.array([20, 30])
    sp._ra_gal = ra_g
    sp._dec_gal = dec_g
    sp._z_gal = z_g
    with pytest.warns(
        UserWarning, match="galaxies are outside survey area or frequency range"
    ):
        stack_3D_map, stack_3D_weight = stack(sp)
    assert stack_3D_weight.sum() == 0
    assert stack_3D_map.sum() == 0

    ra_g = np.array([sp.ra_map[80, 30], sp.ra_map[50, 40]])
    dec_g = np.array([sp.dec_map[80, 30], sp.dec_map[50, 40]])
    z_g = np.array([sp.z_ch[0] + 0.2, sp.z_ch[-1] - 0.2])
    sp._ra_gal = ra_g
    sp._dec_gal = dec_g
    sp._z_gal = z_g
    with pytest.warns(
        UserWarning, match="galaxies are outside survey area or frequency range"
    ):
        stack_3D_map, stack_3D_weight = stack(sp)
    assert stack_3D_weight.sum() == 0
    assert stack_3D_map.sum() == 0


def test_sum_3d_stack():
    test_3D_map = np.zeros((5, 5, 201))
    test_3D_map[:, :, 95:106] = 1.0
    angular_test, spectral_test = sum_3d_stack(
        test_3D_map, vel_ch_avg=5, ang_sum_dist=3.0
    )
    assert np.allclose(angular_test, np.ones_like(angular_test) * 11.0)
    assert np.allclose(spectral_test[95:106], np.ones(11) * 25)
    assert angular_test.sum() == test_3D_map.sum()
    assert spectral_test.sum() == test_3D_map.sum()
    angular_test, spectral_test = sum_3d_stack(
        test_3D_map, vel_ch_avg=2, ang_sum_dist=1.0
    )
    assert np.allclose(angular_test, np.ones_like(angular_test) * 5.0)
    assert np.allclose(spectral_test[95:106], np.ones(11) * 5)


# def test_weight_source_peaks(test_wproj, test_W, test_nu):
#    map_in = np.zeros((test_W.shape[0], test_W.shape[1], len(test_nu)))
#    ra_in = np.array([350.0])
#    dec_in = np.array([-25.0])
#    z_in = np.array([0.45])
#    gal_freq = f_21 / (1 + z_in)
#    indx_1, indx_2 = radec_to_indx(ra_in, dec_in, test_wproj)
#    # which channel each source centre belongs to
#    gal_which_ch = np.argmin(np.abs(gal_freq[None, :] - test_nu[:, None]), axis=0)
#    map_gal_indx, map_gal_weight = weight_source_peaks(
#        map_in,
#        test_wproj,
#        ra_in,
#        dec_in,
#        z_in,
#        test_nu,
#        ignore_double_counting=True,
#    )
#    assert np.allclose(map_gal_indx, -1 * np.ones_like(map_in))
#    # test no vel
#    map_gal_indx, map_gal_weight = weight_source_peaks(
#        map_in,
#        test_wproj,
#        ra_in,
#        dec_in,
#        z_in,
#        test_nu,
#        no_vel=True,
#        ignore_double_counting=False,
#    )
#    assert map_gal_indx[indx_1, indx_2, gal_which_ch] == 0
#    # test with velocity
#    map_gal_indx, map_gal_weight = weight_source_peaks(
#        map_in,
#        test_wproj,
#        ra_in,
#        dec_in,
#        z_in,
#        test_nu,
#        no_vel=False,
#        ignore_double_counting=False,
#    )
#    assert map_gal_indx[indx_1, indx_2, gal_which_ch] == 0
#    test_indx = np.where(map_gal_indx != -1)
#    assert np.unique(test_indx[0]) == indx_1
#    assert np.unique(test_indx[1]) == indx_2
#    num_of_pixel_per_gal = np.sum(map_gal_indx != -1)
#
#    # test with beam
#    map_gal_indx, map_gal_weight = weight_source_peaks(
#        map_in,
#        test_wproj,
#        ra_in,
#        dec_in,
#        z_in,
#        test_nu,
#        no_vel=False,
#        no_sel_weight=False,
#        gal_sel_indx=[0, 1],
#        ignore_double_counting=False,
#        velocity_profile="step",
#        sigma_beam_in=np.ones(len(test_nu)) * 0.3,
#    )
#    assert map_gal_indx[indx_1, indx_2, gal_which_ch] == 0
#    assert map_gal_indx[indx_1 - 1, indx_2, gal_which_ch] == 0
#    assert map_gal_indx[indx_1, indx_2 - 1, gal_which_ch] == 0
#    assert map_gal_indx[indx_1 + 1, indx_2, gal_which_ch] == 0
#    assert map_gal_indx[indx_1, indx_2 + 1, gal_which_ch] == 0
#
#    # test two galaxies right next to each other
#    ra_in = np.array([350.0, 350.0])
#    dec_in = np.array([-27.0, -27.0])
#    z_in = np.array([0.45, 0.4505])
#    gal_freq = f_21 / (1 + z_in)
#    indx_1, indx_2 = radec_to_indx(ra_in, dec_in, test_wproj)
#    map_gal_indx, map_gal_weight = weight_source_peaks(
#        map_in,
#        test_wproj,
#        ra_in,
#        dec_in,
#        z_in,
#        test_nu,
#        no_vel=False,
#        ignore_double_counting=False,
#        velocity_profile="step",
#    )
#    assert map_gal_weight.sum() < 2
#    assert np.sum(map_gal_indx != -1) < 2 * num_of_pixel_per_gal
#    map_gal_indx, map_gal_weight = weight_source_peaks(
#        map_in,
#        test_wproj,
#        ra_in,
#        dec_in,
#        z_in,
#        test_nu,
#        no_vel=False,
#        no_sel_weight=True,
#        ignore_double_counting=False,
#        velocity_profile="step",
#    )
#    assert np.allclose(map_gal_weight.sum(), 2)
#    with pytest.raises(ValueError):
#        map_gal_indx, map_gal_weight = weight_source_peaks(
#            map_in,
#            test_wproj,
#            ra_in,
#            dec_in,
#            z_in,
#            test_nu,
#            no_vel=False,
#            no_sel_weight=True,
#            ignore_double_counting=False,
#            velocity_profile="something",
#        )
#
#
# def test_stack(test_wproj, test_W, test_nu):
#    map_in = np.zeros((test_W.shape[0], test_W.shape[1], len(test_nu)))
#    ra_in = np.array([350.0])
#    dec_in = np.array([-25.0])
#    z_in = np.array([0.45])
#    gal_freq = f_21 / (1 + z_in)
#    indx_1, indx_2 = radec_to_indx(ra_in, dec_in, test_wproj)
#    # which channel each source centre belongs to
#    gal_which_ch = np.argmin(np.abs(gal_freq[None, :] - test_nu[:, None]), axis=0)
#    map_in[indx_1, indx_2, gal_which_ch] = 1.0
#    (
#        stack_3D_map,
#        stack_3D_weight,
#        x_edges,
#        ang_edges,
#        map_gal_indx,
#        map_gal_weight,
#    ) = stack(
#        map_in,
#        test_wproj,
#        ra_in,
#        dec_in,
#        z_in,
#        test_nu,
#        W_map_in=None,
#        w_map_in=None,
#        velocity_width_halfmax=250,
#        velocity_profile="gaussian",
#        sigma_beam_in=None,
#        no_vel=False,
#        internal_step=2000,
#        verbose=False,
#        ignore_double_counting=False,
#        project_mat=None,
#        gal_sel_indx=None,
#        no_sel_weight=False,
#        stack_angular_num_nearby_pix=10,
#        # x_unit=units.km / units.s,
#        return_indx_and_weight=True,
#    )
#    ang_mid_point = stack_3D_map.shape[0] // 2
#    freq_mid_point = stack_3D_map.shape[2] // 2
#    assert stack_3D_map[ang_mid_point, ang_mid_point, freq_mid_point] == 1.0
#    assert np.sum(stack_3D_map > 0) == 1
#    # test spectral direction
#    map_in = np.zeros((test_W.shape[0], test_W.shape[1], len(test_nu)))
#    map_in[indx_1, indx_2, gal_which_ch] = 0.5
#    map_in[indx_1, indx_2, gal_which_ch - 1] = 0.25
#    map_in[indx_1, indx_2, gal_which_ch + 1] = 0.25
#    (stack_3D_map, stack_3D_weight, x_edges, ang_edges,) = stack(
#        map_in,
#        test_wproj,
#        ra_in,
#        dec_in,
#        z_in,
#        test_nu,
#        W_map_in=None,
#        w_map_in=None,
#        velocity_width_halfmax=250,
#        velocity_profile="gaussian",
#        sigma_beam_in=None,
#        no_vel=False,
#        internal_step=2000,
#        verbose=False,
#        ignore_double_counting=False,
#        project_mat=None,
#        gal_sel_indx=None,
#        no_sel_weight=False,
#        stack_angular_num_nearby_pix=10,
#        # x_unit=units.km / units.s,
#        return_indx_and_weight=False,
#    )
#    assert stack_3D_map[ang_mid_point, ang_mid_point, freq_mid_point] == 0.5
#    assert stack_3D_map[ang_mid_point, ang_mid_point, freq_mid_point - 1] == 0.25
#    assert stack_3D_map[ang_mid_point, ang_mid_point, freq_mid_point + 1] == 0.25
#    assert np.sum(stack_3D_map > 0) == 3
#    # test angular
#    map_in = np.zeros((test_W.shape[0], test_W.shape[1], len(test_nu)))
#    map_in[indx_1, indx_2, gal_which_ch] = 0.6
#    map_in[indx_1 - 1, indx_2, gal_which_ch] = 0.15
#    map_in[indx_1 + 1, indx_2, gal_which_ch] = 0.05
#    map_in[indx_1, indx_2 - 1, gal_which_ch] = 0.05
#    map_in[indx_1, indx_2 + 1, gal_which_ch] = 0.15
#    (stack_3D_map, stack_3D_weight, x_edges, ang_edges,) = stack(
#        map_in,
#        test_wproj,
#        ra_in,
#        dec_in,
#        z_in,
#        test_nu,
#        W_map_in=None,
#        w_map_in=None,
#        velocity_width_halfmax=250,
#        velocity_profile="gaussian",
#        sigma_beam_in=None,
#        no_vel=False,
#        internal_step=2000,
#        verbose=False,
#        ignore_double_counting=False,
#        project_mat=None,
#        gal_sel_indx=None,
#        no_sel_weight=False,
#        stack_angular_num_nearby_pix=10,
#        # x_unit=units.km / units.s,
#        return_indx_and_weight=False,
#    )
#    assert stack_3D_map[ang_mid_point, ang_mid_point, freq_mid_point] == 0.6
#    assert stack_3D_map[ang_mid_point - 1, ang_mid_point, freq_mid_point] == 0.15
#    assert stack_3D_map[ang_mid_point + 1, ang_mid_point, freq_mid_point] == 0.05
#    assert stack_3D_map[ang_mid_point, ang_mid_point - 1, freq_mid_point] == 0.05
#    assert stack_3D_map[ang_mid_point, ang_mid_point + 1, freq_mid_point] == 0.15
#
#    # test averaging
#    ra_in = np.array([350.0, 340.0])
#    dec_in = np.array([-27.0, -24.0])
#    z_in = np.array([0.45, 0.4505])
#    indx_1, indx_2 = radec_to_indx(ra_in, dec_in, test_wproj)
#    gal_freq = f_21 / (1 + z_in)
#    gal_which_ch = np.argmin(np.abs(gal_freq[None, :] - test_nu[:, None]), axis=0)
#    map_in = np.zeros((test_W.shape[0], test_W.shape[1], len(test_nu)))
#    map_in[indx_1[0], indx_2[0], gal_which_ch[0]] = 1.5
#    map_in[indx_1[1], indx_2[1], gal_which_ch[1]] = 0.5
#    (stack_3D_map, stack_3D_weight, x_edges, ang_edges,) = stack(
#        map_in,
#        test_wproj,
#        ra_in,
#        dec_in,
#        z_in,
#        test_nu,
#        W_map_in=None,
#        w_map_in=None,
#        velocity_width_halfmax=250,
#        velocity_profile="gaussian",
#        sigma_beam_in=None,
#        no_vel=False,
#        internal_step=2000,
#        verbose=False,
#        ignore_double_counting=False,
#        project_mat=None,
#        gal_sel_indx=None,
#        no_sel_weight=False,
#        stack_angular_num_nearby_pix=10,
#        # x_unit=units.km / units.s,
#        return_indx_and_weight=False,
#    )
#    assert stack_3D_map[ang_mid_point, ang_mid_point, freq_mid_point] == 1.0
#    # test different unit
#    (stack_3D_map, stack_3D_weight, x_edges, ang_edges,) = stack(
#        map_in,
#        test_wproj,
#        ra_in,
#        dec_in,
#        z_in,
#        test_nu,
#        W_map_in=None,
#        w_map_in=None,
#        velocity_width_halfmax=250,
#        velocity_profile="gaussian",
#        sigma_beam_in=None,
#        no_vel=False,
#        internal_step=2000,
#        verbose=False,
#        ignore_double_counting=False,
#        project_mat=None,
#        gal_sel_indx=None,
#        no_sel_weight=False,
#        stack_angular_num_nearby_pix=10,
#        x_unit=units.Hz,
#        return_indx_and_weight=False,
#    )
#    assert stack_3D_map[ang_mid_point, ang_mid_point, freq_mid_point] == 1.0
#    # test raise error
#    with pytest.raises(ValueError):
#        (stack_3D_map, stack_3D_weight, x_edges, ang_edges,) = stack(
#            map_in,
#            test_wproj,
#            ra_in,
#            dec_in,
#            z_in,
#            test_nu,
#            W_map_in=None,
#            w_map_in=None,
#            velocity_width_halfmax=250,
#            velocity_profile="gaussian",
#            sigma_beam_in=None,
#            no_vel=False,
#            internal_step=2000,
#            verbose=False,
#            ignore_double_counting=False,
#            project_mat=None,
#            gal_sel_indx=None,
#            no_sel_weight=False,
#            stack_angular_num_nearby_pix=10,
#            x_unit=units.K,
#            return_indx_and_weight=False,
#        )
