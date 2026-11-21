import numpy as np
from meer21cm.stack import (
    Stacking,
    aperture_weights_2d,
    azimuthal_pixel_window_lag,
    beam_window_amplitude,
    channel_cross_covariance,
    comoving_bin_indices,
    contract_channel_covariance,
    correlate_cubelet,
    data_and_fkp_cubelets,
    discrete_beam_window_amplitude,
    paint_sky_counts,
    rebin_cube,
    rebin_sightlines,
    sky_mesh_positions,
    stack,
    stack_cubelet,
    sum_3d_stack,
)
from meer21cm.util import create_wcs, freq_to_redshift
from meer21cm import Specification
import pytest


def test_stack_invoke_error():
    with pytest.raises(ValueError):
        stack_cubelet(
            1.0,
            1.0,
            [0],
            [0],
            [0],
            weighting="random",
        )


@pytest.mark.parametrize(
    "sym, weighting", [(False, "conventional"), (True, "quadratic")]
)
def test_stack(sym, weighting):
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
    stack_3D_map, stack_3D_weight = stack(sp, symmetrize=sym, weighting=weighting)
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
    stack_3D_map, stack_3D_weight = stack(sp, symmetrize=sym, weighting=weighting)
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
    stack_3D_map, stack_3D_weight = stack(sp, symmetrize=sym, weighting=weighting)
    average_profile = stack_3D_map[10, 10, peak_point - 4 : peak_point + 5]
    # due to double counting
    assert np.allclose(average_profile, source_avg * 2)


@pytest.mark.parametrize("weighting", ["conventional", "quadratic"])
def test_stack_weights(weighting):
    input_map = np.zeros((3, 3, 1))
    input_map[1, 1, 0] = 1.0
    input_map[0, 1, 0] = 1.0
    weights_map = np.zeros((3, 3, 1))
    weights_map[1, 1, 0] = 1.0
    weights_map[0, 1, 0] = 2.0
    stack_3D_map, stack_3D_weight = stack_cubelet(
        input_map,
        weights_map,
        [1],
        [1],
        [0],
        weighting=weighting,
        stack_angular_num_nearby_pix=1,
        symmetrize=False,
    )
    if weighting == "conventional":
        assert np.allclose(stack_3D_map, input_map)
    elif weighting == "quadratic":
        output_map = np.zeros((3, 3, 1))
        output_map[1, 1, 0] = 1.0
        output_map[0, 1, 0] = 2.0
        assert np.allclose(stack_3D_map, output_map)


def test_raise_error():
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
    ra_g = np.array([180, 181])
    dec_g = np.array([sp.dec_map[80, 30], sp.dec_map[50, 40]])
    z_g = np.array([sp.z_ch[80], sp.z_ch[140]])
    sp._ra_gal = ra_g
    sp._dec_gal = dec_g
    sp._z_gal = z_g
    with pytest.raises(ValueError):
        stack(sp)
    ra_g = np.array([sp.ra_map[80, 30], sp.ra_map[50, 40]])
    dec_g = np.array([20, 30])
    sp._ra_gal = ra_g
    sp._dec_gal = dec_g
    sp._z_gal = z_g
    with pytest.raises(ValueError):
        stack(sp)
    ra_g = np.array([sp.ra_map[80, 30], sp.ra_map[50, 40]])
    dec_g = np.array([sp.dec_map[80, 30], sp.dec_map[50, 40]])
    z_g = np.array([sp.z_ch[0] + 0.2, sp.z_ch[-1] - 0.2])
    sp._ra_gal = ra_g
    sp._dec_gal = dec_g
    sp._z_gal = z_g
    with pytest.raises(ValueError):
        stack(sp)


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


def _uniform_chi(worker):
    """Force one comoving bin per native channel."""
    chi = np.arange(worker.nu.size, dtype=float) * worker.dchi_mpc
    worker.channel_chi_mpc = lambda chi=chi: chi
    return chi


def _tiny_worker(data, weight, ra, dec, z, nu, wcs, estimator, centre_pixels=True):
    return Stacking(
        data=data,
        weights_map_pixel=weight,
        nu=nu,
        wproj=wcs,
        num_pix_x=data.shape[0],
        num_pix_y=data.shape[1],
        ra_gal=ra,
        dec_gal=dec,
        z_gal=z,
        stack_estimator=estimator,
        stack_angular_num_nearby_pix=1,
        dchi_mpc=4.0,
        centre_pixels=centre_pixels,
        filter_map_los=False,
        auto_set_radecnu_bounds=False,
        init_box_from_map_data=False,
        ra_range=(0, 360),
        dec_range=(-90, 90),
    )


def test_correlation_agrees_with_loop_on_uniform_chi():
    """Loop and correlation cubelets match, including a shared voxel and an edge."""
    shape = (6, 5, 4)
    temperature = np.arange(np.prod(shape), dtype=float).reshape(shape)
    weight = np.ones(shape)
    chi = np.arange(shape[-1], dtype=float) * 4.0
    # two galaxies in cell (3, 2, 1), one on the corner cell (0, 0, 2)
    positions = np.array(
        [
            [3.5, 2.5, 1.5],
            [3.5, 2.5, 1.5],
            [0.5, 0.5, 2.5],
        ]
    )
    indices = np.array([[3, 2, 1], [3, 2, 1], [0, 0, 2]])
    counts = paint_sky_counts(positions, shape)
    assert counts[3, 2, 1] == 2.0
    assert counts[0, 0, 2] == 1.0
    assert counts.sum() == 3.0
    rebinned_w, rebinned_wt, rebinned_n, _centres = rebin_sightlines(
        weight * temperature, weight, counts, chi, 4.0
    )
    # uniform spacing equal to the bin width keeps one channel per bin
    assert np.allclose(rebinned_w, weight * temperature)
    cube, cube_weight = correlate_cubelet(rebinned_w, rebinned_wt, rebinned_n, 1)
    loop, loop_weight = stack_cubelet(
        temperature,
        weight,
        indices[:, 0],
        indices[:, 1],
        indices[:, 2],
        stack_angular_num_nearby_pix=1,
    )
    assert np.allclose(cube, loop, rtol=1e-8, atol=1e-8)
    assert np.allclose(cube_weight, loop_weight, rtol=1e-8, atol=1e-8)
    # the shared voxel doubles the weight and leaves the average unchanged
    one, one_weight = stack_cubelet(
        temperature,
        weight,
        indices[:1, 0],
        indices[:1, 1],
        indices[:1, 2],
        stack_angular_num_nearby_pix=1,
    )
    two, two_weight = stack_cubelet(
        temperature,
        weight,
        indices[:2, 0],
        indices[:2, 1],
        indices[:2, 2],
        stack_angular_num_nearby_pix=1,
    )
    assert np.allclose(two, one)
    assert np.allclose(two_weight, 2.0 * one_weight)
    # the corner galaxy has no map at negative angular lags
    edge, edge_weight = stack_cubelet(
        temperature,
        weight,
        indices[2:, 0],
        indices[2:, 1],
        indices[2:, 2],
        stack_angular_num_nearby_pix=1,
    )
    centre = edge.shape[-1] // 2
    assert edge_weight[0, :, centre].sum() == 0.0
    assert edge[edge.shape[0] // 2, edge.shape[1] // 2, centre] == temperature[0, 0, 2]


def test_half_pixel_centring():
    """WCS pixel centres land in mesh cell i, and the zero-lag peak follows."""
    n_pix = 8
    n_ch = 3
    wcs = create_wcs(180.0, 0.0, [n_pix, n_pix], 1.0)
    nu = np.linspace(700e6, 710e6, n_ch)
    z = np.array([freq_to_redshift(nu[1])])
    pixel = 3.0
    sky_on_cell = wcs.pixel_to_world(pixel + 0.2, pixel)
    ra = np.array([sky_on_cell.ra.deg])
    dec = np.array([sky_on_cell.dec.deg])
    positions, inside = sky_mesh_positions(
        ra, dec, z, wcs, nu, (n_pix, n_pix, n_ch), centre_pixels=True
    )
    assert inside.all()
    counts = paint_sky_counts(positions, (n_pix, n_pix, n_ch))
    assert counts.sum() == 1.0
    assert counts[int(pixel), int(pixel), 1] == 1.0

    # fractional part 0.7: adding half a cell crosses into the next mesh cell
    sky_cross = wcs.pixel_to_world(pixel + 0.7, pixel)
    ra_c = np.array([sky_cross.ra.deg])
    dec_c = np.array([sky_cross.dec.deg])
    centred, _ = sky_mesh_positions(
        ra_c, dec_c, z, wcs, nu, (n_pix, n_pix, n_ch), centre_pixels=True
    )
    raw, _ = sky_mesh_positions(
        ra_c, dec_c, z, wcs, nu, (n_pix, n_pix, n_ch), centre_pixels=False
    )
    cell_x = int(np.floor(centred[0, 0]))
    cell_y = int(np.floor(centred[0, 1]))
    raw_x = int(np.floor(raw[0, 0]))
    raw_y = int(np.floor(raw[0, 1]))
    assert cell_x == raw_x + 1
    temperature = np.zeros((n_pix, n_pix, n_ch))
    weight = np.ones_like(temperature)
    temperature[cell_x, cell_y, 1] = 1.0
    on_centre = _tiny_worker(
        temperature, weight, ra_c, dec_c, z, nu, wcs, "correlation", True
    )
    off_centre = _tiny_worker(
        temperature, weight, ra_c, dec_c, z, nu, wcs, "correlation", False
    )
    # narrower than the channel spacing, so the spike stays in its own bin
    dchi = 0.5 * np.min(np.diff(np.sort(on_centre.channel_chi_mpc())))
    on_centre.dchi_mpc = dchi
    off_centre.dchi_mpc = dchi
    cube_on, _wt = on_centre.run_stack()
    cube_off, _wt = off_centre.run_stack()
    mid = cube_on.shape[0] // 2
    los = cube_on.shape[-1] // 2
    assert np.unravel_index(np.argmax(cube_on[:, :, los]), cube_on.shape[:2]) == (
        mid,
        mid,
    )
    assert cube_on[mid, mid, los] == pytest.approx(1.0)
    # the map was painted with the half-cell shift, so the raw coordinate
    # reads the neighbouring pixel
    assert cube_off[mid + (cell_x - raw_x), mid + (cell_y - raw_y), los] == (
        pytest.approx(1.0)
    )
    assert np.unravel_index(np.argmax(cube_off[:, :, los]), cube_off.shape[:2]) != (
        mid,
        mid,
    )


def test_los_rebin_conserves_galaxies_and_places_spike():
    chi = np.array([0.0, 1.5, 3.2, 5.0, 7.1])
    counts = np.zeros((2, 2, chi.size))
    counts[0, 0] = np.array([1.0, 0.0, 2.0, 0.0, 4.0])
    weighted = np.zeros_like(counts)
    weighted[1, 0, 3] = 9.0
    weight = np.ones_like(counts)
    out_w, out_wt, out_n, centres = rebin_sightlines(weighted, weight, counts, chi, 4.0)
    assert out_n.sum() == counts.sum()
    # chi = 5 Mpc is inside the bin that starts at 4 Mpc
    spike_bin = int(np.digitize([5.0], np.arange(0.0, 7.1 + 2.0, 4.0))[0] - 1)
    assert out_w[1, 0, spike_bin] == 9.0
    assert out_w[1, 0].sum() == 9.0
    assert centres[spike_bin] == pytest.approx(6.0)
    assert out_wt[..., 0].sum() > 0.0


def test_stacking_loop_and_correlation_dispatch():
    """The worker's two estimators agree on a uniform-chi cube."""
    n_pix = 6
    wcs = create_wcs(180.0, 0.0, [n_pix, n_pix], 1.0)
    nu = np.linspace(700e6, 712e6, 4)
    pixels = [(3, 2, 1), (3, 2, 1), (0, 1, 2)]
    sky = [wcs.pixel_to_world(i, j) for i, j, _ch in pixels]
    ra = np.array([s.ra.deg for s in sky])
    dec = np.array([s.dec.deg for s in sky])
    z = np.array([freq_to_redshift(nu[ch]) for _i, _j, ch in pixels])
    temperature = np.arange(n_pix * n_pix * nu.size, dtype=float).reshape(
        n_pix, n_pix, nu.size
    )
    weight = np.ones_like(temperature)
    loop = _tiny_worker(temperature, weight, ra, dec, z, nu, wcs, "loop")
    correlation = _tiny_worker(temperature, weight, ra, dec, z, nu, wcs, "correlation")
    _uniform_chi(correlation)
    loop_cube, loop_weight = loop.run_stack()
    corr_cube, corr_weight = correlation.run_stack()
    direct, direct_weight = stack(loop, stack_angular_num_nearby_pix=1)
    assert np.allclose(loop_cube, direct)
    assert np.allclose(loop_weight, direct_weight)
    assert np.allclose(corr_cube, loop_cube, rtol=1e-8, atol=1e-8)
    assert np.allclose(corr_weight, loop_weight, rtol=1e-8, atol=1e-8)
    assert correlation.n_gal_stacked == 3
    assert loop.stack_3d is loop_cube


def test_unknown_stack_estimator_and_unset_properties():
    import meer21cm

    assert meer21cm.Stacking is Stacking
    with pytest.raises(ValueError):
        Stacking(stack_estimator="config")
    n_pix = 4
    wcs = create_wcs(180.0, 0.0, [n_pix, n_pix], 1.0)
    nu = np.linspace(700e6, 704e6, 2)
    sky = wcs.pixel_to_world(1, 1)
    worker = _tiny_worker(
        np.zeros((n_pix, n_pix, 2)),
        np.ones((n_pix, n_pix, 2)),
        np.array([sky.ra.deg]),
        np.array([sky.dec.deg]),
        np.array([freq_to_redshift(nu[0])]),
        nu,
        wcs,
        "loop",
    )
    assert worker.stack_3d is None
    assert worker.stack_weight is None


def test_galaxy_weight_scales_the_correlation_weight():
    """A per-galaxy weight scales the cubelet weight and leaves the average."""
    n_pix = 4
    wcs = create_wcs(180.0, 0.0, [n_pix, n_pix], 1.0)
    nu = np.linspace(700e6, 708e6, 3)
    sky = wcs.pixel_to_world(2, 2)
    ra = np.array([sky.ra.deg])
    dec = np.array([sky.dec.deg])
    z = np.array([freq_to_redshift(nu[1])])
    temperature = np.ones((n_pix, n_pix, nu.size))
    weight = np.ones_like(temperature)
    unit = _tiny_worker(temperature, weight, ra, dec, z, nu, wcs, "correlation")
    weighted = Stacking(
        data=temperature,
        weights_map_pixel=weight,
        nu=nu,
        wproj=wcs,
        num_pix_x=n_pix,
        num_pix_y=n_pix,
        ra_gal=ra,
        dec_gal=dec,
        z_gal=z,
        weights_gal=np.array([2.5]),
        stack_estimator="correlation",
        stack_angular_num_nearby_pix=1,
        dchi_mpc=4.0,
        filter_map_los=False,
        auto_set_radecnu_bounds=False,
        init_box_from_map_data=False,
    )
    _uniform_chi(unit)
    _uniform_chi(weighted)
    cube_unit, weight_unit = unit.run_stack()
    cube_weighted, weight_weighted = weighted.run_stack()
    assert np.allclose(cube_weighted, cube_unit)
    assert np.allclose(weight_weighted, 2.5 * weight_unit)
    assert weighted.stack_weight is weight_weighted
    assert weighted.stack_3d is cube_weighted


def test_stacking_rejects_inconsistent_settings():
    empty = paint_sky_counts(np.zeros((0, 3)), (2, 2, 2))
    assert empty.shape == (2, 2, 2)
    assert empty.sum() == 0.0
    with pytest.raises(ValueError):
        Stacking(stack_weighting="nope")
    with pytest.raises(ValueError):
        Stacking(dchi_mpc=0.0)
    with pytest.raises(ValueError):
        Stacking(stack_angular_num_nearby_pix=-1)
    with pytest.raises(ValueError, match="provided together"):
        Stacking(ra_gal=[1.0], dec_gal=[0.0])
    with pytest.raises(ValueError, match="without"):
        Stacking(weights_gal=[1.0])
    with pytest.raises(ValueError, match="equal length"):
        Stacking(ra_gal=[1.0, 2.0], dec_gal=[0.0], z_gal=[0.5, 0.6])
    with pytest.raises(ValueError, match="same length"):
        Stacking(ra_gal=[1.0], dec_gal=[0.0], z_gal=[0.5], weights_gal=[1.0, 2.0])
    with pytest.raises(ValueError, match="provided together"):
        Stacking(ra_rand=[1.0])
    with pytest.raises(ValueError, match="without"):
        Stacking(weights_rand=[1.0])
    n_pix = 4
    wcs = create_wcs(180.0, 0.0, [n_pix, n_pix], 1.0)
    nu = np.linspace(700e6, 704e6, 2)
    sky = wcs.pixel_to_world(1, 1)
    ra = np.array([sky.ra.deg])
    dec = np.array([sky.dec.deg])
    z = np.array([freq_to_redshift(nu[0])])
    worker = _tiny_worker(
        np.ones((n_pix, n_pix, 2)),
        np.ones((n_pix, n_pix, 2)),
        ra,
        dec,
        z,
        nu,
        wcs,
        "correlation",
    )
    worker.stack_weighting = "bad"
    with pytest.raises(ValueError, match="stack_weighting"):
        worker.run_stack()
    worker.stack_weighting = "conventional"
    worker._ra_gal = None
    with pytest.raises(ValueError, match="provided together"):
        worker.run_stack()
    worker._dec_gal = None
    worker._z_gal = None
    with pytest.raises(ValueError, match="ra_gal"):
        worker.run_stack()
    del worker._ra_gal
    with pytest.raises(ValueError, match="ra_gal"):
        worker.run_stack()
    worker.stack_estimator = "mesh"
    with pytest.raises(ValueError, match="stack_estimator"):
        worker.run_stack()


def test_fkp_numerator_uses_data_pair_counts():
    """F drops a spike that sits only on randoms and keeps one that sits on galaxies."""
    shape = (5, 5, 4)
    temperature = np.zeros(shape)
    temperature[2, 2, 1] = 4.0
    weight = np.ones(shape)
    data_counts = np.zeros(shape)
    data_counts[2, 2, 1] = 1.0
    random_counts = np.zeros(shape)
    random_counts[0, 0, 0] = 1.0
    random_counts[0, 1, 3] = 1.0
    random_counts[4, 4, 2] = 1.0
    data, fkp, denominator, alpha = data_and_fkp_cubelets(
        weight * temperature,
        weight,
        data_counts,
        1,
        random_counts=random_counts,
    )
    direct, direct_weight = correlate_cubelet(
        weight * temperature, weight, data_counts, 1
    )
    assert np.allclose(data, direct)
    assert np.allclose(denominator, direct_weight)
    assert alpha == pytest.approx(1.0 / 3.0)
    centre = data.shape[-1] // 2
    mid = data.shape[0] // 2
    assert data[mid, mid, centre] == pytest.approx(4.0)
    assert fkp[mid, mid, centre] == pytest.approx(4.0)
    # a constant temperature is the random-position stack, which F removes
    constant = np.full(shape, 2.5)
    data_c, fkp_c, _den_c, _alpha_c = data_and_fkp_cubelets(
        weight * constant, weight, data_counts, 1, random_counts=random_counts
    )
    assert data_c[mid, mid, centre] == pytest.approx(2.5)
    assert np.allclose(fkp_c, 0.0)
    # a mask on the random voxels leaves that lag empty rather than equal to T
    masked = weight.copy()
    masked[random_counts > 0] = 0.0
    data_m, fkp_m, _den_m, _alpha_m = data_and_fkp_cubelets(
        masked * constant, masked, data_counts, 1, random_counts=random_counts
    )
    assert data_m[mid, mid, centre] == pytest.approx(2.5)
    assert fkp_m[mid, mid, centre] == pytest.approx(0.0)
    _data_z, fkp_z, _den_z, _alpha_z = data_and_fkp_cubelets(
        np.zeros(shape), np.zeros(shape), data_counts, 1, random_counts=random_counts
    )
    assert np.all(fkp_z == 0.0)
    plain, _fkp_none, _den, alpha_none = data_and_fkp_cubelets(
        weight * temperature, weight, data_counts, 1
    )
    assert alpha_none is None
    assert _fkp_none is None
    assert np.allclose(plain, data)


def test_stacking_fkp_matches_the_data_stack_on_a_spike():
    n_pix = 6
    wcs = create_wcs(180.0, 0.0, [n_pix, n_pix], 1.0)
    nu = np.linspace(700e6, 712e6, 4)
    galaxy = wcs.pixel_to_world(3, 2)
    random = wcs.pixel_to_world(1, 1)
    z_gal = np.array([freq_to_redshift(nu[1])])
    z_ran = np.array([freq_to_redshift(nu[3])])
    temperature = np.zeros((n_pix, n_pix, nu.size))
    position = sky_mesh_positions(
        np.array([galaxy.ra.deg]),
        np.array([galaxy.dec.deg]),
        z_gal,
        wcs,
        nu,
        temperature.shape,
        centre_pixels=True,
    )[0][0]
    temperature[int(np.floor(position[0])), int(np.floor(position[1])), 1] = 4.0
    weight = np.ones_like(temperature)
    worker = Stacking(
        data=temperature,
        weights_map_pixel=weight,
        nu=nu,
        wproj=wcs,
        num_pix_x=n_pix,
        num_pix_y=n_pix,
        ra_gal=np.array([galaxy.ra.deg]),
        dec_gal=np.array([galaxy.dec.deg]),
        z_gal=z_gal,
        ra_rand=np.array([random.ra.deg]),
        dec_rand=np.array([random.dec.deg]),
        z_rand=z_ran,
        weights_rand=np.array([1.0]),
        stack_estimator="correlation",
        stack_angular_num_nearby_pix=1,
        dchi_mpc=4.0,
        filter_map_los=False,
        auto_set_radecnu_bounds=False,
        init_box_from_map_data=False,
    )
    _uniform_chi(worker)
    worker.weights_rand = None
    data, _weight = worker.run_stack()
    fkp = worker.stack_fkp
    mid = data.shape[0] // 2
    centre = data.shape[-1] // 2
    assert worker.fkp_alpha == pytest.approx(1.0)
    assert worker.n_ran_stacked == 1
    assert data[mid, mid, centre] == pytest.approx(4.0)
    assert fkp[mid, mid, centre] == pytest.approx(4.0)
    worker.weights_rand = np.array([2.0])
    data_w, _weight_w = worker.run_stack()
    assert worker.fkp_alpha == pytest.approx(0.5)
    assert data_w[mid, mid, centre] == pytest.approx(4.0)
    assert worker.stack_fkp[mid, mid, centre] == pytest.approx(4.0)
    worker.data = np.full_like(temperature, 2.5)
    data_c, _weight_c = worker.run_stack()
    assert data_c[mid, mid, centre] == pytest.approx(2.5)
    assert np.allclose(worker.stack_fkp, 0.0)
    worker.stack_estimator = "loop"
    with pytest.raises(ValueError, match="randoms"):
        worker.run_stack()


def test_stack_cubelet_zero_nearby_and_negative_channel():
    """stack_cubelet supports n_nearby=0 and rejects negative channel/pad."""
    cube = np.arange(2 * 2 * 3, dtype=float).reshape(2, 2, 3)
    weight = np.ones_like(cube)
    out, wt = stack_cubelet(
        cube,
        weight,
        [1],
        [0],
        [1],
        stack_angular_num_nearby_pix=0,
    )
    assert out.shape == (1, 1, 5)
    assert np.allclose(out[0, 0, 1:4], cube[1, 0, :])
    assert np.allclose(wt[0, 0, 1:4], 1.0)
    with pytest.raises(ValueError, match="non-negative"):
        stack_cubelet(cube, weight, [0], [0], [0], stack_angular_num_nearby_pix=-1)
    with pytest.raises(ValueError, match="outside"):
        stack_cubelet(cube, weight, [0], [0], [-1], stack_angular_num_nearby_pix=0)


def test_comoving_bin_indices_and_rebin_cube():
    with pytest.raises(ValueError, match="dchi"):
        comoving_bin_indices([1.0, 2.0], 0.0)
    with pytest.raises(ValueError, match="non-empty"):
        comoving_bin_indices([], 1.0)
    chi_desc = np.array([12.0, 8.0, 4.0, 0.0])
    cube = np.arange(4, dtype=float).reshape(1, 1, 4)
    rebinned, centres = rebin_cube(cube, chi_desc, 4.0)
    # Sorted ascending in chi -> reverses native channel order
    assert np.allclose(rebinned[0, 0], [3.0, 2.0, 1.0, 0.0])
    assert np.allclose(centres, [2.0, 6.0, 10.0, 14.0])


def test_loop_vs_correlation_los_sign_on_descending_chi():
    """Physical ascending nu -> descending chi flips the LOS lag axis between loop and correlation."""
    n_pix = 5
    wcs = create_wcs(180.0, 0.0, [n_pix, n_pix], 1.0)
    nu = np.linspace(700e6, 712e6, 4)
    sky = wcs.pixel_to_world(2, 2)
    ra = np.array([sky.ra.deg])
    dec = np.array([sky.dec.deg])
    # Galaxy at channel 1 (higher chi), signal spike at channel 2 (higher nu, lower chi)
    z = np.array([freq_to_redshift(nu[1])])
    temp = np.zeros((n_pix, n_pix, 4))
    temp[2, 2, 2] = 5.0
    wt = np.ones_like(temp)

    w_loop = _tiny_worker(temp, wt, ra, dec, z, nu, wcs, "loop")
    loop_cube, _ = w_loop.run_stack()
    assert w_loop.s_parallel_mpc is None
    assert w_loop.bin_chi_centres is None

    # Use native channel spacing so each channel stays in its own bin in ascending chi
    w_corr = _tiny_worker(temp, wt, ra, dec, z, nu, wcs, "correlation")
    chi = w_corr.channel_chi_mpc()
    assert np.all(np.diff(chi) < 0)
    w_corr.dchi_mpc = float(np.min(np.abs(np.diff(chi))) * 0.99)
    corr_cube, _ = w_corr.run_stack()

    mid = 1
    loop_c = loop_cube.shape[-1] // 2
    corr_c = corr_cube.shape[-1] // 2
    # In loop (+delta_nu), spike at ch=2 vs gal ch=1 sits at +1 lag
    assert loop_cube[mid, mid, loop_c + 1] == pytest.approx(5.0)
    # In correlation (+delta_chi), ch=2 has smaller chi than ch=1 -> negative s_parallel
    assert corr_cube[mid, mid, corr_c - 1] == pytest.approx(5.0)
    assert w_corr.s_parallel_mpc[corr_c - 1] < 0.0
    assert np.array_equal(w_corr.delta_i, [-1, 0, 1])
    assert np.array_equal(w_corr.delta_j, [-1, 0, 1])
    assert w_corr.bin_chi_edges.size == w_corr.bin_chi_centres.size + 1
    assert w_corr.bin_channel_which.size == nu.size


@pytest.mark.parametrize("weighting", ["conventional", "quadratic"])
@pytest.mark.parametrize("symmetrize", [False, True])
def test_correlation_weighting_and_symmetrize_match_loop(weighting, symmetrize):
    """Correlation matches loop for both conventional/quadratic and symmetrize=False/True."""
    n_pix = 5
    wcs = create_wcs(180.0, 0.0, [n_pix, n_pix], 1.0)
    nu = np.linspace(700e6, 712e6, 4)
    rng = np.random.default_rng(12)
    temp = rng.normal(0.0, 1.0, size=(n_pix, n_pix, 4))
    wt = rng.uniform(0.5, 2.0, size=(n_pix, n_pix, 4))
    pixels = [(2, 2, 1), (3, 1, 2), (2, 2, 1)]
    sky = [wcs.pixel_to_world(i, j) for i, j, _ in pixels]
    ra = np.array([s.ra.deg for s in sky])
    dec = np.array([s.dec.deg for s in sky])
    z = np.array([freq_to_redshift(nu[c]) for _, _, c in pixels])
    w_gal = np.array([1.0, 2.5, 0.5])

    w_loop = _tiny_worker(temp, wt, ra, dec, z, nu, wcs, "loop")
    w_loop.weights_gal = w_gal
    w_loop.stack_weighting = weighting
    w_loop.symmetrize = symmetrize
    c_loop, wt_loop = w_loop.run_stack()

    w_corr = _tiny_worker(temp, wt, ra, dec, z, nu, wcs, "correlation")
    w_corr.weights_gal = w_gal
    w_corr.stack_weighting = weighting
    w_corr.symmetrize = symmetrize
    _uniform_chi(w_corr)
    c_corr, wt_corr = w_corr.run_stack()

    assert np.allclose(c_corr, c_loop, rtol=1e-8, atol=1e-8)
    assert np.allclose(wt_corr, wt_loop, rtol=1e-8, atol=1e-8)


def test_quadratic_fkp_and_details_and_stale_reset():
    """Verify quadratic FKP estimator, exposed random attributes, and stale reset."""
    with pytest.raises(ValueError, match="weighting"):
        data_and_fkp_cubelets(
            np.ones((2, 2, 2)),
            np.ones((2, 2, 2)),
            np.ones((2, 2, 2)),
            0,
            weighting="bad",
        )
    shape = (4, 4, 3)
    temp = np.full(shape, 3.0)
    wt = np.ones(shape)
    d = np.zeros(shape)
    d[2, 2, 1] = 2.0
    r = np.zeros(shape)
    r[2, 2, 1] = 4.0
    data, fkp, den, alpha, r_stk, r_wt, valid = data_and_fkp_cubelets(
        wt * temp,
        wt,
        d,
        1,
        random_counts=r,
        weighting="quadratic",
        symmetrize=True,
        return_details=True,
    )
    assert alpha == pytest.approx(0.5)
    assert data[1, 1, 2] == pytest.approx(3.0)
    assert r_stk[1, 1, 2] == pytest.approx(3.0)
    assert np.allclose(fkp, 0.0)
    assert valid.all()
    assert r_wt[1, 1, 2] == pytest.approx(8.0)

    # Zero denominator quadratic path
    data0, fkp0, _d0, _a0, r0, _rw0, v0 = data_and_fkp_cubelets(
        np.zeros(shape),
        np.zeros(shape),
        d,
        1,
        random_counts=r,
        weighting="quadratic",
        return_details=True,
    )
    assert np.all(data0 == 0.0)
    assert np.all(fkp0 == 0.0)
    assert np.all(r0 == 0.0)
    assert not v0.any()

    # Worker exposes stack_rand, stack_rand_weight, stack_valid_mask and resets cleanly
    n_pix = 5
    wcs = create_wcs(180.0, 0.0, [n_pix, n_pix], 1.0)
    nu = np.linspace(700e6, 708e6, 3)
    sky_g = wcs.pixel_to_world(2, 2)
    sky_r = wcs.pixel_to_world(1, 1)
    w = _tiny_worker(
        np.ones((n_pix, n_pix, 3)),
        np.ones((n_pix, n_pix, 3)),
        np.array([sky_g.ra.deg]),
        np.array([sky_g.dec.deg]),
        np.array([freq_to_redshift(nu[1])]),
        nu,
        wcs,
        "correlation",
    )
    _uniform_chi(w)
    w.ra_rand = np.array([sky_r.ra.deg])
    w.dec_rand = np.array([sky_r.dec.deg])
    w.z_rand = np.array([freq_to_redshift(nu[1])])
    w.run_stack()
    assert w.stack_rand is not None
    assert w.stack_rand_weight is not None
    assert w.stack_valid_mask is not None
    # Clearing randoms and rerunning clears all random outputs
    w.ra_rand = None
    w.dec_rand = None
    w.z_rand = None
    w.run_stack()
    assert w.stack_rand is None
    assert w.stack_rand_weight is None
    assert w.stack_valid_mask is None
    assert w.stack_fkp is None
    assert w.n_ran_stacked == 0


def test_aperture_weights_and_project_1d():
    with pytest.raises(ValueError, match="non-negative"):
        aperture_weights_2d(-1)
    with pytest.raises(ValueError, match="shape"):
        aperture_weights_2d(1, weights_2d=np.ones((2, 2)))
    with pytest.raises(ValueError, match="tophat"):
        aperture_weights_2d(1, aperture="tophat", radius=-1.0)
    with pytest.raises(ValueError, match="gaussian"):
        aperture_weights_2d(1, aperture="gaussian", radius=0.0)
    with pytest.raises(ValueError, match="aperture"):
        aperture_weights_2d(1, aperture="hexagon")
    with pytest.raises(ValueError, match="sum to zero"):
        aperture_weights_2d(1, weights_2d=np.zeros((3, 3)))

    w_c = aperture_weights_2d(2, aperture="central")
    assert w_c[2, 2] == 1.0 and w_c.sum() == 1.0
    w_t = aperture_weights_2d(2, aperture="tophat", radius=1.0, normalize=False)
    assert w_t.sum() == 5.0  # centre + 4 cardinal neighbours
    w_g = aperture_weights_2d(2, aperture="gaussian", radius=1.5)
    assert w_g.sum() == pytest.approx(1.0)
    assert w_g[2, 2] > w_g[2, 3] > w_g[3, 3]

    # Worker project_1d + default beam-matched Gaussian radius
    n_pix = 5
    wcs = create_wcs(180.0, 0.0, [n_pix, n_pix], 1.0)
    nu = np.linspace(700e6, 708e6, 3)
    sky = wcs.pixel_to_world(2, 2)
    w = _tiny_worker(
        np.ones((n_pix, n_pix, 3)),
        np.ones((n_pix, n_pix, 3)),
        np.array([sky.ra.deg]),
        np.array([sky.dec.deg]),
        np.array([freq_to_redshift(nu[1])]),
        nu,
        wcs,
        "correlation",
    )
    with pytest.raises(ValueError, match="run_stack"):
        w.project_1d()
    with pytest.raises(ValueError, match="shape"):
        w.project_1d(cubelet=np.ones((2, 2, 3)))
    _uniform_chi(w)
    w.run_stack()
    spec_c = w.project_1d(aperture="central")
    assert np.allclose(spec_c, w.stack_3d[1, 1])
    w.sigma_beam_ch = np.full(3, 1.2)  # 1.2 deg on 1 deg pixels -> radius 1.2 pix
    spec_g = w.project_1d(aperture="gaussian")
    assert spec_g.shape == (w.stack_3d.shape[-1],)
    assert w._default_gaussian_aperture_pixels() == pytest.approx(1.2, rel=1e-2)
    w.sigma_beam_ch = np.zeros(3)
    assert w._default_gaussian_aperture_pixels() == 1.0
    w.sigma_beam_ch = None
    assert w._default_gaussian_aperture_pixels() == 1.0


def test_read_rand_cat_and_from_power_spectrum(tmp_path):
    from astropy.table import Table
    from meer21cm import PowerSpectrum

    n_pix = 6
    wcs = create_wcs(180.0, 0.0, [n_pix, n_pix], 1.0)
    nu = np.linspace(700e6, 712e6, 4)
    sky_in = wcs.pixel_to_world(3, 3)
    sky_out = wcs.pixel_to_world(3, 3)

    f1 = tmp_path / "ran_0.fits"
    f2 = tmp_path / "ran_1.fits"
    Table(
        {
            "RA": [sky_in.ra.deg, 10.0],
            "DEC": [sky_in.dec.deg, -40.0],
            "Z": [freq_to_redshift(nu[1]), freq_to_redshift(nu[1])],
            "WEIGHT": [1.5, 2.0],
        }
    ).write(f1)
    Table(
        {
            "RA": [sky_out.ra.deg],
            "DEC": [sky_out.dec.deg],
            "Z": [freq_to_redshift(nu[2])],
            "WEIGHT": [3.0],
        }
    ).write(f2)

    ps = PowerSpectrum(
        data=np.ones((n_pix, n_pix, 4)),
        weights_map_pixel=np.ones((n_pix, n_pix, 4)),
        nu=nu,
        wproj=wcs,
        num_pix_x=n_pix,
        num_pix_y=n_pix,
        ra_range=(175.0, 185.0),
        dec_range=(-5.0, 5.0),
        tracer_bias_1=1.4,
        tracer_bias_2=2.2,
        sigma_v_1=250.0,
        sigma_v_2=100.0,
        filter_map_los=False,
        auto_set_radecnu_bounds=False,
        init_box_from_map_data=False,
    )
    ps._ra_gal = np.array([sky_in.ra.deg])
    ps._dec_gal = np.array([sky_in.dec.deg])
    ps._z_gal = np.array([freq_to_redshift(nu[1])])
    ps.weights_gal = np.array([1.0])
    ps.ra_rand = np.array([sky_in.ra.deg])
    ps.dec_rand = np.array([sky_in.dec.deg])
    ps.z_rand = np.array([freq_to_redshift(nu[1])])
    ps.weights_rand = np.array([1.0])

    worker = Stacking.from_power_spectrum(ps, stack_estimator="correlation")
    assert worker.tracer_bias_1 == pytest.approx(1.4)
    assert worker.tracer_bias_2 == pytest.approx(2.2)
    assert worker.ra_gal.size == 1
    assert worker.ra_rand.size == 1

    with pytest.raises(ValueError, match="at least one"):
        worker.read_rand_cat([])
    n_single = worker.read_rand_cat(f1, weight_col="WEIGHT", trim=True)
    assert n_single == 1
    assert worker.weights_rand[0] == pytest.approx(1.5)
    n_multi = worker.read_rand_cat([f1, f2], weight_col=None, trim=False)
    assert n_multi == 3
    assert worker.weights_rand is None


def test_transverse_window_helpers_and_validation():
    from meer21cm.los import azimuthal_pixel_window_power

    kperp = np.linspace(0.0, 0.5, 64)
    w0 = azimuthal_pixel_window_lag(kperp, 25.0, di=0, dj=0, n_phi=64, pixel_power=2)
    w_los = azimuthal_pixel_window_power(kperp, 25.0, n_phi=64)
    assert np.allclose(w0, w_los, rtol=1e-12, atol=1e-12)
    # Power 1 is larger than power 2 at intermediate k_perp
    w1 = azimuthal_pixel_window_lag(kperp, 25.0, di=0, dj=0, pixel_power=1)
    assert w1[10] > w0[10]
    w_frac = azimuthal_pixel_window_lag(kperp, 25.0, di=0, dj=0, pixel_power=1.5)
    assert w0[10] < w_frac[10] < w1[10]
    # Power 0 with lag 0 is identically 1
    assert np.allclose(azimuthal_pixel_window_lag(kperp, 25.0, pixel_power=0), 1.0)
    # Nonzero lag starts at 1 at k=0 and oscillates negative
    w_lag = azimuthal_pixel_window_lag(kperp, 25.0, di=1, dj=0, pixel_power=2)
    assert w_lag[0] == pytest.approx(1.0)
    assert np.min(w_lag) < 0.0

    # Continuous vs discrete beam window
    wb_c = beam_window_amplitude(kperp, 40.0)
    assert wb_c[0] == pytest.approx(1.0)
    assert np.allclose(wb_c, np.exp(-0.5 * (kperp * 40.0) ** 2))
    # For large sigma_pix = sigma_b / dperp, discrete DFT converges to continuous Gaussian
    wd_large = discrete_beam_window_amplitude(
        kperp[:20], 10.0, 40.0, di=0, dj=0, pixel_power=0
    )
    assert np.allclose(wd_large, wb_c[:20], rtol=1e-3, atol=1e-4)
    # Discrete beam with pixel_power=1, 1.5, 2 and zero sigma fallback
    assert np.allclose(
        discrete_beam_window_amplitude(kperp, 10.0, 0.0, pixel_power=2),
        azimuthal_pixel_window_lag(kperp, 10.0, pixel_power=2),
    )
    wd_p1 = discrete_beam_window_amplitude(kperp[:10], 20.0, 25.0, pixel_power=1)
    wd_pf = discrete_beam_window_amplitude(kperp[:10], 20.0, 25.0, pixel_power=1.5)
    wd_p2 = discrete_beam_window_amplitude(kperp[:10], 20.0, 25.0, pixel_power=2)
    assert wd_p2[5] < wd_pf[5] < wd_p1[5]

    with pytest.raises(ValueError, match="dperp"):
        azimuthal_pixel_window_lag(kperp, -1.0)
    with pytest.raises(ValueError, match="n_phi"):
        azimuthal_pixel_window_lag(kperp, 10.0, n_phi=2)
    with pytest.raises(ValueError, match="pixel_power"):
        azimuthal_pixel_window_lag(kperp, 10.0, pixel_power=-1.0)


def test_channel_covariance_exact_algebraic_identity_with_pca_and_masks():
    """Channel covariance contraction matches brute-force plane-wave Monte Carlo to <1e-12
    for non-separable 3D masks, jagged comoving bins, flagged channels, and PCA cleaning M.
    """
    rng = np.random.default_rng(42)
    nx, ny, n_ch = 5, 4, 18
    # Non-uniform comoving channel spacings
    dchi_ch = np.linspace(0.8, 1.4, n_ch)
    chi_ch = 1000.0 + np.cumsum(dchi_ch) - 0.5 * dchi_ch
    dchi_bin = 4.0
    _order, edges, _centres, bin_which = comoving_bin_indices(chi_ch, dchi_bin)
    n_bin = edges.size - 1

    # Non-separable 3D map weight (with flagged channels and spatial holes) & galaxy cube
    w_map = rng.uniform(0.5, 2.0, size=(nx, ny, n_ch))
    w_map[:, :, [3, 11]] = 0.0  # flagged RFI channels
    w_map[0, :, :6] = 0.0  # non-separable corner mask
    n_gal = rng.poisson(2.0, size=(nx, ny, n_ch)).astype(float)

    # Synthetic rank-3 PCA cleaning matrix M = I - V V^T
    v_raw = rng.normal(size=(n_ch, 3))
    q_mat, _ = np.linalg.qr(v_raw)
    clean_mat = np.eye(n_ch) - q_mat @ q_mat.T

    # Single-mode exact plane-wave test: C^ch[c1, c2] = sinc(k*dchi_1/2) sinc(k*dchi_2/2) cos(k*(chi1-chi2))
    k_test = 0.27
    # Build exact covariance from a known 2-point spectrum (analytical Gaussian P1D)
    kpar = np.linspace(0.0, 6.0, 601)
    p1d = 12.5 * np.exp(-0.5 * (kpar / 0.35) ** 2)
    cov_obs = channel_cross_covariance(
        chi_ch, dchi_ch, kpar, p1d, clean_matrix=clean_mat, dchi_gal=dchi_ch
    )

    # Check exact identity against explicit double-sum over binned cubes at lag (di=1, dj=-1):
    # E[(w_L L_clean) ★ N_g](di, dj, db) = sum_{b} sum_{c1 in b+db, c2 in b} W[c1, c2] C_obs[c1, c2]
    di, dj = 1, -1
    pair_w = Stacking._pair_weight_matrix_for_lag(w_map, n_gal, di, dj, separable=False)
    for weighting in ("conventional", "quadratic"):
        for sym in (False, True):
            prof, den = contract_channel_covariance(
                cov_obs,
                pair_w,
                bin_which=bin_which,
                n_bin=n_bin,
                weighting=weighting,
                symmetrize=sym,
            )
            # Verify pair-count denominator matches FFT linear_correlation of rebinned cubes
            rebinned_w, _ = rebin_cube(w_map, chi_ch, dchi_bin)
            rebinned_n, _ = rebin_cube(n_gal, chi_ch, dchi_bin)
            _d, _f, fft_den, _a = data_and_fkp_cubelets(
                rebinned_w,
                rebinned_w,
                rebinned_n,
                2,
                weighting=weighting,
                symmetrize=sym,
            )
            assert np.allclose(den, fft_den[2 + di, 2 + dj], rtol=1e-11, atol=1e-11)

            # Verify numerator via explicit Cholesky/eigendecomposition expectation (exact trace)
            # For random field with E[L_c1 g_c2] = cov_obs[c1, c2], check explicit binned contraction
            s_mat = np.zeros((n_bin, n_ch))
            s_mat[bin_which, np.arange(n_ch)] = 1.0
            n_bin_mat = s_mat @ (pair_w * cov_obs) @ s_mat.T
            d_bin_mat = s_mat @ pair_w @ s_mat.T
            lags = np.arange(-(n_bin - 1), n_bin)
            n_exp = np.array([np.trace(n_bin_mat, offset=-int(db)) for db in lags])
            d_exp = np.array([np.trace(d_bin_mat, offset=-int(db)) for db in lags])
            if sym:
                n_exp = n_exp + n_exp[::-1]
                d_exp = d_exp + d_exp[::-1]
            if weighting == "conventional":
                expected = np.where(d_exp > 1e-8 * d_exp.max(), n_exp / d_exp, 0.0)
            else:
                expected = n_exp / d_exp[n_bin - 1]
            assert np.allclose(prof, expected, rtol=1e-13, atol=1e-13)

    # Also test native (bin_which=None) + dchi_gal=0 and scalar/array validation
    cov_raw = channel_cross_covariance(chi_ch, dchi_ch, kpar, p1d, dchi_gal=dchi_ch)
    cov_pt = channel_cross_covariance(chi_ch, dchi_ch, kpar, p1d, dchi_gal=0.0)
    assert cov_pt[0, 0] > cov_raw[0, 0]
    prof_nat, _ = contract_channel_covariance(
        cov_pt, np.zeros_like(pair_w), weighting="quadratic"
    )
    assert np.all(prof_nat == 0.0)

    # Validation errors on helpers
    with pytest.raises(ValueError, match="kpar"):
        channel_cross_covariance(chi_ch, dchi_ch, [1.0, 0.5], [1.0, 1.0])
    with pytest.raises(ValueError, match="chi_ch"):
        channel_cross_covariance(chi_ch, dchi_ch[:-1], kpar, p1d)
    with pytest.raises(ValueError, match="dchi_gal"):
        channel_cross_covariance(chi_ch, dchi_ch, kpar, p1d, dchi_gal=dchi_ch[:-1])
    with pytest.raises(ValueError, match="p1d_kpar"):
        channel_cross_covariance(chi_ch, dchi_ch, kpar, p1d[:-1])
    with pytest.raises(ValueError, match="clean_matrix"):
        channel_cross_covariance(chi_ch, dchi_ch, kpar, p1d, clean_matrix=np.eye(2))
    with pytest.raises(ValueError, match="weighting"):
        contract_channel_covariance(cov_obs, pair_w, weighting="bad")
    with pytest.raises(ValueError, match="square"):
        contract_channel_covariance(np.ones((2, 3)), np.ones((2, 3)))
    with pytest.raises(ValueError, match="bin_which"):
        contract_channel_covariance(cov_obs, pair_w, bin_which=np.zeros(2, dtype=int))


def test_stacking_model_closed_form_gaussian_pk_and_state_preservation():
    """Verify model_1d_stack and model_3d_stack against closed-form analytical
    3D Gaussian P(k) to < 1e-4 relative accuracy, and verify kmode/mumode are never mutated.
    """
    n_pix = 5
    n_ch = 25
    wcs = create_wcs(180.0, 0.0, [n_pix, n_pix], 0.5)
    nu = 700e6 + np.arange(n_ch) * 0.05e6  # ~0.35 Mpc narrow channels
    sigma_k = 0.06  # Mpc^-1; P(k) = A * exp(-k^2 / (2 sigma_k^2))
    amp = 1e4  # K Mpc^3
    # Analytical 3D correlation function:
    # xi(r) = A * (sigma_k / sqrt(2 pi))^3 * exp(-0.5 * sigma_k^2 * r^2)
    xi_0_exact = amp * (sigma_k / np.sqrt(2.0 * np.pi)) ** 3

    worker = Stacking(
        data=np.zeros((n_pix, n_pix, n_ch)),
        weights_map_pixel=np.ones((n_pix, n_pix, n_ch)),
        nu=nu,
        wproj=wcs,
        num_pix_x=n_pix,
        num_pix_y=n_pix,
        stack_estimator="loop",
        stack_angular_num_nearby_pix=2,
        include_pixel_window=False,
        include_beam_window=False,
        kperp_max=0.6,
        n_kperp=512,
        power_kmu_func=lambda k, mu: amp * np.exp(-0.5 * (k / sigma_k) ** 2),
        filter_map_los=False,
        auto_set_radecnu_bounds=False,
        init_box_from_map_data=False,
    )
    # Point-like channel (dchi_gal=0) with narrow native channels -> exact xi(s_par)
    chi = worker.chi_ch
    dchi_exact = worker.dchi_ch
    assert dchi_exact.shape == (n_ch,)
    # Test model_1d_power directly
    kpar_test = np.array([0.0, 0.05, 0.1])
    p1d_exact = (
        amp * (sigma_k**2 / (2.0 * np.pi)) * np.exp(-0.5 * (kpar_test / sigma_k) ** 2)
    )
    p1d_num = worker.model_1d_power(kpar_test)
    assert np.allclose(p1d_num, p1d_exact, rtol=1e-4)

    # Full 3D model and 1D central model must agree identically and match exact xi(0)
    # Use very thin channels via dchi_gal=0 and compare with channel-averaged analytical integral
    mod_3d = worker.model_3d_stack(dchi_gal=0.0, kpar_max=1.5, n_kpar_eval=128)
    mod_1d = worker.model_1d_stack(
        aperture="central", dchi_gal=0.0, kpar_max=1.5, n_kpar_eval=128
    )
    assert mod_3d.shape == (5, 5, 2 * n_ch - 1)
    assert np.allclose(mod_3d[2, 2], mod_1d, rtol=1e-12)
    # At zero lag, narrow channels (~0.35 Mpc) match exact xi_0 to < 1e-4
    assert mod_1d[n_ch - 1] == pytest.approx(xi_0_exact, rel=1e-4)
    # Off-centre pixel lag (1, 0) decays as exp(-0.5 * sigma_k^2 * D_perp^2)
    dperp_mid = (
        np.sqrt(float(worker.pixel_area)) * (np.pi / 180.0) * float(np.mean(chi))
    )
    ratio_expected = np.exp(-0.5 * (sigma_k * dperp_mid) ** 2)
    assert mod_3d[3, 2, n_ch - 1] / mod_3d[2, 2, n_ch - 1] == pytest.approx(
        ratio_expected, rel=2e-2
    )
    # Tophat & Gaussian aperture 1D models match project_1d(mod_3d) to machine precision
    for ap, rad in [("tophat", 1.0), ("gaussian", 1.2)]:
        m1 = worker.model_1d_stack(
            aperture=ap, radius=rad, dchi_gal=0.0, kpar_max=1.5, n_kpar_eval=128
        )
        m1_proj = worker.project_1d(mod_3d, aperture=ap, radius=rad)
        assert np.allclose(m1, m1_proj, rtol=1e-12, atol=1e-12)


def test_stacking_model_cosmological_kmu_and_options():
    """Verify model_1d_stack / model_3d_stack with real ModelPowerSpectrum.power_kmu,
    beam modes, PCA clean_matrix, correlation rebinning, and state preservation."""
    n_pix = 5
    n_ch = 16
    wcs = create_wcs(145.0, -1.7, [n_pix, n_pix], 0.5)
    nu = 700e6 + np.arange(n_ch) * 0.2e6
    sky_g = wcs.pixel_to_world(2, 2)
    sky_r = wcs.pixel_to_world(3, 2)

    worker = Stacking(
        data=np.zeros((n_pix, n_pix, n_ch)),
        weights_map_pixel=np.ones((n_pix, n_pix, n_ch)),
        nu=nu,
        wproj=wcs,
        num_pix_x=n_pix,
        num_pix_y=n_pix,
        ra_gal=np.array([sky_g.ra.deg]),
        dec_gal=np.array([sky_g.dec.deg]),
        z_gal=np.array([freq_to_redshift(nu[8])]),
        ra_rand=np.array([sky_r.ra.deg]),
        dec_rand=np.array([sky_r.dec.deg]),
        z_rand=np.array([freq_to_redshift(nu[8])]),
        tracer_bias_1=1.47,
        tracer_bias_2=2.30,
        sigma_v_1=250.0,
        sigma_v_2=0.0,
        mean_amp_1="average_hi_temp",
        mean_amp_2=1.0,
        omega_hi=0.0008,
        kaiser_rsd=True,
        sigma_beam_ch=np.full(n_ch, 0.7),
        stack_estimator="correlation",
        stack_angular_num_nearby_pix=1,
        dchi_mpc=4.0,
        n_kperp=64,
        n_chi=3,
        filter_map_los=False,
        auto_set_radecnu_bounds=False,
        init_box_from_map_data=False,
    )
    orig_kmode = None if worker._kmode is None else np.array(worker._kmode, copy=True)
    orig_mumode = (
        None if worker._mumode is None else np.array(worker._mumode, copy=True)
    )

    # Continuous beam + randoms
    spec_cont = worker.model_1d_stack(aperture="central", n_kpar_eval=64)
    # kmode / mumode must be completely untouched
    if orig_kmode is None:
        assert worker._kmode is None
    else:
        assert np.array_equal(worker._kmode, orig_kmode)
    if orig_mumode is None:
        assert worker._mumode is None
    else:
        assert np.array_equal(worker._mumode, orig_mumode)
    assert np.isfinite(spec_cont).all()
    c_mid = spec_cont.size // 2
    assert spec_cont[c_mid] > 0.0

    # Discrete beam mode, Gaussian default aperture, separable weights, shot noise, PCA matrix
    spec_disc = worker.model_1d_stack(
        aperture="gaussian",
        beam_mode="discrete",
        separable_weights=True,
        cross_shot_noise=lambda k: np.zeros_like(k),
        n_kpar_eval=64,
    )
    assert spec_disc.shape == spec_cont.shape
    assert np.isfinite(spec_disc).all()

    # Mean-subtraction cleaning matrix M = I - 11^T / N_ch lowers zero-lag signal
    m_mean = np.eye(n_ch) - np.ones((n_ch, n_ch)) / n_ch
    spec_clean = worker.model_1d_stack(
        aperture="central", clean_matrix=m_mean, n_kpar_eval=64
    )
    assert spec_clean[c_mid] < spec_cont[c_mid]

    # Scalar cross_shot_noise + explicit dchi_gal array + galaxy-only fallback + W_HI fallback
    spec_sn = worker.model_1d_stack(
        aperture="central",
        cross_shot_noise=1e-4,
        dchi_gal=worker.dchi_ch,
        n_kpar_eval=64,
    )
    assert spec_sn[c_mid] > spec_cont[c_mid]
    p1d_sn = worker.model_1d_power([0.05, 0.1], cross_shot_noise=1e-4)
    p1d_call = worker.model_1d_power(
        [0.05, 0.1], cross_shot_noise=lambda k: np.full_like(k, 1e-4)
    )
    assert np.allclose(p1d_sn, p1d_call)

    # Exercise galaxy-only pair weights (no randoms) and dchi_gal=None
    worker.ra_rand = None
    worker.dec_rand = None
    worker.z_rand = None
    spec_gal_only = worker.model_1d_stack(
        aperture="central", dchi_gal=None, n_kpar_eval=64
    )
    assert np.isfinite(spec_gal_only).all()

    # Scalar sigma_beam on multi-channel worker + kperp / dk_perp properties
    worker.sigma_beam_ch = np.array([0.7])
    assert np.isfinite(worker.transverse_window()).all()
    assert worker.kperp.shape == (64,)
    assert worker.dk_perp.shape == (64,)

    # Edge cases: single channel, lag > nx, data=None, w_HI=None, W_HI=None
    w_1ch = Stacking(
        nu=np.array([700e6]),
        wproj=wcs,
        num_pix_x=2,
        num_pix_y=2,
        tracer_bias_1=1.0,
        tracer_bias_2=1.5,
        sigma_beam_ch=np.array([0.7]),
        stack_estimator="correlation",
        stack_angular_num_nearby_pix=3,
        n_kperp=32,
        n_chi=1,
        filter_map_los=False,
        auto_set_radecnu_bounds=False,
        init_box_from_map_data=False,
    )
    w_1ch._data = None
    w_1ch._weights_map_pixel = None
    w_1ch._map_has_sampling = None
    assert w_1ch.dchi_ch.size == 1
    assert w_1ch.model_3d_stack(n_kpar_eval=32).shape == (7, 7, 1)
    w_1ch.W_HI = np.ones((2, 2, 1), dtype=bool)
    assert w_1ch.model_1d_stack(aperture="central", n_kpar_eval=32).shape == (1,)
    assert w_1ch._chi_nodes(channel_weights=np.zeros(1))[1][0] == 1.0
    assert worker._chi_nodes(channel_weights=np.zeros(n_ch))[1].sum() == pytest.approx(
        1.0
    )

    # Validation errors on Stacking model parameters
    with pytest.raises(ValueError, match="n_chi"):
        Stacking(n_chi=0)
    with pytest.raises(ValueError, match="n_kperp"):
        Stacking(n_kperp=4)
    with pytest.raises(ValueError, match="beam_mode"):
        Stacking(beam_mode="airy")
    with pytest.raises(ValueError, match="kperp_edges"):
        Stacking(kperp_edges=[1.0])
    with pytest.raises(ValueError, match="strictly increasing"):
        w_bad = Stacking(
            nu=nu,
            wproj=wcs,
            num_pix_x=2,
            num_pix_y=2,
            kperp_edges=[0.2, 0.1],
            filter_map_los=False,
            auto_set_radecnu_bounds=False,
            init_box_from_map_data=False,
        )
        _ = w_bad.kperp_edges
    with pytest.raises(ValueError, match="kperp_max"):
        w_bad2 = Stacking(
            nu=nu,
            wproj=wcs,
            num_pix_x=2,
            num_pix_y=2,
            kperp_max=-1.0,
            filter_map_los=False,
            auto_set_radecnu_bounds=False,
            init_box_from_map_data=False,
        )
        _ = w_bad2.kperp_edges
    with pytest.raises(ValueError, match="n_chi"):
        worker._chi_nodes(n_chi=0)
    with pytest.raises(ValueError, match="beam_mode"):
        worker.transverse_window(beam_mode="bad")
    with pytest.raises(ValueError, match="kpar"):
        worker.model_1d_power([])
    with pytest.raises(ValueError, match="kpar_max"):
        worker.model_1d_stack(kpar_max=-1.0)
    with pytest.raises(ValueError, match="dchi_gal"):
        worker.model_1d_stack(dchi_gal="invalid")
    worker.power_kmu_func = lambda k, mu: np.zeros(1)
    with pytest.raises(ValueError, match="power_kmu_func"):
        worker.model_1d_power([0.1])


def test_quadratic_model_uses_global_zero_lag_q0():
    """Quadratic forward model divides off-centre lags by global Q0 = D(0,0,0),
    matching data_and_fkp_cubelets and stack_cubelet on asymmetric masks.
    """
    n_pix = 5
    n_ch = 8
    wcs = create_wcs(180.0, 0.0, [n_pix, n_pix], 0.5)
    nu = np.linspace(700e6, 714e6, n_ch)
    # Strongly non-uniform mask so D(di, dj, 0) != D(0, 0, 0) at off-centre lags
    w_map = np.zeros((n_pix, n_pix, n_ch), dtype=float)
    w_map[1:4, 1:4, :] = 2.0
    w_map[2, 2, :] = 5.0
    data = np.ones((n_pix, n_pix, n_ch), dtype=float)

    # Constant P(k, mu) via custom power_kmu_func for fast deterministic evaluation
    p_const = lambda k, mu: np.exp(-0.5 * (k / 0.15) ** 2)

    for estimator in ("correlation", "loop"):
        for sym in (False, True):
            w_conv = Stacking(
                data=data,
                weights_map_pixel=w_map,
                nu=nu,
                wproj=wcs,
                num_pix_x=n_pix,
                num_pix_y=n_pix,
                stack_estimator=estimator,
                stack_weighting="conventional",
                symmetrize=sym,
                stack_angular_num_nearby_pix=2,
                dchi_mpc=8.0,
                n_kperp=48,
                n_chi=2,
                power_kmu_func=p_const,
                filter_map_los=False,
                auto_set_radecnu_bounds=False,
                init_box_from_map_data=False,
            )
            w_quad = Stacking(
                data=data,
                weights_map_pixel=w_map,
                nu=nu,
                wproj=wcs,
                num_pix_x=n_pix,
                num_pix_y=n_pix,
                stack_estimator=estimator,
                stack_weighting="quadratic",
                symmetrize=sym,
                stack_angular_num_nearby_pix=2,
                dchi_mpc=8.0,
                n_kperp=48,
                n_chi=2,
                power_kmu_func=p_const,
                filter_map_los=False,
                auto_set_radecnu_bounds=False,
                init_box_from_map_data=False,
            )
            m_conv = w_conv.model_3d_stack(n_kpar_eval=48)
            m_quad = w_quad.model_3d_stack(n_kpar_eval=48)

            # Compute exact pair-count cubelet for the same weights & selection mask
            use_corr = estimator == "correlation"
            w_native, n_native = w_quad._pair_weight_cubes(sort_chi=False)
            if use_corr:
                chi = w_quad.channel_chi_mpc()
                w_b, _ = rebin_cube(w_native, chi, w_quad.dchi_mpc)
                n_b, _ = rebin_cube(n_native, chi, w_quad.dchi_mpc)
            else:
                w_b, n_b = w_native, n_native
            _d, _f, den_cube, _a = data_and_fkp_cubelets(
                w_b, w_b, n_b, 2, weighting="quadratic", symmetrize=sym
            )
            mid = 2
            c0 = den_cube.shape[-1] // 2
            q0_global = float(den_cube[mid, mid, c0])
            # Exact algebraic relation at every (di, dj, db):
            # I_quad(s) = I_conv(s) * D(s) / Q0_global
            expected_quad = m_conv * (den_cube / q0_global)
            assert np.allclose(m_quad, expected_quad, rtol=1e-12, atol=1e-15)
            # Confirm off-centre lag denominator actually differed from Q0_global
            assert den_cube[mid + 2, mid + 2, c0] < 0.5 * q0_global
