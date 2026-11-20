import numpy as np
from meer21cm.stack import (
    Stacking,
    correlate_cubelet,
    data_and_fkp_cubelets,
    paint_sky_counts,
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


def test_unknown_stack_estimator_and_model_stub():
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
    with pytest.raises(NotImplementedError):
        worker.model_1d_stack()
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
    worker.stack_weighting = "quadratic"
    with pytest.raises(ValueError):
        worker.run_stack()
    worker.stack_weighting = "conventional"
    worker.symmetrize = True
    with pytest.raises(ValueError):
        worker.run_stack()
    worker.symmetrize = False
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
