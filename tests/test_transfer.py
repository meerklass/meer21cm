import multiprocessing as mp

mp.set_start_method("spawn", force=True)
import numpy as np
from meer21cm.transfer import analytic_transfer_function, pca_clean, get_pca_matrix
from meer21cm import MockSimulation
from meer21cm.transfer import (
    TransferFunction,
    run_tf_calculation_auto,
    run_tf_calculation_cross,
    run_null_test,
)
import pytest
from meer21cm.util import create_wcs


def test_analytic_transfer_function():
    mock = MockSimulation(
        survey="meerklass_2021",
        band="L",
        mean_amp_1="average_hi_temp",
        omega_hi=5e-4,
        k1dbins=np.linspace(0.05, 1.5, 11),
    )
    mock.W_HI = np.ones_like(mock.W_HI)
    mock.w_HI = np.ones_like(mock.w_HI)
    mock.use_flat_sky_box()
    mock.propagate_field_k_to_model()
    mock.W_HI = np.ones_like(mock.W_HI)
    mock.w_HI = np.ones_like(mock.w_HI)
    mock.use_flat_sky_box()
    mock.propagate_field_k_to_model()
    mock.field_1 = mock.mock_tracer_field_1
    porig_1d, _, _ = mock.get_1d_power(mock.auto_power_3d_1)
    fg = np.ones_like(mock.field_1) * 20
    fg *= ((mock.nu / 408 / 1e6) ** (-2.7))[None, None, :]
    res_map, A_mat = pca_clean(fg + mock.field_1, 3, return_A=True)
    R_mat = np.eye(mock.nu.size) - A_mat @ A_mat.T
    tf, wab = analytic_transfer_function(R_mat)
    mock.field_1 = res_map
    pdata_1d, _, _ = mock.get_1d_power(mock.auto_power_3d_1 * tf[None, None, :])
    pnotf_1d, _, _ = mock.get_1d_power(mock.auto_power_3d_1)
    no_tf_deviation = np.abs(pnotf_1d - porig_1d) / porig_1d
    tf_deviation = np.abs(pdata_1d - porig_1d) / porig_1d
    assert tf_deviation.mean() < 5e-2
    assert no_tf_deviation.mean() > tf_deviation.mean()
    assert np.diagonal(wab)[0] < 0.1
    assert np.abs(np.diagonal(wab)[6:].mean() - 1) < 0.01


def test_numerical_transfer_function():
    wcs = create_wcs(0.0, 0.0, [10, 10], [1.0, 1.0])
    mock = MockSimulation(
        survey="meerklass_2021",
        band="L",
    )
    mock = MockSimulation(
        nu=mock.nu[:50],
        wproj=wcs,
        num_pix_x=10,
        num_pix_y=10,
        mean_amp_1="average_hi_temp",
        omega_hi=5e-4,
        k1dbins=np.linspace(0.2, 3.0, 5),
        flat_sky=True,
        tracer_bias_2=1.0,
        num_discrete_source=10000,
    )
    mock.W_HI = np.ones_like(mock.W_HI)
    mock.w_HI = np.ones_like(mock.w_HI)
    mock.use_flat_sky_box()
    mock.propagate_field_k_to_model()
    mock.k1dweights = np.ones_like(mock.kmode)
    mock.k1dweights[:, :, :2] = 0.0
    mock.field_1 = mock.mock_tracer_field_1
    porig_1d, _, _ = mock.get_1d_power(mock.auto_power_3d_1)
    fg = np.ones_like(mock.field_1) * 20
    fg *= ((mock.nu / 408 / 1e6) ** (-2.7))[None, None, :]
    res_map, A_mat = pca_clean(fg + mock.field_1, 3, return_A=True, mean_center=True)
    R_mat = np.eye(mock.nu.size) - A_mat @ A_mat.T
    R_mat_test = get_pca_matrix(
        fg + mock.field_1, 3, weights=np.ones_like(mock.field_1), mean_center_map=True
    )
    assert np.allclose(R_mat, R_mat_test)
    mock.field_1 = res_map
    pnotf_1d, _, _ = mock.get_1d_power(mock.auto_power_3d_1)
    tf_true = pnotf_1d / porig_1d
    mock.data = fg + mock.field_1
    tf = TransferFunction(
        mock,
        N_fg=3,
        highres_sim=1,
        upres_transverse=1,
        upres_radial=1,
        num_process=1,
    )
    # test pool invoke
    results_arr = tf.run(
        range(1), type="auto", return_power_3d=True, return_power_1d=True
    )
    results_arr = tf.run(
        range(1), type="cross", return_power_3d=True, return_power_1d=True
    )
    tf.pool = "mpi"
    results_arr = tf.run(range(1), type="null", return_power_3d=True)
    with pytest.raises(ValueError):
        tf.run(range(10), type="test", return_power_3d=False, return_power_1d=False)
    tf.pool = "test"
    with pytest.raises(ValueError):
        tf.run(range(10), type="auto", return_power_3d=False, return_power_1d=False)


def test_transfer_function_value():
    wcs = create_wcs(0.0, 0.0, [10, 10], [1.0, 1.0])
    mock = MockSimulation(
        survey="meerklass_2021",
        band="L",
    )
    mock = MockSimulation(
        nu=mock.nu[:50],
        wproj=wcs,
        num_pix_x=10,
        num_pix_y=10,
        mean_amp_1="average_hi_temp",
        omega_hi=5e-4,
        k1dbins=np.linspace(0.2, 3.0, 5),
        flat_sky=True,
        tracer_bias_2=1.0,
        num_discrete_source=10000,
    )
    mock.W_HI = np.ones_like(mock.W_HI)
    mock.w_HI = np.ones_like(mock.w_HI)
    mock.use_flat_sky_box()
    mock.propagate_field_k_to_model()
    mock.k1dweights = np.ones_like(mock.kmode)
    mock.k1dweights[:, :, :2] = 0.0
    mock.field_1 = mock.mock_tracer_field_1
    porig_1d, _, _ = mock.get_1d_power(mock.auto_power_3d_1)
    fg = np.ones_like(mock.field_1) * 20
    fg *= ((mock.nu / 408 / 1e6) ** (-2.7))[None, None, :]
    res_map, A_mat = pca_clean(fg + mock.field_1, 3, return_A=True, mean_center=True)
    R_mat = np.eye(mock.nu.size) - A_mat @ A_mat.T
    R_mat_test = get_pca_matrix(
        fg + mock.field_1, 3, weights=np.ones_like(mock.field_1), mean_center_map=True
    )
    assert np.allclose(R_mat, R_mat_test)
    mock.field_1 = res_map
    pnotf_1d, _, _ = mock.get_1d_power(mock.auto_power_3d_1)
    tf_true = pnotf_1d / porig_1d
    mock.data = fg + mock.field_1
    tf = TransferFunction(
        mock,
        N_fg=3,
        highres_sim=1,
        upres_transverse=1,
        upres_radial=1,
        num_process=1,
    )
    arglist = tf.get_arg_list_for_parallel_auto(
        range(10), return_power_3d=True, return_power_1d=True
    )
    results_arr = []
    for arg in arglist:
        results_arr.append(run_tf_calculation_auto(*arg))
    results_arr = np.array([results_arr[i][0] for i in range(10)])
    tf_1d = np.median(results_arr, axis=0)
    assert np.abs(tf_true - tf_1d).mean() < 1e-1
    mock.propagate_mock_tracer_to_gal_cat()
    arglist = tf.get_arg_list_for_parallel_cross(
        range(10), return_power_3d=True, return_power_1d=True
    )
    results_arr = []
    for arg in arglist:
        results_arr.append(run_tf_calculation_cross(*arg))
    results_arr = np.array([results_arr[i][0] for i in range(10)])
    tf_1d = np.median(results_arr, axis=0)
    assert np.abs(tf_true - tf_1d).mean() < 1e-1
    arglist = tf.get_arg_list_for_parallel_null(range(10), return_power_3d=True)
    results_arr = []
    for arg in arglist:
        results_arr.append(run_null_test(*arg))
    results_arr = np.array([results_arr[i][0] for i in range(10)])
    null_ps_1d = np.median(results_arr, axis=0)
    # note null is cross, so missing one temperature unit
    avg = (mock.average_hi_temp * null_ps_1d / porig_1d).mean()
    assert np.abs(avg) < 6e-2
