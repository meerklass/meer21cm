import inspect
import multiprocessing as mp
import os
import pickle

mp.set_start_method("spawn", force=True)
import numpy as np
from meer21cm.transfer import analytic_transfer_function, pca_clean, get_pca_matrix
from meer21cm import MockSimulation
from meer21cm.power import PowerSpectrum
from meer21cm.transfer import (
    TransferFunction,
    apply_cleaning_matrix,
    _multipole_ells,
    load_precomputed_mock,
    run_tf_calculation_auto,
    run_tf_calculation_cross,
    run_null_test,
)
import pytest
from meer21cm.util import create_wcs, freq_to_redshift


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
    dndz_arr = [np.linspace(0, 1, 100), np.ones_like(np.linspace(0, 1, 100))]
    tf = TransferFunction(
        mock,
        N_fg=3,
        highres_sim=1,
        upres_transverse=1,
        upres_radial=1,
        num_process=1,
        unmask_during_mock=True,
        discrete_source_dndz=dndz_arr,
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


def _tiny_transfer_ps():
    nu = np.linspace(900e6, 1000e6, 6)
    mock = MockSimulation(
        nu=nu,
        wproj=create_wcs(0.0, 0.0, [4, 4], [1.0, 1.0]),
        num_pix_x=4,
        num_pix_y=4,
        flat_sky=True,
        seed=0,
        num_discrete_source=10,
    )
    mock.field_1 = np.zeros(mock.data.shape)
    mock.use_flat_sky_box()
    return mock


def test_transfer_worker_argument_contract():
    ps = _tiny_transfer_ps()
    tf = TransferFunction(
        ps,
        N_fg=2,
        highres_sim=1,
        upres_transverse=1,
        upres_radial=1,
        num_process=1,
    )
    workers = {
        "auto": (tf.get_arg_list_for_parallel_auto, run_tf_calculation_auto),
        "cross": (tf.get_arg_list_for_parallel_cross, run_tf_calculation_cross),
        "null": (tf.get_arg_list_for_parallel_null, run_null_test),
    }
    for arg_func, worker in workers.values():
        arg = arg_func([0])[0]
        signature = inspect.signature(worker)
        signature.bind(*arg)
        names = list(signature.parameters)
        if "project_combined" in names:
            signature.bind(*arg[: names.index("project_combined")])
        assert pickle.dumps(arg)
        assert not any(isinstance(item, PowerSpectrum) for item in arg)
    tf.convolve_loaded_mock = True
    auto_arg = tf.get_arg_list_for_parallel_auto([1])[0]
    assert auto_arg[-4:] == (False, None, None, True)


def test_apply_cleaning_matrix_projects_selected_cube():
    rng = np.random.default_rng(0)
    mock_data = rng.normal(size=(4, 4, 6))
    uncleaned = rng.normal(size=(4, 4, 6))
    cleaning = np.eye(6)
    cleaning[0, :] = 0.0
    weights = np.ones_like(mock_data)
    cleaned_mock = apply_cleaning_matrix(
        mock_data, uncleaned, cleaning, weights, False, False
    )
    cleaned_both = apply_cleaning_matrix(
        mock_data, uncleaned, cleaning, weights, False, True
    )
    assert np.allclose(cleaned_mock, np.einsum("ij,abj->abi", cleaning, mock_data))
    assert np.allclose(
        cleaned_both, np.einsum("ij,abj->abi", cleaning, uncleaned + mock_data)
    )


def _planted_transfer_ps():
    wcs = create_wcs(0.0, 0.0, [10, 10], [1.0, 1.0])
    survey = MockSimulation(survey="meerklass_2021", band="L")
    nu = survey.nu[:20]
    mock = MockSimulation(
        nu=nu,
        wproj=wcs,
        num_pix_x=10,
        num_pix_y=10,
        flat_sky=True,
        seed=0,
        num_discrete_source=100,
        k1dbins=np.linspace(0.2, 3.0, 5),
        tracer_bias_2=1.0,
        mean_amp_1="average_hi_temp",
        omega_hi=5e-4,
    )
    mock.W_HI = np.ones_like(mock.W_HI)
    mock.w_HI = np.ones_like(mock.w_HI)
    mock.use_flat_sky_box()
    mock.nu_min = float(mock.nu.min()) - 1.0
    mock.nu_max = float(mock.nu.max()) + 1.0
    foreground = 20.0 * (mock.nu / mock.nu.mean()) ** -2.7
    mock.data = np.broadcast_to(foreground, mock.data.shape).copy()
    return mock


def _save_wide_mock(ps, prefix):
    extra = float(ps.nu.max()) + 50e6
    wide_nu = np.concatenate([np.asarray(ps.nu, dtype=float), [extra]])
    phase = np.linspace(0.0, 4.0 * np.pi, ps.nu.size)
    x_grad = np.linspace(-1.0, 1.0, ps.num_pix_x)[:, None, None]
    y_grad = np.linspace(-1.0, 1.0, ps.num_pix_y)[None, :, None]
    pattern = (x_grad + y_grad) * np.cos(phase)[None, None, :]
    pattern[2:8, 2:8, :] += 1.0
    wide = np.concatenate([pattern, np.full((10, 10, 1), 99.0)], axis=-1)
    saver = MockSimulation(
        nu=wide_nu,
        wproj=ps.wproj,
        num_pix_x=10,
        num_pix_y=10,
        flat_sky=True,
        seed=0,
    )
    saver.data = wide
    saver._ra_gal = np.array([0.2, -0.4])
    saver._dec_gal = np.array([0.1, -0.2])
    saver._z_gal = np.full(2, freq_to_redshift(ps.nu[ps.nu.size // 2]))
    saver.save_mock_products(prefix, seed=0)
    return pattern


def test_loaded_mock_frequency_cut_and_multipoles(tmp_path):
    ps = _planted_transfer_ps()
    prefix = str(tmp_path) + os.sep
    pattern = _save_wide_mock(ps, prefix)
    loaded = MockSimulation(
        nu=np.asarray(ps.nu, dtype=float),
        nu_min=ps.nu_min,
        nu_max=ps.nu_max,
        wproj=ps.wproj,
        num_pix_x=10,
        num_pix_y=10,
        flat_sky=True,
        seed=0,
    )
    load_precomputed_mock(
        loaded, prefix, read_galaxies=True, convolve_loaded_mock=False
    )
    assert loaded.nu.size == ps.nu.size
    assert loaded.data.shape == pattern.shape
    assert np.allclose(loaded.data, pattern)
    assert loaded.ra_gal.size == 2
    ramp = np.linspace(0.2, 1.0, ps.nu.size)
    tf = TransferFunction(
        ps,
        N_fg=2,
        highres_sim=1,
        upres_transverse=1,
        upres_radial=1,
        num_process=1,
        mock_directory=prefix,
        mean_center_map=False,
        R_mat=np.diag(ramp),
    )

    def run_worker(kind, **kwargs):
        # Call the worker in this process. tf.run() would hide these lines
        # from the coverage file uploaded by GitHub Actions.
        worker = {
            "cross": run_tf_calculation_cross,
            "auto": run_tf_calculation_auto,
        }[kind]
        arg_func = getattr(tf, f"get_arg_list_for_parallel_{kind}")
        return worker(*arg_func([0], **kwargs)[0])

    monopole = run_worker("cross", return_power_1d=True)
    both = run_worker("cross", multipole_ells=[0, 2], return_power_1d=True)
    assert np.isfinite(monopole[0]).all()
    assert monopole[0].ndim == 1
    assert both[0].shape == (2, monopole[0].size)
    assert np.allclose(both[0][0], monopole[0])
    assert np.isfinite(both[0][1]).all()
    assert not np.allclose(both[0][1], both[0][0])
    auto_default = run_worker("auto", return_power_1d=True)
    auto_ell = run_worker("auto", multipole_ells=[0, 2], return_power_1d=True)
    assert np.allclose(auto_ell[0][0], auto_default[0])
    assert auto_ell[0].shape == (2, auto_default[0].size)
    assert np.isfinite(auto_ell[0][1]).all()
    before_power = auto_ell[1]
    assert before_power.shape == auto_ell[0].shape
    assert not np.allclose(before_power[0], before_power[1])
    os.remove(prefix + "gal_mock_0.fits")
    auto_no_cat = run_worker("auto")
    assert np.isfinite(auto_no_cat[0]).all()


def _high_k_power(cube):
    channel = cube[:, :, cube.shape[-1] // 2]
    spectrum = np.fft.fftshift(np.fft.fft2(channel))
    yy, xx = np.indices(spectrum.shape)
    center = (np.array(spectrum.shape) - 1) / 2.0
    radius = np.sqrt((yy - center[0]) ** 2 + (xx - center[1]) ** 2)
    return np.sum(np.abs(spectrum[radius >= 1.5]) ** 2)


def test_loaded_mock_beam_flag(tmp_path):
    nu = np.linspace(900e6, 1000e6, 4)
    wproj = create_wcs(0.0, 0.0, [8, 8], [1.0, 1.0])
    cube = np.zeros((8, 8, nu.size))
    cube[4, 4, :] = 1.0
    saver = MockSimulation(
        nu=nu,
        wproj=wproj,
        num_pix_x=8,
        num_pix_y=8,
        flat_sky=True,
        seed=0,
    )
    saver.data = cube
    saver._ra_gal = np.array([0.0])
    saver._dec_gal = np.array([0.0])
    saver._z_gal = np.array([freq_to_redshift(nu[1])])
    prefix = str(tmp_path) + os.sep
    saver.save_mock_products(prefix, seed=0)

    def _receiver():
        mock = MockSimulation(
            nu=nu,
            wproj=wproj,
            num_pix_x=8,
            num_pix_y=8,
            flat_sky=True,
            seed=0,
            sigma_beam_ch=2.0,
        )
        mock.include_beam = [True, False]
        return mock

    plain = _receiver()
    load_precomputed_mock(
        plain, prefix, read_galaxies=False, convolve_loaded_mock=False
    )
    assert np.allclose(plain.data, cube)
    beamed = _receiver()
    load_precomputed_mock(
        beamed, prefix, read_galaxies=False, convolve_loaded_mock=True
    )
    assert not np.allclose(beamed.data, cube)
    assert _high_k_power(beamed.data) < _high_k_power(plain.data)


def test_multipole_list_and_loaded_mock_reject_bad_inputs(tmp_path):
    with pytest.raises(ValueError):
        _multipole_ells([])
    nu = np.linspace(900e6, 1000e6, 4)
    wproj = create_wcs(0.0, 0.0, [4, 4], [1.0, 1.0])
    saver = MockSimulation(
        nu=nu,
        wproj=wproj,
        num_pix_x=4,
        num_pix_y=4,
        flat_sky=True,
        seed=0,
    )
    saver.data = np.ones((4, 4, nu.size))
    saver._ra_gal = np.array([0.0])
    saver._dec_gal = np.array([0.0])
    saver._z_gal = np.array([0.4])
    prefix = str(tmp_path) + os.sep
    saver.save_mock_products(prefix, seed=0)

    outside = MockSimulation(
        nu=nu,
        wproj=wproj,
        num_pix_x=4,
        num_pix_y=4,
        flat_sky=True,
        seed=0,
    )
    outside.nu_min = 2e9
    outside.nu_max = 3e9
    with pytest.raises(ValueError):
        load_precomputed_mock(outside, prefix, read_galaxies=False)

    bad_beam = MockSimulation(
        nu=nu,
        wproj=wproj,
        num_pix_x=4,
        num_pix_y=4,
        flat_sky=True,
        seed=0,
    )
    bad_beam.sigma_beam_ch = np.ones(1)
    with pytest.raises(ValueError):
        load_precomputed_mock(
            bad_beam, prefix, read_galaxies=False, convolve_loaded_mock=True
        )


def test_loaded_cube_mask_is_rebuilt(tmp_path):
    nu = np.linspace(900e6, 1000e6, 4)
    wproj = create_wcs(0.0, 0.0, [4, 4], [1.0, 1.0])
    saver = MockSimulation(
        nu=nu,
        wproj=wproj,
        num_pix_x=4,
        num_pix_y=4,
        flat_sky=True,
        seed=0,
    )
    saver.data = np.ones((4, 4, nu.size))
    saver._ra_gal = np.array([0.0])
    saver._dec_gal = np.array([0.0])
    saver._z_gal = np.array([0.4])
    prefix = str(tmp_path) + os.sep
    saver.save_mock_products(prefix, seed=0)

    def receiver():
        return MockSimulation(
            nu=nu,
            wproj=wproj,
            num_pix_x=4,
            num_pix_y=4,
            flat_sky=True,
            seed=0,
        )

    mock = receiver()
    mock.map_has_sampling = np.zeros((4, 4, 2), dtype=bool)
    mock.map_has_sampling[1, 1, :] = True
    mock.weights_map_pixel = np.ones((4, 4, 2))
    load_precomputed_mock(mock, prefix, read_galaxies=False)
    assert mock.map_has_sampling.shape == mock.data.shape
    assert mock.weights_map_pixel.shape == mock.data.shape
    assert mock.map_has_sampling[1, 1].all()
    assert not mock.map_has_sampling[0, 0].any()

    mock = receiver()
    mock.map_has_sampling = np.zeros((4, 4), dtype=bool)
    mock.map_has_sampling[2, 2] = True
    mock.weights_map_pixel = np.ones((4, 4))
    load_precomputed_mock(mock, prefix, read_galaxies=False)
    assert mock.map_has_sampling.shape == mock.data.shape
    assert mock.weights_map_pixel.shape == mock.data.shape
    assert mock.map_has_sampling[2, 2].all()

    mock = receiver()
    mock.map_has_sampling = np.ones((2, 2), dtype=bool)
    mock.weights_map_pixel = np.ones(3)
    load_precomputed_mock(mock, prefix, read_galaxies=False)
    assert mock.map_has_sampling.shape == mock.data.shape
    assert mock.weights_map_pixel.shape == mock.data.shape
    assert mock.map_has_sampling.all()
