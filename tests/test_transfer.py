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
    RadialTransferFunction,
    radial_transfer,
)
import pytest
from meer21cm.power_ops import fkp_count_field
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


def _los_projector(modes):
    """R = I - A A^T for real line-of-sight modes."""
    basis, _ = np.linalg.qr(np.atleast_2d(np.asarray(modes, dtype=float)).T)
    n_chan = basis.shape[0]
    return np.eye(n_chan) - basis @ basis.T


def _quadrature_response(clean, bin_index):
    """Power landed in each real-FFT bin by a unit mode at ``bin_index``.

    A cosine and a sine at the same frequency have opposite +k/-k cross terms,
    so their average is the phase-averaged response of white noise in that bin.
    DC and Nyquist have only one real mode.
    """
    n_chan = clean.shape[0]
    channel = np.arange(n_chan)
    n_k = n_chan // 2 + 1
    angle = 2.0 * np.pi * bin_index * channel / n_chan
    # DC, and the Nyquist mode when N is even, are single real modes.
    # For odd N every non-zero real-FFT bin has a distinct negative partner.
    single_real_mode = bin_index == 0 or (n_chan % 2 == 0 and bin_index == n_chan // 2)
    modes = (np.cos(angle),) if single_real_mode else (np.cos(angle), np.sin(angle))
    landed = np.zeros(n_k)
    injected = 0.0
    for mode in modes:
        spectrum = np.fft.rfft(mode)
        landed += np.abs(np.fft.rfft(clean @ mode)) ** 2
        injected += np.abs(spectrum[bin_index]) ** 2
    return landed / injected


def test_analytic_transfer_function_white_noise():
    """White noise keeps the power given by the sum of H over every frequency.

    Removing one cosine keeps half the power at that k_parallel: the cosine
    part is removed and the sine part remains.
    """
    n_chan = 64
    mode_index = 5
    channel = np.arange(n_chan)
    cosine = np.cos(2.0 * np.pi * mode_index * channel / n_chan)
    coordinate = np.linspace(-1.0, 1.0, n_chan)
    cases = (
        _los_projector(cosine),
        _los_projector(np.vstack([np.ones(n_chan), coordinate, coordinate**2])),
    )
    rng = np.random.default_rng(0)
    n_los = 4000
    for clean in cases:
        transfer, window = analytic_transfer_function(clean)
        assert transfer.shape == (n_chan // 2 + 1,)
        assert window.shape == (n_chan // 2 + 1, n_chan // 2 + 1)
        retained = 1.0 / transfer
        noise = rng.standard_normal((n_los, n_chan))
        power_in = np.mean(np.abs(np.fft.rfft(noise, axis=-1)) ** 2, axis=0)
        power_out = np.mean(np.abs(np.fft.rfft(noise @ clean.T, axis=-1)) ** 2, axis=0)
        assert np.max(np.abs(power_out / power_in - retained)) < 2e-2
        kept = retained > 1e-8
        assert np.allclose(window[kept].sum(axis=1), 1.0)
        for bin_index in range(retained.size):
            response = _quadrature_response(clean, bin_index)
            # A fully removed output mode has retained = 0 and an infinite
            # transfer, so that window row is 0/0. The landed power is zero.
            predicted = retained * window[:, bin_index]
            missing = ~np.isfinite(predicted)
            predicted = np.where(missing, 0.0, predicted)
            assert np.allclose(response, predicted, atol=1e-10)
            assert np.allclose(response[missing], 0.0, atol=1e-10)

    odd = 63
    odd_channel = np.arange(odd)
    odd_clean = _los_projector(np.cos(2.0 * np.pi * mode_index * odd_channel / odd))
    odd_transfer, odd_window = analytic_transfer_function(odd_clean)
    odd_retained = 1.0 / odd_transfer
    assert odd_retained[mode_index] == pytest.approx(0.5, abs=1e-10)
    for bin_index in range(odd_retained.size):
        response = _quadrature_response(odd_clean, bin_index)
        predicted = odd_retained * odd_window[:, bin_index]
        assert np.allclose(response, predicted, atol=1e-10)

    retained = 1.0 / analytic_transfer_function(cases[0])[0]
    assert retained[mode_index] == pytest.approx(0.5, abs=1e-10)
    assert np.allclose(retained[np.arange(retained.size) != mode_index], 1.0)


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
    assert auto_arg[-7:-3] == (True, None, None, True)
    assert auto_arg[-3:] == ("HI_mock_", "gal_mock_", None)


def test_construct_fkp_mock_requires_randoms():
    ps = _tiny_transfer_ps()
    tf = TransferFunction(
        ps,
        N_fg=1,
        highres_sim=1,
        upres_transverse=1,
        upres_radial=1,
        construct_fkp_mock=True,
    )
    with pytest.raises(ValueError, match="field_2_has_random"):
        tf.get_arg_list_for_parallel_cross([0])


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


def test_mock_dilution_reference_power_and_rescale(tmp_path):
    ps = _tiny_transfer_ps()
    ps.k1dbins = np.linspace(0.02, 0.3, 6)
    n_freq = int(ps.nu.size)
    phase = np.linspace(0.0, 2.0 * np.pi, n_freq)
    x_grad = np.linspace(-1.0, 1.0, ps.num_pix_x)[:, None, None]
    y_grad = np.linspace(-1.0, 1.0, ps.num_pix_y)[None, :, None]
    cube = (1.5 + x_grad + 0.4 * y_grad) * (2.0 + np.cos(phase))
    saver = MockSimulation(
        nu=np.asarray(ps.nu, dtype=float),
        wproj=ps.wproj,
        num_pix_x=ps.num_pix_x,
        num_pix_y=ps.num_pix_y,
        flat_sky=True,
        seed=0,
    )
    saver.data = cube
    saver._ra_gal = np.array([0.2, -0.3])
    saver._dec_gal = np.array([0.15, -0.25])
    saver._z_gal = freq_to_redshift(np.asarray(ps.nu, dtype=float)[[1, -2]])
    prefix = str(tmp_path) + os.sep
    saver.save_mock_products(prefix, seed=0)

    def cross_powers(cleaning, dilution):
        tf = TransferFunction(
            ps,
            N_fg=1,
            highres_sim=1,
            upres_transverse=1,
            upres_radial=1,
            num_process=1,
            mock_directory=prefix,
            mean_center_map=False,
            project_combined=False,
            R_mat=cleaning,
            mock_dilution=dilution,
        )
        arg = tf.get_arg_list_for_parallel_cross([0], return_power_1d=True)[0]
        result = run_tf_calculation_cross(*arg)
        return np.asarray(result[-2]), np.asarray(result[-1])

    def occupied(left, right):
        sel = np.isfinite(left) & np.isfinite(right)
        assert np.count_nonzero(sel) >= 1
        return left[sel], right[sel]

    raw, cleaned = cross_powers(np.eye(n_freq), 0.5 * np.ones(cube.shape))
    raw, cleaned = occupied(raw, cleaned)
    assert np.max(np.abs(raw)) > 0.0
    assert np.allclose(cleaned, raw)

    mix = np.eye(n_freq)
    mix[0, 1] = 0.7
    mix[1, 0] = -0.4
    per_channel = np.linspace(0.4, 1.3, n_freq)
    before_diluted, after_diluted = cross_powers(mix, per_channel)
    before_plain, after_plain = cross_powers(mix, None)
    before_diluted, before_plain = occupied(before_diluted, before_plain)
    after_diluted, after_plain = occupied(after_diluted, after_plain)
    assert np.allclose(before_diluted, before_plain)
    assert not np.allclose(after_diluted, after_plain)


def test_subtract_cleaned_data(tmp_path):
    ps = _tiny_transfer_ps()
    ps.k1dbins = np.linspace(0.02, 0.3, 6)
    n_freq = int(ps.nu.size)
    phase = np.linspace(0.0, 2.0 * np.pi, n_freq)
    x_grad = np.linspace(-1.0, 1.0, ps.num_pix_x)[:, None, None]
    y_grad = np.linspace(-1.0, 1.0, ps.num_pix_y)[None, :, None]
    cube = (1.5 + x_grad + 0.4 * y_grad) * (2.0 + np.cos(phase))
    saver = MockSimulation(
        nu=np.asarray(ps.nu, dtype=float),
        wproj=ps.wproj,
        num_pix_x=ps.num_pix_x,
        num_pix_y=ps.num_pix_y,
        flat_sky=True,
        seed=0,
    )
    saver.data = cube
    saver._ra_gal = np.array([0.2, -0.3])
    saver._dec_gal = np.array([0.15, -0.25])
    saver._z_gal = freq_to_redshift(np.asarray(ps.nu, dtype=float)[[1, -2]])
    prefix = str(tmp_path) + os.sep
    saver.save_mock_products(prefix, seed=0)

    rng = np.random.default_rng(3)
    data = 5.0 * (1.0 + x_grad * y_grad) * np.sin(phase + 0.3) + rng.normal(
        size=cube.shape
    )
    mix = np.eye(n_freq)
    mix[0, 1] = 0.7
    mix[1, 0] = -0.4

    def cleaned_power(kind, **kwargs):
        tf = TransferFunction(
            ps,
            N_fg=1,
            highres_sim=1,
            upres_transverse=1,
            upres_radial=1,
            num_process=1,
            mock_directory=prefix,
            uncleaned_data=data,
            **kwargs,
        )
        if kind == "cross":
            arg = tf.get_arg_list_for_parallel_cross([0], return_power_1d=True)[0]
            result = run_tf_calculation_cross(*arg)
        else:
            arg = tf.get_arg_list_for_parallel_auto([0], return_power_1d=True)[0]
            result = run_tf_calculation_auto(*arg)
        power = np.asarray(result[-1])
        return power[np.isfinite(power)]

    for kind in ("cross", "auto"):
        # fixed matrix, no mean-centering: R(d + m) - R d is R m exactly
        fixed = dict(R_mat=mix, mean_center_map=False)
        mock_only = cleaned_power(kind, project_combined=False, **fixed)
        combined = cleaned_power(kind, subtract_cleaned_data=False, **fixed)
        subtracted = cleaned_power(kind, **fixed)
        assert mock_only.size >= 1
        assert not np.allclose(combined, mock_only)
        assert np.allclose(subtracted, mock_only)
        # matrix found from data + mock: the data cleaned with its own matrix is removed
        plain = cleaned_power(kind, mean_center_map=True, subtract_cleaned_data=False)
        with_sub = cleaned_power(kind, mean_center_map=True)
        assert np.all(np.isfinite(with_sub))
        assert not np.allclose(plain, with_sub)

    # the dilution is divided out of the difference (cross path only)
    fixed = dict(R_mat=mix, mean_center_map=False)
    dilution = np.linspace(0.4, 1.3, n_freq)
    diluted = cleaned_power("cross", mock_dilution=dilution, **fixed)
    diluted_mock_only = cleaned_power(
        "cross", project_combined=False, mock_dilution=dilution, **fixed
    )
    assert np.allclose(diluted, diluted_mock_only)

    tf = TransferFunction(ps, N_fg=1, uncleaned_data=data)
    assert tf.project_combined is True
    assert tf.subtract_cleaned_data is True
    assert tf.get_cleaned_data().shape == data.shape
    assert tf.get_arg_list_for_parallel_cross([0])[0][-1] is not None
    # nothing to subtract: mock-only projection, flag off, or no data
    tf.project_combined = False
    assert tf.get_cleaned_data() is None
    tf.project_combined = True
    tf.subtract_cleaned_data = False
    assert tf.get_cleaned_data() is None
    assert TransferFunction(ps, N_fg=1).get_cleaned_data() is None

    for arg_func, worker in (
        (tf.get_arg_list_for_parallel_cross, run_tf_calculation_cross),
        (tf.get_arg_list_for_parallel_auto, run_tf_calculation_auto),
    ):
        arg = list(arg_func([0])[0])
        names = list(inspect.signature(worker).parameters)
        arg[names.index("project_combined")] = False
        arg[names.index("cleaned_data")] = np.zeros_like(data)
        with pytest.raises(ValueError, match="project_combined"):
            worker(*arg)
        arg[names.index("cleaned_data")] = "invalid"
        with pytest.raises(ValueError, match="auto"):
            worker(*arg)

    # Direct worker default (cleaned_data="auto") matches TransferFunction defaults
    tf_direct = TransferFunction(
        ps,
        N_fg=1,
        highres_sim=1,
        upres_transverse=1,
        upres_radial=1,
        num_process=1,
        mock_directory=prefix,
        uncleaned_data=data,
    )
    for kind, arg_func, worker in (
        ("cross", tf_direct.get_arg_list_for_parallel_cross, run_tf_calculation_cross),
        ("auto", tf_direct.get_arg_list_for_parallel_auto, run_tf_calculation_auto),
    ):
        full_arg = arg_func([0], return_power_1d=True)[0]
        from_tf = np.asarray(worker(*full_arg)[-1])
        # Omitting cleaned_data uses worker default "auto"
        from_worker_default = np.asarray(worker(*full_arg[:-1])[-1])
        occ = np.isfinite(from_tf)
        assert np.allclose(from_worker_default[occ], from_tf[occ])
        # project_combined=False with default cleaned_data="auto" does not subtract or raise
        no_proj_arg = list(full_arg[:-1])
        names = list(inspect.signature(worker).parameters)
        no_proj_arg[names.index("project_combined")] = False
        from_auto_false = np.asarray(worker(*no_proj_arg)[-1])
        expected_mock_only = cleaned_power(kind, project_combined=False)
        assert np.allclose(
            from_auto_false[np.isfinite(from_auto_false)], expected_mock_only
        )


def test_zero_weight_gap_and_nan_channels(tmp_path):
    ps = _tiny_transfer_ps()
    ps.k1dbins = np.linspace(0.02, 0.3, 6)
    n_freq = int(ps.nu.size)
    gap = 2
    phase = np.linspace(0.0, 2.0 * np.pi, n_freq)
    x_grad = np.linspace(-1.0, 1.0, ps.num_pix_x)[:, None, None]
    y_grad = np.linspace(-1.0, 1.0, ps.num_pix_y)[None, :, None]
    cube = (1.5 + x_grad + 0.4 * y_grad) * (2.0 + np.cos(phase))

    saver = MockSimulation(
        nu=np.asarray(ps.nu, dtype=float),
        wproj=ps.wproj,
        num_pix_x=ps.num_pix_x,
        num_pix_y=ps.num_pix_y,
        flat_sky=True,
        seed=0,
    )
    saver.data = cube
    saver._ra_gal = np.array([0.2, -0.3])
    saver._dec_gal = np.array([0.15, -0.25])
    saver._z_gal = freq_to_redshift(np.asarray(ps.nu, dtype=float)[[1, -2]])
    prefix = str(tmp_path) + os.sep
    saver.save_mock_products(prefix, seed=0)

    weights = np.ones_like(cube)
    weights[:, :, gap] = 0.0
    ps.W_HI = weights.copy()
    ps.w_HI = weights.copy()
    ps.weights_field_1 = weights.copy()
    ps.weights_grid_1 = weights.copy()

    rng = np.random.default_rng(7)
    data_zero = 5.0 * (1.0 + x_grad * y_grad) * np.sin(phase + 0.3) + rng.normal(
        size=cube.shape
    )
    data_zero[:, :, gap] = 0.0
    data_nan = data_zero.copy()
    data_nan[:, :, gap] = np.nan

    # Direct apply_cleaning_matrix: zero-weight and NaN gaps must not poison active
    # channels or emit divide-by-zero warnings
    R_gap = get_pca_matrix(data_zero, 1, weights, True)
    expected_pca = pca_clean(
        data_zero, 1, weights=weights, mean_center=True, ignore_nan=True
    )
    with np.errstate(divide="raise", invalid="raise"):
        cleaned_zero = apply_cleaning_matrix(0.0, data_zero, R_gap, weights, True, True)
        cleaned_nan = apply_cleaning_matrix(0.0, data_nan, R_gap, weights, True, True)
        cleaned_mock_only = apply_cleaning_matrix(
            cube, data_nan, R_gap, weights, True, False
        )
    assert np.all(np.isfinite(cleaned_zero))
    assert np.all(np.isfinite(cleaned_nan))
    assert np.all(np.isfinite(cleaned_mock_only))
    assert np.allclose(cleaned_zero[:, :, gap], 0.0)
    assert np.allclose(cleaned_mock_only[:, :, gap], 0.0)
    assert np.allclose(cleaned_zero, expected_pca)
    assert np.allclose(cleaned_nan, expected_pca)

    def run_worker(kind, uncleaned, **kwargs):
        tf = TransferFunction(
            ps,
            N_fg=1,
            highres_sim=1,
            upres_transverse=1,
            upres_radial=1,
            num_process=1,
            mock_directory=prefix,
            uncleaned_data=uncleaned,
            pca_map_weights=weights,
            mean_center_map=True,
            **kwargs,
        )
        if kind == "cross":
            arg = tf.get_arg_list_for_parallel_cross([0], return_power_1d=True)[0]
            return tf, run_tf_calculation_cross(*arg)
        arg = tf.get_arg_list_for_parallel_auto([0], return_power_1d=True)[0]
        return tf, run_tf_calculation_auto(*arg)

    tf_zero, cross_zero = run_worker("cross", data_zero)
    tf_nan, cross_nan = run_worker("cross", data_nan)
    assert np.allclose(tf_zero.get_cleaned_data(), expected_pca)
    assert np.allclose(tf_nan.get_cleaned_data(), expected_pca)

    tf_vals_zero = np.asarray(cross_zero[0])
    tf_vals_nan = np.asarray(cross_nan[0])
    occ = np.isfinite(tf_vals_zero)
    assert np.count_nonzero(occ) >= 1
    assert np.allclose(tf_vals_zero[occ], tf_vals_nan[occ])

    # Zero or NaN in mock_dilution on the flagged channel must not produce NaNs
    for gap_dil in (0.0, np.nan):
        dilution = np.ones(n_freq, dtype=float)
        dilution[gap] = gap_dil
        _, cross_dil = run_worker("cross", data_zero, mock_dilution=dilution)
        tf_vals_dil = np.asarray(cross_dil[0])
        assert np.all(np.isfinite(tf_vals_dil[occ]))
        assert np.allclose(tf_vals_dil[occ], tf_vals_zero[occ])

    # Auto path: both project_combined=True and project_combined=False (in-place branch)
    for proj in (True, False):
        _, auto_res = run_worker("auto", data_zero, project_combined=proj)
        auto_tf = np.asarray(auto_res[0])
        assert np.count_nonzero(np.isfinite(auto_tf)) >= 1
        assert np.all(np.isfinite(auto_res[-1][np.isfinite(auto_res[-2])]))


def test_cross_worker_fkp_matches_parent_randoms(tmp_path):
    ps = _tiny_transfer_ps()
    ps.k1dbins = np.linspace(0.02, 0.3, 6)
    n_freq = int(ps.nu.size)
    phase = np.linspace(0.0, 2.0 * np.pi, n_freq)
    x_grad = np.linspace(-1.0, 1.0, ps.num_pix_x)[:, None, None]
    y_grad = np.linspace(-1.0, 1.0, ps.num_pix_y)[None, :, None]
    cube = (1.5 + x_grad + 0.4 * y_grad) * (2.0 + np.cos(phase))
    saver = MockSimulation(
        nu=np.asarray(ps.nu, dtype=float),
        wproj=ps.wproj,
        num_pix_x=ps.num_pix_x,
        num_pix_y=ps.num_pix_y,
        flat_sky=True,
        seed=0,
    )
    saver.data = cube
    saver._ra_gal = np.array([0.2, -0.3])
    saver._dec_gal = np.array([0.15, -0.25])
    saver._z_gal = freq_to_redshift(np.asarray(ps.nu, dtype=float)[[1, -2]])
    prefix = str(tmp_path) + os.sep
    saver.save_mock_products(prefix, seed=0)

    tf = TransferFunction(
        ps,
        N_fg=1,
        highres_sim=1,
        upres_transverse=1,
        upres_radial=1,
        num_process=1,
        mock_directory=prefix,
        mean_center_map=False,
        project_combined=False,
        R_mat=np.eye(n_freq),
        construct_fkp_mock=True,
    )
    painted = MockSimulation(**tf.get_mock_instance_attr_dict(0))
    load_precomputed_mock(painted, prefix, read_galaxies=True)
    painted.ra_range = ps.ra_range
    painted.dec_range = ps.dec_range
    painted.trim_map_to_range()
    painted.trim_gal_to_range()
    painted.downres_factor_radial = ps.downres_factor_radial
    painted.downres_factor_transverse = ps.downres_factor_transverse
    painted.get_enclosing_box()
    hi_field, _, _ = painted.grid_data_to_field()
    data_counts, _, _ = painted.grid_gal_to_field()
    random_counts = np.linspace(0.3, 1.4, data_counts.size).reshape(data_counts.shape)
    template = np.linspace(0.5, 1.1, data_counts.size).reshape(data_counts.shape)
    ps.field_2_R = random_counts
    ps.field_2_has_random = True
    ps.weights_grid_2 = template

    def uncleaned(construct_fkp):
        tf.construct_fkp_mock = construct_fkp
        arg = tf.get_arg_list_for_parallel_cross([0], return_power_1d=True)[0]
        result = run_tf_calculation_cross(*arg)
        return np.asarray(result[-2])

    measured = uncleaned(True)
    raw_counts = uncleaned(False)
    field_ref, alpha = fkp_count_field(data_counts, random_counts)
    painted.field_1 = hi_field
    painted.weights_field_1 = ps.weights_field_1
    painted.weights_grid_1 = ps.weights_grid_1
    painted.field_2_D = data_counts
    painted.field_2_R = random_counts
    painted.field_2_alpha = alpha
    painted.field_2_has_random = True
    painted.weights_field_2 = alpha * random_counts * template
    painted.weights_grid_2 = template
    painted.mean_center_2 = False
    painted.unitless_2 = False
    expected, _, _ = painted.get_1d_power(
        painted.cross_power_3d, k1dweights=ps.k1dweights, multipole_ell=0
    )
    expected = np.asarray(expected)
    occupied = np.isfinite(measured) & np.isfinite(expected)
    assert np.count_nonzero(occupied) >= 1
    assert np.max(np.abs(measured[occupied])) > 0.0
    assert np.allclose(measured[occupied], expected[occupied])
    raw_occupied = np.isfinite(raw_counts) & occupied
    assert np.count_nonzero(raw_occupied) >= 1
    assert not np.allclose(raw_counts[raw_occupied], measured[raw_occupied])


def test_loaded_mock_custom_file_stems(tmp_path):
    nu = np.linspace(900e6, 1000e6, 4)
    wproj = create_wcs(0.0, 0.0, [4, 4], [1.0, 1.0])
    cube = np.arange(4 * 4 * nu.size, dtype=float).reshape(4, 4, nu.size)
    saver = MockSimulation(
        nu=nu,
        wproj=wproj,
        num_pix_x=4,
        num_pix_y=4,
        flat_sky=True,
        seed=3,
    )
    saver.data = cube
    saver._ra_gal = np.array([0.2, -0.4])
    saver._dec_gal = np.array([0.1, -0.2])
    saver._z_gal = np.array([0.4, 0.45])
    prefix = str(tmp_path) + os.sep
    hi_stem = "cube_"
    gal_stem = "catalogue_"
    hi_path, gal_path = saver.save_mock_products(
        prefix, seed=3, hi_stem=hi_stem, gal_stem=gal_stem
    )
    assert hi_path == prefix + "cube_3.pkl"
    assert gal_path == prefix + "catalogue_3.fits"
    assert os.path.isfile(hi_path)
    assert os.path.isfile(gal_path)
    assert not os.path.isfile(prefix + "HI_mock_3.pkl")
    assert not os.path.isfile(prefix + "gal_mock_3.fits")

    loaded = MockSimulation(
        nu=nu,
        nu_min=float(nu.min()) - 1.0,
        nu_max=float(nu.max()) + 1.0,
        wproj=wproj,
        num_pix_x=4,
        num_pix_y=4,
        flat_sky=True,
        seed=3,
    )
    load_precomputed_mock(
        loaded,
        prefix,
        read_galaxies=True,
        hi_stem=hi_stem,
        gal_stem=gal_stem,
    )
    assert np.allclose(loaded.data, cube)
    assert np.allclose(loaded.ra_gal, [0.2, -0.4])
    assert np.allclose(loaded.dec_gal, [0.1, -0.2])
    assert loaded.gal_file == gal_path
    with pytest.raises(FileNotFoundError):
        load_precomputed_mock(loaded, prefix, read_galaxies=False)

    ps = _tiny_transfer_ps()
    tf = TransferFunction(
        ps,
        N_fg=1,
        highres_sim=1,
        upres_transverse=1,
        upres_radial=1,
        num_process=1,
        hi_stem=hi_stem,
        gal_stem=gal_stem,
    )
    assert tf.get_arg_list_for_parallel_auto([3])[0][-3:-1] == (hi_stem, gal_stem)
    assert tf.get_arg_list_for_parallel_cross([3])[0][-3:-1] == (hi_stem, gal_stem)


def test_loaded_mock_gal_cat_selects_tracer_stem(tmp_path):
    nu = np.linspace(900e6, 1000e6, 4)
    wproj = create_wcs(0.0, 0.0, [4, 4], [1.0, 1.0])
    cube = np.ones((4, 4, nu.size))
    saver = MockSimulation(
        nu=nu,
        wproj=wproj,
        num_pix_x=4,
        num_pix_y=4,
        flat_sky=True,
        seed=1,
    )
    saver.data = cube
    saver._ra_gal = np.array([1.0])
    saver._dec_gal = np.array([-1.0])
    saver._z_gal = np.array([0.4])
    prefix = str(tmp_path) + os.sep
    _, gal_path = saver.save_mock_products(prefix, seed=1, gal_stem="gal_LRG_mock_")
    loaded = MockSimulation(
        nu=nu,
        nu_min=float(nu.min()) - 1.0,
        nu_max=float(nu.max()) + 1.0,
        wproj=wproj,
        num_pix_x=4,
        num_pix_y=4,
        flat_sky=True,
        seed=1,
    )
    load_precomputed_mock(loaded, prefix, gal_cat="LRG")
    assert loaded.gal_file == gal_path
    assert np.allclose(loaded.ra_gal, [1.0])
    assert np.allclose(loaded.data, cube)
    with pytest.raises(ValueError):
        load_precomputed_mock(loaded, prefix, gal_cat="LRG", gal_stem="catalogue_")


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


def test_radial_transfer_identity():
    """No channelisation: the transfer is unity."""
    chi_u = np.linspace(2272.0, 2868.0, 255)
    dchi_u = float(chi_u[1] - chi_u[0])
    k = np.array([0.02, 0.1, 0.3])
    transfer = radial_transfer(k, chi_u, dchi_u)
    assert np.allclose(transfer, 1.0)


def test_radial_transfer_channel_tophat():
    """One voxel per channel reproduces the top-hat transfer at low k."""
    n = 128
    chi_ch = np.linspace(2272.4, 2867.9, n)
    dchi_ch = np.full(n, float(np.median(np.diff(chi_ch))))
    chi_u = np.linspace(chi_ch.min(), chi_ch.max(), 512)
    dchi_u = float(chi_u[1] - chi_u[0])
    k = np.array([0.02, 0.05])
    transfer = radial_transfer(
        k, chi_u, dchi_u, chi_ch=chi_ch, dchi_ch=dchi_ch, dchi_vox=dchi_ch[0]
    )
    expected = np.sinc(k * dchi_ch[0] / 2.0 / np.pi) ** 2
    assert np.allclose(transfer, expected, rtol=0.05)
    assert np.all(transfer <= 1.0 + 1e-12)
    # the transfer decreases with k
    assert transfer[0] > transfer[1]


def test_radial_transfer_cleaning_matrix():
    """An identity cleaning matrix leaves the transfer unchanged; zero kills it."""
    n = 32
    chi_ch = np.linspace(2272.4, 2867.9, n)
    dchi_ch = np.full(n, float(np.median(np.diff(chi_ch))))
    chi_u = np.linspace(chi_ch.min(), chi_ch.max(), 200)
    dchi_u = float(chi_u[1] - chi_u[0])
    k = np.array([0.05, 0.1])
    base = radial_transfer(k, chi_u, dchi_u, chi_ch=chi_ch, dchi_ch=dchi_ch)
    ident = radial_transfer(
        k, chi_u, dchi_u, chi_ch=chi_ch, dchi_ch=dchi_ch, clean_matrix=np.eye(n)
    )
    assert np.allclose(base, ident)
    zeroed = radial_transfer(
        k, chi_u, dchi_u, chi_ch=chi_ch, dchi_ch=dchi_ch, clean_matrix=np.zeros((n, n))
    )
    assert np.allclose(zeroed, 0.0)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(chi_u=np.ones((2, 2)), dchi_u=1.0), "chi_u must be a 1-D array"),
        (dict(chi_u=np.array([1.0]), dchi_u=1.0), "chi_u must be a 1-D array"),
        (
            dict(chi_u=np.array([1.0, 0.0]), dchi_u=1.0),
            "chi_u must be strictly ascending",
        ),
        (
            dict(chi_u=np.linspace(0.0, 10.0, 8), dchi_u=0.0),
            "dchi_u must be a positive finite number",
        ),
        (
            dict(chi_u=np.linspace(0.0, 10.0, 8), dchi_u=np.nan),
            "dchi_u must be a positive finite number",
        ),
        (
            dict(
                chi_u=np.linspace(0.0, 10.0, 8),
                dchi_u=1.0,
                clean_matrix=np.eye(2),
            ),
            "clean_matrix requires chi_ch",
        ),
        (
            dict(
                chi_u=np.linspace(0.0, 10.0, 8),
                dchi_u=1.0,
                chi_ch=np.array([1.0]),
            ),
            "chi_ch must be a 1-D array",
        ),
        (
            dict(
                chi_u=np.linspace(0.0, 10.0, 8),
                dchi_u=1.0,
                chi_ch=np.array([1.0, 2.0, 3.0]),
                dchi_ch=np.ones((3, 2)),
            ),
            "dchi_ch must have the same size",
        ),
        (
            dict(
                chi_u=np.linspace(0.0, 10.0, 8),
                dchi_u=1.0,
                chi_ch=np.array([1.0, 2.0]),
                dchi_vox=0.0,
            ),
            "dchi_vox must be a positive finite number",
        ),
        (
            dict(
                chi_u=np.linspace(0.0, 10.0, 8),
                dchi_u=1.0,
                chi_ch=np.array([1.0, 2.0]),
                clean_matrix=np.ones(3),
            ),
            "clean_matrix must be a square 2-D array",
        ),
        (
            dict(
                chi_u=np.linspace(0.0, 10.0, 8),
                dchi_u=1.0,
                chi_ch=np.array([1.0, 2.0]),
                clean_matrix=np.eye(3),
            ),
            "clean_matrix must have shape",
        ),
    ],
)
def test_radial_transfer_function_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        RadialTransferFunction(**kwargs)


def test_radial_transfer_function_defaults_and_helpers():
    chi_u = np.linspace(0.0, 10.0, 20)
    chi_ch = np.array([3.0, 1.0, 2.0])
    dchi_ch = np.array([0.3, 0.1, 0.2])
    radial = RadialTransferFunction(chi_u, 0.05, chi_ch=chi_ch, dchi_ch=dchi_ch)
    assert np.allclose(radial.chi_ch, np.array([1.0, 2.0, 3.0]))
    assert np.allclose(radial.dchi_ch, np.array([0.1, 0.2, 0.3]))
    centres = np.array([1.0, 2.0, 4.0])
    defaulted = RadialTransferFunction(chi_u, 0.05, chi_ch=centres)
    assert np.allclose(defaulted.dchi_ch, np.median(np.diff(defaulted.chi_ch)))
    assert defaulted.dchi_vox == pytest.approx(float(np.median(defaulted.dchi_ch)))
    identity = RadialTransferFunction(chi_u, 0.05)
    with pytest.raises(ValueError, match="voxel_grid requires chi_ch"):
        identity.voxel_grid()
    with pytest.raises(ValueError, match="values must be sampled on the voxel grid"):
        radial._channelise(np.zeros(3, dtype=complex))
    k = np.array([0.05, 0.1])
    assert np.allclose(radial(k), radial.transfer(k))
    assert np.allclose(
        radial_transfer(k, chi_u, 0.05, chi_ch=chi_ch, dchi_ch=dchi_ch),
        radial.transfer(k),
    )


def test_radial_transfer_zero_taper_is_nan():
    chi_u = np.linspace(0.0, 10.0, 32)
    radial = RadialTransferFunction(
        chi_u, float(chi_u[1] - chi_u[0]), taper=lambda n: np.zeros(n)
    )
    assert np.all(np.isnan(radial.transfer(np.array([0.1, 0.2]))))


def test_radial_transfer_channel_order_invariant():
    chi_ch = np.linspace(100.0, 200.0, 16)
    dchi_ch = np.linspace(1.0, 1.5, 16)
    chi_u = np.linspace(chi_ch.min(), chi_ch.max(), 64)
    dchi_u = float(chi_u[1] - chi_u[0])
    k = np.array([0.02, 0.08])
    forward = radial_transfer(
        k, chi_u, dchi_u, chi_ch=chi_ch, dchi_ch=dchi_ch, dchi_vox=0.5
    )
    backward = radial_transfer(
        k,
        chi_u,
        dchi_u,
        chi_ch=chi_ch[::-1],
        dchi_ch=dchi_ch[::-1],
        dchi_vox=0.5,
    )
    assert np.allclose(forward, backward, rtol=1e-12, atol=0.0)
