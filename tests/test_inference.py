import multiprocessing as mp

mp.set_start_method("spawn", force=True)
from meer21cm import PowerSpectrum
import numpy as np
from meer21cm.inference import (
    extract_model_fitting_inputs,
    SamplerEmcee,
    SamplerBase,
    SamplerNautilus,
)
import emcee
import os
import pytest


def test_emcee_run():
    ps = PowerSpectrum(
        band="L",
        survey="meerklass_2021",
        sigma_v_1=100,
        tracer_bias_1=1.5,
        tracer_bias_2=2.0,
    )
    ps._box_len = np.array([500, 500, 500])
    ps._box_ndim = np.array([2, 2, 2])
    ps.propagate_field_k_to_model()
    ps.k1dbins = np.linspace(0.0, 0.03, 3)
    pmodauto, keff, nmodes = ps.get_1d_power(ps.auto_power_tracer_1_model)
    pmodcross, keff, nmodes = ps.get_1d_power(ps.cross_power_tracer_model)
    pmodggauto, keff, nmodes = ps.get_1d_power(ps.auto_power_tracer_2_model)
    ps_dict = extract_model_fitting_inputs(ps)
    data_vector = np.array([pmodauto, pmodcross, pmodggauto]).ravel()
    data_covariance = np.diag(data_vector * 0.1) ** 2
    sampler = SamplerEmcee(
        ps_dict=ps_dict,
        data_vector=data_vector,
        data_covariance=data_covariance,
        params_name=["tracer_bias_1", "sigma_v_1"],
        params_prior=[
            ("uniform", 0.5, 2.5),
            ("gaussian", 100, 10),
        ],
        observables=["hi-auto", "cross", "gg-auto"],
        nwalkers=2,
        nsteps=1,
        nthreads=1,
        mp_backend="multiprocessing",
        save=True,
        save_filename="test_fit.h5",
        save_model_blobs=True,
    )
    assert sampler.log_prior(np.array([1.5, 100])) == 0.0
    sampler = SamplerEmcee(
        ps_dict=ps_dict,
        data_vector=data_vector,
        data_covariance=data_covariance,
        params_name=["tracer_bias_1"],
        params_prior=[
            ("uniform", 0.5, 2.5),
        ],
        observables=["hi-auto", "cross", "gg-auto"],
        nwalkers=2,
        nsteps=1,
        nthreads=1,
        mp_backend="multiprocessing",
        save=True,
        save_filename="test_fit.h5",
        save_model_blobs=True,
    )
    assert sampler.hartlap_factor == 1.0
    assert sampler.percival_factor == 1.0
    sampler.do_hartlap_correction = True
    sampler.do_percival_correction = True
    sampler.num_mocks = 1000000000
    assert np.allclose(sampler.hartlap_factor, 1.0)
    assert np.allclose(sampler.percival_factor, 1.0)
    ll_test = sampler.log_likelihood(np.array([1.5]))[0]
    assert np.isclose(ll_test, 0.0)
    ll_test, blob_test = sampler.log_likelihood(np.array([3]))
    assert not np.isfinite(ll_test)
    assert np.allclose(blob_test, np.zeros((3, 1)))
    sampler.save_model_blobs = False
    # test switching off blob
    ll_test = sampler.log_likelihood(np.array([1.5]))
    assert np.isclose(ll_test, 0.0)
    ll_test = sampler.log_likelihood(np.array([3]))
    assert not np.isfinite(ll_test)
    sampler.save_model_blobs = True
    sampler.run(resume=False, progress=False)
    # run another step
    sampler.nsteps = 2
    sampler.run(resume=True, progress=False)
    backend = emcee.backends.HDFBackend("test_fit.h5")
    assert backend.iteration == 2
    # run another step with reset
    sampler.nsteps = 1
    sampler.run(resume=False, progress=False)
    backend = emcee.backends.HDFBackend("test_fit.h5")
    assert backend.iteration == 1
    points = sampler.get_points()
    assert points.shape == (1, 2, 1)
    blobs = sampler.get_blobs()
    assert blobs.shape == (1, 2, 3, 2)
    log_prob = sampler.get_log_prob()
    assert log_prob.shape == (1, 2)
    sampler.save = False
    sampler.mp_backend = "mpi"
    sampler.nsteps = 1
    mcmc = sampler.run(resume=False, progress=False)
    with pytest.raises(ValueError):
        sampler.get_points()
    with pytest.raises(ValueError):
        sampler.get_blobs()
    points = sampler.get_points(mcmc)
    assert points.shape == (1, 2, 1)
    blobs = sampler.get_blobs(mcmc)
    assert blobs.shape == (1, 2, 3, 2)
    log_prob = sampler.get_log_prob(mcmc)
    assert log_prob.shape == (1, 2)
    os.remove("test_fit.h5")


def test_validate_input():
    ps = PowerSpectrum(
        band="L",
        survey="meerklass_2021",
        sigma_v_1=100,
        tracer_bias_1=1.5,
        tracer_bias_2=2.0,
    )
    ps._box_len = np.array([500, 500, 500])
    ps._box_ndim = np.array([20, 20, 20])
    ps.propagate_field_k_to_model()
    ps.k1dbins = np.linspace(0.05, 0.2, 11)
    ps_dict = extract_model_fitting_inputs(ps)
    data_vector = np.ones(30)
    data_covariance = np.eye(20)
    with pytest.raises(ValueError):
        sampler = SamplerBase(
            ps_dict=ps_dict,
            data_vector=data_vector,
            data_covariance=data_covariance,
            params_name=["tracer_bias_1", "sigma_v_1"],
            params_prior=[
                ("uniform", 0.5, 2.5),
                ("uniform", 0, 400),
            ],
            observables=["hi-auto", "cross", "gg-auto"],
        )
    data_covariance = np.eye(30)
    with pytest.raises(ValueError):
        sampler = SamplerBase(
            ps_dict=ps_dict,
            data_vector=data_vector,
            data_covariance=data_covariance,
            params_name=["tracer_bias_1", "sigma_v_1"],
            params_prior=[
                ("uniform", 0.5, 2.5),
                ("uniform", 0, 400),
                ("uniform", 0, 400),
            ],
        )
    with pytest.raises(ValueError):
        sampler = SamplerBase(
            ps_dict=ps_dict,
            data_vector=data_vector,
            data_covariance=data_covariance,
            params_name=["tracer_bias_1", "sigma_v_1"],
            params_prior=[
                ("uniform", 0.5, 2.5),
                ("uniform", 0, 400),
            ],
            # wrong observable name
            observables=["gg-cross"],
        )


def test_nautilus_run():
    ps = PowerSpectrum(
        band="L",
        survey="meerklass_2021",
        sigma_v_1=100,
        tracer_bias_1=1.5,
        tracer_bias_2=2.0,
    )
    ps._box_len = np.array([500, 500, 500])
    ps._box_ndim = np.array([20, 20, 20])
    ps.propagate_field_k_to_model()
    ps.k1dbins = np.linspace(0.05, 0.2, 11)
    pmodauto, keff, nmodes = ps.get_1d_power(ps.auto_power_tracer_1_model)
    pmodcross, keff, nmodes = ps.get_1d_power(ps.cross_power_tracer_model)
    pmodggauto, keff, nmodes = ps.get_1d_power(ps.auto_power_tracer_2_model)
    ps_dict = extract_model_fitting_inputs(ps)
    data_vector = np.array([pmodauto, pmodcross, pmodggauto]).ravel()
    data_covariance = np.diag(data_vector * 0.1) ** 2
    sampler = SamplerNautilus(
        ps_dict=ps_dict,
        data_vector=data_vector,
        data_covariance=data_covariance,
        # nautilus requires minumum 2 parameters
        params_name=["tracer_bias_1", "sigma_v_1"],
        params_prior=[
            ("uniform", 0.5, 2.5),
            ("gaussian", 100, 10),
        ],
        observables=["hi-auto", "cross", "gg-auto"],
        n_live_points=3,
        f_live=0.01,
        n_shell=1,
        n_eff=10000,
        nthreads=1,
        save=True,
        save_filename="tests/test_fit2.h5",
        save_model_blobs=True,
        timeout=1,
    )
    ll, blob = sampler.compute_log_likelihood(np.array([1.5, 100]))
    assert np.isclose(ll, 0.0)
    assert blob.shape == (3, 10)
    sampler.save_model_blobs = False
    sampler.run(resume=False, progress=False, run_sampler=False)
    points, log_w, log_l = sampler.get_posterior()
    sampler.save = False
    sampler.mp_backend = "mpi"
    sampler.run(resume=False, progress=False, run_sampler=False)
    # os.remove("test_fit2.h5")
