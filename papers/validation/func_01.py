from meer21cm import MockSimulation
import numpy as np
import matplotlib.pyplot as plt
from meer21cm.plot import plot_map
from meer21cm.util import create_wcs, redshift_to_freq
from specs import *
from scipy.interpolate import interp1d
from meer21cm.telescope import dish_beam_sigma
import scipy.signal.windows as windows
from multiprocessing import Pool
from meer21cm.power import bin_3d_to_cy, bin_3d_to_1d

# one process uses about 64GB mem for 0.4-1.1
# mpi needed
window_name = 'boxcar'
def get_3d_power(seed):
    z_func = interp1d(z_cen, z_count/dV_arr, kind="linear", bounds_error=False, fill_value=0)
    sigma_beam_ch = dish_beam_sigma(13.5,nu_arr)
    mock = MockSimulation(
        wproj=wcs,
        num_pix_x=num_pix_x,
        num_pix_y=num_pix_y,
        ra_range=ra_range,
        dec_range=dec_range,
        nu=nu_arr,
        discrete_source_dndz=z_func,
        seed=seed,
        tracer_bias_2=1.0,
        tracer_bias_1=1.0,
        mean_amp_1="average_hi_temp",
        omega_hi=5e-4,
        #sigma_beam_ch=sigma_beam_ch,
        sigma_v_1=100,
        sigma_v_2=100,
    )
    num_gal = int(mock.survey_volume * n_gal)
    mock.num_discrete_source = num_gal
    #mock.sigma_beam_ch = np.ones_like(mock.nu) * 0.00001
    #cov_dist = mock.cosmo.comoving_distance(mock.z_ch).value
    #mock.sigma_beam_ch = cov_dist.max()/cov_dist * sigma_beam_ch.mean()
    mock.W_HI = np.ones_like(mock.W_HI)
    mock.w_HI = np.ones_like(mock.w_HI)
    mock.downres_factor_transverse = 1/2
    mock.downres_factor_radial = 1/2
    mock.get_enclosing_box()
    mock.data = mock.propagate_mock_field_to_data(mock.mock_tracer_field_1)
    mock.trim_map_to_range()
    mock.downres_factor_transverse = 3
    mock.downres_factor_radial = 6
    mock.get_enclosing_box()
    mock.grid_scheme = 'nnb'
    himap_rg,_,_ = mock.grid_data_to_field()
    mock.field_1 = himap_rg
    mock.weights_1 = mock.counts_in_box.astype(np.float32)
    mock.taper_func = getattr(windows, window_name)
    mock.apply_taper_to_field(1,axis=[0,1,2])
    mock.include_sky_sampling = [True, False]
    mock.compensate = [True, True]
    mock.include_beam = [True, False]
    pdata3d = mock.auto_power_3d_1
    phimod3d = mock.auto_power_tracer_1_model
    return pdata3d, phimod3d

if __name__ == "__main__":
    # run the simulations
    pdata3d_arr = []
    phimod3d_arr = []
    pg3d_arr = []
    pgmod3d_arr = []
    pcross3d_arr = []
    pcrossmod3d_arr = []
    with Pool(16) as p:
        for pdata3d, phimod3d in p.map(
            get_3d_power, range(32)
        ):
            pdata3d_arr.append(pdata3d)
            phimod3d_arr.append(phimod3d)
    pdata3d_arr = np.array(pdata3d_arr)
    phimod3d_arr = np.array(phimod3d_arr)
    # get the dimension
    mock = MockSimulation(
        wproj=wcs,
        num_pix_x=num_pix_x,
        num_pix_y=num_pix_y,
        ra_range=ra_range,
        dec_range=dec_range,
        nu=nu_arr,
        mean_amp_1="average_hi_temp",
        omega_hi=5e-4,
    )
    mock.downres_factor_transverse = 3
    mock.downres_factor_radial = 6
    mock.get_enclosing_box()
    mock.k1dbins = k1dbins
    mock.kperpbins = kperpbins
    mock.kparabins = kparabins
    # bin the power spectra
    pdatacy_arr = bin_3d_to_cy(pdata3d_arr, mock.k_perp, mock.kperpbins, vectorize=True)
    pdatacy_arr = bin_3d_to_cy(
        pdatacy_arr, np.abs(mock.k_para), mock.kparabins, vectorize=True
    )
    phimodcy_arr = bin_3d_to_cy(
        phimod3d_arr[0], mock.k_perp, mock.kperpbins, vectorize=False
    )
    phimodcy_arr = bin_3d_to_cy(
        phimodcy_arr, np.abs(mock.k_para), mock.kparabins, vectorize=False
    )
    fig = plot_cy_power(
        mock.kperpbins,
        mock.kparabins,
        pdatacy_arr,
        phimodcy_arr,
        0.5,
        1.5,
    )
    fig.savefig(f"plots/01_hicy_{window_name}_nobeam.png",bbox_inches="tight")
    k1dsel = (
        (mock.k_vec[0] < 0.7 * mock.k_nyquist[0])[:,None,None]
        * (mock.k_vec[1] < 0.7 * mock.k_nyquist[1])[None,:,None]
        * (mock.k_vec[2] < 1.0 * mock.k_nyquist[2])[None,None,:]
    )
    k1dsel[:2] = 0.0
    k1dsel[:,:2,:] = 0.0
    k1dsel[:,:,:1] = 0.0
    pdata1d_arr, keff, nmodes = bin_3d_to_1d(
        pdata3d_arr, mock.kmode, mock.k1dbins, vectorize=True,
        weights=k1dsel,
    )
    phimod1d_arr, keff, nmodes = bin_3d_to_1d(
        phimod3d_arr[0], mock.kmode, mock.k1dbins, vectorize=False,
        weights=k1dsel,
    )
    fig = plot_1d_power(
        keff,
        pdata1d_arr,
        phimod1d_arr,
        -0.2,
        0.2,
    )
    fig.savefig(f"plots/01_hi1d_{window_name}_nobeam.png",bbox_inches="tight")
    np.savez(
        f'data/01_{window_name}_nobeam.npz',
        pdata1d_arr=pdata1d_arr,
        phimod1d_arr=phimod1d_arr,
        pdatacy_arr=pdatacy_arr,
        phimodcy_arr=phimodcy_arr,
        pdata3d_arr=pdata3d_arr,
        phimod3d_arr=phimod3d_arr,
    )