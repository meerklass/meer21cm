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
from meer21cm.grid import shot_noise_correction_from_gridding
from meer21cm.power import get_shot_noise_galaxy


# one process uses about 64GB mem for 0.4-1.1
# mpi needed
window_name = 'tukey'
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
    mock.taper_func = getattr(windows, window_name)
    num_gal = int(mock.survey_volume * n_gal)
    mock.num_discrete_source = num_gal
    mock.W_HI = np.ones_like(mock.W_HI)
    mock.w_HI = np.ones_like(mock.w_HI)
    mock.downres_factor_transverse = 1/2
    mock.downres_factor_radial = 1/2
    mock.get_enclosing_box()
    mock.data = mock.propagate_mock_field_to_data(mock.mock_tracer_field_1)
    mock.propagate_mock_tracer_to_gal_cat()
    mock.trim_map_to_range()
    mock.trim_gal_to_range()
    mock.downres_factor_transverse = 3
    mock.downres_factor_radial = 6
    mock.get_enclosing_box()
    mock.grid_scheme = 'cic'
    himap_rg,_,_ = mock.grid_data_to_field()
    galmap_rg,_,_ = mock.grid_gal_to_field()
    dndz_box = mock.discrete_source_dndz(mock._box_voxel_redshift)
    box_counts = mock.counts_in_box.copy()
    box_len = mock.box_len.copy()
    box_ndim = mock.box_ndim.copy()
    box_resol = mock.box_resol.copy()
    split_indx = box_ndim[1]//2
    mock._box_ndim = np.array([box_ndim[0],split_indx,box_ndim[2]])
    mock._box_len = mock._box_ndim * box_resol
    mock.propagate_field_k_to_model()
    mock._counts_in_box = box_counts[:,:split_indx,:]
    mock.field_1 = himap_rg[:,:split_indx,:]
    mock.weights_1 = mock.counts_in_box.astype('float')
    mock.apply_taper_to_field(1,axis=[0,1,2])
    mock.include_sky_sampling = [True, False]
    mock.compensate = [True, True]
    mock.include_beam = [True, False]
    mock.field_2 = galmap_rg[:,:split_indx,:]
    mock.weights_field_2 = dndz_box[:,:split_indx,:]
    #mock.weights_grid_2 = ((dndz_box>0)*box_counts).astype('float')[:,:split_indx,:]
    mock.weights_grid_2 = ((dndz_box*box_counts)>0).astype('float')[:,:split_indx,:]
    mock.apply_taper_to_field(2, axis=[0, 1, 2])
    mock.mean_center_2 = True
    mock.unitless_2 = True
    shot_noise = get_shot_noise_galaxy(
        galmap_rg[:,:split_indx,:], mock.box_len, mock.weights_grid_2, mock.weights_field_2
    ) * shot_noise_correction_from_gridding(
        mock.box_ndim,
        mock.grid_scheme
    )
    pdata3d_1 = mock.auto_power_3d_1
    phimod3d_1 = mock.auto_power_tracer_1_model
    pg3d_1 = mock.auto_power_3d_2 - shot_noise
    pgmod3d_1 = mock.auto_power_tracer_2_model
    pcross3d_1 = mock.cross_power_3d
    pcrossmod3d_1 = mock.cross_power_tracer_model
    mock._box_ndim = np.array([box_ndim[0],box_ndim[1] - split_indx,box_ndim[2]])
    mock._box_len = mock._box_ndim * box_resol
    mock.propagate_field_k_to_model()
    mock._counts_in_box = box_counts[:,split_indx:,:]
    mock.field_1 = himap_rg[:,split_indx:,:]
    mock.weights_1 = mock.counts_in_box.astype('float')
    mock.apply_taper_to_field(1,axis=[0,1,2])
    mock.include_sky_sampling = [True, False]
    mock.compensate = [True, True]
    mock.include_beam = [True, False]
    mock.field_2 = galmap_rg[:,split_indx:,:]
    mock.weights_field_2 = dndz_box[:,split_indx:,:]
    #mock.weights_grid_2 = ((dndz_box>0)*box_counts).astype('float')[:,split_indx:,:]
    mock.weights_grid_2 = ((dndz_box*box_counts)>0).astype('float')[:,split_indx:,:]
    mock.apply_taper_to_field(2, axis=[0, 1, 2])
    mock.mean_center_2 = True
    mock.unitless_2 = True
    shot_noise = get_shot_noise_galaxy(
        galmap_rg[:,split_indx:,:], mock.box_len, mock.weights_grid_2, mock.weights_field_2
    ) * shot_noise_correction_from_gridding(
        mock.box_ndim,
        mock.grid_scheme
    )
    pdata3d_2 = mock.auto_power_3d_1
    phimod3d_2 = mock.auto_power_tracer_1_model
    pg3d_2 = mock.auto_power_3d_2 - shot_noise
    pgmod3d_2 = mock.auto_power_tracer_2_model
    pcross3d_2 = mock.cross_power_3d
    pcrossmod3d_2 = mock.cross_power_tracer_model
    return pdata3d_1, phimod3d_1, pdata3d_2, phimod3d_2, pg3d_1, pgmod3d_1,pg3d_2, pgmod3d_2,pcross3d_1,pcrossmod3d_1,pcross3d_2,pcrossmod3d_2

if __name__ == "__main__":
    # run the simulations
    pdata3d_arr_1 = []
    phimod3d_arr_1 = []
    pdata3d_arr_2 = []
    phimod3d_arr_2 = []
    pg3d_arr_1 = []
    pgmod3d_arr_1 = []
    pg3d_arr_2 = []
    pgmod3d_arr_2 = []
    pcross3d_arr_1 = []
    pcrossmod3d_arr_1 = []
    pcross3d_arr_2 = []
    pcrossmod3d_arr_2 = []
    with Pool(16) as p:
        for pdata3d_1, phimod3d_1, pdata3d_2, phimod3d_2, pg3d_1, pgmod3d_1,pg3d_2, pgmod3d_2,pcross3d_1,pcrossmod3d_1,pcross3d_2,pcrossmod3d_2 in p.map(
            get_3d_power, range(128)
        ):
            pdata3d_arr_1.append(pdata3d_1)
            phimod3d_arr_1.append(phimod3d_1)
            pdata3d_arr_2.append(pdata3d_2)
            phimod3d_arr_2.append(phimod3d_2)
            pg3d_arr_1.append(pg3d_1)
            pg3d_arr_2.append(pg3d_2)
            pgmod3d_arr_1.append(pgmod3d_1)
            pgmod3d_arr_2.append(pgmod3d_2)
            pcross3d_arr_1.append(pcross3d_1)
            pcross3d_arr_2.append(pcross3d_2)
            pcrossmod3d_arr_1.append(pcrossmod3d_1)
            pcrossmod3d_arr_2.append(pcrossmod3d_2)
    pdata3d_arr_1 = np.array(pdata3d_arr_1)
    phimod3d_arr_1 = np.array(phimod3d_arr_1)
    pdata3d_arr_2 = np.array(pdata3d_arr_2)
    phimod3d_arr_2 = np.array(phimod3d_arr_2)
    pg3d_arr_1 = np.array(pg3d_arr_1)
    pg3d_arr_2 = np.array(pg3d_arr_2)
    pgmod3d_arr_1 = np.array(pgmod3d_arr_1)
    pgmod3d_arr_2 = np.array(pgmod3d_arr_2)
    pcross3d_arr_1 = np.array(pcross3d_arr_1)
    pcross3d_arr_2 = np.array(pcross3d_arr_2)
    pcrossmod3d_arr_1 = np.array(pcrossmod3d_arr_1)
    pcrossmod3d_arr_2 = np.array(pcrossmod3d_arr_2)
    
    np.savez(
        f'data/01_{window_name}_split_nobeam.npz',
        pdata3d_arr_1=pdata3d_arr_1,
        phimod3d_arr_1=phimod3d_arr_1,
        pdata3d_arr_2=pdata3d_arr_2,
        phimod3d_arr_2=phimod3d_arr_2,
        pg3d_arr_1=np.array(pg3d_arr_1),
        pg3d_arr_2=np.array(pg3d_arr_2),
        pgmod3d_arr_1=np.array(pgmod3d_arr_1),
        pgmod3d_arr_2=np.array(pgmod3d_arr_2),
        pcross3d_arr_1=np.array(pcross3d_arr_1),
        pcross3d_arr_2=np.array(pcross3d_arr_2),
        pcrossmod3d_arr_1=np.array(pcrossmod3d_arr_1),
        pcrossmod3d_arr_2=np.array(pcrossmod3d_arr_2),
    )