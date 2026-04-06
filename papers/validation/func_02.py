# transfer function analysis
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
from meer21cm.power import get_shot_noise_galaxy
from meer21cm.grid import shot_noise_correction_from_gridding
from astropy.cosmology import Planck18
from meer21cm.util import pca_clean
from meer21cm.transfer import TransferFunction, run_tf_calculation_auto, run_tf_calculation_cross
import sys

batch_id = int(sys.argv[-1])

window_name = "blackmanharris"
# generate the foreground template once
# from meer21cm.fg import ForegroundSimulation
# fgsim = ForegroundSimulation(
#    hp_nside=128,
#    wproj=wcs,
#    num_pix_x=num_pix_x,
#    num_pix_y=num_pix_y,
# )
# fg_map = fgsim.fg_wcs_cube(nu_arr)
# np.save('fg_map',fg_map)

fg_map = np.load("fg_map.npy")
#fg_map = np.zeros_like(fg_map)
N_fg = 5
seed_arr = np.array_split(np.arange(1000),10)[batch_id]

def get_3d_power(seed):
    z_func = interp1d(
        z_cen, z_count / dV_arr, kind="linear", bounds_error=False, fill_value=0
    )
    sigma_beam_ch = dish_beam_sigma(13.5, nu_arr)
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
        sigma_beam_ch=sigma_beam_ch,
        sigma_v_1=100,
        sigma_v_2=100,
    )
    mock.taper_func = getattr(windows, window_name)
    mock.W_HI = np.ones_like(mock.W_HI)
    mock.w_HI = np.ones_like(mock.w_HI)
    num_gal = int(mock.survey_volume * n_gal)
    mock.num_discrete_source = num_gal
    mock.downres_factor_transverse = 1 / 2
    mock.downres_factor_radial = 1 / 2
    mock.get_enclosing_box()
    mock.data = mock.propagate_mock_field_to_data(mock.mock_tracer_field_1)
    mock.propagate_mock_tracer_to_gal_cat()
    mock.trim_map_to_range()
    mock.trim_gal_to_range()
    hi_map = mock.data.copy()
    mock.data = fg_map.copy()
    mock.convolve_data(mock.beam_image)
    mock.w_HI = mock.W_HI.astype("float")
    mock.trim_map_to_range()
    fg_map_beam = mock.data.copy()
    tot_map = fg_map_beam + hi_map
    cov_tot, _, eival, eigvec = pca_clean(
        tot_map, 1, weights=mock.w_HI, return_analysis=True, mean_center=True
    )
    res_map, A_mat = pca_clean(
        tot_map,
        N_fg,
        weights=mock.w_HI,
        mean_center=True,
        covariance=cov_tot,
        return_A=True,
    )
    R_mat = np.eye(mock.nu.size) - A_mat @ A_mat.T
    mock.data = hi_map
    mock.downres_factor_transverse = 3
    mock.downres_factor_radial = 6
    mock.get_enclosing_box()
    mock.grid_scheme = "cic"
    himap_rg, _, _ = mock.grid_data_to_field()
    galmap_rg, _, _ = mock.grid_gal_to_field()
    dndz_box = mock.discrete_source_dndz(mock._box_voxel_redshift)
    mock.field_1 = himap_rg
    mock.weights_1 = mock.counts_in_box.astype(np.float32)
    mock.apply_taper_to_field(1, axis=[0, 1, 2])
    mock.include_sky_sampling = [True, False]
    mock.compensate = [True, True]
    mock.include_beam = [True, False]
    mock.field_2 = galmap_rg
    mock.weights_field_2 = dndz_box
    mock.weights_grid_2 = ((dndz_box > 0) * mock.counts_in_box).astype("float")  # test
    # mock.weights_grid_2 = ((dndz_box*mock.counts_in_box)>0).astype('float') # test2
    mock.apply_taper_to_field(2, axis=[0, 1, 2])
    shot_noise = get_shot_noise_galaxy(
        galmap_rg,
        mock.box_len,
        mock.weights_grid_2,
        mock.weights_field_2,
    ) * shot_noise_correction_from_gridding(mock.box_ndim, mock.grid_scheme)
    pdata3d = mock.auto_power_3d_1
    phimod3d = mock.auto_power_tracer_1_model
    pg3d = mock.auto_power_3d_2 - shot_noise
    pgmod3d = mock.auto_power_tracer_2_model
    pcross3d = mock.cross_power_3d
    pcrossmod3d = mock.cross_power_tracer_model
    mock.data = res_map
    resmap_rg, _, _ = mock.grid_data_to_field()
    mock.field_1 = resmap_rg
    mock.weights_1 = mock.counts_in_box.astype(np.float32)
    mock.apply_taper_to_field(1, axis=[0, 1, 2])
    mock.field_2 = galmap_rg
    mock.weights_field_2 = dndz_box
    mock.weights_grid_2 = ((dndz_box > 0) * mock.counts_in_box).astype("float")  # test
    # mock.weights_grid_2 = ((dndz_box*mock.counts_in_box)>0).astype('float') # test2
    mock.apply_taper_to_field(2, axis=[0, 1, 2])
    phiclean3d = mock.auto_power_3d_1
    pxclean3d = mock.cross_power_3d
    mock.k1dbins = k1dbins
    tf = TransferFunction(
        mock,
        N_fg=N_fg,
        highres_sim=None,
        # generate mock data on a high-resolution grid, then to average it to sky map for injection
        upres_transverse=2,
        upres_radial=2,
        uncleaned_data=tot_map,  # inject into the map data to reperform PCA
        #R_mat=R_mat,
        num_process=1,  # number of available cpus to run parallel calculation
        pca_map_weights=mock.W_HI.astype("float"),
        unmask_during_mock=True,
        discrete_source_dndz=[z_cen, z_count / dV_arr,],
    )
    #arglist = tf.get_arg_list_for_parallel_auto(
    arglist = tf.get_arg_list_for_parallel_cross(
        np.array([mock.seed + 10000]),  # make sure it is a different seed
        return_power_3d=True,
        return_power_1d=False,
    )
    #result = run_tf_calculation_auto(*arglist[0])
    result = run_tf_calculation_cross(*arglist[0])
    power_tf_before = result[1]
    power_tf_after = result[2]
    np.savez(
        f"/scratch3/users/ztchen/validation/02/{seed}_cross.npz",
        phi3d_arr=np.array(pdata3d),
        phimod3d_arr=np.array(phimod3d),
        pg3d_arr=np.array(pg3d),
        pgmod3d_arr=np.array(pgmod3d),
        pcross3d_arr=np.array(pcross3d),
        pcrossmod3d_arr=np.array(pcrossmod3d),
        phiclean3d_arr=phiclean3d,
        pcrossclean3d_arr=pxclean3d,
        power_tf_before=power_tf_before,
        power_tf_after=power_tf_after,
    )
    return 1


if __name__ == "__main__":
    with Pool(10) as p:
        p.map(get_3d_power, seed_arr)
