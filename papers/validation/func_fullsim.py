# full simulation
import numpy as np
from specs import *
from scipy.interpolate import interp1d
from meer21cm.telescope import dish_beam_sigma
import scipy.signal.windows as windows
from multiprocessing import Pool
from meer21cm.power import get_shot_noise_galaxy
from meer21cm.grid import shot_noise_correction_from_gridding
from astropy.cosmology import Planck18
from meer21cm.util import pca_clean

# generate the foreground template once
# from meer21cm.fg import ForegroundSimulation
# fgsim = ForegroundSimulation(
#    hp_nside=128,
#    wproj=wcs,
#    num_pix_x=num_pix_x,
#    num_pix_y=num_pix_y,
#    backend='pysm',
# )
# fg_map = fgsim.fg_wcs_cube(nu_arr)
# np.save('fg_map',fg_map)
fg_map = np.load("fg_map.npy")


def get_3d_power(seed):
    mock = get_mock(seed)
    num_gal = int(mock.survey_volume * n_gal)
    mock.num_discrete_source = num_gal
    # remove buffer for a bigger simulation box
    mock.W_HI = np.ones_like(mock.W_HI)
    mock.w_HI = np.ones_like(mock.w_HI)
    mock.downres_factor_transverse = sim_upres_transverse
    mock.downres_factor_radial = sim_upres_radial
    mock.get_enclosing_box()
    mock.data = mock.propagate_mock_field_to_data(mock.mock_tracer_field_1)
    mock.propagate_mock_tracer_to_gal_cat()
    mock.trim_map_to_range()
    mock.trim_gal_to_range()
    mock.downres_factor_transverse = ps_downres_transverse
    mock.downres_factor_radial = ps_downres_radial
    mock.get_enclosing_box()
    # keep a copy of the hi map and the count map
    hi_map = mock.data.copy()
    count_map = mock.w_HI.copy()
    mask_map = mock.W_HI.copy()
    mock.grid_scheme = grid_scheme

    # get the no-foreground power
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
    mock.weights_grid_2 = ((dndz_box * mock.counts_in_box) > 0).astype("float")
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

    # add in the foreground
    mock.data = fg_map.copy()
    mock.w_HI = np.ones_like(mock.w_HI)
    mock.W_HI = np.ones_like(mock.W_HI)
    # convolve with the beam
    mock.convolve_data(mock.beam_image)
    mock.trim_map_to_range()
    fg_map_beam = mock.data.copy()
    tot_map = fg_map_beam + hi_map
    mock.w_HI = count_map
    mock.W_HI = mask_map
    cov_tot, _, eival, eigvec = pca_clean(
        tot_map,
        1,
        weights=mock.W_HI,
        return_analysis=True,
        mean_center=True,
    )
    # perform PCA cleaning
    res_map, A_mat = pca_clean(
        tot_map,
        N_fg,
        weights=mock.W_HI,
        mean_center=True,
        covariance=cov_tot,
        return_A=True,
        ignore_nan=True,
    )
    R_mat = np.eye(mock.nu.size) - A_mat @ A_mat.T
    mock.data = res_map
    resmap_rg, _, _ = mock.grid_data_to_field()
    mock.field_1 = resmap_rg
    mock.weights_1 = mock.counts_in_box.astype(np.float32)
    mock.apply_taper_to_field(1, axis=[0, 1, 2])
    mock.field_2 = galmap_rg
    mock.weights_field_2 = dndz_box
    mock.weights_grid_2 = ((dndz_box * mock.counts_in_box) > 0).astype("float")  # test2
    mock.apply_taper_to_field(2, axis=[0, 1, 2])
    phiclean3d = mock.auto_power_3d_1
    pxclean3d = mock.cross_power_3d
    np.savez(
        f"/scratch3/users/ztchen/validation/full_sim_seed_{seed}.npz",
        phi3d=pdata3d,
        phimod3d=phimod3d,
        pg3d=pg3d,
        pgmod3d=pgmod3d,
        pcross3d=pcross3d,
        pcrossmod3d=pcrossmod3d,
        phiclean3d=phiclean3d,
        pxclean3d=pxclean3d,
        R_mat=R_mat,
    )
    return 1


if __name__ == "__main__":
    # run the simulations
    with Pool(10) as p:
        p.map(get_3d_power, range(512))
