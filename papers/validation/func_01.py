# project to sky coordinates and back
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

# one process uses about 64GB mem for 0.4-1.1
# mpi needed
# window_name = "tukey"
window_name = "blackmanharris"


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
        # sigma_beam_ch=sigma_beam_ch,
        sigma_v_1=100,
        sigma_v_2=100,
    )
    comov_dist = Planck18.comoving_distance(mock.z_ch).value
    sigma_beam_new = 1 / comov_dist * sigma_beam_ch
    sigma_beam_new *= sigma_beam_ch.mean() / sigma_beam_new.mean()
    mock.sigma_beam_ch = sigma_beam_new
    mock.taper_func = getattr(windows, window_name)
    num_gal = int(mock.survey_volume * n_gal)
    mock.num_discrete_source = num_gal
    mock.W_HI = np.ones_like(mock.W_HI)
    mock.w_HI = np.ones_like(mock.w_HI)
    mock.downres_factor_transverse = 1 / 2
    mock.downres_factor_radial = 1 / 2
    mock.get_enclosing_box()
    mock.data = mock.propagate_mock_field_to_data(mock.mock_tracer_field_1)
    mock.propagate_mock_tracer_to_gal_cat()
    mock.trim_map_to_range()
    mock.trim_gal_to_range()
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
    # mock.weights_grid_2 = ((dndz_box>0)*mock.counts_in_box).astype('float') # test
    mock.weights_grid_2 = ((dndz_box * mock.counts_in_box) > 0).astype("float")  # test2
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
    print(seed)
    return pdata3d, phimod3d, pg3d, pgmod3d, pcross3d, pcrossmod3d


if __name__ == "__main__":
    # run the simulations
    pdata3d_arr = []
    phimod3d_arr = []
    pg3d_arr = []
    pgmod3d_arr = []
    pcross3d_arr = []
    pcrossmod3d_arr = []
    with Pool(7) as p:
        for pdata3d, phimod3d, pg3d, pgmod3d, pcross3d, pcrossmod3d in p.map(
            get_3d_power, range(256)
        ):
            pdata3d_arr.append(pdata3d)
            phimod3d_arr.append(phimod3d)
            pg3d_arr.append(pg3d)
            pgmod3d_arr.append(pgmod3d)
            pcross3d_arr.append(pcross3d)
            pcrossmod3d_arr.append(pcrossmod3d)
    np.savez(
        f"/scratch3/users/ztchen/validation/01_{window_name}_2",
        pdata3d_arr=np.array(pdata3d_arr),
        phimod3d_arr=np.array(phimod3d_arr),
        pg3d_arr=np.array(pg3d_arr),
        pgmod3d_arr=np.array(pgmod3d_arr),
        pcross3d_arr=np.array(pcross3d_arr),
        pcrossmod3d_arr=np.array(pcrossmod3d_arr),
    )
