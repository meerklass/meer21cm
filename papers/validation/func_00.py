import numpy as np
from meer21cm.mock import MockSimulation
from specs import *
from meer21cm.grid import project_particle_to_regular_grid
from scipy.interpolate import interp1d
from meer21cm.power import get_shot_noise_galaxy
from multiprocessing import Pool
from meer21cm.power import bin_3d_to_cy, bin_3d_to_1d
import scipy.signal.windows as windows

def get_3d_power(
    seed,
):
    # z_count is in redshift bins, convert to Mpc-3 unit
    z_func = interp1d(z_cen, z_count/dV_arr, kind="linear", bounds_error=False, fill_value=0)
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
        sigma_v_1=100,
        sigma_v_2=100,
    )
    num_gal = int(mock.survey_volume * n_gal)
    mock.num_discrete_source = num_gal
    mock.get_enclosing_box()
    mock.field_1 = mock.mock_tracer_field_1
    mock.weights_1 = np.ones_like(mock.field_1)
    mock.taper_func = windows.tukey
    mock.apply_taper_to_field(1, axis=[0, 1, 2])
    pdata3d = mock.auto_power_3d_1
    phimod3d = mock.auto_power_tracer_1_model
    _, _, gal_counts = project_particle_to_regular_grid(
        mock.mock_tracer_position_in_box,
        mock.box_len,
        mock.box_ndim,
    )
    dndz_box = mock.discrete_source_dndz(mock._box_voxel_redshift)
    mock.field_2 = gal_counts
    mock.weights_field_2 = dndz_box
    mock.weights_grid_2 = np.ones_like(gal_counts)
    mock.apply_taper_to_field(2, axis=[0, 1, 2])
    mock.mean_center_2 = True
    mock.unitless_2 = True
    mock.compensate = [False, False]
    shot_noise = get_shot_noise_galaxy(
        gal_counts, mock.box_len, mock.weights_grid_2, mock.weights_field_2
    )
    pg3d = mock.auto_power_3d_2 - shot_noise
    pgmod3d = mock.auto_power_tracer_2_model
    pcross3d = mock.cross_power_3d
    pcrossmod3d = mock.cross_power_tracer_model
    return pdata3d, phimod3d, pg3d, pgmod3d, pcross3d, pcrossmod3d


def main():
    pdata3d_arr = []
    phimod3d_arr = []
    pg3d_arr = []
    pgmod3d_arr = []
    pcross3d_arr = []
    pcrossmod3d_arr = []
    with Pool(8) as p:
        for pdata3d, phimod3d, pg3d, pgmod3d, pcross3d, pcrossmod3d in p.map(
            get_3d_power, range(512)
        ):
            pdata3d_arr.append(pdata3d)
            phimod3d_arr.append(phimod3d)
            pg3d_arr.append(pg3d)
            pgmod3d_arr.append(pgmod3d)
            pcross3d_arr.append(pcross3d)
            pcrossmod3d_arr.append(pcrossmod3d)
    pdata3d_arr = np.array(pdata3d_arr)
    phimod3d_arr = np.array(phimod3d_arr)[0][None]
    pg3d_arr = np.array(pg3d_arr)
    pgmod3d_arr = np.array(pgmod3d_arr)[0][None]
    pcross3d_arr = np.array(pcross3d_arr)
    pcrossmod3d_arr = np.array(pcrossmod3d_arr)[0][None]
    
    np.savez(
        'data/00.npz',
        pdata3d_arr = pdata3d_arr,
        phimod3d_arr = phimod3d_arr,
        pg3d_arr = pg3d_arr,
        pgmod3d_arr = pgmod3d_arr,
        pcross3d_arr = pcross3d_arr,
        pcrossmod3d_arr = pcrossmod3d_arr,
    )


if __name__ == "__main__":
    # run the simulations
    main()