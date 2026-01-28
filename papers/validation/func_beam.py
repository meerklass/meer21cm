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
window_name = 'blackmanharris'
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
        sigma_beam_ch=sigma_beam_ch,
        sigma_v_1=100,
        sigma_v_2=100,
    )
    rng = np.random.default_rng(seed)
    noise = rng.normal(0,1,mock.data.shape)
    mock.data = noise.copy()
    mock.trim_map_to_range()
    mock.downres_factor_transverse = 3
    mock.downres_factor_radial = 6
    mock.get_enclosing_box()
    mock.grid_scheme = 'cic'
    noise_rg,_,_ = mock.grid_data_to_field()
    mock.taper_func = getattr(windows, window_name)
    mock.field_1 = noise_rg
    mock.weights_1 = mock.counts_in_box.astype(np.float32)
    mock.apply_taper_to_field(1, axis=[0, 1, 2])
    power_beam_theory = mock.auto_power_3d_1.copy() * mock.beam_attenuation()**2
    mock.data = mock.data.copy()
    mock.W_HI = np.ones_like(mock.W_HI)
    mock.w_HI = np.ones_like(mock.w_HI)
    mock.convolve_data(mock.beam_image)
    mock.trim_map_to_range()
    noise_rg,_,_ = mock.grid_data_to_field()
    mock.field_1 = noise_rg
    mock.weights_1 = mock.counts_in_box.astype(np.float32)
    mock.apply_taper_to_field(1, axis=[0, 1, 2])
    power_beam_noise = mock.auto_power_3d_1.copy()
    return power_beam_theory, power_beam_noise

if __name__ == "__main__":
    power_beam_theory_arr = []
    power_beam_noise_arr = []
    with Pool(16) as p:
        for power_beam_theory, power_beam_noise in p.map(
            get_3d_power, range(128)
        ):
            power_beam_theory_arr.append(power_beam_theory)
            power_beam_noise_arr.append(power_beam_noise)
    power_beam_theory_arr = np.array(power_beam_theory_arr).mean(0)
    power_beam_noise_arr = np.array(power_beam_noise_arr).mean(0)
    np.savez(
        f'data/beam_theory_split_{window_name}.npz', 
        power_beam_theory_arr=power_beam_theory_arr, power_beam_noise_arr=power_beam_noise_arr
    )