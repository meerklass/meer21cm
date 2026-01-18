import numpy as np
import matplotlib.pyplot as plt
from meer21cm.plot import plot_map
from meer21cm.util import create_wcs, redshift_to_freq

num_pix_x = 120
num_pix_y = 40

z_min = 0.4
z_max = 1.1
nu_min = redshift_to_freq(z_max)
nu_max = redshift_to_freq(z_min)
nu_resol = 132812.5
num_ch = int((nu_max - nu_min) / nu_resol)
nu_arr = np.linspace(nu_min, nu_min + (num_ch - 1) * nu_resol, num_ch)

wcs = create_wcs(
    ra_cr=150,
    dec_cr=-2.5,
    ngrid=[num_pix_x, num_pix_y],
    resol=0.5,
)

ra_range = [125, 175]
dec_range = [-10.1, 5]

dndz_data = np.load("LRG_dndz.npz")
