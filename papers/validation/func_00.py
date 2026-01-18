import numpy as np
from meer21cm.mock import MockSimulation
from specs import *
from meer21cm.grid import project_particle_to_regular_grid
from scipy.interpolate import interp1d
from meer21cm.power import get_shot_noise_galaxy
from multiprocessing import Pool
from meer21cm.power import bin_3d_to_cy, bin_3d_to_1d


def get_3d_power(
    seed,
):
    dndz_arr = np.load("LRG_dndz.npz")
    z_bin = dndz_arr["z_bin"]
    z_count = dndz_arr["z_count"]
    z_cen = (z_bin[:-1] + z_bin[1:]) / 2
    z_func = interp1d(z_cen, z_count, kind="linear", bounds_error=False, fill_value=0)
    mock = MockSimulation(
        wproj=wcs,
        num_pix_x=num_pix_x,
        num_pix_y=num_pix_y,
        ra_range=ra_range,
        dec_range=dec_range,
        nu=nu_arr,
        discrete_source_dndz=z_func,
        num_discrete_source=int(1e6),
        seed=seed,
        tracer_bias_2=1.0,
        tracer_bias_1=1.0,
        mean_amp_1="average_hi_temp",
        omega_hi=5e-4,
    )
    mock.get_enclosing_box()
    mock.field_1 = mock.mock_tracer_field_1
    mock.weights_1 = np.ones_like(mock.field_1)
    mock.apply_taper_to_field(1, axis=[0, 1, 2])
    pdata3d = mock.auto_power_3d_1
    phimod3d = mock.auto_power_tracer_1_model
    # hack
    # mock._box_voxel_redshift = np.ones(mock.box_ndim) * mock.z
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


if __name__ == "__main__":
    # run the simulations
    pdata3d_arr = []
    phimod3d_arr = []
    pg3d_arr = []
    pgmod3d_arr = []
    pcross3d_arr = []
    pcrossmod3d_arr = []
    with Pool(10) as p:
        for pdata3d, phimod3d, pg3d, pgmod3d, pcross3d, pcrossmod3d in p.map(
            get_3d_power, range(100)
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
    mock.get_enclosing_box()
    mock.k1dbins = np.linspace(0.003, 0.27, 41)
    mock.kperpbins = np.linspace(0, 0.16, 33)
    mock.kparabins = np.linspace(0, 0.5, 51)
    # bin the power spectra
    pdatacy_arr = bin_3d_to_cy(pdata3d_arr, mock.k_perp, mock.kperpbins, vectorize=True)
    pdatacy_arr = bin_3d_to_cy(
        pdatacy_arr, np.abs(mock.k_para), mock.kparabins, vectorize=True
    )
    phimodcy_arr = bin_3d_to_cy(
        phimod3d_arr, mock.k_perp, mock.kperpbins, vectorize=True
    )
    phimodcy_arr = bin_3d_to_cy(
        phimodcy_arr, np.abs(mock.k_para), mock.kparabins, vectorize=True
    )[0]
    pgcy_arr = bin_3d_to_cy(pg3d_arr, mock.k_perp, mock.kperpbins, vectorize=True)
    pgcy_arr = bin_3d_to_cy(
        pgcy_arr, np.abs(mock.k_para), mock.kparabins, vectorize=True
    )
    pgmodcy_arr = bin_3d_to_cy(pgmod3d_arr, mock.k_perp, mock.kperpbins, vectorize=True)
    pgmodcy_arr = bin_3d_to_cy(
        pgmodcy_arr, np.abs(mock.k_para), mock.kparabins, vectorize=True
    )[0]
    pcrosscy_arr = bin_3d_to_cy(
        pcross3d_arr, mock.k_perp, mock.kperpbins, vectorize=True
    )
    pcrosscy_arr = bin_3d_to_cy(
        pcrosscy_arr, np.abs(mock.k_para), mock.kparabins, vectorize=True
    )
    pcrossmodcy_arr = bin_3d_to_cy(
        pcrossmod3d_arr, mock.k_perp, mock.kperpbins, vectorize=True
    )
    pcrossmodcy_arr = bin_3d_to_cy(
        pcrossmodcy_arr, np.abs(mock.k_para), mock.kparabins, vectorize=True
    )[0]
    fig, axes = plt.subplots(1, 3)
    axes[0].pcolormesh(
        mock.kperpbins,
        mock.kparabins,
        np.log10(pdatacy_arr.mean(axis=0).T),
    )
    axes[1].pcolormesh(
        mock.kperpbins,
        mock.kparabins,
        np.log10(phimodcy_arr.T),
    )
    im = axes[2].pcolormesh(
        mock.kperpbins,
        mock.kparabins,
        (pdatacy_arr.mean(axis=0).T) / (phimodcy_arr.T),
        vmin=0.95,
        vmax=1.05,
        cmap="bwr",
    )
    plt.colorbar(im)
    plt.savefig("plots/pdatacy_pmodcy_00.png")
    fig, axes = plt.subplots(1, 3)
    axes[0].pcolormesh(
        mock.kperpbins,
        mock.kparabins,
        np.log10(pgcy_arr.mean(axis=0).T),
    )
    axes[1].pcolormesh(
        mock.kperpbins,
        mock.kparabins,
        np.log10(pgmodcy_arr.T),
    )
    im = axes[2].pcolormesh(
        mock.kperpbins,
        mock.kparabins,
        (pgcy_arr.mean(axis=0).T) / (pgmodcy_arr.T),
        vmin=0.95,
        vmax=1.05,
        cmap="bwr",
    )
    plt.colorbar(im)
    plt.savefig("plots/pgcy_pmodcy_00.png")
    fig, axes = plt.subplots(1, 3)
    axes[0].pcolormesh(
        mock.kperpbins,
        mock.kparabins,
        np.log10(pcrosscy_arr.mean(axis=0).T),
    )
    axes[1].pcolormesh(
        mock.kperpbins,
        mock.kparabins,
        np.log10(pcrossmodcy_arr.T),
    )
    im = axes[2].pcolormesh(
        mock.kperpbins,
        mock.kparabins,
        (pcrosscy_arr.mean(axis=0).T) / (pcrossmodcy_arr.T),
        vmin=0.95,
        vmax=1.05,
        cmap="bwr",
    )
    plt.colorbar(im)
    plt.savefig("plots/pcrosscy_pcrossmodcy_00.png")
    pdata1d_arr, keff, nmodes = bin_3d_to_1d(
        pdata3d_arr, mock.kmode, mock.k1dbins, vectorize=True
    )
    phimod1d_arr, keff, nmodes = bin_3d_to_1d(
        phimod3d_arr, mock.kmode, mock.k1dbins, vectorize=True
    )
    phimod1d_arr = phimod1d_arr[0]
    pg1d_arr, keff, nmodes = bin_3d_to_1d(
        pg3d_arr, mock.kmode, mock.k1dbins, vectorize=True
    )
    pgmod1d_arr, keff, nmodes = bin_3d_to_1d(
        pgmod3d_arr, mock.kmode, mock.k1dbins, vectorize=True
    )
    pgmod1d_arr = pgmod1d_arr[0]
    pcross1d_arr, keff, nmodes = bin_3d_to_1d(
        pcross3d_arr, mock.kmode, mock.k1dbins, vectorize=True
    )
    pcrossmod1d_arr, keff, nmodes = bin_3d_to_1d(
        pcrossmod3d_arr, mock.kmode, mock.k1dbins, vectorize=True
    )
    pcrossmod1d_arr = pcrossmod1d_arr[0]
    plt.figure()
    plt.errorbar(
        keff,
        (pdata1d_arr.mean(axis=0)) * keff,
        yerr=(pdata1d_arr.std(axis=0)) * keff,
        label="mock HI",
    )
    plt.plot(keff, (phimod1d_arr) * keff, label="model", ls="--")
    # plt.plot(keff,np.log10(pg1d_arr.mean(axis=0)),label='mock galaxy',ls=':')
    plt.legend()
    plt.savefig("plots/pdata1d_pmod1d_00.png")
    plt.figure()
    plt.errorbar(
        keff,
        (pg1d_arr.mean(axis=0)) * keff,
        yerr=(pg1d_arr.std(axis=0)) * keff,
        label="mock galaxy",
    )
    plt.plot(keff, (pgmod1d_arr) * keff, label="model", ls="--")
    # plt.plot(keff,np.log10(pg1d_arr.mean(axis=0)),label='mock galaxy',ls=':')
    plt.legend()
    plt.savefig("plots/pg1d_pmod1d_00.png")
    plt.figure()
    plt.errorbar(
        keff,
        (pcross1d_arr.mean(axis=0)) * keff,
        yerr=(pcross1d_arr.std(axis=0)) * keff,
        label="mock cross",
    )
    plt.plot(keff, (pcrossmod1d_arr) * keff, label="model", ls="--")
    plt.legend()
    plt.savefig("plots/pcross1d_pcrossmod1d_00.png")
