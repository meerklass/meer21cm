"""Galaxy catalogue weights, the FKP field, and the galaxy auto-power."""

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table
from scipy.interpolate import interp1d

from meer21cm import MockSimulation, PowerSpectrum, Specification
from meer21cm.mock import selection_sampling_weight
from meer21cm.io import read_catalogue_fits
from meer21cm.grid import shot_noise_correction_from_gridding
from meer21cm.power import get_shot_noise_galaxy
from meer21cm.power_ops import fkp_count_field, get_shot_noise_counts
from meer21cm.util import angle_in_range, f_21, freq_to_redshift, redshift_to_freq


def _survey_ps(test_gal_fits, test_W):
    ps = PowerSpectrum(
        gal_file=test_gal_fits,
        survey="meerklass_2021",
        band="L",
    )
    ps.W_HI = (test_W * ps.nu[None, None, :]) > 0
    ps.data = ps.W_HI
    ps.w_HI = ps.W_HI
    return PowerSpectrum(
        gal_file=test_gal_fits,
        data=ps.data,
        map_has_sampling=ps.W_HI,
        weights_map_pixel=ps.w_HI,
        init_box_from_map_data=True,
        include_sky_sampling=[True, True],
        survey="meerklass_2021",
        band="L",
        tracer_bias_2=1.0,
        grid_scheme="nnb",
    )


def test_unit_weights_match_default_paint(test_gal_fits, test_W):
    ps = _survey_ps(test_gal_fits, test_W)
    ps.read_gal_cat()
    assert ps.weights_gal is None
    ps.grid_gal_to_field()
    field = np.array(ps.field_2, copy=True)
    weights_field = np.array(ps.weights_field_2, copy=True)
    weights_grid = np.array(ps.weights_grid_2, copy=True)
    ones = np.ones(ps.ra_gal.size)
    ps.grid_gal_to_field(weights=ones)
    assert np.allclose(ps.field_2, field)
    assert np.allclose(ps.weights_field_2, weights_field)
    assert np.allclose(ps.weights_grid_2, weights_grid)
    ps.weights_gal = ones
    ps.grid_gal_to_field()
    assert np.allclose(ps.field_2, field)
    assert np.allclose(ps.weights_field_2, weights_field)
    assert np.allclose(ps.weights_grid_2, weights_grid)


def test_painted_unit_weight_sum_equals_galaxies_in_box(test_gal_fits, test_W):
    ps = _survey_ps(test_gal_fits, test_W)
    ps.get_enclosing_box()
    ndim = np.asarray(ps.box_ndim, dtype=int)
    length = np.asarray(ps.box_len, dtype=float)
    centers = np.array(
        [
            [ndim[0] // 2, ndim[1] // 2, ndim[2] // 2],
            [ndim[0] // 2 + 1, ndim[1] // 2, ndim[2] // 3],
            [ndim[0] // 3, ndim[1] // 2 + 1, ndim[2] // 2 + 1],
        ],
        dtype=float,
    )
    pos = (centers + 0.5) * (length / ndim)
    ra, dec, z, _ = ps.ra_dec_z_for_coord_in_box(pos)
    counts = ps.paint_catalogue_counts(ra, dec, z)
    assert counts.sum() == pytest.approx(pos.shape[0])


def test_trim_slices_weights(test_gal_fits):
    sp = Specification(survey="meerklass_2021", band="L", gal_file=test_gal_fits)
    sp.read_gal_cat()
    n_in = sp.ra_gal.size
    sp._z_gal = np.array(sp.z_gal, copy=True)
    sp._z_gal[0] = 10.0
    sp.weights_gal = np.arange(n_in, dtype=float)
    dropped = float(sp.weights_gal[0])
    sp.ra_rand = np.array(sp.ra_gal[:2], copy=True)
    sp.dec_rand = np.array(sp.dec_gal[:2], copy=True)
    sp.z_rand = np.array([10.0, sp.z_gal[1]])
    sp.weights_rand = np.array([3.0, 4.0])
    sp.trim_gal_to_range()
    assert sp.weights_gal.size == sp.ra_gal.size == n_in - 1
    assert dropped not in sp.weights_gal
    assert sp.ra_rand.size == sp.dec_rand.size == sp.z_rand.size == 1
    assert sp.weights_rand[0] == pytest.approx(4.0)
    assert sp.z_rand[0] == pytest.approx(sp.z_gal[0])
    sp.weights_gal = None
    assert sp.weights_gal is None


def test_read_catalogue_list_keeps_weights(tmp_path):
    data_path = tmp_path / "data.fits"
    rand_path = tmp_path / "rand.fits"
    Table(
        {
            "RA": [150.0, 151.0],
            "DEC": [0.0, 1.0],
            "Z": [0.4, 0.5],
            "WEIGHT": [2.0, 0.5],
        }
    ).write(data_path, overwrite=True)
    Table(
        {
            "RA": [150.0, 151.0, 152.0, 153.0],
            "DEC": [0.0, 1.0, 0.5, -0.5],
            "Z": [0.4, 0.5, 0.45, 0.55],
            "WEIGHT": [1.0, 1.0, 1.0, 3.0],
        }
    ).write(rand_path, overwrite=True)
    ra, dec, z, weight = read_catalogue_fits(
        [data_path, rand_path], weight_col="WEIGHT"
    )
    assert ra.size == 6
    assert np.allclose(weight, [2.0, 0.5, 1.0, 1.0, 1.0, 3.0])
    empty = read_catalogue_fits([])
    assert all(column.size == 0 for column in empty)
    unit = read_catalogue_fits([data_path], weight_col=None)
    assert np.allclose(unit[3], 1.0)
    sp = Specification(survey="meerklass_2021", band="L", gal_file=str(data_path))
    sp.read_gal_cat(weight_col="WEIGHT", trim=False)
    assert np.allclose(sp.weights_gal, [2.0, 0.5])
    with fits.open(data_path) as hdul:
        assert "WEIGHT_FKP" not in hdul[1].columns.names


def _interior_radecz(ps, centers):
    ndim = np.asarray(ps.box_ndim, dtype=int)
    length = np.asarray(ps.box_len, dtype=float)
    pos = (np.asarray(centers, dtype=float) + 0.5) * (length / ndim)
    ra, dec, z, _ = ps.ra_dec_z_for_coord_in_box(pos)
    return ra, dec, z


def test_construct_fkp_is_data_minus_alpha_randoms(test_gal_fits, test_W):
    ps = _survey_ps(test_gal_fits, test_W)
    ps.get_enclosing_box()
    ndim = np.asarray(ps.box_ndim, dtype=int)
    data_ra, data_dec, data_z = _interior_radecz(
        ps,
        [
            [ndim[0] // 2, ndim[1] // 2, ndim[2] // 2],
            [ndim[0] // 2 + 1, ndim[1] // 2, ndim[2] // 3],
        ],
    )
    rand_ra, rand_dec, rand_z = _interior_radecz(
        ps,
        [
            [ndim[0] // 3, ndim[1] // 2, ndim[2] // 2],
            [ndim[0] // 3 + 1, ndim[1] // 2 + 1, ndim[2] // 2],
            [ndim[0] // 2, ndim[1] // 3, ndim[2] // 4],
            [ndim[0] // 2, ndim[1] // 3 + 1, ndim[2] // 4 + 1],
        ],
    )
    data_counts = ps.paint_catalogue_counts(data_ra, data_dec, data_z)
    random_counts = ps.paint_catalogue_counts(rand_ra, rand_dec, rand_z)
    field_ref, alpha_ref = fkp_count_field(data_counts, random_counts)
    freq = f_21 / (1.0 + np.asarray(data_z, dtype=float))
    field, window, data_paint = ps.grid_gal_to_field(
        radecfreq=(data_ra, data_dec, freq),
        construct_fkp=True,
        random_radecz=(rand_ra, rand_dec, rand_z),
    )
    assert np.allclose(field, field_ref)
    assert np.allclose(ps.field_2, field_ref)
    assert np.allclose(data_paint, data_counts)
    assert ps.field_2_has_random is True
    assert ps.field_2_alpha == pytest.approx(alpha_ref)
    assert np.allclose(ps.field_2_D, data_counts)
    assert np.allclose(ps.field_2_R, random_counts)
    assert np.allclose(window, alpha_ref * random_counts)
    assert np.allclose(ps.weights_field_2, alpha_ref * random_counts)
    assert np.allclose(ps.weights_grid_2, 1.0)
    assert ps.mean_center_2 is False
    assert ps.unitless_2 is False
    assert field.sum() == pytest.approx(0.0, abs=1e-8)


def test_construct_fkp_uses_stored_randoms(test_gal_fits, test_W):
    ps = _survey_ps(test_gal_fits, test_W)
    ps.get_enclosing_box()
    ndim = np.asarray(ps.box_ndim, dtype=int)
    data_ra, data_dec, data_z = _interior_radecz(
        ps, [[ndim[0] // 2, ndim[1] // 2, ndim[2] // 2]]
    )
    rand_ra, rand_dec, rand_z = _interior_radecz(
        ps,
        [
            [ndim[0] // 3, ndim[1] // 2, ndim[2] // 2],
            [ndim[0] // 2, ndim[1] // 3, ndim[2] // 3],
        ],
    )
    ps.ra_rand = rand_ra
    ps.dec_rand = rand_dec
    ps.z_rand = rand_z
    freq = f_21 / (1.0 + np.asarray(data_z, dtype=float))
    field_attr, _, _ = ps.grid_gal_to_field(
        radecfreq=(data_ra, data_dec, freq),
        construct_fkp=True,
    )
    field_arg, _, _ = ps.grid_gal_to_field(
        radecfreq=(data_ra, data_dec, freq),
        construct_fkp=True,
        random_radecz=(rand_ra, rand_dec, rand_z),
    )
    assert np.allclose(field_attr, field_arg)


def test_construct_fkp_requires_randoms(test_gal_fits, test_W):
    ps = _survey_ps(test_gal_fits, test_W)
    ps.get_enclosing_box()
    ndim = np.asarray(ps.box_ndim, dtype=int)
    ra, dec, z = _interior_radecz(ps, [[ndim[0] // 2, ndim[1] // 2, ndim[2] // 2]])
    freq = f_21 / (1.0 + np.asarray(z, dtype=float))
    with pytest.raises(ValueError, match="random catalogue"):
        ps.grid_gal_to_field(
            radecfreq=(ra, dec, freq),
            construct_fkp=True,
        )


def test_get_shot_noise_galaxy():
    gal_count = np.ones(100000)
    box_len = [1, 1, 1]
    shot_noise = get_shot_noise_galaxy(gal_count, box_len)
    assert np.allclose(shot_noise, 1e-5)


def test_shot_noise_attribute_matches_legacy_and_fkp():
    counts = np.array(
        [
            [[1.0, 0.0, 2.0], [0.0, 1.0, 0.0], [2.0, 1.0, 0.0]],
            [[0.0, 2.0, 1.0], [1.0, 0.0, 1.0], [0.0, 0.0, 2.0]],
            [[1.0, 1.0, 0.0], [2.0, 0.0, 1.0], [0.0, 2.0, 1.0]],
        ]
    )
    box_len = np.array([10.0, 20.0, 30.0])
    grid_w = np.array(
        [
            [[1.0, 0.5, 1.0], [0.8, 1.0, 0.4], [1.0, 0.7, 1.0]],
            [[0.6, 1.0, 0.9], [1.0, 0.3, 1.0], [0.5, 1.0, 0.8]],
            [[1.0, 0.2, 1.0], [0.9, 1.0, 0.6], [1.0, 0.4, 1.0]],
        ]
    )
    field_w = np.array(
        [
            [[1.0, 1.0, 0.5], [1.0, 0.2, 1.0], [0.7, 1.0, 1.0]],
            [[1.0, 0.4, 1.0], [0.8, 1.0, 1.0], [1.0, 0.6, 1.0]],
            [[0.3, 1.0, 1.0], [1.0, 1.0, 0.9], [1.0, 1.0, 0.5]],
        ]
    )
    ps = PowerSpectrum(
        np.ones_like(counts),
        box_len,
        field_2=counts,
        weights_grid_2=grid_w,
        weights_field_2=field_w,
        mean_center_2=True,
        unitless_2=True,
        grid_scheme="cic",
        compensate=[False, False],
        include_beam=[False, False],
        include_sky_sampling=[False, False],
    )
    correction = shot_noise_correction_from_gridding(ps.box_ndim, "cic")
    legacy = get_shot_noise_galaxy(counts, box_len, grid_w, field_w) * correction
    assert np.allclose(ps.shot_noise_2, legacy)

    random_counts = np.array(
        [
            [[4.0, 5.0, 4.0], [6.0, 4.0, 5.0], [4.0, 7.0, 4.0]],
            [[5.0, 4.0, 6.0], [4.0, 5.0, 4.0], [8.0, 4.0, 5.0]],
            [[4.0, 6.0, 4.0], [5.0, 4.0, 7.0], [4.0, 5.0, 4.0]],
        ]
    )
    alpha = float(counts.sum() / random_counts.sum())
    ps.field_2_D = counts
    ps.field_2_R = random_counts
    ps.field_2_alpha = alpha
    ps.field_2_has_random = True
    ps.weights_field_2 = alpha * random_counts
    ps.weights_grid_2 = grid_w
    ps.set_corr_type("gal", 2)
    assert ps.mean_center_2 is False
    assert ps.unitless_2 is False
    assert np.allclose(ps.field_2, counts - alpha * random_counts)
    with pytest.raises(ValueError, match="field_2_D"):
        ps.field_2 = counts
    window = grid_w * (alpha * random_counts)
    amplitude = (
        np.prod(box_len)
        / counts.size
        * np.sum(grid_w**2 * (counts + alpha**2 * random_counts))
        / np.sum(window**2)
    )
    assert np.allclose(ps.shot_noise_2, amplitude * correction)
    ps.field_2_has_random = False
    assert ps.mean_center_2 is True
    assert ps.unitless_2 is True

    assert ps.field_1_R == 0.0
    assert ps.field_1_has_random is False
    ps.field_1_D = counts
    ps.field_1_R = random_counts
    ps.field_1_alpha = alpha
    ps.field_1_has_random = True
    assert ps.field_1_has_random is True
    ps.weights_field_1 = alpha * random_counts
    ps.weights_1 = grid_w
    assert np.allclose(ps.field_1, counts - alpha * random_counts)
    assert ps.mean_center_1 is False
    assert ps.unitless_1 is False
    assert np.allclose(ps.shot_noise_1, amplitude * correction)


def test_flat_fkp_shot_noise_is_one_plus_alpha_over_nbar():
    counts = np.full((4, 4, 4), 2.0)
    random_counts = np.full_like(counts, 8.0)
    box_len = np.array([40.0, 40.0, 40.0])
    field, alpha = fkp_count_field(counts, random_counts)
    assert alpha == pytest.approx(0.25)
    assert field.sum() == pytest.approx(0.0, abs=1e-8)
    cell_volume = np.prod(box_len) / counts.size
    expected = (1.0 + alpha) / (2.0 / cell_volume)
    amplitude = get_shot_noise_counts(
        counts,
        box_len,
        weights_field=alpha * random_counts,
        random_counts=random_counts,
        alpha=alpha,
    )
    assert amplitude == pytest.approx(expected)
    ps = PowerSpectrum(
        np.ones_like(counts),
        box_len,
        field_2=counts,
        grid_scheme="cic",
        compensate=[False, False],
        include_beam=[False, False],
        include_sky_sampling=[False, False],
    )
    ps.field_2_D = counts
    ps.field_2_R = random_counts
    ps.field_2_alpha = alpha
    ps.field_2_has_random = True
    ps.weights_field_2 = alpha * random_counts
    ps.weights_grid_2 = np.ones_like(counts)
    correction = shot_noise_correction_from_gridding(ps.box_ndim, "cic")
    assert correction[0, 0, 0] == pytest.approx(1.0)
    assert np.allclose(ps.shot_noise_2, expected * correction)

    ps.field_2_R = None
    ps.field_2_alpha = 3.0
    ps.weights_field_2 = np.ones_like(counts)
    assert ps.field_2_R == 0.0
    assert np.allclose(ps.field_2, counts)
    data_only = 2.0 * cell_volume
    assert get_shot_noise_counts(
        counts,
        box_len,
        random_counts=ps.field_2_R,
        alpha=ps.field_2_alpha,
    ) == pytest.approx(data_only)
    assert np.allclose(ps.shot_noise_2, data_only * correction)


def test_fkp_inputs_are_required():
    with pytest.raises(ValueError, match="random counts"):
        fkp_count_field(np.ones(4), np.zeros(4))
    ps = PowerSpectrum(
        np.ones((3, 3, 3)),
        np.array([9.0, 9.0, 9.0]),
        grid_scheme="nnb",
        compensate=[False, False],
        include_beam=[False, False],
        include_sky_sampling=[False, False],
    )
    assert ps.shot_noise_2 is None
    ps.field_2_has_random = True
    with pytest.raises(ValueError, match="field_2_D"):
        ps.field_2
    ps.field_1_has_random = True
    with pytest.raises(ValueError, match="field_1"):
        ps.field_1 = np.ones((3, 3, 3))
    with pytest.raises(ValueError, match="tracer"):
        ps._shot_noise_tracer(0)


def test_grid_gal(test_gal_fits, test_W):
    ps = PowerSpectrum(
        gal_file=test_gal_fits,
        survey="meerklass_2021",
        band="L",
    )
    ps.W_HI = (test_W * ps.nu[None, None, :]) > 0
    ps.data = ps.W_HI
    ps.w_HI = ps.W_HI
    ps = PowerSpectrum(
        gal_file=test_gal_fits,
        data=ps.data,
        map_has_sampling=ps.W_HI,
        weights_map_pixel=ps.w_HI,
        init_box_from_map_data=True,
        include_sky_sampling=[True, True],
        survey="meerklass_2021",
        band="L",
        tracer_bias_2=1.0,  # just for invoking some tests
    )
    ps.read_gal_cat()
    ps.grid_gal_to_field()
    ps.apply_taper_to_field(2)


def test_grid_gal_to_field_zero_galaxy(test_W):
    ps = PowerSpectrum(
        data=(test_W > 0).astype(float),
        map_has_sampling=(test_W > 0),
        weights_map_pixel=(test_W > 0).astype(float),
        init_box_from_map_data=True,
        include_sky_sampling=[True, True],
        survey="meerklass_2021",
        band="L",
        tracer_bias_2=1.0,
    )
    ps.get_enclosing_box()
    ps._counts_in_box = np.ones(tuple(ps.box_ndim.tolist()), dtype=ps.real_dtype)
    ps._ra_gal = np.array([])
    ps._dec_gal = np.array([])
    ps._z_gal = np.array([])
    galmap_rg, galweights_rg, galcounts_rg = ps.grid_gal_to_field()
    assert np.allclose(galmap_rg, 0.0)
    assert np.allclose(galweights_rg, 0.0)
    assert np.allclose(galcounts_rg, 0.0)
    assert np.allclose(ps.field_2, 0.0)


def test_poisson_gal_gen():
    raminMK, ramaxMK = 334, 357
    decminMK, decmaxMK = -35, -26.5
    ra_range = (raminMK, ramaxMK)
    dec_range = (decminMK, decmaxMK)
    ps = PowerSpectrum(
        ra_range=ra_range,
        dec_range=dec_range,
        omega_hi=5.4e-4,
        mean_amp_1="average_hi_temp",
        tracer_bias_1=1.5,
        tracer_bias_2=1.9,
        survey="meerklass_2021",
        band="L",
        # seed=42,
        kmax=10.0,
        num_particle_per_pixel=2,
        box_buffkick=[5, 5, 5],
        seed=1,
    )
    ps._ra_gal = np.ones(40000)
    ps._dec_gal = np.ones(40000)
    ps._z_gal = np.ones(40000)
    radecfreq = ps.gen_random_poisson_galaxy(seed=1)
    ps.compensate = False
    ps.grid_gal_to_field(radecfreq)
    volume = (
        (ps.W_HI[:, :, 0].sum() * ps.pixel_area * (np.pi / 180) ** 2)
        / 3
        * (
            ps.astropy_cosmo_true.comoving_distance(ps.z_ch.max()) ** 3
            - ps.astropy_cosmo_true.comoving_distance(ps.z_ch.min()) ** 3
        ).value
    )
    k1dedges = np.geomspace(0.05, 1, 21)
    ps.k1dbins = k1dedges
    psn = volume / ps.ra_gal.size
    psn1d, _, _ = ps.get_1d_power(
        "auto_power_3d_2",
    )
    plateau = psn1d[-5:].mean()
    assert np.abs(plateau - psn) / psn < 2.5e-1


def test_poisson_gal_gen_chi2_radial():
    """Default radial sampling is p(chi) ∝ chi**2 (constant comoving density)."""
    from meer21cm.util import freq_to_redshift

    zmin = 0.6
    zmax = 0.8
    nu = np.linspace(redshift_to_freq(zmax), redshift_to_freq(zmin), 100)
    raminMK, ramaxMK = 320, 380
    decminMK, decmaxMK = -35, -26.5
    ra_range = (raminMK, ramaxMK)
    dec_range = (decminMK, decmaxMK)
    ps = PowerSpectrum(
        hp_nside=128,
        ra_range=ra_range,
        dec_range=dec_range,
        omega_hi=5.4e-4,
        mean_amp_1="average_hi_temp",
        tracer_bias_1=1.5,
        tracer_bias_2=1.9,
        nu=nu,
        # seed=42,
        kmax=10.0,
        num_particle_per_pixel=2,
        box_buffkick=[5, 5, 5],
        seed=2,
    )
    ps._ra_gal = np.ones(400000)
    ps._dec_gal = np.ones(400000)
    ps._z_gal = np.ones(400000)
    dndz_func = lambda z: 0.01 * np.exp(-((z - 0.7) ** 2) / 0.01)
    radecfreq = ps.gen_random_poisson_galaxy(dndz=dndz_func(ps.z_ch), seed=2)
    ps.compensate = False
    gal_count, _, _ = ps.grid_gal_to_field(radecfreq)
    volume = (
        (ps.W_HI[..., 0].sum() * ps.pixel_area * (np.pi / 180) ** 2)
        / 3
        * (
            ps.astropy_cosmo_true.comoving_distance(ps.z_ch.max()) ** 3
            - ps.astropy_cosmo_true.comoving_distance(ps.z_ch.min()) ** 3
        ).value
    )
    k1dedges = np.linspace(0.01, 0.3, 21)
    ps.k1dbins = k1dedges
    ps.field_2 = gal_count
    ps.weights_field_2 = dndz_func(ps._box_voxel_redshift)
    ps.weights_2 = (ps.counts_in_box > 0).astype(float)
    ps.apply_taper_to_field(2, axis=(0, 1, 2))
    psn = volume / ps.ra_gal.size
    psn1d, _, _ = ps.get_1d_power(
        "auto_power_3d_2",
    )
    # remove first 2 bins to avoid windowing and large var
    psn1d = psn1d[2:]
    assert psn1d.mean() == pytest.approx(psn, rel=2e-1)
    # roughly a plateau
    assert psn1d.std() < (psn * 0.2)

    # check z dist
    z_rand = freq_to_redshift(radecfreq[-1])
    z_interp = np.linspace(z_rand.min(), z_rand.max(), 100)
    diff_V = ps.cosmo.differential_comoving_volume(z_interp).value
    diff_V_interp = interp1d(z_interp, diff_V)
    diff_V_rand = diff_V_interp(z_rand)
    counts, bins = np.histogram(z_rand, weights=1 / diff_V_rand, bins=50)
    counts /= counts.sum()
    bins = (bins[:-1] + bins[1:]) / 2
    counts_func = dndz_func(bins)
    counts_func /= counts_func.sum()
    assert np.abs(counts / counts_func - 1).max() < 5e-2

    chi_min = ps.astropy_cosmo_fiducial.comoving_distance(ps.z_ch.min()).to("Mpc").value
    chi_max = ps.astropy_cosmo_fiducial.comoving_distance(ps.z_ch.max()).to("Mpc").value

    def _chi3_hist_scatter(freq):
        z = freq_to_redshift(freq)
        chi = ps.astropy_cosmo_fiducial.comoving_distance(z).to("Mpc").value
        # CDF of p(χ)∝χ² is uniform in χ³.
        u = (chi**3 - chi_min**3) / (chi_max**3 - chi_min**3)
        hist, _ = np.histogram(u, bins=10, range=(0, 1))
        return hist.std() / hist.mean()

    _, _, freq = ps.gen_random_poisson_galaxy(num_g_rand=80000, seed=3)
    assert _chi3_hist_scatter(freq) < 0.08

    # Per-volume ones_like matches the same χ² measure (mock default convention).
    _, _, freq2 = ps.gen_random_poisson_galaxy(
        num_g_rand=80000, seed=4, dndz=np.ones_like
    )
    assert _chi3_hist_scatter(freq2) < 0.08


def test_flat_sky():
    mock = MockSimulation(
        survey="meerklass_2021",
        band="L",
        highres_sim=None,
        num_discrete_source=1000000,
        tracer_bias_2=1.0,
        kmax=10.0,
        flat_sky=True,
        mean_amp_1="average_hi_temp",
        seed=1,
    )
    mock.data = mock.propagate_mock_field_to_data(mock.mock_tracer_field_1)
    mock.grid_data_to_field()
    mock.weights_field_1 = None
    mock.weights_grid_1 = None
    mock.include_sky_sampling = [False, False]
    mock.compensate = False
    ratio = mock.auto_power_3d_1 / mock.auto_power_tracer_1_model
    assert np.abs(ratio.mean() - 1) < 2e-1
    mock.propagate_mock_tracer_to_gal_cat()
    mock.grid_gal_to_field()
    mock.weights_field_2 = None
    mock.weights_grid_2 = None
    mock.compensate = False
    shot_noise = np.prod(mock.box_len) / mock.field_2.sum()
    ratio = (mock.auto_power_3d_2 - shot_noise) / mock.auto_power_tracer_2_model
    assert np.abs(ratio.mean() - 1) < 2e-1


def test_mock_tracer_grid():
    """
    Generate a mock galaxy caralogue,
    grid it onto regular grids, and test input/output matching.
    """
    raminGAMA, ramaxGAMA = 339, 351
    decminGAMA, decmaxGAMA = -35, -30
    ra_range = (raminGAMA, ramaxGAMA)
    dec_range = (decminGAMA, decmaxGAMA)
    k1dedges = np.geomspace(0.05, 1.5, 20)
    pmap_1d = []
    pmod_1d = []
    # run 10 realizations
    for i in range(10):
        mock = MockSimulation(
            survey="meerklass_2021",
            band="L",
            ra_range=ra_range,
            dec_range=dec_range,
            kaiser_rsd=True,
            discrete_base_field=2,
            k1dbins=k1dedges,
            target_relative_to_num_g=2.0,
            seed=i,
        )
        mock.data = np.ones(mock.W_HI.shape)
        mock.w_HI = np.ones(mock.W_HI.shape)
        mock.counts = np.ones(mock.W_HI.shape)
        mock.downres_factor_radial = 1 / 2.0
        mock.downres_factor_transverse = 1 / 2.0
        mock.get_enclosing_box()
        mock.tracer_bias_2 = 1.9
        mock.num_discrete_source = 2700
        # galaxy catalogue
        mock.propagate_mock_tracer_to_gal_cat()
        mock.downres_factor_radial = 1.5
        mock.downres_factor_transverse = 1.5
        mock.compensate = False
        gal_map_rg, gal_weights_rg, pixel_counts_gal_rg = mock.grid_gal_to_field()
        _, _, pixel_counts_hi_rg = mock.grid_data_to_field()
        mock.get_n_bar_correction()
        taper = mock.taper_func(mock.box_ndim[-1])
        mock.weights_2 = (pixel_counts_hi_rg > 0) * taper[None, None, :]
        shot_noise_g = (
            np.prod(mock.box_len) * (pixel_counts_hi_rg > 0).mean() / mock.ra_gal.size
        )
        mock.sampling_resol = None
        mock.has_resol = False
        pmod_1d_gg, keff, _ = mock.get_1d_power("auto_power_tracer_2_model")
        pdata_1d_gg, keff, nmodes = mock.get_1d_power(
            "auto_power_3d_2",
        )
        pdata_1d_gg -= shot_noise_g
        pmap_1d += [
            pdata_1d_gg,
        ]
        pmod_1d += [
            pmod_1d_gg,
        ]
    pmap_1d = np.array(pmap_1d)
    pmod_1d = np.array(pmod_1d)
    avg_deviation = ((pmap_1d.mean(0) - pmod_1d.mean(0)) / pmap_1d.std(0)).mean()
    # 3 sigma
    assert np.abs(avg_deviation) < 3


def test_galaxy_selection_count_and_legacy_ratio():
    mock = MockSimulation(
        survey="meerklass_2021",
        band="L",
        ra_range=(334, 357),
        dec_range=(-35, -26.5),
        tracer_bias_1=1.5,
        num_discrete_source=100,
    )
    mock.get_enclosing_box()
    assert mock.galaxy_selection is None
    legacy = mock.num_discrete_source * np.prod(mock.box_len) / mock.survey_volume
    assert mock.tot_num_source_in_box == pytest.approx(legacy)
    density = np.full(tuple(int(n) for n in mock.box_ndim), 1.0e-4)
    weight, total = selection_sampling_weight(density, mock.box_resol)
    assert np.allclose(weight, 1.0)
    assert total == pytest.approx(1.0e-4 * np.prod(mock.box_resol) * density.size)
    zeros, zero_total = selection_sampling_weight(
        np.zeros_like(density), mock.box_resol
    )
    assert zero_total == 0.0
    assert np.all(zeros == 0.0)
    step = np.array([1.0, 3.0])
    step_weight, step_total = selection_sampling_weight(step, np.array([2.0]))
    assert step_total == pytest.approx(8.0)
    assert np.allclose(step_weight, step / step.mean())
    mock.galaxy_selection = density
    assert mock.tot_num_source_in_box == pytest.approx(total)
    weight, enclosed = mock._selection_weight()
    assert np.allclose(weight, 1.0)
    assert enclosed == pytest.approx(total)
    with pytest.raises(ValueError, match="galaxy_selection shape"):
        mock._selection_weight(np.ones(2))
    mock.galaxy_selection = None
    assert mock.tot_num_source_in_box == pytest.approx(legacy)


def _small_lightcone():
    redshift = np.linspace(0.6, 0.8, 6)
    return MockSimulation(
        nu=redshift_to_freq(redshift[::-1]),
        hp_nside=16,
        ra_range=(0.0, 20.0),
        dec_range=(-10.0, 10.0),
        downres_factor_transverse=4,
        downres_factor_radial=2,
        num_discrete_source=20,
        grid_scheme="nnb",
    )


def _write_selection_catalogues(tmp_path):
    from astropy.table import Table

    random_path = tmp_path / "random.fits"
    data_path = tmp_path / "data.fits"
    Table(
        {
            "RA": [9.83, 9.83, 9.83, 100.0],
            "DEC": [0.0, 0.0, 0.0, 0.0],
            "Z": [0.62, 0.69, 0.76, 0.65],
            "WEIGHT": [1.0, 2.0, 1.0, 5.0],
            "WEIGHT_COMP": [2.0, 2.0, 1.0, 1.0],
            "WEIGHT_SYS": [1.0, 2.0, 1.0, 1.0],
            "WEIGHT_ZFAIL": [0.5, 0.5, 1.0, 1.0],
        }
    ).write(random_path)
    Table(
        {
            "RA": [9.83, 100.0, 9.83],
            "DEC": [0.0, 0.0, 0.0],
            "Z": [0.65, 0.65, 0.2],
            "WEIGHT": [8.0, 9.0, 7.0],
        }
    ).write(data_path)
    return random_path, data_path


def _selection_integral(angular, n_w, z_edges, nside, cosmo):
    import healpy as hp

    n_pix = int(np.sum(np.asarray(angular) > 0))
    omega = n_pix * float(hp.nside2pixarea(int(nside)))
    chi = np.asarray(cosmo.comoving_distance(np.asarray(z_edges)).value, dtype=float)
    volume = omega / 3.0 * (chi[1:] ** 3 - chi[:-1] ** 3)
    dvol = volume / max(n_pix, 1)
    return float(np.sum(np.asarray(n_w) * dvol * np.sum(angular)))


def test_selection_weight_changes_the_sampled_counts():
    mock = _small_lightcone()
    mock.get_enclosing_box()
    shape = tuple(int(n) for n in mock.box_ndim)
    density = np.full(shape, 1.0e-6)
    density[..., -1] = 3.0e-6
    mock.galaxy_selection = density
    mock.seed = 0
    mock.get_mock_tracer_position_in_box(2, density_field=np.zeros(shape, dtype=float))
    positions = mock._mock_tracer_position_in_box
    edges = np.linspace(0.0, mock.box_len[2], shape[2] + 1)
    counts, _bins = np.histogram(positions[:, 2], bins=edges)
    expected = np.array([1.0, 1.0, 3.0])
    expected = expected / expected.sum() * counts.sum()
    chi2 = float(np.sum((counts - expected) ** 2 / expected))
    assert counts.sum() == pytest.approx(
        float(np.sum(density) * np.prod(mock.box_resol)), rel=0.15
    )
    assert chi2 < 20.0


def test_constant_random_weight_rescales_alpha_and_keeps_F():
    mock = _small_lightcone()
    mock.get_enclosing_box()
    data_ra, data_dec, data_z = _interior_radecz(mock, [[0, 0, 0]])
    rand_ra, rand_dec, rand_z = _interior_radecz(mock, [[0, 0, 1]] * 4)
    freq = redshift_to_freq(data_z)
    field_unit, _, _ = mock.grid_gal_to_field(
        radecfreq=(data_ra, data_dec, freq),
        weights=np.ones(1),
        construct_fkp=True,
        random_radecz=(rand_ra, rand_dec, rand_z),
        random_weights=np.ones(4),
    )
    alpha_unit = float(mock.field_2_alpha)
    field_weighted, _, _ = mock.grid_gal_to_field(
        radecfreq=(data_ra, data_dec, freq),
        weights=np.ones(1),
        construct_fkp=True,
        random_radecz=(rand_ra, rand_dec, rand_z),
        random_weights=np.full(4, 2.0),
    )
    assert mock.field_2_alpha == pytest.approx(alpha_unit / 2.0)
    assert np.allclose(mock.weights_rand, 2.0)
    assert np.allclose(field_weighted, field_unit)
    assert field_weighted.sum() == pytest.approx(0.0, abs=1e-8)
    positions = mock._sky_positions_in_box(
        rand_ra, rand_dec, redshift_to_freq(rand_z), mock.flat_sky
    )
    weights = np.full(4, 2.0)
    painted = mock._cached_random_paint(positions, weights)
    assert mock._cached_random_paint(positions, weights) is painted
    assert painted.sum() == pytest.approx(8.0)


def test_selection_normalises_to_the_data_and_paints_cell_centres(tmp_path):
    import healpy as hp
    from astropy.cosmology import Planck18

    from meer21cm.io import selection_from_random_files

    random_path, data_path = _write_selection_catalogues(tmp_path)
    z_edges = np.array([0.6, 0.67, 0.74, 0.8])
    ra_range = (0.0, 20.0)
    dec_range = (-10.0, 10.0)
    columns = ("WEIGHT_COMP", "WEIGHT_SYS", "WEIGHT_ZFAIL")
    result = selection_from_random_files(
        [random_path],
        ra_range=ra_range,
        dec_range=dec_range,
        z_edges=z_edges,
        cosmo=Planck18,
        data_path=data_path,
        nside=8,
        random_density_deg2=1.0,
        source_weight_columns=columns,
    )
    pix = hp.ang2pix(8, 9.83, 0.0, lonlat=True)
    omega_deg = float(hp.nside2pixarea(8, degrees=True))
    assert result["angular"][pix] == pytest.approx(3.0 / omega_deg)
    assert result["angular"].sum() == pytest.approx(3.0 / omega_deg)
    assert result["data_weight"] == pytest.approx(8.0)
    assert result["random_weight"] == pytest.approx(4.0)
    assert result["scale"] == pytest.approx(2.0)
    assert _selection_integral(
        result["angular"], result["n_w"], z_edges, 8, Planck18
    ) == pytest.approx(8.0)
    unnormalised = selection_from_random_files(
        [random_path],
        ra_range=ra_range,
        dec_range=dec_range,
        z_edges=z_edges,
        cosmo=Planck18,
        nside=8,
        random_density_deg2=1.0,
        source_weight_columns=columns,
    )
    assert unnormalised["data_weight"] == pytest.approx(unnormalised["random_weight"])
    assert _selection_integral(
        unnormalised["angular"], unnormalised["n_w"], z_edges, 8, Planck18
    ) == pytest.approx(4.0)
    bare = selection_from_random_files(
        [random_path],
        ra_range=ra_range,
        dec_range=dec_range,
        z_edges=z_edges,
        cosmo=Planck18,
        nside=8,
        random_density_deg2=1.0,
        source_weight_columns=("NOT_A_COLUMN",),
    )
    assert bare["angular"][pix] == pytest.approx(4.0 / omega_deg)

    mock = _small_lightcone()
    assert mock.random_selection is None
    assert mock.sky_selection_density is None
    uniform = mock.selection_density_from_sky(lambda z: np.full(np.shape(z), 1.0e-4))
    mock._sky_selection_density = np.zeros(3)
    stored = mock.read_random_selection(
        [random_path],
        data_path=data_path,
        nside=8,
        z_edges=z_edges,
        random_density_deg2=1.0,
        source_weight_columns=columns,
    )
    assert mock._sky_selection_density is None
    assert mock.random_selection is stored
    assert stored["data_weight"] == pytest.approx(8.0)
    assert "_sky_selection_density" in mock.selection_dep_attr
    density = np.asarray(mock.sky_selection_density)
    assert mock.sky_selection_density is density
    xx, yy, zz = np.meshgrid(mock.x_vec[0], mock.x_vec[1], mock.x_vec[2], indexing="ij")
    centres = np.stack((xx.ravel(), yy.ravel(), zz.ravel()), axis=1)
    ra, dec, redshift, _distance = mock.ra_dec_z_for_coord_in_box(centres)
    expected = mock._selection_n_of_z()(redshift) * mock._selection_angular_at()(
        ra, dec
    )
    assert np.allclose(density.ravel(), expected)
    assert np.all(expected > 0)
    sky_total = float(np.sum(density) * np.prod(mock.box_resol))
    assert mock.galaxy_selection is None
    assert mock.tot_num_source_in_box == pytest.approx(sky_total)
    weight, enclosed = mock._selection_weight()
    assert enclosed == pytest.approx(sky_total)
    assert np.allclose(weight * np.mean(density), density)
    window = np.asarray(angle_in_range(ra, ra_range[0], ra_range[1]), dtype=float)
    window *= (dec > dec_range[0]) & (dec < dec_range[1])
    assert np.allclose(uniform.ravel(), 1.0e-4 * window)
    default_shells = mock.read_random_selection(
        [random_path],
        data_path=data_path,
        nside=8,
        random_density_deg2=1.0,
        source_weight_columns=columns,
    )
    assert default_shells["z_edges"].size == 21
    assert _selection_integral(
        default_shells["angular"],
        default_shells["n_w"],
        default_shells["z_edges"],
        8,
        mock.astropy_cosmo_true,
    ) == pytest.approx(8.0)
    mock._random_selection = None
    mock._selection_z_edges = None
    rebuilt = mock.random_selection
    assert rebuilt["z_edges"].size == 21
    assert _selection_integral(
        rebuilt["angular"],
        rebuilt["n_w"],
        rebuilt["z_edges"],
        rebuilt["nside"],
        mock.astropy_cosmo_true,
    ) == pytest.approx(rebuilt["data_weight"])
