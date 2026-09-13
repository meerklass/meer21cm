"""
Tests for the jackknife covariance weighting and box consistency.

These cover the argument handling and the box-geometry guard added so that the
jackknife realisations use the same weighting and the same Cartesian box as the
data measurement. They are deliberately cheap: no mock realisation is generated
here, so they do not depend on the gridding or on a survey-scale map.
"""

import numpy as np
import pytest

from meer21cm import PowerSpectrum
from meer21cm.util import f_21, freq_to_redshift
from meer21cm.jackknife import (
    JackknifeCovariance,
    _apply_jackknife_field_settings,
    _check_box_geometry,
    _check_k1dweights,
    _digitize_sky_to_patch,
    _get_dndz_box,
    _grid_jackknifed_map,
    _map_realisations,
    _materialise_none_grid_weights,
    _normalise_weights_argument,
    box_geometry,
    box_voxel_patch_labels,
    resolve_tracer_weights,
    run_jackknife_auto,
    run_jackknife_boxmask,
    run_jackknife_cross,
)
from meer21cm.power import get_renormed_field


def _weights_array(value=1.0, size=3):
    return np.full(size, value)


# --------------------------------------------------------------------------- #
# the weights argument
# --------------------------------------------------------------------------- #
def test_normalise_weights_none():
    assert _normalise_weights_argument(None) is None


def test_normalise_weights_single_array_is_grid_weights_of_tracer_1():
    w = _weights_array()
    out = _normalise_weights_argument(w)
    assert out[0] == (None, w)
    assert out[1] == (None, None)


def test_normalise_weights_single_array_for_auto():
    w = _weights_array()
    out = _normalise_weights_argument(w, type_default="auto")
    assert out == ((None, w), (None, None))


def test_normalise_weights_two_arrays():
    w1, w2 = _weights_array(1.0), _weights_array(2.0)
    out = _normalise_weights_argument((w1, w2))
    assert out[0] == (None, w1)
    assert out[1] == (None, w2)


def test_normalise_weights_field_grid_pairs():
    f1, g1, f2, g2 = (_weights_array(v) for v in (1.0, 2.0, 3.0, 4.0))
    out = _normalise_weights_argument(((f1, g1), (f2, g2)))
    assert out[0] == (f1, g1)
    assert out[1] == (f2, g2)


def test_normalise_weights_mixed_entry():
    f2, g2 = _weights_array(3.0), _weights_array(4.0)
    g1 = _weights_array(1.0)
    out = _normalise_weights_argument((g1, (f2, g2)))
    assert out[0] == (None, g1)
    assert out[1] == (f2, g2)


def test_normalise_weights_none_entries():
    out = _normalise_weights_argument((None, None))
    assert out == ((None, None), (None, None))


def test_normalise_weights_rejects_wrong_length():
    with pytest.raises(ValueError):
        _normalise_weights_argument((_weights_array(),) * 3)
    with pytest.raises(ValueError):
        _normalise_weights_argument(((_weights_array(),),))  # pair of length 1


# --------------------------------------------------------------------------- #
# box geometry
# --------------------------------------------------------------------------- #
def test_box_geometry_records_the_box():
    ps = PowerSpectrum(survey="meerklass_2021", band="L")
    ps.data = np.ones(ps.W_HI.shape)
    ps.w_HI = np.ones(ps.W_HI.shape)
    ps.get_enclosing_box()
    geo = box_geometry(ps)
    assert set(geo) == {"box_len", "box_ndim", "box_origin", "box_resol"}
    assert np.array_equal(geo["box_ndim"], ps.box_ndim)


def test_check_box_geometry_accepts_the_same_box_and_rejects_a_different_one():
    ps = PowerSpectrum(survey="meerklass_2021", band="L")
    ps.data = np.ones(ps.W_HI.shape)
    ps.w_HI = np.ones(ps.W_HI.shape)
    ps.get_enclosing_box()
    geo = box_geometry(ps)

    _check_box_geometry(ps, geo)  # must not raise

    for attr in ("box_ndim", "box_len", "box_origin", "box_resol"):
        bad = dict(geo)
        bad[attr] = np.asarray(geo[attr]) + 1
        with pytest.raises(ValueError, match="identical box"):
            _check_box_geometry(ps, bad)


def test_box_voxel_patch_labels_have_box_shape_and_valid_indices():
    ps = PowerSpectrum(survey="meerklass_2021", band="L")
    ps.data = np.ones(ps.W_HI.shape)
    ps.w_HI = np.ones(ps.W_HI.shape)
    ps.ra_range = (334.0, 357.0)
    ps.dec_range = (-35.0, -26.5)
    ps.get_enclosing_box()
    nu_range = (
        float(ps.nu.min() - ps.freq_resol / 2),
        float(ps.nu.max() + ps.freq_resol / 2),
    )
    labels = box_voxel_patch_labels(ps, 2, 2, 2, ps.ra_range, ps.dec_range, nu_range)
    assert labels.shape == tuple(int(n) for n in ps.box_ndim)
    assert labels.min() >= -1
    assert labels.max() < 8


def test_jackknife_covariance_records_the_box_on_construction():
    ps = PowerSpectrum(survey="meerklass_2021", band="L")
    ps.data = np.ones(ps.W_HI.shape)
    ps.w_HI = np.ones(ps.W_HI.shape)
    ps.ra_range = (334.0, 357.0)
    ps.dec_range = (-35.0, -26.5)
    ps.get_enclosing_box()
    jc = JackknifeCovariance(ps, 2, 2, 2, pool="serial")
    assert set(jc.box_geometry) == set(box_geometry(ps))
    assert np.array_equal(jc.box_geometry["box_ndim"], ps.box_ndim)


# --------------------------------------------------------------------------- #
# the dN/dz fallback
# --------------------------------------------------------------------------- #
def test_get_dndz_box_is_none_without_a_mock():
    ps = PowerSpectrum(survey="meerklass_2021", band="L")
    assert _get_dndz_box(ps) is None


def test_validation_scheme_is_the_default():
    import inspect

    sig = inspect.signature(JackknifeCovariance.__init__)
    assert sig.parameters["weights_scheme"].default == "validation"
    assert sig.parameters["weights"].default is None
    assert sig.parameters["freeze_mean"].default is False
    assert sig.parameters["regrid"].default is False
    assert sig.parameters["mean_center_1"].default is None
    assert sig.parameters["mean_center_2"].default is None


# --------------------------------------------------------------------------- #
# mean-center overrides / freeze_mean
# --------------------------------------------------------------------------- #
def test_resolve_tracer_weights_restricts_to_kept_by_default():
    grid = np.array([1.0, 2.0, 3.0, 4.0])
    field = np.array([10.0, 20.0, 30.0, 40.0])
    weights_rg = np.array([1.0, 0.0, 1.0, 0.0])
    kept = weights_rg > 0
    weights = _normalise_weights_argument(((field, grid), None))
    field_w, grid_w = resolve_tracer_weights(
        tracer=1,
        weights_scheme="validation",
        weights=weights,
        jk=None,
        counts_keep_rg=None,
        weights_rg=weights_rg,
    )
    np.testing.assert_array_equal(field_w, field * kept)
    np.testing.assert_array_equal(grid_w, grid * kept)


def test_apply_jackknife_field_settings_overrides_after_gal_defaults():
    ps = PowerSpectrum(survey="meerklass_2021", band="L")
    ps.mean_center_2 = True
    ps.unitless_2 = True
    _apply_jackknife_field_settings(
        ps,
        2,
        {"mean_center_2": True, "unitless_2": True},
        mean_center_override=False,
        unitless_override=False,
        freeze_mean=False,
        field_mean=None,
    )
    assert ps.mean_center_2 is False
    assert ps.unitless_2 is False


def test_apply_jackknife_field_settings_freeze_mean_sets_field_mean():
    ps = PowerSpectrum(survey="meerklass_2021", band="L")
    _apply_jackknife_field_settings(
        ps,
        1,
        {"mean_center_1": False, "unitless_1": False},
        mean_center_override=True,
        unitless_override=True,
        freeze_mean=True,
        field_mean=3.5,
    )
    assert ps.mean_center_1 is True
    assert ps.unitless_1 is True
    assert ps.field_mean_1 == 3.5


def test_get_renormed_field_uses_supplied_field_mean():
    field = np.array([2.0, 4.0, 6.0])
    weights = np.ones(3)
    # weighted mean of the field is 4; freeze it to 2
    out = get_renormed_field(
        field, weights=weights, mean_center=True, unitless=True, field_mean=2.0
    )
    np.testing.assert_allclose(out, (field - 2.0) / 2.0)
    default = get_renormed_field(
        field, weights=weights, mean_center=True, unitless=True
    )
    np.testing.assert_allclose(default, (field - 4.0) / 4.0)


def test_w_hi_setter_clears_counts_in_box_cache():
    ps = PowerSpectrum(survey="meerklass_2021", band="L")
    ps.data = np.ones(ps.W_HI.shape)
    ps.w_HI = np.ones(ps.W_HI.shape)
    ps.get_enclosing_box()
    cached = ps.counts_in_box
    assert ps._counts_in_box is not None
    assert np.array_equal(cached, ps._counts_in_box)
    ps.w_HI = np.zeros(ps.W_HI.shape)
    assert ps._counts_in_box is None


def _boxed_ps(k1dbins=None):
    """A survey PowerSpectrum with a fixed enclosing box and 1D k-bins."""
    ps = PowerSpectrum(survey="meerklass_2021", band="L")
    ps.data = np.ones(ps.W_HI.shape)
    ps.w_HI = np.ones(ps.W_HI.shape)
    ps.ra_range = (334.0, 357.0)
    ps.dec_range = (-35.0, -26.5)
    ps.get_enclosing_box()
    ps.k1dbins = np.linspace(0.02, 0.15, 6) if k1dbins is None else k1dbins
    return ps


def _nu_range(ps):
    return (
        float(ps.nu.min() - ps.freq_resol / 2),
        float(ps.nu.max() + ps.freq_resol / 2),
    )


# --------------------------------------------------------------------------- #
# Tukey covariance
# --------------------------------------------------------------------------- #
def test_jackknife_covariance_matches_tukey_formula():
    power = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    cov, mean = JackknifeCovariance.jackknife_covariance(power)
    n = power.shape[0]
    expected_mean = power.mean(axis=0)
    delta = power - expected_mean
    expected = (n - 1) / n * delta.T @ delta
    np.testing.assert_allclose(mean, expected_mean)
    np.testing.assert_allclose(cov, expected)


def test_jackknife_covariance_vanishes_for_identical_realisations():
    power = np.ones((5, 4))
    cov, mean = JackknifeCovariance.jackknife_covariance(power)
    np.testing.assert_allclose(mean, 1.0)
    np.testing.assert_allclose(cov, 0.0)


def test_jackknife_covariance_needs_at_least_two_realisations():
    with pytest.raises(ValueError, match="at least 2"):
        JackknifeCovariance.jackknife_covariance(np.ones((1, 3)))


# --------------------------------------------------------------------------- #
# voxel labels and the default box-mask estimator
# --------------------------------------------------------------------------- #
def test_digitize_sky_to_patch_matches_galaxy_labels():
    ps = _boxed_ps()
    n_ra, n_dec, n_nu = 2, 2, 2
    nu_range = _nu_range(ps)
    rng = np.random.default_rng(1)
    ra = rng.uniform(ps.ra_range[0], ps.ra_range[1], 40)
    dec = rng.uniform(ps.dec_range[0], ps.dec_range[1], 40)
    z = rng.uniform(freq_to_redshift(ps.nu.max()), freq_to_redshift(ps.nu.min()), 40)
    ps._ra_gal = ra
    ps._dec_gal = dec
    ps._z_gal = z
    gal = ps.get_gal_patch_labels(
        n_ra, n_dec, n_nu, ps.ra_range, ps.dec_range, nu_range
    )
    sky = _digitize_sky_to_patch(
        ra, dec, ps.freq_gal, ps.ra_range, ps.dec_range, nu_range, n_ra, n_dec, n_nu
    )
    np.testing.assert_array_equal(sky, gal)


def test_box_voxel_labels_match_galaxies_at_voxel_centres():
    ps = _boxed_ps()
    n_ra, n_dec, n_nu = 2, 2, 2
    nu_range = _nu_range(ps)
    labels = box_voxel_patch_labels(
        ps, n_ra, n_dec, n_nu, ps.ra_range, ps.dec_range, nu_range
    )
    inside = np.argwhere(labels >= 0)
    assert len(inside) > 0
    pick = inside[:: max(len(inside) // 12, 1)][:12]
    x_vec, y_vec, z_vec = ps.x_vec
    pos = np.column_stack([x_vec[pick[:, 0]], y_vec[pick[:, 1]], z_vec[pick[:, 2]]])
    ra, dec, z, _ = ps.ra_dec_z_for_coord_in_box(pos)
    ps._ra_gal = ra
    ps._dec_gal = dec
    ps._z_gal = z
    gal = ps.get_gal_patch_labels(
        n_ra, n_dec, n_nu, ps.ra_range, ps.dec_range, nu_range
    )
    np.testing.assert_array_equal(gal, labels[tuple(pick.T)])


def test_boxmask_keep_all_matches_full_sample_power():
    ps = _boxed_ps()
    rng = np.random.default_rng(0)
    ps.field_1 = rng.normal(size=ps.box_ndim).astype(ps.real_dtype)
    ps.weights_field_1 = None
    ps.weights_grid_1 = np.array(ps.counts_in_box, copy=True)
    p_full, _, _ = ps.get_1d_power("auto_power_3d_1")

    jc = JackknifeCovariance(ps, 2, 2, 1, pool="serial", apply_taper=False)
    p_jk = run_jackknife_boxmask(
        jc.get_ps_instance_attr_dict(),
        np.ones(ps.box_ndim, dtype=bool),
        ps.field_1,
        None,
        ps.counts_in_box,
        ps.k1dweights,
        "validation",
        None,
        False,
        (2,),
        "binary",
        False,
        False,
        jc.box_geometry,
        None,
    )[0]
    np.testing.assert_allclose(p_jk, p_full, rtol=1e-10, atol=0.0)


def test_boxmask_zeroing_a_localized_spike_reduces_power():
    ps = _boxed_ps()
    jc = JackknifeCovariance(ps, 2, 2, 1, pool="serial", apply_taper=False)
    labels = jc.box_voxel_labels()
    surveyed = ps.counts_in_box > 0
    occupied = [j for j in range(jc.num_patches) if np.any((labels == j) & surveyed)]
    assert occupied
    patch = max(occupied, key=lambda j: np.sum((labels == j) & surveyed))
    field = np.zeros(ps.box_ndim, dtype=ps.real_dtype)
    field[(labels == patch) & surveyed] = 1.0
    ps.field_1 = field
    keep_cut = labels != patch
    args = (
        jc.get_ps_instance_attr_dict(),
        None,
        ps.counts_in_box,
        ps.k1dweights,
        "validation",
        None,
        False,
        (2,),
        "binary",
        False,
        False,
        jc.box_geometry,
        None,
    )
    p_full = run_jackknife_boxmask(
        args[0], np.ones(ps.box_ndim, dtype=bool), field, *args[1:]
    )[0]
    p_cut = run_jackknife_boxmask(args[0], keep_cut, field, *args[1:])[0]
    assert np.isfinite(p_full).all() and np.isfinite(p_cut).all()
    assert np.nansum(np.abs(p_full)) > 0.0
    assert np.nansum(np.abs(p_cut)) < 0.05 * np.nansum(np.abs(p_full))


def test_run_auto_boxmask_gives_positive_diagonal_covariance():
    ps = _boxed_ps()
    rng = np.random.default_rng(2)
    ps.field_1 = rng.normal(size=ps.box_ndim).astype(ps.real_dtype)
    jc = JackknifeCovariance(ps, 2, 2, 1, pool="serial", apply_taper=False)
    results = jc.run(type="auto")
    assert jc.regrid is False
    assert len(results) == len(jc.patch_indices_used)
    assert len(results) >= 2
    cov, mean = jc.get_covariance(results)
    n_k = len(ps.k1dbins) - 1
    assert cov.shape == (n_k, n_k)
    assert mean.shape == (n_k,)
    assert np.isfinite(cov).all()
    assert np.all(np.diag(cov) >= 0.0)


def test_run_rejects_empty_patch_list():
    ps = _boxed_ps()
    ps.field_1 = np.ones(ps.box_ndim, dtype=ps.real_dtype)
    jc = JackknifeCovariance(ps, 2, 2, 1, pool="serial", apply_taper=False)
    with pytest.raises(ValueError, match="no jackknife patch"):
        jc.run(type="auto", patch_indices=[])


def _attach_galaxies_at_surveyed_voxels(ps, n=12):
    """Place galaxies on surveyed voxel centres so sky and box labels match."""
    nu_range = _nu_range(ps)
    labels = box_voxel_patch_labels(ps, 2, 2, 1, ps.ra_range, ps.dec_range, nu_range)
    pick = np.argwhere((ps.counts_in_box > 0) & (labels >= 0))[:n]
    assert len(pick) > 0
    x_vec, y_vec, z_vec = ps.x_vec
    pos = np.column_stack([x_vec[pick[:, 0]], y_vec[pick[:, 1]], z_vec[pick[:, 2]]])
    ra, dec, z, _ = ps.ra_dec_z_for_coord_in_box(pos)
    ps._ra_gal = ra
    ps._dec_gal = dec
    ps._z_gal = z
    return labels, pick


# --------------------------------------------------------------------------- #
# constructor ranges and documented errors
# --------------------------------------------------------------------------- #
def test_z_range_sets_frequency_bins_and_inverts():
    ps = _boxed_ps()
    z_lo, z_hi = 0.05, 0.15
    jc = JackknifeCovariance(
        ps, 2, 2, 1, z_range=(z_lo, z_hi), pool="serial", apply_taper=False
    )
    np.testing.assert_allclose(jc.nu_range, (f_21 / (1.0 + z_hi), f_21 / (1.0 + z_lo)))
    np.testing.assert_allclose(jc.z_range, (z_lo, z_hi), rtol=1e-12)


def test_nu_range_and_z_range_are_mutually_exclusive():
    ps = _boxed_ps()
    with pytest.raises(ValueError, match="mutually exclusive"):
        JackknifeCovariance(
            ps,
            2,
            2,
            1,
            nu_range=_nu_range(ps),
            z_range=(0.05, 0.15),
            pool="serial",
        )


def test_inverted_z_range_is_rejected():
    ps = _boxed_ps()
    with pytest.raises(AssertionError):
        JackknifeCovariance(ps, 2, 2, 1, z_range=(0.2, 0.1), pool="serial")


def test_invalid_weights_scheme_and_gal_grid_weights_are_rejected():
    ps = _boxed_ps()
    with pytest.raises(ValueError, match="Invalid weights_scheme"):
        JackknifeCovariance(ps, 2, 2, 1, weights_scheme="fkp", pool="serial")
    with pytest.raises(ValueError, match="Invalid gal_grid_weights"):
        JackknifeCovariance(ps, 2, 2, 1, gal_grid_weights="fkp", pool="serial")


def test_run_rejects_invalid_type_and_pool():
    ps = _boxed_ps()
    ps.field_1 = np.ones(ps.box_ndim, dtype=ps.real_dtype)
    jc = JackknifeCovariance(ps, 2, 2, 1, pool="serial", apply_taper=False)
    with pytest.raises(ValueError, match="Invalid type"):
        jc.run(type="wedges", patch_indices=[0, 1])
    jc.pool = "threads"
    with pytest.raises(ValueError, match="Invalid pool"):
        jc.run(type="auto", patch_indices=[0, 1])


def test_mismatched_k1dweights_are_rejected():
    ps = _boxed_ps()
    with pytest.raises(ValueError, match="k1dweights"):
        _check_k1dweights(ps, np.ones((2, 2, 2)))


# --------------------------------------------------------------------------- #
# weight schemes
# --------------------------------------------------------------------------- #
def test_validation_hi_uses_jackknifed_counts():
    counts = np.array([1.0, 0.0, 3.0])
    weights_rg = np.array([1.0, 0.0, 1.0])
    field_w, grid_w = resolve_tracer_weights(
        1, "validation", None, None, counts, weights_rg
    )
    assert field_w is None
    np.testing.assert_array_equal(grid_w, counts)


def test_validation_galaxy_uses_dndz_times_surveyed_counts():
    counts = np.array([1.0, 2.0, 0.0, 4.0])
    weights_rg = np.ones(4)
    dndz = np.array([0.5, 0.0, 0.2, 0.3])
    field_w, grid_w = resolve_tracer_weights(
        2, "validation", None, None, counts, weights_rg, gal_dndz=dndz
    )
    np.testing.assert_array_equal(field_w, dndz)
    np.testing.assert_array_equal(grid_w, (dndz > 0) * counts)


def test_validation_galaxy_without_dndz_falls_back_to_occupancy():
    ps = PowerSpectrum(survey="meerklass_2021", band="L")
    counts = np.array([1.0, 0.0, 3.0])
    weights_rg = np.ones(3)
    with pytest.warns(UserWarning, match="dN/dz"):
        field_w, grid_w = resolve_tracer_weights(
            2, "validation", None, ps, counts, weights_rg
        )
    np.testing.assert_array_equal(field_w, counts > 0)
    np.testing.assert_array_equal(grid_w, np.ones_like(weights_rg))


def test_counts_scheme_is_inverse_noise_for_hi_and_occupancy_for_galaxies():
    counts = np.array([2.0, 0.0, 5.0])
    weights_rg = np.ones(3)
    hi_f, hi_g = resolve_tracer_weights(1, "counts", None, None, counts, weights_rg)
    gal_f, gal_g = resolve_tracer_weights(2, "counts", None, None, counts, weights_rg)
    assert hi_f is None
    np.testing.assert_array_equal(hi_g, counts)
    np.testing.assert_array_equal(gal_f, counts > 0)
    np.testing.assert_array_equal(gal_g, np.ones_like(weights_rg))


def test_gridded_scheme_copies_the_jackknifed_window():
    weights_rg = np.array([1.0, 0.0, 2.0])
    field_w, grid_w = resolve_tracer_weights(1, "gridded", None, None, None, weights_rg)
    np.testing.assert_array_equal(field_w, weights_rg)
    np.testing.assert_array_equal(grid_w, weights_rg)


def test_gridded_galaxy_binary_window_is_restricted_to_kept_voxels():
    weights_rg = np.array([1.0, 0.0, 1.0, 1.0])
    window = np.array([4.0, 5.0, 6.0, 7.0])
    field_w, grid_w = resolve_tracer_weights(
        2,
        "gridded",
        None,
        None,
        None,
        weights_rg,
        gal_grid_weights="binary",
        gal_window_rg=window,
    )
    expected = window * (weights_rg > 0)
    np.testing.assert_array_equal(field_w, expected)
    np.testing.assert_array_equal(grid_w, expected)


def test_gridded_galaxy_falls_back_to_field_weights_then_occupancy():
    weights_rg = np.array([1.0, 0.0, 1.0])
    counts = np.array([2.0, 3.0, 0.0])
    ps = PowerSpectrum(survey="meerklass_2021", band="L")
    ps.weights_field_2 = np.array([9.0, 8.0, 7.0])
    field_w, _ = resolve_tracer_weights(
        2, "gridded", None, ps, counts, weights_rg, gal_grid_weights="binary"
    )
    np.testing.assert_array_equal(field_w, ps.weights_field_2 * (weights_rg > 0))
    ps.weights_field_2 = None
    field_w, _ = resolve_tracer_weights(
        2, "gridded", None, ps, counts, weights_rg, gal_grid_weights="binary"
    )
    np.testing.assert_array_equal(field_w, (counts > 0) * (weights_rg > 0))
    field_w, _ = resolve_tracer_weights(
        2, "gridded", None, ps, None, weights_rg, gal_grid_weights="binary"
    )
    np.testing.assert_array_equal(field_w, weights_rg > 0)


def test_gridded_galaxy_uses_supplied_galaxy_grid_weights():
    weights_rg = np.ones(3)
    gal_w = np.array([1.0, 2.0, 3.0])
    field_w, grid_w = resolve_tracer_weights(
        2,
        "gridded",
        None,
        None,
        None,
        weights_rg,
        gal_grid_weights="gridded",
        gal_weights_rg=gal_w,
    )
    np.testing.assert_array_equal(field_w, gal_w)
    np.testing.assert_array_equal(grid_w, gal_w)


def test_weight_scheme_errors():
    weights_rg = np.ones(2)
    with pytest.raises(ValueError, match="counts_keep_rg is required"):
        resolve_tracer_weights(1, "validation", None, None, None, weights_rg)
    with pytest.raises(ValueError, match="Invalid weights_scheme"):
        resolve_tracer_weights(1, "fkp", None, None, np.ones(2), weights_rg)
    with pytest.raises(ValueError, match="gridded galaxy weights"):
        resolve_tracer_weights(
            2,
            "gridded",
            None,
            None,
            None,
            weights_rg,
            gal_grid_weights="gridded",
        )
    with pytest.raises(ValueError, match="Invalid gal_grid_weights"):
        resolve_tracer_weights(
            2, "gridded", None, None, None, weights_rg, gal_grid_weights="fkp"
        )


def test_get_dndz_box_evaluates_the_callable_on_voxel_redshifts():
    ps = _boxed_ps()
    ps.discrete_source_dndz = lambda z: np.full_like(np.asarray(z), 2.5, dtype=float)
    dndz = _get_dndz_box(ps)
    assert dndz.shape == tuple(int(n) for n in ps.box_ndim)
    np.testing.assert_allclose(dndz, 2.5)


def test_none_grid_weights_become_uniform_before_tapering():
    ps = _boxed_ps()
    ps.weights_grid_1 = None
    _materialise_none_grid_weights(ps, 1)
    np.testing.assert_array_equal(ps.weights_grid_1, np.ones(ps.box_ndim))


def test_tracer_1_field_settings_restore_the_input_instance_flags():
    ps = PowerSpectrum(survey="meerklass_2021", band="L")
    ps.mean_center_1 = False
    ps.unitless_1 = False
    _apply_jackknife_field_settings(
        ps,
        1,
        {"mean_center_1": True, "unitless_1": False},
        mean_center_override=None,
        unitless_override=None,
        freeze_mean=False,
        field_mean=None,
    )
    assert ps.mean_center_1 is True
    assert ps.unitless_1 is False


def test_tracer_2_without_override_leaves_galaxy_defaults():
    ps = PowerSpectrum(survey="meerklass_2021", band="L")
    ps.mean_center_2 = True
    ps.unitless_2 = True
    _apply_jackknife_field_settings(
        ps,
        2,
        {},
        mean_center_override=None,
        unitless_override=None,
        freeze_mean=False,
        field_mean=None,
    )
    assert ps.mean_center_2 is True
    assert ps.unitless_2 is True


def test_weighted_field_mean_is_recorded_when_the_cube_exists():
    ps = _boxed_ps()
    ps.field_1 = np.arange(int(np.prod(ps.box_ndim)), dtype=ps.real_dtype).reshape(
        ps.box_ndim
    )
    ps.weights_1 = np.ones(ps.box_ndim, dtype=ps.real_dtype)
    jc = JackknifeCovariance(ps, 2, 2, 1, pool="serial", apply_taper=False)
    expected = float(np.sum(ps.weights_1 * ps.field_1) / np.sum(ps.weights_1))
    assert jc.field_mean_1 == pytest.approx(expected)
    assert jc.field_mean_2 is None


# --------------------------------------------------------------------------- #
# painting, labels, and default patch selection
# --------------------------------------------------------------------------- #
def test_missing_hi_cube_is_painted_from_the_map():
    ps = _boxed_ps()
    jc = JackknifeCovariance(ps, 2, 2, 1, pool="serial", apply_taper=False)
    # PowerSpectrum always plants a dummy 1x1x1 cube; a missing HI field is None.
    ps.field_1 = None
    expected, _, _ = ps.grid_data_to_field()
    ps.field_1 = None
    jc._ensure_gridded_fields("auto")
    np.testing.assert_allclose(ps.field_1, expected)


def test_already_painted_hi_cube_is_not_replaced():
    ps = _boxed_ps()
    custom = np.arange(int(np.prod(ps.box_ndim)), dtype=ps.real_dtype).reshape(
        ps.box_ndim
    )
    ps.field_1 = custom
    jc = JackknifeCovariance(ps, 2, 2, 1, pool="serial", apply_taper=False)
    jc._ensure_gridded_fields("auto")
    np.testing.assert_array_equal(ps.field_1, custom)


def test_missing_galaxy_cube_is_painted_from_the_catalogue():
    ps = _boxed_ps()
    _attach_galaxies_at_surveyed_voxels(ps)
    jc = JackknifeCovariance(ps, 2, 2, 1, pool="serial", apply_taper=False)
    assert ps.field_2 is None
    expected, _, _ = ps.grid_gal_to_field()
    ps.field_2 = None
    args = jc.get_arg_list_for_parallel_boxmask([0], type="cross")
    np.testing.assert_allclose(ps.field_2, expected)
    np.testing.assert_allclose(args[0][3], expected)


def test_cross_without_galaxies_is_rejected():
    ps = _boxed_ps()
    ps.field_1 = np.ones(ps.box_ndim, dtype=ps.real_dtype)
    ps._ra_gal = np.array([])
    ps._dec_gal = np.array([])
    ps._z_gal = np.array([])
    jc = JackknifeCovariance(ps, 2, 2, 1, pool="serial", apply_taper=False)
    with pytest.raises(ValueError, match="empty"):
        jc._ensure_gridded_fields("cross")
    with pytest.raises(ValueError, match="empty"):
        jc.get_arg_list_for_parallel_cross([0])


def test_class_galaxy_labels_match_box_voxel_labels():
    ps = _boxed_ps()
    _attach_galaxies_at_surveyed_voxels(ps)
    jc = JackknifeCovariance(ps, 2, 2, 1, pool="serial", apply_taper=False)
    gal = jc.get_gal_patch_labels()
    box = jc.box_voxel_labels()
    pick = np.argwhere((ps.counts_in_box > 0) & (box >= 0))[: gal.size]
    np.testing.assert_array_equal(gal, box[tuple(pick.T)])


def test_light_patches_are_dropped_by_the_weight_threshold():
    ps = _boxed_ps()
    jc = JackknifeCovariance(ps, 2, 2, 1, pool="serial", apply_taper=False)
    frac = jc.patch_weight_fraction()
    thresh = float(np.median(frac))
    jc.min_patch_weight_fraction = thresh
    used = jc.get_default_patch_indices()
    np.testing.assert_array_equal(used, np.where(frac > thresh)[0])
    assert 0 < len(used) < jc.num_patches


# --------------------------------------------------------------------------- #
# boxmask cross, taper, and 3D return
# --------------------------------------------------------------------------- #
def test_boxmask_keep_all_cross_matches_full_sample():
    ps = _boxed_ps()
    rng = np.random.default_rng(5)
    ps.field_1 = rng.normal(size=ps.box_ndim).astype(ps.real_dtype)
    ps.field_2 = rng.normal(size=ps.box_ndim).astype(ps.real_dtype)
    dndz = np.ones(ps.box_ndim, dtype=ps.real_dtype)
    ps.weights_field_1 = None
    ps.weights_grid_1 = np.array(ps.counts_in_box, copy=True)
    ps.weights_field_2 = dndz
    ps.weights_grid_2 = (dndz > 0) * ps.counts_in_box
    p_cross, _, _ = ps.get_1d_power("cross_power_3d")
    p_hi, _, _ = ps.get_1d_power("auto_power_3d_1")
    p_gg, _, _ = ps.get_1d_power("auto_power_3d_2")
    jc = JackknifeCovariance(ps, 2, 2, 1, pool="serial", apply_taper=False)
    result = run_jackknife_boxmask(
        jc.get_ps_instance_attr_dict(),
        np.ones(ps.box_ndim, dtype=bool),
        ps.field_1,
        ps.field_2,
        ps.counts_in_box,
        ps.k1dweights,
        "validation",
        None,
        False,
        (2,),
        "binary",
        True,
        True,
        jc.box_geometry,
        dndz,
    )
    np.testing.assert_allclose(result[0], p_cross, rtol=1e-10, atol=0.0)
    p3d_to_1d, _, _ = ps.get_1d_power(result[1])
    np.testing.assert_allclose(p3d_to_1d, p_cross, rtol=1e-10, atol=0.0)
    np.testing.assert_allclose(result[2], p_hi, rtol=1e-10, atol=0.0)
    np.testing.assert_allclose(result[3], p_gg, rtol=1e-10, atol=0.0)


def test_boxmask_keep_all_with_taper_matches_tapered_full_sample():
    ps = _boxed_ps()
    rng = np.random.default_rng(6)
    ps.field_1 = rng.normal(size=ps.box_ndim).astype(ps.real_dtype)
    ps.weights_field_1 = None
    ps.weights_grid_1 = np.array(ps.counts_in_box, copy=True)
    ps.apply_taper_to_field(1, axis=[2])
    p_full, _, _ = ps.get_1d_power("auto_power_3d_1")
    field = np.array(ps.field_1, copy=True)
    jc = JackknifeCovariance(ps, 2, 2, 1, pool="serial", apply_taper=True)
    p_jk = run_jackknife_boxmask(
        jc.get_ps_instance_attr_dict(),
        np.ones(ps.box_ndim, dtype=bool),
        field,
        None,
        ps.counts_in_box,
        ps.k1dweights,
        "validation",
        None,
        True,
        (2,),
        "binary",
        False,
        False,
        jc.box_geometry,
        None,
    )[0]
    np.testing.assert_allclose(p_jk, p_full, rtol=1e-10, atol=0.0)


def test_serial_and_multiprocessing_delete_one_spectra_agree():
    ps = _boxed_ps()
    rng = np.random.default_rng(7)
    ps.field_1 = rng.normal(size=ps.box_ndim).astype(ps.real_dtype)
    patches = [0, 1]
    serial = JackknifeCovariance(ps, 2, 2, 1, pool="serial", apply_taper=False)
    parallel = JackknifeCovariance(
        ps, 2, 2, 1, pool="multiprocessing", num_process=2, apply_taper=False
    )
    r_s = serial.run(type="auto", patch_indices=patches)
    r_p = parallel.run(type="auto", patch_indices=patches)
    np.testing.assert_allclose(r_s[0][0], r_p[0][0], rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(r_s[1][0], r_p[1][0], rtol=1e-12, atol=0.0)


def test_mpi_pool_collects_starmap_results_in_order(monkeypatch):
    class _Executor:
        def __init__(self, n):
            self.n = n

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def starmap(self, fn, args):
            return [fn(*a) for a in args]

    import sys
    import types

    fake_futures = types.ModuleType("mpi4py.futures")
    fake_futures.MPIPoolExecutor = _Executor
    monkeypatch.setitem(sys.modules, "mpi4py.futures", fake_futures)
    out = _map_realisations(lambda a, b: a - b, [(5, 2), (9, 1)], "mpi", 2)
    assert out == [3, 8]


# --------------------------------------------------------------------------- #
# re-grid operator: empty mask recovers the full-sample measurement
# --------------------------------------------------------------------------- #
def test_regrid_empty_mask_recovers_full_sample_hi_power():
    ps = _boxed_ps()
    rng = np.random.default_rng(8)
    ps.data = rng.normal(size=ps.W_HI.shape)
    ps.w_HI = np.ones(ps.W_HI.shape)
    ps.field_1, _, _ = ps.grid_data_to_field()
    ps.weights_field_1 = None
    ps.weights_grid_1 = np.array(ps.counts_in_box, copy=True)
    p_full, _, _ = ps.get_1d_power("auto_power_3d_1")
    jc = JackknifeCovariance(ps, 2, 2, 1, pool="serial", apply_taper=False, regrid=True)
    empty = np.zeros(ps.W_HI.shape, dtype=bool)
    result = run_jackknife_auto(
        jc.get_ps_instance_attr_dict(),
        empty,
        ps.k1dweights,
        "validation",
        None,
        False,
        (2,),
        True,
        jc.box_geometry,
        None,
    )
    np.testing.assert_allclose(result[0], p_full, rtol=1e-8, atol=0.0)
    p3d_to_1d, _, _ = ps.get_1d_power(result[1])
    np.testing.assert_allclose(p3d_to_1d, p_full, rtol=1e-8, atol=0.0)


def test_regrid_mask_reduces_kept_counts():
    ps = _boxed_ps()
    jc = JackknifeCovariance(ps, 2, 2, 1, pool="serial", apply_taper=False)
    masks = jc.get_patch_masks()
    frac = jc.patch_weight_fraction()
    j = int(np.argmax(frac))
    jk = PowerSpectrum(**jc.get_ps_instance_attr_dict())
    _, _, counts_keep = _grid_jackknifed_map(
        jk,
        masks[j],
        return_kept_counts=False,
        box_reference=jc.box_geometry,
        return_kept_window=True,
    )
    assert counts_keep is not None
    assert counts_keep.sum() < ps.counts_in_box.sum()


def test_regrid_cross_args_drop_galaxies_in_the_removed_patch():
    ps = _boxed_ps()
    labels, _ = _attach_galaxies_at_surveyed_voxels(ps)
    jc = JackknifeCovariance(ps, 2, 2, 1, pool="serial", apply_taper=False, regrid=True)
    gal_labels = jc.get_gal_patch_labels()
    j = int(gal_labels[gal_labels >= 0][0])
    arg_list = jc.get_arg_list_for_parallel_cross([j])
    ra_keep, dec_keep, z_keep = arg_list[0][2]
    kept = gal_labels != j
    np.testing.assert_array_equal(ra_keep, ps.ra_gal[kept])
    np.testing.assert_array_equal(dec_keep, ps.dec_gal[kept])
    np.testing.assert_array_equal(z_keep, ps.z_gal[kept])
    assert kept.sum() < gal_labels.size


def test_regrid_empty_mask_cross_matches_full_sample():
    ps = _boxed_ps()
    rng = np.random.default_rng(9)
    ps.data = rng.normal(size=ps.W_HI.shape)
    _attach_galaxies_at_surveyed_voxels(ps, n=20)
    ps.field_1, _, _ = ps.grid_data_to_field()
    ps.field_2, _, _ = ps.grid_gal_to_field()
    dndz = np.ones(ps.box_ndim, dtype=ps.real_dtype)
    ps.weights_field_1 = None
    ps.weights_grid_1 = np.array(ps.counts_in_box, copy=True)
    ps.weights_field_2 = dndz
    ps.weights_grid_2 = (dndz > 0) * ps.counts_in_box
    p_full, _, _ = ps.get_1d_power("cross_power_3d")
    jc = JackknifeCovariance(ps, 2, 2, 1, pool="serial", apply_taper=False, regrid=True)
    empty = np.zeros(ps.W_HI.shape, dtype=bool)
    result = run_jackknife_cross(
        jc.get_ps_instance_attr_dict(),
        empty,
        (ps.ra_gal, ps.dec_gal, ps.z_gal),
        ps.k1dweights,
        "validation",
        None,
        False,
        (2,),
        "binary",
        False,
        True,
        jc.box_geometry,
        dndz,
    )
    np.testing.assert_allclose(result[0], p_full, rtol=1e-8, atol=0.0)
    p_hi, _, _ = ps.get_1d_power("auto_power_3d_1")
    p_gg, _, _ = ps.get_1d_power("auto_power_3d_2")
    np.testing.assert_allclose(result[1], p_hi, rtol=1e-8, atol=0.0)
    np.testing.assert_allclose(result[2], p_gg, rtol=1e-8, atol=0.0)


def test_regrid_run_auto_matches_direct_worker():
    ps = _boxed_ps()
    rng = np.random.default_rng(10)
    ps.data = rng.normal(size=ps.W_HI.shape)
    ps.field_1, _, _ = ps.grid_data_to_field()
    jc = JackknifeCovariance(ps, 2, 2, 1, pool="serial", apply_taper=False, regrid=True)
    patches = jc.get_default_patch_indices()[:2]
    results = jc.run(type="auto", patch_indices=patches)
    args = jc.get_arg_list_for_parallel_auto(patches)
    direct = [run_jackknife_auto(*a) for a in args]
    np.testing.assert_allclose(results[0][0], direct[0][0], rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(results[1][0], direct[1][0], rtol=1e-12, atol=0.0)
