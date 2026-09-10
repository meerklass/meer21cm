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
from meer21cm.jackknife import (
    JackknifeCovariance,
    _check_box_geometry,
    _get_dndz_box,
    _normalise_weights_argument,
    box_geometry,
)


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
