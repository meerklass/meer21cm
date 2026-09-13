import logging

import numpy as np
import pytest

import meer21cm.stack as stack_mod
from meer21cm.stack import (
    Stacking,
    accumulate_stack_chunk_results,
    init_stack_chunk_worker,
    run_stack_chunk,
    stack,
)


def _meerklass_lband_kwargs():
    return dict(
        ra_range=(334, 357),
        dec_range=(-35, -26.5),
        survey="meerklass_2021",
        band="L",
    )


def _two_source_maps(st):
    data = st.data.copy()
    source_1 = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])
    source_2 = np.array([1, 2, 3, 4, 3, 2, 1])
    data[80, 30, 80 - 4 : 80 + 5] = source_1
    data[50, 40, 140 - 3 : 140 + 4] = source_2
    ra_g = np.array([st.ra_map[80, 30], st.ra_map[50, 40]])
    dec_g = np.array([st.dec_map[80, 30], st.dec_map[50, 40]])
    z_g = np.array([st.z_ch[80], st.z_ch[140]])
    st.data = data
    st._ra_gal = ra_g
    st._dec_gal = dec_g
    st._z_gal = z_g
    return st


def test_run_stack_matches_stack_function():
    st = Stacking(**_meerklass_lband_kwargs())
    _two_source_maps(st)
    ref_map, ref_weight = stack(
        st,
        weighting=st.stack_weighting,
        stack_angular_num_nearby_pix=st.stack_angular_num_nearby_pix,
        symmetrize=st.symmetrize,
    )
    out_map, out_weight = st.run_stack()
    assert np.array_equal(out_map, ref_map)
    assert np.array_equal(out_weight, ref_weight)
    assert np.array_equal(st._stack_3d, ref_map)
    assert np.array_equal(st._stack_weight, ref_weight)


def test_invalid_weighting_and_stack_space():
    with pytest.raises(ValueError, match="weighting"):
        Stacking(weighting="random", **_meerklass_lband_kwargs())
    with pytest.raises(ValueError, match="stack_space"):
        Stacking(stack_space="sky", **_meerklass_lband_kwargs())
    with pytest.raises(NotImplementedError, match="config"):
        Stacking(stack_space="config", **_meerklass_lband_kwargs())


def test_stack_3d_warns_before_run(caplog):
    st = Stacking(**_meerklass_lband_kwargs())
    with caplog.at_level(logging.WARNING, logger="meer21cm.stack"):
        got = st.stack_3d
    assert got is None
    assert "run_stack" in caplog.text
    assert st._stack_3d is None
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="meer21cm.stack"):
        got_w = st.stack_weight
    assert got_w is None
    assert "run_stack" in caplog.text


def test_lazy_import():
    import meer21cm

    assert meer21cm.Stacking is Stacking


@pytest.mark.parametrize("weighting", ["conventional", "quadratic"])
@pytest.mark.parametrize("use_worker_object", [False, True])
def test_galaxy_chunks_match_run_stack(weighting, use_worker_object):
    st = Stacking(weighting=weighting, **_meerklass_lband_kwargs())
    _two_source_maps(st)
    ref_map, ref_weight = st.run_stack()
    args = st.get_arg_list_for_galaxy_chunks(
        n_chunks=2, use_worker_object=use_worker_object
    )
    assert len(args) == 2
    if use_worker_object:
        init_stack_chunk_worker(st.stack_chunk_worker_kwargs())
    try:
        results = [run_stack_chunk(kw, chunk) for kw, chunk in args]
        out_map, out_weight = st.accumulate_stack_chunks(results)
    finally:
        stack_mod._STACK_CHUNK_WORKER = None
    assert np.allclose(out_map, ref_map)
    assert np.allclose(out_weight, ref_weight)


def test_galaxy_chunks_n_chunks_and_worker_error():
    stack_mod._STACK_CHUNK_WORKER = None
    st = Stacking(**_meerklass_lband_kwargs())
    _two_source_maps(st)
    with pytest.raises(ValueError, match="n_chunks"):
        st.get_arg_list_for_galaxy_chunks(n_chunks=0)
    args = st.get_arg_list_for_galaxy_chunks(n_chunks=8)
    assert len(args) == 2
    with pytest.raises(RuntimeError, match="init_stack_chunk_worker"):
        run_stack_chunk({"use_worker_object": True}, args[0][1])
    with pytest.raises(ValueError, match="at least one"):
        accumulate_stack_chunk_results([], "conventional")
