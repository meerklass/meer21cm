import os

import numpy as np
import pytest

from meer21cm.mpca import (
    apply_mpca_matrices,
    get_mpca_matrices,
    mpca_clean,
    starlet_forward_healpix,
    starlet_forward_wcs,
    starlet_split,
)
from meer21cm.util import pca_clean

REF_FILE = os.path.join(os.path.dirname(__file__), "mpca_reference_synthetic.npz")


@pytest.fixture(scope="module")
def ref():
    """
    Output of the code of Carucci et al. 2025 (A&A 703, A222) on a small synthetic cube
    (24 channels x 20 x 16 pixels), stored channel first as (n_ch, nx, ny).
    """
    r = np.load(REF_FILE)
    out = {k: r[k] for k in r.files}
    out["d"] = np.moveaxis(out["X"], 0, -1)  # meer21cm layout (nx, ny, n_ch)
    out["W"] = np.repeat(out["w"][:, :, None], out["d"].shape[-1], axis=-1)
    return out


def _close(res, ref_cube, rtol=1e-5):
    res = np.moveaxis(res, -1, 0)
    assert np.abs(res - ref_cube).max() <= rtol * np.abs(ref_cube).max()


def test_paper_pipelines(ref):
    d, W = ref["d"], ref["W"]
    n, nl, ns = int(ref["n"]), int(ref["n_l"]), int(ref["n_s"])
    _close(pca_clean(d, n, mean_center=True), ref["PCA"])
    _close(pca_clean(d, n), ref["SVD"])
    _close(pca_clean(d, n, weights=W), ref["SVDw"])
    _close(mpca_clean(d, nl, ns, mean_center=True), ref["mPCA"])
    _close(mpca_clean(d, nl, ns), ref["mSVD"])
    _close(
        mpca_clean(
            d,
            nl,
            ns,
            weights=W,
            mean_center=True,
            mean_center_weights=np.ones_like(W),
        ),
        ref["mPCAw"],
    )
    _close(mpca_clean(d, nl, ns, weights=W), ref["mSVDw"])


def test_embedded_footprint(ref):
    # the paper cleaned a cropped rectangle; meer21cm keeps the full map with zeros outside
    d, W = ref["d"], ref["W"]
    big = np.zeros((d.shape[0] + 7, d.shape[1] + 5, d.shape[2]))
    big[3:-4, 2:-3] = d
    Wb = np.zeros_like(big)
    Wb[3:-4, 2:-3] = W
    res = mpca_clean(
        big,
        int(ref["n_l"]),
        int(ref["n_s"]),
        weights=Wb,
        mean_center=True,
        mean_center_weights=(Wb > 0).astype(float),
    )
    _close(res[3:-4, 2:-3], ref["mPCAw"])
    assert np.all(res[:3] == 0) and np.all(res[:, -3:] == 0)


def test_starlet_wcs_reconstruction(ref):
    x = ref["X"]
    for J in (1, 2, 3):
        wt = starlet_forward_wcs(x, J=J)
        assert wt.shape == x.shape + (J + 1,)
        assert np.allclose(wt.sum(-1), x, rtol=0, atol=1e-12 * np.abs(x).max())
    # coarse scale of a constant map is the constant
    c = starlet_forward_wcs(np.ones((2, 9, 7)))
    assert np.allclose(c[..., -1], 1) and np.allclose(c[..., 0], 0)


def test_los_axis(ref):
    d = ref["d"]
    r1 = mpca_clean(d, 3, 5, mean_center=True)
    r0 = mpca_clean(np.moveaxis(d, -1, 0), 3, 5, mean_center=True, los_axis=0)
    assert np.allclose(np.moveaxis(r0, 0, -1), r1)
    s1 = starlet_split(d)
    s0 = starlet_split(np.moveaxis(d, -1, 0), los_axis=0)
    assert s1.shape == d.shape + (2,)
    assert np.allclose(np.moveaxis(s0, 0, -2), s1)


def test_no_modes_and_outputs(ref):
    d = ref["d"]
    res = mpca_clean(d, 0, 0, mean_center=True)
    assert np.allclose(res, d - d.mean(axis=(0, 1), keepdims=True))
    res, A_l, A_s, scales = mpca_clean(d, 3, 5, return_A=True, return_scales=True)
    nz = d.shape[-1]
    assert A_l.shape == (nz, 3) and A_s.shape == (nz, 5)
    assert np.allclose(A_l.T @ A_l, np.eye(3)) and np.allclose(A_s.T @ A_s, np.eye(5))
    assert scales.shape == d.shape + (2,)
    assert np.allclose(scales.sum(-1), res)


def test_irregular_footprint(ref):
    d = ref["d"].copy()
    nx, ny, _ = d.shape
    ix, iy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    fp = (ix - nx / 2) ** 2 / 64 + (iy - ny / 2) ** 2 / 36 < 1
    res = mpca_clean(d, 3, 5, mean_center=True, footprint=fp)
    assert np.all(res[~fp] == 0)
    # values outside the footprint (even NaN) do not matter
    d2 = d.copy()
    d2[~fp] = np.nan
    res2 = mpca_clean(d2, 3, 5, mean_center=True, footprint=fp)
    assert np.allclose(res, res2)
    # default footprint from non-zero weights
    W = np.repeat(fp[:, :, None].astype(float), d.shape[-1], axis=-1)
    res3 = mpca_clean(d2, 3, 5, weights=W, mean_center=True)
    assert np.allclose(res, res3)
    # fill value only affects the starlet near the edge, not the definition of the output
    res4 = mpca_clean(d, 3, 5, mean_center=True, footprint=fp, fill=0.0)
    assert np.all(res4[~fp] == 0) and np.all(np.isfinite(res4))


def test_errors(ref):
    d = ref["d"].copy()
    with pytest.raises(ValueError):
        mpca_clean(d, 3, 5, weights=np.ones(d.shape[:2]))
    d[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        mpca_clean(d, 3, 5, footprint=np.ones(d.shape[:2], dtype=bool))
    with pytest.raises(ValueError):
        mpca_clean(np.ones(5), 1, 1)


# ---------------------------------------------------------------------------- HEALPix
def _healpix_maps(nside=16, n_ch=12, seed=1):
    hp = pytest.importorskip("healpy")
    rng = np.random.default_rng(seed)
    lmax = 3 * nside
    nu = np.linspace(0.97, 1.02, n_ch)
    cl = 1.0 / (np.arange(lmax + 1) + 5.0) ** 2
    maps = np.zeros((n_ch, hp.nside2npix(nside)))
    for beta in (-2.7, -2.1):
        np.random.seed(int(rng.integers(1e6)))
        m = hp.synfast(cl, nside, lmax=lmax)
        maps += nu[:, None] ** beta * (10.0 + m)[None]
    maps += 1e-3 * rng.standard_normal(maps.shape)
    return hp, nside, maps


def test_starlet_healpix_full_sky():
    from meer21cm.mpca import _h_harmonic

    hp, nside, maps = _healpix_maps(nside=32)
    wt = starlet_forward_healpix(maps, nside)
    assert wt.shape == maps.shape + (2,)
    assert np.allclose(wt.sum(-1), maps)
    # dyadic split: coarse-scale power = h(ell)^2 x input power (zero for ell >= lmax / 2).
    # Checked up to ell = 2 nside, where map2alm is accurate (healpy quadrature errors
    # of ~1e-2 in the power appear above).
    lmax = 3 * nside
    cl = hp.anafast(wt[0, :, 1], lmax=lmax, iter=3)
    cl_in = hp.anafast(maps[0], lmax=lmax, iter=3)
    ell = np.arange(2, 2 * nside + 1)
    ratio = cl[ell] / cl_in[ell]
    assert np.abs(ratio - _h_harmonic(lmax, 1)[ell] ** 2).max() < 2e-2


def test_mpca_healpix_sparse():
    hp, nside, maps = _healpix_maps()
    from meer21cm.skymap import HealpixSkyMap

    sky = HealpixSkyMap(nside, ra_range=(100.0, 200.0), dec_range=(-40.0, 0.0))
    pid = sky.pixel_id
    d = maps[:, pid].T  # meer21cm layout (n_pix, n_ch)
    res = mpca_clean(d, 2, 3, mean_center=True, hp_nside=nside, pixel_id=pid)
    assert res.shape == d.shape and np.all(np.isfinite(res))
    res_sky = mpca_clean(d, 2, 3, mean_center=True, skymap=sky)
    assert np.allclose(res, res_sky)
    # two smooth components: few modes remove almost everything but the noise
    assert res.std() < 1e-2 * d.std()
    s = starlet_split(d, skymap=sky)
    assert np.allclose(s.sum(-1), d)
    with pytest.raises(ValueError):
        mpca_clean(d, 2, 3, hp_nside=nside)


@pytest.mark.parametrize("weighted", [False, True])
def test_fixed_matrices(ref, weighted):
    """Matrices from one cube, applied to it, give mpca_clean; with fixed matrices it is linear."""
    d, W = ref["d"], ref["W"]
    kw = dict(weights=W, mean_center=True) if weighted else dict(mean_center=True)
    rng = np.random.default_rng(0)
    m = 0.1 * d.std() * rng.normal(size=d.shape)
    A_c, A_f = get_mpca_matrices(d + m, 4, 8, **kw)
    res = mpca_clean(d + m, 4, 8, **kw)
    scale = np.abs(res).max()
    assert np.abs(apply_mpca_matrices(d + m, A_c, A_f, **kw) - res).max() <= 1e-10 * scale
    lin = (apply_mpca_matrices(d, A_c, A_f, **kw) + apply_mpca_matrices(m, A_c, A_f, **kw)
           - apply_mpca_matrices(d + m, A_c, A_f, **kw))
    assert np.abs(lin).max() <= 1e-9 * scale
    # spectral axis first
    d0 = np.moveaxis(d + m, -1, 0)
    kw0 = dict(kw, los_axis=0)
    if weighted:
        kw0["weights"] = np.moveaxis(W, -1, 0)
    res0 = apply_mpca_matrices(d0, A_c, A_f, **kw0)
    assert np.abs(np.moveaxis(res0, 0, -1) - res).max() <= 1e-10 * scale
