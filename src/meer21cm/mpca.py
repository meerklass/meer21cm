r"""
Multiscale PCA (mPCA) foreground cleaning.

Implements the multiscale cleaning of Carucci et al. 2025 (A&A 703, A222) [1]:
the maps are split along the sky directions into a coarse (large angular scales)
and a fine (small angular scales) part with an isotropic undecimated wavelet
("starlet") transform, a separate PCA is run on each part, and the foreground
model of each part is subtracted.

.. math::
    \vec{d} = \vec{d}_{\rm coarse} + \vec{d}_{\rm fine}, \qquad
    \vec{d}_{\rm res} = (I - A_{\rm L} A_{\rm L}^T)\,\vec{d}_{\rm coarse}
    + (I - A_{\rm S} A_{\rm S}^T)\,\vec{d}_{\rm fine},

where :math:`A_{\rm L}` (:math:`A_{\rm S}`) are the first :math:`N_{\rm L}`
(:math:`N_{\rm S}`) eigenvectors of the frequency-frequency covariance of the coarse
(fine) maps. Each PCA is done by :func:`meer21cm.util.pca_clean`, so weights and
mean-centring follow the same conventions as standard PCA cleaning in meer21cm.

Both map formats of meer21cm are supported:

- WCS cubes ``(nx, ny, n_ch)``: planar starlet (B3-spline "a trous" algorithm,
  mirror boundary), as in [1];
- sparse HEALPix maps ``(n_pix, n_ch)`` with their ``pixel_id`` (RING ordering):
  spherical starlet in harmonic space (as pyMRS / gmca4im).

Only the dyadic starlet split with ``J`` detail scales is implemented (``J = 1`` in [1]).

References
----------
.. [1] Carucci, I. P. et al., "Hydrogen intensity mapping with MeerKAT: preserving
       cosmological signal by optimising contaminant separation", A&A 703, A222 (2025).
"""

import numpy as np
from scipy.ndimage import convolve1d

from .util import pca_clean, mean_center_signal

__all__ = [
    "B3_SPLINE_FILTER",
    "starlet_forward_wcs",
    "starlet_forward_healpix",
    "starlet_split",
    "mpca_clean",
]

#: B3-spline scaling filter of the starlet transform.
B3_SPLINE_FILTER = np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0


# ----------------------------------------------------------------------------
# Starlet transforms. Internal layout: channel first, (n_ch, ...spatial).
# ----------------------------------------------------------------------------
def starlet_forward_wcs(cube, J=1, h=B3_SPLINE_FILTER):
    """
    Planar starlet transform of a channel-first cube.

    Parameters
    ----------
        cube: array.
            Maps of shape ``(n_ch, nx, ny)``. Must be finite everywhere.
        J: int, default 1.
            Number of detail (wavelet) scales.
        h: array, default :data:`B3_SPLINE_FILTER`.
            1D filter of the "a trous" algorithm.

    Returns
    -------
        wt: array.
            Shape ``(n_ch, nx, ny, J + 1)``. ``wt[..., :J]`` are the detail scales
            (finest first), ``wt[..., J]`` the coarse scale; ``wt.sum(-1) == cube``.
    """
    cube = np.asarray(cube, dtype=float)
    if cube.ndim != 3:
        raise ValueError(f"expected (n_ch, nx, ny), got shape {cube.shape}")
    if not np.all(np.isfinite(cube)):
        raise ValueError("starlet_forward_wcs: input must be finite")
    h = np.asarray(h, dtype=float)
    wt = np.empty(cube.shape + (J + 1,))
    c = cube
    for j in range(J):
        step = 2**j
        g = np.zeros((len(h) - 1) * step + 1)
        g[::step] = h
        c_next = convolve1d(c, g, axis=1, mode="reflect")
        c_next = convolve1d(c_next, g, axis=2, mode="reflect")
        wt[..., j] = c - c_next
        c = c_next
    wt[..., J] = c
    return wt


def _spline_b3(x):
    """B3-spline scaling profile (pyMRS ``spline2``)."""
    x = np.abs(x)
    return (3.0 / 2.0) * (1.0 / 12.0) * (
        np.abs(x - 2.0) ** 3
        - 4.0 * np.abs(x - 1.0) ** 3
        + 6.0 * x**3
        - 4.0 * np.abs(x + 1.0) ** 3
        + np.abs(x + 2.0) ** 3
    )


def _h_harmonic(lmax, scale):
    """Low-pass ratio filter h(ell), ell = 0..lmax, of the spherical starlet (pyMRS ``compute_h``)."""
    ell = np.arange(lmax + 1, dtype=float)
    h = _spline_b3(2.0 * 2.0 * scale * ell / lmax) / (
        _spline_b3(2.0 * scale * ell / lmax) + 1e-6
    )
    h[int(lmax / (2.0 * scale)) :] = 0.0
    return h


def _inpaint_healpix(m, bad, lmax, n_inpaint):
    """
    Fill unobserved pixels of a full-sky RING map by multi-scale diffusive inpainting:
    start from the mean of the observed pixels, then replace the unobserved pixels by a
    Gaussian-smoothed version of the map, with FWHM decreasing from 60 deg to ~2 pixels.
    Observed pixels are never changed. Linear in the input map.
    """
    import healpy as hp

    nside = hp.npix2nside(m.size)
    m[bad] = np.mean(m[~bad])
    for fwhm in np.geomspace(np.radians(60.0), 2 * hp.nside2resol(nside), n_inpaint):
        alm = hp.map2alm(m, lmax=lmax, iter=0)
        ms = hp.alm2map(
            hp.almxfl(alm, hp.gauss_beam(fwhm, lmax=lmax)), nside=nside, lmax=lmax
        )
        m[bad] = ms[bad]
    return m


def starlet_forward_healpix(
    cube, hp_nside, pixel_id=None, J=1, lmax=None, fill="mean", n_iter=3, n_inpaint=10
):
    """
    Spherical starlet transform of channel-first HEALPix maps.

    Parameters
    ----------
        cube: array.
            Maps of shape ``(n_ch, n_pix)``, RING ordering; either full-sky
            (``n_pix = 12 * hp_nside**2``, ``pixel_id=None``) or sparse, with the
            HEALPix index of each column in ``pixel_id``. Must be finite.
        hp_nside: int.
            HEALPix nside.
        pixel_id: array, default None.
            HEALPix index of each pixel for sparse maps.
        J: int, default 1.
            Number of detail scales. The coarse scale keeps ell < lmax / 2**J.
        lmax: int, default None.
            Maximum multipole, default ``3 * hp_nside``.
        fill: str or float, default "mean".
            Value of the unobserved pixels before the harmonic transform:
            ``"mean"`` (per channel mean of the observed pixels), ``"inpaint"``
            (smooth continuation across the footprint edge) or a number.
            All choices are linear in the data.
        n_iter: int, default 3.
            Iterations of ``healpy.map2alm``.
        n_inpaint: int, default 10.
            Smoothing steps for ``fill="inpaint"``.

    Returns
    -------
        wt: array.
            Shape ``(n_ch, n_pix, J + 1)`` on the input pixels; ``wt.sum(-1) == cube``.
    """
    import healpy as hp

    cube = np.asarray(cube, dtype=float)
    if cube.ndim != 2:
        raise ValueError(f"expected (n_ch, n_pix), got shape {cube.shape}")
    if not np.all(np.isfinite(cube)):
        raise ValueError("starlet_forward_healpix: input must be finite")
    n_ch, n_pix = cube.shape
    npix_sphere = hp.nside2npix(int(hp_nside))
    if pixel_id is None:
        if n_pix != npix_sphere:
            raise ValueError(
                f"{n_pix} pixels is not full sky for nside={hp_nside}; pass pixel_id"
            )
        pixel_id = np.arange(npix_sphere)
    pixel_id = np.asarray(pixel_id, dtype=np.int64).ravel()
    if pixel_id.size != n_pix:
        raise ValueError("pixel_id must have one entry per map pixel")
    lmax = 3 * int(hp_nside) if lmax is None else int(lmax)
    if lmax // 2**J < 1:
        raise ValueError(f"J = {J} is too large for lmax = {lmax}")
    if isinstance(fill, str) and fill not in ("mean", "inpaint"):
        raise ValueError("fill must be 'mean', 'inpaint' or a number")
    filters = [_h_harmonic(lmax, 2**j) for j in range(J)]
    bad = np.ones(npix_sphere, dtype=bool)
    bad[pixel_id] = False
    wt = np.empty((n_ch, n_pix, J + 1))
    for i in range(n_ch):
        m = np.zeros(npix_sphere)
        m[pixel_id] = cube[i]
        if bad.any():
            if fill == "mean":
                m[bad] = cube[i].mean()
            elif fill == "inpaint":
                m = _inpaint_healpix(m, bad, lmax, n_inpaint)
            else:
                m[bad] = float(fill)
        alm = hp.map2alm(m, lmax=lmax, iter=n_iter)
        current = m
        for j in range(J):
            alm = hp.almxfl(alm, filters[j])
            smooth = hp.alm2map(alm, nside=int(hp_nside), lmax=lmax)
            wt[i, :, j] = (current - smooth)[pixel_id]
            current = smooth
        wt[i, :, J] = current[pixel_id]
    return wt


def _fill_box(cube, inside, fill):
    """Give the pixels of a channel-first box outside the footprint a value (planar case)."""
    if inside.all():
        return cube
    if isinstance(fill, str) and fill != "mean":
        raise ValueError("fill must be 'mean' or a number for WCS maps")
    cube = cube.copy()
    if fill == "mean":
        val = cube[:, inside].mean(axis=1)
        cube[:, ~inside] = val[:, None]
    else:
        cube[:, ~inside] = float(fill)
    return cube


def starlet_split(
    signal,
    J=1,
    los_axis=-1,
    footprint=None,
    fill="mean",
    hp_nside=None,
    pixel_id=None,
    skymap=None,
    lmax=None,
    n_iter=3,
):
    """
    Split maps into starlet scales, in meer21cm's map layout.

    For WCS cubes the transform is done on the smallest rectangle containing the
    footprint; pixels of that rectangle outside the footprint are first given the
    value ``fill``. Results outside the footprint are zero.

    Parameters
    ----------
        signal: array.
            3D WCS ``(nx, ny, n_ch)`` or 2D HEALPix ``(n_pix, n_ch)`` maps,
            spectral axis ``los_axis``.
        J: int, default 1.
            Number of detail scales.
        los_axis: int, default -1.
            Index of the spectral axis.
        footprint: array of bool, default None.
            Sky pixels to use, shape of the map without the spectral axis.
            Default: pixels that are finite in every channel.
        fill: str or float, default "mean".
            See :func:`starlet_forward_healpix` (``"inpaint"`` is HEALPix only).
        hp_nside, pixel_id: default None.
            HEALPix geometry (or pass ``skymap``).
        skymap: :class:`meer21cm.skymap.HealpixSkyMap`, default None.
            Takes ``hp_nside`` and ``pixel_id`` from it.
        lmax, n_iter:
            HEALPix only, see :func:`starlet_forward_healpix`.

    Returns
    -------
        scales: array.
            Shape ``signal.shape + (J + 1,)``; ``scales[..., :J]`` detail scales,
            ``scales[..., J]`` coarse scale. Inside the footprint ``scales.sum(-1) == signal``.
    """
    x = np.moveaxis(np.asarray(signal, dtype=float), los_axis, 0)
    inside = _footprint(x, footprint)
    out = np.zeros(x.shape + (J + 1,))
    if x.ndim == 3:
        bx, by = _bounding_box(inside)
        box = _fill_box(np.nan_to_num(x[:, bx, by]), inside[bx, by], fill)
        out[:, bx, by] = starlet_forward_wcs(box, J=J)
    elif x.ndim == 2:
        hp_nside, pixel_id = _healpix_geometry(x.shape[1], hp_nside, pixel_id, skymap)
        wt = starlet_forward_healpix(
            x[:, inside],
            hp_nside,
            pixel_id[inside],
            J=J,
            lmax=lmax,
            fill=fill,
            n_iter=n_iter,
        )
        out[:, inside] = wt
    else:
        raise ValueError("expected a 3D WCS cube or 2D HEALPix maps")
    out[:, ~inside] = 0.0
    return np.moveaxis(out, 0, los_axis if los_axis >= 0 else los_axis - 1)


def _footprint(x, footprint):
    """Boolean sky footprint for a channel-first array."""
    finite = np.all(np.isfinite(x), axis=0)
    if footprint is None:
        inside = finite
    else:
        inside = np.asarray(footprint, dtype=bool)
        if inside.shape != x.shape[1:]:
            raise ValueError(
                f"footprint shape {inside.shape} does not match the sky shape {x.shape[1:]}"
            )
        if not np.all(finite[inside]):
            raise ValueError("signal is not finite everywhere inside the footprint")
    if not inside.any():
        raise ValueError("empty footprint")
    return inside


def _bounding_box(inside):
    ix, iy = np.where(inside)
    return slice(ix.min(), ix.max() + 1), slice(iy.min(), iy.max() + 1)


def _healpix_geometry(n_pix, hp_nside, pixel_id, skymap):
    if skymap is not None:
        if getattr(skymap, "format", None) == "wcs" or not hasattr(skymap, "pixel_id"):
            raise ValueError("skymap must be a HealpixSkyMap for HEALPix maps")
        hp_nside, pixel_id = skymap.hp_nside, skymap.pixel_id
    if hp_nside is None:
        raise ValueError("HEALPix maps need hp_nside (and pixel_id), or skymap")
    if pixel_id is None:
        if n_pix != 12 * int(hp_nside) ** 2:
            raise ValueError("sparse HEALPix maps need pixel_id")
        pixel_id = np.arange(n_pix)
    pixel_id = np.asarray(pixel_id, dtype=np.int64).ravel()
    if pixel_id.size != n_pix:
        raise ValueError("pixel_id must have one entry per map pixel")
    return int(hp_nside), pixel_id


# ----------------------------------------------------------------------------
# mPCA
# ----------------------------------------------------------------------------
def mpca_clean(
    signal,
    N_coarse,
    N_fine,
    weights=None,
    mean_center=False,
    mean_center_weights=None,
    los_axis=-1,
    J=1,
    footprint=None,
    fill="mean",
    hp_nside=None,
    pixel_id=None,
    skymap=None,
    lmax=None,
    n_iter=3,
    return_A=False,
    return_scales=False,
):
    r"""
    Multiscale PCA (mPCA) cleaning of map data (3D WCS cube or 2D HEALPix ``(n_pix, n_ch)``),
    following Carucci et al. 2025 (A&A 703, A222).

    The (optionally mean-centred) maps are split with the starlet transform into a coarse
    scale and ``J`` detail scales (:func:`starlet_split`). :func:`meer21cm.util.pca_clean`
    is then run on the coarse scale removing ``N_coarse`` modes, and on the detail scales
    (all ``J`` stacked together, one mixing matrix) removing ``N_fine`` modes, with the
    same ``weights``. The cleaned maps are the sum of the cleaned scales.

    The 2025 paper's pipelines, with ``w`` the hits averaged along frequency
    (one weight per pixel, repeated over channels), correspond to:

    - mPCA: ``mpca_clean(d, N_L, N_S, mean_center=True)``;
    - mSVD: ``mpca_clean(d, N_L, N_S)``;
    - mPCAw: ``mpca_clean(d, N_L, N_S, weights=w, mean_center=True, mean_center_weights=np.ones_like(w))``;
    - mSVDw: ``mpca_clean(d, N_L, N_S, weights=w)``.

    With weights that change with frequency (e.g. the hits per channel) the covariance is
    the weighted one of :func:`meer21cm.util.pca_clean`, which differs from the paper's.

    The mean-centring (if any) is done on the full maps before the split, over the footprint
    pixels. Mean-centring a single scale is not supported.

    For WCS cubes the transform is done on the smallest rectangle containing the footprint;
    if the footprint does not fill it, the remaining pixels are given ``fill`` before the
    transform (they do not enter the covariances). The paper used a rectangular footprint.
    The output is zero outside the footprint.

    The cleaning is not a frequency-only operation, so it can not be used with
    :class:`meer21cm.transfer.TransferFunction` in its current form.

    Parameters
    ----------
        signal: array.
            Map cube: 3D WCS ``(nx, ny, n_ch)`` or 2D HEALPix ``(n_pix, n_ch)``,
            spectral axis ``los_axis`` (default last).
        N_coarse: int.
            Number of modes removed on the coarse scale (:math:`N_{\rm L}` in the paper).
        N_fine: int.
            Number of modes removed on the detail scales (:math:`N_{\rm S}` in the paper).
        weights: array, default None.
            Weights of each element, as in :func:`meer21cm.util.pca_clean`.
            Must have the shape of ``signal``. Default: uniform.
        mean_center: bool, default False.
            Whether to mean-centre the maps (per channel, over the footprint) before the split.
        mean_center_weights: array, default None.
            Weights for the mean; default follows ``weights``.
        los_axis: int, default -1.
            Index of the spectral axis.
        J: int, default 1.
            Number of starlet detail scales (``J = 1`` in the paper).
        footprint: array of bool, default None.
            Sky pixels to clean (shape of the map without the spectral axis).
            Default: pixels finite in every channel and, if ``weights`` is given,
            with non-zero weight in at least one channel.
        fill: str or float, default "mean".
            Value given to the pixels outside the footprint before the starlet transform
            (``"mean"``, a number, or ``"inpaint"`` for HEALPix).
        hp_nside: int, default None.
            HEALPix nside (HEALPix maps only).
        pixel_id: array, default None.
            HEALPix index (RING) of each map pixel, for sparse maps.
        skymap: :class:`meer21cm.skymap.HealpixSkyMap`, default None.
            Alternative to ``hp_nside`` and ``pixel_id``.
        lmax: int, default None.
            HEALPix: maximum multipole of the transform, default ``3 * hp_nside``.
        n_iter: int, default 3.
            HEALPix: iterations of ``healpy.map2alm``.
        return_A: bool, default False.
            Whether to also return the coarse and fine mixing matrices.
        return_scales: bool, default False.
            Whether to also return the cleaned scales, shape ``signal.shape + (J + 1,)``
            (coarse last).

    Returns
    -------
        residual: array.
            The cleaned maps, same shape as ``signal``.
        A_coarse, A_fine: array, if ``return_A=True``.
            Mixing matrices ``(n_ch, N_coarse)`` and ``(n_ch, N_fine)``.
        residual_scales: array, if ``return_scales=True``.
            The cleaned maps of each scale.
    """
    signal = np.asarray(signal, dtype=float)
    if signal.ndim not in (2, 3):
        raise ValueError("expected a 3D WCS cube or 2D HEALPix maps")
    x = np.moveaxis(signal, los_axis, 0)
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != signal.shape:
            raise ValueError("weights must have the same shape as signal")
        w = np.moveaxis(weights, los_axis, 0)
    else:
        w = None
    if footprint is None:
        inside = np.all(np.isfinite(x), axis=0)
        if w is not None:
            inside &= np.any(w > 0, axis=0)
    else:
        inside = _footprint(x, footprint)
    if not inside.any():
        raise ValueError("empty footprint")
    n_ch = x.shape[0]

    # data and weights of the footprint pixels, (n_ch, n_obs)
    d = x[:, inside]
    w_obs = None if w is None else w[:, inside]
    if mean_center:
        if mean_center_weights is None:
            mw = w_obs
        else:
            mw = np.moveaxis(np.asarray(mean_center_weights, dtype=float), los_axis, 0)
            mw = mw[:, inside]
        d = mean_center_signal(d, mw, los_axis=0)
    x_mc = np.zeros_like(x)
    x_mc[:, inside] = d

    scales = starlet_split(
        x_mc,
        J=J,
        los_axis=0,
        footprint=inside,
        fill=fill,
        hp_nside=hp_nside,
        pixel_id=pixel_id,
        skymap=skymap,
        lmax=lmax,
        n_iter=n_iter,
    )
    s_obs = scales[:, inside]  # (n_ch, n_obs, J + 1)
    n_obs = s_obs.shape[1]

    coarse = s_obs[..., J]
    res_coarse, A_coarse = pca_clean(
        coarse, N_coarse, weights=w_obs, los_axis=0, return_A=True
    )
    fine = s_obs[..., :J].reshape(n_ch, n_obs * J)
    w_fine = None if w_obs is None else np.repeat(w_obs[:, :, None], J, axis=2)
    if w_fine is not None:
        w_fine = w_fine.reshape(n_ch, n_obs * J)
    res_fine, A_fine = pca_clean(fine, N_fine, weights=w_fine, los_axis=0, return_A=True)
    res_fine = res_fine.reshape(n_ch, n_obs, J)

    res_scales = np.zeros(x.shape + (J + 1,))
    res_scales[:, inside, :J] = res_fine
    res_scales[:, inside, J] = res_coarse
    residual = np.moveaxis(res_scales.sum(axis=-1), 0, los_axis)
    out = [residual]
    if return_A:
        out += [A_coarse, A_fine]
    if return_scales:
        out.append(np.moveaxis(res_scales, 0, los_axis if los_axis >= 0 else los_axis - 1))
    return out[0] if len(out) == 1 else tuple(out)
