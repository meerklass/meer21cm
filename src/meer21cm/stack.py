"""Galaxy--intensity stacking on a map cube.

:func:`stack` loops over galaxies on the native channels.
:class:`Stacking` with ``stack_estimator='correlation'`` paints those
galaxies on the sky voxels, sums the channels into uniform comoving bins,
and estimates the same cubelet with a real FFT. With a random catalogue
the worker also stores the data stack minus the random-position stack.
"""

import logging
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

from .grid import project_particle_to_regular_grid
from .io import read_catalogue_fits
from .los_model import (
    OneDimPowerModel,
    azimuthal_pixel_window,
    beam_window,
    channel_comoving_distances,
    channel_comoving_widths,
    chi_nodes_and_weights,
    discrete_beam_window,
)
from .power import PowerSpectrum
from .power_ops import step_window_attenuation
from .util import find_ch_id, radec_to_indx, redshift_to_freq

logger = logging.getLogger(__name__)


def azimuthal_pixel_window_lag(kperp, dperp, di=0, dj=0, n_phi=64, pixel_power=2):
    r"""Azimuthally averaged square-pixel window at angular lag :math:`(\Delta i, \Delta j)`.

    Thin alias of :func:`meer21cm.los_model.azimuthal_pixel_window`.
    For sky-cube stacking where the intensity map is averaged over square
    pixels and galaxies are snapped to the same pixel grid, both legs
    contribute one factor of the pixel top-hat so ``pixel_power=2``
    (:math:`|W_{\rm pix}|^2`).

    Parameters
    ----------
    kperp : array_like
        Transverse wavenumbers, in Mpc^-1.
    dperp : float
        Comoving pixel side length :math:`D_\perp`, in Mpc.
    di, dj : int, default 0
        Integer angular lag offsets in map-pixel units.
    n_phi : int, default 64
        Number of azimuthal quadrature angles on :math:`[0, 2\pi)`.
    pixel_power : int or float, default 2
        Exponent :math:`p` of the pixel top-hat window (``2`` for
        :math:`|W_{\rm pix}|^2`, ``1`` for single-leg diagnostics, ``0`` for
        no pixel smoothing).

    Returns
    -------
    window : ndarray
        Azimuthally averaged pixel window, shape ``(len(kperp),)``.
    """
    return azimuthal_pixel_window(
        kperp, dperp, di=di, dj=dj, n_phi=n_phi, pixel_power=pixel_power
    )


def beam_window_amplitude(kperp, sigma_b):
    r"""Field-level Gaussian dish-beam amplitude window :math:`\exp(-\tfrac12 k_\perp^2 \sigma_B^2)`.

    Thin alias of :func:`meer21cm.los_model.beam_window` with ``power=1``.
    Unlike the HI auto-power spectrum (which uses :math:`W_{\rm beam}^2`),
    galaxy--HI stacking carries only one factor of the beam because only the
    intensity map is convolved with the primary beam.

    Parameters
    ----------
    kperp : array_like
        Transverse wavenumbers, in Mpc^-1.
    sigma_b : float
        Comoving Gaussian beam standard deviation :math:`\sigma_B`, in Mpc.

    Returns
    -------
    window : ndarray
        Beam amplitude window with the same shape as ``kperp``.
    """
    return beam_window(kperp, sigma_b, power=1.0)


def discrete_beam_window_amplitude(
    kperp,
    dperp,
    sigma_b,
    di=0,
    dj=0,
    n_phi=64,
    pixel_power=2,
    truncate_sigma=5.0,
):
    r"""Azimuthally averaged pixel + discrete pixel-grid Gaussian beam window.

    Thin alias of :func:`meer21cm.los_model.discrete_beam_window` with
    ``beam_power=1`` for galaxy--HI stacking.

    Parameters
    ----------
    kperp : array_like
        Transverse wavenumbers, in Mpc^-1.
    dperp : float
        Comoving pixel side length :math:`D_\perp`, in Mpc.
    sigma_b : float
        Comoving Gaussian beam width :math:`\sigma_B`, in Mpc.
    di, dj : int, default 0
        Integer angular lag offsets in map-pixel units.
    n_phi : int, default 64
        Number of azimuthal quadrature angles.
    pixel_power : int or float, default 2
        Exponent of the square-pixel window.
    truncate_sigma : float, default 5.0
        Kernel truncation half-width in units of :math:`\sigma_{\rm pix}`.

    Returns
    -------
    window : ndarray
        Azimuthally averaged product of lag phase, pixel window, and discrete
        beam DFT, shape ``(len(kperp),)``.
    """
    return discrete_beam_window(
        kperp,
        dperp,
        sigma_b,
        di=di,
        dj=dj,
        n_phi=n_phi,
        pixel_power=pixel_power,
        beam_power=1.0,
        truncate_sigma=truncate_sigma,
    )


def _trapezoid_weights(x):
    """Trapezoidal quadrature weights for a strictly ascending 1D grid."""
    x_arr = np.asarray(x, dtype=float).ravel()
    if x_arr.size < 2 or np.any(np.diff(x_arr) <= 0.0):
        raise ValueError("kpar must be a strictly ascending 1D array with >= 2 points")
    w = np.empty_like(x_arr)
    w[0] = 0.5 * (x_arr[1] - x_arr[0])
    w[-1] = 0.5 * (x_arr[-1] - x_arr[-2])
    if x_arr.size > 2:
        w[1:-1] = 0.5 * (x_arr[2:] - x_arr[:-2])
    return w


def channel_cross_covariance(
    chi_ch,
    dchi_ch,
    kpar,
    p1d_kpar,
    clean_matrix=None,
    dchi_gal=None,
):
    r"""Native channel-to-channel cross-covariance matrix from :math:`P_{1D}(k_\parallel)`.

    Evaluates the exact continuous-to-channel cross-covariance

    .. math::
        C^{\rm ch}_{c_1, c_2}
        = \frac{1}{\pi}\int_0^{k_{\parallel,\max}}\mathrm{d}k_\parallel\;
          P_{1D}(k_\parallel)\,
          \operatorname{sinc}\!\left(\frac{k_\parallel\Delta\chi_{c_1}}{2}\right)
          \operatorname{sinc}\!\left(\frac{k_\parallel\Delta\chi_{{\rm gal},c_2}}{2}\right)
          \cos\!\big(k_\parallel(\chi_{c_1} - \chi_{c_2})\big)

    via a single BLAS matrix product, supporting arbitrary signed
    :math:`P_{1D}(k_\parallel)` (for nonzero angular lags) and an optional
    linear frequency-axis cleaning matrix :math:`\mathbb{M}` acting on the map
    leg (:math:`\mathbf{C}^{\rm obs} = \mathbb{M}\,\mathbf{C}^{\rm ch}`).

    Parameters
    ----------
    chi_ch : array_like
        Comoving channel centres, in Mpc, shape ``(n_ch,)``.
    dchi_ch : array_like
        Comoving channel widths on the intensity-map leg, in Mpc, shape
        ``(n_ch,)``.
    kpar : array_like
        Strictly ascending non-negative :math:`k_\parallel` quadrature nodes
        in Mpc^-1, shape ``(n_kpar,)``.
    p1d_kpar : array_like
        One-dimensional cross-power :math:`P_{1D}(k_\parallel)` evaluated at
        ``kpar``, shape ``(n_kpar,)``.
    clean_matrix : array_like, optional
        Linear cleaning operator :math:`\mathbb{M}` of shape
        ``(n_ch, n_ch)``, ordered identically to ``chi_ch``. Applied on the
        left (map leg).
    dchi_gal : array_like or float, optional
        Comoving channel widths on the galaxy leg. ``None`` defaults to
        ``dchi_ch`` (galaxies binned into native channels); ``0.0`` sets the
        galaxy radial window to unity (:math:`W_\parallel^g = 1`).

    Returns
    -------
    cov : ndarray
        Channel cross-covariance matrix of shape ``(n_ch, n_ch)``.
    """
    chi = np.asarray(chi_ch, dtype=float).ravel()
    dchi_l = np.asarray(dchi_ch, dtype=float).ravel()
    if chi.size == 0 or dchi_l.shape != chi.shape:
        raise ValueError(
            "chi_ch and dchi_ch must be non-empty 1D arrays of equal length"
        )
    if dchi_gal is None:
        dchi_g = dchi_l
    elif np.ndim(dchi_gal) == 0:
        dchi_g = np.full(chi.size, float(dchi_gal), dtype=float)
    else:
        dchi_g = np.asarray(dchi_gal, dtype=float).ravel()
        if dchi_g.shape != chi.shape:
            raise ValueError("dchi_gal must match chi_ch length")

    kpar_arr = np.asarray(kpar, dtype=float).ravel()
    p1d_arr = np.asarray(p1d_kpar, dtype=float).ravel()
    if p1d_arr.shape != kpar_arr.shape:
        raise ValueError("p1d_kpar must have the same shape as kpar")
    weights = _trapezoid_weights(kpar_arr) / np.pi

    # Shift chi by its midpoint to keep complex phases well-conditioned.
    chi_rel = chi - 0.5 * (chi.min() + chi.max())
    phase = np.exp(1j * chi_rel[:, None] * kpar_arr[None, :])
    a_l = step_window_attenuation(kpar_arr[None, :], dchi_l[:, None], p=1) * phase
    if np.array_equal(dchi_g, dchi_l):
        a_g = a_l
    else:
        a_g = step_window_attenuation(kpar_arr[None, :], dchi_g[:, None], p=1) * phase

    cov = np.real((a_l * (weights * p1d_arr)[None, :]) @ a_g.conj().T)
    if clean_matrix is not None:
        m_mat = np.asarray(clean_matrix, dtype=float)
        if m_mat.shape != (chi.size, chi.size):
            raise ValueError(
                f"clean_matrix must have shape {(chi.size, chi.size)}, got {m_mat.shape}"
            )
        cov = m_mat @ cov
    return cov


def contract_channel_covariance(
    cov_ch,
    pair_weight,
    bin_which=None,
    n_bin=None,
    weighting="conventional",
    symmetrize=False,
    q0=None,
):
    r"""Contract a channel covariance matrix with pair weights and LOS binning.

    Computes the exact expected stacked 1D lag profile for one angular lag
    :math:`(\Delta i, \Delta j)`:

    .. math::
        \mathbf{N}^{\rm bin} = S\,\big(W \odot \mathbf{C}^{\rm obs}\big)\,S^T,
        \qquad
        \mathbf{D}^{\rm bin} = S\,W\,S^T,

    where :math:`S_{b,c} \in \{0, 1\}` maps native channels into uniform
    comoving bins (or :math:`S = \mathbb{I}` when ``bin_which is None``), and
    sums each diagonal :math:`\Delta b = -(N_b - 1), \dots, +(N_b - 1)` of
    :math:`\mathbf{N}^{\rm bin}` and :math:`\mathbf{D}^{\rm bin}` identically
    to :func:`correlate_cubelet` and :func:`stack_cubelet`.

    Parameters
    ----------
    cov_ch : array_like
        Channel-to-channel cross-covariance :math:`\mathbf{C}^{\rm obs}`,
        shape ``(n_ch, n_ch)``.
    pair_weight : array_like
        Exact pair-weight matrix :math:`W_{c_1, c_2} = \sum_{i,j} w_L(i+\Delta i,
        j+\Delta j, c_1)\,\bar{N}_g(i, j, c_2)`, shape ``(n_ch, n_ch)``.
    bin_which : array_like of int, optional
        Bin index ``0 .. n_bin - 1`` for each channel. ``None`` keeps native
        channels without binning.
    n_bin : int, optional
        Total number of LOS bins (required when empty trailing bins exist;
        defaults to ``max(bin_which) + 1``).
    weighting : {'conventional', 'quadratic'}, default 'conventional'
        Normalisation scheme matching the estimator.
    symmetrize : bool, default False
        Fold :math:`\pm \Delta b` before normalising, matching the estimator.
    q0 : float, optional
        Unsymmetrised global zero-lag scalar normalisation
        :math:`Q_0 = \sum_b D^{\rm bin,(0,0)}_{b,b}` used when
        ``weighting='quadratic'`` at off-centre angular lags
        :math:`(\Delta i, \Delta j) \neq (0, 0)`. When ``symmetrize=True``,
        ``q0`` is doubled automatically to match :func:`stack_cubelet` and
        :func:`data_and_fkp_cubelets`. Defaults to the zero-LOS-lag element of
        ``pair_weight``.

    Returns
    -------
    profile : ndarray
        Model stacked spectrum of shape ``(2 * n_los - 1,)``.
    denom : ndarray
        Model pair-count denominator of shape ``(2 * n_los - 1,)``.
    """
    if weighting not in ("conventional", "quadratic"):
        raise ValueError(
            f"weighting must be 'conventional' or 'quadratic', got {weighting!r}"
        )
    cov = np.asarray(cov_ch, dtype=float)
    wt = np.asarray(pair_weight, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or wt.shape != cov.shape:
        raise ValueError(
            "cov_ch and pair_weight must be square 2D arrays of equal shape"
        )
    n_ch = cov.shape[0]
    if bin_which is not None:
        which = np.asarray(bin_which, dtype=int).ravel()
        if which.size != n_ch:
            raise ValueError("bin_which length must match cov_ch size")
        nb = int(which.max()) + 1 if n_bin is None else int(n_bin)
        s_mat = np.zeros((nb, n_ch), dtype=float)
        s_mat[which, np.arange(n_ch)] = 1.0
        num_bin = s_mat @ (wt * cov) @ s_mat.T
        den_bin = s_mat @ wt @ s_mat.T
    else:
        nb = n_ch
        num_bin = wt * cov
        den_bin = wt

    lags = np.arange(-(nb - 1), nb, dtype=int)
    num_diag = np.array(
        [np.trace(num_bin, offset=-int(db)) for db in lags], dtype=float
    )
    den_diag = np.array(
        [np.trace(den_bin, offset=-int(db)) for db in lags], dtype=float
    )
    if symmetrize:
        num_diag = num_diag + num_diag[::-1]
        den_diag = den_diag + den_diag[::-1]

    if weighting == "conventional":
        out = np.zeros_like(num_diag)
        hit = _pair_hit(den_diag)
        out[hit] = num_diag[hit] / den_diag[hit]
    else:
        if q0 is None:
            q0_eff = float(den_diag[nb - 1])
        else:
            q0_eff = float(q0) * (2.0 if symmetrize else 1.0)
        out = num_diag / q0_eff if q0_eff != 0.0 else np.zeros_like(num_diag)
    return out, den_diag


def _validate_catalogue_arrays(name, ra, dec, z, weights=None):
    """Validate that a catalogue's coordinate and optional weight arrays agree."""
    coords = (ra, dec, z)
    present = [arr is not None for arr in coords]
    if not any(present):
        if weights is not None:
            raise ValueError(
                f"weights_{name} cannot be set without ra_{name}, dec_{name}, and z_{name}"
            )
        return None
    if not all(present):
        raise ValueError(
            f"ra_{name}, dec_{name}, and z_{name} must all be provided together"
        )
    ra_arr = np.asarray(ra, dtype=float)
    dec_arr = np.asarray(dec, dtype=float)
    z_arr = np.asarray(z, dtype=float)
    if ra_arr.ndim != 1 or dec_arr.shape != ra_arr.shape or z_arr.shape != ra_arr.shape:
        raise ValueError(
            f"ra_{name}, dec_{name}, and z_{name} must be 1D arrays of equal length"
        )
    w_arr = None
    if weights is not None:
        w_arr = np.asarray(weights, dtype=float)
        if w_arr.shape != ra_arr.shape:
            raise ValueError(
                f"weights_{name} must have the same length as ra_{name} "
                f"({ra_arr.size}), got {w_arr.shape}"
            )
    return ra_arr, dec_arr, z_arr, w_arr


def stack_cubelet(
    map_in,
    w_map_in,
    indx_0_g,
    indx_1_g,
    indx_z_g,
    weights_gal=None,
    weighting="conventional",
    stack_angular_num_nearby_pix=10,
    symmetrize=False,
):
    r"""
    Workhorse routine that builds the 3D stacked cubelet from an intensity map and a set of
    source pixel positions. This function does not depend on any :class:`meer21cm.Specification`
    object; all inputs are plain arrays.

    Following the stacking formalism, the 3D cubelet around the :math:`i`-th source is

    .. math::
        \bm{I}_{\bm{s}\bm{x}_i} = \sum_{\bm{x}} \mathcal{S}^{\bm{s}}_{\bm{x}_i\bm{x}}\, \bm{L}_{\bm{x}},
        \qquad
        \mathcal{S}^{\bm{s}}_{\bm{x}_i\bm{x}} = \delta^{\rm K}_{(\bm{x}-\bm{x}_i)\bm{s}}\, w_{\bm{x}},

    i.e. the cubelet voxel at separation :math:`\bm{s}` is the map value :math:`L_{\bm{x}_i+\bm{s}}`
    weighted by the map weight :math:`w_{\bm{x}_i+\bm{s}}`. The 3D stacked signal is the
    galaxy-weighted sum over all sources, normalised by a factor :math:`Q_0`,

    .. math::
        \bm{I}_{\bm{s}} = \frac{1}{Q_0} \sum_i \bm{w}^{\rm gal}_i\, \bm{I}_{\bm{s}\bm{x}_i}
        = \frac{1}{Q_0}\sum_i \bm{w}^{\rm gal}_i\, w_{\bm{x}_i+\bm{s}}\, L_{\bm{x}_i+\bm{s}}.

    The galaxy weight :math:`\bm{w}^{\rm gal}_i` is a single constant per source and is **not** a
    function of the separation :math:`\bm{s}`; for example it can be a binary weight that selects a
    particular subsample of the galaxy catalogue.

    In both cases the numerator is the same galaxy-weighted sum
    :math:`\sum_i \bm{w}^{\rm gal}_i\, w_{\bm{x}_i+\bm{s}}\, L_{\bm{x}_i+\bm{s}}`; the two
    normalisation schemes differ only in :math:`Q_0`:

    - ``"conventional"``: :math:`Q_0` is the per-separation sum of the effective weights, using the
      map pixel weight evaluated at each separation coordinate :math:`\bm{x}_i+\bm{s}`, so that the
      cubelet is the weighted **average** over the contributing sources,

      .. math::
          \bm{I}_{\bm{s}} = \frac{\sum_i \bm{w}^{\rm gal}_i\, w_{\bm{x}_i+\bm{s}}\, L_{\bm{x}_i+\bm{s}}}
               {\sum_i \bm{w}^{\rm gal}_i\, w_{\bm{x}_i+\bm{s}}}.

    - ``"quadratic"``: :math:`Q_0` is the single scalar normalisation of the quadratic estimator,
      i.e. the sum over sources of the galaxy weight times the map pixel weight evaluated at the
      **centre pixel** :math:`\bm{x}_i` (Eq. 30 of the formalism),

      .. math::
          Q_0 = \sum_i \bm{w}^{\rm gal}_i\, w_{\bm{x}_i}.

      Unlike the conventional scheme, :math:`Q_0` is independent of the separation :math:`\bm{s}`
      because the map pixel weight is taken at the source centre rather than at each separation.

    The cubelet extends over the entire frequency range of the map so the spectral separation is
    sampled at [:math:`-N_{\rm ch}\delta\nu`,...,0,..., :math:`N_{\rm ch}\delta\nu`].
    Positive spectral lag index corresponds to ``ch_map - ch_gal > 0`` (ascending frequency
    :math:`+\Delta\nu`, i.e. descending comoving distance :math:`-\Delta\chi` for standard
    ascending-frequency cubes). By contrast, :func:`rebin_sightlines` and the ``'correlation'``
    estimator sort channels in ascending :math:`\chi`, so positive lag index corresponds to
    :math:`+\Delta\chi`.
    The angular sampling of the cubelet corresponds to the map pixels, and the size of the angular
    plane is set by ``stack_angular_num_nearby_pix``. Note that ``stack_angular_num_nearby_pix`` is
    the number of pixels **each side of the centre** so the size of the angular plane is
    ``(2 * stack_angular_num_nearby_pix + 1)**2``. ``0`` keeps only the centre pixel.

    If ``symmetrize``, a mirroring of the individual cubelets is performed along :math:`\Delta\nu=0`.
    This corresponds to the 180deg rotation along the spectral axis described in Sinigaglia et al.
    (2022) [1] and is the only symmetry that single-dish IM stacking is sensitive to.

    Parameters
    ----------
        map_in: array.
            The intensity map data cube, with shape ``(n_ra, n_dec, n_ch)``.
        w_map_in: array.
            The per-voxel map weights :math:`w_{\bm{x}}`, with the same shape as ``map_in``.
        indx_0_g: array.
            The first angular pixel index of each source centre.
        indx_1_g: array.
            The second angular pixel index of each source centre.
        indx_z_g: array.
            The frequency channel index of each source centre.
        weights_gal: array, optional, default None.
            The per-source weights :math:`\bm{w}^{\rm gal}_i`. If None, uniform weights are used.
        weighting: str, optional, default "conventional".
            The normalisation scheme, either ``"conventional"`` (per-separation weighted average,
            map pixel weight at each separation) or ``"quadratic"`` (scalar quadratic-estimator
            normalisation :math:`\sum_i w^{\rm gal}_i w_{\bm{x}_i}`, map pixel weight at the centre pixel).
        stack_angular_num_nearby_pix: optional, default 10.
            The number of map pixels sampled on each side relative to the source centre.
        symmetrize: optional, default False.
            Whether to symmetrize the stacking.

    Returns
    -------
        stack_3D_map: array.
            The normalised cubelet for the stacking.
        stack_3D_weight: array.
            The accumulated per-voxel effective weights :math:`\sum_i \bm{w}^{\rm gal}_i w_{\bm{x}_i+\bm{s}}`.
            This is the per-voxel normalisation used by the ``"conventional"`` scheme.

    References
    ----------
    .. [1] Sinigaglia, F. et al., "Optimizing spectral stacking for 21-cm observations of galaxies: accuracy assessment and symmetrized stacking", https://ui.adsabs.harvard.edu/abs/2022MNRAS.514.4205S.

    """
    if weighting not in ("conventional", "quadratic"):
        raise ValueError(
            f"weighting must be 'conventional' or 'quadratic', got '{weighting}'"
        )
    pad = int(stack_angular_num_nearby_pix)
    if pad < 0:
        raise ValueError(
            f"stack_angular_num_nearby_pix must be non-negative, got {pad}"
        )
    map_in = np.asarray(map_in)
    w_map_in = np.asarray(w_map_in)
    num_ch = map_in.shape[-1]
    # copy so that the in-place padding shift does not mutate the caller's arrays
    indx_0_g = np.array(indx_0_g)
    indx_1_g = np.array(indx_1_g)
    indx_z_g = np.asarray(indx_z_g)
    num_g = indx_0_g.size
    if weights_gal is None:
        weights_gal = np.ones(num_g)
    weights_gal = np.asarray(weights_gal, dtype=float)
    # check if some galaxies are outside the range
    sel = (
        (indx_0_g < 0)
        | (indx_0_g >= map_in.shape[0])
        | (indx_1_g < 0)
        | (indx_1_g >= map_in.shape[1])
        | (indx_z_g < 0)
        | (indx_z_g >= num_ch)
    )
    if sel.sum() > 0:
        raise ValueError("some galaxies are outside survey area or frequency range")
    # zero pad the sky map and the weights
    padded_shape = (
        map_in.shape[0] + 2 * pad,
        map_in.shape[1] + 2 * pad,
        num_ch,
    )
    map_stack = np.zeros(padded_shape)
    w_stack = np.zeros(padded_shape)
    if pad > 0:
        map_stack[pad:-pad, pad:-pad] = map_in
        w_stack[pad:-pad, pad:-pad] = w_map_in
    else:
        map_stack[...] = map_in
        w_stack[...] = w_map_in
    # indices are shifted by zero-padding
    indx_0_g += pad
    indx_1_g += pad

    num_angular_bin = 2 * pad + 1
    # take a nearby area around each source
    indx_xx, indx_yy = np.meshgrid(
        *((np.arange(-pad, pad + 1),) * 2),
        indexing="ij",
    )
    indx_0_sample = indx_0_g[None, None, :] + indx_xx[:, :, None]
    indx_1_sample = indx_1_g[None, None, :] + indx_yy[:, :, None]
    # the results to be stacked
    stack_3D_map = np.zeros((num_angular_bin, num_angular_bin, 2 * num_ch - 1))
    stack_3D_weight = np.zeros((num_angular_bin, num_angular_bin, 2 * num_ch - 1))
    # loop over frequency channel should be a good balance between speed and memory
    for ch_id in range(num_ch):
        # the centre image around each source in channel i
        map_source_i = map_stack[
            indx_0_sample.ravel(), indx_1_sample.ravel(), ch_id
        ].reshape(indx_0_sample.shape)
        weight_source_i = w_stack[
            indx_0_sample.ravel(), indx_1_sample.ravel(), ch_id
        ].reshape(indx_0_sample.shape)
        # fold the per-source galaxy weight into the effective voxel weight so that both the
        # numerator and the normalisation are weighted by w_gal_i (see Eq. 6 of the formalism)
        weight_source_i = weight_source_i * weights_gal[None, None, :]
        # each source is added to a different channel in the final cube
        # this is wrong because repeating indices are only added in the last occurance
        # stack_3D_map[:, :, ch_id - indx_z_g + num_ch - 1] += (
        #    map_source_i * weight_source_i
        # )
        # stack_3D_weight[:, :, ch_id - indx_z_g + num_ch - 1] += weight_source_i
        add_id = ch_id - indx_z_g + num_ch - 1
        if symmetrize:
            add_id = np.append(add_id, 2 * num_ch - 2 - add_id)
            weight_source_i = np.concatenate(
                [weight_source_i, weight_source_i], axis=-1
            )
            map_source_i = np.concatenate([map_source_i, map_source_i], axis=-1)
        # some new np black magic
        np.add.at(stack_3D_weight, (slice(None), slice(None), add_id), weight_source_i)
        np.add.at(
            stack_3D_map,
            (slice(None), slice(None), add_id),
            weight_source_i * map_source_i,
        )

    # normalise
    if weighting == "conventional":
        # per-separation weighted average: divide by the accumulated weights at each separation,
        # i.e. the map pixel weight evaluated at each separation coordinate x_i + s
        stack_3D_map[stack_3D_weight > 0] = (
            stack_3D_map[stack_3D_weight > 0] / stack_3D_weight[stack_3D_weight > 0]
        )
    else:
        # quadratic estimator normalisation Q0 = sum_i w_gal_i w_L(x_i) (Eq. 30 of the formalism):
        # the sum over sources of the galaxy weight times the map pixel weight at the centre pixel
        # (indx_0_g, indx_1_g are the zero-padding-shifted centre indices of w_stack)
        centre_w = w_stack[indx_0_g, indx_1_g, indx_z_g]
        q0 = np.sum(weights_gal * centre_w)
        if symmetrize:
            # each cubelet is mirrored and added twice, so the numerator is doubled accordingly
            q0 = 2 * q0
        if q0 != 0:
            stack_3D_map = stack_3D_map / q0
    return stack_3D_map, stack_3D_weight


def stack(
    sp,
    weights_gal=None,
    weighting="conventional",
    stack_angular_num_nearby_pix=10,
    symmetrize=False,
):
    r"""
    Calculate a stacked 3D cubelet using the intensity maps and source positions stored in ``sp``.

    This is a thin wrapper around :func:`stack_cubelet`. It extracts the intensity map
    (``sp.data``), the map weights (``sp.w_HI``) and the source positions
    (``sp.ra_gal``, ``sp.dec_gal``, ``sp.z_gal``) from the input object, converts the source sky
    positions into map pixel/channel indices, and delegates the actual stacking to
    :func:`stack_cubelet`.

    Parameters
    ----------
        sp: :class:`meer21cm.Specification` object.
            The data used for stacking.
        weights_gal: array, optional, default None.
            The per-source weights :math:`\bm{w}^{\rm gal}_i`. If None, uniform weights are used.
        weighting: str, optional, default "conventional".
            The normalisation scheme, either ``"conventional"`` (per-separation weighted average,
            map pixel weight at each separation) or ``"quadratic"`` (scalar quadratic-estimator
            normalisation :math:`\sum_i w^{\rm gal}_i w_{\bm{x}_i}`, map pixel weight at the centre pixel).
        stack_angular_num_nearby_pix: optional, default 10.
            The number of map pixels sampled on each side relative to the source centre.
        symmetrize: optional, default False.
            Whether to symmetrize the stacking.

    Returns
    -------
        stack_3D_map: array.
            The normalised cubelet for the stacking.
        stack_3D_weight: array.
            The accumulated per-voxel effective weights in the cubelet.

    See Also
    --------
    stack_cubelet : The underlying ``sp``-independent stacking routine.
    """
    map_in = sp.data.copy()
    w_map_in = sp.w_HI.copy()
    ra_g_in = sp.ra_gal.copy()
    dec_g_in = sp.dec_gal.copy()
    z_g_in = sp.z_gal.copy()
    wproj = sp.wproj
    # retrive the centre pixel positions
    indx_0_g, indx_1_g = radec_to_indx(ra_g_in, dec_g_in, wproj)
    indx_z_g = find_ch_id(redshift_to_freq(z_g_in), sp.nu)
    return stack_cubelet(
        map_in,
        w_map_in,
        indx_0_g,
        indx_1_g,
        indx_z_g,
        weights_gal=weights_gal,
        weighting=weighting,
        stack_angular_num_nearby_pix=stack_angular_num_nearby_pix,
        symmetrize=symmetrize,
    )


def sum_3d_stack(stack_3D_map, vel_ch_avg=5, ang_sum_dist=3.0):
    """
    Collapse a stacked cubelet into stacked image and stacked spectrum.

    Note that for stacked image, `vel_ch_avg` is the number of channels that go into
    the summation on each side of the centre channel so that the total number of
    channels that are summed is `(2 * vel_ch_avg + 1)`.

    For stacked spectrum, the angular pixels that go into the summation are determined
    by the distance to the center pixel. Note that the distance is in cell length not physical angular unit.


    Parameters
    ----------
        stack_3D_map: array.
            The stacked cubelet.
        vel_ch_avg: optional, default 5.
            How many channels on each side of the center to sum into stacked image.
        ang_sum_dist: optional, default 3.0.
            The distance within which the angular pixels are summed to stacked spectrum

    Returns
    -------
        angular_stack_map: array.
            The stacked image.
        spectral_stack_map: array.
            The stacked spectrum.
    """
    mid_point = stack_3D_map.shape[-1] // 2
    ang_centre = stack_3D_map.shape[0] // 2
    xx, yy = np.meshgrid(
        np.linspace(-ang_centre, ang_centre, stack_3D_map.shape[0]),
        np.linspace(-ang_centre, ang_centre, stack_3D_map.shape[0]),
    )
    pix_dist = np.sqrt(xx**2 + yy**2)
    pix_sel = pix_dist <= (ang_sum_dist)
    angular_stack_map = stack_3D_map[
        :, :, mid_point - vel_ch_avg : mid_point + vel_ch_avg + 1
    ].sum(axis=-1)
    spectral_stack_map = stack_3D_map[pix_sel].sum(axis=0)
    return angular_stack_map, spectral_stack_map


def sky_mesh_positions(ra, dec, z, wproj, nu, shape, centre_pixels=True):
    """
    Mesh coordinates of galaxies that land inside a WCS map cube.

    ``radec_to_indx(..., to_int=False)`` supplies the continuous WCS
    coordinate. Astropy pixel ``i`` is centred at ``i``. Mesh cell ``i``
    covers ``[i, i + 1)``, so ``centre_pixels`` adds half a cell and the
    pixel centre falls on the mesh-cell centre. The channel coordinate is
    the centre of the channel from :func:`meer21cm.util.find_ch_id`.
    An object is kept when that shifted coordinate lies in ``[0, n)`` on
    each angular axis and the channel index lies in the cube.

    Parameters
    ----------
        ra, dec, z: arrays.
            Galaxy coordinates. ``z`` is the redshift.
        wproj: :class:`astropy.wcs.WCS`
            The map WCS.
        nu: array.
            Channel frequencies in Hz, monotonically increasing.
        shape: tuple of int.
            ``(n_ra, n_dec, n_ch)`` of the map cube.
        centre_pixels: bool, default True.
            Add half a cell to the angular WCS coordinate.

    Returns
    -------
        positions: array, shape ``(n_inside, 3)``.
            Mesh coordinates ``(i + shift, j + shift, channel + 0.5)``.
        inside: bool array.
            Mask of the input galaxies that were kept.
    """
    channel = find_ch_id(redshift_to_freq(np.asarray(z, dtype=float)), nu)
    ix, iy = radec_to_indx(np.asarray(ra), np.asarray(dec), wproj, to_int=False)
    shift = 0.5 if centre_pixels else 0.0
    x = np.asarray(ix, dtype=float) + shift
    y = np.asarray(iy, dtype=float) + shift
    z = np.asarray(channel, dtype=float) + 0.5
    inside = (
        (channel >= 0)
        & (channel < shape[2])
        & (x >= 0)
        & (x < shape[0])
        & (y >= 0)
        & (y < shape[1])
    )
    positions = np.column_stack([x[inside], y[inside], z[inside]])
    return positions, np.asarray(inside, dtype=bool)


def paint_sky_counts(positions, shape, mass=None):
    """
    Nearest-grid deposit of galaxy weights onto the map mesh.

    ``average=False``, so two galaxies in one voxel add. Mesh cell ``i``
    covers ``[i, i + 1)``.

    Parameters
    ----------
        positions: array, shape ``(n, 3)``.
            Mesh coordinates.
        shape: tuple of int.
            Mesh shape.
        mass: array, optional.
            Per-galaxy weight. Uniform weights if omitted.

    Returns
    -------
        counts: array.
            Painted weights, shape ``shape``.
    """
    positions = np.asarray(positions, dtype=float)
    painted = np.zeros(shape, dtype=float)
    if positions.size == 0:
        return painted
    if mass is None:
        mass = np.ones(positions.shape[0], dtype=float)
    mesh, _weights, _counts = project_particle_to_regular_grid(
        positions,
        np.asarray(shape, dtype=float),
        np.asarray(shape, dtype=int),
        grid_scheme="nnb",
        particle_mass=np.asarray(mass, dtype=float),
        average=False,
        compensate=False,
    )
    painted[...] = np.asarray(mesh, dtype=float)
    return painted


def comoving_bin_indices(chi, dchi):
    """Build uniform comoving line-of-sight bin edges and channel assignments.

    Channels are sorted in ascending comoving distance ``chi``, so positive
    lags after rebinning correspond to :math:`+\\Delta\\chi` (opposite to
    ascending frequency :math:`+\\Delta\\nu`).

    Parameters
    ----------
    chi : array_like
        Comoving distance of each channel, in Mpc.
    dchi : float
        Uniform comoving bin width, in Mpc. Must be positive.

    Returns
    -------
    order : ndarray of int
        Index array sorting ``chi`` in ascending order.
    edges : ndarray of float
        Bin edges in Mpc, shape ``(n_bin + 1,)``.
    centres : ndarray of float
        Bin centres in Mpc, shape ``(n_bin,)``.
    which : ndarray of int
        Bin index ``0 .. n_bin - 1`` assigned to each sorted channel
        ``chi[order]``.
    """
    dchi_f = float(dchi)
    if not np.isfinite(dchi_f) or dchi_f <= 0.0:
        raise ValueError(f"dchi must be a finite positive float, got {dchi!r}")
    chi_arr = np.asarray(chi, dtype=float).ravel()
    if chi_arr.size == 0:
        raise ValueError("chi must be non-empty")
    order = np.argsort(chi_arr)
    chi_s = chi_arr[order]
    edges = np.arange(chi_s[0], chi_s[-1] + 0.5 * dchi_f, dchi_f)
    if edges.size == 0 or edges[-1] <= chi_s[-1]:
        edges = np.append(edges, (chi_s[0] if edges.size == 0 else edges[-1]) + dchi_f)
    which = np.clip(np.digitize(chi_s, edges) - 1, 0, edges.size - 2)
    centres = 0.5 * (edges[:-1] + edges[1:])
    return order, edges, centres, which


def rebin_cube(cube, chi, dchi):
    """Sum a single map cube along its last axis into uniform comoving bins.

    Parameters
    ----------
    cube : array_like
        Array with channel axis last, shape ``(..., n_ch)``.
    chi : array_like
        Comoving distance of each channel, in Mpc.
    dchi : float
        Uniform comoving bin width, in Mpc.

    Returns
    -------
    rebinned : ndarray
        Summed array of shape ``(..., n_bin)``, ordered by ascending
        :math:`\\chi`.
    centres : ndarray
        Comoving bin centres, in Mpc.
    """
    order, edges, centres, which = comoving_bin_indices(chi, dchi)
    nbin = edges.size - 1
    data = np.asarray(cube, dtype=float)[..., order]
    out = np.zeros(data.shape[:-1] + (nbin,), dtype=float)
    for bin_id in range(nbin):
        sel = np.flatnonzero(which == bin_id)
        if sel.size > 0:
            out[..., bin_id] = data[..., sel].sum(axis=-1)
    return out, centres


def rebin_sightlines(weighted, weight, counts, chi, dchi):
    """
    Sum map sightlines into uniform comoving bins.

    ``weighted`` is :math:`w_L L` and ``weight`` is :math:`w_L`. Both are
    summed over the native channels that fall in each bin. Galaxy counts
    are summed, so the number of galaxies is unchanged. Channels are sorted
    in ascending :math:`\\chi`, so positive lags correspond to
    :math:`+\\Delta\\chi`.

    Parameters
    ----------
        weighted, weight, counts: arrays.
            Map cubes, channel axis last. ``chi`` is one comoving distance
            per channel, in Mpc.
        chi: array.
            Comoving distance of each channel, in Mpc.
        dchi: float.
            Bin width in Mpc.

    Returns
    -------
        weighted, weight, counts: arrays.
            Rebinned cubes.
        centres: array.
            Comoving centre of each bin, in Mpc.
    """
    order, edges, centres, which = comoving_bin_indices(chi, dchi)
    nbin = edges.size - 1
    weighted = np.asarray(weighted, dtype=float)[..., order]
    weight = np.asarray(weight, dtype=float)[..., order]
    counts = np.asarray(counts, dtype=float)[..., order]
    out_w = np.zeros(weighted.shape[:2] + (nbin,), dtype=float)
    out_wt = np.zeros_like(out_w)
    out_n = np.zeros_like(out_w)
    for bin_id in range(nbin):
        sel = np.flatnonzero(which == bin_id)
        if sel.size == 0:
            continue
        out_w[..., bin_id] = weighted[..., sel].sum(axis=-1)
        out_wt[..., bin_id] = weight[..., sel].sum(axis=-1)
        out_n[..., bin_id] = counts[..., sel].sum(axis=-1)
    return out_w, out_wt, out_n, centres


def aperture_weights_2d(
    n_nearby,
    aperture="central",
    radius=None,
    weights_2d=None,
    normalize=True,
):
    """Build a 2D angular projection kernel for collapsing a 3D cubelet to 1D.

    Parameters
    ----------
    n_nearby : int
        Angular half-width of the cubelet in pixels (kernel side length is
        ``2 * n_nearby + 1``).
    aperture : {'central', 'tophat', 'gaussian'}, default 'central'
        Preset aperture shape used when ``weights_2d`` is ``None``:

        - ``'central'``: central pixel ``(0, 0)`` only.
        - ``'tophat'``: uniform weight on pixels with
          :math:`\\sqrt{\\Delta i^2 + \\Delta j^2} \\le \\texttt{radius}`.
        - ``'gaussian'``: :math:`\\exp(-(\\Delta i^2 + \\Delta j^2) / (2\\,\\texttt{radius}^2))`.
    radius : float, optional
        Aperture radius (top-hat) or Gaussian :math:`\\sigma` in **map-pixel**
        units. Defaults to ``1.0`` for ``'tophat'`` and ``'gaussian'``.
    weights_2d : array_like, optional
        Custom 2D weight matrix of shape ``(2 * n_nearby + 1, 2 * n_nearby + 1)``.
        Overrides ``aperture`` and ``radius`` when provided.
    normalize : bool, default True
        If True, divide the weights by their sum so the projection is a
        weighted average rather than a raw sum.

    Returns
    -------
    kernel : ndarray
        2D float array of shape ``(2 * n_nearby + 1, 2 * n_nearby + 1)``.
    """
    n_pix = int(n_nearby)
    if n_pix < 0:
        raise ValueError(f"n_nearby must be non-negative, got {n_pix}")
    side = 2 * n_pix + 1
    if weights_2d is not None:
        kernel = np.asarray(weights_2d, dtype=float).copy()
        if kernel.shape != (side, side):
            raise ValueError(
                f"weights_2d must have shape {(side, side)}, got {kernel.shape}"
            )
    else:
        lags = np.arange(-n_pix, n_pix + 1, dtype=float)
        di, dj = np.meshgrid(lags, lags, indexing="ij")
        r2 = di**2 + dj**2
        mode = str(aperture).lower()
        if mode == "central":
            kernel = np.zeros((side, side), dtype=float)
            kernel[n_pix, n_pix] = 1.0
        elif mode == "tophat":
            rad = 1.0 if radius is None else float(radius)
            if not np.isfinite(rad) or rad < 0.0:
                raise ValueError(f"tophat radius must be finite and >= 0, got {rad!r}")
            kernel = (r2 <= rad**2 + 1e-12).astype(float)
        elif mode == "gaussian":
            rad = 1.0 if radius is None else float(radius)
            if not np.isfinite(rad) or rad <= 0.0:
                raise ValueError(f"gaussian radius must be finite and > 0, got {rad!r}")
            kernel = np.exp(-0.5 * r2 / (rad**2))
        else:
            raise ValueError(
                f"aperture must be 'central', 'tophat', or 'gaussian', got {aperture!r}"
            )
    if normalize:
        total = float(np.sum(kernel))
        if total == 0.0:
            raise ValueError("aperture weights sum to zero; cannot normalize")
        kernel = kernel / total
    return kernel


def linear_correlation(field, weight, n_nearby):
    """
    ``sum_x field[x + s] weight[x]`` on the cubelet lags.

    Zero-padding makes the real FFT a linear correlation. Negative lags
    sit at the wrapped end of the padded axis.

    Parameters
    ----------
        field, weight: arrays.
            Same shape. ``weight`` is the galaxy field for a stack.
        n_nearby: int.
            Angular lags kept on each side of zero. The line-of-sight axis
            is kept in full, from ``-(N-1)`` to ``N-1``.

    Returns
    -------
        correlated: array.
            Shape ``(2 n_nearby + 1, 2 n_nearby + 1, 2 N_los - 1)``.
    """
    nx, ny, nz = field.shape
    padded = (nx + n_nearby, ny + n_nearby, 2 * nz - 1)
    field_pad = np.zeros(padded, dtype=float)
    weight_pad = np.zeros(padded, dtype=float)
    field_pad[:nx, :ny, :nz] = field
    weight_pad[:nx, :ny, :nz] = weight
    axes = (0, 1, 2)
    correlated = np.fft.irfftn(
        np.fft.rfftn(field_pad, axes=axes)
        * np.conj(np.fft.rfftn(weight_pad, axes=axes)),
        s=padded,
        axes=axes,
    )
    di = np.arange(-n_nearby, n_nearby + 1)
    dj = np.arange(-n_nearby, n_nearby + 1)
    dc = np.arange(-(nz - 1), nz)
    return correlated[
        di[:, None, None] % padded[0],
        dj[None, :, None] % padded[1],
        dc[None, None, :] % padded[2],
    ]


def _divide_cubelet(numerator, denominator):
    """Ratio of two cubelets. Empty lags stay zero."""
    stacked = np.zeros_like(numerator)
    scale = np.max(np.abs(denominator))
    hit = np.abs(denominator) > (0.0 if scale == 0.0 else 1e-8 * scale)
    stacked[hit] = numerator[hit] / denominator[hit]
    return stacked


def correlate_cubelet(weighted_map, weight, galaxy, n_nearby):
    """
    Conventional cubelet from a painted galaxy field.

    The numerator correlates :math:`w_L L` with the galaxy field. The
    denominator correlates :math:`w_L` with the galaxy field.

    Parameters
    ----------
        weighted_map: array.
            :math:`w_L L` on the map mesh.
        weight: array.
            :math:`w_L` on the map mesh.
        galaxy: array.
            Painted galaxy weights.
        n_nearby: int.
            Angular lags on each side of the centre pixel.

    Returns
    -------
        cubelet, cubelet_weight: arrays.
            Normalised stack and the denominator.
    """
    numerator = linear_correlation(weighted_map, galaxy, n_nearby)
    denominator = linear_correlation(weight, galaxy, n_nearby)
    return _divide_cubelet(numerator, denominator), denominator


def _pair_hit(denominator):
    """Lags whose pair-count denominator is above the FFT roundoff floor."""
    scale = np.max(np.abs(denominator))
    if scale == 0.0:
        return np.zeros(denominator.shape, dtype=bool)
    return np.abs(denominator) > 1e-8 * scale


def data_and_fkp_cubelets(
    weighted_map,
    weight,
    data_counts,
    n_nearby,
    random_counts=None,
    weighting="conventional",
    symmetrize=False,
    return_details=False,
):
    r"""
    Number-count stack, and that stack minus the random-position stack.

    For ``weighting='conventional'``:

    - The data cubelet divides :math:`(w_L L) \star D` by the galaxy pair
      counts :math:`w_L \star D`.
    - The random-position stack divides :math:`(w_L L) \star R` by the random
      pair counts :math:`w_L \star R`.
    - ``fkp`` is their difference :math:`I_D - I_R`, which cancels any constant
      map offset identically at every lag with both data and random pairs. A
      lag missing either pair count stays zero.

    For ``weighting='quadratic'``:

    - The data cubelet divides :math:`(w_L L) \star D` by the scalar
      zero-lag normalisation :math:`Q_{0,D} = (w_L \star D)(\bm{0})`.
    - With randoms, ``fkp`` uses the FKP quadratic estimator
      :math:`[(w_L L) \star (D - \alpha R)] / Q_{0,D}` with
      :math:`\alpha = \sum D / \sum R`.

    If ``symmetrize=True``, positive and negative line-of-sight lags are folded
    by summing both the numerator and denominator with their reversed-LOS
    mirrors before normalisation, matching :func:`stack_cubelet`.

    Parameters
    ----------
        weighted_map: array.
            :math:`w_L L` on the map mesh.
        weight: array.
            :math:`w_L` on the map mesh.
        data_counts: array.
            Painted galaxy counts :math:`D`.
        n_nearby: int.
            Angular lags on each side of the centre pixel.
        random_counts: array, optional.
            Painted random counts :math:`R`. ``None`` skips the subtraction.
        weighting: {'conventional', 'quadratic'}, default 'conventional'.
            Normalisation scheme.
        symmetrize: bool, default False.
            Fold :math:`\pm s_\parallel` before normalising.
        return_details: bool, default False.
            If True, append ``(random_stack, random_denominator, valid_mask)``
            to the returned tuple.

    Returns
    -------
        data, fkp, denominator, alpha: arrays and float (plus
            ``random_stack, random_denominator, valid_mask`` when
            ``return_details=True``).
            Random-related entries are ``None`` when ``random_counts`` is
            omitted. ``denominator`` is the galaxy pair count.
    """
    if weighting not in ("conventional", "quadratic"):
        raise ValueError(
            f"weighting must be 'conventional' or 'quadratic', got {weighting!r}"
        )
    numerator = linear_correlation(weighted_map, data_counts, n_nearby)
    denominator = linear_correlation(weight, data_counts, n_nearby)
    if symmetrize:
        numerator = numerator + numerator[..., ::-1]
        denominator = denominator + denominator[..., ::-1]

    mid = int(n_nearby)
    centre = denominator.shape[-1] // 2
    if weighting == "conventional":
        data = _divide_cubelet(numerator, denominator)
    else:
        q0_d = float(denominator[mid, mid, centre])
        data = numerator / q0_d if q0_d != 0.0 else np.zeros_like(numerator)

    if random_counts is None:
        if return_details:
            return data, None, denominator, None, None, None, None
        return data, None, denominator, None

    r_sum = float(np.sum(random_counts))
    alpha = float(np.sum(data_counts)) / r_sum if r_sum > 0.0 else 0.0
    random_numerator = linear_correlation(weighted_map, random_counts, n_nearby)
    random_denominator = linear_correlation(weight, random_counts, n_nearby)
    if symmetrize:
        random_numerator = random_numerator + random_numerator[..., ::-1]
        random_denominator = random_denominator + random_denominator[..., ::-1]

    if weighting == "conventional":
        random_stack = _divide_cubelet(random_numerator, random_denominator)
        both = _pair_hit(denominator) & _pair_hit(random_denominator)
        fkp = np.zeros_like(data)
        fkp[both] = data[both] - random_stack[both]
    else:
        q0_d = float(denominator[mid, mid, centre])
        q0_r = float(random_denominator[mid, mid, centre])
        random_stack = (
            random_numerator / q0_r if q0_r != 0.0 else np.zeros_like(random_numerator)
        )
        if q0_d != 0.0 and r_sum > 0.0:
            fkp = (numerator - alpha * random_numerator) / q0_d
            both = np.ones(data.shape, dtype=bool)
        else:
            fkp = np.zeros_like(data)
            both = np.zeros(data.shape, dtype=bool)

    if return_details:
        return (
            data,
            fkp,
            denominator,
            float(alpha),
            random_stack,
            random_denominator,
            both,
        )
    return data, fkp, denominator, float(alpha)


class Stacking(PowerSpectrum):
    r"""
    Galaxy--intensity stacking on a map cube.

    ``stack_estimator='loop'`` calls :func:`stack`, which samples each
    galaxy on the native channels (positive LOS lag index corresponds to
    :math:`+\Delta\nu`). ``stack_estimator='correlation'`` paints the
    weighted galaxy counts on the map mesh, sums the sightlines into
    uniform comoving bins sorted in ascending :math:`\chi` (positive LOS
    lag index corresponds to :math:`+\Delta\chi`), and estimates the
    cubelet with a linear real FFT.

    Both ``'conventional'`` and ``'quadratic'`` weighting and
    ``symmetrize=True`` are supported by both estimators. When a random
    catalogue is set in ``'correlation'`` mode, the worker also stores
    :attr:`stack_fkp` (:math:`I_D - I_R` for ``'conventional'`` weighting,
    or :math:`[(w_L L)\star(D - \alpha R)]/Q_{0,D}` for ``'quadratic'``),
    :attr:`stack_rand` (:math:`I_R`), :attr:`stack_rand_weight`
    (:math:`w_L \star R`), :attr:`stack_valid_mask`, and :attr:`fkp_alpha`
    (:math:`\sum D / \sum R`). The loop estimator raises if randoms are
    set.

    The sky stack does not use the survey enclosing box. :attr:`n_gal_stacked`
    and :attr:`n_ran_stacked` count objects whose shifted coordinates fall
    inside the cube.

    Parameters
    ----------
        stack_estimator: ``{'loop', 'correlation'}``, default ``'loop'``.
            Which measurement to run.
        stack_angular_num_nearby_pix: int, default 4.
            Map pixels kept on each side of the centre. ``4`` is a 9×9 cubelet;
            ``0`` keeps only the centre pixel.
        stack_weighting: ``{'conventional', 'quadratic'}``, default ``'conventional'``.
            Normalisation scheme used by both estimators.
        symmetrize: bool, default False.
            Mirror the cubelet about zero line-of-sight separation.
        dchi_mpc: float, default 4.
            Comoving width, in Mpc, of the line-of-sight sum used by the
            correlation estimator.
        centre_pixels: bool, default True.
            Add half a cell before the nearest-grid deposit, matching the
            lightcone paint of a WCS pixel.
        ra_gal, dec_gal, z_gal: arrays, optional.
            Galaxy catalogue. Stored on the worker.
        weights_gal: array, optional.
            Per-galaxy weights. ``None`` is unit weight.
        ra_rand, dec_rand, z_rand: arrays, optional.
            Random catalogue. The correlation estimator then stores the
            random-subtracted cubelet on :attr:`stack_fkp`.
        weights_rand: array, optional.
            Per-random weights. ``None`` is unit weight.
        **params:
            Passed to :class:`meer21cm.power.PowerSpectrum`. ``data`` and
            ``weights_map_pixel`` are the intensity map and :math:`w_L`.
    """

    def __init__(
        self,
        stack_estimator="loop",
        stack_angular_num_nearby_pix=4,
        stack_weighting="conventional",
        symmetrize=False,
        dchi_mpc=4.0,
        centre_pixels=True,
        ra_gal=None,
        dec_gal=None,
        z_gal=None,
        weights_gal=None,
        ra_rand=None,
        dec_rand=None,
        z_rand=None,
        weights_rand=None,
        n_chi=5,
        kperp_edges=None,
        n_kperp=256,
        kperp_max=None,
        include_pixel_window=True,
        include_beam_window=True,
        pixel_power=2,
        beam_mode="continuous",
        power_kmu_func=None,
        **params,
    ):
        if stack_estimator not in ("loop", "correlation"):
            raise ValueError(
                "stack_estimator must be 'loop' or 'correlation', "
                f"got {stack_estimator!r}"
            )
        if stack_weighting not in ("conventional", "quadratic"):
            raise ValueError(
                "stack_weighting must be 'conventional' or 'quadratic', "
                f"got {stack_weighting!r}"
            )
        if int(stack_angular_num_nearby_pix) < 0:
            raise ValueError(
                "stack_angular_num_nearby_pix must be non-negative, "
                f"got {stack_angular_num_nearby_pix!r}"
            )
        if not np.isfinite(float(dchi_mpc)) or float(dchi_mpc) <= 0.0:
            raise ValueError("dchi_mpc must be positive")
        if int(n_chi) < 1:
            raise ValueError("n_chi must be >= 1")
        if int(n_kperp) < 8:
            raise ValueError("n_kperp must be at least 8")
        if beam_mode not in ("continuous", "discrete"):
            raise ValueError(
                f"beam_mode must be 'continuous' or 'discrete', got {beam_mode!r}"
            )
        kperp_edges_arr = (
            None if kperp_edges is None else np.asarray(kperp_edges, dtype=float)
        )
        if kperp_edges_arr is not None and (
            kperp_edges_arr.ndim != 1 or kperp_edges_arr.size < 2
        ):
            raise ValueError("kperp_edges must be a 1-D array with at least two edges")
        gal_cat = _validate_catalogue_arrays("gal", ra_gal, dec_gal, z_gal, weights_gal)
        rand_cat = _validate_catalogue_arrays(
            "rand", ra_rand, dec_rand, z_rand, weights_rand
        )
        super().__init__(**params)
        self.stack_estimator = stack_estimator
        self.stack_angular_num_nearby_pix = int(stack_angular_num_nearby_pix)
        self.stack_weighting = stack_weighting
        self.symmetrize = bool(symmetrize)
        self.dchi_mpc = float(dchi_mpc)
        self.centre_pixels = bool(centre_pixels)
        self.n_chi = int(n_chi)
        self._kperp_edges = kperp_edges_arr
        self.n_kperp = int(n_kperp)
        self.kperp_max = None if kperp_max is None else float(kperp_max)
        self.include_pixel_window = bool(include_pixel_window)
        self.include_beam_window = bool(include_beam_window)
        self.pixel_power = float(pixel_power)
        self.beam_mode = beam_mode
        self.power_kmu_func = power_kmu_func
        self._stack_3d = None
        self._stack_weight = None
        self._stack_fkp = None
        self._stack_rand = None
        self._stack_rand_weight = None
        self._stack_valid_mask = None
        self._bin_chi_edges = None
        self._bin_chi_centres = None
        self._bin_channel_which = None
        self._s_parallel_mpc = None
        self.fkp_alpha = None
        self.n_gal_stacked = 0
        self.n_ran_stacked = 0
        if gal_cat is not None:
            self._ra_gal, self._dec_gal, self._z_gal, w_g = gal_cat
            if w_g is not None:
                self.weights_gal = w_g
        if rand_cat is not None:
            self.ra_rand, self.dec_rand, self.z_rand, w_r = rand_cat
            if w_r is not None:
                self.weights_rand = w_r

    @classmethod
    def from_power_spectrum(cls, ps, data=None, **kwargs):
        """Build a :class:`Stacking` worker from a :class:`PowerSpectrum`-like object.

        Copies the map geometry, frequency axis, weights, beam, cosmological
        model parameters, and any attached galaxy/random catalogues so the
        stacking measurement and forward model match ``ps``.

        Parameters
        ----------
        ps : object
            A :class:`~meer21cm.power.PowerSpectrum`,
            :class:`~meer21cm.mock.MockSimulation`, or compatible object.
        data : array_like, optional
            Map cube to attach. Defaults to ``ps.data``.
        **kwargs
            Constructor overrides forwarded to :class:`Stacking`.

        Returns
        -------
        Stacking
        """
        params = {
            "filter_map_los": False,
            "auto_set_radecnu_bounds": False,
            "init_box_from_map_data": False,
        }
        skymap = getattr(ps, "skymap", None)
        if skymap is not None:
            params["skymap"] = skymap
        for key in (
            "nu",
            "wproj",
            "num_pix_x",
            "num_pix_y",
            "weights_map_pixel",
            "W_HI",
            "sigma_beam_ch",
            "beam_model",
            "beam_unit",
            "ra_range",
            "dec_range",
            "precision",
            "batch_number",
            "mean_amp_1",
            "mean_amp_2",
            "sigma_v_1",
            "sigma_v_2",
            "tracer_bias_1",
            "tracer_bias_2",
            "omega_hi",
            "fog_profile",
            "cross_coeff",
            "kaiser_rsd",
            "ps_type",
            "fiducial_cosmology",
            "true_cosmology",
            "backend",
        ):
            if key in kwargs or (
                key in ("wproj", "num_pix_x", "num_pix_y") and "skymap" in params
            ):
                continue
            val = getattr(ps, key, None)
            if val is not None:
                params[key] = val
        # Copy catalogues only when complete (and not overridden)
        if all(k not in kwargs for k in ("ra_gal", "dec_gal", "z_gal")):
            if (
                getattr(ps, "_ra_gal", None) is not None
                and getattr(ps, "_dec_gal", None) is not None
                and getattr(ps, "_z_gal", None) is not None
            ):
                params["ra_gal"] = ps.ra_gal
                params["dec_gal"] = ps.dec_gal
                params["z_gal"] = ps.z_gal
                if (
                    "weights_gal" not in kwargs
                    and getattr(ps, "weights_gal", None) is not None
                ):
                    params["weights_gal"] = ps.weights_gal
        if all(k not in kwargs for k in ("ra_rand", "dec_rand", "z_rand")):
            if (
                getattr(ps, "ra_rand", None) is not None
                and getattr(ps, "dec_rand", None) is not None
                and getattr(ps, "z_rand", None) is not None
            ):
                params["ra_rand"] = ps.ra_rand
                params["dec_rand"] = ps.dec_rand
                params["z_rand"] = ps.z_rand
                if (
                    "weights_rand" not in kwargs
                    and getattr(ps, "weights_rand", None) is not None
                ):
                    params["weights_rand"] = ps.weights_rand
        if data is None:
            data = getattr(ps, "data", None)
        params.update(kwargs)
        return cls(data=data, **params)

    def read_rand_cat(
        self,
        rand_files,
        ra_col="RA",
        dec_col="DEC",
        z_col="Z",
        weight_col=None,
        trim=True,
    ):
        """Read one or more FITS random catalogues and attach them to the worker.

        Uses :func:`meer21cm.io.read_catalogue_fits` to concatenate multiple
        random files.

        Parameters
        ----------
        rand_files : path or sequence of path
            Single FITS file path or sequence of FITS file paths.
        ra_col, dec_col, z_col : str, default 'RA', 'DEC', 'Z'
            Coordinate column names in HDU 1.
        weight_col : str, optional
            Per-object weight column. ``None`` leaves :attr:`weights_rand`
            unset (unit weights).
        trim : bool, default True
            If True, trim the randoms to the worker's ``ra_range``,
            ``dec_range``, and frequency/redshift window.

        Returns
        -------
        n_rand : int
            Number of random objects stored after optional trimming.
        """
        if isinstance(rand_files, (str, Path)):
            paths = [Path(rand_files)]
        else:
            paths = [Path(p) for p in rand_files]
        if len(paths) == 0:
            raise ValueError("rand_files must contain at least one FITS path")
        ra_r, dec_r, z_r, w_r = read_catalogue_fits(
            paths,
            ra_col=ra_col,
            dec_col=dec_col,
            z_col=z_col,
            weight_col=weight_col,
        )
        if trim:
            keep = self._objects_in_survey(ra_r, dec_r, z_r)
            ra_r = ra_r[keep]
            dec_r = dec_r[keep]
            z_r = z_r[keep]
            w_r = w_r[keep]
        self.ra_rand = ra_r
        self.dec_rand = dec_r
        self.z_rand = z_r
        self.weights_rand = None if weight_col is None else w_r
        return int(ra_r.size)

    @property
    def stack_3d(self):
        """
        Number-count cubelet from the last :meth:`run_stack`.

        This is the average of the map around the galaxies. A run with
        randoms leaves this cubelet unchanged and stores the subtraction
        on :attr:`stack_fkp`. Returns ``None`` if :meth:`run_stack` has
        not been called.
        """
        if self._stack_3d is None:
            logger.warning("stack_3d is unset; call run_stack() first")
            return None
        return self._stack_3d

    @property
    def stack_weight(self):
        """
        Cubelet weight from the last :meth:`run_stack`.

        Conventional weighting stores :math:`\\sum_i w^{\\rm gal}_i w_L`
        at each lag. Returns ``None`` if :meth:`run_stack` has not been called.
        """
        if self._stack_weight is None:
            logger.warning("stack_weight is unset; call run_stack() first")
            return None
        return self._stack_weight

    @property
    def stack_fkp(self):
        """
        Random-subtracted cubelet from the last correlation run with randoms.

        Equals :math:`I_D - I_R` for ``stack_weighting='conventional'`` and
        :math:`[(w_L L)\\star(D - \\alpha R)] / Q_{0,D}` for
        ``stack_weighting='quadratic'``. ``None`` when the last run had no
        randoms.
        """
        return self._stack_fkp

    @property
    def stack_rand(self):
        """Random-position cubelet :math:`I_R` from the last correlation run, or ``None``."""
        return self._stack_rand

    @property
    def stack_rand_weight(self):
        """Random pair-count cubelet :math:`w_L \\star R` from the last correlation run, or ``None``."""
        return self._stack_rand_weight

    @property
    def stack_valid_mask(self):
        """Boolean mask of lags with valid data and random pairs, or ``None``."""
        return self._stack_valid_mask

    @property
    def bin_chi_edges(self):
        """Comoving LOS bin edges in Mpc from the last correlation run, or ``None``."""
        return self._bin_chi_edges

    @property
    def bin_chi_centres(self):
        """Comoving LOS bin centres in Mpc from the last correlation run, or ``None``."""
        return self._bin_chi_centres

    @property
    def bin_channel_which(self):
        """Bin assignment of sorted channels from the last correlation run, or ``None``."""
        return self._bin_channel_which

    @property
    def s_parallel_mpc(self):
        """Comoving LOS lag axis in Mpc for the last correlation run, or ``None``."""
        return self._s_parallel_mpc

    @property
    def delta_i(self):
        """Integer angular lag indices along axis 0 (``-n_nearby .. +n_nearby``)."""
        n = int(self.stack_angular_num_nearby_pix)
        return np.arange(-n, n + 1, dtype=int)

    @property
    def delta_j(self):
        """Integer angular lag indices along axis 1 (``-n_nearby .. +n_nearby``)."""
        n = int(self.stack_angular_num_nearby_pix)
        return np.arange(-n, n + 1, dtype=int)

    def channel_chi_mpc(self):
        """
        Comoving distance of each frequency channel, in Mpc.

        Uses the fiducial cosmology.
        """
        return channel_comoving_distances(self)

    def _clear_random_outputs(self):
        """Reset random-catalogue outputs so stale values never leak across runs."""
        self._stack_fkp = None
        self._stack_rand = None
        self._stack_rand_weight = None
        self._stack_valid_mask = None
        self.fkp_alpha = None
        self.n_ran_stacked = 0

    def _default_gaussian_aperture_pixels(self):
        """Band-median Gaussian beam width in map pixels, falling back to 1.0."""
        sigma_ch = getattr(self, "sigma_beam_ch", None)
        if sigma_ch is None:
            return 1.0
        sigma_arr = np.atleast_1d(np.asarray(sigma_ch, dtype=float))
        sigma_deg = float(np.nanmedian(np.abs(sigma_arr)))
        pix_deg = float(self.pix_resol)
        if not np.isfinite(sigma_deg) or sigma_deg <= 0.0 or pix_deg <= 0.0:
            return 1.0
        return sigma_deg / pix_deg

    def project_1d(
        self,
        cubelet=None,
        aperture="central",
        radius=None,
        weights_2d=None,
        normalize=True,
    ):
        """Project a 3D stacked cubelet onto a 1D line-of-sight spectrum.

        Parameters
        ----------
        cubelet : ndarray, optional
            3D array of shape ``(2*n_nearby+1, 2*n_nearby+1, n_los)``.
            Defaults to :attr:`stack_3d`.
        aperture : {'central', 'tophat', 'gaussian'}, default 'central'
            Preset 2D angular aperture when ``weights_2d`` is omitted.
        radius : float, optional
            Aperture radius / Gaussian :math:`\\sigma` in map pixels. For
            ``aperture='gaussian'`` with ``radius=None``, uses the band-median
            ``sigma_beam_ch`` in pixel units when available (else ``1.0``).
        weights_2d : array_like, optional
            Custom 2D angular weight array overriding ``aperture``.
        normalize : bool, default True
            Normalise the 2D aperture weights to unit sum before contraction.

        Returns
        -------
        spectrum : ndarray
            1D projected spectrum of shape ``(n_los,)``.
        """
        if cubelet is None:
            cubelet = self.stack_3d
            if cubelet is None:
                raise ValueError(
                    "cubelet is None and stack_3d is unset; call run_stack() first"
                )
        cube = np.asarray(cubelet, dtype=float)
        if cube.ndim != 3 or cube.shape[0] != cube.shape[1] or cube.shape[0] % 2 == 0:
            raise ValueError(
                f"cubelet must have shape (2*n+1, 2*n+1, n_los), got {cube.shape}"
            )
        n_nearby = cube.shape[0] // 2
        if (
            weights_2d is None
            and str(aperture).lower() == "gaussian"
            and radius is None
        ):
            radius = self._default_gaussian_aperture_pixels()
        kernel = aperture_weights_2d(
            n_nearby,
            aperture=aperture,
            radius=radius,
            weights_2d=weights_2d,
            normalize=normalize,
        )
        return np.tensordot(kernel, cube, axes=([0, 1], [0, 1]))

    def run_stack(self):
        """
        Measure the cubelet with the selected estimator.

        The return value is the number-count stack and its pair-count
        weight. A correlation run with randoms also sets :attr:`stack_fkp`,
        :attr:`stack_rand`, :attr:`stack_rand_weight`,
        :attr:`stack_valid_mask`, and :attr:`fkp_alpha`.

        Returns
        -------
            stack_3d, stack_weight: arrays.
                Number-count cubelet and its pair-count weight.
        """
        if self.stack_weighting not in ("conventional", "quadratic"):
            raise ValueError(
                "stack_weighting must be 'conventional' or 'quadratic', "
                f"got {self.stack_weighting!r}"
            )
        if self.stack_estimator == "loop":
            if self.ra_rand is not None:
                raise ValueError("randoms are used by the correlation estimator")
            self._clear_random_outputs()
            self._bin_chi_edges = None
            self._bin_chi_centres = None
            self._bin_channel_which = None
            self._s_parallel_mpc = None
            cube, weight = stack(
                self,
                weights_gal=self.weights_gal,
                weighting=self.stack_weighting,
                stack_angular_num_nearby_pix=self.stack_angular_num_nearby_pix,
                symmetrize=self.symmetrize,
            )
            self.n_gal_stacked = int(np.size(self.ra_gal))
        elif self.stack_estimator == "correlation":
            cube, weight = self._correlation_stack()
        else:
            raise ValueError(
                "stack_estimator must be 'loop' or 'correlation', "
                f"got {self.stack_estimator!r}"
            )
        self._stack_3d = cube
        self._stack_weight = weight
        return cube, weight

    def _correlation_stack(self):
        """Paint, rebin, and correlate. The galaxy field is the weighted counts."""
        try:
            ra = self.ra_gal
            dec = self.dec_gal
            redshift = self.z_gal
        except AttributeError as err:
            raise ValueError("ra_gal, dec_gal and z_gal are required") from err
        gal_cat = _validate_catalogue_arrays("gal", ra, dec, redshift, self.weights_gal)
        if gal_cat is None:
            raise ValueError("ra_gal, dec_gal and z_gal are required")
        ra, dec, redshift, w_gal = gal_cat
        positions, inside = sky_mesh_positions(
            ra,
            dec,
            redshift,
            self.wproj,
            self.nu,
            self.data.shape,
            centre_pixels=self.centre_pixels,
        )
        if w_gal is None:
            mass = np.ones(ra.size, dtype=float)[inside]
        else:
            mass = w_gal[inside]
        self.n_gal_stacked = int(inside.sum())
        counts = paint_sky_counts(positions, self.data.shape, mass=mass)
        chi = self.channel_chi_mpc()
        _order, edges, centres, which = comoving_bin_indices(chi, self.dchi_mpc)
        self._bin_chi_edges = edges
        self._bin_chi_centres = centres
        self._bin_channel_which = which
        n_bin = centres.size
        self._s_parallel_mpc = (
            np.arange(-(n_bin - 1), n_bin, dtype=float) * self.dchi_mpc
        )
        weighted, weight, counts_b, _centres = rebin_sightlines(
            np.asarray(self.w_HI, dtype=float) * np.asarray(self.data, dtype=float),
            np.asarray(self.w_HI, dtype=float),
            counts,
            chi,
            self.dchi_mpc,
        )
        random_b = None
        self._clear_random_outputs()
        if (
            self.ra_rand is not None
            or self.dec_rand is not None
            or self.z_rand is not None
            or self.weights_rand is not None
        ):
            random_b, self.n_ran_stacked = self._rebinned_random_counts(chi)
        (
            data,
            fkp,
            denominator,
            alpha,
            rand_stack,
            rand_weight,
            valid_mask,
        ) = data_and_fkp_cubelets(
            weighted,
            weight,
            counts_b,
            self.stack_angular_num_nearby_pix,
            random_counts=random_b,
            weighting=self.stack_weighting,
            symmetrize=self.symmetrize,
            return_details=True,
        )
        self._stack_fkp = fkp
        self._stack_rand = rand_stack
        self._stack_rand_weight = rand_weight
        self._stack_valid_mask = valid_mask
        self.fkp_alpha = alpha
        return data, denominator

    def _rebinned_random_counts(self, chi):
        """Paint the random catalogue and sum it once onto the stack's LOS bins."""
        rand_cat = _validate_catalogue_arrays(
            "rand", self.ra_rand, self.dec_rand, self.z_rand, self.weights_rand
        )
        ra, dec, redshift, w_rand = rand_cat
        positions, inside = sky_mesh_positions(
            ra,
            dec,
            redshift,
            self.wproj,
            self.nu,
            self.data.shape,
            centre_pixels=self.centre_pixels,
        )
        if w_rand is None:
            mass = np.ones(ra.size, dtype=float)[inside]
        else:
            mass = w_rand[inside]
        painted = paint_sky_counts(positions, self.data.shape, mass=mass)
        counts, _centres = rebin_cube(painted, chi, self.dchi_mpc)
        return counts, int(inside.sum())

    # -- forward model -------------------------------------------------
    @property
    def _one_dim_model(self):
        """Live :class:`~meer21cm.los_model.OneDimPowerModel` bound to ``self``."""
        return OneDimPowerModel(
            self,
            n_chi=self.n_chi,
            kperp_edges=self._kperp_edges,
            n_kperp=self.n_kperp,
            kperp_max=self.kperp_max,
            include_pixel_window=self.include_pixel_window,
            include_beam_window=self.include_beam_window,
            pixel_power=self.pixel_power,
            beam_power=1.0,
            beam_mode=self.beam_mode,
            power_kmu_func=self.power_kmu_func,
            enforce_first_zero=False,
        )

    @property
    def chi_ch(self):
        """Comoving distance to each frequency channel centre, in Mpc."""
        return channel_comoving_distances(self)

    @property
    def dchi_ch(self):
        """Comoving width of each frequency channel, in Mpc."""
        return channel_comoving_widths(self)

    @property
    def kperp_edges(self):
        """Edges of the :math:`k_\\perp` integration bins, in Mpc^-1."""
        return self._one_dim_model.kperp_edges

    @property
    def kperp(self):
        """Centres of the :math:`k_\\perp` integration bins, in Mpc^-1."""
        return self._one_dim_model.kperp

    @property
    def dk_perp(self):
        """Widths of the :math:`k_\\perp` integration bins, in Mpc^-1."""
        return self._one_dim_model.dk_perp

    def _pair_weight_cubes(self, sort_chi=True):
        """Return ``(w_map, n_cube)`` cubes ordered by ascending chi or native channels."""
        if self.data is not None:
            shape = tuple(int(n) for n in self.data.shape)
        else:
            shape = (int(self.num_pix_x), int(self.num_pix_y), int(np.size(self.nu)))
        w_hi = getattr(self, "w_HI", None)
        if w_hi is not None and tuple(w_hi.shape) == shape:
            w_map = np.asarray(w_hi, dtype=float)
        else:
            w_map = np.ones(shape, dtype=float)

        # Prefer randoms (smooth selection estimate), fall back to galaxies, else W_HI / uniform
        if (
            self.ra_rand is not None
            and self.dec_rand is not None
            and self.z_rand is not None
        ):
            positions, inside = sky_mesh_positions(
                self.ra_rand,
                self.dec_rand,
                self.z_rand,
                self.wproj,
                self.nu,
                shape,
                centre_pixels=self.centre_pixels,
            )
            mass = (
                np.ones(int(np.size(self.ra_rand)), dtype=float)[inside]
                if self.weights_rand is None
                else np.asarray(self.weights_rand, dtype=float)[inside]
            )
            n_cube = paint_sky_counts(positions, shape, mass=mass)
        elif (
            getattr(self, "_ra_gal", None) is not None
            and getattr(self, "_dec_gal", None) is not None
            and getattr(self, "_z_gal", None) is not None
        ):
            positions, inside = sky_mesh_positions(
                self.ra_gal,
                self.dec_gal,
                self.z_gal,
                self.wproj,
                self.nu,
                shape,
                centre_pixels=self.centre_pixels,
            )
            mass = (
                np.ones(int(np.size(self.ra_gal)), dtype=float)[inside]
                if self.weights_gal is None
                else np.asarray(self.weights_gal, dtype=float)[inside]
            )
            n_cube = paint_sky_counts(positions, shape, mass=mass)
        else:
            w_mask = getattr(self, "W_HI", None)
            if w_mask is not None and tuple(w_mask.shape) == shape:
                n_cube = np.asarray(w_mask, dtype=float)
            else:
                n_cube = np.ones(shape, dtype=float)

        if sort_chi:
            order = np.argsort(self.chi_ch)
            return w_map[..., order], n_cube[..., order]
        return w_map, n_cube

    @staticmethod
    def _pair_weight_matrix_for_lag(w_map, n_cube, di, dj, separable=False):
        """Build the ``(n_ch, n_ch)`` pair-weight matrix at angular lag ``(di, dj)``."""
        nx, ny, n_ch = w_map.shape
        if abs(int(di)) >= nx or abs(int(dj)) >= ny:
            return np.zeros((n_ch, n_ch), dtype=float)
        di_i, dj_i = int(di), int(dj)
        x_m = slice(max(0, di_i), nx + min(0, di_i))
        x_g = slice(max(0, -di_i), nx - max(0, di_i))
        y_m = slice(max(0, dj_i), ny + min(0, dj_i))
        y_g = slice(max(0, -dj_i), ny - max(0, dj_i))
        w_sub = w_map[x_m, y_m, :]
        n_sub = n_cube[x_g, y_g, :]
        if separable:
            return np.outer(w_sub.sum(axis=(0, 1)), n_sub.sum(axis=(0, 1)))
        w_flat = w_sub.reshape(-1, n_ch)
        n_flat = n_sub.reshape(-1, n_ch)
        return w_flat.T @ n_flat

    def _chi_nodes(self, n_chi=None, channel_weights=None):
        """Comoving distance nodes across the band and their pair weights."""
        return chi_nodes_and_weights(
            np.sort(self.chi_ch),
            n_chi=self.n_chi if n_chi is None else int(n_chi),
            weights=channel_weights,
        )

    def transverse_window(
        self,
        di=0,
        dj=0,
        n_chi=None,
        n_phi=64,
        include_pixel_window=None,
        include_beam_window=None,
        pixel_power=None,
        beam_mode=None,
        channel_weights=None,
    ):
        r"""Band-averaged transverse stacking window at angular lag :math:`(\Delta i, \Delta j)`.

        Delegates to :meth:`meer21cm.los_model.OneDimPowerModel.transverse_window`
        with ``beam_power=1`` (linear map-leg dish beam) and default
        ``pixel_power=2`` (:math:`|W_{\rm pix}|^2`).

        Parameters
        ----------
        di, dj : int, default 0
            Integer angular lag offsets in map-pixel units.
        n_chi : int, optional
            Number of comoving distance nodes. Defaults to :attr:`n_chi`.
        n_phi : int, default 64
            Number of azimuthal angles for the square-pixel average.
        include_pixel_window : bool, optional
            Include the pixel top-hat window. Defaults to
            :attr:`include_pixel_window`.
        include_beam_window : bool, optional
            Include the dish beam window when :attr:`sigma_beam_ch` is set.
            Defaults to :attr:`include_beam_window`.
        pixel_power : int or float, optional
            Pixel-window exponent (default :attr:`pixel_power`, ``2``).
        beam_mode : {'continuous', 'discrete'}, optional
            ``'continuous'`` uses :func:`beam_window_amplitude` (continuous sky
            smoothing prior to pixelisation); ``'discrete'`` uses
            :func:`discrete_beam_window_amplitude` (pixel-grid Gaussian
            convolution). Defaults to :attr:`beam_mode`.
        channel_weights : array_like, optional
            Non-negative per-channel weights (in ascending :math:`\chi` order)
            used to weight the comoving distance nodes.

        Returns
        -------
        window : ndarray
            Band-averaged transverse window on :attr:`kperp`, shape
            ``(len(kperp),)``.
        """
        return self._one_dim_model.transverse_window(
            di=di,
            dj=dj,
            chi_grid=np.sort(self.chi_ch),
            node_weights=channel_weights,
            n_chi=n_chi,
            n_phi=n_phi,
            include_pixel_window=include_pixel_window,
            include_beam_window=include_beam_window,
            pixel_power=pixel_power,
            beam_mode=beam_mode,
        )

    def model_1d_power(
        self,
        kpar,
        di=0,
        dj=0,
        which="cross",
        include_mean_amp=True,
        cross_shot_noise=0.0,
        **window_kwargs,
    ):
        r"""Transverse-projected 1D cross-power :math:`P_{1D}^{(\Delta i,\Delta j)}(k_\parallel)`.

        Evaluates

        .. math::
            P_{1D}^{(\Delta i, \Delta j)}(k_\parallel)
            = \frac{1}{2\pi} \int_0^{k_{\perp,\max}} k_\perp\,\mathrm{d}k_\perp\;
              \left[P_{Lg}(k, \mu) + P_{\rm SN}\right]\,
              \overline{W}_\times^{(\Delta i, \Delta j)}(k_\perp)

        with :math:`k = \sqrt{k_\perp^2 + k_\parallel^2}` and
        :math:`\mu = k_\parallel / k`, delegating to
        :meth:`meer21cm.los_model.OneDimPowerModel.model_1d_power`.

        Parameters
        ----------
        kpar : array_like
            Line-of-sight wavenumbers in Mpc^-1.
        di, dj : int, default 0
            Integer angular lag offsets in map-pixel units.
        which : {'cross', 'auto_1', 'auto_2'}, default 'cross'
            Model power spectrum passed to :meth:`power_kmu`.
        include_mean_amp : bool, default True
            Multiply by ``mean_amp_1 * mean_amp_2`` (for ``'cross'``).
        cross_shot_noise : float or callable, default 0.0
            Optional cross shot noise added to :math:`P_{Lg}(k, \mu)`.
        **window_kwargs
            Forwarded to :meth:`transverse_window`.

        Returns
        -------
        p1d : ndarray
            Projected 1D power spectrum, shape ``(len(kpar),)``.
        """
        kw = dict(window_kwargs)
        ch_weights = kw.pop("channel_weights", None)
        return self._one_dim_model.model_1d_power(
            kpar,
            di=di,
            dj=dj,
            which=which,
            include_mean_amp=include_mean_amp,
            cross_shot_noise=cross_shot_noise,
            preserve_cache=True,
            chi_grid=np.sort(self.chi_ch),
            node_weights=ch_weights,
            **kw,
        )

    def _default_kpar_grids(self, chi_s, dchi_s, kpar_max=None, n_kpar_eval=256):
        """Build coarse P_1D evaluation grid and Nyquist-safe fine quadrature grid."""
        l_band = max(float(chi_s[-1] - chi_s[0]), float(np.sum(dchi_s)), 1.0)
        dchi_min = max(float(np.min(dchi_s)), 0.1)
        if kpar_max is None:
            kmax = max(4.0, min(16.0, 2.5 * np.pi / dchi_min))
        else:
            kmax = float(kpar_max)
            if not np.isfinite(kmax) or kmax <= 0.0:
                raise ValueError(f"kpar_max must be positive, got {kmax!r}")
        n_eval = max(32, int(n_kpar_eval))
        k_split = min(0.5, 0.25 * kmax)
        n_lin = n_eval // 2
        n_log = n_eval - n_lin
        k_lin = np.linspace(0.0, k_split, n_lin, endpoint=False)
        k_log = np.geomspace(k_split, kmax, n_log)
        kpar_eval = np.concatenate([k_lin, k_log])

        # Oversample pi / L_band by 4x so cos(k_par * Delta chi) has zero aliasing
        dk_quad = np.pi / (4.0 * l_band)
        n_quad = max(n_eval, int(np.ceil(kmax / dk_quad)) + 1)
        kpar_quad = np.linspace(0.0, kmax, n_quad)
        return kpar_eval, kpar_quad

    def _model_for_lags(
        self,
        lags,
        clean_matrix=None,
        which="cross",
        include_mean_amp=True,
        cross_shot_noise=0.0,
        separable_weights=False,
        dchi_gal="channel",
        kpar_max=None,
        n_kpar_eval=256,
        **window_kwargs,
    ):
        """Compute exact channel-contracted model profiles for a list of ``(di, dj)`` lags."""
        use_corr = self.stack_estimator == "correlation"
        chi_all = self.chi_ch
        dchi_all = self.dchi_ch
        if use_corr:
            order, edges, _centres, bin_which = comoving_bin_indices(
                chi_all, self.dchi_mpc
            )
            chi_work = chi_all[order]
            dchi_work = dchi_all[order]
            n_bin = edges.size - 1
            m_clean = (
                None
                if clean_matrix is None
                else np.asarray(clean_matrix, dtype=float)[np.ix_(order, order)]
            )
        else:
            order = None
            chi_work = chi_all
            dchi_work = dchi_all
            bin_which = None
            n_bin = None
            m_clean = (
                None if clean_matrix is None else np.asarray(clean_matrix, dtype=float)
            )

        if isinstance(dchi_gal, str):
            if dchi_gal != "channel":
                raise ValueError(
                    f"dchi_gal must be 'channel', None, a scalar, or an array; got {dchi_gal!r}"
                )
            dchi_g_work = dchi_work
        elif dchi_gal is None:
            dchi_g_work = 0.0
        elif np.ndim(dchi_gal) == 0:
            dchi_g_work = float(dchi_gal)
        else:
            dchi_g_arr = np.asarray(dchi_gal, dtype=float).ravel()
            dchi_g_work = dchi_g_arr[order] if order is not None else dchi_g_arr

        w_map, n_cube = self._pair_weight_cubes(sort_chi=use_corr)
        # Channel pair weights (in ascending chi order) for redshift-node averaging
        ch_weights = (w_map * n_cube).sum(axis=(0, 1))
        if not use_corr:
            ch_weights = ch_weights[np.argsort(chi_all)]

        kpar_eval, kpar_quad = self._default_kpar_grids(
            np.sort(chi_all),
            dchi_all[np.argsort(chi_all)],
            kpar_max=kpar_max,
            n_kpar_eval=n_kpar_eval,
        )
        model_1d = self._one_dim_model
        edges_kp = model_1d.kperp_edges
        kperp = 0.5 * (edges_kp[:-1] + edges_kp[1:])
        dk_perp = np.diff(edges_kp)
        _km, _mm, power_3d = model_1d.eval_power_kmu(
            kpar_eval,
            which=which,
            include_mean_amp=include_mean_amp,
            cross_shot_noise=cross_shot_noise,
            preserve_cache=True,
            kperp=kperp,
        )
        measure = (kperp * dk_perp / (2.0 * np.pi))[:, None]
        chi_sorted = np.sort(chi_all)
        win_kw = dict(window_kwargs)
        win_kw.pop("channel_weights", None)

        q0_global = None
        if self.stack_weighting == "quadratic":
            pair_w0 = self._pair_weight_matrix_for_lag(
                w_map, n_cube, 0, 0, separable=separable_weights
            )
            if bin_which is not None:
                s_mat = np.zeros((n_bin, pair_w0.shape[0]), dtype=float)
                s_mat[bin_which, np.arange(pair_w0.shape[0])] = 1.0
                q0_global = float(np.trace(s_mat @ pair_w0 @ s_mat.T))
            else:
                q0_global = float(np.trace(pair_w0))

        # Exploit 8-fold square symmetry for transverse window & channel covariance
        cov_cache = {}
        results = {}
        for di, dj in lags:
            sym_key = tuple(sorted((abs(int(di)), abs(int(dj)))))
            if sym_key not in cov_cache:
                w_perp = model_1d.transverse_window(
                    di=sym_key[0],
                    dj=sym_key[1],
                    chi_grid=chi_sorted,
                    node_weights=ch_weights,
                    kperp=kperp,
                    **win_kw,
                )
                p1d_eval = np.sum(measure * power_3d * w_perp[:, None], axis=0)
                p1d_quad = CubicSpline(kpar_eval, p1d_eval)(kpar_quad)
                cov_cache[sym_key] = channel_cross_covariance(
                    chi_work,
                    dchi_work,
                    kpar_quad,
                    p1d_quad,
                    clean_matrix=m_clean,
                    dchi_gal=dchi_g_work,
                )
            pair_w = self._pair_weight_matrix_for_lag(
                w_map, n_cube, di, dj, separable=separable_weights
            )
            profile, _denom = contract_channel_covariance(
                cov_cache[sym_key],
                pair_w,
                bin_which=bin_which,
                n_bin=n_bin,
                weighting=self.stack_weighting,
                symmetrize=self.symmetrize,
                q0=q0_global,
            )
            results[(int(di), int(dj))] = profile
        return results

    def model_3d_stack(
        self,
        clean_matrix=None,
        which="cross",
        include_mean_amp=True,
        cross_shot_noise=0.0,
        separable_weights=False,
        dchi_gal="channel",
        kpar_max=None,
        n_kpar_eval=256,
        **window_kwargs,
    ):
        r"""Forward model of the full 3D stacked cubelet.

        Evaluates the exact channel-space covariance contraction matching
        :attr:`stack_estimator`, :attr:`stack_weighting`, :attr:`symmetrize`,
        :attr:`dchi_mpc`, map weights :attr:`w_HI`, galaxy/random selection,
        square-pixel window (:math:`|W_{\rm pix}|^2`), dish beam
        (:math:`W_{\rm beam}`), and an optional linear cleaning matrix
        :math:`\mathbb{M}` (in the map's native channel order).

        Parameters
        ----------
        clean_matrix : array_like, optional
            Linear frequency-axis cleaning operator :math:`\mathbb{M}` of shape
            ``(n_ch, n_ch)`` in the map's native ``self.nu`` channel order.
        which : {'cross', 'auto_1', 'auto_2'}, default 'cross'
            Model power spectrum passed to :meth:`power_kmu`.
        include_mean_amp : bool, default True
            Include ``mean_amp`` factors (e.g. :math:`\bar{T}_{\rm HI}`).
        cross_shot_noise : float or callable, default 0.0
            Optional cross shot noise added to :math:`P_{Lg}(k, \mu)`.
        separable_weights : bool, default False
            If True, approximate the per-lag pair-weight matrix by the outer
            product of sky-summed map and galaxy weights. ``False`` uses the
            exact 3D non-separable pair-weight matrix.
        dchi_gal : {'channel', None} or float or array_like, default 'channel'
            Radial window width on the galaxy leg.
        kpar_max : float, optional
            Upper limit of the :math:`k_\parallel` integral, in Mpc^-1.
        n_kpar_eval : int, default 256
            Number of coarse :math:`k_\parallel` nodes at which ``power_kmu``
            is evaluated before interpolating onto the fine quadrature grid.
        **window_kwargs
            Forwarded to :meth:`transverse_window`.

        Returns
        -------
        cubelet : ndarray
            Model 3D cubelet of shape
            ``(2 * n_nearby + 1, 2 * n_nearby + 1, n_los)`` matching
            :attr:`stack_3d`.
        """
        n_nearby = int(self.stack_angular_num_nearby_pix)
        lag_vals = np.arange(-n_nearby, n_nearby + 1, dtype=int)
        lags = [(int(di), int(dj)) for di in lag_vals for dj in lag_vals]
        profiles = self._model_for_lags(
            lags,
            clean_matrix=clean_matrix,
            which=which,
            include_mean_amp=include_mean_amp,
            cross_shot_noise=cross_shot_noise,
            separable_weights=separable_weights,
            dchi_gal=dchi_gal,
            kpar_max=kpar_max,
            n_kpar_eval=n_kpar_eval,
            **window_kwargs,
        )
        n_los = next(iter(profiles.values())).size
        side = 2 * n_nearby + 1
        cube = np.zeros((side, side, n_los), dtype=float)
        for di, dj in lags:
            cube[di + n_nearby, dj + n_nearby, :] = profiles[(di, dj)]
        return cube

    def model_1d_stack(
        self,
        aperture="central",
        radius=None,
        weights_2d=None,
        normalize=True,
        clean_matrix=None,
        which="cross",
        include_mean_amp=True,
        cross_shot_noise=0.0,
        separable_weights=False,
        dchi_gal="channel",
        kpar_max=None,
        n_kpar_eval=256,
        **window_kwargs,
    ):
        r"""Forward model of a 1D aperture-projected stacked spectrum.

        Evaluates only the angular lags with nonzero aperture weight and
        projects them identically to :meth:`project_1d`.

        Parameters
        ----------
        aperture : {'central', 'tophat', 'gaussian'}, default 'central'
            Preset 2D angular aperture when ``weights_2d`` is omitted.
        radius : float, optional
            Aperture radius / Gaussian :math:`\sigma` in map pixels. For
            ``aperture='gaussian'`` with ``radius=None``, uses the band-median
            ``sigma_beam_ch`` in pixel units when available (else ``1.0``).
        weights_2d : array_like, optional
            Custom 2D angular weight array overriding ``aperture``.
        normalize : bool, default True
            Normalise the 2D aperture weights to unit sum before contraction.
        clean_matrix, which, include_mean_amp, cross_shot_noise, separable_weights, dchi_gal, kpar_max, n_kpar_eval, **window_kwargs
            Forwarded to the channel-space model (see :meth:`model_3d_stack`).

        Returns
        -------
        spectrum : ndarray
            Model 1D stacked spectrum of shape ``(n_los,)`` matching
            ``self.project_1d(self.stack_3d, ...)``.
        """
        n_nearby = int(self.stack_angular_num_nearby_pix)
        if (
            weights_2d is None
            and str(aperture).lower() == "gaussian"
            and radius is None
        ):
            radius = self._default_gaussian_aperture_pixels()
        kernel = aperture_weights_2d(
            n_nearby,
            aperture=aperture,
            radius=radius,
            weights_2d=weights_2d,
            normalize=normalize,
        )
        active_idx = np.argwhere(kernel != 0.0)
        active_lags = [
            (int(ix - n_nearby), int(iy - n_nearby)) for ix, iy in active_idx
        ]
        profiles = self._model_for_lags(
            active_lags,
            clean_matrix=clean_matrix,
            which=which,
            include_mean_amp=include_mean_amp,
            cross_shot_noise=cross_shot_noise,
            separable_weights=separable_weights,
            dchi_gal=dchi_gal,
            kpar_max=kpar_max,
            n_kpar_eval=n_kpar_eval,
            **window_kwargs,
        )
        n_los = next(iter(profiles.values())).size
        spec = np.zeros(n_los, dtype=float)
        for (di, dj), (ix, iy) in zip(active_lags, active_idx):
            spec += float(kernel[ix, iy]) * profiles[(di, dj)]
        return spec
