"""
Pure helper functions for power spectrum estimation and modelling.

Used by :mod:`meer21cm.estimator`, :mod:`meer21cm.model`, and
:mod:`meer21cm.power`. Kept free of survey I/O and cosmology state.
"""

import numpy as np
from .util import real_dtype_from_array


def get_renormed_field(
    real_field,
    weights=None,
    mean_center=False,
    unitless=False,
    field_mean=None,
):
    """
    Mean center the field and renormalise it by dividing the mean.

    Parameters
    ----------
    real_field: np.ndarray
        The real-space field.
    weights: np.ndarray, default None
        The weights of the field.
    mean_center: bool, default False
        Whether to mean center the field.
    unitless: bool, default False
        Whether to make the field unitless.
    field_mean: float, default None
        If given, use this mean instead of the weighted mean of ``real_field``.
        Ignored when both ``mean_center`` and ``unitless`` are False.

    Returns
    -------
    field: np.ndarray
        The renormalized field.
    """
    field = np.asarray(real_field)
    real_dtype = real_dtype_from_array(field)
    field = field.astype(real_dtype, copy=True)
    if weights is None:
        weights = np.ones_like(field, dtype=real_dtype)
    weights = np.asarray(weights, dtype=real_dtype)
    if mean_center or unitless:
        if field_mean is None:
            field_mean = np.sum(weights * field) / np.sum(weights)
        else:
            field_mean = np.asarray(field_mean, dtype=real_dtype)
    else:
        return real_field
    if mean_center:
        field -= field_mean
    if unitless:
        field /= field_mean
    return field


def get_fourier_density(
    real_field,
    weights=None,
    mean_center=False,
    unitless=False,
    norm="forward",
    field_mean=None,
):
    """
    Perform Fourier transform of a density field in real space. Note that
    this is deliberately written in a way that is not dimension specific.
    It can be used to calculate power spectrum of arbitrary dimension.

    Note that, the field is multiplied by the weights
    and then Fourier-transformed, and is **not weight normalised**.

    Parameters
    ----------
    real_field: np.ndarray
        The real-space field.
    weights: np.ndarray, default None
        The weights of the field.
    mean_center: bool, default False
        Whether to mean center the field.
    unitless: bool, default False
        Whether to make the field unitless.
    norm: str, default "forward"
        The normalization of the Fourier transform. Naming is the same as np.fft.
    field_mean: float, default None
        If given, forwarded to :func:`get_renormed_field` so the mean used
        for centering / unitless conversion is not recomputed from the field.

    Returns
    -------
    fourier_field: np.ndarray
        The Fourier transform of the field.
    """
    field = get_renormed_field(
        real_field,
        weights=weights,
        mean_center=mean_center,
        unitless=unitless,
        field_mean=field_mean,
    )
    if weights is None:
        weights = np.ones_like(field, dtype=real_dtype_from_array(field))
    weights = np.asarray(weights, dtype=real_dtype_from_array(field))
    fourier_field = np.fft.rfftn(field * weights, norm=norm)
    return fourier_field


def get_x_vector(box_ndim, box_resol):
    """
    Get the position vector along each direction for a given box.

    Parameters
    ----------
    box_ndim: int
        The number of dimensions of the box.
    box_resol: float
        The resolution of the box.

    Returns
    -------
    xvecarr: tuple
        The position vector along each direction.
    """
    xvecarr = tuple(
        box_resol[i] * (np.arange(box_ndim[i]) + 0.5) for i in range(len(box_ndim))
    )
    return xvecarr


def get_k_vector(box_ndim, box_resol):
    """
    Get the wavenumber vector along each direction
    for a given box.

    Parameters
    ----------
    box_ndim: int
        The number of dimensions of the box.
    box_resol: float
        The resolution of the box.

    Returns
    -------
    kvecarr: tuple
        The wavenumber vector along each direction.
    """
    kvecarr = [
        2
        * np.pi
        * np.fft.fftfreq(
            box_ndim[i],
            d=box_resol[i],
        )
        for i in range(len(box_ndim))
    ]
    kvecarr[-1] = np.abs(kvecarr[-1][: box_ndim[-1] // 2 + 1])
    return kvecarr


def get_vec_mode(vecarr):
    """
    Calculate the mode of the n-dimensional vectors on the grids

    Parameters
    ----------
    vecarr: tuple
        The vectors.

    Returns
    -------
    mode: np.ndarray
        The mode of the vectors.
    """
    result = np.sqrt(
        np.sum(
            (np.meshgrid(*([(vec) ** 2 for vec in vecarr]), indexing="ij")),
            0,
        )
    )
    return result


def get_shot_noise_galaxy(
    gal_count,
    box_len,
    weights_grid=None,
    weights_field=None,
):
    """
    Calculate the shot noise of a galaxy number count field.
    """
    gal_count = np.asarray(gal_count)
    real_dtype = real_dtype_from_array(gal_count)
    if weights_grid is None:
        weights_grid = np.ones(gal_count.shape, dtype=real_dtype)
    if weights_field is None:
        weights_field = np.ones(gal_count.shape, dtype=real_dtype)
    weights_grid = np.asarray(weights_grid, dtype=real_dtype)
    weights_field = np.asarray(weights_field, dtype=real_dtype)
    w_g_n = (weights_grid * gal_count).sum() / gal_count.sum()
    w_2_g_n = (weights_grid**2 * gal_count).sum() / gal_count.sum()
    wfwg_2_v = ((weights_field * weights_grid) ** 2).mean()
    wfwg_v = (weights_field * weights_grid).mean()
    shot_noise = (
        np.prod(box_len)
        / gal_count.sum()
        * w_2_g_n
        / w_g_n**2
        * (wfwg_v**2 / wfwg_2_v)
    )
    return shot_noise


def get_shot_noise(
    real_field,
    box_len,
    weights=None,
):
    """
    Calculate the shot noise of a field.

    Parameters
    ----------
    real_field: np.ndarray
        The real-space field.
    box_len: tuple
        The length of the box along each direction.
    weights: np.ndarray, default None
        The weights of the field.

    Returns
    -------
    shot_noise: float
        The shot noise of the field.
    """
    real_dtype = real_dtype_from_array(real_field)
    box_len = np.asarray(box_len, dtype=real_dtype)
    box_volume = np.prod(box_len)
    if weights is None:
        weights = np.ones(real_field.shape, dtype=real_dtype)
    weights = np.asarray(weights, dtype=real_dtype)
    weights_renorm = power_weights_renorm(weights, weights)
    shot_noise = (
        box_volume
        * np.sum((weights * real_field) ** 2)
        / np.sum(weights * real_field) ** 2
        * weights_renorm
        * np.mean(weights) ** 2
    )
    return shot_noise


def get_modelpk_conv(psmod, weights1_in_real=None, weights2=None, renorm=True):
    """
    Convolve a model power spectrum with real-space weights.

    Parameters
    ----------
    psmod: np.ndarray
        The model power spectrum.
    weights1_in_real: np.ndarray, default None
        The real-space weights for the first field. Default is None, which means no weights.
    weights2: np.ndarray, default None
        The real-space weights for the second field. Default is None, which means no weights.
    renorm: bool, default True
        Whether to renormalize the power spectrum.

    Returns
    -------
    power_conv: np.ndarray
        The convolved power spectrum.
    """
    if weights1_in_real is None and weights2 is None:
        return psmod
    if weights1_in_real is None:
        weights1_in_real = np.ones_like(weights2)
    if weights2 is None:
        weights2 = np.ones_like(weights1_in_real)
    assert np.allclose(weights1_in_real.shape, weights2.shape)
    if np.allclose(weights1_in_real, 1) and np.allclose(weights2, 1):
        return psmod
    weights_fourier = np.fft.rfftn(weights1_in_real)
    weights_fourier *= np.conj(np.fft.rfftn(weights2))
    # using fft instead of rfft somehow is wrong, I have no idea why
    # the behaviour of convolving with uniform weights is incorrect at k_xyz=0 due to rfft
    power_conv = (
        np.fft.rfftn(
            np.fft.irfftn(psmod, axes=[0, 1, 2], s=weights2.shape)
            * np.fft.irfftn(weights_fourier, axes=[0, 1, 2], s=weights2.shape)
        )
        / weights1_in_real.size
    ) * (
        (weights2.shape[-1] // 2 * 2) / weights2.shape[-1]
    )  # rfft correction for odd number?
    if renorm:
        weights_renorm = power_weights_renorm(weights1_in_real, weights2=weights2)
        power_conv *= weights_renorm
    return power_conv.real


def power_weights_renorm(weights1_in_real=None, weights2=None):
    r"""
    Calculate the renormalization coefficient based on the weights
    on the density field when calculating power spectrum.
    The renormalization is defined as

    .. math::
        \frac{{N_{\rm grid}}} {\sum_{i} w_1(x_i) w_2(x_i)},

    where :math:`N_{\rm grid}` is the number of grids in the box and
    :math:`i` loops over all the grids.

    Note that this renormaliszation corresponds to the diagonal
    renormalisation matrix that does not change the window function convolution,
    but only renormalises the sum of each row of the window function matrix.
    See Chen (2025) [1] for more details.

    Parameters
    ----------
        weights1_in_real: array, default None.
            The weights of the density field in real space.
            Must be in the shape of the regular grid field.
        weights2: array, default None.
            If cross-correlation, the weights for the second field.

    Returns
    -------
        weights_norm: float.
           The renormalization coefficient.

    References
    ----------
        [1] Chen, Z., 2025, "A quadratic estimator view of the transfer function correction in intensity mapping surveys",
        https://ui.adsabs.harvard.edu/abs/2025MNRAS.542L...1C/abstract.
    """
    if weights1_in_real is None and weights2 is None:
        return 1.0
    if weights1_in_real is None:
        weights1_in_real = np.ones_like(weights2)
    if weights2 is None:
        weights2 = np.ones_like(weights1_in_real)
    weights_norm = weights1_in_real.size / np.sum(weights1_in_real * weights2)
    return weights_norm


def get_power_spectrum(
    fourier_field,
    box_len,
    weights=None,
    field_2=None,
    weights_2=None,
    renorm=True,
):
    """
    Calculate the power spectrum for one/two given Fourier fields.

    Parameters
    ----------
    fourier_field: np.ndarray
        The Fourier field of the first tracer.
    box_len: tuple
        The length of the box along each direction.
    weights: np.ndarray, default None
        The weights of the first tracer **in real space**.
    field_2: np.ndarray, default None
        The Fourier field of the second tracer. If None, it is set to be the same as the first field.
    weights_2: np.ndarray, default None
        The weights of the second tracer **in real space**. **Must be provided if field_2 is provided.**
    renorm: bool, default True
        Whether to renormalize the power spectrum by the weights.

    Returns
    -------
    power: np.ndarray
        The power spectrum.
    """
    box_len = np.array(box_len)
    box_volume = np.prod(box_len)
    if field_2 is None:
        field_2 = fourier_field
    fourier_field = np.array(fourier_field)
    field_2 = np.array(field_2)
    power = np.real(fourier_field * np.conj(field_2))
    if weights is None and weights_2 is None:
        return power * box_volume
    if weights is None:
        weights = np.ones(weights_2.shape)
    if weights_2 is None:
        weights_2 = weights
    # if weights_2 is None, the renormalisation sets it to weights
    weights_norm = power_weights_renorm(weights, weights_2)
    if renorm:
        power *= weights_norm
    return power * box_volume


def get_gaussian_noise_floor(
    sigma_n,
    box_ndim,
    box_volume=1.0,
    counts=None,
):
    """
    Calculate the Gaussian noise floor for a given field.

    Parameters
    ----------
    sigma_n: float
        The standard deviation of the noise before being averaged down by the sampling.
    box_ndim: tuple
        The number of grids of the box along each direction.
    box_volume: float, default 1.0
        The volume of the box.
    counts: np.ndarray, default None
        The number of sampling in the box. If None, it is set to be 1.0.

    Returns
    -------
    noise_floor: float
        The noise floor.
    """
    box_ndim = np.array(box_ndim)
    if counts is None:
        counts = np.ones(box_ndim.tolist())
    counts = np.array(counts)
    box_std = sigma_n / np.sqrt(counts)
    fourier_var = np.sum(box_std**2) / np.prod(box_ndim) ** 2
    return fourier_var * box_volume


def bin_3d_to_1d(
    ps3d,
    kfield,
    k1dedges,
    weights=None,
    error=False,
    vectorize=False,
):
    r"""
    Bin a 3d distribution, e.g. power spectrum :math:`P_{3D}(\vec{k})`, into 1D average.

    Note that, the distribution is unraveled to a 1D array, so essentially an array of any
    dimension would do, as long as ``ps3d``, ``kfield``, and ``weights`` have the same size.

    The mean of the 1D average is calculated as

    .. math::
        \hat{P}_{\rm 1D}^{i} = \big(\sum_j P_{\rm 3D}^{ j} w_{ j} \big)/\big(\sum_j w_{ j}\big),

    where j loops over all the modess that fall into the :math:`i^{\rm th}` bin
    and :math:`w_{ j}` is the weights.

    If ``error`` is set to ``True``, a sampling error is calculated and returned so that

    .. math::
        (\Delta P_{\rm 1D}^{\rm i})^2 = \big(\sum_j (P_{\rm 3D}^{\rm j}-\hat{P}_{\rm 1D}^{\rm i})^2 w_{\rm j}^2 \big) \Big/ \big(\sum_j w_{\rm j}\big)^2.

    Parameters
    ----------
    ps3d: np.ndarray
        The 3D distribution to be binned.
    kfield: np.ndarray
        The k-field of the 3D distribution.
    k1dedges: np.ndarray
        The bin edges for the 1D power spectrum.
    weights: np.ndarray, default None
        The weights for each 3D k-mode of the power spectrum.
    error: bool, default False
        Whether to calculate the sampling error.
    vectorize: bool, default False
        Whether to vectorize the calculation, assuming the first axis is independent realisations.

    Returns
    -------
    ps1d: np.ndarray
        The 1D power spectrum.
    ps1derr: np.ndarray
        The sampling error for the 1D power spectrum. Returned only if ``error`` is ``True``.
    k1deff: np.ndarray
        The effective k-mode for each bin.
    nmodes: np.ndarray
        The number of modes in each bin.
    """
    if not vectorize:
        ps3d = np.array(ps3d)[None, ...]
    if weights is None:
        weights = np.ones_like(ps3d[0])
    ps3d = np.array(ps3d).reshape(len(ps3d), -1)
    kfield = np.array(kfield).ravel()
    weights = np.array(weights).ravel()

    indx = (kfield[:, None] >= k1dedges[None, :-1]) * (
        kfield[:, None] < k1dedges[None, 1:]
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        ps1d = np.sum(
            ps3d[:, :, None] * indx[None, :, :] * weights[None, :, None], 1
        ) / np.sum(indx[None, :, :] * weights[None, :, None], 1)
        k1deff = np.sum(kfield[:, None] * indx * weights[:, None], 0) / np.sum(
            indx * weights[:, None], 0
        )
    if error is True:
        with np.errstate(divide="ignore", invalid="ignore"):
            ps1derr = np.sqrt(
                np.sum(
                    (ps3d[:, :, None] - ps1d[:, None, :]) ** 2
                    * (indx[None, :, :] * weights[None, :, None]) ** 2,
                    1,
                )
                / np.sum((indx[None, :, :] * weights[None, :, None]), 1) ** 2
            )
        if not vectorize:
            ps1derr = ps1derr[0]
    if not vectorize:
        ps1d = ps1d[0]
    nmodes = np.sum(indx * (weights[:, None] > 0), 0)

    if error is True:
        return ps1d, ps1derr, k1deff, nmodes
    else:
        return ps1d, k1deff, nmodes


def bin_3d_to_cy(
    ps3d,
    kperp_i,
    kperpedges,
    weights=None,
    average=True,
    vectorize=False,
):
    """
    Function to bin a 3D distribution (e.g. power spectrum) into cylindrical average.
    The arrays are unravelled to 2D before binning by keeping the last axis.
    The 2D array is then binned along the first axis.

    The output is flipped so that the first axis is the original last axis.
    Therefore, to bin a 3D power spectrum to a cylindrical average,
    one can simply run ``bin_3d_to_cy`` twice
    (see ``PowerSpectrum.get_cy_power``).

    Parameters
    ----------
    ps3d: array.
        The 3D distribution to be binned.
    kperp_i: array.
        The perpendicular k-mode corresponding to the first axis.
    kperpedges: array.
        The bin edges for the perpendicular k-mode.
    weights: array, None.
        The weights for each 3D k-mode of the power spectrum.
    average: bool, default True.
        If ``True``, calculate the weighted average of the power spectrum
        in each bin. Else, calculate the weighted sum.
    vectorize: bool, default False
        Whether to vectorize the calculation, assuming the first axis is independent realisations.

    Returns
    -------
    pscy: np.ndarray
        The cylindrical average of the 3D distribution.
    """
    ps3d = np.array(ps3d)
    if not vectorize:
        ps3d = ps3d[None, ...]
    kperpedges = np.array(kperpedges)
    kperp_i = np.array(kperp_i).ravel()
    ps3d = ps3d.reshape((len(ps3d), len(kperp_i), -1))
    if weights is None:
        weights = np.ones_like(ps3d[0])
    weights = np.array(weights).reshape((len(kperp_i), -1))
    indx = (kperp_i[:, None] >= kperpedges[None, :-1]) * (
        kperp_i[:, None] < kperpedges[None, 1:]
    )
    weights = indx[:, None, :] * weights[:, :, None]
    pscy = np.sum(ps3d[:, :, :, None] * weights[None], 1)
    if average:
        pscy = pscy / np.sum(weights, 0)[None]
    if not vectorize:
        pscy = pscy[0]
    return pscy


def gaussian_beam_attenuation(k_perp, beam_sigma_in_mpc):
    """
    The beam attenuation term to be multiplied to model power
    spectrum assuming a Gaussian beam.

    Parameter
    ---------
    k_perp: np.ndarray.
        The transverse k-scale in Mpc^-1
    beam_sigma_in_mpc: float.
        The sigma of the Gaussian beam in Mpc.

    Returns
    -------
    beam_attenuation: np.ndarray.
        The beam attenuation factor.
    """
    return np.exp(-(k_perp**2) * beam_sigma_in_mpc**2 / 2)


def step_window_attenuation(k_dir, step_size_in_mpc, p=1):
    """
    The beam attenuation term to be multiplied to model power
    spectrum assuming a Gaussian beam.

    Parameter
    ---------
    k_perp: float.
        The transverse k-scale in Mpc^-1
    beam_sigma_in_mpc: float.
        The sigma of the Gaussian beam in Mpc.
    p: int, default 1
        The index of assignment scheme.

    Returns
    -------
    window_attenuation: np.ndarray.
        The window attenuation factor.
    """
    # note np.sinc is sin(pi x)/(pi x)
    return np.sinc(k_dir * step_size_in_mpc / np.pi / 2) ** p
