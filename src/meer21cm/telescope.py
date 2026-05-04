import numpy as np
from .util import get_wcs_coor, get_ang_between_coord, freq_to_redshift, tagging
from astropy import units, constants
from scipy.signal import convolve
from astropy.cosmology import Planck18
import healpy as hp
from astropy.wcs import WCS
import meer21cm

default_data_dir = meer21cm.__file__.rsplit("/", 1)[0] + "/data/"

meerkat_L_band_nu_min = 856.0 * 1e6  # in Hz
meerkat_L_band_nu_max = 1712.0 * 1e6  # in Hz
meerkat_L_4k_delta_nu = 0.208984375 * 1e6  # in Hz

meerklass_L_deep_nu_min = 971 * 1e6
meerklass_L_deep_nu_max = 1023.8 * 1e6

meerklass_L_pilot_nu_min = 971 * 1e6
meerklass_L_pilot_nu_max = 1023.2 * 1e6

meerkat_UHF_band_nu_min = 544.0 * 1e6  # in Hz
meerkat_UHF_band_nu_max = 1088.0 * 1e6  # in Hz
meerkat_UHF_4k_delta_nu = 0.1328125 * 1e6  # in Hz

meerklass_UHF_deep_nu_min = 610.0 * 1e6
meerklass_UHF_deep_nu_max = 929.2 * 1e6

default_nu_min = {
    "meerkat_L": meerkat_L_band_nu_min,
    "meerkat_UHF": meerkat_UHF_band_nu_min,
    "meerklass_2021_L": meerklass_L_deep_nu_min,
    "meerklass_2019_L": meerklass_L_pilot_nu_min,
    "meerklass_UHF": meerklass_UHF_deep_nu_min,
}

default_nu_max = {
    "meerkat_L": meerkat_L_band_nu_max,
    "meerkat_UHF": meerkat_UHF_band_nu_max,
    "meerklass_2021_L": meerklass_L_deep_nu_max,
    "meerklass_2019_L": meerklass_L_pilot_nu_max,
    "meerklass_UHF": meerklass_UHF_deep_nu_max,
}

default_num_pix_x = {
    "meerkat_L": None,
    "meerkat_UHF": None,
    "meerklass_2021_L": 133,
    "meerklass_2019_L": None,
    "meerklass_UHF": None,
}

default_num_pix_y = {
    "meerkat_L": None,
    "meerkat_UHF": None,
    "meerklass_2021_L": 73,
    "meerklass_2019_L": None,
    "meerklass_UHF": None,
}

default_wproj = {
    "meerkat_L": None,
    "meerkat_UHF": None,
    "meerklass_2021_L": WCS(default_data_dir + "test_fits.fits").dropaxis(-1),
    "meerklass_2019_L": None,
    "meerklass_UHF": None,
}


def weighted_convolution(
    signal,
    kernel,
    weights,
    kernel_renorm=True,
    los_axis=-1,
):
    r"""
    Perform weighted convolution of signal. The weighted convolution of the signal is defined as

    .. math::
        \tilde{s} = [(s \cdot w) * b]/[w * b],

    where :math:`s` is the signal, :math:`w` is the weight,
    :math:`b` is the convolution kernel and :math:`w` denotes convolution.

    The convolution also creates new weights for the output signal so that

    .. math::
        \tilde{w} = [w * b]^2 / [w * b^2]

    Parameters
    ----------
        signal: float.
            The input signal to be convolved
        kernel: float.
            The convolution kernel
        weights: float.
            The weights for the signal
        kernel_renorm: boolean, default True.
            Whether to renormalise the kernel so that the sum of the kernel is one.
            Should be set to ``True`` for temperature and ``False`` for flux density.
        los_axis: int, default -1.
            which axis is the los.


    Returns
    -------
        conv_signal: float.
            The convolved signal.
        conv_weights: float.
            The convolved weights.
    """
    if los_axis < 0:
        los_axis += 3
    # make sure los is the last axis
    axes = [0, 1, 2]
    axes.remove(los_axis)
    axes = axes + [
        los_axis,
    ]
    signal = np.transpose(signal * weights, axes=axes)
    kernel = np.transpose(kernel, axes=axes)
    weights = np.transpose(weights, axes=axes)
    if kernel_renorm:
        kernel /= kernel.sum(axis=(0, 1))[None, None, :]
    kernel_square = kernel**2
    kernel_square /= kernel_square.sum(axis=(0, 1))[None, None, :]
    conv_signal = np.zeros_like(signal)
    conv_variance = np.zeros_like(signal)
    for ch_i in range(signal.shape[-1]):
        weight_renorm = convolve(
            weights[:, :, ch_i],
            kernel[:, :, ch_i],
            mode="same",
        )
        weight_renorm[weight_renorm == 0] = np.inf
        conv_signal[:, :, ch_i] = (
            convolve(
                signal[:, :, ch_i],
                kernel[:, :, ch_i],
                mode="same",
            )
            / weight_renorm
        )
        conv_variance[:, :, ch_i] = (
            convolve(
                weights[:, :, ch_i],
                kernel_square[:, :, ch_i],
                mode="same",
            )
            / weight_renorm**2
        )
    conv_variance[conv_variance == 0] = np.inf
    conv_weights = 1 / conv_variance
    conv_weights = np.transpose(conv_weights, axes=axes)
    conv_signal = np.transpose(conv_signal, axes=axes)
    return conv_signal, conv_weights


def get_beam_xy(wproj, xdim, ydim):
    """
    Get the x and y angular coordinates of the given wcs.
    """
    x_cen, y_cen = xdim // 2, ydim // 2
    ra_cen, dec_cen = get_wcs_coor(
        wproj,
        x_cen,
        y_cen,
    )
    xx, yy = np.meshgrid(np.arange(xdim), np.arange(ydim), indexing="ij")
    ra, dec = get_wcs_coor(wproj, xx, yy)
    vec = hp.ang2vec(ra, dec, lonlat=True)
    vec_cen = hp.ang2vec(ra_cen, dec_cen, lonlat=True)
    xx = np.arcsin(vec - vec_cen[None, None, :])[:, :, 0].T * 180 / np.pi
    yy = np.arcsin(vec - vec_cen[None, None, :])[:, :, 1].T * 180 / np.pi
    return xx, yy


@tagging("anisotropic")
def kat_beam(nu, wproj, xdim, ydim, band="L"):
    r"""
    Returns a beam model from the ``katbeam`` model, which is a simplification of
    the model reported in Asad et al. [1].
    The katbeam implementation here still needs validation. Use it
    with caution, especially if you want correct orientation of the beam.

    References
    ----------
    .. [1] Asad et al., "Primary beam effects of radio astronomy antennas -- II. Modelling the MeerKAT L-band beam", https://arxiv.org/abs/1904.07155
    """
    from katbeam import JimBeam

    xx, yy = get_beam_xy(
        wproj,
        xdim,
        ydim,
    )
    beam = JimBeam(f"MKAT-AA-{band}-JIM-2020")
    freqMHz = nu / 1e6
    beam_image = np.zeros((xdim, ydim, len(nu)))
    for i, freq in enumerate(freqMHz):
        beam_image[:, :, i] = beam.I(xx, yy, freq) ** 2
    return beam_image


@tagging("isotropic")
def gaussian_beam(sigma):
    r"""
    Returns a Gaussian beam function

    .. math::
        B(\theta) = {\rm exp}[-\frac{\theta^2}{2\sigma^2}]

    when the beam width :math:`\sigma` is specified.

    Parameters
    ----------
        sigma: float.
            The width of the gaussian beam profile.
    Returns
    -------
        beam_func: function.
            The beam function.
    """
    return lambda x: np.exp(-(x**2) / 2 / sigma**2)


@tagging("isotropic")
def cos_beam(sigma):
    r"""
    Returns a cosine-tapered beam function [1]

    .. math::
        B(\theta) = \bigg[
        \frac{\cos \big( 1.189 \pi \theta / \theta_b \big)}
        {1-4\big( 1.189 \theta / \theta_b \big)}
        \bigg]^2

    for given input parameter :math:`\sigma`, the FWHM is set to
    :math:`\theta_b = 2\sqrt{2{\rm log}2 \sigma}`.

    Parameters
    ----------
        sigma: float.
            The width of the beam profile.

    Returns
    -------
        beam_func: function.
            The beam function.

    References
    ----------
    .. [1] Mauch et al., "The 1.28 GHz MeerKAT DEEP2 Image", https://arxiv.org/abs/1912.06212
    """
    theta_b = 2 * np.sqrt(2 * np.log(2)) * sigma

    def beam_func(ang_dist):
        beam = (
            np.cos(1.189 * ang_dist * np.pi / theta_b)
            / (1 - 4 * (1.189 * ang_dist / theta_b) ** 2)
        ) ** 2
        return beam

    return beam_func


def isotropic_beam_profile(
    xdim,
    ydim,
    wproj,
    beam_func,
    ang_unit=units.deg,
):
    """
    Generate an isotropic image of the beam for given ``wproj`` and ``beam_func``. The image can later be used to convolve or deconvolve with intensity maps.

    Parameters
    ----------
        xdim: int.
            The number of pixels in the first axis.
        ydim: int.
            The number of pixels in the second axis.
        wproj: :class:`astropy.wcs.WCS` object.
            The two-dimensional wcs object for the map.
        beam_func: function.
            The beam function.
        ang_unit: str or :class:`astropy.units.Unit`.
            The unit of input values for ``beam_func``.
    Returns
    -------
        beam_image: float array.
            The image of the beam.
    """
    xx, yy = np.meshgrid(np.arange(xdim), np.arange(ydim), indexing="ij")
    ra, dec = get_wcs_coor(wproj, xx, yy)
    ra_cen, dec_cen = get_wcs_coor(wproj, xdim // 2, ydim // 2)
    ang_dist = (
        (get_ang_between_coord(ra, dec, ra_cen, dec_cen) * units.deg).to(ang_unit).value
    )
    beam_image = beam_func(ang_dist)
    return beam_image


def gaussian_beam_window(sigma_rad, lmax):
    r"""
    Closed-form Gaussian beam window function in spherical-harmonic space.

    .. math::

        B_\ell = \exp\left(-\frac{\ell(\ell+1)\sigma^2}{2}\right),

    where :math:`\sigma` is the Gaussian beam dispersion in radians.

    Parameters
    ----------
    sigma_rad : float
        Gaussian beam dispersion in radians.
    lmax : int
        Maximum multipole; the returned array has length ``lmax + 1``.

    Returns
    -------
    b_ell : ndarray of shape ``(lmax + 1,)``
    """
    lmax_i = int(lmax)
    ell = np.arange(lmax_i + 1, dtype=np.float64)
    sigma = float(sigma_rad)
    return np.exp(-0.5 * ell * (ell + 1.0) * sigma**2)


def isotropic_beam_window(beam_func, sigma, lmax, theta_grid=None):
    r"""
    Numerical beam window function for an isotropic beam via ``healpy.sphtfunc.beam2bl``.

    The beam profile ``beam_func(theta)`` is sampled on a fine ``theta`` grid
    covering :math:`[0, \theta_{\rm max}]` with :math:`\theta_{\rm max} \approx 8\sigma`
    (truncated at :math:`\pi`), and the resulting :math:`B(\theta)` is converted
    to :math:`B_\ell` with ``hp.sphtfunc.beam2bl``.

    Parameters
    ----------
    beam_func : callable
        Maps ``theta`` (radians or the ``ang_unit`` chosen by the caller's beam
        factory) to beam amplitude. Must accept vector inputs.
    sigma : float
        Beam dispersion in the same angular units as ``beam_func`` expects.
        Used to pick the default ``theta_grid`` when none is provided.
    lmax : int
        Maximum multipole.
    theta_grid : ndarray, optional
        Explicit ``theta`` sample points (same units as expected by
        ``beam_func``). Must start at 0 and be monotonically increasing.

    Returns
    -------
    b_ell : ndarray of shape ``(lmax + 1,)``
    """
    lmax_i = int(lmax)
    if theta_grid is None:
        theta_max = min(8.0 * float(sigma), np.pi)
        theta_grid = np.linspace(0.0, theta_max, 1024)
    theta_grid = np.asarray(theta_grid, dtype=np.float64)
    if theta_grid[0] != 0:
        raise ValueError("theta_grid must start at 0.")
    profile = np.asarray(beam_func(theta_grid), dtype=np.float64)
    return hp.sphtfunc.beam2bl(profile, theta_grid, lmax_i)


def weighted_smoothing_healpix(
    data_pix,
    weights_pix,
    beam_window_ch,
    hp_nside,
    pixel_id,
    kernel_renorm=True,
    nside_out=None,
    pixel_id_out=None,
):
    r"""
    Weighted HEALPix smoothing, mirroring :func:`weighted_convolution` semantics.

    For each channel :math:`c` with beam window :math:`B^{(c)}_\ell`, this
    returns

    .. math::

        \tilde s^{(c)} = \frac{\mathcal{S}[s^{(c)} w^{(c)} \, B^{(c)}_\ell]}
                               {\mathcal{S}[w^{(c)} \, B^{(c)}_\ell]},\qquad
        \tilde w^{(c)} = \frac{\bigl(\mathcal{S}[w^{(c)} B^{(c)}_\ell]\bigr)^2}
                               {\mathcal{S}[w^{(c)} (B^{(c)}_\ell)^2]},

    where :math:`\mathcal{S}` denotes harmonic-space smoothing via
    ``hp.smoothing(..., beam_window=B_\ell)`` on a full-sphere scratch buffer
    (allocated per channel, not kept).

    Parameters
    ----------
    data_pix : (n_pix, n_ch) ndarray
    weights_pix : (n_pix, n_ch) ndarray
    beam_window_ch : (n_ch, lmax+1) ndarray or (lmax+1,) ndarray
        Per-channel beam window. A 1D array is broadcast over all channels.
    hp_nside : int
        HEALPix :math:`N_{\rm side}` that ``pixel_id`` refers to.
    pixel_id : ndarray of int
        HEALPix indices at ``hp_nside``, matching rows of ``data_pix`` /
        ``weights_pix``, used to paint the high-resolution sphere before smoothing.
    pixel_id_out : ndarray of int, optional
        If ``nside_out`` is below ``hp_nside``, indices at ``nside_out`` where outputs
        are sampled (survey footprint). Required in that case. When no downgrade,
        defaults to ``pixel_id``.
    kernel_renorm : bool, default True
        Kept for API symmetry with :func:`weighted_convolution`. ``B_\ell`` is
        already normalised by ``hp.smoothing`` (``B_0 = 1`` for Gaussian
        windows), so this flag is informational only.
    nside_out : int, optional
        If set to a HEALPix :math:`N_{\\rm side}` strictly less than ``hp_nside``,
        harmonic smoothing is carried out at ``hp_nside`` on full spheres, maps are
        ``hp.ud_grade`` down to ``nside_out``, then values are sampled at
        ``pixel_id_out``.

    Returns
    -------
    conv_data_pix : (n_pix_out, n_ch) ndarray
    conv_weights_pix : (n_pix_out, n_ch) ndarray
    """
    del kernel_renorm
    nside_i = int(hp_nside)
    if nside_out is None:
        nside_o = nside_i
    else:
        nside_o = int(nside_out)
        if nside_o >= nside_i:
            raise ValueError("nside_out must be no greater than hp_nside.")
        if nside_i % nside_o != 0:
            raise ValueError(
                f"hp_nside={nside_i} must be divisible by nside_out={nside_o} "
                "(standard HEALPix downgrade)."
            )
    npix_full = hp.nside2npix(nside_i)
    pid = np.asarray(pixel_id, dtype=np.int64).ravel()
    data_pix = np.asarray(data_pix)
    weights_pix = np.asarray(weights_pix)
    if data_pix.ndim != 2 or weights_pix.ndim != 2:
        raise ValueError(
            "data_pix and weights_pix must be 2D arrays of shape (n_pix, n_ch); "
            f"got {data_pix.shape} and {weights_pix.shape}."
        )
    if data_pix.shape != weights_pix.shape:
        raise ValueError(
            f"data/weights shape mismatch: {data_pix.shape} vs {weights_pix.shape}."
        )
    n_pix, n_ch = data_pix.shape
    if n_pix != pid.size:
        raise ValueError(f"pixel_id length {pid.size} does not match n_pix={n_pix}.")
    if nside_o == nside_i:
        pid_out = pid
    else:
        if pixel_id_out is None:
            raise ValueError(
                "pixel_id_out is required when nside_out is less than hp_nside."
            )
        pid_out = np.asarray(pixel_id_out, dtype=np.int64).ravel()
    n_pix_out = int(pid_out.size)
    bwin = np.asarray(beam_window_ch, dtype=np.float64)
    if bwin.ndim == 1:
        bwin = np.broadcast_to(bwin, (n_ch, bwin.size))
    elif bwin.ndim == 2:
        if bwin.shape[0] != n_ch:
            raise ValueError(
                f"beam_window_ch first axis must equal n_ch={n_ch}; got {bwin.shape[0]}."
            )
    else:
        raise ValueError("beam_window_ch must be 1D or 2D.")
    real_dtype = np.result_type(data_pix.dtype, weights_pix.dtype, np.float64)
    conv_data = np.zeros((n_pix_out, n_ch), dtype=real_dtype)
    conv_weights = np.zeros((n_pix_out, n_ch), dtype=real_dtype)
    for ci in range(n_ch):
        s = np.asarray(data_pix[:, ci], dtype=np.float64)
        w = np.asarray(weights_pix[:, ci], dtype=np.float64)
        b_ell = np.asarray(bwin[ci], dtype=np.float64)
        b_ell_sq = b_ell**2
        sw_full = np.zeros(npix_full, dtype=np.float64)
        w_full = np.zeros(npix_full, dtype=np.float64)
        np.add.at(sw_full, pid, s * w)
        np.add.at(w_full, pid, w)
        smoothed_sw = hp.smoothing(
            sw_full,
            beam_window=b_ell,
            iter=0,
            pol=False,
            use_weights=False,
        )
        smoothed_w = hp.smoothing(
            w_full,
            beam_window=b_ell,
            iter=0,
            pol=False,
            use_weights=False,
        )
        smoothed_w_sq = hp.smoothing(
            w_full,
            beam_window=b_ell_sq,
            iter=0,
            pol=False,
            use_weights=False,
        )
        if nside_o != nside_i:
            smoothed_sw = hp.ud_grade(smoothed_sw, nside_o, order_in="RING")
            smoothed_w = hp.ud_grade(smoothed_w, nside_o, order_in="RING")
            smoothed_w_sq = hp.ud_grade(smoothed_w_sq, nside_o, order_in="RING")
        denom = smoothed_w[pid_out]
        num = smoothed_sw[pid_out]
        with np.errstate(divide="ignore", invalid="ignore"):
            cdata = np.where(np.abs(denom) > 0, num / denom, 0.0)
        denom_sq = smoothed_w_sq[pid_out]
        with np.errstate(divide="ignore", invalid="ignore"):
            cvar = np.where(np.abs(denom_sq) > 0, denom_sq / denom**2, np.inf)
            cvar = np.where(denom != 0, cvar, np.inf)
        cw = np.where(np.isfinite(cvar) & (cvar > 0), 1.0 / cvar, 0.0)
        conv_data[:, ci] = cdata
        conv_weights[:, ci] = cw
    return conv_data, conv_weights


def dish_beam_sigma(dish_diameter, nu, gamma=1.0, ang_unit=units.deg):
    r"""
    Calculate the beam size of a dish telescope assuming

    .. math::
        \theta_{\rm FWHM} = \gamma \frac{\lambda}{D},

    where :math:`\theta_{\rm FWHM}` is the FWHM of the beam,
    :math:`\gamma` is the aperture efficiency,
    :math:`\lambda` is the observing wavelength,
    and D is the dish diameter.

    The sigma of the Gaussian beam is then
    :math:`\sigma = \theta_{\rm FWHM}/ 2\sqrt{2 {\rm ln}2}`.

    Parameters
    ----------
        dish_diameter: float.
            The diameter of the dish in metre.
        nu: float.
            The observing frequency in Hz.
        gamma: float, default 1.0.
            The aperture efficiency.
        ang_unit: str or :class:`astropy.units.Unit`, default ``deg``.
            The unit of the output.
    Returns
    -------
        beam_sigma: float.
            The sigma of the beam.
    """
    beam_fwhm = (
        constants.c / (nu * units.Hz * dish_diameter * units.m) * units.rad
    ).to(ang_unit).value * gamma
    beam_sigma = beam_fwhm / (2 * np.sqrt(2 * np.log(2)))
    return beam_sigma


def cmb_temperature(nu, tcmb0=Planck18.Tcmb0.value):
    """
    Calculate the background CMB temperature at given frequencies.

    Parameters
    ----------
        nu: float.
            The observing frequency in Hz.
        tcmb0: float, default ``Planck18.Tcmb0.value``.
            The background CMB temperature at z=0.
    Returns
    -------
        tcmb: float.
            The CMB temperature at given frequencies in Kelvin.
    """
    redshift = freq_to_redshift(nu)
    return tcmb0 * (1 + redshift)


def receiver_temperature_meerkat(nu):
    """
    The receiver temperature of MeerKAT.

    Parameters
    ----------
        nu: float.
            The observing frequency in Hz.
    Returns
    -------
        Trx: float.
            The receiver temperature at given frequencies in Kelvin.
    """
    Trx = 7.5 + 10 * (nu / 1e9 - 0.75) ** 2
    return Trx


def galaxy_temperature(nu, tgal_408MHz=25, sp_indx=-2.75):
    """
    The temperature template of the Milky Way.

    Note that, for an accurate T_sky, you can instead use
    :class:`meer21cm.fg.ForegroundSimulation` to generate the foregrounds.

    Parameters
    ----------
        nu: float.
            The observing frequency in Hz.
        tgal_408MHz: float.
            The average galaxy temperature at 408MHz in Kelvin.
        sp_indx: float.
            The spectral index to extrapolate it to input frequencies.
    Returns
    -------
        Tgal: float.
            The galaxy temperature at given frequencies in Kelvin.
    """
    Tgal = tgal_408MHz * (nu / 408 / 1e6) ** sp_indx
    return Tgal
