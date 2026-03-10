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
