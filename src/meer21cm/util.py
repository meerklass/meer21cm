import numpy as np
from astropy import constants, units
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import healpy as hp
from astropy.io import fits
from astropy.cosmology import Planck18
import inspect
import sys, os
from scipy.special import erf
from scipy.interpolate import interp1d
from numpy.random import default_rng
from halomod.hod import HODBulk
from astropy.wcs import WCS
import re
from scipy.special import legendre

f_21 = 1420405751.7667  # in Hz
A_10 = 2.85 * 1e-15 / units.s
lamb_21 = (constants.c / f_21 * units.s).to("m")
# for HI galaxy
nhi_per_jyhzdl2 = 16 * np.pi / 3 / constants.h / (f_21 * units.Hz) / A_10
nhi_per_jyhzdl2 = nhi_per_jyhzdl2.to("Mpc-2 Jy-1 Hz-1")
# correct coeff for eq 45 in 1705.04210
mass_intflux_coeff = (
    (
        nhi_per_jyhzdl2
        * (constants.m_e + constants.m_p)
        * units.Mpc**2
        * units.Jy
        * units.Hz
    )
    .to("Msun")
    .value
)


class HiddenPrints:
    """
    A context manager that suppresses print statements.
    This is useful for suppressing print statements in the middle of a calculation.
    """

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def get_nd_slicer(ndim=3):
    """
    Get a list of slice objects that can be used to slice an ndarray.
    For example, if you have a list of k-vectors, then you can this to match
    the vectors for array index propagation.

    Parameters
    ----------
    ndim: int, default 3.
        The number of dimensions of the array to slice.

    Returns
    -------
    out: list of slice objects.
    """
    out = []
    for i in range(ndim):
        slice_i = [
            None,
        ] * ndim
        slice_i[i] = slice(None)
        slice_i = tuple(slice_i)
        out += [
            slice_i,
        ]
    return out


def create_wcs_with_range(
    ra_range,
    dec_range,
    resol=[0.3, 0.3],
    buffer=[1.2, 1.2],
    ctype=["RA---ZEA", "DEC--ZEA"],
):
    """
    Create a wcs object that can be used to map a index array to sky cooridnates,
    based on an approximate survey area specified by ``ra_range`` and ``dec_range``.

    A naive number of pixels will be calculated for both directions based on the
    difference between the lower and upper limit of the range, divided by the input
    resolution. It is then multiplied by the buffer because usually you need more than
    that due to curved sky.

    Parameters
    ----------
    ra_range: array-like of 2 elements.
        The lower and upper limit of the RA range in degree.
    dec_range: array-like of 2 elements.
        The lower and upper limit of the Dec range in degree.
    resol: array-like of 2 elements, default [0.3,0.3].
        The resolution along the two axis in degree.
    buffer: array-like of 2 elements, default [1.2,1.2].
        The proportional increase to the range.
    ctype: list of two strings, default ['RA---ZEA', 'DEC--ZEA'].
        The projection type.

    Returns
    -------
    w: :class:`astropy.wcs.WCS` object.
        The output wcs.

    num_pix_x: int.
        Number of pixels on the first axis.

    num_pix_y: int.
        Number of pixels on the second axis.
    """
    w = WCS(naxis=2)
    ra_min = ra_range[0]
    ra_max = ra_range[1]
    if ra_range[1] < ra_range[0]:
        ra_min -= 360
    ra_cen = (ra_min + ra_max) / 2
    dec_cen = (dec_range[0] + dec_range[1]) / 2
    ra_scale = ra_max - ra_min
    dec_scale = dec_range[1] - dec_range[0]
    num_pix_x = int(ra_scale / resol[0] * buffer[0])
    num_pix_y = int(dec_scale / resol[1] * buffer[1])
    crpix = [num_pix_x // 2, num_pix_y // 2]
    w.wcs.crpix = crpix
    w.wcs.cdelt = resol
    w.wcs.crval = [ra_cen, dec_cen]
    w.wcs.ctype = ctype
    return w, num_pix_x, num_pix_y


def create_wcs(
    ra_cr,
    dec_cr,
    ngrid,
    resol,
    crpix=None,
    ctype=["RA---ZEA", "DEC--ZEA"],
):
    """
    Create a wcs object that can be used to map a index array to sky cooridnates,
    for a given central coordinate and grid dimensions.

    Parameters
    ----------
    ra_cr: float.
        The coordinate of the critical pixel in RA in degree.
    dec_cr: float.
        The coordinate of the critical pixel in Dec in degree.
    ngrid: array-like of 2 elements.
        The number of pixels on the two axis.
    resol: array-like of 2 elements.
        The resolution along the two axis in degree.
    crpix: array-like of 2 elements, default None.
        The index of the critical pixel. Default is the center of the array.
    ctype: list of two strings, default ['RA---ZEA', 'DEC--ZEA'].
        The projection type.

    Returns
    -------
    w: :class:`astropy.wcs.WCS` object.
        The output wcs.
    """
    if type(resol) == float:
        resol = [resol, resol]
    if type(ngrid) == int:
        ngrid = [ngrid, ngrid]
    w = WCS(naxis=2)
    num_pix_x, num_pix_y = ngrid
    if crpix is None:
        crpix = [num_pix_x // 2, num_pix_y // 2]
    w.wcs.crpix = crpix
    w.wcs.cdelt = resol
    w.wcs.crval = [ra_cr, dec_cr]
    w.wcs.ctype = ctype
    return w


def angle_in_range(alpha, lower, upper):
    """
    Determines whether the angle alpha is in a range.
    All inputs in degree.
    Stolen from
    [this url](https://stackoverflow.com/questions/66799475/how-to-elegantly-find-if-an-angle-is-between-a-range).
    """
    # no selection at all
    if (upper - lower) % 360 == 0:
        return alpha <= np.inf
    return (alpha - lower) % 360 <= (upper - lower) % 360


def sample_map_from_highres(
    map_highres, ra_map, dec_map, wproj_lowres, num_pix_x, num_pix_y, average=True
):
    """
    grid a highres map into lowres.

    Parameters
    ----------
    map_highres: array
        The input high-res map.
    ra_map: array
        The RA coordinates of the input map pixels.
    dec_map: array
        The Dec coordinates of the input map pixels.
    wproj_lowres: :class:`astropy.wcs.WCS` object.
        The wcs object for the low-res map.
    num_pix_x: int
        The number of pixels along the first axis for the low-res map.
    num_pix_y: int
        The number of pixels along the second axis for the low-res map.
    average: bool, default True.
        Whether the sampling is an average of the high-res pixels (True)
        or the sum (False).
    """
    indx_1, indx_2 = radec_to_indx(ra_map, dec_map, wproj_lowres, to_int=False)
    indx_1 = indx_1.ravel()
    indx_2 = indx_2.ravel()
    map_lowres = np.zeros((num_pix_x, num_pix_y, map_highres.shape[-1]))
    indx_num = [num_pix_x, num_pix_y]
    indx_bins = [center_to_edges(np.arange(indx_num[i])) for i in range(2)]
    for indx_z in range(map_highres.shape[-1]):
        count, _ = np.histogramdd(np.array([indx_1, indx_2]).T, bins=indx_bins)
        map_i, _ = np.histogramdd(
            np.array([indx_1, indx_2]).T,
            bins=indx_bins,
            weights=map_highres[:, :, indx_z].ravel(),
        )
        if average:
            map_i /= count
        map_lowres[:, :, indx_z] = map_i
    return map_lowres


def create_udres_wproj(
    wproj,
    udres_scale,
):
    """
    Take an input wcs and create a wcs object that upgrades/downgrades the resolution.
    The ratio between the output resolution and input is determined by
    ``udres_scale``.

    Parameters
    ----------
    wproj: :class:`astropy.wcs.WCS` object.
        The input wcs.
    udres_scale: float
        The ratio between input and output resolution.

    Returns
    -------
    w: :class:`astropy.wcs.WCS` object.
        The output wcs.
    """
    w = WCS(naxis=2)
    w.wcs.crpix = wproj.wcs.crpix
    w.wcs.cdelt = wproj.wcs.cdelt / udres_scale
    w.wcs.crval = wproj.wcs.crval
    w.wcs.ctype = wproj.wcs.ctype
    return w


def super_sample_array(arr_in, super_factor):
    """
    Super-sample an array along each direction by a factor

    Parameters
    ----------
    arr_in: array.
        The input array to sample from.

    super_factor: list of int.
        super sample factor along each axis.

    Returns
    -------
    arr_out: array.
        The super-sampled array.
    """
    if arr_in is None:
        return None
    assert len(arr_in.shape) == len(super_factor)
    # maybe a faster approach is to use np.repeat
    arr_shape_in = arr_in.shape
    arr_shape_out = ()
    slice_indx = ()
    for i, dim in enumerate(arr_shape_in):
        arr_shape_out += (dim, super_factor[i])
        slice_indx += (slice(None), None)
    arr_out = np.zeros(arr_shape_out)
    arr_out += arr_in[slice_indx]
    arr_out = arr_out.reshape(np.array(arr_shape_in) * super_factor)
    return arr_out


def random_sample_indx(tot_len, num_sub_sample, seed=None):
    """
    Generate a random sub-sample indices.

    Parameters
    ----------
    tot_len: int.
        The total number of data points to sample from.

    num_sub_sample: int.
        Number of sub-samples.

    seed: int, default None.
        The seed for the random number generator.

    Returns
    -------
    sub_indx: array.
        The sub-sample indices.
    """
    rng = default_rng(seed)
    sub_indx = rng.choice(
        np.arange(tot_len),
        size=num_sub_sample,
        replace=False,
    )
    return sub_indx


def find_property_with_tags(obj):
    """
    Retrieve a dictionary for all the properties of a class that has tags.
    The keys of the dictionary are the property names and the values are the tags of each property.
    """
    func_dependency_dict = dict()
    for func in dir(type(obj)):
        if func[0] != "_":
            if isinstance(getattr(type(obj), func), property):
                if "tags" in dir(getattr(type(obj), func).fget):
                    func_tags = getattr(type(obj), func).fget.tags
                    func_dependency_dict.update({func: func_tags})
    return func_dependency_dict


def tagging(*tags):
    """
    A decorator that does one simple thing: Adding tags to functions.

    For example, you can add tags when defining this function

    .. highlight:: python
    .. code-block:: python

       from meer21cm.util import tagging
       @tagging('test')
       def foo(x):
          return x
       print(foo.tags)

    You will find that ``foo.tags`` is ``('test',)``. ``meer21cm`` uses this function to keep track of
    parameter dependencies of functions in classes.
    """

    def tagged_decorator(func):
        func.tags = tags
        return func

    return tagged_decorator


def center_to_edges(arr):
    """
    Extend a linear spaced monotonic array
    so that the original array is the middle point of the output array.
    """
    result = arr.copy()
    dx = np.diff(arr)
    result = np.append(result[:-1] - dx / 2, result[-2:] + dx[-2:] / 2)
    return result


def find_ch_id(nu_inp, nu_ch):
    r"""
    For which channel does the input frequency fall into.
    The channel ids are zero-indexed.
    Input frequencies outside the frequency range are assigned
    :math:`N_{\rm ch}` (note the last channel id is :math:`N_{\rm ch}-1`)

    Parameters
    ----------
    nu_inp: array.
        The input frequencies

    nu_ch: array.
        Must be monotonically increasing.
        The centre frequencies of each channel.

    Returns
    -------
    which_ch: int array.
        The ch ids.
    """
    nu_edges = center_to_edges(nu_ch)
    nu_edges_extend = center_to_edges(center_to_edges(nu_edges))
    which_ch = np.digitize(nu_inp, nu_edges_extend) - 2
    # first and last bins are out of range
    which_ch[which_ch < 0] = len(nu_ch)
    which_ch[which_ch > len(nu_ch)] = len(nu_ch)
    return which_ch


def coeff_hi_density_to_temp(z=0, cosmo=Planck18):
    r"""
    The conversion coefficient :math:`C_{\rm HI}` so that

    .. math::
        \bar{T}_{\rm HI} = C_{\rm HI} \rho_{\rm HI}

    Parameters
    ----------
    z: float, defulat 0.0.
        The redshift

    cosmo: cosmology, default Planck18.
        The cosmology used

    Returns
    -------
    C_HI: quantity.
        The coefficient
    """
    C_HI = (
        3
        * A_10
        * constants.h
        * constants.c**3
        * (1 + z) ** 2
        / 32
        / np.pi
        / (constants.m_e + constants.m_p)
        / constants.k_B
        / (f_21 * units.Hz) ** 2
        / cosmo.H(z)
    ).to(units.K / units.M_sun * units.Mpc**3)
    return C_HI


def omega_hi_to_average_temp(omega_hi, z=0, cosmo=Planck18):
    """
    The average HI brightness temperature from given HI density.

    Parameters
    ----------
    omega_hi:float
        The HI density relative to the z=0 critical density.

    z: float, defulat 0.0.
        The redshift

    cosmo: cosmology, default Planck18.
        The cosmology used

    Returns
    -------
    t_bar: float.
        The average HI temperature in Kelvin
    """
    c_hi = coeff_hi_density_to_temp(z=z, cosmo=cosmo)
    t_bar = (c_hi * cosmo.critical_density0 * omega_hi).to("K").value
    return t_bar


def freq_to_redshift(freq):
    """
    Convert frequency of 21cm emission to redshift

    Parameters
    ----------
    freq: float

    Returns
    -------
    redshift: float
    """
    return f_21 / freq - 1


def redshift_to_freq(redshift):
    """
    Convert redshift to frequency

    Parameters
    ----------
    redshift: float

    Returns
    -------
    freq: float
    """
    return f_21 / (1 + redshift)


def get_ang_between_coord(ra1, dec1, ra2, dec2, unit="deg"):
    """
    Calculate the angle between two points on the sphere.

    Parameters
    ----------
        ra1: float array
            The RA of the first point
        dec1: float array
            The Dec of the first point
        ra2: float array
            The RA of the second point
        dec2: float array
            The Dec of the second point
        unit: str, default 'deg'.
    Returns
    -------
        result: float array.
            The angle in the specified unit.

    """
    vec1 = hp.ang2vec(ra1, dec1, lonlat=True)
    vec2 = hp.ang2vec(ra2, dec2, lonlat=True)
    # extremely rarely, due to precision errors vec1*vec2 can be bigger than 1
    # triggering a nan in arccos
    v1v2cross = (vec1 * vec2).sum(axis=-1)
    v1v2cross[v1v2cross > 1] = 1
    result = (np.arccos(v1v2cross) * units.rad).to(unit).value
    return result.T


def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def read_healpix_fits(file):
    hp_map = hp.read_map(file)
    hp_nside = hp.get_nside(hp_map)
    with fits.open(file) as hdul:
        header = hdul[1].header
        map_unit = units.Unit(header["TUNIT1"])
        map_freq = units.Quantity(header["FREQ"]).to("Hz").value
    return hp_map, hp_nside, map_unit, map_freq


def get_wcs_coor(wcs, xx, yy, ang_unit="deg"):
    """
    Retrieve RA and Dec coordinates of pixels in the WCS.

    Parameters
    ----------
        wproj: :class:`astropy.wcs.WCS` object.
            The two-dimensional wcs object for the map.
        xx: array
            The first indx of the pixel.
        yy: array
            The second indx of the pixel.

    Returns
    -------
        ra: array.
            The RA of the pixels.
        dec: array.
            The Dec of the pixels.
    """
    assert wcs.naxis == 2, "input wcs must be 2-dimensional."
    coor = wcs.pixel_to_world(xx, yy)
    ra = getattr(coor.ra, ang_unit)
    dec = getattr(coor.dec, ang_unit)
    return ra, dec


def mean_center_signal(signal, weights=None, los_axis=-1):
    """
    Mean-center the signal. Assume the first axis is the LOS axis.
    """
    assert len(signal.shape) == 3, "map must be 3D."
    if los_axis < 0:
        # change -1 to 2
        los_axis = 3 + los_axis
    # make sure los is the fist axis
    axes = [0, 1, 2]
    axes.remove(los_axis)
    axes = [
        los_axis,
    ] + axes
    # transpose map data
    signal = np.transpose(signal, axes=axes)
    nz, nx, ny = signal.shape
    signal = signal.reshape((nz, -1))
    if weights is not None:
        weights = np.transpose(weights, axes=axes)
        weights = weights.reshape((nz, -1))
    else:
        weights = np.ones_like(signal)
    signal = signal - np.sum(signal * weights, 1)[:, None] / np.sum(weights, 1)[:, None]
    signal = signal.reshape((nz, nx, ny))
    signal = np.transpose(signal, axes=np.argsort(axes))
    return signal


def weighted_covariance(signal, weights, renorm=True):
    """
    Calculate the weighted covariance matrix of the signal.

    The signal is assumed to be mean-centered, with the shape of
    (num_ch, num_pix).

    Parameters
    ----------
        signal: array.
            The input signal
        weights: array.
            The weights of each element in the signal.
        renorm: bool, default True.
            Whether to renormalize the covariance matrix
            over the sum of weights.

    Returns
    -------
        covariance: array.
            The covariance matrix of the input signal.
    """
    covariance = np.einsum("ia,ja->ij", signal * weights, signal * weights)
    if renorm:
        covariance /= np.einsum("ia,ja->ij", weights, weights)
    return covariance


def pca_clean(
    signal,
    N_fg,
    weights=None,
    return_analysis=False,
    mean_center=False,
    los_axis=-1,
    return_A=False,
    mean_center_weights=None,
    ignore_nan=False,
    covariance=None,
):
    r"""
    Performs PCA cleaning of the map data. If ``mean_center`` is set to ``True``,
    then the input signal is first mean-centered so that

    .. math::
        \vec{d}^{\rm mc} = \vec{d} - \langle  \vec{d} \rangle,

    where the mean of the data vector at a specific channel i is

    .. math::
        \langle \vec{d} \rangle_i = \frac{\sum_a w_{ia} d_{ia}}{\sum_a w_{ia}},

    where a loops over each sampling in channel i and :math:`w` is the weight of each element.

    A covariance matrix is then calculated on the mean-centered data

    .. math::
        C_{ij} = \frac{\sum_{a,b }w_{ia} d^{\rm mc}_{ia}
             w_{jb} d^{\rm mc}_{jb}}{\sum_{a,b }w_{ia}w_{jb}},

    over which the eigenvalue decomposition is performed.
    See Section 4.3. of MeerKLASS Collaboration 2024 [1] for more details.

    Note that, in the rigorous definition, the data vector should be zero-meaned,
    and the weights used to calculate the mean and the covariance should be the same.
    However, in practice, many people don't remove the mean of the data.
    Some use one type of weights (often just binary masks) for mean calculation, and then
    use different weights for covariance calculation. While it is not encouraged, that flexibility
    is allowed in the function, by setting a different weight ``mean_center_weights``.

    If there are frequency gaps in the data, you can set ``ignore_nan=True`` to ignore these
    channels, and treat the rest of the data as a continuous spectrum.

    Parameters
    ----------
        signal: array
            The input signal to be cleaned.
        N_fg: int.
            Number of modes to be removed.
        weights: array, default None
            The weights of each element in the signal.
            Default will set uniform weights for each element.
        return_analysis: bool, default False
            If True, instead of residual maps the function will return eigenanalysis quantities.
        mean_center: bool, default False
            Whether to mean-center the input data vector
        los_axis: int, default -1.
            Which axis is the line-of-sight, i.e. spectral axis.
        return_A: bool, default False.
            Whether to return the mixing matrix A.
        mean_center_weights: array, default None.
            The weights of each element for mean calculation.
            Default follows the ``weights`` argument.
        ignore_nan: bool, default False.
            Whether to ignore NaN values in the data.
            If True, channels with NaN values are skipped.
        covariance: array, default None.
            The covariance matrix of the input signal.
            If not provided, it will be calculated from the input signal.

    Returns
    -------
        residual: array.
            The residual after PCA cleaning.
        A: array, if ``return_A=True``.
            The mixing matrix.
        covariance: array, if ``return_analysis=True``.
            The covariance matrix of the input signal.
        eignumb: array, if ``return_analysis=True``.
            The number indexing of the eigenvalues starting from 1.
        eigenval: array, if ``return_analysis=True``.
            The eigenvalues of the covariance matrix.
        V: array, if ``return_analysis=True``.
            The eigenvectors of the covariance matrix.

    References
    ----------
    .. [1] MeerKLASS collab, "MeerKLASS L-band deep-field intensity maps: entering the HI dominated regime", https://arxiv.org/abs/2407.21626.
    """
    if mean_center:
        if mean_center_weights is None:
            signal = mean_center_signal(signal, weights, los_axis)
        else:
            signal = mean_center_signal(signal, mean_center_weights, los_axis)
    assert len(signal.shape) == 3, "map must be 3D."
    if los_axis < 0:
        # change -1 to 2
        los_axis = 3 + los_axis
    # make sure los is the fist axis
    axes = [0, 1, 2]
    axes.remove(los_axis)
    axes = [
        los_axis,
    ] + axes
    # transpose map data
    signal = np.transpose(signal, axes=axes)
    nz, nx, ny = signal.shape
    signal = signal.reshape((nz, -1))
    if weights is not None:
        weights = np.transpose(weights, axes=axes)
        weights = weights.reshape((nz, -1))
    else:
        weights = np.ones_like(signal)

    if mean_center_weights is not None:
        mean_center_weights = np.transpose(mean_center_weights, axes=axes)
        mean_center_weights = mean_center_weights.reshape((nz, -1))
    # if mean_center:
    #    if mean_center_weights is None:
    #        signal = (
    #            signal
    #            - np.sum(signal * weights, 1)[:, None] / np.sum(weights, 1)[:, None]
    #        )
    #    else:
    #        signal = (
    #            signal
    #            - np.sum(signal * mean_center_weights, 1)[:, None]
    #            / np.sum(mean_center_weights, 1)[:, None]
    #        )
    ### Covariance calculation:
    if covariance is None:
        covariance = weighted_covariance(signal, weights)
    nan_flag = ignore_nan and np.any(np.isnan(covariance))
    if nan_flag:
        sel = np.logical_not(np.isnan(np.diagonal(covariance)))
        covariance = covariance[sel][:, sel]
        signal = np.nan_to_num(signal)
    eigenval, V = np.linalg.eigh(covariance)
    if nan_flag:
        eigenvec = np.zeros((nz, nz)) * np.nan
        eigenvec[np.where(sel)[0][:, None], np.where(sel)[0]] = V
        V = eigenvec
        eval = np.zeros(nz) * np.nan
        eval[sel] = eigenval
        eigenval = eval
    V = V[:, ::-1]  # Eigenvectors from covariance matrix with most dominant first
    if return_analysis:
        eignumb = np.linspace(1, len(eigenval), len(eigenval))
        eigenval = eigenval[::-1]  # Put largest eigenvals first
        return covariance, eignumb, eigenval, V
    A = V[:, :N_fg]  # Mixing matrix, first N_fg most dominant modes of eigenvectors
    R_pca = np.eye(nz) - A @ A.T
    if nan_flag:
        R_pca = np.nan_to_num(R_pca)
    residual = R_pca @ signal
    residual = np.reshape(residual, (nz, nx, ny))
    residual = np.transpose(residual, axes=np.argsort(axes))
    if return_A:
        return residual, A
    return residual


def radec_to_indx(ra_arr, dec_arr, wproj, to_int=True, ang_unit="deg"):
    coor = SkyCoord(ra_arr, dec_arr, unit=ang_unit)
    indx_1, indx_2 = wproj.world_to_pixel(coor)
    if to_int:
        indx_1 = np.round(indx_1).astype("int")
        indx_2 = np.round(indx_2).astype("int")
    return indx_1, indx_2


def convert_hpmap_in_jy_to_temp(hp_map, freq):
    nside = hp.get_nside(hp_map)
    hp_map = jy_to_kelvin(hp_map, hp.nside2resol(nside), freq)
    return hp_map


def healpix_to_wcs(hp_map, xx, yy, wcs):
    """
    Project the healpix map onto a 2-D wcs grid.
    Map has to be in temperature unit.
    """
    nside = hp.get_nside(hp_map)
    ra_map, dec_map = get_wcs_coor(wcs, xx, yy)
    pix_indx = hp.ang2pix(nside, ra_map, dec_map, lonlat=True)
    output_map = hp_map[pix_indx]
    return output_map


def rebin_spectrum(spectrum, rebin_width=3, mode="avg"):
    assert spectrum.size % 2 == 1
    assert rebin_width % 2 == 1
    rebin_pad = spectrum.size % rebin_width
    if rebin_pad % 2 == 1:
        rebin_pad += rebin_width
    rebin_pad = rebin_pad // 2
    spectrum_rebin = (
        spectrum[rebin_pad:-rebin_pad].reshape((-1, rebin_width)).mean(axis=-1)
    )
    if mode == "sum":
        spectrum_rebin *= rebin_width
    return spectrum_rebin


def hod_obuljen18(
    logmh,
    m0h=9.52,
    mminh=11.27,
    alpha=0.44,
    cosmo=Planck18,
    input_has_h=True,
    output_has_h=False,
):
    r"""
    HI-halo mass relation reported in Obuljen et al. (2018) [1].
    The default settings assume the mass is in M_sun/h, and returns mass **without** h unit.
    Note that, if you want input without h unit, you must change the ``m0h`` and ``mminh`` manually as well.
    The HI-halo mass relation follows

    .. math::
        M_{\rm HI} (M_h) = M_0 (M_h/M_{\rm min})^\alpha \, {\rm exp}[-M_{\rm min}/M_h],



    Parameters
    ----------
        logmh: float.
            input halo mass in log10
        m0h: optional, default 9.52.
            The :math:`M_0` parameter in the HOD in log10
        mminh: optional, default 11.27.
            The :math:`M_{min}` parameter in the HOD in log10
        alpha: optional, default 0.44.
            The :math:`\alpha` parameter in the HOD.
        cosmo: optional, default Planck18.
            The default cosmology, used only for h unit.
        input_has_h: optional, default True.
            Whether the input mass has h unit.
        output_has_h: optional, default False.
            Whether the output mass has h unit.

    Returns
    -------
        The HI mass for the input halo mass.

    References
    ----------
    .. [1] Obuljen, A. et al., "The H I content of dark matter haloes at z ≈ 0 from ALFALFA ", https://ui.adsabs.harvard.edu/abs/2019MNRAS.486.5124O.
    """
    input_h_unit = 1.0 if input_has_h else cosmo.h
    output_h_unit = 1.0 if output_has_h else 1 / cosmo.h
    marr = 10**logmh * input_h_unit  # in Msun/h
    himass = 10 ** (m0h) * (marr / 10**mminh) ** alpha * np.exp(-(10**mminh) / marr)
    return himass * output_h_unit


def check_unit_equiv(u1, u2):
    """
    Check if two units are equivelant

    Parameters
    ----------
        u1: ``astropy.units.Unit`` object.
            The first input unit.
        u2: ``astropy.units.Unit`` object.
            The second input unit.

    Returns
    -------
        result: bool.
            Whether they are the same unit.
    """
    return (1 * u1 / u2).si.unit == units.dimensionless_unscaled


def jy_to_kelvin(val, omega, freq):
    """
    convert Jy/beam to brightness temperature in Kelvin.

    Parameters
    ----------
        val: numpy array.
            The input value(s) in Jy/beam or Jy/pix
        omega: float.
            beam or pixel area in Steradian.
        freq: float.
            the frequency for conversion in Hz.

    Returns
    -------
        result: float array.
            The brightness temperature in Kelvin.
    """
    freq = freq * units.Hz
    omega = omega * units.sr
    result = (
        (val * units.Jy / omega)
        .to(units.K, equivalencies=units.brightness_temperature(freq))
        .value
    )
    return result


def busy_function_simple(xarr, par_a, par_b, par_c, width):
    r"""
    The simplified busy function that assumes mirror symmetry around x=0 [1].


    .. math::
       B_2(x) = \frac{a}{2} \times ({\rm erf}[b(w^2-x^2)]+1) \times (cx^2+1)


    Parameters
    ----------
        xarr: float array.
            the input x values
        par_a: float.
            amplitude parameter
        par_b: float.
            b parameter that controls the sharpness of the double peaks
        par_c: float.
            c parameter that controls the height of the double peaks
        width: float.
            the width of the profile

    Returns
    -------
        b2x: float array.
            the busy function values at xarr

    References
    ----------
    .. [1] Westmeier, T. et al., "The busy function: a new analytic function for describing the integrated 21-cm spectral profile of galaxies",
           https://ui.adsabs.harvard.edu/abs/arXiv:1311.5308 .
    """
    b2x = (
        par_a
        / 2
        * (erf(par_b * (width**2 - xarr**2)) + 1)
        * (par_c * xarr**2 + 1)
    )
    return b2x


def find_indx_for_subarr(subarr, arr):
    """
    Find the indices of the elements of an array in another array.

    Parameters
    ----------
        subarr: numpy array.
            The sub-array to search for. Elements can be repeated.
        arr: numpy array.
            The larger array that contains all elements of subarr.
    Returns
    -------
        indices: int array.
            The position of the elements of subarr in arr.
    """
    assert np.unique(arr).size == arr.size, "the larger array must be unique"
    # Actually preform the operation...
    arrsorted = np.argsort(arr)
    subpos = np.searchsorted(arr[arrsorted], subarr)
    indices = arrsorted[subpos]
    return indices


def himf(m, phi_s, m_s, alpha_s):
    r"""
    Analytical HIMF function (or any other Schechter function).

    .. math::
       \phi = {\rm ln}(10) \times \phi_* \times \bigg(\frac{m}{m_*} \bigg)^{\alpha_*+1}\times e^{-m/m_*}

    While the units are arbitrary, it is recommended that
    phi_s is in the unit of Mpc:sup:`-3`dex:sup:`-1`,
    m_s is in the unit of log10 M_sun,
    and alpha_s has no unit.


    Parameters
    ----------
        m: float array.
            mass in log10.
        phi_s: float.
            HIMF amplitude
        m_s: float.
            knee mass in log10.
        alpha_s: float.
            slope

    Returns
    -------
        out: float array.
            The HIMF values at m
    """
    out = (
        np.log(10)
        * phi_s
        * (10**m / 10**m_s) ** (alpha_s + 1)
        * np.exp(-(10**m) / 10**m_s)
    )
    return out


def cal_himf(x, mmin, cosmo, mmax=11, integrate_step=500):
    """
    Calculate the integrated quantity related to the HIMF.

    Parameters
    ----------
        x: list of float.
            need to be [phi_s,log10(m_s),alpha_s]
        mmin: float.
            The minimum mass to integrate from in log10
        cosmo: an :obj:`astropy.cosmology.Cosmology` object.
            The cosmology object to calculate critical density
        mmax: Optional float, default 11.
            The maximum mass to integrate to in log10.
        integrate_step: Optional int, default 500.
            The number of steps to integrate the HIMF.

    Returns
    -------
        nhi: float.
            The number density of HI galaxies, in the units of phi_s * dex
        omegahi: float.
            The HI density over the critical density of the present day (assuming the recommended units for x are used)
        psn: float.
            The shot noise in the units of Mpc:sup:`3` (assuming the recommended units for x are used)
    """
    marr = np.logspace(mmin, mmax, num=integrate_step)
    omegahi = (
        (
            np.trapz(himf(np.log10(marr), x[0], x[1], x[2]) * marr, x=np.log10(marr))
            * units.M_sun
            / units.Mpc**3
            / cosmo.critical_density0
        )
        .to("")
        .value
    )
    psn = (
        np.trapz(himf(np.log10(marr), x[0], x[1], x[2]) * marr**2, x=np.log10(marr))
        / np.trapz(himf(np.log10(marr), x[0], x[1], x[2]) * marr, x=np.log10(marr)) ** 2
    )
    nhi = np.trapz(himf(np.log10(marr), x[0], x[1], x[2]), x=np.log10(marr))
    return nhi, omegahi, psn


def himf_pars_jones18(h_70):
    """
    The HIMF parameters measured in Jones+18 [1].

    Parameters
    ----------
        h_70: float.
            The Hubble parameter over 70 km/s/Mpc.

    Returns
    -------
        phi_star: float.
            The amplitude of HIMF in Mpc-3 dex-1.

        m_star: float.
            The knee mass of HIMF in log10 solar mass.

        alpha: float.
            The slope of HIMF.

    References
    ----------
    .. [1] Jones, M. et al., "The ALFALFA HI mass function: A dichotomy in the low-mass slope and a locally suppressed 'knee' mass", https://ui.adsabs.harvard.edu/abs/arXiv:1802.00053.
    """
    phi_star = 4.5 * 1e-3 * h_70**3  # in Mpc-3 dex-1
    m_star = np.log10(10 ** (9.94) / h_70**2)  # in log10 Msun
    alpha = -1.25
    return phi_star, m_star, alpha


def cumu_nhi_from_himf(m, mmin, x):
    """
    The integrated source number density from HIMF.

    Parameters
    ----------
        m: float array.
            The higher end of integration in log10
        mmin: float.
            The minimum mass to integrate from in log10
        x: list of float.
            need to be [phi_s,log10(m_s),alpha_s]

    Returns
    -------
        nhi: float array.
            The integrated number density of HI galaxies, in the units of phi_s * dex
    """
    marr = np.logspace(mmin, m, num=500)
    nhi = np.trapz(himf(np.log10(marr), x[0], x[1], x[2]), x=np.log10(marr), axis=0)
    return nhi


def sample_from_dist(func, xmin, xmax, size=1, cdf=False, seed=None):
    """
    Sample from custom distribution.

    Parameters
    ----------
        func: distribution function.
            The probability distribution function (cdf=False) or the cumulative distribution function (cdf=True).
        xmin: float.
            The minimum value to sample from
        xmax: float.
            The maximum value to sample from
        size: int or list of int, default 1.
            The size of the sample array
        cdf: bool, default False.
            Wheter PDF or CDF is used.
        seed: int, default None.
            Seed for the random number generator. Fix for reproducible samples.

    Returns
    -------
        sample: float array.
            The random sample following the input distribution function.
    """
    xarr = np.linspace(xmin, xmax, 1001)
    if cdf is False:
        pdf_arr = func(xarr)
        cdf_arr = np.cumsum(pdf_arr)
    else:
        cdf_arr = func(xarr)
    cdf_arr -= cdf_arr[0]
    cdf_arr /= cdf_arr[-1]
    cdf_inv = interp1d(cdf_arr, xarr)
    rng = default_rng(seed=seed)
    sample = cdf_inv(rng.uniform(low=0, high=1, size=size))
    return sample


def tully_fisher(xarr, slope, zero_point, inv=False):
    """
    Tully-Fisher relation.

    Note that, **regardless of inv**, the slope and zero_point always refer to the Tully-Fisher relation
    and **not the inverse**.
    For example, zero_point is always in the unit of log10 mass.

    Parameters
    ----------
        xarr: float array.
            input velocity if inv=False and mass if inv=True.
        slope: float.
            the slope of Tully-Fisher relation.
        zero_point: float.
            the intercept of Tully-Fisher relation
        inv: bool, default False.
            if True, calculate velocity based on input mass.

    Returns
    -------
        out: float array.
            The output mass if inv=False and velocity if inv=True.
    """

    if inv:
        out = 10 ** ((np.log10(xarr) - zero_point) / slope)
    else:
        out = 10 ** (slope * np.log10(xarr) + zero_point)
    return out


class Obuljen18(HODBulk):
    """
    A :class:`halomod.hod.HODBulk` object for the HI-halo mass relation
    reported in Obuljen et al. (2018) [1].
    """

    _defaults = {
        "m0h": 9.52,
        "mminh": 11.27,
        "alpha": 0.44,
        "M_min": 9.0,
    }

    def _satellite_occupation(self, m):
        m0h = self.params["m0h"]
        mminh = self.params["mminh"]
        alpha = self.params["alpha"]
        himass = 10 ** (m0h) * (m / 10**mminh) ** alpha * np.exp(-(10**mminh) / m)
        return himass

    def sigma_satellite(self, m):
        return np.zeros_like(m)


def dft_matrix(N, norm="backward"):
    """
    Generate the DFT matrix for a given N.
    The default is the backward normalization, which is the same as np.fft.fft.

    Parameters
    ----------
        N: int
            The size of the matrix.
        norm: str, default 'backward'
            The normalization of the DFT matrix.

    Returns
    -------
        dft_mat: np.ndarray
            The DFT matrix.
    """
    return np.fft.fft(np.eye(N), norm=norm)


def inv_dft_matrix(N, norm="backward"):
    """
    Generate the inverse DFT matrix for a given N.
    The default is the backward normalization, which is the same as np.fft.fft.

    Parameters
    ----------
        N: int
            The size of the matrix.
        norm: str, default "backward"
            The normalization of the DFT matrix.

    Returns
    -------
        inv_dft_mat: np.ndarray
            The inverse DFT matrix.
    """
    return np.fft.ifft(np.eye(N), norm=norm)


def find_block_id(filename):
    """
    Find the MeerKAT data block id from the filename.
    MeerKAT data blocks are identified by a 10-digit number.
    Use `vfind_id` to find the block id for an array of filenames.

    Parameters
    ----------
        filename: str
            The filename to find the block id from.

    Returns
    -------
        block_id: str
            The block id.
    """
    reex = "[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]"
    result = re.findall(reex, filename)
    if result.count(result[0]) != len(result):
        raise ValueError("ambiguous block id from filename " + filename)
    result = result[0]
    return result


vfind_id = np.vectorize(find_block_id)


def legendre_polynomial_with_factor(ell: int, return_coeff: bool = True):
    """
    Return the Legendre polynomial with the given ell, with the (2*ell+1) factor.

    Parameters
    ----------
        ell: int
            The Legendre polynomial order.
        return_coeff: bool, default True
            Whether to return the coefficients or the polynomial or the callable function.

    Returns
    -------
        coeff: np.ndarray
            The coefficients of the Legendre polynomial.
        poly: np.poly1d
            The Legendre polynomial function.
    """
    coeff = legendre(ell).c * (2 * ell + 1)
    if return_coeff:
        return coeff
    return np.poly1d(coeff)
