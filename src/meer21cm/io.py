"""
Module for reading and pre-processing MeerKLASS maps.
"""

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from .util import get_wcs_coor
import meer21cm.telescope as telescope
import pickle


def cal_freq(
    ch_id,
    band="L",
    nu_min=None,
    delta_nu=None,
):
    """
    returns the centre of the frequency channel for channel id `ch_id`
    of the meerkat telescope.

    Parameters
    ----------
        ch_id: int.
            The channel id.
        band: str, default 'L'.
            Frequency band, can either be 'L' or 'UHF'.
            Retrieves default MeerKAT setting.
            If `nu_min` and `delta_nu` are passed,
            the default settings are overridden.
        nu_min: float, default 856.0*1e6 Hz.
            The lower end of the frequency range.
        delta_nu: float, default 0.208984375*1e6 Hz.
            The channel bandwidth.

    Returns
    -------
        freq: float.
           The frequency of the channel.
    """
    if band == "":
        band = "L"
    if nu_min is None:
        nu_min = getattr(telescope, f"meerkat_{band}_band_nu_min")
    if delta_nu is None:
        delta_nu = getattr(telescope, f"meerkat_{band}_4k_delta_nu")
    return ch_id * delta_nu + nu_min


def filter_incomplete_los(
    map_intensity,
    map_has_sampling,
    map_weight,
    map_pix_counts,
    los_axis=-1,
    soft_mask=False,
    threshold_instead_of_filter=None,
):
    """
    Filter the map so that along the line-of-sight, only pixels that has sampling at every channel gets selected.

    If `soft_mask` is True, instead of filtering out incomplete los,
    the filtering is applied by checking the maximum sampling fraction along the los, and
    the pixels with less than the maximum sampling fraction are masked.

    If `threshold_instead_of_filter` is given, instead of filtering out incomplete los,
    the filtering is applied by checking the maximum sampling fraction along the los, and
    the pixels with less than the maximum sampling fraction are masked.

    Parameters
    ----------
        map_intensity: array.
            The input map.
        map_has_sampling: boolean array.
            Whether the pixel has measurement.
        map_weight: array.
            the pixel weights.
        map_pix_counts: array.
            The channel bandwidth.
        los_axis: int, default -1.
            which axis is the los.
        soft_mask: boolean, default False.
            whether to apply soft masking.
        threshold_instead_of_filter: float, default None.
            if given, instead of filtering out incomplete los,
            the filtering is applied by checking the sampling fraction along the los,
            and pixels with less than the threshold are masked.

    Returns
    -------
        map_intensity: array.
           map after pixels that have incomplete los sampling are masked.
        map_has_sampling: array.
           sampling after pixels that have incomplete los sampling are masked.
        map_weight: array.
           weights after pixels that have incomplete los sampling are masked.
        map_pix_counts: array.
           counts after pixels that have incomplete los sampling are masked.
    """
    if los_axis < 0:
        los_axis += 3
    axes = [0, 1, 2]
    axes.remove(los_axis)
    # make sure los is the last axis
    axes = axes + [
        los_axis,
    ]
    map_intensity = np.transpose(map_intensity, axes=axes)
    map_has_sampling = np.transpose(map_has_sampling, axes=axes)
    map_weight = np.transpose(map_weight, axes=axes)
    map_pix_counts = np.transpose(map_pix_counts, axes=axes)
    sampling_fraction = map_has_sampling.mean(axis=-1)
    if soft_mask:
        full_sample_los = sampling_fraction == sampling_fraction.max()
    elif threshold_instead_of_filter is not None:
        full_sample_los = sampling_fraction >= threshold_instead_of_filter
    else:
        full_sample_los = sampling_fraction == 1.0
    map_intensity *= full_sample_los[:, :, None]
    map_has_sampling *= full_sample_los[:, :, None]
    map_weight *= full_sample_los[:, :, None]
    map_pix_counts *= full_sample_los[:, :, None]

    # back to original shape
    map_intensity = np.transpose(map_intensity, axes=np.argsort(axes))
    map_has_sampling = np.transpose(map_has_sampling, axes=np.argsort(axes))
    map_weight = np.transpose(map_weight, axes=np.argsort(axes))
    map_pix_counts = np.transpose(map_pix_counts, axes=np.argsort(axes))
    return (
        map_intensity,
        map_has_sampling,
        map_weight,
        map_pix_counts,
    )


def read_pickle(
    pickle_file,
    nu_min=-np.inf,
    nu_max=np.inf,
    los_axis=-1,
    data_column="map",
    counts_column="hit",
    freq_column="freq",
    wcs_column="wcs",
):
    """
    Read pickle file of MeerKLASS UHF-band data into arrays.
    The file format requires the following keys:
    - map: the map data.
    - hit: the number of sampling for each pixel.
    - freq: the frequencies of each channel in the data in MHz. ``meer21cm`` then converts it to Hz.
    - wcs: the :class:`astropy.wcs.WCS` object for the map.

    Parameters
    ----------
        pickle_file: str.
            The input pickle file.
        nu_min: float, default -np.inf.
            The lower end of frequency cut.
            Channels below this frequency will be thrown away.
        nu_max: float, default np.inf.
            The higher end of freuqency cut.
            Channels above this frequency will be thrown away.
        los_axis: int, default -1.
            which axis is the los.
        data_column: str, default "map".
            The column name of the map data.
        counts_column: str, default "hit".
            The column name of the number of sampling for each pixel.
        freq_column: str, default "freq".
            The column name of the frequencies of each channel in the data.
        wcs_column: str, default "wcs".
            The column name of the :class:`astropy.wcs.WCS` object for the map.

    Returns
    -------
        map_data: array.
            The map data.
    """
    with open(pickle_file, "rb") as f:
        data = pickle.load(f)
    map_data = data[data_column]
    map_has_sampling = np.logical_not(map_data.mask)
    counts = data[counts_column]
    nu = data[freq_column] * 1e6  # MHz to Hz
    wproj = data[wcs_column]
    nu_sel = np.where((nu > nu_min) & (nu < nu_max))[0]
    nu_sel_min, nu_sel_max = nu_sel.min(), nu_sel.max()
    sel_indx = [
        slice(None, None, 1),
    ] * 3
    sel_indx[los_axis] = slice(nu_sel_min, nu_sel_max + 1, 1)
    sel_indx = tuple(sel_indx)
    nu = nu[nu_sel]
    # masked pixels are filled with 0
    # in case there is nan in the map
    map_data = np.nan_to_num(map_data.filled(0)[sel_indx])
    if isinstance(counts, np.ma.MaskedArray):
        counts = counts.filled(0)
    counts = counts[sel_indx]
    map_has_sampling = map_has_sampling[sel_indx]
    # in case there is inconsistency between hit and map_has_sampling
    map_has_sampling = (map_has_sampling * (counts > 0)).astype("bool")
    counts = counts * map_has_sampling
    if los_axis < 0:
        los_axis += 3
    axes = [0, 1, 2]
    axes.remove(los_axis)
    xx, yy = np.meshgrid(
        np.arange(map_data.shape[axes[0]]),
        np.arange(map_data.shape[axes[1]]),
        indexing="ij",
    )
    ra, dec = get_wcs_coor(wproj, xx, yy)
    return map_data, counts, map_has_sampling, ra, dec, nu, wproj


def read_map(
    map_file,
    counts_file=None,
    nu_min=-np.inf,
    nu_max=np.inf,
    ch_start=1,
    los_axis=-1,
    band="L",
):
    """
    Read fits files of MeerKLASS 4k L-band data into arrays.

    Parameters
    ----------
        map_file: str.
            The input map file.
        counts_file: str, default None.
            The input pixel counts file.
        nu_min: float, default -np.inf.
            The lower end of frequency cut.
        nu_max: float, default np.inf.
            The higher end of freuqency cut.
        ch_start: int, default 1.
            The starting channel of the data.
        los_axis: int, default -1.
            which axis is the los.

    Returns
    -------
        map_data: array.
            The map data.
        counts: array.
            The number of sampling for each pixel. If no ``counts_file`` is specified, it is the same as ``map_has_sampling``.
        map_has_sampling: boolean array.
            Whether the pixels are samplied.
        ra: array.
            The RA coordinates of each pixel
        dec: array.
            The Dec coordinates of each pixel
        nu: array.
            The frequencies of each channel in the data.
        wproj: :class:`astropy.wcs.WCS` object.
            The two-dimensional wcs object for the map.
    """
    map_data = fits.open(map_file)[0].data
    num_ch = map_data.shape[los_axis]
    nu_data = cal_freq(
        np.arange(ch_start, ch_start + num_ch),
        band=band,
    )
    nu_sel = np.where((nu_data > nu_min) & (nu_data < nu_max))[0]
    nu_sel_min, nu_sel_max = nu_sel.min(), nu_sel.max()
    sel_indx = [
        slice(None, None, 1),
    ] * 3
    sel_indx[los_axis] = slice(nu_sel_min, nu_sel_max + 1, 1)
    sel_indx = tuple(sel_indx)
    nu = nu_data[nu_sel]
    map_data = np.nan_to_num(map_data[sel_indx])

    map_has_sampling = map_data != 0
    if counts_file is not None:
        counts = fits.open(counts_file)[0].data
        counts = counts[sel_indx]
    else:
        counts = map_has_sampling
    wproj = WCS(map_file).dropaxis(los_axis)
    if los_axis < 0:
        los_axis += 3
    axes = [0, 1, 2]
    axes.remove(los_axis)
    xx, yy = np.meshgrid(
        np.arange(map_data.shape[axes[0]]),
        np.arange(map_data.shape[axes[1]]),
        indexing="ij",
    )
    ra, dec = get_wcs_coor(wproj, xx, yy)
    return map_data, counts, map_has_sampling, ra, dec, nu, wproj
