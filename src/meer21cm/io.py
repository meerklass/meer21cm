"""
Module for reading and pre-processing MeerKLASS maps and galaxy catalogues.
"""

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import healpy as hp
from .util import get_wcs_coor
from . import telescope
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


def read_catalogue_fits(
    paths,
    ra_col="RA",
    dec_col="DEC",
    z_col="Z",
    weight_col=None,
):
    """
    Read a list of catalogue FITS files and concatenate them.

    ``weight_col=None`` assigns unit weight to every row. ``WEIGHT_FKP`` is
    not read.

    Parameters
    ----------
    paths : sequence of path
        FITS tables. Each must have a binary table in HDU 1.
    ra_col, dec_col, z_col : str
        Column names for right ascension, declination and redshift.
    weight_col : str, optional
        Column of per-object weights. ``None`` uses unit weight.

    Returns
    -------
    ra, dec, z, weight : ndarray
        Concatenated columns. All four are empty if ``paths`` is empty.
    """
    ra_parts = []
    dec_parts = []
    z_parts = []
    w_parts = []
    for path in paths:
        with fits.open(path, memmap=True) as hdul:
            table = hdul[1].data
            ra_i = np.asarray(table[ra_col], dtype=float)
            ra_parts.append(ra_i)
            dec_parts.append(np.asarray(table[dec_col], dtype=float))
            z_parts.append(np.asarray(table[z_col], dtype=float))
            if weight_col is None:
                w_parts.append(np.ones(ra_i.size, dtype=float))
            else:
                w_parts.append(np.asarray(table[weight_col], dtype=float))
    if not ra_parts:
        empty = np.zeros(0, dtype=float)
        return empty, empty, empty, empty
    return (
        np.concatenate(ra_parts),
        np.concatenate(dec_parts),
        np.concatenate(z_parts),
        np.concatenate(w_parts),
    )


def _native_float(values):
    """Copy a FITS numeric column into a native-endian float array.

    Parameters
    ----------
    values : array
        Column from a memory-mapped FITS table.

    Returns
    -------
    ndarray
        Float copy. Big-endian columns are byte-swapped.
    """
    values = np.asarray(values)
    if values.dtype.byteorder == ">":
        values = values.astype(values.dtype.newbyteorder("="))
    return np.asarray(values, dtype=float)


def _source_weight(table, columns):
    """Product of the shuffled source-weight columns.

    Parameters
    ----------
    table : FITS table
        Rows of one catalogue.
    columns : sequence of str
        Column names whose product is :math:`w'_{\\rm tot}`. Missing columns
        are skipped. If none are present the weight is 1.

    Returns
    -------
    ndarray
        One weight per row.
    """
    names = set(table.columns.names)
    present = [name for name in columns if name in names]
    if not present:
        return np.ones(len(table), dtype=float)
    weight = np.ones(len(table), dtype=float)
    for name in present:
        weight *= _native_float(table[name])
    return weight


def selection_from_random_files(
    random_paths,
    ra_range,
    dec_range,
    z_edges,
    cosmo,
    data_path=None,
    nside=512,
    random_density_deg2=2500.0,
    source_weight_columns=("WEIGHT_COMP", "WEIGHT_SYS", "WEIGHT_ZFAIL"),
):
    """HEALPix angular factor and radial density from random catalogues.

    :math:`A` is ``WEIGHT`` divided by the shuffled source weight, summed in
    HEALPix pixels and divided by the nominal random density. :math:`n_w(z)`
    is ``WEIGHT / A`` per comoving shell of the pixels where :math:`A > 0`.
    If a data catalogue is given, :math:`n_w` is scaled so that
    :math:`\\int n_w A\\,{\\rm d}V` equals the summed data ``WEIGHT``.

    Parameters
    ----------
    random_paths : sequence of path
        FITS random catalogues with ``RA``, ``DEC``, ``Z`` and ``WEIGHT``.
    ra_range, dec_range : pair of float
        Sky window in degrees. Rows outside it are ignored.
    z_edges : array
        Redshift edges of the radial shells. :math:`A` uses every redshift
        in the sky window. :math:`n_w` uses shells inside these edges.
    cosmo : cosmology
        Astropy-like cosmology. Shell volumes use ``comoving_distance``.
    data_path : path, optional
        Data catalogue. ``None`` leaves :math:`n_w` on the random normalisation.
    nside : int, default 512
        HEALPix resolution of :math:`A`.
    random_density_deg2 : float, default 2500
        Nominal random density of one file, in deg\\(:sup:`-2`\\), before vetoes.
    source_weight_columns : sequence of str
        Columns multiplied to give the shuffled source weight.

    Returns
    -------
    dict
        ``angular`` (HEALPix), ``n_w``, ``z_edges``, ``nside``,
        ``data_weight``, ``random_weight`` and ``scale``.
    """
    z_edges = np.asarray(z_edges, dtype=float)
    n_pix = hp.nside2npix(int(nside))
    sum_ratio = np.zeros(n_pix, dtype=float)
    for path in random_paths:
        _accumulate_angular(
            sum_ratio,
            path,
            ra_range,
            dec_range,
            nside,
            source_weight_columns,
        )
    omega_deg = float(hp.nside2pixarea(int(nside), degrees=True))
    angular = sum_ratio / (
        float(random_density_deg2) * len(list(random_paths)) * omega_deg
    )
    footprint = angular > 0
    sum_over_a = np.zeros(z_edges.size - 1, dtype=float)
    random_weight = 0.0
    for path in random_paths:
        shells, weight_sum = _accumulate_radial(
            path,
            ra_range,
            dec_range,
            z_edges,
            nside,
            angular,
        )
        sum_over_a += shells
        random_weight += weight_sum
    if data_path is None:
        data_weight = random_weight
    else:
        _shells, data_weight = _accumulate_radial(
            data_path,
            ra_range,
            dec_range,
            z_edges,
            nside,
            np.ones(n_pix, dtype=float),
        )
    volume = _shell_volume(z_edges, int(footprint.sum()), cosmo, nside)
    n_w = sum_over_a / np.maximum(volume, 1e-30)
    dvol_pix = volume / max(int(footprint.sum()), 1)
    integral = float(np.sum(n_w * dvol_pix * angular.sum()))
    scale = float(data_weight) / integral if integral > 0.0 else 1.0
    n_w = n_w * scale
    return {
        "angular": angular,
        "n_w": n_w,
        "z_edges": z_edges,
        "nside": int(nside),
        "data_weight": float(data_weight),
        "random_weight": float(random_weight),
        "scale": scale,
    }


def _sky_rows(table, ra_range, dec_range):
    """Rows inside a right ascension and declination window.

    Parameters
    ----------
    table : FITS table
        Catalogue with ``RA`` and ``DEC``.
    ra_range, dec_range : pair of float
        Window in degrees.

    Returns
    -------
    ndarray
        Boolean mask, one entry per row.
    """
    ra = _native_float(table["RA"])
    dec = _native_float(table["DEC"])
    return (
        (ra >= ra_range[0])
        & (ra <= ra_range[1])
        & (dec >= dec_range[0])
        & (dec <= dec_range[1])
    )


def _accumulate_angular(sum_ratio, path, ra_range, dec_range, nside, source_columns):
    """Add ``WEIGHT / w'_tot`` of one file into HEALPix pixels.

    Parameters
    ----------
    sum_ratio : ndarray
        Pixel sums, updated in place.
    path : path
        FITS catalogue.
    ra_range, dec_range : pair of float
        Sky window in degrees.
    nside : int
        HEALPix resolution.
    source_columns : sequence of str
        Columns multiplied into the shuffled source weight.

    Returns
    -------
    None
    """
    with fits.open(path, memmap=True) as hdul:
        table = hdul[1].data
        keep = _sky_rows(table, ra_range, dec_range)
        weight = _native_float(table["WEIGHT"])
        source = _source_weight(table, source_columns)
        ok = keep & np.isfinite(weight) & np.isfinite(source) & (source > 0)
        pix = hp.ang2pix(
            int(nside),
            _native_float(table["RA"])[ok],
            _native_float(table["DEC"])[ok],
            lonlat=True,
        )
        np.add.at(sum_ratio, pix, weight[ok] / source[ok])


def _accumulate_radial(path, ra_range, dec_range, z_edges, nside, angular):
    """Sum ``WEIGHT / A`` in redshift shells for one catalogue.

    Parameters
    ----------
    path : path
        FITS catalogue.
    ra_range, dec_range : pair of float
        Sky window in degrees.
    z_edges : array
        Redshift shell edges.
    nside : int
        HEALPix resolution of ``angular``.
    angular : ndarray
        HEALPix :math:`A`. A value of 1 sums ``WEIGHT`` itself.

    Returns
    -------
    shells : ndarray
        ``WEIGHT / A`` in each shell.
    weight_sum : float
        Sum of ``WEIGHT`` inside the sky window and the redshift edges.
    """
    shells = np.zeros(len(z_edges) - 1, dtype=float)
    with fits.open(path, memmap=True) as hdul:
        table = hdul[1].data
        keep = _sky_rows(table, ra_range, dec_range)
        redshift = _native_float(table["Z"])
        weight = _native_float(table["WEIGHT"])
        in_z = (
            keep
            & (redshift >= z_edges[0])
            & (redshift < z_edges[-1])
            & np.isfinite(weight)
        )
        pix = hp.ang2pix(
            int(nside),
            _native_float(table["RA"])[in_z],
            _native_float(table["DEC"])[in_z],
            lonlat=True,
        )
        amp = np.asarray(angular, dtype=float)[pix]
        use = amp > 0
        index = np.digitize(redshift[in_z][use], z_edges) - 1
        np.add.at(shells, index, weight[in_z][use] / amp[use])
        weight_sum = float(weight[in_z].sum())
    return shells, weight_sum


def _shell_volume(z_edges, n_pix, cosmo, nside):
    """Comoving volume of the footprint in each redshift shell.

    Parameters
    ----------
    z_edges : array
        Redshift edges.
    n_pix : int
        Number of HEALPix pixels with :math:`A > 0`.
    cosmo : cosmology
        Provides ``comoving_distance`` in Mpc.
    nside : int
        HEALPix resolution.

    Returns
    -------
    ndarray
        Shell volumes in Mpc\\(:sup:`3`\\).
    """
    omega = int(n_pix) * float(hp.nside2pixarea(int(nside)))
    chi = np.asarray(cosmo.comoving_distance(z_edges).value, dtype=float)
    return omega / 3.0 * (chi[1:] ** 3 - chi[:-1] ** 3)
