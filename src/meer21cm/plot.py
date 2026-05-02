import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as mticker
import numpy as np
import copy
import matplotlib

from .util import tightest_ra_interval


def plot_pixels_along_los(
    map_in,
    map_has_sampling,
    zaxis=None,
    xlabel="",
    ylabel="",
    lw=0.01,
    title="",
    los_axis=-1,
):
    plt.figure()
    map_plot = map_in.copy()
    map_plot[map_has_sampling == 0] = np.nan
    if los_axis < 0:
        los_axis += 3
    axes = [0, 1, 2]
    axes.remove(los_axis)
    # make sure los is the last axis
    axes = axes + [
        los_axis,
    ]
    map_plot = np.transpose(map_plot, axes=axes)
    nz = map_plot.shape[-1]
    map_plot = map_plot.reshape((-1, nz))
    if zaxis is None:
        zaxis = np.arange(nz)
    for i in range(len(map_plot)):
        plt.plot(zaxis, map_plot[i], lw=lw, color="black")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)


def _healpix_ra_tick_formatter(ax_map):
    """
    R.A. labels in :math:`[0^\\circ, 360^\\circ)`.

    healpy's Cartesian plane uses negated longitude when ``flip='astro'`` (east left),
    so raw axis values can be negative even when the sky R.A. is :math:`310^\\circ`–
    :math:`320^\\circ`. We label from ``ax_map.get_lonlat`` (true lon/lat at the tick).
    """

    def _fmt(x, pos):
        if not np.isfinite(x):
            return ""
        y0, y1 = ax_map.get_ylim()
        y_ref = 0.5 * (float(y0) + float(y1))
        pos_ll = ax_map.get_lonlat(x, y_ref)
        if pos_ll is None:
            return ""
        lon, _lat = pos_ll
        if np.isnan(lon):
            return ""
        ra = float(np.mod(lon, 360.0))
        if ra >= 360.0 - 1e-5:
            ra = 0.0
        rar = round(ra)
        if abs(ra - rar) < 1e-3:
            return f"{int(rar)}°"
        return f"{ra:g}°"

    return mticker.FuncFormatter(_fmt)


def _healpix_dec_tick_formatter(ax_map):
    """Declination tick labels from sky coordinates at the tick (not raw plane *x*)."""

    def _fmt(y, pos):
        if not np.isfinite(y):
            return ""
        x0, x1 = ax_map.get_xlim()
        x_ref = 0.5 * (float(x0) + float(x1))
        pos_ll = ax_map.get_lonlat(x_ref, y)
        if pos_ll is None:
            return ""
        _lon, lat = pos_ll
        if np.isnan(lat):
            return ""
        latr = round(lat)
        if abs(lat - latr) < 1e-3:
            return f"{int(latr)}°"
        return f"{lat:g}°"

    return mticker.FuncFormatter(_fmt)


def _finish_healpix_cartview_figure(
    fig,
    ax_map,
    have_cbar,
    cbar_label,
    cbarshrink,
    *,
    cbar_yoffset=0.0,
    xlabel_pad=5.0,
):
    """Axis labels, degree ticks, grid, and matplotlib-style colorbar label after ``cartview``."""
    # healpy ``SphericalProjAxes`` calls ``axis('off')`` in ``__init__``, which suppresses
    # matplotlib axis labels and ticks even after ``set_xlabel`` / ``set_ylabel``.
    ax_map.set_axis_on()
    ax_map.set_xlabel("R.A [deg]", labelpad=xlabel_pad)
    ax_map.set_ylabel("Dec. [deg]")
    ax_map.xaxis.set_major_formatter(_healpix_ra_tick_formatter(ax_map))
    ax_map.yaxis.set_major_formatter(_healpix_dec_tick_formatter(ax_map))
    ax_map.grid(True, color="grey", ls="solid", lw=0.5)
    if have_cbar and len(fig.axes) > 1:
        for cb_ax in fig.axes[1:]:
            if cb_ax is ax_map:
                continue
            for t in list(cb_ax.texts):
                t.remove()
            if cbar_label:
                cb_ax.set_xlabel(cbar_label, labelpad=2.0)
            pos = cb_ax.get_position()
            w = float(pos.width)
            x0 = float(pos.x0)
            if cbarshrink is not None and 0 < float(cbarshrink) < 1:
                w *= float(cbarshrink)
                x0 = float(pos.x0) + (float(pos.width) - w) / 2
            # Figure coordinates: lower *y0* moves the colorbar down (healpy's ``pad`` is tight vs x-label).
            y0 = float(pos.y0) + float(cbar_yoffset)
            y0 = max(y0, 0.02)
            cb_ax.set_position([x0, y0, w, float(pos.height)])


def _healpix_pixel_scale_deg(nside):
    """Typical linear pixel scale (~ :math:`\\sqrt{\\Omega_{\\rm pix}}`) in degrees."""
    import healpy as hp

    omega_deg2 = hp.nside2pixarea(int(nside), degrees=True)
    return float(np.sqrt(max(omega_deg2, 1e-30)))


def _normalize_lonra(lo, hi):
    """Return ``(lon_lo, lon_hi)`` with positive span (``healpy`` Cartesian convention)."""
    lo = float(lo)
    hi = float(hi)
    span = np.mod(hi - lo, 360.0)
    if span == 0:
        span = 360.0
    return lo, lo + span


def _apply_ra_buffer(lo, hi, buffer_deg, pad_deg):
    """
    Widen ``(lo, hi)`` by ``buffer_deg`` on each side in longitude; respect wrap and
    full-sky. ``pad_deg`` is a minimum half-width when the interval collapses to a point.
    """
    L0, L1 = _normalize_lonra(lo, hi)
    span = L1 - L0
    if span >= 360.0 - 1e-9:
        return [0.0, 360.0]
    if span < 1e-9:
        w = max(float(buffer_deg), float(pad_deg))
        return list(_normalize_lonra(lo - w, lo + w))
    return list(_normalize_lonra(L0 - buffer_deg, L1 + buffer_deg))


def _apply_dec_buffer(lat0, lat1, buffer_deg):
    """Widen declination bounds by ``buffer_deg``, clamped to :math:`[-90^\\circ, 90^\\circ]`."""
    a = float(lat0) - float(buffer_deg)
    b = float(lat1) + float(buffer_deg)
    a = max(a, -90.0)
    b = min(b, 90.0)
    if a >= b:
        mid = float(np.clip(0.5 * (float(lat0) + float(lat1)), -90.0, 90.0))
        half = float(min(max(buffer_deg, 0.25), 45.0))
        a = max(mid - half, -90.0)
        b = min(mid + half, 90.0)
        if a >= b:
            a, b = mid - 0.25, mid + 0.25
    return [a, b]


def _cart_extent_from_ranges(lonra, latra, scale_deg, min_pixels=96, max_pixels=2400):
    """Infer ``xsize`` / ``ysize`` from RA/Dec span and HEALPix pixel scale (degrees)."""
    dlon = float(lonra[1] - lonra[0])
    dlat = float(latra[1] - latra[0])
    if dlon <= 0 or dlat <= 0:
        raise ValueError("lonra and latra must have positive span.")
    xsize = int(np.ceil(dlon / scale_deg))
    ysize = int(np.ceil(dlat / scale_deg))
    xsize = int(np.clip(xsize, min_pixels, max_pixels))
    ysize = int(np.clip(ysize, min_pixels, max_pixels))
    return xsize, ysize


def _lonlat_ranges_cartview(lon_deg, lat_deg, ra_range, dec_range, pad_deg, buffer_deg):
    """Build ``lonra`` / ``latra`` for :func:`healpy.cartview` with edge-aware buffering."""
    lon_deg = np.asarray(lon_deg, dtype=float).ravel()
    lat_deg = np.asarray(lat_deg, dtype=float).ravel()

    if dec_range is not None:
        d0, d1 = float(dec_range[0]), float(dec_range[1])
    else:
        d0, d1 = float(np.min(lat_deg)), float(np.max(lat_deg))
    latra = _apply_dec_buffer(d0, d1, buffer_deg)

    if ra_range is not None:
        lo, hi = float(ra_range[0]), float(ra_range[1])
    else:
        lo, hi = tightest_ra_interval(lon_deg)
    lonra = _apply_ra_buffer(lo, hi, buffer_deg, pad_deg)

    return lonra, latra


def plot_map_wcs(
    map_plot,
    wproj,
    W=None,
    title="",
    have_cbar=True,
    cbar_label="",
    cbarshrink=1,
    ZeroCentre=False,
    vmin=None,
    vmax=None,
    cmap="magma",
    invert_x=True,
    dpi=100,
    cbar_aspect=25,
    ax=None,
):
    """
    Plot a map with WCS :class:`astropy.wcs.WCS` projection (original ``plot_map`` behaviour).
    """
    if ax is None:
        plt.figure(dpi=dpi)
        plt.subplot(projection=wproj)
        ax = plt.gca()
    lon = ax.coords[0]
    lat = ax.coords[1]
    lon.set_major_formatter("d")
    lon.set_ticks_position("b")
    lat.set_ticks_position("l")
    ax.grid(True, color="grey", ls="solid", lw=0.5)
    map_in = map_plot.copy()
    if len(np.shape(map_in)) == 3:
        map_in = np.mean(
            map_in, 2
        )  # Average along 3rd dimention (LoS) as default if 3D map given
        if W is not None:
            W = W.copy()
            W = np.mean(W, 2)
    if vmax is not None:
        map_in[map_in > vmax] = vmax
    if vmin is not None:
        map_in[map_in < vmin] = vmin
    if ZeroCentre == True:
        divnorm = colors.TwoSlopeNorm(
            vmin=np.min(map_in), vcenter=0, vmax=np.max(map_in)
        )
        cmap = copy.copy(matplotlib.cm.get_cmap("seismic"))
        cmap.set_bad(color="grey")
    else:
        divnorm = None
    if W is not None:
        map_in[W == 0] = np.nan
    im = ax.imshow(map_in.T, cmap=cmap, norm=divnorm)
    if vmax is not None or vmin is not None:
        ax.get_images()[0].set_clim(vmin, vmax)
    if have_cbar:
        cbar = plt.colorbar(
            im,
            orientation="horizontal",
            shrink=cbarshrink,
            pad=0.2,
            aspect=cbar_aspect,
            ax=ax,
        )
        cbar.set_label(cbar_label)
    if invert_x:
        ax.invert_xaxis()
    ax.set_xlabel("R.A [deg]")
    ax.set_ylabel("Dec. [deg]")
    ax.set_title(title)


def plot_map_healpix(
    map_plot,
    pixel_id,
    hp_nside,
    W=None,
    title="",
    have_cbar=True,
    cbar_label="",
    cbarshrink=1,
    ZeroCentre=False,
    vmin=None,
    vmax=None,
    cmap="magma",
    dpi=100,
    ax=None,
    ra_range=None,
    dec_range=None,
    pad_deg=1.0,
    buffer_pixel_scale=2.0,
    xsize=None,
    ysize=None,
    pixel_scale_deg=None,
    invert_x=True,
    cbar_yoffset=-0.045,
    healpix_xlabel_pad=5.0,
):
    """
    Cartesian HEALPix view (``healpy.cartview``) for sparse ``pixel_id`` samples.

    The map extent is set by ``ra_range`` / ``dec_range`` when given (same idea as
    :class:`~meer21cm.dataanalysis.Specification`), otherwise inferred from pixel
    centres. Each side is padded by ``buffer_pixel_scale`` times the typical HEALPix
    pixel scale in degrees (see ``pixel_scale_deg``); declination is clipped to
    :math:`[-90^\\circ, 90^\\circ]`, and full-sky RA is detected so padding does not
    inflate beyond :math:`360^\\circ`.

    Parameters
    ----------
    map_plot : ndarray
        Length ``n_pix``, or ``(n_pix, n_chan)`` / ``(nx, ny, n_chan)``. Multiple channels are
        averaged along the line-of-sight (last axis), like :func:`plot_map_wcs`.
    pixel_id : ndarray
        HEALPix indices (RING, ``nest=False``), same length as the flattened map.
    hp_nside : int
        :math:`N_{side}` of the map.
    W : ndarray, optional
        Sampling weights; same shape rules as ``map_plot``.
    ra_range : tuple of float, optional
        ``(ra_min, ra_max)`` in degrees for ``lonra`` (full sphere allowed). Default: from data.
    dec_range : tuple of float, optional
        ``(dec_min, dec_max)`` in degrees for ``latra``. Default: min/max declination of sample.
    pad_deg : float
        Minimum half-width in degrees when the RA interval collapses to one longitude
        (used together with the automatic buffer).
    buffer_pixel_scale : float
        Sky padding on each RA/Dec edge is ``buffer_pixel_scale * pixel_scale_deg``
        (default ``2`` × approximate pixel linear size in degrees).
    xsize, ysize : int, optional
        Raster size. Default: from sky span / typical HEALPix pixel scale (degrees).
    pixel_scale_deg : float, optional
        Degrees per image pixel for auto ``xsize``/``ysize``. Default: :math:`\\sqrt{\\Omega_{\\rm pix}}`
        in degrees at this ``hp_nside``.
    invert_x : bool
        If True (default), use healpy ``flip='astro'`` (matches :func:`plot_map_wcs`). If False,
        ``flip='geo'``. R.A. tick labels are shown in :math:`[0^\\circ,360^\\circ)` using the
        on-sky longitude (the axes still use healpy's internal *x* when ``flip='astro'``).
    cbar_yoffset : float
        Extra vertical shift for the healpy horizontal colorbar in **figure fraction** units
        (negative moves the colorbar **down**, clearing overlap with the map's R.A. label).
    healpix_xlabel_pad : float
        Passed as ``labelpad`` on the map's ``R.A [deg]`` label (points).
    ax : optional
        Not supported yet; use WCS :func:`plot_map_wcs` for subplot grids.
    """
    import healpy as hp

    if ax is not None:
        raise ValueError(
            "plot_map_healpix does not support ax= yet; use plot_map_wcs for subplot grids."
        )

    nside = int(hp_nside)
    map_in = np.asarray(map_plot, dtype=float).copy()
    if W is not None:
        W = np.asarray(W, dtype=float).copy()
    # Same intent as plot_map_wcs: average over frequency / LOS on the last axis.
    # HEALPix cubes are typically (n_pix, n_chan); WCS uses (nx, ny, n_chan).
    if map_in.ndim == 3:
        map_in = np.mean(map_in, axis=-1)
        if W is not None:
            W = np.mean(W, axis=-1)
    elif map_in.ndim == 2:
        if map_in.shape[-1] > 1:
            map_in = np.mean(map_in, axis=-1)
            if W is not None:
                W = np.mean(W, axis=-1)
        else:
            map_in = map_in[:, 0]
            if W is not None:
                W = W[:, 0]
    if map_in.ndim != 1:
        raise ValueError(
            "HEALPix plot expects a 1D map or a cube with LOS/frequency on the last "
            f"axis (after averaging); got shape {map_plot.shape}."
        )
    pid = np.asarray(pixel_id, dtype=np.int64).ravel()
    if map_in.size != pid.size:
        raise ValueError(
            f"map_plot length {map_in.size} does not match pixel_id length {pid.size}."
        )

    vals = map_in.ravel().copy()
    if W is not None:
        W = np.asarray(W).ravel()
        if W.size != vals.size:
            raise ValueError("W must match map_plot shape.")
        vals[W == 0] = np.nan

    if vmax is not None:
        vals = np.clip(vals, None, vmax)
    if vmin is not None:
        vals = np.clip(vals, vmin, None)

    npix = hp.nside2npix(nside)
    full = np.full(npix, np.nan, dtype=float)
    if (pid < 0).any() or (pid >= npix).any():
        raise ValueError("pixel_id out of range for this hp_nside.")
    full[pid] = vals

    lon, lat = hp.pix2ang(nside, pid, nest=False, lonlat=True)
    scale_deg = (
        float(pixel_scale_deg)
        if pixel_scale_deg is not None
        else _healpix_pixel_scale_deg(nside)
    )
    buffer_deg = float(buffer_pixel_scale) * scale_deg
    lonra, latra = _lonlat_ranges_cartview(
        lon, lat, ra_range, dec_range, pad_deg, buffer_deg
    )
    if xsize is None or ysize is None:
        auto_x, auto_y = _cart_extent_from_ranges(lonra, latra, scale_deg)
        if xsize is None:
            xsize = auto_x
        if ysize is None:
            ysize = auto_y

    if ZeroCentre:
        cmap_use = copy.copy(matplotlib.cm.get_cmap("seismic"))
        cmap_use.set_bad(color="grey")
        norm = colors.TwoSlopeNorm(
            vmin=np.nanmin(vals), vcenter=0, vmax=np.nanmax(vals)
        )
        cmap_arg = cmap_use
        min_arg, max_arg = None, None
    else:
        norm = None
        cmap_arg = cmap
        min_arg, max_arg = vmin, vmax

    hp.cartview(
        full,
        fig=None,
        rot=None,
        coord="C",
        nest=False,
        title=title,
        unit="",
        xsize=int(xsize),
        ysize=int(ysize),
        lonra=lonra,
        latra=latra,
        min=min_arg,
        max=max_arg,
        flip="astro" if invert_x else "geo",
        cbar=have_cbar,
        cmap=cmap_arg,
        norm=norm,
        notext=True,
    )
    fig = plt.gcf()
    fig.set_dpi(dpi)
    ax_out = plt.gca()
    _finish_healpix_cartview_figure(
        fig,
        ax_out,
        have_cbar,
        cbar_label,
        cbarshrink,
        cbar_yoffset=cbar_yoffset,
        xlabel_pad=healpix_xlabel_pad,
    )


def plot_map(
    map_plot,
    wproj=None,
    W=None,
    title="",
    have_cbar=True,
    cbar_label="",
    cbarshrink=1,
    ZeroCentre=False,
    vmin=None,
    vmax=None,
    cmap="magma",
    invert_x=True,
    dpi=100,
    cbar_aspect=25,
    ax=None,
    pixel_id=None,
    hp_nside=None,
    ra_range=None,
    dec_range=None,
    pad_deg=1.0,
    buffer_pixel_scale=2.0,
    xsize=None,
    ysize=None,
    pixel_scale_deg=None,
    cbar_yoffset=-0.045,
    healpix_xlabel_pad=5.0,
):
    """
    Plot a sky map: WCS image by default, or HEALPix Cartesian view when ``pixel_id`` is given.

    Parameters
    ----------
    map_plot : ndarray
        Map values (WCS 2D/3D or HEALPix 1D/3D sparse).
    wproj : :class:`astropy.wcs.WCS`, optional
        Required for WCS plots (when ``pixel_id`` is None).
    pixel_id : ndarray, optional
        If set, plot with :func:`plot_map_healpix` using ``healpy.cartview``.
    hp_nside : int, optional
        Required together with ``pixel_id``.
    ra_range, dec_range : tuple of float, optional
        Passed to :func:`plot_map_healpix` (``lonra`` / ``latra`` for ``cartview``).
    pad_deg : float
        HEALPix-only; see :func:`plot_map_healpix`.
    buffer_pixel_scale : float
        HEALPix-only edge padding in units of HEALPix pixel scale; see :func:`plot_map_healpix`.
    xsize, ysize, pixel_scale_deg
        HEALPix-only raster controls; see :func:`plot_map_healpix`.
    cbar_yoffset, healpix_xlabel_pad
        HEALPix-only layout for the horizontal colorbar vs R.A. label; see :func:`plot_map_healpix`.
    """
    if pixel_id is not None:
        if wproj is not None:
            raise ValueError(
                "Pass either ``wproj`` (WCS) or ``pixel_id`` (HEALPix), not both."
            )
        if hp_nside is None:
            raise ValueError("hp_nside is required when pixel_id is set.")
        return plot_map_healpix(
            map_plot,
            pixel_id,
            hp_nside,
            W=W,
            title=title,
            have_cbar=have_cbar,
            cbar_label=cbar_label,
            cbarshrink=cbarshrink,
            ZeroCentre=ZeroCentre,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            dpi=dpi,
            ax=None,
            ra_range=ra_range,
            dec_range=dec_range,
            pad_deg=pad_deg,
            buffer_pixel_scale=buffer_pixel_scale,
            xsize=xsize,
            ysize=ysize,
            pixel_scale_deg=pixel_scale_deg,
            invert_x=invert_x,
            cbar_yoffset=cbar_yoffset,
            healpix_xlabel_pad=healpix_xlabel_pad,
        )
    if wproj is None:
        raise TypeError(
            "plot_map requires ``wproj`` for WCS plots, or pass ``pixel_id`` and "
            "``hp_nside`` for HEALPix."
        )
    return plot_map_wcs(
        map_plot,
        wproj,
        W=W,
        title=title,
        have_cbar=have_cbar,
        cbar_label=cbar_label,
        cbarshrink=cbarshrink,
        ZeroCentre=ZeroCentre,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        invert_x=invert_x,
        dpi=dpi,
        cbar_aspect=cbar_aspect,
        ax=ax,
    )


def plot_eigenspectrum(
    eigenval,
    eignumb=None,
    eignumb_cut=40,
    title="",
):
    plt.figure()
    if len(eigenval.shape) == 1:
        eigenval = np.array(
            [
                eigenval,
            ]
        )
    if eignumb is None:
        eignumb = np.arange(eigenval.shape[1])
    for i in range(len(eigenval)):
        plt.plot(eignumb[:eignumb_cut], eigenval[i][:eignumb_cut], "-o")
    plt.yscale("log")
    plt.xlabel("Eigennumber")
    plt.ylabel("Eigenvalue")
    plt.title(title)


def plot_projected_map(A, data, wproj, W=None):
    S_pca_full = np.nan_to_num(A[:, None, :] * A[None, :, :])
    num_of_maps = A.shape[1]
    num_rows = int(np.ceil(num_of_maps / 4))
    num_pix_x, num_pix_y = data.shape[:2]
    ratio = num_pix_x / num_pix_y
    fig, axes = plt.subplots(
        num_rows,
        4,
        figsize=(int(np.ceil(16 * ratio)), int(np.ceil(6 * num_rows))),
        subplot_kw={"projection": wproj},
        gridspec_kw={"hspace": 0.1, "wspace": 0.1},
    )
    for i in range(num_of_maps):
        res_i = np.einsum("ij,abj->abi", S_pca_full[:, :, i], np.nan_to_num(data))
        plot_map(res_i, wproj, W=W, ax=axes.ravel()[i], title=f"mode {i}")
    return fig, axes


def visualise_patch_split(mask_arr, wproj):
    """
    Visualise the patch split by plotting the mask array.
    """
    for nu_indx in range(mask_arr.shape[2]):
        mask_map = np.zeros_like(mask_arr[0, 0, 0, :, :, 0]) + np.nan
        for i in range(mask_arr.shape[0]):
            for j in range(mask_arr.shape[1]):
                sel = mask_arr[i, j, nu_indx].sum(-1) > 0
                mask_map[sel] = i + j * mask_arr.shape[0]
        plot_map(mask_map, wproj, have_cbar=False, title=f"frequency bin {nu_indx}")
