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

from .grid import project_particle_to_regular_grid
from .io import read_catalogue_fits
from .power import PowerSpectrum
from .util import find_ch_id, radec_to_indx, redshift_to_freq

logger = logging.getLogger(__name__)


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
        distance = self.astropy_cosmo_fiducial.comoving_distance(self.z_ch)
        return np.asarray(distance.to("Mpc").value, dtype=float)

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
        pix_deg = float(np.sqrt(self.pixel_area))
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

    def model_1d_stack(self):
        """
        One-dimensional stacked spectrum.

        The spectrum is the inverse Fourier transform, along the line of
        sight, of the transverse integral of the galaxy--intensity
        cross-power, including the pixel, beam, and radial windows.

        Raises
        ------
        NotImplementedError
            The integral is not available.
        """
        raise NotImplementedError("model_1d_stack is not implemented")
