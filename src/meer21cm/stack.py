import logging

import numpy as np

from .grid import project_particle_to_regular_grid
from .power import PowerSpectrum
from .util import find_ch_id, radec_to_indx, redshift_to_freq

logger = logging.getLogger(__name__)


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
    The angular sampling of the cubelet corresponds to the map pixels, and the size of the angular
    plane is set by ``stack_angular_num_nearby_pix``. Note that ``stack_angular_num_nearby_pix`` is
    the number of pixels **each side of the centre** so the size of the angular plane is
    ``(2 * stack_angular_num_nearby_pix + 1)**2``.

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
        + (indx_0_g >= map_in.shape[0])
        + (indx_1_g < 0)
        + (indx_1_g >= map_in.shape[1])
        + (indx_z_g == num_ch)
    )
    if sel.sum() > 0:
        raise ValueError("some galaxies are outside survey area or frequency range")
    # zero pad the sky map and the weights
    map_stack = np.zeros(
        (
            np.array(map_in.shape)
            + np.array(
                [2 * stack_angular_num_nearby_pix, 2 * stack_angular_num_nearby_pix, 0]
            )
        )
    )
    map_stack[
        stack_angular_num_nearby_pix:-stack_angular_num_nearby_pix,
        stack_angular_num_nearby_pix:-stack_angular_num_nearby_pix,
    ] = map_in.copy()
    w_stack = np.zeros(
        (
            np.array(map_in.shape)
            + np.array(
                [2 * stack_angular_num_nearby_pix, 2 * stack_angular_num_nearby_pix, 0]
            )
        )
    )
    w_stack[
        stack_angular_num_nearby_pix:-stack_angular_num_nearby_pix,
        stack_angular_num_nearby_pix:-stack_angular_num_nearby_pix,
    ] = w_map_in.copy()
    # indices are shifted by zero-padding
    indx_0_g += stack_angular_num_nearby_pix
    indx_1_g += stack_angular_num_nearby_pix

    num_angular_bin = 2 * stack_angular_num_nearby_pix + 1
    # take a nearby area around each source
    indx_xx, indx_yy = np.meshgrid(
        *(
            (
                np.arange(
                    -stack_angular_num_nearby_pix,
                    stack_angular_num_nearby_pix + 1,
                ),
            )
            * 2
        ),
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
    inside = (
        (channel >= 0)
        & (channel < shape[2])
        & (ix >= 0)
        & (ix < shape[0])
        & (iy >= 0)
        & (iy < shape[1])
    )
    positions = np.column_stack(
        [
            np.asarray(ix, dtype=float)[inside] + shift,
            np.asarray(iy, dtype=float)[inside] + shift,
            np.asarray(channel, dtype=float)[inside] + 0.5,
        ]
    )
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


def rebin_sightlines(weighted, weight, counts, chi, dchi):
    """
    Sum map sightlines into uniform comoving bins.

    ``weighted`` is :math:`w_L L` and ``weight`` is :math:`w_L`. Both are
    summed over the native channels that fall in each bin. Galaxy counts
    are summed, so the number of galaxies is unchanged.

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
    order = np.argsort(np.asarray(chi, dtype=float))
    chi_s = np.asarray(chi, dtype=float)[order]
    edges = np.arange(chi_s[0], chi_s[-1] + 0.5 * dchi, dchi)
    if edges.size == 0 or edges[-1] <= chi_s[-1]:
        edges = np.append(edges, (chi_s[0] if edges.size == 0 else edges[-1]) + dchi)
    which = np.clip(np.digitize(chi_s, edges) - 1, 0, edges.size - 2)
    nbin = edges.size - 1
    weighted = np.asarray(weighted, dtype=float)[..., order]
    weight = np.asarray(weight, dtype=float)[..., order]
    counts = np.asarray(counts, dtype=float)[..., order]
    out_w = np.zeros(weighted.shape[:2] + (nbin,))
    out_wt = np.zeros_like(out_w)
    out_n = np.zeros_like(out_w)
    for bin_id in range(nbin):
        sel = np.flatnonzero(which == bin_id)
        if sel.size == 0:
            continue
        out_w[..., bin_id] = weighted[..., sel].sum(axis=-1)
        out_wt[..., bin_id] = weight[..., sel].sum(axis=-1)
        out_n[..., bin_id] = counts[..., sel].sum(axis=-1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    return out_w, out_wt, out_n, centres


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


class Stacking(PowerSpectrum):
    r"""
    Galaxy--intensity stacking on a map cube.

    ``stack_estimator='loop'`` calls :func:`stack`, which samples each
    galaxy on the native channels. ``stack_estimator='correlation'`` paints
    the weighted galaxy counts on the map mesh, sums the sightlines into
    uniform comoving bins, and estimates the same conventional cubelet
    with a linear real FFT. The galaxy field is the weighted counts.

    The sky stack does not use the survey enclosing box.

    Parameters
    ----------
        stack_estimator: ``{'loop', 'correlation'}``, default ``'loop'``.
            Which measurement to run.
        stack_angular_num_nearby_pix: int, default 4.
            Map pixels kept on each side of the centre. ``4`` is a 9×9 cubelet.
        stack_weighting: ``{'conventional', 'quadratic'}``, default ``'conventional'``.
            Passed to :func:`stack`. The correlation estimator uses
            conventional weighting.
        symmetrize: bool, default False.
            Mirror the loop cubelet about zero frequency separation.
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
        if dchi_mpc <= 0:
            raise ValueError("dchi_mpc must be positive")
        super().__init__(**params)
        self.stack_estimator = stack_estimator
        self.stack_angular_num_nearby_pix = int(stack_angular_num_nearby_pix)
        self.stack_weighting = stack_weighting
        self.symmetrize = bool(symmetrize)
        self.dchi_mpc = float(dchi_mpc)
        self.centre_pixels = bool(centre_pixels)
        self._stack_3d = None
        self._stack_weight = None
        self.n_gal_stacked = 0
        if ra_gal is not None:
            self._ra_gal = np.asarray(ra_gal)
            self._dec_gal = np.asarray(dec_gal)
            self._z_gal = np.asarray(z_gal, dtype=float)
        if weights_gal is not None:
            self.weights_gal = weights_gal

    @property
    def stack_3d(self):
        """
        Normalised cubelet from the last :meth:`run_stack`.

        Returns ``None`` if :meth:`run_stack` has not been called.
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

    def channel_chi_mpc(self):
        """
        Comoving distance of each frequency channel, in Mpc.

        Uses the fiducial cosmology.
        """
        distance = self.astropy_cosmo_fiducial.comoving_distance(self.z_ch)
        return np.asarray(distance.to("Mpc").value, dtype=float)

    def run_stack(self):
        """
        Measure the cubelet with the selected estimator.

        Returns
        -------
            stack_3d, stack_weight: arrays.
                Normalised cubelet and its weight.
        """
        if self.stack_estimator == "loop":
            cube, weight = stack(
                self,
                weights_gal=self.weights_gal,
                weighting=self.stack_weighting,
                stack_angular_num_nearby_pix=self.stack_angular_num_nearby_pix,
                symmetrize=self.symmetrize,
            )
            self.n_gal_stacked = int(np.size(self.ra_gal))
        elif self.stack_estimator == "correlation":
            if self.stack_weighting != "conventional":
                raise ValueError(
                    "the correlation estimator uses conventional weighting"
                )
            if self.symmetrize:
                raise ValueError(
                    "the correlation estimator does not symmetrize the cubelet"
                )
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
        positions, inside = sky_mesh_positions(
            ra,
            dec,
            redshift,
            self.wproj,
            self.nu,
            self.data.shape,
            centre_pixels=self.centre_pixels,
        )
        if self.weights_gal is None:
            mass = np.ones(int(np.size(ra)), dtype=float)[inside]
        else:
            mass = np.asarray(self.weights_gal, dtype=float)[inside]
        self.n_gal_stacked = int(inside.sum())
        counts = paint_sky_counts(positions, self.data.shape, mass=mass)
        weighted, weight, counts_b, _centres = rebin_sightlines(
            np.asarray(self.w_HI, dtype=float) * np.asarray(self.data, dtype=float),
            np.asarray(self.w_HI, dtype=float),
            counts,
            self.channel_chi_mpc(),
            self.dchi_mpc,
        )
        return correlate_cubelet(
            weighted,
            weight,
            counts_b,
            self.stack_angular_num_nearby_pix,
        )

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
