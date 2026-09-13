import logging

import numpy as np

from .model import ModelPowerSpectrum
from .util import radec_to_indx, find_ch_id, redshift_to_freq

logger = logging.getLogger(__name__)

_STACK_NOT_RUN = (
    "Stacked cubelet has not been computed. Call run_stack() or "
    "get_arg_list_for_galaxy_chunks() then accumulate_stack_chunks() "
    "before reading stack_3d / stack_weight."
)
_STACK_SPACES = ("angular", "config")
_WEIGHTINGS = ("conventional", "quadratic")
_STACK_CHUNK_WORKER: dict | None = None


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


def galaxy_pixel_indices(sp):
    """
    WCS pixel and frequency-channel indices of the galaxy catalogue on ``sp``.

    Parameters
    ----------
    sp : :class:`meer21cm.dataanalysis.Specification`
        Object with ``ra_gal``, ``dec_gal``, ``z_gal``, ``wproj``, and ``nu``.

    Returns
    -------
    indx_0_g, indx_1_g, indx_z_g : ndarray
        Angular pixel indices and channel index of each source.
    """
    indx_0_g, indx_1_g = radec_to_indx(
        np.asarray(sp.ra_gal), np.asarray(sp.dec_gal), sp.wproj
    )
    indx_z_g = find_ch_id(redshift_to_freq(np.asarray(sp.z_gal)), sp.nu)
    return indx_0_g, indx_1_g, indx_z_g


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
    galaxy_pixel_indices : RA/Dec/z to map pixel and channel indices.
    """
    map_in = sp.data.copy()
    w_map_in = sp.w_HI.copy()
    indx_0_g, indx_1_g, indx_z_g = galaxy_pixel_indices(sp)
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


def init_stack_chunk_worker(kwargs):
    """
    Pool initializer: cache shared map arrays for :func:`run_stack_chunk`.

    Use with ``get_arg_list_for_galaxy_chunks(..., use_worker_object=True)``
    so each worker pickles the intensity map once instead of once per chunk.
    """
    global _STACK_CHUNK_WORKER
    _STACK_CHUNK_WORKER = dict(kwargs)


def run_stack_chunk(kwargs, chunk):
    """
    Pickleable worker for one galaxy-index chunk.

    ``kwargs`` is the first element of a tuple from
    :meth:`Stacking.get_arg_list_for_galaxy_chunks`.  ``chunk`` is
    ``(indx_0, indx_1, indx_z, weights_gal)``.

    Returns an un-normalised ``(numerator, weight_map, q0)`` so chunks
    combine exactly.  For ``conventional``, ``q0`` is unused (``0.0``).
    For ``quadratic``, ``q0`` is the chunk's scalar :math:`Q_0`.
    """
    if kwargs.get("use_worker_object"):
        if _STACK_CHUNK_WORKER is None:
            raise RuntimeError(
                "run_stack_chunk needs init_stack_chunk_worker(kwargs) "
                "when use_worker_object is True"
            )
        kw = _STACK_CHUNK_WORKER
    else:
        kw = kwargs
    indx_0, indx_1, indx_z, weights_gal = chunk
    weighting = kw["weighting"]
    symmetrize = bool(kw["symmetrize"])
    stack_map, stack_weight = stack_cubelet(
        kw["map_in"],
        kw["w_map_in"],
        indx_0,
        indx_1,
        indx_z,
        weights_gal=weights_gal,
        weighting=weighting,
        stack_angular_num_nearby_pix=kw["stack_angular_num_nearby_pix"],
        symmetrize=symmetrize,
    )
    if weighting == "conventional":
        return stack_map * stack_weight, stack_weight, 0.0
    w_map = np.asarray(kw["w_map_in"])
    q0 = float(
        np.sum(np.asarray(weights_gal, dtype=float) * w_map[indx_0, indx_1, indx_z])
    )
    if symmetrize:
        q0 = 2.0 * q0
    if q0 != 0:
        return stack_map * q0, stack_weight, q0
    return stack_map, stack_weight, q0


def accumulate_stack_chunk_results(results, weighting):
    """
    Combine :func:`run_stack_chunk` outputs into one cubelet.

    Parameters
    ----------
    results : sequence of (numerator, weight_map, q0)
        Worker returns.
    weighting : {'conventional', 'quadratic'}
        Same scheme used for the chunks.

    Returns
    -------
    stack_3d : ndarray
        Normalised cubelet (same convention as :func:`stack_cubelet`).
    stack_weight : ndarray
        Sum of per-voxel accumulated weights.
    """
    if weighting not in _WEIGHTINGS:
        raise ValueError(
            f"weighting must be 'conventional' or 'quadratic', got '{weighting}'"
        )
    results = list(results)
    if len(results) == 0:
        raise ValueError("accumulate_stack_chunk_results needs at least one chunk")
    numerator = np.zeros_like(results[0][0], dtype=float)
    stack_weight = np.zeros_like(results[0][1], dtype=float)
    q0 = 0.0
    for num, weight, q0_i in results:
        numerator = numerator + num
        stack_weight = stack_weight + weight
        q0 = q0 + float(q0_i)
    stack_3d = np.zeros_like(numerator)
    if weighting == "conventional":
        mask = stack_weight > 0
        stack_3d[mask] = numerator[mask] / stack_weight[mask]
    elif q0 != 0:
        stack_3d = numerator / q0
    else:
        stack_3d = numerator
    return stack_3d, stack_weight


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


class Stacking(ModelPowerSpectrum):
    """
    Combined stacking estimator with survey / cosmology / ``power_kmu``
    from :class:`~meer21cm.model.ModelPowerSpectrum`.

    This class does **not** inherit
    :class:`~meer21cm.estimator.FieldPowerSpectrum` or
    :class:`~meer21cm.grid.LightconeGriddingMixin` and never calls
    ``get_enclosing_box()``.  The data estimator is the existing angular
    :func:`stack` / :func:`stack_cubelet` kernel.

    Call :meth:`run_stack` to fill :attr:`stack_3d` / :attr:`stack_weight`,
    or split the catalogue with :meth:`get_arg_list_for_galaxy_chunks`,
    map :func:`run_stack_chunk` externally, and
    :meth:`accumulate_stack_chunks`.  There is no in-library pool.
    Accessing the cubelet before either path warns and returns ``None``.
    ``stack_space='config'`` is not implemented.

    .. code-block:: python

        >>> from multiprocessing import Pool
        >>> from meer21cm.stack import (
        ...     init_stack_chunk_worker,
        ...     run_stack_chunk,
        ... )
        >>> args = st.get_arg_list_for_galaxy_chunks(
        ...     n_chunks, use_worker_object=True
        ... )
        >>> with Pool(
        ...     n_chunks,
        ...     initializer=init_stack_chunk_worker,
        ...     initargs=(st.stack_chunk_worker_kwargs(),),
        ... ) as pool:
        ...     results = pool.starmap(run_stack_chunk, args)
        >>> st.accumulate_stack_chunks(results)

    Parameters
    ----------
    stack_space : {'angular', 'config'}, default 'angular'
        Coordinate system of the stacked cubelet.  Only ``'angular'`` is
        implemented.
    stack_angular_num_nearby_pix : int, default 10
        Map pixels on each side of the source centre (angular path).
    weighting : {'conventional', 'quadratic'}, default 'conventional'
        Cubelet normalisation passed to :func:`stack`.  Stored as
        :attr:`stack_weighting` so it does not override
        :attr:`~meer21cm.dataanalysis.Specification.weighting` (map
        hit-count scheme).
    symmetrize : bool, default False
        180° flip in :math:`\\Delta\\nu` (Sinigaglia et al. 2022).
    weights_gal : array, optional
        Per-source weights.  If None, uniform weights are used.
    mean_amp_1 : float or str, default 'average_hi_temp'
        HI mean amplitude for the cross-correlation model.
    include_sky_sampling : list, default [False, False]
        Off by default: survey-box sampling is not a stack operator.
    compensate : list, default [False, False]
        Off by default: MAS compensation is not a stack operator.
    **params
        Forwarded to :class:`~meer21cm.model.ModelPowerSpectrum`
        (and therefore :class:`~meer21cm.dataanalysis.Specification`).
    """

    def __init__(
        self,
        stack_space="angular",
        stack_angular_num_nearby_pix=10,
        weighting="conventional",
        symmetrize=False,
        weights_gal=None,
        mean_amp_1="average_hi_temp",
        include_sky_sampling=None,
        compensate=None,
        **params,
    ):
        if include_sky_sampling is None:
            include_sky_sampling = [False, False]
        if compensate is None:
            compensate = [False, False]
        super().__init__(
            mean_amp_1=mean_amp_1,
            include_sky_sampling=include_sky_sampling,
            compensate=compensate,
            **params,
        )
        self.stack_space = stack_space
        self.stack_angular_num_nearby_pix = int(stack_angular_num_nearby_pix)
        self.stack_weighting = weighting
        self.symmetrize = bool(symmetrize)
        self.weights_gal = weights_gal
        self._stack_3d = None
        self._stack_weight = None

    @property
    def stack_space(self):
        """``'angular'`` or ``'config'``.  Only ``'angular'`` is implemented."""
        return self._stack_space

    @stack_space.setter
    def stack_space(self, value):
        space = str(value).lower()
        if space not in _STACK_SPACES:
            raise ValueError(
                f"stack_space must be 'angular' or 'config', got '{value}'"
            )
        if space == "config":
            raise NotImplementedError(
                "stack_space='config' is not implemented yet; use 'angular'."
            )
        self._stack_space = space

    @property
    def stack_weighting(self):
        """``'conventional'`` or ``'quadratic'`` cubelet normalisation."""
        return self._stack_weighting

    @stack_weighting.setter
    def stack_weighting(self, value):
        weighting = str(value).lower()
        if weighting not in _WEIGHTINGS:
            raise ValueError(
                f"weighting must be 'conventional' or 'quadratic', got '{value}'"
            )
        self._stack_weighting = weighting

    def _warn_stack_status(self):
        """Warn if :meth:`run_stack` has not been called; do not compute."""
        if self._stack_3d is None:
            logger.warning(_STACK_NOT_RUN)

    @property
    def stack_3d(self):
        """
        Normalised stacked cubelet, or ``None`` if :meth:`run_stack` has
        not been called.
        """
        self._warn_stack_status()
        return self._stack_3d

    @property
    def stack_weight(self):
        """
        Accumulated per-voxel stack weights, or ``None`` if
        :meth:`run_stack` has not been called.
        """
        self._warn_stack_status()
        return self._stack_weight

    def run_stack(self):
        """
        Stack the intensity map around the stored galaxy catalogue.

        Angular path only: delegates to :func:`stack` and stores
        :attr:`stack_3d` / :attr:`stack_weight`.  For external
        chunk-parallel stacking use
        :meth:`get_arg_list_for_galaxy_chunks` + :func:`run_stack_chunk`
        + :meth:`accumulate_stack_chunks`.

        Returns
        -------
        stack_3d : ndarray
            Normalised cubelet.
        stack_weight : ndarray
            Accumulated per-voxel effective weights.
        """
        if self.stack_space != "angular":
            raise NotImplementedError(
                f"run_stack for stack_space={self.stack_space!r} "
                "is not implemented yet."
            )
        stack_3d, stack_weight = stack(
            self,
            weights_gal=self.weights_gal,
            weighting=self.stack_weighting,
            stack_angular_num_nearby_pix=self.stack_angular_num_nearby_pix,
            symmetrize=self.symmetrize,
        )
        self._stack_3d = stack_3d
        self._stack_weight = stack_weight
        return stack_3d, stack_weight

    def stack_chunk_worker_kwargs(self):
        """
        Shared arrays and knobs for :func:`run_stack_chunk`.

        Pass to :func:`init_stack_chunk_worker` when mapping with
        ``use_worker_object=True``.
        """
        if self.stack_space != "angular":
            raise NotImplementedError(
                f"galaxy chunks for stack_space={self.stack_space!r} "
                "are not implemented yet."
            )
        return {
            "map_in": np.asarray(self.data),
            "w_map_in": np.asarray(self.w_HI),
            "stack_angular_num_nearby_pix": int(self.stack_angular_num_nearby_pix),
            "weighting": self.stack_weighting,
            "symmetrize": bool(self.symmetrize),
        }

    def get_arg_list_for_galaxy_chunks(self, n_chunks=1, use_worker_object=False):
        """
        Pickleable ``(kwargs, chunk)`` tuples for external mapping.

        Splits the galaxy catalogue into ``n_chunks`` index batches.
        Map with :func:`run_stack_chunk`, then
        :meth:`accumulate_stack_chunks`.  Does **not** start a pool.

        Parameters
        ----------
        n_chunks : int, default 1
            Number of galaxy batches.  Empty splits are dropped, so the
            returned list may be shorter than ``n_chunks``.
        use_worker_object : bool, default False
            If True, ``kwargs`` is ``{"use_worker_object": True}`` and
            the pool must be started with
            :func:`init_stack_chunk_worker` and
            :meth:`stack_chunk_worker_kwargs`.  If False, each tuple
            carries the map arrays (serial ``starmap`` without an
            initializer).

        Returns
        -------
        args : list of (dict, tuple)
            Each ``chunk`` is ``(indx_0, indx_1, indx_z, weights_gal)``.
        """
        n_chunks = int(n_chunks)
        if n_chunks < 1:
            raise ValueError(f"n_chunks must be >= 1, got {n_chunks}")
        shared = self.stack_chunk_worker_kwargs()
        indx_0, indx_1, indx_z = galaxy_pixel_indices(self)
        num_g = int(np.asarray(indx_0).size)
        if self.weights_gal is None:
            weights_gal = np.ones(num_g, dtype=float)
        else:
            weights_gal = np.asarray(self.weights_gal, dtype=float)
            if weights_gal.size != num_g:
                raise ValueError(
                    "weights_gal length must match the galaxy catalogue, "
                    f"got {weights_gal.size} vs {num_g}"
                )
        chunk_ids = np.array_split(np.arange(num_g), n_chunks)
        chunks = []
        for ids in chunk_ids:
            if ids.size == 0:
                continue
            chunks.append(
                (
                    np.asarray(indx_0)[ids],
                    np.asarray(indx_1)[ids],
                    np.asarray(indx_z)[ids],
                    weights_gal[ids],
                )
            )
        if use_worker_object:
            kw = {"use_worker_object": True}
            return [(kw, chunk) for chunk in chunks]
        return [(dict(shared), chunk) for chunk in chunks]

    def accumulate_stack_chunks(self, results):
        """
        Sum galaxy-chunk numerators and attach :attr:`stack_3d`.

        Parameters
        ----------
        results : sequence of (numerator, weight_map, q0)
            Outputs of :func:`run_stack_chunk`.

        Returns
        -------
        stack_3d : ndarray
            Normalised cubelet.
        stack_weight : ndarray
            Accumulated per-voxel weights.
        """
        stack_3d, stack_weight = accumulate_stack_chunk_results(
            results, self.stack_weighting
        )
        self._stack_3d = stack_3d
        self._stack_weight = stack_weight
        return stack_3d, stack_weight

    def stack_image(self, vel_ch_avg=5, ang_sum_dist=3.0):
        """
        Collapse :attr:`stack_3d` to a stacked image.

        See :func:`sum_3d_stack`.  Requires :meth:`run_stack`.
        """
        cube = self.stack_3d
        if cube is None:
            raise RuntimeError(
                "stack_image requires run_stack(); the cubelet is not set."
            )
        image, _spectrum = sum_3d_stack(
            cube, vel_ch_avg=vel_ch_avg, ang_sum_dist=ang_sum_dist
        )
        return image

    def stack_spectrum(self, vel_ch_avg=5, ang_sum_dist=3.0):
        """
        Collapse :attr:`stack_3d` to a stacked spectrum.

        See :func:`sum_3d_stack`.  Requires :meth:`run_stack`.
        """
        cube = self.stack_3d
        if cube is None:
            raise RuntimeError(
                "stack_spectrum requires run_stack(); the cubelet is not set."
            )
        _image, spectrum = sum_3d_stack(
            cube, vel_ch_avg=vel_ch_avg, ang_sum_dist=ang_sum_dist
        )
        return spectrum
