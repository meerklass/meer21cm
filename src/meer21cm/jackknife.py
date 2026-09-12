"""
Delete-one jackknife covariance of a measured 1D power spectrum.

Pass the same :class:`meer21cm.power.PowerSpectrum` instance used for the data
measurement to :class:`JackknifeCovariance`. The survey is split into
``ra_patch_num x dec_patch_num x los_patch_num`` patches. Line-of-sight
patches are linear in frequency (not redshift): galaxy redshifts are converted
to :math:`\\nu_g = f_{21}/(1+z_g)` so that map voxels and galaxies in the same
slab are removed together. Give the radial range as ``nu_range`` (Hz) or
``z_range`` (mutually exclusive).

Default estimator (``regrid=False``): grid the map (and galaxies) once, label
each Cartesian voxel from its centre (RA, Dec, frequency; the same bins as
:meth:`meer21cm.dataanalysis.Specification.get_gal_patch_labels`), and zero
those voxels in the painted fields and in ``counts_in_box``. The older
sky-mask + re-TSC operator is ``regrid=True``.

Each realisation is a fresh :class:`~meer21cm.power.PowerSpectrum` built from
the input attributes (same pattern as
:class:`~meer21cm.transfer.TransferFunction`). The enclosing box is fixed
once from the full ``W_HI`` so every realisation shares the data k-grid.
Realisations can run in serial, with ``multiprocessing``, or with MPI.

.. code-block:: python

    >>> from meer21cm.jackknife import JackknifeCovariance
    >>> jc = JackknifeCovariance(ps, ra_patch_num=4, dec_patch_num=3, los_patch_num=4)
    >>> results = jc.run(type="cross")
    >>> cov, mean_p1d = jc.get_covariance(results)
"""

from __future__ import annotations

from collections.abc import Sequence
from multiprocessing import Pool
from typing import Any, Literal
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .power import PowerSpectrum
from .util import f_21, redshift_to_freq
from .transfer import required_attrs

# string-literal aliases for the string-valued keyword arguments,
# following the convention of the explicitly typed modules.
JackknifeType = Literal["cross", "auto"]
WeightsScheme = Literal["validation", "gridded", "counts"]
GalaxyGridWeights = Literal["binary", "gridded"]
PoolKind = Literal["multiprocessing", "mpi", "serial"]

# the result of one jackknife realisation: the requested power spectrum,
# optionally followed by the 3D power spectrum and the two 1D auto-powers.
JackknifeResult = list[NDArray[np.floating]]

# box attributes held fixed across realisations, so that every realisation lands
# on the identical Cartesian grid, k-modes and mass-assignment compensation.
BOX_GEOMETRY_ATTRS = ("box_len", "box_ndim", "box_origin", "box_resol")

# attributes of the input PowerSpectrum instance that are propagated
# to the per-realisation instances, on top of ``meer21cm.transfer.required_attrs``.
# note that ra_range and dec_range are not propagated:
# Specification.__init__ invokes trim_map_to_range, which would trim the map of
# the per-realisation instances to the ranges, whereas the input map data must be
# reproduced exactly as it is (the input instance may have been used with an
# untrimmed map, and trimming would change the enclosing box and the k-modes).
extra_required_attrs: list[str] = [
    "seed",
    "downres_factor_radial",
    "downres_factor_transverse",
    "k1dbins",
]


def box_geometry(ps: PowerSpectrum) -> dict[str, NDArray]:
    """
    The box attributes that must be identical in every jackknife realisation.

    The enclosing box fixes the k-grid (hence the 1D binning), the power
    normalisation, and the mass-assignment compensation
    (:meth:`meer21cm.power.PowerSpectrum.gridding_compensation`, which is built
    from ``box_ndim``). Realisations that drifted onto a different box would not
    be comparable with the data measurement, so the geometry is recorded once
    and checked on every realisation.

    Parameters
    ----------
    ps: :class:`meer21cm.power.PowerSpectrum`
        An instance whose enclosing box has already been computed.

    Returns
    -------
    geometry: dict
        A mapping of attribute name to value for ``box_len``, ``box_ndim``,
        ``box_origin`` and ``box_resol``.
    """
    return {attr: np.array(getattr(ps, attr)) for attr in BOX_GEOMETRY_ATTRS}


def _digitize_axis(values: ArrayLike, edges: ArrayLike) -> NDArray[np.integer]:
    """
    ``np.digitize`` with values below the first edge mapped to the last bin.

    Matches :meth:`meer21cm.dataanalysis.Specification.get_gal_patch_labels`.
    Out-of-range indices are filtered by the caller.
    """
    indx = np.digitize(values, edges)
    indx[indx == 0] = len(edges)
    return indx - 1


def _digitize_sky_to_patch(
    ra: ArrayLike,
    dec: ArrayLike,
    nu: ArrayLike,
    ra_range: tuple[float, float],
    dec_range: tuple[float, float],
    nu_range: tuple[float, float],
    n_ra: int,
    n_dec: int,
    n_nu: int,
) -> NDArray[np.integer]:
    """
    Flattened jackknife patch index for sky coordinates.

    Same RA / Dec / frequency bins as
    :meth:`meer21cm.dataanalysis.Specification.get_gal_patch_labels`.
    Points outside the ranges get ``-1``.
    """
    ra = np.asarray(ra)
    ra_indx = _digitize_axis(
        (ra - ra_range[0]) % 360,
        np.linspace(0, (ra_range[1] - ra_range[0]) % 360, n_ra + 1),
    )
    dec_indx = _digitize_axis(dec, np.linspace(dec_range[0], dec_range[1], n_dec + 1))
    nu_indx = _digitize_axis(nu, np.linspace(nu_range[0], nu_range[1], n_nu + 1))
    inside = (
        (ra_indx >= 0)
        & (ra_indx < n_ra)
        & (dec_indx >= 0)
        & (dec_indx < n_dec)
        & (nu_indx >= 0)
        & (nu_indx < n_nu)
    )
    label = np.full(ra.shape, -1, dtype=int)
    label[inside] = (
        ra_indx[inside] * n_dec * n_nu + dec_indx[inside] * n_nu + nu_indx[inside]
    )
    return label


def box_voxel_patch_labels(
    ps: PowerSpectrum,
    ra_patch_num: int,
    dec_patch_num: int,
    los_patch_num: int,
    ra_range: tuple[float, float],
    dec_range: tuple[float, float],
    nu_range: tuple[float, float],
) -> NDArray[np.integer]:
    """
    Assign each Cartesian box voxel to a jackknife patch.

    The voxel centre is converted to (RA, Dec, frequency) and digitized
    with the same bins as
    :meth:`meer21cm.dataanalysis.Specification.get_gal_patch_labels`.
    Voxels outside the ranges get ``-1`` and are never removed.

    Parameters
    ----------
    ps: :class:`meer21cm.power.PowerSpectrum`
        An instance whose enclosing box has already been computed.
    ra_patch_num, dec_patch_num, los_patch_num: int
        The number of patches along each axis.
    ra_range, dec_range: tuple
        Sky ranges in degrees.
    nu_range: tuple
        Frequency range in Hz.

    Returns
    -------
    labels: np.ndarray of int
        Patch index per box voxel, shape ``ps.box_ndim``.
    """
    x_vec, y_vec, z_vec = ps.x_vec
    nx, ny, nz = (int(n) for n in ps.box_ndim)
    pos_xyz = np.empty((nx * ny * nz, 3), dtype=ps.real_dtype)
    pos_xyz[:, 0] = np.repeat(x_vec, ny * nz)
    pos_xyz[:, 1] = np.tile(np.repeat(y_vec, nz), nx)
    pos_xyz[:, 2] = np.tile(z_vec, nx * ny)
    ra, dec, z, _ = ps.ra_dec_z_for_coord_in_box(pos_xyz)
    return _digitize_sky_to_patch(
        ra,
        dec,
        redshift_to_freq(z),
        ra_range,
        dec_range,
        nu_range,
        ra_patch_num,
        dec_patch_num,
        los_patch_num,
    ).reshape(nx, ny, nz)


def _check_box_geometry(jk: PowerSpectrum, reference: dict[str, NDArray]) -> None:
    """
    Check that a realisation landed on the reference box.

    Parameters
    ----------
    jk: :class:`meer21cm.power.PowerSpectrum`
        The per-realisation instance, after ``get_enclosing_box()``.
    reference: dict
        The geometry recorded on the input instance, from :func:`box_geometry`.

    Raises
    ------
    ValueError
        If any geometry attribute differs from the reference.
    """
    for attr, ref in reference.items():
        got = np.asarray(getattr(jk, attr))
        ref = np.asarray(ref)
        if got.shape != ref.shape or not np.array_equal(got, ref):
            raise ValueError(
                f"the jackknife realisation has {attr}={got!r}, but the input "
                f"instance has {attr}={ref!r}. Every realisation must live on the "
                "identical box: the k-grid, the 1D binning and the gridding "
                "compensation all depend on it. Check that the input instance's "
                "box was computed with the final gridding settings (call "
                "ps.get_enclosing_box()) and that W_HI / map_has_sampling, the "
                "downres factors and the seed are not changed between the data "
                "measurement and the jackknife."
            )


def _normalise_weights_argument(weights, type_default: str = "cross"):
    """
    Validate and normalise the ``weights`` argument of :class:`JackknifeCovariance`.

    Accepts ``None`` (use the scheme's defaults), a single array (tracer 1), or a
    tuple with one entry per tracer. Each entry is either an array, used as the
    *grid* weights -- the library convention, since ``weights_1``/``weights_2``
    are aliases of ``weights_grid_1``/``weights_grid_2`` -- or a
    ``(field_weights, grid_weights)`` pair in which either element may be
    ``None``.

    Parameters
    ----------
    weights: None or array_like or tuple
        The user-supplied weights.
    type_default: str, default "cross"
        Set to ``"auto"`` to allow a single array without wrapping it in a tuple.

    Returns
    -------
    weights: None or tuple
        ``None``, or a 2-tuple of ``(field_weights, grid_weights)`` entries. A
        missing entry is ``(None, None)``.
    """
    if weights is None:
        return None
    if not isinstance(weights, (tuple, list)):
        weights = (weights,)
    if len(weights) == 1 and type_default == "auto":
        weights = (weights[0], None)
    if len(weights) not in (1, 2):
        raise ValueError(
            "weights must be None, a single array, or a tuple with one entry per "
            f"tracer; got {len(weights)} entries"
        )
    out = []
    for entry in weights:
        if entry is None:
            out.append((None, None))
        elif isinstance(entry, (tuple, list)):
            if len(entry) != 2:
                raise ValueError(
                    "a per-tracer weights entry must be an array or a "
                    f"(field_weights, grid_weights) pair; got {len(entry)} elements"
                )
            out.append((entry[0], entry[1]))
        else:
            out.append((None, entry))
    while len(out) < 2:
        out.append((None, None))
    return tuple(out)


class JackknifeCovariance:
    """
    Delete-one jackknife covariance of a 1D power spectrum.

    Each realisation zeros one patch on the already-gridded Cartesian fields
    (voxel-centre RA / Dec / frequency labels) and recomputes the power
    spectrum with the input settings. ``regrid=True`` is the older sky-mask
    + re-TSC operator. ``type="auto"`` returns the HI monopole;
    ``type="cross"`` returns the HI × galaxy cross-power (optionally the
    two autos). Realisations run on a fresh
    :class:`~meer21cm.power.PowerSpectrum`; if ``field_1`` / ``field_2``
    are missing, they are painted once onto the input instance so ``run``
    can be called without a prior ``grid_*``.

    .. code-block:: python

        >>> jc = JackknifeCovariance(ps, 4, 3, 4)
        >>> results = jc.run(type="cross")
        >>> cov, mean_p1d = jc.get_covariance(results)

    Parameters
    ----------
    ps: :class:`meer21cm.power.PowerSpectrum`
        The power spectrum instance used for the data measurement.
        It must have the map data (and the galaxy catalogue for ``type="cross"``)
        already read in, and the gridding settings
        (``downres_factor_transverse``, ``downres_factor_radial``,
        ``box_buffkick``, ``grid_scheme``, ...) already set.
        For ``type="cross"``, the instance must provide
        :meth:`meer21cm.dataanalysis.Specification.get_gal_patch_labels`.
    ra_patch_num: int
        The number of patch grids in the right ascension direction.
    dec_patch_num: int
        The number of patch grids in the declination direction.
    los_patch_num: int
        The number of patch grids in the line-of-sight direction.
        The line-of-sight is binned linearly in frequency.
    ra_range: tuple, default None
        The range of the right ascension in degrees. Default uses ``ps.ra_range``.
    dec_range: tuple, default None
        The range of the declination in degrees. Default uses ``ps.dec_range``.
    nu_range: tuple, default None
        The range of the frequency in Hz used for the line-of-sight split.
        Mutually exclusive with ``z_range``. If neither is provided, default uses
        ``[ps.nu.min() - ps.freq_resol/2, ps.nu.max() + ps.freq_resol/2]``,
        matching :meth:`meer21cm.dataanalysis.Specification.get_jackknife_patches`.
    z_range: tuple, default None
        The redshift range used for the line-of-sight split, as an alternative
        to ``nu_range``. It is converted internally to a frequency range via
        :math:`\\nu = f_{21}/(1+z)`, and the split is performed linearly in
        frequency for both the map and the galaxies.
    weights_scheme: str, default "validation"
        The scheme used to assign the power spectrum weights of each
        realisation. Make sure this matches what you did for the data
        measurement.

        ``"validation"`` (default) follows
        ``papers/validation/func_fullsim.py``: the HI grid weights are the
        jackknifed ``counts_in_box`` with uniform field weights, and the galaxy
        field weights are the dN/dz with grid weights
        ``(dN/dz > 0) * counts_in_box``. The dN/dz comes from
        ``ps.discrete_source_dndz`` when the instance has one (mock
        simulations); for real data pass ``weights`` explicitly, or the galaxy
        falls back to its binary occupancy.

        If ``"gridded"``, the field and grid weights of the HI map are the
        gridded jackknifed pixel weights ``w_HI`` (second output of
        ``grid_data_to_field``), and the galaxy weights are controlled by
        ``gal_grid_weights``.

        If ``"counts"``, the weights follow the cookbook convention:
        the HI grid weights are ``counts_in_box`` (inverse noise variance
        weighting) and the HI field weights are None; the galaxy field
        weights are the binary occupancy ``counts_in_box > 0`` and the
        galaxy grid weights are uniform. Note that in each realisation
        ``counts_in_box`` of the deleted voxels is zeroed (or, with
        ``regrid=True``, recomputed after zeroing the sky-pixel weights).
    weights: tuple, default None
        Explicit weights, overriding ``weights_scheme``. ``None`` uses the
        scheme's defaults.

        Otherwise a tuple with one entry per tracer (a single array is accepted
        and treated as tracer 1). Each entry is either

        * an array, used as the **grid** weights of that tracer -- the library
          convention, since ``weights_1``/``weights_2`` are aliases of
          ``weights_grid_1``/``weights_grid_2``; or
        * a ``(field_weights, grid_weights)`` pair, in which either element may
          be ``None`` (``None`` means uniform).

        For example, to reproduce the validation weighting of a simulation
        explicitly, pass ``weights=(counts, (dndz, (dndz > 0) * counts))`` where
        ``counts = ps.get_counts_in_box()``.

        The supplied arrays are restricted to the kept box voxels, so the
        removed patch always carries zero weight. They must live on the box
        of the input instance (call ``ps.get_enclosing_box()`` first).
    mean_center_1, unitless_1, mean_center_2, unitless_2: bool, default None
        Overrides for the per-realisation ``mean_center_*`` / ``unitless_*``
        flags. ``None`` keeps the input instance (tracer 1) or the values set
        by ``grid_gal_to_field`` (tracer 2, which forces both True). Used to
        run the HI cube through the galaxy pipeline and vice versa.
    freeze_mean: bool, default False
        If True, each subsample is centred / made unitless with the
        **full-sample** weighted field means recorded in ``__init__``, not
        the mean of the remaining volume. No effect when both mean-centering
        and unitless conversion are off.
    apply_taper: bool, default True
        Whether to apply the taper function ``ps.taper_func`` to the gridded
        weights of each field in each realisation
        (see :meth:`meer21cm.power.PowerSpectrum.apply_taper_to_field`).
        The taper is applied to the (already masked) grid weights of each
        realisation; make sure this setting matches the data measurement.
    taper_axis: tuple, default (2,)
        The box axes along which the taper is applied.
        Default is the z-axis, which is approximately the line-of-sight.
    gal_grid_weights: str, default "binary"
        Only used if ``weights_scheme="gridded"``.
        The scheme for the grid/field weights of the galaxy field.
        If ``"binary"``, the weights are the binary window of the kept
        voxels. If ``"gridded"``, the full-sample galaxy grid weights are
        restricted to the kept voxels.
    regrid: bool, default False
        If False (default), mask already-gridded box voxels. If True, zero
        the sky patch and the galaxy catalogue and TSC-paint each
        realisation from scratch.
    min_patch_weight_fraction: float, default 0.0
        Patches whose fraction of the total map weight ``(w_HI * W_HI)``
        is not greater than this value are skipped when ``run`` is called with
        ``patch_indices=None``. The default skips only completely empty patches
        (e.g. patches fully outside the survey footprint), whose removal would
        produce a realisation identical to the data and artificially shrink
        the covariance.
    pool: str, default "multiprocessing"
        The pool to use for parallelisation.
        Can be "multiprocessing", "mpi", or "serial" (no parallelisation).
    num_process: int, default None
        The number of processes to use for parallelisation.
        If not provided, the number of processes is set to the number of
        cores available.
    """

    def __init__(
        self,
        ps: PowerSpectrum,
        ra_patch_num: int,
        dec_patch_num: int,
        los_patch_num: int,
        ra_range: tuple[float, float] | None = None,
        dec_range: tuple[float, float] | None = None,
        nu_range: tuple[float, float] | None = None,
        z_range: tuple[float, float] | None = None,
        weights_scheme: WeightsScheme | str = "validation",
        weights: tuple | None = None,
        mean_center_1: bool | None = None,
        unitless_1: bool | None = None,
        mean_center_2: bool | None = None,
        unitless_2: bool | None = None,
        freeze_mean: bool = False,
        apply_taper: bool = True,
        taper_axis: tuple[int, ...] = (2,),
        gal_grid_weights: GalaxyGridWeights | str = "binary",
        regrid: bool = False,
        min_patch_weight_fraction: float = 0.0,
        pool: PoolKind | str = "multiprocessing",
        num_process: int | None = None,
    ) -> None:
        self.ps = ps
        self.ra_patch_num = ra_patch_num
        self.dec_patch_num = dec_patch_num
        self.los_patch_num = los_patch_num
        if ra_range is None:
            ra_range = ps.ra_range
        if dec_range is None:
            dec_range = ps.dec_range
        self.ra_range = tuple(ra_range)
        self.dec_range = tuple(dec_range)
        if (nu_range is not None) and (z_range is not None):
            raise ValueError(
                "nu_range and z_range are mutually exclusive, provide only one"
            )
        if z_range is not None:
            assert z_range[0] < z_range[1], "z_range[0] must be less than z_range[1]"
            nu_range = (f_21 / (1 + z_range[1]), f_21 / (1 + z_range[0]))
        if nu_range is None:
            nu_range = (
                ps.nu.min() - ps.freq_resol / 2,
                ps.nu.max() + ps.freq_resol / 2,
            )
        self.nu_range = tuple(nu_range)
        if weights_scheme not in ("validation", "gridded", "counts"):
            raise ValueError(f"Invalid weights_scheme: {weights_scheme}")
        self.weights_scheme = weights_scheme
        self.weights = _normalise_weights_argument(weights, type_default="cross")
        self.mean_center_1 = mean_center_1
        self.unitless_1 = unitless_1
        self.mean_center_2 = mean_center_2
        self.unitless_2 = unitless_2
        self.freeze_mean = bool(freeze_mean)
        self.apply_taper = apply_taper
        self.taper_axis = tuple(taper_axis)
        if gal_grid_weights not in ("binary", "gridded"):
            raise ValueError(f"Invalid gal_grid_weights: {gal_grid_weights}")
        self.gal_grid_weights = gal_grid_weights
        self.regrid = bool(regrid)
        self.min_patch_weight_fraction = min_patch_weight_fraction
        self.pool = pool
        self.num_process = num_process
        self.patch_indices_used: NDArray[np.integer] | None = None

        # Fix the box once, on the input instance, and record its geometry.
        # Every realisation re-derives the box from the propagated window and
        # seed; the geometry is checked against this reference so that a
        # realisation can never silently land on a different k-grid (which would
        # change the binning and the gridding compensation).
        ps.get_enclosing_box()
        self.box_geometry = box_geometry(ps)
        # The per-realisation instances are plain PowerSpectrum objects built
        # from required_attrs, which does not carry discrete_source_dndz. The
        # dN/dz is therefore evaluated once here, on the input instance's box,
        # which is identical to every realisation's box (see box_geometry).
        self.gal_dndz = _get_dndz_box(ps)
        # Full-sample field means, used when freeze_mean is on. Recorded after
        # the box is fixed so they match every realisation.
        self.field_mean_1 = _weighted_field_mean(ps, 1)
        self.field_mean_2 = _weighted_field_mean(ps, 2)

    @property
    def num_patches(self) -> int:
        """
        The total number of jackknife patches,
        ``ra_patch_num * dec_patch_num * los_patch_num``.
        """
        return self.ra_patch_num * self.dec_patch_num * self.los_patch_num

    @property
    def z_range(self) -> tuple[float, float]:
        """
        The redshift range corresponding to ``self.nu_range``.
        """
        return (f_21 / self.nu_range[1] - 1, f_21 / self.nu_range[0] - 1)

    def get_patch_masks(self) -> NDArray[np.bool_]:
        """
        The jackknife patch masks of the map, computed with
        :meth:`meer21cm.dataanalysis.Specification.get_jackknife_patches`
        and flattened along the patch dimensions.
        ``mask[i] = True`` marks the pixels that are removed in
        realisation ``i``.

        Returns
        -------
        mask_arr: np.ndarray of bool
            The patch masks, of shape ``(num_patches, *ps.W_HI.shape)``.
        """
        mask_arr = self.ps.get_jackknife_patches(
            self.ra_patch_num,
            self.dec_patch_num,
            self.los_patch_num,
            ra_range=self.ra_range,
            dec_range=self.dec_range,
            nu_range=self.nu_range,
        )
        return mask_arr.reshape((self.num_patches,) + self.ps.W_HI.shape)

    def get_gal_patch_labels(self) -> NDArray[np.integer]:
        """
        The jackknife patch label of each galaxy in ``ps``, computed with
        :meth:`meer21cm.dataanalysis.Specification.get_gal_patch_labels`.
        The galaxy redshifts are converted internally to 21cm line frequencies
        and digitized into the same frequency bins as the map patches, so the
        galaxy labels match the map patch indices exactly.
        Galaxy ``g`` is removed in realisation ``i`` if ``label[g] == i``.

        Returns
        -------
        label: np.ndarray of int
            Patch index for each galaxy, -1 for galaxies outside the ranges.
        """
        return self.ps.get_gal_patch_labels(
            self.ra_patch_num,
            self.dec_patch_num,
            self.los_patch_num,
            ra_range=self.ra_range,
            dec_range=self.dec_range,
            nu_range=self.nu_range,
        )

    def box_voxel_labels(self) -> NDArray[np.integer]:
        """
        Patch index of every Cartesian box voxel, from :func:`box_voxel_patch_labels`.
        """
        return box_voxel_patch_labels(
            self.ps,
            self.ra_patch_num,
            self.dec_patch_num,
            self.los_patch_num,
            self.ra_range,
            self.dec_range,
            self.nu_range,
        )

    def _ensure_gridded_fields(self, type: JackknifeType | str) -> None:
        """
        Paint missing Cartesian fields onto the input instance once.

        Realisations copy these arrays; they do not re-grid unless
        ``regrid=True``.
        """
        ps = self.ps
        painters = [("field_1", ps.grid_data_to_field)]
        if type == "cross":
            if getattr(ps, "ra_gal", None) is None or np.size(ps.ra_gal) == 0:
                raise ValueError(
                    "the galaxy catalogue of the input instance is empty, "
                    "cannot run type='cross'"
                )
            painters.append(("field_2", ps.grid_gal_to_field))
        for attr, paint in painters:
            if getattr(ps, attr, None) is None:
                painted, _, _ = paint()
                setattr(ps, attr, painted)

    def get_arg_list_for_parallel_boxmask(
        self,
        patch_indices: ArrayLike,
        type: JackknifeType | str = "cross",
        return_power_3d: bool = False,
        return_auto_power: bool = False,
    ) -> list[tuple[Any, ...]]:
        """
        Arguments for :func:`run_jackknife_boxmask` (the default estimator).
        """
        self._ensure_gridded_fields(type)
        labels = self.box_voxel_labels()
        field_1 = np.array(self.ps.field_1, copy=True)
        field_2 = np.array(self.ps.field_2, copy=True) if type == "cross" else None
        counts = np.array(self.ps.counts_in_box, copy=True)
        extra = self._jackknife_option_args()
        arg_list = []
        for j in patch_indices:
            arg_list.append(
                (
                    self.get_ps_instance_attr_dict(),
                    labels != j,
                    field_1,
                    field_2,
                    counts,
                    self.ps.k1dweights,
                    self.weights_scheme,
                    self.weights,
                    self.apply_taper,
                    self.taper_axis,
                    self.gal_grid_weights,
                    return_power_3d,
                    return_auto_power,
                    self.box_geometry,
                    self.gal_dndz,
                )
                + extra
            )
        return arg_list

    def patch_weight_fraction(self) -> NDArray[np.floating]:
        """
        The fraction of the total map weight ``(w_HI * W_HI)`` contained
        in each patch. Useful for identifying (nearly) empty patches and
        for checking how uniform the jackknife split is.

        Returns
        -------
        frac: np.ndarray
            The weight fraction of each patch, of shape ``(num_patches,)``.
        """
        # nan_to_num: real maps can store nan outside the sampled window,
        # which would otherwise propagate and flag every patch as empty
        w = np.nan_to_num(self.ps.w_HI) * self.ps.W_HI
        masks = self.get_patch_masks()
        return np.array([(w * m).sum() for m in masks]) / w.sum()

    def get_ps_instance_attr_dict(self) -> dict[str, Any]:
        """
        Generate the attribute dictionary for the per-realisation
        power spectrum instances.
        It reads the attributes listed in
        :data:`meer21cm.transfer.required_attrs` and
        :data:`meer21cm.jackknife.extra_required_attrs`
        from the input power spectrum instance.
        Note that ``seed`` is propagated so that the enclosing box
        (which uses random sampling of the pixels) is identical in every
        realisation and identical to the one of the data measurement.

        Returns
        -------
        attr_dict: dict
            The attribute dictionary for the per-realisation instances.
        """
        attr_dict = {}
        for attr in required_attrs + extra_required_attrs:
            attr_dict[attr] = getattr(self.ps, attr)
        return attr_dict

    def _jackknife_option_args(self) -> tuple:
        """
        The pickleable extra arguments shared by the auto and cross workers.

        Order must match the trailing parameters of ``run_jackknife_auto``
        and ``run_jackknife_cross``.
        """
        return (
            self.mean_center_1,
            self.unitless_1,
            self.mean_center_2,
            self.unitless_2,
            self.freeze_mean,
            self.field_mean_1,
            self.field_mean_2,
        )

    def get_default_patch_indices(self) -> NDArray[np.integer]:
        """
        The indices of the patches used by default in ``run``:
        all patches whose weight fraction is greater than
        ``min_patch_weight_fraction``.

        Returns
        -------
        patch_indices: np.ndarray of int
        """
        frac = self.patch_weight_fraction()
        return np.where(frac > self.min_patch_weight_fraction)[0]

    def get_arg_list_for_parallel_auto(
        self,
        patch_indices: ArrayLike,
        return_power_3d: bool = False,
        return_auto_power: bool = False,
    ) -> list[tuple[Any, ...]]:
        """
        Generate a list of arguments for parallelisation of the auto-power runs.
        This list is then used for ``pool.starmap``.

        Parameters
        ----------
        patch_indices: list
            The indices of the patches to remove, one per realisation.
        return_power_3d: bool, default False
            Whether to also return the 3D power spectrum of each realisation.
        return_auto_power: bool, default False
            Dummy argument to keep consistency with the cross runs.

        Returns
        -------
        arg_list: list
            The list of arguments for the parallelisation.
        """
        masks = self.get_patch_masks()
        extra = self._jackknife_option_args()
        arg_list = []
        for j in patch_indices:
            attr_dict = self.get_ps_instance_attr_dict()
            arg_list.append(
                (
                    attr_dict,
                    masks[j],
                    self.ps.k1dweights,
                    self.weights_scheme,
                    self.weights,
                    self.apply_taper,
                    self.taper_axis,
                    return_power_3d,
                    self.box_geometry,
                    self.gal_dndz,
                )
                + extra
            )
        return arg_list

    def get_arg_list_for_parallel_cross(
        self,
        patch_indices: ArrayLike,
        return_power_3d: bool = False,
        return_auto_power: bool = False,
    ) -> list[tuple[Any, ...]]:
        """
        Generate a list of arguments for parallelisation of the cross-power runs.
        This list is then used for ``pool.starmap``.

        Parameters
        ----------
        patch_indices: list
            The indices of the patches to remove, one per realisation.
        return_power_3d: bool, default False
            Whether to also return the 3D cross-power spectrum of each realisation.
        return_auto_power: bool, default False
            Whether to also return the 1D auto-power spectra of the two fields
            of each realisation.

        Returns
        -------
        arg_list: list
            The list of arguments for the parallelisation.
        """
        if self.ps.ra_gal.size == 0:
            raise ValueError(
                "the galaxy catalogue of the input instance is empty, "
                "cannot run type='cross'"
            )
        masks = self.get_patch_masks()
        labels = self.get_gal_patch_labels()
        ra_gal = self.ps.ra_gal
        dec_gal = self.ps.dec_gal
        z_gal = self.ps.z_gal
        extra = self._jackknife_option_args()
        arg_list = []
        for j in patch_indices:
            attr_dict = self.get_ps_instance_attr_dict()
            keep_gal = labels != j
            gal_radecz = (ra_gal[keep_gal], dec_gal[keep_gal], z_gal[keep_gal])
            arg_list.append(
                (
                    attr_dict,
                    masks[j],
                    gal_radecz,
                    self.ps.k1dweights,
                    self.weights_scheme,
                    self.weights,
                    self.apply_taper,
                    self.taper_axis,
                    self.gal_grid_weights,
                    return_power_3d,
                    return_auto_power,
                    self.box_geometry,
                    self.gal_dndz,
                )
                + extra
            )
        return arg_list

    def run(
        self,
        patch_indices: ArrayLike | None = None,
        type: JackknifeType | str = "cross",
        return_power_3d: bool = False,
        return_auto_power: bool = False,
    ) -> list[JackknifeResult]:
        """
        Run the jackknife realisations.

        Note that, ``run`` automatically uses a parallel pool to loop over
        the patches (unless ``pool="serial"``).
        If you believe the parallel behaviour is not as expected,
        you can manually extract the argument list and map the function yourself.
        For example:

        .. code-block:: python

            >>> jc = JackknifeCovariance(ps, 4, 3, 4)
            >>> results = jc.run(type="auto")

        is the same as:

        .. code-block:: python

            >>> jc = JackknifeCovariance(ps, 4, 3, 4)
            >>> arg_list = jc.get_arg_list_for_parallel_boxmask(
            ...     jc.get_default_patch_indices(), type="auto"
            ... )
            >>> results = []
            >>> with Pool(jc.num_process) as pool:
            >>>     for result_i in pool.starmap(run_jackknife_boxmask, arg_list):
            >>>         results.append(result_i)

        Parameters
        ----------
        patch_indices: list, default None
            The indices of the patches to remove, one per realisation.
            If None, all patches with weight fraction greater than
            ``min_patch_weight_fraction`` are used
            (see :meth:`get_default_patch_indices`).
            The indices actually used are stored in ``self.patch_indices_used``.
        type: str, default "cross"
            The type of power spectrum to compute for each realisation.
            Can be "cross" (HI x galaxy) or "auto" (HI x HI).
        return_power_3d: bool, default False
            Whether to also return the 3D power spectrum of each realisation.
        return_auto_power: bool, default False
            Only used if ``type="cross"``. Whether to also return the 1D
            auto-power spectra of the HI map and of the galaxy field of
            each realisation.

        Returns
        -------
        results_arr: list
            The list of results, with each element being a sublist for the
            result of one realisation.
            The first element of each sublist is the 1D power spectrum.
            If ``return_power_3d`` is True, the next element is the 3D power
            spectrum. If ``return_auto_power`` is True (cross only), the next
            two elements are the 1D auto-power of field 1 (HI) and of
            field 2 (galaxy).
        """
        if type not in ("cross", "auto"):
            raise ValueError(f"Invalid type: {type}")
        if patch_indices is None:
            patch_indices = self.get_default_patch_indices()
        if len(patch_indices) == 0:
            raise ValueError(
                "no jackknife patch to use. Check that the patch splits overlap "
                "the survey footprint (see self.patch_weight_fraction()) and "
                "that ra_range, dec_range and nu_range/z_range are correct."
            )
        self.patch_indices_used = np.asarray(patch_indices)
        run_func = {
            (False, "auto"): run_jackknife_boxmask,
            (False, "cross"): run_jackknife_boxmask,
            (True, "auto"): run_jackknife_auto,
            (True, "cross"): run_jackknife_cross,
        }[self.regrid, type]
        arg_list = (
            getattr(self, f"get_arg_list_for_parallel_{type}")(
                patch_indices, return_power_3d, return_auto_power
            )
            if self.regrid
            else self.get_arg_list_for_parallel_boxmask(
                patch_indices, type, return_power_3d, return_auto_power
            )
        )
        return _map_realisations(run_func, arg_list, self.pool, self.num_process)

    @staticmethod
    def jackknife_covariance(
        power_arr: ArrayLike,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        r"""
        The delete-one jackknife covariance matrix of the input realisations,

        .. math::
            C = \frac{N - 1}{N}
            \sum_{j=1}^{N} (P_j - \bar{P}) (P_j - \bar{P})^T

        where :math:`N` is the number of realisations and
        :math:`\bar{P}` is the mean of the realisations.

        Parameters
        ----------
        power_arr: np.ndarray
            The 1D power spectra of the realisations,
            of shape ``(num_realisations, num_kbins)``.

        Returns
        -------
        cov: np.ndarray
            The jackknife covariance matrix,
            of shape ``(num_kbins, num_kbins)``.
        mean: np.ndarray
            The mean of the realisations, of shape ``(num_kbins,)``.
        """
        power_arr = np.asarray(power_arr)
        num_jack = power_arr.shape[0]
        if num_jack < 2:
            raise ValueError(f"at least 2 realisations are needed, got {num_jack}")
        mean = power_arr.mean(axis=0)
        delta = power_arr - mean[None, :]
        cov = (num_jack - 1) / num_jack * np.einsum("ja,jb->ab", delta, delta)
        return cov, mean

    def get_covariance(
        self,
        results_arr: Sequence[Sequence[NDArray[np.floating]]],
        element: int = 0,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        Compute the jackknife covariance matrix from the output of ``run``.

        Parameters
        ----------
        results_arr: list
            The output of :meth:`run`.
        element: int, default 0
            The element of each result sublist to use.
            Default is the first element, i.e. the 1D power spectrum
            of the requested type.

        Returns
        -------
        cov: np.ndarray
            The jackknife covariance matrix.
        mean: np.ndarray
            The mean of the realisations.
        """
        power_arr = np.array([result[element] for result in results_arr])
        return self.jackknife_covariance(power_arr)


# this must be pickleable inputs for multiprocessing
def _check_k1dweights(jk: PowerSpectrum, k_sel_3d_to_1d: ArrayLike | None) -> None:
    """
    Check that the input ``k1dweights`` (``k_sel``) matches the k-mode grid
    of the realisation. A mismatched array (e.g. built from ``ps.k_mode``
    before ``ps.get_enclosing_box()`` was called, when the box is still
    the placeholder unit box and the k arrays have shape ``(1, 1, 1)``) would
    otherwise broadcast silently and typically zero out every k-bin,
    turning the whole 1D power spectrum into nan.
    """
    if k_sel_3d_to_1d is None:
        return
    k_sel_3d_to_1d = np.asarray(k_sel_3d_to_1d)
    if k_sel_3d_to_1d.shape != jk.k_mode.shape:
        raise ValueError(
            f"k1dweights (k_sel) has shape {k_sel_3d_to_1d.shape}, but the "
            f"k-mode grid of the jackknife realisation has shape "
            f"{jk.k_mode.shape}. k1dweights must be computed on the same "
            "enclosing box as the data measurement: call "
            "ps.get_enclosing_box() (with the final gridding settings) "
            "before building k_sel from ps.k_mode / k_perp / k_para, "
            "and then set ps.k1dweights = k_sel."
        )


def _map_realisations(
    run_func,
    arg_list: list[tuple[Any, ...]],
    pool: PoolKind | str,
    num_process: int | None,
) -> list[JackknifeResult]:
    """Map ``run_func`` over ``arg_list`` with the requested pool."""
    if pool == "serial":
        return [run_func(*args) for args in arg_list]
    pool_cls = {"multiprocessing": Pool}.get(pool)
    if pool == "mpi":
        from mpi4py.futures import MPIPoolExecutor

        pool_cls = MPIPoolExecutor
    if pool_cls is None:
        raise ValueError(f"Invalid pool: {pool}")
    with pool_cls(num_process) as executor:
        return list(executor.starmap(run_func, arg_list))


def _fresh_ps(
    ps_attr_dict: dict[str, Any],
    box_reference: dict[str, NDArray] | None,
    k_sel_3d_to_1d: ArrayLike | None,
) -> PowerSpectrum:
    """A realisation instance on the reference box, with matching k-weights."""
    jk = PowerSpectrum(**ps_attr_dict)
    jk.get_enclosing_box()
    if box_reference is not None:
        _check_box_geometry(jk, box_reference)
    _check_k1dweights(jk, k_sel_3d_to_1d)
    jk.include_beam = ps_attr_dict["include_beam"]
    jk.include_sky_sampling = ps_attr_dict["include_sky_sampling"]
    return jk


def _bind_tracer(
    jk: PowerSpectrum,
    tracer: int,
    field: ArrayLike,
    *,
    weights_scheme: WeightsScheme | str,
    weights: tuple | None,
    counts_keep_rg: NDArray[np.floating] | None,
    weights_rg: NDArray[np.floating],
    gal_grid_weights: GalaxyGridWeights | str,
    ps_attr_dict: dict[str, Any],
    mean_center: bool | None,
    unitless: bool | None,
    freeze_mean: bool,
    field_mean: float | None,
    apply_taper: bool,
    taper_axis: tuple[int, ...],
    gal_dndz: NDArray[np.floating] | None = None,
    gal_weights_rg: NDArray[np.floating] | None = None,
    gal_window_rg: NDArray[np.floating] | None = None,
) -> None:
    """Attach one tracer's field, weights, flags, and optional taper."""
    setattr(jk, f"field_{tracer}", np.asarray(field))
    field_w, grid_w = resolve_tracer_weights(
        tracer=tracer,
        weights_scheme=weights_scheme,
        weights=weights,
        jk=jk,
        counts_keep_rg=counts_keep_rg,
        weights_rg=weights_rg,
        gal_grid_weights=gal_grid_weights,
        gal_weights_rg=gal_weights_rg,
        gal_window_rg=gal_window_rg,
        gal_dndz=gal_dndz,
    )
    setattr(jk, f"weights_field_{tracer}", field_w)
    setattr(jk, f"weights_grid_{tracer}", grid_w)
    _apply_jackknife_field_settings(
        jk, tracer, ps_attr_dict, mean_center, unitless, freeze_mean, field_mean
    )
    _materialise_none_grid_weights(jk, tracer)
    if apply_taper:
        jk.apply_taper_to_field(tracer, axis=list(taper_axis))


def _measure_result(
    jk: PowerSpectrum,
    power_name: str,
    k_sel_3d_to_1d: ArrayLike | None,
    return_power_3d: bool,
    extra_names: tuple[str, ...] = (),
) -> JackknifeResult:
    """1D power of ``power_name``, optionally the 3D cube and extra 1D autos."""
    p1d, _, _ = jk.get_1d_power(power_name, k1dweights=k_sel_3d_to_1d)
    extras_3d = (getattr(jk, power_name),) if return_power_3d else ()
    extras_1d = tuple(
        jk.get_1d_power(name, k1dweights=k_sel_3d_to_1d)[0] for name in extra_names
    )
    return [p1d, *extras_3d, *extras_1d]


# this must be pickleable inputs for multiprocessing
def _materialise_none_grid_weights(jk: PowerSpectrum, tracer: int) -> None:
    """
    Replace ``None`` grid weights of a tracer with an explicit uniform array.

    ``None`` means uniform throughout this pipeline, but
    :meth:`meer21cm.power.PowerSpectrum.apply_taper_to_field` multiplies the
    weights in place and therefore cannot handle ``None``. Materialising the
    array keeps the semantics identical (``None`` is read as ones by
    ``get_weights_none_to_one``) while allowing the taper to be applied.

    Parameters
    ----------
    jk: :class:`meer21cm.power.PowerSpectrum`
        The per-realisation instance.
    tracer: int
        ``1`` for the HI map, ``2`` for the galaxy field.
    """
    if getattr(jk, f"weights_grid_{tracer}") is None:
        setattr(
            jk,
            f"weights_grid_{tracer}",
            np.ones(jk.box_ndim, dtype=jk.real_dtype),
        )


def resolve_tracer_weights(
    tracer: int,
    weights_scheme: WeightsScheme | str,
    weights: tuple | None,
    jk: PowerSpectrum,
    counts_keep_rg: NDArray[np.floating] | None,
    weights_rg: NDArray[np.floating],
    gal_grid_weights: GalaxyGridWeights | str = "binary",
    gal_weights_rg: NDArray[np.floating] | None = None,
    gal_window_rg: NDArray[np.floating] | None = None,
    gal_dndz: NDArray[np.floating] | None = None,
) -> tuple[ArrayLike | None, ArrayLike | None]:
    """
    Resolve the ``(field_weights, grid_weights)`` of one tracer in one realisation.

    In this pipeline the *grid* weights are the ones that matter for the data
    power spectrum: they are multiplied into the field before the FFT and used
    for the mean-centring, while the *field* weights enter only through the
    renormalisation factor ``power_weights_renorm(grid_w * field_w, ...)``.

    The default, ``weights_scheme="validation"``, reproduces the weighting of
    ``papers/validation/func_fullsim.py``:

    ``tracer=1`` (HI)
        field weights ``None`` (uniform), grid weights ``counts_in_box`` of the
        jackknifed map.
    ``tracer=2`` (galaxy)
        field weights ``dN/dz`` evaluated on the box voxel redshifts, grid
        weights ``(dN/dz > 0) * counts_in_box`` of the jackknifed map. The
        dN/dz is taken from ``ps.discrete_source_dndz`` when available (mock
        simulations); otherwise the tracer-2 grid weights fall back to the
        binary occupancy and the field weights to ``None``, and the user should
        pass ``weights`` explicitly.

    When ``weights`` is given it takes precedence over the scheme. The supplied
    arrays are restricted to the region kept by the jackknife (box cells with
    ``weights_rg > 0``) so that the removed patch carries zero weight; arrays
    that are already zero inside the patch are unaffected.

    Parameters
    ----------
    tracer: int
        ``1`` for the HI map, ``2`` for the galaxy field.
    weights_scheme: str
        The scheme to use when ``weights`` is None.
    weights: tuple or None
        The normalised user weights, from
        :func:`_normalise_weights_argument`. Entry ``tracer - 1`` is used.
    jk: :class:`meer21cm.power.PowerSpectrum`
        The per-realisation instance.
    counts_keep_rg: np.ndarray or None
        The jackknifed ``counts_in_box``; required by the ``"validation"`` and
        ``"counts"`` schemes and by ``gal_grid_weights="binary"``.
    weights_rg: np.ndarray
        The gridded jackknifed pixel weights, ``(weights_rg > 0)`` marking the
        kept window.
    gal_grid_weights: str, default "binary"
        Only used by ``weights_scheme="gridded"`` for tracer 2.

    Returns
    -------
    field_weights: np.ndarray or None
        The field-level weights of the tracer (``None`` means uniform).
    grid_weights: np.ndarray or None
        The grid-level weights of the tracer (``None`` means uniform).
    """
    kept = weights_rg > 0
    if weights is not None:
        return _restrict_user_weights(weights[tracer - 1], kept)
    schemes = {
        "validation": lambda: _validation_tracer_weights(
            tracer, counts_keep_rg, weights_rg, jk, gal_dndz
        ),
        "counts": lambda: _counts_tracer_weights(tracer, counts_keep_rg, weights_rg),
        "gridded": lambda: _gridded_tracer_weights(
            tracer,
            kept,
            weights_rg,
            counts_keep_rg,
            jk,
            gal_grid_weights,
            gal_weights_rg,
            gal_window_rg,
        ),
    }
    try:
        return schemes[weights_scheme]()
    except KeyError as exc:
        raise ValueError(f"Invalid weights_scheme: {weights_scheme}") from exc


def _restrict_user_weights(
    entry: tuple[ArrayLike | None, ArrayLike | None],
    kept: NDArray[np.bool_],
) -> tuple[ArrayLike | None, ArrayLike | None]:
    """Restrict a ``(field, grid)`` pair to the kept box voxels."""
    field_w, grid_w = entry
    field_w = None if field_w is None else np.asarray(field_w) * kept
    grid_w = None if grid_w is None else np.asarray(grid_w) * kept
    return field_w, grid_w


def _require_counts(
    counts_keep_rg: NDArray[np.floating] | None, scheme: str
) -> NDArray[np.floating]:
    if counts_keep_rg is None:
        raise ValueError(f"counts_keep_rg is required by weights_scheme={scheme!r}")
    return counts_keep_rg


def _validation_tracer_weights(
    tracer: int,
    counts_keep_rg: NDArray[np.floating] | None,
    weights_rg: NDArray[np.floating],
    jk: PowerSpectrum,
    gal_dndz: NDArray[np.floating] | None,
) -> tuple[ArrayLike | None, ArrayLike | None]:
    counts = _require_counts(counts_keep_rg, "validation")
    if tracer == 1:
        return None, counts
    dndz = gal_dndz if gal_dndz is not None else _get_dndz_box(jk)
    if dndz is not None:
        return dndz, (dndz > 0) * counts
    warnings.warn(
        "weights_scheme='validation' needs a dN/dz for the galaxy field "
        "weights, but none is available on this instance (it is not a "
        "mock simulation and gal_dndz was not passed). Falling back to "
        "the binary occupancy of the jackknifed lightcone, which is NOT "
        "the validation weighting. Pass weights=(w1, (dndz, grid2)) or "
        "gal_dndz=dndz explicitly.",
        stacklevel=2,
    )
    return (counts > 0).astype(weights_rg.dtype), np.ones_like(weights_rg)


def _counts_tracer_weights(
    tracer: int,
    counts_keep_rg: NDArray[np.floating] | None,
    weights_rg: NDArray[np.floating],
) -> tuple[ArrayLike | None, ArrayLike | None]:
    counts = _require_counts(counts_keep_rg, "counts")
    if tracer == 1:
        return None, counts
    return (counts > 0).astype(weights_rg.dtype), np.ones_like(weights_rg)


def _gridded_tracer_weights(
    tracer: int,
    kept: NDArray[np.bool_],
    weights_rg: NDArray[np.floating],
    counts_keep_rg: NDArray[np.floating] | None,
    jk: PowerSpectrum,
    gal_grid_weights: GalaxyGridWeights | str,
    gal_weights_rg: NDArray[np.floating] | None,
    gal_window_rg: NDArray[np.floating] | None,
) -> tuple[ArrayLike | None, ArrayLike | None]:
    if tracer == 1:
        return weights_rg, weights_rg
    if gal_grid_weights == "gridded":
        if gal_weights_rg is None:
            raise ValueError(
                "gal_grid_weights='gridded' needs the gridded galaxy weights, "
                "which are only available for the cross jackknife"
            )
        return gal_weights_rg, gal_weights_rg
    if gal_grid_weights != "binary":
        raise ValueError(f"Invalid gal_grid_weights: {gal_grid_weights}")
    window = gal_window_rg
    if window is None:
        window = getattr(jk, "weights_field_2", None)
    if window is None:
        window = (counts_keep_rg > 0) if counts_keep_rg is not None else kept
    window = np.asarray(window, dtype=weights_rg.dtype) * kept
    return window, window


def _weighted_field_mean(ps: PowerSpectrum, tracer: int) -> float | None:
    """
    The weighted mean of one tracer's field on ``ps``.

    Uses the current grid weights (``weights_{tracer}``). Returns ``None``
    when that field is missing.
    """
    field = getattr(ps, f"field_{tracer}", None)
    if field is None:
        return None
    weights = ps.get_weights_none_to_one(f"weights_{tracer}")
    return float(np.sum(weights * field) / np.sum(weights))


def _apply_jackknife_field_settings(
    jk: PowerSpectrum,
    tracer: int,
    ps_attr_dict: dict[str, Any],
    mean_center_override: bool | None,
    unitless_override: bool | None,
    freeze_mean: bool,
    field_mean: float | None,
) -> None:
    """
    Apply mean-center / unitless overrides and an optional frozen field mean.

    Tracer 1 restores the input-instance flags when the override is ``None``.
    Tracer 2 leaves whatever ``grid_gal_to_field`` set unless an override is
    given.
    """
    for name, override in (
        (f"mean_center_{tracer}", mean_center_override),
        (f"unitless_{tracer}", unitless_override),
    ):
        value = (
            override
            if override is not None
            else (ps_attr_dict[name] if tracer == 1 else None)
        )
        if value is not None:
            setattr(jk, name, value)
    if freeze_mean and field_mean is not None:
        setattr(jk, f"field_mean_{tracer}", field_mean)


def _get_dndz_box(jk: PowerSpectrum) -> NDArray[np.floating] | None:
    """
    The dN/dz of the discrete tracers evaluated on the box voxel redshifts.

    Parameters
    ----------
    jk: :class:`meer21cm.power.PowerSpectrum`
        The per-realisation instance.

    Returns
    -------
    dndz: np.ndarray or None
        The dN/dz per box voxel, or ``None`` when the instance has no
        ``discrete_source_dndz`` callable (i.e. it is not a mock simulation).
    """
    dndz_func = getattr(jk, "discrete_source_dndz", None)
    if dndz_func is None or not callable(dndz_func):
        return None
    dndz = np.asarray(dndz_func(jk.box_voxel_redshift))
    return np.broadcast_to(dndz, jk.box_ndim).astype(jk.real_dtype, copy=True)


def _grid_jackknifed_map(
    jk: PowerSpectrum,
    patch_mask: ArrayLike,
    return_kept_counts: bool = False,
    box_reference: dict[str, NDArray] | None = None,
    return_kept_window: bool = False,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating] | None,]:
    """
    Apply the jackknife mask to the map data and pixel weights of a
    per-realisation instance, and grid the jackknifed map to the Cartesian box.

    The enclosing box is computed before masking, from the full ``W_HI``
    window, so that every realisation lives on the identical grid and k-modes.
    ``W_HI`` is left untouched (the pixel coordinates used for the gridding are
    tied to the ``W_HI`` footprint used to compute the box); the masking enters
    through the map data, the pixel weights ``w_HI``, and consequently the
    gridded weights, which define the jackknifed window on the box.

    Parameters
    ----------
    jk: :class:`meer21cm.power.PowerSpectrum`
        The per-realisation instance.
    patch_mask: np.ndarray of bool
        The mask of the removed patch, True for removed pixels.
    return_kept_counts: bool, default False
        Whether to also compute the number of kept map voxels falling in
        each box cell (the jackknife-consistent analogue of
        ``counts_in_box``). This requires a second gridding pass.
    box_reference: dict, default None
        The box geometry recorded on the input instance, from
        :func:`box_geometry`. If given, the realisation's own box is checked
        against it and a ``ValueError`` is raised on any mismatch.
    return_kept_window: bool, default False
        Whether to also return the boolean mask of box cells that are covered by
        the jackknifed window ``(weights_rg > 0)``. Used to restrict
        user-supplied weights to the kept region.

    Returns
    -------
    map_rg: np.ndarray
        The gridded jackknifed map.
    weights_rg: np.ndarray
        The gridded jackknifed weights.
    counts_keep_rg: np.ndarray or None
        The number of kept map voxels per box cell,
        or None if ``return_kept_counts`` is False.
    """
    # identical box for all realisations: computed from the full window
    # with the seed propagated from the input instance
    jk.get_enclosing_box()
    if box_reference is not None:
        _check_box_geometry(jk, box_reference)
    keep = (~np.asarray(patch_mask, dtype=bool)) * (jk.W_HI > 0)
    keep = keep.astype(jk.data.dtype)
    # nan_to_num: real maps can store nan outside the sampled window;
    # nan in the pixel weights would propagate through the gridding and
    # turn the whole power spectrum into nan
    jk.data = np.nan_to_num(jk.data) * keep
    jk.weights_map_pixel = np.nan_to_num(jk.weights_map_pixel) * keep
    map_rg, weights_rg, _ = jk.grid_data_to_field()
    counts_keep_rg = None
    if return_kept_counts or return_kept_window:
        # the jackknife-consistent analogue of ``counts_in_box``:
        # get_counts_in_box grids the pixel weights ``w_HI`` (single particle
        # pass, no interlacing), which at this point are already jackknifed,
        # so this reduces exactly to ``ps.counts_in_box`` when no patch
        # is removed
        counts_keep_rg = jk.get_counts_in_box()
    return map_rg, weights_rg, counts_keep_rg


def run_jackknife_boxmask(
    ps_attr_dict: dict[str, Any],
    keep: ArrayLike,
    field_1: ArrayLike,
    field_2: ArrayLike | None,
    counts: ArrayLike,
    k_sel_3d_to_1d: ArrayLike | None = None,
    weights_scheme: WeightsScheme | str = "validation",
    weights: tuple | None = None,
    apply_taper: bool = True,
    taper_axis: tuple[int, ...] = (2,),
    gal_grid_weights: GalaxyGridWeights | str = "binary",
    return_power_3d: bool = False,
    return_auto_power: bool = False,
    box_reference: dict[str, NDArray] | None = None,
    gal_dndz: NDArray[np.floating] | None = None,
    mean_center_1: bool | None = None,
    unitless_1: bool | None = None,
    mean_center_2: bool | None = None,
    unitless_2: bool | None = None,
    freeze_mean: bool = False,
    field_mean_1: float | None = None,
    field_mean_2: float | None = None,
) -> JackknifeResult:
    """
    One delete-one realisation on already-gridded Cartesian fields.

    ``keep`` is the boolean box mask of voxels that remain. The painted
    fields and ``counts`` are multiplied by it; the survey is not re-gridded.
    """
    keep = np.asarray(keep, dtype=bool)
    jk = _fresh_ps(ps_attr_dict, box_reference, k_sel_3d_to_1d)
    weights_rg = keep.astype(jk.real_dtype, copy=False)
    counts_keep = np.asarray(counts) * keep
    bind_kw = dict(
        weights_scheme=weights_scheme,
        weights=weights,
        counts_keep_rg=counts_keep,
        weights_rg=weights_rg,
        gal_grid_weights=gal_grid_weights,
        ps_attr_dict=ps_attr_dict,
        freeze_mean=freeze_mean,
        apply_taper=apply_taper,
        taper_axis=taper_axis,
        gal_dndz=gal_dndz,
    )
    _bind_tracer(
        jk,
        1,
        np.asarray(field_1) * keep,
        mean_center=mean_center_1,
        unitless=unitless_1,
        field_mean=field_mean_1,
        **bind_kw,
    )
    if field_2 is None:
        return _measure_result(jk, "auto_power_3d_1", k_sel_3d_to_1d, return_power_3d)
    _bind_tracer(
        jk,
        2,
        np.asarray(field_2) * keep,
        mean_center=mean_center_2,
        unitless=unitless_2,
        field_mean=field_mean_2,
        **bind_kw,
    )
    extra = ("auto_power_3d_1", "auto_power_3d_2") if return_auto_power else ()
    return _measure_result(jk, "cross_power_3d", k_sel_3d_to_1d, return_power_3d, extra)


def run_jackknife_auto(
    ps_attr_dict: dict[str, Any],
    patch_mask: ArrayLike,
    k_sel_3d_to_1d: ArrayLike | None = None,
    weights_scheme: WeightsScheme | str = "validation",
    weights: tuple | None = None,
    apply_taper: bool = True,
    taper_axis: tuple[int, ...] = (2,),
    return_power_3d: bool = False,
    box_reference: dict[str, NDArray] | None = None,
    gal_dndz: NDArray[np.floating] | None = None,
    mean_center_1: bool | None = None,
    unitless_1: bool | None = None,
    mean_center_2: bool | None = None,
    unitless_2: bool | None = None,
    freeze_mean: bool = False,
    field_mean_1: float | None = None,
    field_mean_2: float | None = None,
) -> JackknifeResult:
    """
    Compute the 1D auto-power spectrum of the HI map for one jackknife
    realisation, with the patch given by ``patch_mask`` removed.

    Parameters
    ----------
    ps_attr_dict: dict
        The attribute dictionary to initialize the per-realisation instance.
    patch_mask: np.ndarray of bool
        The mask of the removed patch, True for removed pixels.
    k_sel_3d_to_1d: np.ndarray, default None
        The weights for averaging the 3D power spectrum k-modes to
        the 1D power spectrum.
    weights_scheme: str, default "validation"
        The scheme used to assign the power spectrum weights,
        see :class:`JackknifeCovariance`.
    weights: tuple, default None
        Explicit weights, see :class:`JackknifeCovariance`.
    apply_taper: bool, default True
        Whether to apply the taper function to the gridded weights.
    taper_axis: tuple, default (2,)
        The box axes along which the taper is applied.
    return_power_3d: bool, default False
        Whether to also return the 3D power spectrum.
    box_reference: dict, default None
        The box geometry to check each realisation against, from
        :func:`box_geometry`.

    Returns
    -------
    result: list
        The list of results. The first element is the 1D auto-power spectrum.
        If ``return_power_3d`` is True, the second element is the
        3D auto-power spectrum.
    """
    jk = PowerSpectrum(**ps_attr_dict)
    map_rg, weights_rg, counts_keep_rg = _grid_jackknifed_map(
        jk,
        patch_mask,
        return_kept_counts=(weights_scheme in ("validation", "counts")),
        box_reference=box_reference,
    )
    _check_k1dweights(jk, k_sel_3d_to_1d)
    jk.include_beam = ps_attr_dict["include_beam"]
    jk.include_sky_sampling = ps_attr_dict["include_sky_sampling"]
    _bind_tracer(
        jk,
        1,
        map_rg,
        weights_scheme=weights_scheme,
        weights=weights,
        counts_keep_rg=counts_keep_rg,
        weights_rg=weights_rg,
        gal_grid_weights="binary",
        ps_attr_dict=ps_attr_dict,
        mean_center=mean_center_1,
        unitless=unitless_1,
        freeze_mean=freeze_mean,
        field_mean=field_mean_1,
        apply_taper=apply_taper,
        taper_axis=taper_axis,
        gal_dndz=gal_dndz,
    )
    return _measure_result(jk, "auto_power_3d_1", k_sel_3d_to_1d, return_power_3d)


def run_jackknife_cross(
    ps_attr_dict: dict[str, Any],
    patch_mask: ArrayLike,
    gal_radecz: tuple[ArrayLike, ArrayLike, ArrayLike],
    k_sel_3d_to_1d: ArrayLike | None = None,
    weights_scheme: WeightsScheme | str = "validation",
    weights: tuple | None = None,
    apply_taper: bool = True,
    taper_axis: tuple[int, ...] = (2,),
    gal_grid_weights: GalaxyGridWeights | str = "binary",
    return_power_3d: bool = False,
    return_auto_power: bool = False,
    box_reference: dict[str, NDArray] | None = None,
    gal_dndz: NDArray[np.floating] | None = None,
    mean_center_1: bool | None = None,
    unitless_1: bool | None = None,
    mean_center_2: bool | None = None,
    unitless_2: bool | None = None,
    freeze_mean: bool = False,
    field_mean_1: float | None = None,
    field_mean_2: float | None = None,
) -> JackknifeResult:
    """
    Compute the 1D HI x galaxy cross-power spectrum for one jackknife
    realisation, with the patch given by ``patch_mask`` removed from the map
    and the galaxies inside the patch removed from the catalogue.

    Parameters
    ----------
    ps_attr_dict: dict
        The attribute dictionary to initialize the per-realisation instance.
    patch_mask: np.ndarray of bool
        The mask of the removed patch, True for removed pixels.
    gal_radecz: tuple of np.ndarray
        The (ra, dec, z) of the galaxies kept in this realisation.
    k_sel_3d_to_1d: np.ndarray, default None
        The weights for averaging the 3D power spectrum k-modes to
        the 1D power spectrum.
    weights_scheme: str, default "validation"
        The scheme used to assign the power spectrum weights,
        see :class:`JackknifeCovariance`.
    weights: tuple, default None
        Explicit weights, see :class:`JackknifeCovariance`.
    apply_taper: bool, default True
        Whether to apply the taper function to the gridded weights
        of both fields.
    taper_axis: tuple, default (2,)
        The box axes along which the taper is applied.
    gal_grid_weights: str, default "binary"
        The scheme for the grid/field weights of the galaxy field,
        see :class:`JackknifeCovariance`.
    return_power_3d: bool, default False
        Whether to also return the 3D cross-power spectrum.
    return_auto_power: bool, default False
        Whether to also return the 1D auto-power spectra of the two fields.

    Returns
    -------
    result: list
        The list of results. The first element is the 1D cross-power spectrum.
        If ``return_power_3d`` is True, the next element is the 3D cross-power
        spectrum. If ``return_auto_power`` is True, the next two elements are
        the 1D auto-power of field 1 (HI) and of field 2 (galaxy).
    """
    jk = PowerSpectrum(**ps_attr_dict)
    map_rg, weights_rg, counts_keep_rg = _grid_jackknifed_map(
        jk,
        patch_mask,
        return_kept_counts=(weights_scheme in ("validation", "counts")),
        box_reference=box_reference,
    )
    _check_k1dweights(jk, k_sel_3d_to_1d)
    ra_gal, dec_gal, z_gal = gal_radecz
    jk._ra_gal = np.asarray(ra_gal)
    jk._dec_gal = np.asarray(dec_gal)
    jk._z_gal = np.asarray(z_gal)
    gal_map_rg, gal_weights_rg, _ = jk.grid_gal_to_field()
    gal_window_rg = np.array(jk.weights_field_2, copy=True)
    jk.include_beam = ps_attr_dict["include_beam"]
    jk.include_sky_sampling = ps_attr_dict["include_sky_sampling"]
    bind_kw = dict(
        weights_scheme=weights_scheme,
        weights=weights,
        counts_keep_rg=counts_keep_rg,
        weights_rg=weights_rg,
        gal_grid_weights=gal_grid_weights,
        ps_attr_dict=ps_attr_dict,
        freeze_mean=freeze_mean,
        apply_taper=apply_taper,
        taper_axis=taper_axis,
        gal_dndz=gal_dndz,
        gal_weights_rg=gal_weights_rg,
        gal_window_rg=gal_window_rg,
    )
    _bind_tracer(
        jk,
        1,
        map_rg,
        mean_center=mean_center_1,
        unitless=unitless_1,
        field_mean=field_mean_1,
        **bind_kw,
    )
    _bind_tracer(
        jk,
        2,
        gal_map_rg,
        mean_center=mean_center_2,
        unitless=unitless_2,
        field_mean=field_mean_2,
        **bind_kw,
    )
    extra = ("auto_power_3d_1", "auto_power_3d_2") if return_auto_power else ()
    return _measure_result(jk, "cross_power_3d", k_sel_3d_to_1d, return_power_3d, extra)
