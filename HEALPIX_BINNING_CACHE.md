# HEALPix gridding: cached voxel→pixel binning

Implementation notes for the caching of the HEALPix binning used by
`grid_field_to_sky_map(..., fmt="healpix")` and, transitively,
`MockSimulation.propagate_mock_field_to_data(...)`.

- **Status**: implemented and verified (see [§8](#8-verification-performed)).
- **Files touched**: `src/meer21cm/grid.py`, `CHANGELOG.rst`, this document.
- **Public API**: unchanged. No signature, default or return-value change; the
  WCS path and the mock propagation loop are untouched.

| | |
|---|---|
| Problem | Every gridding call recomputed the full voxel → (HEALPix pixel, frequency channel) geometry (~75 % of runtime) |
| Fix | Build that mapping **once**, cache it on the instance, and reduce each call to two `np.bincount` passes |
| Steady-state speed-up | **13–30×** (1.55 s → 0.05–0.11 s for the profiled box) |
| First call | Unchanged (~1.6 s; it performs the build) |
| Outputs | Bitwise identical to the previous implementation at `precision=True`; ≤ 2.5×10⁻⁷ relative at `precision=False` |
| Extra memory | 4 bytes/voxel (20 MB for the profiled box) |

---

## 1. Call path and where the time went

### 1.1 Call path

```
MockSimulation.propagate_mock_field_to_data(field, beam, average)     mock.py:1406
└── _propagate_mock_field_to_data_healpix(field, beam, average)       mock.py:1368
    └── for each LOS batch ch_sel in _iter_field_los_chunks(field):   mock.py:1381
        └── grid_field_to_sky_map(field_chunk, los_sel=ch_sel, ...)   grid.py:1836
            └── _grid_field_to_sky_map_healpix(...)                   grid.py:1783
                ├── geometry: ra_dec_z_for_coord_in_box               grid.py:1575
                │            hp.ang2pix, find_ch_id, searchsorted
                └── accumulate: np.add.at  →  now np.bincount
```

`PowerSpectrum`/`MockSimulation` obtain the gridding mixin through
`PowerSpectrum(LightconeGriddingMixin, FieldPowerSpectrum, ModelPowerSpectrum)`
(`power.py:104`), so everything below lives in `LightconeGriddingMixin`
(`grid.py:642`).

The LOS batching helper is `_iter_last_axis_batches`
(`dataanalysis.py:455`, re-defined identically in `mock.py:165`): it splits
`np.arange(nz)` with `np.array_split(..., batch_number)` and drops empty
splits, so `batch_number=1` still exercises one full-width batch.

### 1.2 Profile of the *previous* implementation

Measured with `cProfile` on a 181×181×155 box (5,077,955 voxels), 1881
HEALPix pixels at `nside=128`, 50 frequency channels, `precision=True`:

| component | time | share |
|---|---:|---:|
| **total `grid_field_to_sky_map` call** | **1.556 s** | 100 % |
| `ra_dec_z_for_coord_in_box` (cumulative) | 0.880 s | 57 % |
| ↳ `hp.vec2ang` | 0.352 s | 23 % |
| ↳ `np.sum` over the rotated coordinates | 0.147 s | 9 % |
| ↳ `interp1d` redshift-from-comoving-distance | 0.129 s | 8 % |
| ↳ `np.einsum` (inverse rotation) | 0.113 s | 7 % |
| `hp.ang2pix` | 0.131 s | 8 % |
| `np.searchsorted` against `pixel_id` | 0.095 s | 6 % |
| `find_ch_id` (`np.digitize`) | 0.058 s | 4 % |
| `np.add.at` × 2 (the actual binning) | 0.050 s | **3 %** |
| `_grid_field_to_sky_map_healpix` self time (allocation, masks, fancy indexing) | 0.281 s | 18 % |
| `np.repeat` / `np.tile` (position cube) | 0.037 s | 2 % |
| `redshift_to_freq` | 0.032 s | 2 % |

Two conclusions drove the design:

1. **~75 % of the cost is geometry** (rotation, `z(χ)` interpolation,
   `vec2ang`, `ang2pix`, `searchsorted`, `digitize`) that is *identical* on
   every call, because it only depends on the box, the cosmology, the channel
   grid and the skymap — never on the field being gridded.
2. **`np.add.at` is only ~3 %** at this size. Replacing the accumulation
   alone (e.g. with `np.bincount`) would have bought almost nothing; the win
   comes from caching the geometry. The accumulation was replaced anyway
   because it falls out of the chosen representation for free.

---

## 2. Design decisions

### 2.1 Representation: flat index array + `np.bincount` (chosen) vs `scipy.sparse` matrix

The original proposal was to build a matrix `S` with shape
`(n_pix · n_ch, n_voxel)` such that binning is `S @ field.ravel()`. That is
mathematically exactly what the index array encodes — with one nonzero per
column, `S` is fully determined by a single integer per voxel:

```
idx[m] = row * n_ch + ch        (m = flat voxel index, C order)
```

and `S @ x` becomes `bincount(idx, weights=x)`.

| | flat `idx` array + `bincount` (**chosen**) | `scipy.sparse.csr_matrix` |
|---|---|---|
| Memory per voxel | 4 B (int32) / 8 B (int64) | ≈ 12–17 B (int32/int64 column index + `data` + `indptr`) |
| Memory for the profiled box | **20.3 MB** | ≈ 61–86 MB |
| Build cost | one pass, no sort | COO→CSR conversion (counting sort over rows) |
| Accumulation kernel | `np.bincount` (single tight C loop, one scatter) | `csr_matvec` (indirect gather + scatter) |
| Arbitrary `los_sel` | `idx[..., los_sel]` — a strided slice, works for *any* selection | column slicing of a CSR, or a per-selection rebuild |
| Extra dependency | none | `scipy.sparse` (scipy is already a dependency) |
| Inspectability | one integer array | a single linear-operator object |

The index array was chosen because it is leaner, needs no sparse build, and —
decisively — supports the *arbitrary* `los_sel` of the public API (tests pass
e.g. `np.arange(7)` or strided selections) as a plain slice instead of a
column subset of a sparse matrix. `scipy.sparse` remains a drop-in
alternative if a reusable operator object is ever wanted.

### 2.2 Cache scope: per-instance (chosen) vs process-wide LRU

The cache lives on the instance (`self._hp_binning_idx`), validated by a
fingerprint. A process-wide LRU keyed by geometry was considered and
rejected: flows such as `transfer.py` rebuild `MockSimulation` for every
realisation with identical geometry, so a shared cache would help, but it
introduces global mutable state, forces a fully value-based key (cosmology
parameters, digest of `pixel_id`), and needs an eviction policy for what can
be a large array. Per-instance keeps the lifetime and ownership of the array
obvious. This can be added later *on top of* the current design without
touching the binning code (only the key and lookup change).

### 2.3 Invalidation: value fingerprint + held object identity (chosen) vs the `@tagging`/`*_dep_attr` framework

The codebase has a dependency framework: public properties carry
`@tagging("box", "nu", ...)` (`util.py:518`), `find_property_with_tags`
(`util.py:503`) collects them in `Specification.__init__`
(`dataanalysis.py:238-252`) into `<tag>_dep_attr` lists, and parameter setters
call `clean_cache` (`dataanalysis.py:431`) which sets the backing `_name`
attributes to `None` so the property recomputes lazily.

It was **not** usable here, for two reasons:

1. `get_enclosing_box()` writes `self._box_origin` and
   `self._rot_mat_sky_to_box` **directly** (`grid.py:949-957`), bypassing the
   `box_origin` setter — a `"box"`-tagged cache would go stale after a box
   regeneration with an unchanged `box_ndim` value. (Only the subsequent
   `self.box_ndim = ndim_rg` at `grid.py:1019` happens to clean it.)
2. There is no tag for the skymap/`hp_nside`/`pixel_id` at all: `hp_nside` and
   `pixel_id` are plain read-through properties of `self.skymap`
   (`dataanalysis.py:501-512`).

A value fingerprint cannot go stale: it is recomputed from the *actual*
quantities the builder consumes, and any mismatch simply triggers a rebuild
(the failure mode is only a redundant build, never a wrong map).

### 2.4 The mock LOS-chunk loop was left in place

`_propagate_mock_field_to_data_healpix` (`mock.py:1381-1392`) still loops over
LOS batches and merges `map_i`/`counts_i`. Collapsing it into a single
full-box call would avoid `batch_number` full-map additions, but:

- it preserves the existing memory bound (the `bincount` weights buffer is
  `n_voxel · 8` bytes for a float32 field, i.e. `1/batch_number` of it per
  batch);
- it keeps the chunk-merge behaviour under test
  (`tests/test_mock.py::test_propagate_mock_field_average_false_chunk_merge`);
- it keeps the diff confined to one file.

`mock.py` therefore needed **no changes at all**.

---

## 3. Data structure

`self._hp_binning_idx` — built once per geometry.

| property | value |
|---|---|
| shape | `tuple(self.box_ndim)` = `(nx, ny, nz)` |
| dtype | `int32` if `n_row + 1 ≤ 2³¹ − 1`, else `int64` (`grid.py:1743`) |
| semantics | flat output index `row * n_ch + ch` of the voxel at that grid cell |
| out-of-survey / out-of-band value | `n_row` (the *scratch bin*, see below) where `n_row = n_pix · n_ch` |
| memory | `nx · ny · nz · itemsize` — 20.3 MB for the profiled box |
| ordering | C order, i.e. `idx.ravel()[m]` ↔ `field.ravel()[m]` for a field of shape `box_ndim` |

Accompanying cache attributes (all set together at `grid.py:1778-1780`):

| attribute | content |
|---|---|
| `_hp_binning_fingerprint_cache` | the geometry fingerprint tuple of §4.1 |
| `_hp_binning_refs_cache` | `(skymap, pixel_id array, z(χ) interpolator)` — held by strong reference so `is` comparisons are sound |
| `_hp_binning_idx` | the index array |

The **scratch bin** trick: invalid voxels are given the index `n_row` rather
than being masked out. `np.bincount` therefore produces exactly
`n_row + 1` bins (guaranteed by `minlength=n_row + 1` and
`max(idx) + 1 = n_row + 1`), and the caller drops the last one with
`[:n_row]`. Because every voxel contributes to exactly one bin, invalid
voxels can only ever write to the discarded bin — no cross-talk, and no
per-voxel boolean masks are needed at apply time.

---

## 4. Implementation walkthrough

All references are to `src/meer21cm/grid.py` as of this change.

### 4.1 `_hp_binning_fingerprint()` — `grid.py:1665`

```python
nu = np.asarray(self.nu)
return (
    tuple(int(v) for v in self.box_ndim),
    tuple(float(v) for v in np.asarray(self.box_len, dtype=float)),
    tuple(float(v) for v in np.asarray(self.box_origin, dtype=float)),
    np.asarray(self.rot_mat_sky_to_box, dtype=float).ravel().tobytes(),
    nu.shape,
    nu.tobytes(),
    np.dtype(self.real_dtype).str,
)
```

| component | guards against |
|---|---|
| `box_ndim` | grid resolution changes (downres factors, regeneration) |
| `box_len` | box extent (`get_enclosing_box`, `box_buffkick`) |
| `box_origin` | box translation — written *directly* by `get_enclosing_box` |
| `rot_mat_sky_to_box` | box re-orientation (72 bytes, exact) |
| `nu.shape`, `nu.tobytes()` | channel grid changes (count *or* values) |
| `real_dtype` | dtype of the coordinate temporaries |

Everything is a `tuple`/`bytes`/`str`/`int`/`float` — never a NumPy array — so
`fingerprint == cached_fingerprint` is a plain, exact Python comparison with
no ambiguous-truth-value errors. Floats are compared bit-exactly (a
*stricter* test than needed; the failure mode is a harmless rebuild).

Not in the fingerprint, handled by identity instead (§4.2):
`self.skymap` (carries `hp_nside` and `pixel_id`), `self.skymap.pixel_id`,
`self.z_as_func_of_comov_dist` (the `interp1d` built from the fiducial
cosmology, `cosmology.py:916-937`).

### 4.2 `_get_hp_binning_indices()` — `grid.py:1685`

```python
fingerprint = self._hp_binning_fingerprint()
refs = (self.skymap, self.skymap.pixel_id, self.z_as_func_of_comov_dist)
cached_fingerprint = getattr(self, "_hp_binning_fingerprint_cache", None)
cached_refs = getattr(self, "_hp_binning_refs_cache", None)
if (
    cached_fingerprint is not None
    and cached_fingerprint == fingerprint
    and cached_refs is not None
    and all(a is b for a, b in zip(cached_refs, refs))
):
    return self._hp_binning_idx
return self._build_hp_binning_indices(fingerprint, refs)
```

Why `is` for the three objects is safe: the cache *holds a strong reference*
to each of them, so their identities cannot be recycled by the allocator
while they are cached. If any is replaced (new skymap assigned, `pixel_id`
array re-created, cosmology change invalidating `_z_as_func_of_comov_dist`
via `cosmo_fid_dep_attr`), the comparison fails and the map is rebuilt.

Cost of a cache hit: three attribute lookups, one tuple build over ~1500
floats + `nu.tobytes()` (a few KB) and three identity checks — microseconds,
against a 1.5 s build.

`getattr(..., None)` also makes the first call (no attributes yet) and any
external `clean_cache` (which sets attributes to `None`) behave correctly.

### 4.3 `_build_hp_binning_indices(fingerprint, refs)` — `grid.py:1716`

Setup (`grid.py:1737-1750`):

```python
nside = int(self.hp_nside)
pixel_id = np.asarray(self.pixel_id, dtype=np.int64)
n_out, n_ch = pixel_id.size, int(self.nu.size)
n_row = n_out * n_ch
scratch = n_row + 1
idx_dtype = np.int32 if scratch <= np.iinfo(np.int32).max else np.int64
order = np.argsort(pixel_id, kind="mergesort")   # stable: original indices
pix_sorted = pixel_id[order]

nx, ny, nz = (int(n) for n in self.box_ndim)
idx = np.full((nx, ny, nz), scratch, dtype=idx_dtype)
```

Per LOS batch (`grid.py:1751-1777`), for `sel` in
`self._iter_last_axis_batches(nz)`:

1. **Build the voxel-centre cube for this batch** — identical arithmetic to
   the old implementation, so the geometry is bit-for-bit the same:
   ```python
   pos_xyz[:, 0] = np.repeat(x_vec, ny * nz_sel)   # i slowest
   pos_xyz[:, 1] = np.tile(np.repeat(y_vec, nz_sel), nx)
   pos_xyz[:, 2] = np.tile(z_vec, nx * ny)         # k fastest
   ```
2. **Sky coordinates** — `self.ra_dec_z_for_coord_in_box(pos_xyz)`
   (`grid.py:1575`): inverse rotation via `np.einsum`, comoving distance,
   `z(χ)` via `interp1d`, `hp.vec2ang`.
3. **Pixel and channel per voxel** —
   `hp.ang2pix(nside, ..., lonlat=True)` and
   `find_ch_id(redshift_to_freq(pos_z), self.nu)` (`util.py:555`).
4. **Two filters** (unchanged logic from the old code):
   ```python
   valid_ch = (ch_idx >= 0) & (ch_idx < n_ch)
   valid_pos = np.flatnonzero(valid_ch)          # positions in the batch cube
   ...
   row_s = np.searchsorted(pix_sorted, hpix)
   in_bounds = row_s < n_out                     # searchsorted may land past the end
   in_survey = np.zeros(hpix.shape, dtype=bool)
   in_survey[in_bounds] = pix_sorted[row_s[in_bounds]] == hpix[in_bounds]
   ```
5. **Scatter into the batch cube** (the new part):
   ```python
   chunk = np.full((nx, ny, nz_sel), scratch, dtype=idx_dtype)
   flat = order[row_s[in_survey]].astype(np.int64) * n_ch + ch_idx[in_survey]
   chunk.reshape(-1)[valid_pos[in_survey]] = flat.astype(idx_dtype)
   idx[..., sel] = chunk
   ```
   `chunk.reshape(-1)` is a view (C-contiguous), so the assignment writes in
   place. `order[row_s[...]]` maps the sorted-pixel rank back to the original
   `pixel_id` position — the output row index — exactly as the old `row`
   variable did.

Finally the three cache attributes are stored (`grid.py:1778-1780`) and the
array returned.

**Ordering invariant.** For a field of shape `(nx, ny, nz)` in C order the
flat voxel index is `m = (i·ny + j)·nz + k`, which is precisely the order in
which `pos_xyz` is generated (repeat on the slowest axis, tile on the
fastest). Therefore `idx.ravel()[m]` describes `field.ravel()[m]`, and for a
LOS selection `sel` the pair
`(field[..., sel].ravel(), idx[..., sel].ravel())` stays aligned — which is
what makes the chunked merge in `mock.py` correct.

**Batching.** Geometry is evaluated batch by batch, so the largest temporary
is `nx · ny · (nz/batch_number) · 3 · 8` bytes for `pos_xyz` (122 MB at
`batch_number=1` for the profiled box) — the same peak as before the change.
The *cached* array is the only new long-lived allocation.

### 4.4 `_grid_field_to_sky_map_healpix(field, average, mask, los_sel)` — `grid.py:1783`

```python
los_sel = np.arange(self.box_ndim[2], dtype=int) if los_sel is None else np.asarray(los_sel, dtype=int)
expected_shape = (self.box_ndim[0], self.box_ndim[1], los_sel.size)
if field.shape != expected_shape:
    raise ValueError(...)                              # unchanged behaviour

idx = self._get_hp_binning_indices()
nz = int(self.box_ndim[2])
if not (los_sel.size == nz and np.array_equal(los_sel, np.arange(nz))):
    idx = idx[..., los_sel]                            # copy only for partial/permuted LOS

n_out = int(np.asarray(self.pixel_id).size)
n_ch  = int(self.nu.size)
n_row = n_out * n_ch
real_dtype = self.real_dtype

flat  = idx.ravel()
mass  = np.asarray(field, dtype=real_dtype).ravel()
map_sum = np.bincount(flat, weights=mass, minlength=n_row + 1)[:n_row]
cnt     = np.bincount(flat,              minlength=n_row + 1)[:n_row]
map_bin    = map_sum.reshape(n_out, n_ch).astype(real_dtype, copy=False)
count_bin  = cnt.reshape(n_out, n_ch).astype(real_dtype, copy=False)
if average:
    with np.errstate(divide="ignore", invalid="ignore"):
        map_bin = np.where(count_bin > 0, map_bin / count_bin, 0.0)
if mask:
    map_bin *= self.W_HI
return map_bin, count_bin
```

Notes:

- **`np.array_equal(los_sel, arange(nz))`** avoids copying the index array
  for the common full-LOS case (the cache is then shared, not mutated). Any
  partial, strided or permuted selection takes the slicing branch, which is
  correct for *any* index array (duplicates included — the old code would also
  have applied such a selection twice).
- **Weights**: `np.asarray(field, dtype=real_dtype)` is a no-op for an
  already-`real_dtype` array, so a contiguous field is passed to `bincount`
  by reference; a non-contiguous LOS chunk is copied in C order by
  `.ravel()`, matching the indexing order.
- **Accumulation dtype**: `np.bincount` accumulates in `float64`
  (`int64` for the counts pass) and the result is cast to `real_dtype`. The
  old code accumulated directly into a `real_dtype` buffer via `np.add.at`.
  See §5.3.
- **`average` / `mask` / return contract** are byte-for-byte the previous
  logic, including `np.where(count > 0, ...)` (no NaN for empty pixels) and
  masking only the map, not the counts.
- The full `(n_pix, n_ch)` output is allocated per call (as before); for
  `batch_number > 1` the caller still sums the per-batch outputs.

### 4.5 What did *not* change

- `grid_field_to_sky_map` dispatch (`grid.py:1836`, healpix branch at
  `grid.py:1897`) and its docstring contract for `los_sel`.
- `_grid_field_to_sky_map_wcs` — same caching opportunity exists but was out
  of scope.
- `_propagate_mock_field_to_data_healpix` and the WCS variant in `mock.py`.
- `beam` handling (`convolve_data(..., kernel=None)`), `W_HI` masking,
  `average` semantics.

---

## 5. Correctness argument

### 5.1 Equivalence of the mapping

For every voxel, both the old and the new code compute

```
row = order[searchsorted(sorted_pixel_id, ang2pix(nside, ra(i,j,k), dec(i,j,k)))]
ch  = find_ch_id(redshift_to_freq(z(i,j,k)), nu)
```

with the same two validity tests (`0 ≤ ch < n_ch`, pixel present in
`pixel_id`) and the same stable `argsort`. The old code then ran
`np.add.at(map, (row, ch), mass)`; the new code writes `row·n_ch + ch` and
runs `np.bincount`. Both traverse voxels in the same `(i, j, k)` order, and
within a given output cell the additions therefore happen in that same
traversal order — by `np.add.at` in the old code, by `bincount` (which walks
its weights array in index order) in the new one. The per-cell sums are thus
*bitwise* identical for `float64`, not merely close; this was confirmed
experimentally (§5.3).

`np.add.at(cnt, (row, ch), 1.0)` is replaced by an unweighted `bincount`,
which counts into `int64` and is cast to `real_dtype` — exact for any
realistic count (integers are representable in `float32` up to 2²⁴).

### 5.2 The scratch bin cannot leak

`idx` holds either a valid value in `[0, n_row)` or exactly `n_row`.
`bincount(..., minlength=n_row + 1)` yields indices `0 … n_row`; `[:n_row]`
drops only the scratch bin. Valid bins never receive contributions from
invalid voxels, and the scratch bin is discarded before `reshape`. If a field
contains `NaN`/`Inf`, its contribution likewise lands in the scratch bin when
the voxel is invalid — identical observable behaviour to the old code, which
filtered those voxels out before `add.at`.

### 5.3 Numerical semantics

| quantity | old | new | effect |
|---|---|---|---|
| mass accumulation | `np.add.at` into a `real_dtype` buffer | `np.bincount(weights=...)` in `float64`, cast to `real_dtype` | float64: **bitwise identical**; float32: differences at float32 round-off |
| counts | `np.add.at(..., 1.0)` into `real_dtype` | `bincount` in `int64`, cast to `real_dtype` | exact; bitwise identical |
| averaging | `np.where(cnt > 0, sum / cnt, 0.0)` | unchanged | identical |
| masking | `map_bin *= self.W_HI` | unchanged | identical |

Measured against a line-for-line re-implementation of the previous algorithm:

| `precision` | `average` | bitwise equal | counts equal | max abs diff | max rel diff |
|---|---|---|---|---|---|
| `True` (float64) | True | ✅ | ✅ | 0 | 0 |
| `True` (float64) | False | ✅ | ✅ | 0 | 0 |
| `False` (float32) | True | ❌ | ✅ | 1.8×10⁻⁷ | 1.3×10⁻⁷ |
| `False` (float32) | False | ❌ | ✅ | 7.6×10⁻⁶ | 2.5×10⁻⁷ |

The float32 differences are the expected consequence of accumulating in
double precision before the cast — i.e. the new path is, if anything, the
more accurate of the two. All existing tests use statistical tolerances and
pass in both precisions.

---

## 6. Cache invalidation matrix

| Trigger | Detected by | Rebuild? |
|---|---|---|
| `get_enclosing_box()` (writes `_box_origin`, `_rot_mat_sky_to_box`, `_box_len`, `box_ndim` directly) | fingerprint (`box_ndim`, `box_len`, `box_origin`, `rot_mat`) | ✅ |
| `downres_factor_transverse/radial`, `box_buffkick`, `num_particle_per_pixel`, `ra_range`/`dec_range` + regeneration | change the box → fingerprint | ✅ |
| `nu = ...` (channel grid) | fingerprint `nu.shape` + `nu.tobytes()` | ✅ |
| `fiducial_cosmology = ...` (`cosmology.py:546-550` clears `_z_as_func_of_comov_dist`) | identity of `z_as_func_of_comov_dist` | ✅ |
| `self.skymap = HealpixSkyMap(...)` (different `nside`/`pixel_id`) | identity of `self.skymap` and of `self.skymap.pixel_id` | ✅ |
| `precision` | no setter exists (immutable after `__init__`); fingerprint also carries `real_dtype` | ✅ (defensive) |
| `batch_number` | *not* a key component — batches only shape the build, the cached array is full-box | n/a (correct by construction) |
| unchanged geometry, repeated call | all checks pass | ❌ cache hit |
| first call | `getattr(...) is None` | ✅ build |

Verified experimentally: after each of `nu`, box regeneration,
`fiducial_cosmology` and skymap replacement the cached array object changed
(`id(...)` differs), and after a mutate-and-restore cycle
(`nu` → regenerate box → original `nu` → regenerate box) the map reproduced
the original **bitwise**, confirming both invalidation and rebuild
correctness.

**Deliberately not covered** (unsupported mutation patterns):

- in-place modification of `self._nu[...]` or `self.skymap._pixel_id[...]`
  (both properties are documented as read-only; a value change without a new
  array object would not be detected);
- changing `z_interp_max` without touching the cosmology (the existing
  `z_as_func_of_comov_dist` cache has the same gap);
- mutating `box_origin`/`rot_mat_sky_to_box` in place (no setters exist).

In all of these the *old* code would also have been affected only through
the mutated values — but the old code recomputed geometry from scratch each
call and would therefore have seen them. They are noted as known limits of
the fingerprint approach; switching to a digest of those arrays would close
them at the cost of an O(n) check per call.

---

## 7. Performance and memory

### 7.1 Wall-clock (181×181×155 box, 1881 px, 50 ch, `precision=True`)

| case | before | after | speed-up |
|---|---:|---:|---:|
| `grid_field_to_sky_map(average=True, mask=True)` (first call = build) | 1.475 s | 1.643 s | 0.9× |
| `grid_field_to_sky_map(average=True, mask=False)` | 1.521 s | 0.054 s | 28.3× |
| `grid_field_to_sky_map(average=False, mask=True)` | 1.542 s | 0.052 s | 29.5× |
| `grid_field_to_sky_map(average=False, mask=False)` | 1.557 s | 0.058 s | 27.0× |
| repeat call (steady state) | 1.561 s | **0.051 s** | **30.4×** |
| mock propagate, `batch_number=1`, `average=True` (build) | 1.608 s | 1.700 s | 0.9× |
| mock propagate, `batch_number=1`, `average=False` | 1.692 s | 0.115 s | 14.7× |
| mock propagate, `batch_number=3`, `average=True` (build) | 1.356 s | 1.435 s | 0.9× |
| mock propagate, `batch_number=3`, `average=False` | 1.380 s | 0.109 s | 12.7× |

Reading: the build costs about one old call (it runs the same geometry once).
Every later grid call on the same object is ~0.05 s; a full mock propagation
— which additionally copies each LOS chunk out of the field and merges the
per-batch maps — settles at ~0.11 s. A caller that grids *N* fields with a
fixed geometry pays `1.6 + 0.05·(N−1)` seconds instead of `1.55·N` — at
*N* = 10 that is ≈ 7.5× overall, and the ratio tends to 30× as *N* grows.

### 7.2 Memory

| item | bytes | profiled box |
|---|---|---|
| cached `idx` (int32 when `n_row + 1 < 2³¹`) | `4 · n_voxel` | 20.3 MB |
| cached `idx` (int64 fallback for very large maps) | `8 · n_voxel` | — |
| (rejected) CSR alternative | ≈ 12–17 · `n_voxel` | ≈ 61–86 MB |
| transient `pos_xyz` during build | `24 · n_voxel / batch_number` | 122 MB at `batch_number=1` (unchanged vs. before) |
| per-call `bincount` output | `8 · (n_pix·n_ch + 1)` | 0.75 MB |
| per-call counts output | same | 0.75 MB |

The `int32`/`int64` switch (`grid.py:1743`) triggers when
`n_pix · n_ch + 1 > 2³¹ − 1`, e.g. `nside=1024` full-sky (12.6 M pixels) with
more than ~170 channels.

---

## 8. Verification performed

1. **Reference capture (before editing).** For a fixed seed, the previous
   implementation was run on: `average ∈ {T,F}` × `mask ∈ {T,F}`; two partial
   `los_sel` selections (`np.arange(7)`, strided); mock propagation with
   `batch_number ∈ {1,3}` × `average ∈ {T,F}`. Outputs and timings were
   pickled to `/tmp/hp_ref.pkl`.
2. **Equivalence.** The new code was run on the identical inputs and compared
   with `np.array_equal` — **all outputs bitwise identical**, counts included.
   (An `np.allclose(rtol=1e-12)` comparison was also run and passed.)
3. **Precision cross-check.** A line-for-line re-implementation of the old
   `np.add.at` algorithm was executed against the new path for
   `precision ∈ {True, False}` — results in §5.3.
4. **Invalidation.** Cache hit, and rebuild-on-change for `nu`, box
   regeneration, `fiducial_cosmology` and skymap replacement; rebuilt map
   reproduces the original bitwise (§6).
5. **Test suite.** `pytest tests/` → **408 passed, 20 failed**. All 20
   failures reproduce identically on the untouched, installed copy of the
   package (missing optional deps `baccoemu`, `emcee`, `nautilus`, `mpi4py`,
   and two NumPy-version `AttributeError`s) — i.e. pre-existing and unrelated.
   Directly relevant tests that pass:
   - `tests/test_power.py::test_grid_field_to_sky_map_healpix` (average T/F)
   - `tests/test_power.py::test_grid_field_to_sky_map_los_sel_shape_validation`
   - `tests/test_pipeline.py::test_mock_tracer_sim_healpix` (10 realisations)
   - `tests/test_mock.py::test_propagate_mock_field_average_false_chunk_merge`
   - `tests/test_galaxy_auto.py` healpix cases
6. **Style.** `black --check src/meer21cm/grid.py` clean.
7. **Changelog.** Entry added under `dev` in `CHANGELOG.rst`.

### Reproducing

```bash
# the venv holds a *non-editable* copy of the package, so either:
pip install -e .                # recommended, per DEVELOPING.md
# or, without touching the environment:
PYTHONPATH=src pytest tests/

PYTHONPATH=src pytest tests/ -q
black --check src/meer21cm/grid.py
```

Minimal smoke/benchmark snippet:

```python
import time, numpy as np
from meer21cm import PowerSpectrum
from meer21cm.util import redshift_to_freq

nu = np.linspace(redshift_to_freq(1.1), redshift_to_freq(0.4), 50)
ps = PowerSpectrum(nu=nu, hp_nside=128, ra_range=(200, 220), dec_range=(-5, 15),
                   downres_factor_radial=1 / 3, downres_factor_transverse=1 / 3)
ps.get_enclosing_box()
f = np.random.default_rng(0).normal(size=ps.box_ndim).astype(ps.real_dtype)

for i in range(3):
    t0 = time.perf_counter()
    ps.grid_field_to_sky_map(f, average=False, mask=False)
    print(f"call {i}: {time.perf_counter() - t0:.3f} s")   # 1st ≈ build, then ≈ 0.05 s
```

---

## 9. Compatibility and limitations

- **No public API change.** `grid_field_to_sky_map` signatures, return
  `(map_bin, count_bin)` shapes/dtypes, error messages for shape mismatches,
  and `propagate_mock_field_to_data` behaviour are unchanged. The only new
  surface is three private attributes and three private methods.
- **Pickling**: the cache holds NumPy arrays, a tuple of floats/bytes, a
  `SkyMap` and an `interp1d` — all picklable. Unpickling keeps the cache;
  if a reconstructed object's geometry differs, the fingerprint check
  rebuilds it.
- **Threading**: not thread-safe (a benign double build is the worst case —
  both threads write the same values to the same attributes).
- **`los_sel` with duplicates** is applied twice, exactly as before (the old
  code also advanced through the duplicated columns).
- **WCS path** still recomputes its geometry on every call.
- **Beam convolution** (`convolve_data` → `weighted_smoothing_healpix`) is
  untouched and remains the dominant cost once `sigma_beam_ch` is set.

---

## 10. Future work

1. **WCS gridding** (`_grid_field_to_sky_map_wcs`, `grid.py:1604`) has the
   same structure (`ra_dec_z_for_coord_in_box` → `radec_to_indx` →
   `project_particle_to_regular_grid`) and would benefit from the identical
   treatment.
2. **Process-wide LRU** keyed by a fully value-based fingerprint (cosmology
   parameters + digest of `pixel_id`) for ensembles that rebuild the object
   per realisation (`transfer.py`, `tests/test_pipeline.py`'s 10-realisation
   loop).
3. **Skip the LOS copy in the mock loop** when a batch spans all channels
   (`field` is already contiguous), saving one `n_voxel` copy per call.
4. **Cache the counts array** for the full-LOS case (it is geometry-only) to
   save the second `bincount`; requires returning a copy to avoid aliasing.
5. **Coarser fingerprint**: replace `nu.tobytes()`/float tuples with a
   cheaper digest if profiling ever shows the key check to matter (it is
   currently microseconds against a 50 ms call).
