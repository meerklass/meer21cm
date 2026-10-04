Releases
========

dev
------

Features
++++++++
* implement local Yamamoto multipoles on ``FieldPowerSpectrum`` (``los='firstpoint'`` /
  ``'endpoint'``): 3D :math:`F_\ell F_0` cube then the same 1D ``|k|`` binning as global PP
* add ``multipole_power_3d``, ``los_observer`` (defaults to ``box_origin``), and real
  :math:`Y_{\ell m}` helpers in ``spherical.py`` (pypower / Hand et al. convention)
* odd Yamamoto poles use :math:`\mathrm{Im}(A_\ell F_0^*)` (even stay real); auto
  firstpoint :math:`\approx` minus endpoint on odd :math:`\ell`
* voxel-averaged local-:math:`\mu` shell projector removed from the
  windowed-multipole matrix: discrete-shell :math:`W` is the Yamamoto
  :math:`|k|` average of :math:`W_{\ell\ell'}` (identity = k-rebin only).
  Global PP modelling stays on 3D ``get_modelpk_conv`` then ``get_1d_power``
* wa_order=1 odd wide-angle matrix (``wide_angle.py``) joined via
  ``DiscreteShellWindowMatrix.resum_input_odd_wide_angle``
* forward ``los_observer`` through ``PowerSpectrum`` and ``SmoothWindowEstimator``
* validate no-RSD Gaussian lightcone + far-observer box under ``misc/yamamoto/no_rsd``

Enhancements
++++++++++++
* ``k_para`` stays box-:math:`z` for local LOS; ``mu_mode`` is box-centre
  :math:`\hat k\cdot\hat n` (diagnostic); discrete-shell :math:`W` does not
  multiply by :math:`\mathcal{L}_\ell(\mu)`
* ``DiscreteShellWindowMatrix`` tracks ``ells_in`` / ``ells_out`` separately
* ``los='midpoint'`` remains reserved (``NotImplementedError``)
* smooth-window Hankel can include an optional :math:`k=0` pair-count term
  (``W_zero``); HealPix validation uses the excess
  :math:`\max(W(0)-W(k_{\mathrm{fund}}),0)` so FFTLog edge extrapolation is
  not double-counted

Changed
+++++++
* rename dish-beam kwargs and helpers to physical names:
  ``beam_at_input`` → ``beam_at_theory_mode``,
  ``beam_in_kernel`` → ``beam_at_output_mode``,
  ``beam_leg_scale`` → ``beam_diag_as_ratio``;
  ``beam_input_*`` → ``beam_theory_*``;
  ``beam_kernel_bin_masses`` → ``beam_output_cell_masses``;
  ``beam_out_mode_scale(level=)`` → ``frame='box'|'cells'``.
* rename ``multipole_power.py`` → ``multipole.py`` and split helpers from
  ``multipole_model.py`` into ``multipole_ops.py`` (same layout as
  ``power.py`` / ``power_ops.py``).  Classes
  (``MultipolePowerSpectrum``, ``WindowedMultipoleModel``,
  ``SmoothWindowEstimator``) live in ``meer21cm.multipole``; beam /
  sampling / k-grid helpers live in ``meer21cm.multipole_ops``.
  ``tests/test_multipole_power.py`` → ``tests/test_multipole.py``.
* rename ``smooth_window.py`` → ``window.py`` (the module now holds both the
  smooth Hankel/Wigner layer and the exact mesh-level FFT window
  ``build_mesh_window_matrix``); imports change from
  ``meer21cm.smooth_window`` to ``meer21cm.window``, and
  ``tests/test_smooth_window.py`` → ``tests/test_window.py``
* add exact mesh-level window ``build_mesh_window_matrix`` (estimator's own
  mesh response, exact for any LOS incl. the true lightcone observer; the
  discrete-:math:`\mu` projector is its :math:`1/d\to 0` limit) with tests in
  ``tests/test_mesh_window.py``
* add the exact b=b' map-sampling shot diagonal to the mesh window:
  ``build_mesh_window_matrix(..., map_m2=...)`` replaces the model's own
  mode_scale-suppressed diagonal
  :math:`(R/N^2)\sum_q t(q)P(q)\sum_b |W_b(k-q)|^2` with the data's actual
  diagonal :math:`(V R/N^2)\sum_b m_b^2 |W_b(k)|^2` (per-lag stencil
  machinery in ``window.map_sampling_shot_diagonal``; monopole offset on
  ``DiscreteShellWindowMatrix.offset``); test-03 ``exact_window_models`` /
  check-C smooth ladder and test-04 mesh models mirror it via the cached
  per-seed ``map_m2`` (see ``misc/rsd_sims/p0_shot_fix_todo.md``).  The
  post-correction P0 residual is the still-open **coherent deficit** (the
  model over-predicts the data's coherent part by ~2–5% at mid-k; ruled out:
  sampling kernel, mode_scale placement, MAS commutation, and — via a
  z-taper experiment — window leakage)

v0.10.0
-------
Features
++++++++
* build the FKP field ``F = D - alpha R`` in ``grid_gal_to_field(construct_fkp=True)``,
  keeping the painted data, randoms and alpha on ``field_i_D``, ``field_i_R`` and
  ``field_i_alpha`` when ``field_i_has_random`` is set
* add ``get_shot_noise_counts`` and expose it as ``shot_noise_1`` / ``shot_noise_2``,
  including the mass-assignment factor, for both a count field and an FKP field
* read a separable selection ``A(theta) n_w(z)`` from random catalogues
  (``selection_from_random_files``, ``PowerSpectrum.read_random_selection``) and evaluate it
  on the simulation box (``MockSimulation.sky_selection_density``, ``galaxy_selection``)
* add galaxy jackknife patches, one weighting convention across jackknife schemes, and
  reuse of a gridded field under a different mask

Enhancements
++++++++++++
* cache the enclosed source count and normalise ``dndz_renorm`` to that count
* refactor the stacking helpers
* add the jackknife cookbook, the multipole-k cookbook and citation metadata
* cover the FKP field, shot noise and selection with physics tests

Fixes
+++++
* pass ``field_mean`` through the observed-power renormalisation
* stop the real FFT from overweighting the ``k_z = 0`` mode in the lognormal variance
* correct the galaxy-window normalisation
* rename the smooth-window helpers and test tapering weights
* replace the deprecated Matplotlib ``get_cmap`` call

v0.9.0
------
Features
++++++++
* add opt-in multipole survey-window path (global plane-parallel): ``SmoothWindowEstimator``,
  ``DiscreteShellWindowMatrix``, and ``WindowedMultipoleModel``
* measure HI window multipoles from the selection / weight field; galaxy windows from Poisson randoms
* support continuous kernels ``smooth`` (Hankel / Wigner) and ``identity`` (discrete FFT ``μ``-selection only)
* add continuous theory multipoles via ``ModelPowerSpectrum.get_theory_multipoles_kmu`` /
  ``power_kmu`` hierarchy (beam / sampling / MAS deferred to the window path)
* add ``fftlog`` Hankel utilities for smooth-window transforms
* add multipole shell map / LOS stubs on ``FieldPowerSpectrum`` for future Yamamoto estimators
* add window-matrix plotting helpers ``plot_discrete_shell_window_row`` and
  ``plot_discrete_shell_window_matrix``
* optimize PR CI by running test/coverage only when source-impacting files change
* add Codecov carryforward configuration for flagged test coverage uploads
* switch CI change detection to latest-commit scope for docs-only follow-up commits

Enhancements
++++++++++++
* refactor power-spectrum code into ``estimator``, ``model``, ``grid``, and ``power_ops`` modules
* clarify ``power_kmu`` vs 3D ``get_modelpk_conv`` modelling paths for future multipole compatibility
* improve Poisson random galaxy generation for window / randoms workflows
* add explicit typing on the estimator module

Fixes
+++++
* remove a stray debug print from ``util._ra_range_is_subset_of``
* clarify that CI change detection checks only the latest commit diff (``HEAD^..HEAD``), so docs-only follow-up commits skip tests
* include workflow file changes in CI change detection so workflow edits trigger tests

v0.8.0
------
Features
++++++++
* add single/double precision option and robust single-precision `mu` bin edge handling
* add batch processing support in mock generation and gridding routines
* add auto range setting when reading maps and support flexible column names
* add Hartlap/Percival correction factors for inference
* add lazy imports for top-level package classes and add foreground tools/docs
* add additional validation/filter testing support and transfer-function utility updates

Enhancements
++++++++++++
* refactor cosmology and AP-parameter handling in model/matter power workflows
* improve transfer-function and simulation options, including optional high-resolution simulation inputs
* improve plotting/validation helper scripts and documentation notebooks

Bugfixes
++++++++
* fix `ps_type` propagation to both true and fiducial settings
* fix weighting, gridding, and ordering edge cases (including NaN handling)
* fix data-injection behavior when unclean maps are missing and improve mean-centering for `R_mat`
* fix model/cosmology dependency propagation and related growth-factor calls
* pin `katbeam` dependency to a known working commit

v0.7.0
------
Features
++++++++
* transfer function class
* parallelization for transfer function
* numerical transfer function calculation from cross and auto mock
* parameter fitting sampler class
* parallelization for parameter fitting
* support emcee and nautilus for sampling

Enhancements
++++++++++++
* allow k-mode cut in 1D power spectrum
* add validation tests 00 and 01 in paper

Bugfixes
++++++++
* fix a bug in FPS initialization
* fix a bug in incorrect sigma_z usage in model power calculation

v0.6.1
------
Features
++++++++
* allow dndz input for mock galaxy simulation
* allow flat-sky approximation in mock and ps calculation
* allow read in UHF pickle file

Bugfixes
++++++++
* fix a bug in get_enclosing_box

v0.6.0
------
Features
++++++++
* mock tracer positions
* new HI galaxy simulation class
* new cosmology parameter class

Enhancements
++++++++++++
* better RSD routine for lognormal simulation
* allow using baccoemu instead of camb

v0.5.0
------
Features
++++++++
* mock simulation
* consistent model power to mock to field power
* explicit dependency checks and cache
* different beam models

Enhancements
++++++++++++
* gridding now part of PS class
* consistent tests for sky map to power spectrum
* remove hiimtool dependency
* end-to-end test from input ps to final power spectrum estimation

v0.4.0
------
Features
++++++++
* a base class for better structure
* model power spectrum
* power spectrum weights and convolution

Enhancements
++++++++++++
* allows more flexibility in PCA
* more precise HI average temp

v0.3.0
------
Features
++++++++
* Gridding functionalities to grid sky map to regular grids
* Basic power spectrum estimation functionalities

Enhancements
++++++++++++
* find enclosing box functions migrated to grid module

v0.2.0
------

Features
++++++++
* MeerKLASS map i/o functionalities consistent with meerpower
* Basic cosmological calculator
* plot functionalities as a separate module
* telescope-related functions including beam size and convolution

Enhancements
++++++++++++
* consistent mean and covariance calculation in PCA

Bugfixes
++++++++
* Fixed a los-axis tranpose back issue

v0.1.1
------

Enhancements
++++++++++++
* allow fixed RA and Dec in lognormal simulations
* add calculation of angles between coorindates on the sphere


v0.1.0
------

Features
++++++++
* Generation of colored noise for simulating systematics
* A bit more docs

Enhancements
++++++++++++
* RSD effect in lognormal mocks based on Kaiser effects

Bugfixes
++++++++
* Fixed a mismatch of h unit in the lognormal simulation

v0.0.1
------
This is the first version

Features
++++++++
* Log-Normal and Poisson generation of HI galaxy signals based on HIMF and velocity dispersion
* Stacking in 3D space
* Calculating effective weights for correcting signal loss for PCA
* Simulation of synchrotron foreground emission using Haslam template
* Docs with API summary
* Unit test coverage
