# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!--changelog-start-->

## [Unreleased]

## [3.14.0] — 2026-11-13

### Behaviour changes for existing users

These affect a `Problem` with an objective constant (`add_obj_constant`)
or with objective-valued solver options:

- **`objective_bound` on a maximisation is now given in the objective's
  own sense.**  Before, HiGHS got it as given and read it in its
  minimisation space, so a bound on a maximisation had to be passed
  negated; polar-high now negates it.  A caller that negated it itself
  must drop its own negation, or the bound flips (e.g. a bound that was
  feasible becomes infeasible).
- **HiGHS' relative MIP gap is measured without the objective constant.**
  The constant no longer reaches HiGHS, so `mip_rel_gap` (and the gap
  HiGHS reports) is relative to the variable part of the objective: a
  solve with a constant stops at a different point than before.
- **The HiGHS objective value no longer includes the constant.**  A raw
  `Solution.highs.getObjectiveValue()` / `getInfo()` (and the LP's
  `offset_`) exclude it; add `Solution.obj_offset` to a raw HiGHS read.
  `Solution.obj`, `SolverResult.objective` and
  `SolveDiagnostics.objective_value` still include it.
- **The `build_only` MPS file no longer carries the constant.**  Before,
  `build_only` wrote its MPS with HiGHS `writeModel` and the constant was
  in it; it is not now (it warns when the constant is non-zero).  Every
  MPS polar-high writes (`write_mps`, `build_only`, the file-based
  fallback) now leaves the constant out; add the constant you passed to
  `add_obj_constant` to an objective read from such a file.

### Added

- **Objective-valued solver options in user units:
  `Problem.set_objective_scale(factor)` and
  `polar_high.objective_options`.**  A caller that builds its objective
  scaled (the objective given to `set_objective` / `add_obj_constant` is
  `factor` × its own objective) declares the factor once.  Objective
  targets, bounds, cutoffs and absolute gaps are then written in user
  units (the units the caller thinks in) and in the sense of its
  objective, and polar-high maps them to what the solver compares: levels `q = f·(k·v − O) + O_solver`,
  absolute gaps `q = f·k·g`, with `k` the factor, `f = 2**N` for HiGHS'
  `user_objective_scale = N` (1 for every other solver), `O` the objective
  constant and `O_solver` the constant the solver holds itself.  HiGHS
  `objective_bound`, which HiGHS reads in its minimisation space, is
  negated for a maximisation.  The options known: HiGHS `objective_target`,
  `objective_bound`, `mip_abs_gap`; Gurobi `BestObjStop`, `BestBdStop`,
  `Cutoff`, `MIPGapAbs`, `PoolGapAbs`; CPLEX lower/upper cutoff, MIP
  lower/upper objective stop, simplex lower/upper objective limit, absolute
  MIP gap, objective difference, pool absolute gap, polish-after absolute
  gap, barrier objective range; Xpress `MIPABSCUTOFF`, `MIPADDCUTOFF`,
  `MIPABSSTOP`; COPT `AbsGap`.  Relative gaps, non-numbers, non-finite
  values, `|v| >= 1e20` and options whose mapping is the identity pass
  unchanged; translated values are rounded to 15 significant digits (a
  string comes back as the `repr` of the float).
  Applied to every in-process HiGHS solve (`Problem.solve`,
  `WarmProblem.solve`, `WarmProblem.solve_with_fallback`) and to
  `polar_high.solvers.solve` for every solver, always in a new dict: the
  stored options keep the values as written, so a re-solve or retry never
  translates twice.  For a solver that reads a `write_mps` file,
  `Problem.objective_transform(solver, "mps")` and
  `translate_objective_options(solver, options, transform)` give the
  mapping (a dict, or a sequence of `(name, value, ...)` tuples; named
  tuples keep their type); `objective_option_kind(solver, name)` says
  whether an option is a level (`"value"`), a difference (`"diff"`) or
  neither.  `Solution.objective_in_user_units` is `obj / factor`.

### Changed

- **Objective-valued HiGHS options are now read in user units.**
  Before, `objective_target`, `objective_bound` and `mip_abs_gap` reached
  HiGHS as given, and HiGHS compared them in its `2**user_objective_scale`
  scaled space, so with Layer 3 autoscaling on they were off by `2**N`.
  When the mapping is the identity (factor 1, no `user_objective_scale`,
  and for a level no constant to take off and no negation) the value
  passes untouched, exactly as written.
- **The objective constant (`add_obj_constant`) is kept out of every
  solver that would not report it consistently, and added to the reported
  objective.**  In-process HiGHS (streaming, `streaming=False`,
  `WarmProblem`) no longer gets the constant as its objective offset; the
  new `Solution.obj_offset` holds it.  The file-based solver fallback
  (`io_api=IOMode.MPS`, and COPT whenever `highspy` is loaded) no longer
  writes the constant into the MPS file, whose objective-row constant
  solvers read differently, and adds it to the parsed objective.

### Fixed

- **HiGHS MIP bound and gap with an objective constant and
  `user_objective_scale`.**  HiGHS scales the costs by `2**N` but not the
  objective offset, so the dual bound mixed scaled and unscaled parts and
  the reported relative gap could be far too small (e.g. 2.8e-6 against a
  true 0.42 %), stopping a `mip_rel_gap` solve early with a false
  optimum.  The constant no longer reaches HiGHS (see Changed), so Layer 3
  autoscaling stays correct with any constant.

## [3.13.0] — 2026-11-13

### Added

- **`SolveDiagnostics.mip_gap`**: the relative MIP gap HiGHS reports for
  the solve (`info.mip_gap`), read in the same snapshot as the other
  counters.  `inf` for an LP or a MIP without a feasible solution (HiGHS
  reports `NaN` for a maximisation without one; that is normalised to
  `inf`).  The field has a default, so code that builds a
  `SolveDiagnostics` itself keeps working.

### Fixed

- **The same model gives the same coefficient bits in every run.** Two
  operations depended on how polars happened to lay the data out, which
  changes from run to run with the thread count, and on polars 2.x's
  default streaming engine with its scheduling:
  - *A Param divided by a scalar* (`Param / 2.5`, and a scalar divisor
    in the RHS / autoscale Param-chain folds): polars divides a column
    by a literal per chunk — the true quotient for a one-row chunk, a
    multiply by the reciprocal for a longer one — so about half of the
    values could differ in the last bit between runs.  It is now always
    a multiply by the reciprocal (what polars did for every row of a
    longer chunk, and what `Expr / float` has always done).
  - *A `Sum` that adds several coefficients of one cell* (a reduced dim
    that is not a dim of the Var, e.g. `Sum(x[p] · b[t], over="t")`):
    the hash `group_by` added a group in the order its rows arrived, on
    polars 1.x too.  A group of three or more is now added in the order
    of the term's dims (`numpy.add.reduceat` on the sorted rows, as a Var
    with shared columns already was), lazily when the plan is collected;
    a group of nearly cancelling entries such as `1e16, 1, −1e16` could
    come out `0` or `1`.  The block-COO combining reduce sums in the same
    order, so both paths give the same bits.  A Sum over Var dims only (a
    relabel, one coefficient per group) keeps the hash `group_by`.  A
    null coefficient is skipped, as polars' `sum` skips it.
    *Cost:* such a Sum no longer streams — its unsummed rows are sorted
    in memory when the plan is collected (about +36 bytes per unsummed
    row at peak, and a sort: a few seconds for tens of millions of rows).
    A Sum with one coefficient per cell is not affected.
  - *The autoscale log2 histogram* (`Log2HistogramReducer`, the
    coefficient-bucket walk): a bucket's `sum log2` was a running float
    sum over the walk's rows, whose order (hash joins in the prune-down
    backstop) changes from run to run.  A batch's sum is now correctly
    rounded (`math.fsum`), the same bits whatever the row order (about
    40 ns per coefficient).

## [3.12.4] — 2026-11-11

### Fixed

- **`write_mps` writes every column, so a solution read back by position
  lines up.** A column with no constraint coefficient and no objective
  coefficient was left out of the COLUMNS section: it was written only
  under BOUNDS, or not at all when its bounds were the default.  HiGHS then
  read it at the end of the column list, or dropped it, so every later
  column moved one place and a caller mapping the solution back by
  position got wrong values (an integer column of this kind also lost its
  integrality).  Such a column is now written in place with a zero
  objective coefficient (`cost 0`), keeping its bounds and integrality.
  Columns with constraint coefficients and a zero cost are written as
  before.
- **`write_mps(emit_names=False)` now records the model's fingerprint.**
  `_last_mps_fingerprint` is taken over the real column and row names
  also when the file carries generic `C…` / `R…` names, and equals the one
  a named write of the same model gives.  Before, it was `None`.
- **Warm-start basis caches written before this release are not reused.**
  The fingerprint format moves to `phbasis-v2`: a basis saved from an MPS
  with a column out of place has the same name-set as the fixed model but
  its column statuses are in the wrong order, so it must not be loaded.

### Changed

- **`write_mps(emit_names=False)` uses less memory.** Each generic `C…` /
  `R…` name is built when its line is written, instead of holding lists of
  all of them for the whole write (about 0.8 GB at 7 M columns and 7 M
  rows). The file is byte-for-byte the same.
- **The log-routing tests pass with highspy 1.16.**  The two real-solve
  tests now solve a 3-row LP and check for its solve-log lines.  Before,
  they checked for the "Running HiGHS" version banner of an LP with no
  rows.  HiGHS 1.16 prints that banner once per process, when `Highs()` is
  created and before a log callback can be attached, so it no longer goes
  through the routing.

## [3.12.3] — 2026-11-07

### Fixed

- **polars 2.x: joins run in the order polar-high writes them.** polars 2.x
  reorders runs of inner joins by estimated cardinality
  (`QueryOptFlags.join_order`, on by default).  On the `Sum` relabel emit
  (`result ⋈ map(p, s → n) ⋈ row_index(n, d, t)`) it joined the row index to
  the coefficients on `(d, t)` alone first, a many-to-many product: on a
  FlexTool model the build peaked at 25 GB instead of 9 GB, with the same LP.
  Every collect of a plan polar-high composes (Param and Var products,
  `Where`, the `Sum` emits, the row-index attach, the autoscale walks) now
  runs with `join_order` off.  On polars 1.x, which has no such flag, the
  collects are unchanged.

## [3.12.2] — 2026-11-06

### Fixed

- **`import polar_high` no longer sets `POLARS_MAX_THREADS` when polars is
  already imported.** polars reads the variable only at import, so setting
  it afterwards did not change the running process; it only leaked into
  child processes started from it (an application that imported polars
  first and then polar_high gave every child one polars thread).  The
  single-thread default still applies when polar_high is imported before
  polars and the variable is unset.

## [3.12.1] — 2026-11-04

### Fixed

- **`WarmProblem.solve(retry_on_unknown=True)` reloads the LP when the cold
  retry is still `kUnknown`.** On a badly scaled LP (a Benders master with
  cut slopes near 1e9), `clearSolver()` can keep solver state that leaves
  the cold run `kUnknown` with a few rows just outside the tolerance, while
  the same LP solves to optimality in a fresh instance.  The retry now
  passes the instance's own LP back to HiGHS and runs once more; the LP,
  names, options and column / row ids are unchanged.
- **A Var whose rows share columns** (built directly from a registered
  family's ``col_id``s, e.g. one column held over a run of time steps and
  indexed on every step; new `Var.col_ids_unique`): a `Sum` over its rows
  combines coefficients even when every summed dim is a Var dim.  The
  block-COO builders, the warm-start parameter tracker and the range
  readout no longer treat such a Sum as a relabel: the readout reported
  the per-row coefficients instead of the summed cells (for example 24
  times too small for a daily column on hourly rows), and
  `WarmProblem.update_param` rewrote a summed cell with one row's
  coefficient.  `Sum` now adds the rows of one cell of such a term in a
  fixed order (by the term's dims, with `numpy.add.reduceat` on one
  contiguous array): a polars group-by summed them per chunk, so the
  last bits of the coefficient changed from run to run.  Problems built
  only from `Problem.add_var` families are unchanged.
- **`WarmProblem.update_param` refuses a Param whose tracked cells repeat
  a `(row, col)`** (the Param in two terms of one row): `changeCoeff`
  sets the whole cell, so only the last term was kept.

## [3.12.0] — 2026-11-03

### Added

- **`ArrayClassifier` for `Log2HistogramReducer`**
  (`polar_high.autoscale._coef_walk`).  A column classification can now
  be given as one integer bucket index per LP column plus the list of
  bucket keys (`-1` = no bucket), instead of a Python callable backed by a
  per-column dict.  The reducer looks the buckets up with numpy and
  returns the same accumulator as the equivalent callable: same keys, same
  order, bit-identical values.  Callables keep working unchanged.
- **`highs_names="always" | "auto" | "never"`** on `Problem.solve` and
  `WarmProblem`: when the LP names are passed to HiGHS.  HiGHS keeps its
  own copy of every name (about 3 GB on a 6 M-column / 7.5 M-row LP) and
  never reads them while solving.  `"auto"` passes them just before
  `run()` only when a name-writing HiGHS option is on
  (`polar_high.engine.HIGHS_NAME_OPTIONS`); `"never"` only on request.
  New `Solution.highs_names_passed`, `Solution.pass_names_to_highs()` and
  `WarmProblem.pass_names_to_highs()`.  The solution, the Python
  `Solution.col_names` / `row_names`, `write_mps` and every HiGHS-written
  file once names are passed are the same in every mode.
  `save_memory=True` always passes the names (its MPS round trip goes
  through HiGHS).  The default stays `"always"`.

### Changed

- **Lower peak memory for terms that multiply by an expanding Param
  product.**  A `Param * Param` result now records its operation tree.
  When a `Var * P` or `Expr * P` term multiplies by a product `P` that
  expands (a cross join, or a join wider than either operand, e.g.
  `size[p] * duration[d,t] * limit[g,d]`), the term joins a rebuild of `P`
  in which every leaf Param is first semi-joined to bounds of the term's
  keys: the Var's frame and the term's `Where` frames when the term still
  has its Var source (cheap; the term's own plan is not re-run), else the
  term's rows after its deferred pure-filter `Where`s.  Before, the whole product was
  materialised and then mostly thrown away by the term's join, once per
  batch in every coefficient walk.  The rebuild keeps the association,
  operand roles and dtypes, so coefficients are bit-identical.  Products
  that only filter keep their plan unchanged.  Trees that divide, and
  `Expr / P`, are not pruned: polars picks its float-division kernel by
  operand length (a length-1 divisor becomes a multiply by its
  reciprocal), so a pruned quotient could differ in the last bit.
  `POLAR_HIGH_DISABLE_COMPOSITE_PRUNE=1` turns the pruning off.
- **`WarmProblem` frees its LP staging copies earlier.**  The
  sentinel-translated bound arrays are freed once the `HighsLp` holds
  them, and the `HighsLp` itself once `passModel` has copied it into
  HiGHS, instead of at the end of the build (about 0.6 GB on a
  13 M-nonzero LP, before the names are passed).  The LP is unchanged.

## [3.11.0] — 2026-10-26

### Added

- **Optional floor for tiny objective coefficients**:
  `Problem.obj_coef_zero_threshold` (absolute) and
  `Problem.obj_coef_zero_rel_threshold` (relative to the largest
  `|cost|`).  A column's summed cost whose user-space magnitude is
  strictly below `max(abs, 2**floor(log2(rel * max|c|)))` becomes exactly
  `0.0` — judged before the Layer-2 side vectors and Layer-3
  `user_objective_scale`, and applied in the canonical matrix
  (`write_mps`, passModel, `WarmProblem`), the streaming solve and its
  `streamed_lp_ranges`, the pre-solve `detect_ranges` readout and
  `WarmProblem.update_obj_coef(_array)`.  A float residue (e.g. `1e-17`)
  can no longer stretch the cost range and drive the objective centring
  to `2**25`+ (HiGHS dual simplex "excessive dual values").  Both default
  to `0.0` (off, byte-identical).  The matrix/RHS floor
  `coef_zero_threshold` is unchanged and still never touches costs.

## [3.10.0] — 2026-10-25

### Added

- `InOutStabilizer.effective_weight()` (the weight `separation_point`
  applies now: 0 for a pass-through), `InOutStabilizer.centre_value()`,
  `polar_high.consistent_separation_points` and the exception
  `PinnedMasterNotOptimal` (raised by a master adapter's `native_cost_at`
  when the pinned master is not optimal), and the optional `BendersMaster`
  member `pinned_solve_at(point) -> (native cost, Solution)`: with in-out ON
  the loop hands its Solution to `on_incumbent`, so the incumbent plan (e.g.
  the investments) is the plan the upper bound priced.

### Fixed

- **Benders in-out (λ > 0) upper bound could fall below the optimum.**
  A coupling column pinned in several subproblems (e.g. the exporter and
  importer half-flows of one cross-region arc) was pinned at different
  values whenever their per-subproblem stabilizers diverged (one forced
  out-step, one interpolating), so the summed recourse belonged to no
  single point and the loop could stop on a bound below the optimum. Every
  owner of a shared column now gets one value: that of the owner with the
  smallest effective weight (`consistent_separation_points`; a forced
  out-step keeps the master point verbatim). The in-out upper bound is now
  always the master's own cost at the same point (`native_cost_at`, one
  extra master solve per iteration) plus the subproblem costs, whether or
  not the master declares `native_cost_flow_dependent`: a flow cost on a
  coupling column also makes it flow-dependent. An iteration whose point
  the pinned master cannot reach, or whose shared columns are split (a
  guard), contributes no upper bound (warning; `on_iteration` reports NaN).
  This costs one extra master solve per in-out iteration; iteration counts
  are unchanged on the test models. A flow-dependent master without
  `native_cost_at` is now an error under in-out.
- **Warm re-solve with `threads` after the global scheduler was re-pinned.**
  HiGHS' task scheduler is process-global; after another caller
  re-initialised it with a different thread count (e.g.
  `prewarm_global_scheduler(1)` before a parallel pass), a `WarmProblem`
  re-solve with an explicit `threads` option was rejected by HiGHS and came
  back not solved. `WarmProblem.solve` now re-initialises the scheduler
  before a re-solve of an instance that sets `threads` (as the first build
  already did); call it while no other HiGHS instance runs.
- Duplicate `(row, col)` coefficients of the COO matrix build (global and
  per-family streaming) are summed in a fixed order, so a group of nearly
  cancelling coefficients gives the same matrix in every process.

## [3.9.0] — 2026-10-24

### Added

- **`frame=` keyword on the warm update methods.**
  `WarmProblem.update_rhs`, `update_obj_coef` and `update_obj_coef_array`
  take `frame="user"` (default) or `frame="scaled"`. In the user frame the
  values are in the model's own units and the Layer-2 side vectors are
  applied as at build; in the scaled frame they are written to HiGHS as
  given (the previous behaviour, for callers that pre-scale).

### Fixed

- **Warm updates under Layer-2 autoscaling.** `update_rhs` wrote the new
  RHS without the Layer-2 row factor, and `update_obj_coef` /
  `update_obj_coef_array` wrote the new costs without the column side
  vector, so a warm re-solve of a Layer-2 scaled LP solved a different
  model than a fresh build with the same values. They now apply the side
  vectors (`frame="user"`). Without side vectors the pushed values are
  unchanged.
- **Scale-invariant small-value floor on the warm path.** `update_rhs`
  and `update_param` judged `coef_zero_threshold` on the scaled value,
  so a user value of 1 scaled to `2**-14` could be floored to 0. The
  floor is now judged on the user-space magnitude, as at build.
- `update_rhs` / `update_obj_coef` label joins keep the row order of the
  constraint / variable explicitly (`maintain_order="left"`).
- Docstrings and the warm-starting guide state the value frame of every
  `WarmProblem` method (`fix_cols`, `fix_col_ids`, `set_col_bounds`,
  `update_coef`, `add_cut_row` write in the scaled frame), that the live
  LP keeps its build labels, and the scope of `update_param` (tracked
  matrix cells only, matched by label).

## [3.8.0] — 2026-10-09

### Added

- **Per-element variable bounds.** `Problem.add_var(lower=..., upper=...)`
  now also accepts a `Param` whose dims are a subset of the variable's
  dims; it is joined onto the variable's index (broadcasting over the
  remaining dims). Elements without a matching row, and `null`/`NaN`
  values, keep the default bound (lower 0, upper +inf); `-inf` lower /
  `+inf` upper allowed. Param rows that match no variable row are
  ignored (sparse tighten), including keys outside the variable's
  `pl.Enum` vocabulary; null keys never match (element keeps the
  default). `Enum` vs `Utf8` key mismatches are aligned automatically;
  `Enum` keys with disjoint category sets raise `ValueError`; `Utf8` vs
  integer keys raise. Duplicate Param rows or dims outside the
  variable's raise `ValueError`. An element with lower > upper is
  accepted and makes the LP infeasible, as for scalar bounds.
  The per-column values flow through the streaming and `passModel`
  solves, `LpView` (all solver adapters), `write_mps`, `WarmProblem`
  and autoscale range detection. New `Var.col_lower()` /
  `Var.col_upper()` / `Var.scale_bounds()` / `Var.has_elementwise_bounds`
  / `Var.bound_param_names` helpers. Scalar bounds are unchanged
  (byte-identical LP); scalar bounds still accept any `numbers.Real` or
  0-d numeric numpy array.
- Bound Params are resolved at `add_var` and **not tracked**:
  `WarmProblem.declare_mutable` / `update_param` raise `ValueError` for
  the name of a Param used as a bound — use `WarmProblem.set_col_bounds`.

### Changed

- `Problem.add_var` now raises `ValueError` for a `+inf` lower bound or
  a `-inf` upper bound (scalar or per-element; previously such a bound
  was silently dropped from `write_mps`, so the exported model differed
  from the solved one), and `TypeError` for a `bool` bound. All valid
  scalar bounds behave exactly as before.

### Fixed

- polars 2.0 compatibility: `Problem.add_var` with `dims=()` (a dimless
  variable) failed under polars >= 2.0 ("Series col_id, length 1 doesn't
  match the DataFrame height of 0"). The one-column frame is now built
  directly; output is unchanged under polars 1.x.

## [3.7.0] — 2026-07-29

### Added

- **`Solution.solve_diagnostics()` — status/feasibility snapshot.** A new
  frozen `SolveDiagnostics` carrier and `Solution.solve_diagnostics()`
  method expose a policy-free snapshot of `getModelStatus()` + `getInfo()`:
  primal/dual solution status, primal/dual infeasibility counts, absolute
  and relative max infeasibilities, `primal_dual_objective_error`, and the
  objective value. It reads the live solver handle in a single `getInfo()`
  call and returns `None` when no queryable handle is attached (a
  synthesised `Solution`, or a read-only subprocess/commercial shim without
  `getInfo`/`getModelStatus`). No policy or thresholds live here and
  `self.optimal` is untouched, so existing consumers are unaffected. This
  lets a downstream caller decide whether a solve HiGHS could not certify as
  optimal — e.g. an interior-point solve run without crossover whose
  post-solve dual objective is slightly inconsistent — is nonetheless usable
  (feasible primal, small primal-dual gap).

## [3.6.1] — 2026-07-10

### Fixed

- **Scale-invariant small-coefficient cutoff.** The `coef_zero_threshold`
  floor was applied to Layer-2-*scaled* matrix and row-bound values with a
  *user-space* absolute threshold. Because Layer 2 is a lossless
  power-of-two conditioning transform, a structurally-essential coefficient
  (e.g. a `1.0`) whose column/row factors shrank it below the threshold
  (e.g. to `2**-14`) was floored to `0.0`, silently corrupting the LP: HiGHS
  then solved a genuinely different problem to a real optimum that
  un-scaled to a wrong point (reported `Optimal`). On a commodity-ladder
  trade model this collapsed the `commodity_ladder_balance` to `0 == flow`,
  forcing the priced supply to 0 and all demand onto slack — a ~10,000x
  objective error under `full` autoscaling.

  The floor now judges negligibility on the **user-space** magnitude
  (`|scaled| / (row_factor * col_factor)`), so Layer-2 conditioning never
  drops a coefficient that is significant in the original problem.
  `_floor_small_coefs` gains an optional per-entry `unscale` multiplier and
  a new `_layer2_matrix_unscale` helper builds it for COO/CSC matrices;
  applied at every cold floor site (non-streaming `_build_canonical_matrix`
  matrix + row bounds, and the streaming `_solve_streaming` RHS + matrix).
  `unscale=None` (no Layer-2 factors) is the exact pre-change path, so
  `ScalingMode.OFF` / `BASIC` / `SOLVER_ONLY` are byte-identical; only
  `FULL` changes, and only for entries scaling pushed across the threshold.
  New regression: `tests/test_layer2_floor_scale_invariant.py`.

## [3.6.0] — 2026-07-09

### Added

- **Warm-start basis primitives (name-keyed, opt-in).** New building blocks
  for warm-starting a solve from a related run's basis or partial primal.
  All record intent only and are byte-identical to before when unused:
  - `NamedBasis` carrier + `basis_fingerprint(col_names, row_names)`
    (exported from `polar_high`) — a structural fingerprint of a model's
    name-set so a stored basis can be matched to a compatible model.
  - `engine.get_named_basis()` / `set_named_basis(nb, *, policy="exact")` —
    extract the in-process HiGHS basis keyed by column/row *name*, and inject
    a stored basis back (name-matched, `"exact"` or alien-tolerant policy).
    A solve-time streaming hook builds and applies the HiGHS basis just before
    `run()`, so injection works on the streaming (save-memory) path too.
  - `Problem.basis_name_fingerprint()` and `WarmProblem.set_named_basis(...)`
    so the retained-handle `WarmProblem` path can also seed a basis.
  - `seed_primal(values, *, frame=...)` — partial primal-seed primitive
    (`setSolution`); mutually exclusive with a named basis ("if a basis is
    set, it wins").
  - `write_mps(...)` now stashes `_last_mps_fingerprint` (the structural
    `basis_fingerprint` of the emitted name-set) so a subprocess `.bas` file
    can be matched to the MPS it was produced from.
- **Trust-region master stabilizer** (`TrustRegionStabilizer`;
  `BendersLoopOptions.trust_region_radius`/`trust_region_scale`). A generic
  box (∞-norm) trust region on the master's coupling point: the master is
  solved WITH the box, so its own primal and the subproblem queries share one
  feasible iterate and the whole-objective upper bound is valid **by
  construction** (no separate re-evaluation). Mutually exclusive with in-out.
  `None` ⇒ off ⇒ byte-identical to the prior loop.
- **`native_cost_at` / `evaluate_at_point` single-point objective
  primitive.** The coordinator can evaluate the whole objective —
  `master.native_cost_at(point) + Σ subproblem cost at that point` — at one
  feasible coupling point, for a valid single-point L-shaped upper bound.

### Fixed

- **Consistent-point upper bound for flow-dependent masters.** When the
  master's native cost depends on the coupling flows (a cost-bearing master),
  the previous mixed-point UB (`native_cost(outer) + Σ cost(separation)`) is
  not a single feasible point and is invalid. The coordinator now evaluates
  the master native cost at the SAME point the subproblems used, so in-out
  stabilization is valid — and re-enabled — for such masters. Flow-independent
  masters are byte-identical to before.

## [3.5.0] — 2026-07-06

### Added

- **`polar_high.benders` — generic multicut Benders coordinator (core
  loop).** `solve_benders_loop(master, subproblems, *, options,
  initial_point, ...)` owns the loop mechanics of a multicut Benders
  scheme: the bootstrap subsolve pass at the caller-supplied
  `initial_point` (whose key set IS the coupling column universe),
  recourse-floor sizing (`−eta_floor_mult·max|bootstrap cost|`), cut
  append + recourse relaxation, LB-monotonicity pinning with a gross-drop
  hard-fail, the always-on LB≤best-UB sandwich guard, an optional
  test-time monolith-bound guard, per-cut satisfaction re-checks with
  row-scale tolerances, `read_point` → `project_point(hard_fail=True)`
  feasibility projection, the deterministic parallel subproblem fan-out
  (via `solve_indexed_parallel`), and UB/incumbent bookkeeping with an
  `on_incumbent` payload capture hook. All problem knowledge stays behind
  two adapter surfaces: `BendersMaster` (solve / read_point / native_cost
  / project_point / add_cut / relax_recourse / set_recourse_floor, plus
  an optional `compact_cuts`) and `BendersSubproblem` (`solve_at(point)
  -> SubproblemResult` owns pin+solve end to end, including any scaling
  transform and slope aggregation keyed by MASTER col id);
  `SubproblemHandle` is a plain-data adapter for callers that don't want
  a class. Structured exceptions: `SubproblemNotOptimal` (raised
  domain-side, propagates untouched — including through the parallel
  fan-out) and `BendersBoundInvalid` (kinds `lb_drop` / `sandwich` /
  `cut_violated` / `cut_nonfinite` / `monolith`, carrying the numeric
  fields). The coordinator works entirely in the caller's scale.
- **`polar_high.benders` — in-out stabilization, stall guard, cut
  compaction.** The coordinator's non-default extensions, each gated on
  its option so the default (λ=0, `compact_at=0`) trajectory is
  byte-identical to the core loop: **in-out stabilization**
  (`in_out_weight` in `[0, 1)`) runs one `InOutStabilizer` per
  subproblem, seeded with the `initial_point` centre — each subproblem is
  solved (and its cut generated) at its own interior separation point
  `f_sep = λ·centre + (1−λ)·f_out`, re-projected through
  `project_point(hard_fail=False)` (an interior point built from an old
  incumbent may legitimately exceed the current feasible set; the main
  post-master projection stays `hard_fail=True`), with the per-subproblem
  separation test driving the register/forced-out-step logic and the
  incumbent point overlaying each subproblem's `f_sep` onto exactly the
  master columns it owns (ownership = its `SubproblemResult.slopes` key
  set; a shared column takes the last owner's value); the **stall guard**
  (`StallMonitor`, `stall_window` / `gap_floor` with the
  `max(20·tol, 0.02)` derivation) raises the structured `BendersStalled`
  (iteration / gap / tol / window / reference_scale / per-subproblem
  current + bootstrap costs) on a frozen-blowup stall, with
  `extra_reference_cost` called once post-bootstrap and its absolute
  value folded into the reference scale; **periodic cut compaction**
  (`compact_at` / `cut_policy` / `cut_window`) invokes the master
  adapter's optional `compact_cuts` at the end of the iteration body with
  the raw master vertex and the bounded trailing trial-point window,
  resetting the tracked row count to the returned `kept` — a master
  without the member disables compaction with a warning (clean skip).
- **`WarmProblem.compact_cuts`** — generic cut-pool compaction for
  cutting-plane / Benders masters. `add_cut_row` now retains each appended
  cut row's `(col_ids, coefs, lower)`; `compact_cuts(solution)` classifies
  every retained cut by PRIMAL slack at the given optimum (binding iff
  `slack <= tol`), deletes the strictly-slack rows (never a build-time row),
  and — with `verify=True` — re-solves and rolls back if the objective
  drifted. Removing a strictly-slack (zero-dual) cut is LB-preserving by
  complementary slackness, so this bounds the active master LP without
  harming the certified lower bound.
- **`compact_cuts(policy="dominance")`** — a second, non-default selection
  policy (de Matos–Philpott–Finardi / Guigues Limited-Memory Level-1) for
  DEGENERATE masters where almost every cut is primal-binding and slack
  deletion prunes nothing: cuts are grouped by the recourse column they bound
  (`add_recourse_col` now registers its ids) and only the cuts active at a
  caller-supplied window of trial points are kept. Same verify-restore belt
  as the slack policy.
- **`WarmProblem.fix_col_ids(col_ids, values)`** — fix columns BY RAW COL ID
  (`changeColsBounds` with `lo = hi = value`). The fast-path counterpart of
  `fix_cols` for callers that already hold the ids: skips the per-call
  dim-tuple → col-id join. Typical consumer: a Benders driver temporarily
  pinning coupling columns.
- **`WarmProblem.get_col_bounds(col_ids)` / `set_col_bounds(col_ids, lower,
  upper)`** — read / write column bounds by raw col id; the save/restore
  pair for temporary pins (save bounds → `fix_col_ids` → solve → restore).
  `get_col_bounds` accepts arbitrary order and repeats (HiGHS' `getCols`
  itself requires an ordered duplicate-free set) and aligns its results
  positionally; `±inf` bounds round-trip exactly.
- **`WarmProblem.solve_with_fallback(fallback_options)`** — one-shot
  option-swap retry: solve, and ONLY if the primary fails to certify
  `kOptimal`, apply `fallback_options` via `setOptionValue` on the live
  handle (`solve(options=...)` is honoured on the first solve only), drop
  the retained solver state with `clearSolver()`, re-run once, and RESTORE
  the prior option values. Returns `(solution, "primary" | "fallback")`.
  The belt for drivers whose primary solve runs an aggressive option set
  (e.g. crossover-free interior point) that can come back non-optimal.

## [3.4.0] — 2026-07-01

### Added

- **`InOutStabilizer` (`polar_high.decomposition`)** — a domain-agnostic in-out
  separation point picker (Ben-Ameur & Neto 2007) for cutting-plane / Benders
  drivers. Cutting-plane methods that generate each cut at the raw master vertex
  `f_out` tail off badly when the recourse is flat in the coupling variable
  (dual-degenerate slopes): the master wanders among cost-equivalent vertices and
  the bound closes very slowly. The stabilizer instead returns an interior
  *separation point* `f_sep = λ·centre + (1-λ)·f_out` at which to generate the cut
  — better-centred cuts, no wandering, faster bound closure, at zero extra
  subproblem solves. Domain-free (operates only on `{col_id -> value}` point dicts
  and scalar weights); constructed ONE PER REGION. `λ = 0.0` is a verbatim no-op
  (returns its input unchanged, so byte-parity with exact Benders holds by
  construction). Convergence guarantee: the moment a region's cut fails to
  separate its `f_out`, the next `separation_point` returns the master point
  verbatim (`λ = 0` ⇒ exact Benders), and the null-step weight-shrink bottoms out
  at a forced `0` rather than a positive floor. Stateful, deterministic,
  side-effect-free.

## [3.3.0] — 2026-07-01

### Added

- **`StallMonitor` / `StallVerdict` (`polar_high.decomposition`)** — a
  domain-agnostic tail-off / stall detector for cutting-plane / Benders
  drivers (the outer loop that consumes `WarmProblem.add_cut_row` /
  `add_recourse_col`). It notices when the outer iteration has stopped making
  progress and lets the driver bail out with a diagnostic instead of silently
  burning the iteration cap. The monitor knows only the two scalars every
  Benders-style loop already maintains — a lower bound and a best-so-far upper
  bound — plus one caller-supplied `reference_scale` (a "sane objective
  magnitude" the driver computes from its own problem); it carries no domain
  concepts. A run is declared stalled only when a CONJUNCTION holds over a full
  trailing window: the relative gap exceeds `gap_floor` (far from converged),
  the best upper bound has not improved by more than `min_rel` (incumbent
  frozen), and the best upper bound is still above `blowup_mult * reference_scale`
  (frozen *far above* any sane magnitude — the penalty / complete-recourse
  regime). The conjunction is what separates a genuine tail-off from the benign
  frozen windows a converging run exhibits (early blow-up that shrinks fast,
  short flat stretches at a sane magnitude). Stateful (bounded `deque`),
  deterministic, and side-effect-free — it only reports.

## [3.2.0] — 2026-06-30

### Changed

- **Autoscale Layer 3 now centres the objective over HiGHS' cost comfort
  zone.** When a cost coefficient would trip a HiGHS warning — smallest
  `|c| < 1e-4` ("excessively small costs") or worst `|c| > 1e+4` — Layer 3
  picks the power-of-two `user_objective_scale` exponent that places the
  band's geometric centre `sqrt(min·max)` at the zone's geometric centre
  (`1.0`), instead of clamping the offending end to a boundary. For a band
  narrower than the zone both ends land inside `[1e-4, 1e+4]`; for a band
  wider than the zone the unavoidable violation falls **symmetrically** on
  both ends in log-space. Bands already inside the zone are untouched
  (`N = 0`), so well-scaled models are never perturbed. The exponent is a
  power of two and HiGHS unscales the objective and duals on output, so the
  reported solution is unchanged — only the magnitudes the simplex pivots
  on. The new audit tag is `center`.
- The previously-unhandled **small-cost** case (HiGHS' "Problem has some
  excessively small costs" warning) is now corrected: it was silently
  ignored before — only large costs were scaled.
- The pure **large-cost** case now centres as well (previously clamped the
  worst `|c|` to the `1e+4` working ceiling). The result is a slightly
  larger-magnitude down-scale; being power-of-two it remains
  output-invariant.

## [3.1.0] — 2026-06-25

### Added

- **`Solution.max_primal_infeasibility`** and
  **`Solution.primal_feasibility_tolerance`**: expose the solver's achieved
  unscaled primal slack and its primal feasibility tolerance, so a caller that
  hand-checks a constraint on the returned solution (e.g. a Benders master
  coupling self-check) can size its tolerance from the solver's own achieved
  feasibility instead of a hard-coded magic constant. HiGHS enforces feasibility
  on the internally-scaled problem, so the unscaled slack reported here can
  exceed the nominal scaled tolerance — a normal solver artifact callers must
  allow for. Both return `0.0` for a synthesised `Solution` with no live HiGHS
  handle.

## [3.0.0] — 2026-06-25

Refocuses regional decomposition: the built-in dual-subgradient
`LagrangianProblem` driver is **removed** and replaced by small, generic
cutting-plane primitives, on top of which the caller builds a Benders
decomposition. `Problem` / `WarmProblem` / `Param` and all engine + solver
exports are otherwise unchanged. (The subgradient driver shipped only in the
unreleased 2.7–2.9 line; those tags never reached PyPI, so **3.0.0 is the first
published release to carry this decomposition rework** — a `2.x → 3.0.0`
upgrade gains the primitives, not just loses the driver.)

### Removed

- **`LagrangianProblem` and the whole subgradient decomposition** (`lagrangian.py`:
  `LagrangianProblem`, `CouplingSpec`, `CouplingEntry`, `LagrangianSolution`, and
  the parallel-subsolve pool), plus their top-level exports. Its one consumer
  (FlexTool) moved to a Benders decomposition built from the cut-append /
  warm-restart / parallel primitives below, which give a tight bound without a
  subgradient tail. **Pin `polar-high<3` for the old driver.**

### Added

- **`WarmProblem.add_cut_row(col_ids, coefs, lower) -> int`**: append an
  optimality-cut row `Σ coefs·x >= lower` to the live (already-built) master and
  return its `row_id`; the cut dual is read by `Solution.row_dual[row_id]`. Plus
  **`add_recourse_col(name, cost, lower, upper)`** for lazy recourse columns. Both
  are post-build live edits that deliberately bypass the build-time DSL lock (it
  only guards the fixed-size autoscale side vectors). **No auto-scaling for
  appended rows** — keep the master autoscale off, or pre-scale `coefs` so the cut
  lives on the built columns' scale.
- **`WarmProblem.solve(*, retry_on_unknown=False)`**: warm re-solve after a cut
  append — the retained basis lets dual simplex hot-start across the appended `>=`
  row, falling back to a single `clearSolver` cold re-presolve only if the warm run
  fails to certify `kOptimal` (`kUnknown` / transient `kSolveError` / spurious
  `kUnbounded` off a stale basis). Default `False` is byte-identical to before. On
  a well-scaled master the warm path holds with no fallback, removing the
  super-linear cold-presolve cost that dominated the Benders master at scale.
- **`polar_high.parallel`** (exported at top level): `solve_indexed_parallel`
  (fan `fn(i)` across a thread pool, collect into per-index slots so the result is
  timing-independent; requires every `WarmProblem` already built, raises on an
  unbuilt one), `prewarm_global_scheduler` (pin the process-global HiGHS scheduler
  to one thread ONCE so concurrent `run()` calls stay single-threaded and
  deterministic), and `resolve_worker_count`. `workers <= 1` keeps a sequential
  path byte-identical to a plain `for` loop. Recovers the scheduler-pin pattern
  from the removed subgradient pool; enables parallel Benders region solves.
- **`WarmProblem.set_output_flag(enabled)`**: enable/disable HiGHS' native solve
  log for this problem; the preference persists on the handle across cold / warm /
  retry solves (applied immediately if built, else at the initial build — before
  the log routing, so `False` also suppresses the HiGHS version banner). Lets a
  driver that fans out many sub-solves (Benders regions) mute the per-sub-solve
  output and show its own concise log instead.

### Changed

- Coverage for two live `WarmProblem` methods (`update_obj_coef_array`,
  `fix_cols`) that the deleted Lagrangian tests exercised is relocated into
  `test_warm_problem.py`.
- Docs: the "Lagrangian" guide is replaced by **"Decomposition building blocks"**
  (cut-append / warm-restart / parallel); the API reference and cross-links are
  updated to the new public surface.

## [2.9.0] — 2026-06-24

Adds opt-in thread-parallel subsolves and a per-subsolve callback to the
Lagrangian driver, with HiGHS-log silencing for that path. Default
behaviour is unchanged — with no new kwargs the solve stays sequential,
fires no per-subsolve callback, and keeps today's verbose native log.

### Added

- **`LagrangianProblem.solve(max_workers=...)`**: optional cap on the number
  of worker threads used to solve subproblems concurrently within each
  barrier (initial build, per-iteration, primal recovery). `None` (default)
  or `1` keeps the fully sequential path. The effective count is clamped to
  `[1, n_subproblems]`. When `> 1`, every subsolve is forced to `threads=1`
  so each `h.run()` is deterministic (HiGHS is non-deterministic with
  `threads > 1`) and the box is not oversubscribed — two parallel solves with
  different `max_workers` are byte-identical. The COLD initial build also
  parallelizes ACROSS regions: the process-global HiGHS scheduler is pre-pinned
  to a single thread ONCE up front (`_prewarm_global_scheduler`), after which
  the per-region first solves fan out concurrently WITHOUT passing `threads`
  (so no per-instance `resetGlobalScheduler`). Parallelism is across regions
  only — each individual solve stays single-threaded on the pinned pool. If the
  one-time prewarm fails, the build falls back to a sequential cold loop on the
  calling thread (`threads=1` per first solve pins the scheduler) and the warm
  iterations still parallelize; the cold-parallel and cold-sequential builds
  are bit-identical. The executor is shut down on every exit path, including the
  no-coupling early return and any raised exception (fail-fast on the lowest
  non-optimal subproblem index, queued siblings cancelled).
- **`LagrangianProblem.solve(subsolve_callback=...)`**: optional callable
  invoked at the start and finish of every individual subproblem solve. It
  fires from worker threads when `max_workers > 1` and MUST be thread-safe;
  exceptions are suppressed so a faulty observer can never abort the solve.
  The callback dict schema (pinned):
  - start: `{"event": "start", "iter": <int>, "subproblem": <int>, "phase": <"initial"|"iterate"|"recovery">}`
  - finish: `{"event": "finish", "iter": <int>, "subproblem": <int>, "phase": <same>, "obj": <float>}`
    — `"obj"` is present only when that subsolve reached optimality.
  `phase` is `"initial"` for the build solve (`iter == 0`), `"iterate"` for
  outer iterations (`iter >= 1`), and `"recovery"` for the primal-recovery
  solve (`iter == -1`).

### Changed

- When the caller uses the new functionality (`max_workers > 1` or a
  `subsolve_callback`), the per-subsolve HiGHS native log is silenced. Set
  `POLAR_HIGH_LAGRANGIAN_VERBOSE=1` to force the verbose native log back on.
  Plain existing callers keep today's verbose native log unchanged.

## [2.8.0] — 2026-06-17

Retains each region's recovered-primal column values in the Lagrangian
result so a downstream caller can reconstruct every subproblem's primal
(e.g. investment-variable values) after the solve. Opt-in/backward-
compatible — the new field defaults to an empty list and existing callers
are unaffected.

### Added

- **`LagrangianSolution.subproblem_col_values`**: one numpy float64
  `col_value` array per subproblem (region), in subproblem order, each the
  region's FINAL recovered-primal column values. Populated on every return
  path of `LagrangianProblem.solve()` — the main subgradient/primal-recovery
  path (a region whose recovery solve is skipped/non-optimal falls back to its
  most recent iterate, so the list is always full-length and index-aligned)
  and the trivial no-coupling early-return path (from each subproblem's initial
  solve). Each array is layout-aligned with that region's own
  `subproblems[i]._vars[name].frame['col_id']`, so a caller can index those
  col_ids into entry `i` to assemble a whole-system primal from the per-region
  solves.

## [2.7.0] — 2026-06-17

Adds opt-in live progress reporting to the Lagrangian driver. Default
behaviour is unchanged — `solve()` stays silent unless a callback is passed.

### Added

- **`LagrangianProblem.solve(progress_callback=...)`**: an optional callable
  invoked once per outer subgradient iteration with that iteration's log dict
  (`iter`, `alpha_k`, `max_abs_residual`, `total_obj`), and once at the end with
  the final-summary dict (`iter == -1`, carrying `best_dual_total` /
  `recovered_total`). Lets callers stream live decomposition progress instead of
  only inspecting `iteration_log` after the fact. `None` (default) is a no-op and
  preserves the prior silent behaviour; callback exceptions are suppressed so a
  faulty observer can never abort the solve.

## [2.6.0] — 2026-06-16

Adds an opt-in small-coefficient cutoff. Default behaviour is unchanged —
byte-identical to 2.5.1 unless a caller sets the new threshold.

### Added

- **`Problem.coef_zero_threshold`** (default `0.0` = off): floors any LP matrix
  coefficient or RHS row-bound whose absolute value is below the threshold to
  exactly `0.0`, narrowing the numerical range (conditioning) of the LP for
  callers that opt in. Applied at every coefficient/RHS finalize point so it is
  independent of the build path — initial build (`_solve_streaming`,
  `_build_canonical_matrix`, including `row_lb`/`row_ub`) and warm in-place
  updates (`WarmProblem.update_param`, `update_rhs`). `±inf` / `NaN` sentinels
  are preserved and entries are replaced, never dropped, so matrix structure and
  determinism are unchanged; a threshold of `0.0` short-circuits to a no-op.

## [2.5.1] — 2026-06-11

Hardens the 2.5.0 HiGHS-log routing so it can never lose the log. The LP build,
autoscale, and solve numerics are unchanged from 2.5.0.

### Fixed

- `route_highs_log_to_stdout` suppressed HiGHS' native console write
  (`log_to_console=False`) and re-emitted via the `sys.stdout` callback on every
  routed solve. Suppressing native logging is a bet that the callback fires, and
  some `highspy` builds register the `kCallbackLogging` callback but never
  deliver a message — so suppress-native + silent-callback dropped the **entire**
  HiGHS log (observed in a Linux GUI subprocess on one machine, while an
  otherwise-identical machine with a different `highspy` was fine). The routing
  now suppresses native logging **only when `sys.stdout` is not backed by the
  native stdout fd (fd 1)**. When the sink already is fd 1 — a terminal, a pipe
  (e.g. a GUI reading a subprocess), or a file on fd 1 — HiGHS' own native log
  already reaches it, so native logging is left intact and the callback is
  skipped (new helper `_sink_is_native_stdout`). Routing + suppression now
  happens only where `sys.stdout` genuinely diverges from fd 1 (Windows Basic
  Console `OutStream`, `redirect_stdout`, pytest `capsys`), i.e. where the native
  write is unreachable anyway. `POLAR_HIGH_NATIVE_LOG=1` still opts out
  everywhere.

## [2.5.0] — 2026-06-10

Routes the HiGHS solver log through Python's `sys.stdout` so it is visible in
consoles that only capture the Python-level stream (not the native fd-1 write).
The LP build, autoscale, and solve numerics are byte-identical to 2.4.5; only
where the log *appears* changes.

### Added

- **`route_highs_log_to_stdout(h, *, stream=None)`** (new module
  `polar_high._log_routing`): registers a HiGHS `kCallbackLogging` callback that
  re-emits each message through `sys.stdout` (resolved lazily, so it follows
  later redirection such as ipykernel's) and suppresses the duplicate native
  console write (`log_to_console=False`) once the callback is confirmed
  registered. Idempotent (safe on a reused `Highs`), a no-op on silent solves
  (`output_flag` false), and fully defensive — any `highspy` error leaves the
  native logging path untouched rather than risking a lost log.

### Changed

- The in-process solve sites (`solvers/_highs.py`, the streaming
  `Problem.solve`, and `WarmProblem._initial_build`) now route the HiGHS log
  through `sys.stdout` by default, right after options are applied so the
  version banner is captured. This makes the solver log visible under the
  **Jupyter / Spine-Toolbox Basic Console on Windows** (where `ipykernel` only
  redirects fd 1 on POSIX, so the entire HiGHS log was previously lost) and
  under `redirect_stdout` / pytest `capsys`. Set **`POLAR_HIGH_NATIVE_LOG=1`**
  to opt out and keep HiGHS' native fd-1 logging.

## [2.4.5] — 2026-06-02

Docs-polish release. **No runtime or public-API changes** — the LP
build, autoscale, and solve paths are byte-identical to 2.4.4.

### Changed

- **Loading-data Memory section** rewritten to lead with what keeps the
  footprint down — the integer-keyed coefficient matrix (`col_id` /
  row id / `float64`, no string labels) and the section-by-section
  streaming build that releases each constraint family's input frames
  before the next — before the long-format constant factor. Notes that
  HiGHS column names embed the dim labels and carry their own cost
  (~1.1 GB at the 3000² grid), shed via `save_memory=True` or
  `write_mps(emit_names=False)`. Conclusion updated to match the current
  benchmarks: polar-high matches or beats linopy/xarray peak memory on
  the irregular network LP and on the dense `N × N` LP with
  `save_memory`.
- **Scaling guide**: the "falsely infeasible" risk is now scoped to
  badly-scaled models (eight or nine decades) rather than implied for
  any wide spread; the "who this is for" bullet list is replaced with an
  inline sentence; and a new section explains the three scaling layers
  (detection, semantic rewrites, HiGHS-native global scaling) and how
  they compose with HiGHS' own `simplex_scale_strategy`.
- **Performance guide**: the Threading section is rewritten — raising
  the thread count does speed up the build, with the trade-offs being
  per-thread scratch memory and fewer concurrent runs; the default of 1
  is tuned for the "many independent solves" deployment. Scaling is
  added to the solver-options list.

### Fixed

- **env-vars guide**: dropped the retired
  `POLAR_HIGH_RANGES_MAX_FAMILY_ROWS` "Workload tuning" knob — it was
  retired when autoscale range detection moved to the per-term walk that
  bounds peak memory regardless of family size; the env var is no longer
  read and setting it is a no-op. Replaced the dead FlexTool
  `dev/env_vars` documentation link (404) with the repository.
- Benchmark harness: sorted the `pulp_net` import block (ruff `I001`).

## [2.4.4] — 2026-06-02

Docs-polish release. **No runtime or public-API changes** — the LP
build, autoscale, and solve paths are byte-identical to 2.4.3.

### Changed

- **`Problem(dense_axes=...)`** now has substantive documentation: a
  new section in the Performance guide covers what the block-COO arm
  does (slice the dense suffix of each `Var`'s frame as a zero-copy
  numpy view, multiply with ufuncs), the row-sort contract the caller
  signs up to, suffix-matching against each `Var`'s dims, and
  `POLAR_HIGH_DISABLE_BLOCK_COO=1` for A/B rollback. The Concepts
  page picks up a short mention pointing readers in. API reference
  was already covered via the `Problem.__init__` docstring.
- **Enum dtype alignment**: the depth that lived in the README is
  rebuilt under `docs/guide/loading-data.md` with a concrete
  side-by-side example (`capacity_df` on a subset Enum vocab,
  `cost_df` on the full vocab) before the alignment-table contract.
  The README section is trimmed; the documentation index already
  links to the Guide.
- Benchmark page picks up the v2.4 numbers and a clarification that
  the `save_memory` trade-off axis is "how much work HiGHS does",
  not model size.
- `mkdocs.yml` drops a legacy CDN script from `extra_javascript`
  (MathJax 3 renders on every current browser without it).

### Fixed

- Benchmark plot harness: `polar_da` rows fold into the `polar` line
  on the dense plots so the published figure tracks the forthcoming
  block-COO-by-default behaviour without a redundant overlapping
  series. The network plot keeps `polar_da_net` as a distinct line
  where the irregular topology surfaces a small visible delta.

## [2.4.3] — 2026-06-01

Benchmark-methodology + docs release. **No runtime or public-API
changes** — the LP build, autoscale, and solve paths are byte-identical
to 2.4.1.

> **Note on 2.4.2.** The `v2.4.2` git tag was pushed before the
> `pyproject.toml` version bump landed, so the published 2.4.2 wheel
> carries `version = "2.4.1"` in its metadata. 2.4.2 has been yanked
> on PyPI; use 2.4.3, which contains the same source as 2.4.2 plus
> the correct version string. Pinned `polar-high==2.4.2` installs
> are unaffected — yanking does not remove the wheel.

### Changed

- Benchmark harness wraps each cell in a fresh `systemd-run --user
  --scope`. The worker reads cgroup v2 `memory.peak` from its own
  cgroup and emits it as a new `cgroup_peak_mb` column — the
  kernel-level peak the OOM killer would charge against a budget,
  less noisy across reps than the process-level `ru_maxrss` we
  previously plotted. Auto-falls back to plain subprocess on hosts
  where `systemd-run --user` is unavailable; `--no-cgroup-scope`
  forces the fallback. In the fallback path `cgroup_peak_mb` is NaN
  and `peak_rss_mb` continues to work.
- Benchmark grows two new polar variants alongside the existing
  `polar` / `polar_net` tools:
  - `polar_sm` / `polar_sm_net` — exercise `save_memory=True` so the
    harness produces directly comparable regular-mode and
    one-shot-mode numbers on the same hardware.
  - `polar_da` / `polar_da_net` — exercise the explicit
    `Problem(dense_axes=...)` contract on the dense and network LPs.
- `docs/compare/benchmark.md` refreshed against v2.4.x:
  - headline cells show both polar modes side-by-side with the
    cgroup peaks;
  - "Measuring memory" leads with `cgroup_peak_mb` as the canonical
    peak metric and demotes `peak_rss_mb` to a process-level note;
  - dense full-solve N=3000 peak drops 38.1 GB → 33.2 GB on polar
    regular (the autoscale memory-bounding work from 2.4.0 shows up
    here);
  - network-LP threading speedup at N=10 000 ticks up from 1.35× to
    1.40×; other rows refreshed accordingly.

## [2.4.1] — 2026-06-01

CI / test-tooling release. **No runtime or public-API changes** — the LP
build, autoscale, and solve paths are byte-identical to 2.4.0.

### Fixed

- `psutil` added to the `test` optional-dependencies. The bounded-memory
  branch-fired profile tests detect which evaluation branch ran via the
  env-gated profile lines on stderr, and those lines are `psutil`-gated:
  without `psutil` installed, `autoscale/_ranges.py` sets `_profile=False`
  and the signals never print, so the four branch-fired tests
  (`test_ranges_block_coo_branch_fired`, `test_ranges_rhs_bound_branch_fired`,
  `test_walk_fallback_profile_signal_fires`,
  `test_rhs_walk_fallback_no_skip_and_fires`) skipped their assertions in
  CI's lean `pip install -e ".[test]"` environment. Declaring `psutil` as a
  test dependency makes CI exercise the profiling signals. The bounded-memory
  feature itself was always correct and is unaffected — the byte-identical
  parity gate passes with or without `psutil`.

### Changed

- `ruff` lint + format cleanup of `src/` and `tests/` (import sorting, an
  unnecessary `open(..., "r")` mode argument, a stray unused local, and a
  whole-tree `ruff format` reflow) so the `lint` CI job passes under the
  latest `ruff`. No behavioural change.

## [2.4.0] — 2026-06-01

### Block-COO coefficient evaluation + bounded-memory autoscale

This release makes LP **build** and **autoscale** memory-bounded: no
constraint or objective family can spike RAM by materialising a wide
coefficient product. Two pillars:

**Block-COO evaluation (Phase C).** A polars frame sorted with its dense
axis trailing is physically a sequence of contiguous Arrow blocks; the
new path slices each block as a zero-copy numpy view and multiplies the
factors with ufuncs — no wide relational join, no wide intermediate. The
client declares its dense trailing axes once via the new
`Problem(dense_axes=...)` / `Problem.declare_dense_axes(...)` contract
(verified cheaply, O(n), no re-sort). `Sum`-wrapped chains are evaluated
in one pass via a captured `SumBlockMeta` reconstruction recipe, with a
relabel fast-path (reduce dims ⊆ var dims ⇒ a pure relabel, bit-identical
to the polars reduce) — the memory win for `Sum`-heavy families. Genuine
coefficient-combining sums stay bit-equivalent (FP reassociation ~1e-9).

**Bounded-memory autoscale (Phase D).** The autoscale Layer-1 range
readout and Layer-2 magnitude-bucketing used to materialise the wide
coefficient product just to read a statistic. Both now route through a
new general primitive, `bounded_coefficient_walk`, which iterates the
constraint/column spine in fixed row-batches and rebuilds each batch's
`(rid/col_id, coef)` via the block-COO builders + a prune-down backstop —
never holding more than one batch's product. Pluggable reducers compute
min/max (byte-identical, order-free) and the log2-magnitude histogram
(exponents may shift ±1 = an objective-invariant scaling change). The
reconstruction recipe is forwarded through post-`Sum` Expr-algebra
(scalar/Param mul+div, negation, subtraction, `Where`, and
`set_objective`'s collapse-all `Sum`) so every wide-product term — the
objective and negated-`Sum` constraints included — routes through the
walk instead of a materialising collect. The size-blind family-row skip
is retired: every shape is now bounded, with no silent coverage gaps.

Validated on a 9-roll rolling-horizon LP (FlexTool DES / RETO-Africa):
autoscale priv_dirty peak 46 → 23 GB, all objectives byte-identical,
~15% faster. The DSL is unchanged.

### Added
- `Problem(dense_axes=...)` / `Problem.declare_dense_axes(...)` — declare
  the pre-sorted dense trailing axes that enable block-COO evaluation.
- `bounded_coefficient_walk` + `CoefWalkRecipe` (`.from_term`,
  `.from_rhs_chain`, `.from_rhs_param`, `.is_buildable`) +
  `MinMaxAbsReducer` / `Log2HistogramReducer` in `autoscale/_coef_walk.py`.
- `POLAR_HIGH_DISABLE_BLOCK_COO=1` — fall every term back to the polars
  path (A/B rollback).
- `POLAR_HIGH_BLOCK_COO_PROFILE` / `POLAR_HIGH_RANGES_PROFILE` /
  `POLAR_HIGH_LAYER2_PROFILE` — env-gated instrumentation (no-op when off).

### Changed
- Autoscale Layer-1 (`_ranges.py`) and Layer-2 (`_layer2.py`) read
  coefficient ranges / magnitudes via the bounded walk. The ranges-PRE
  pass no longer gates the walk on the (not-yet-installed) Layer-2 side
  vectors, so it bounds every family there too.
- The `Sum` block-COO path defers map-effect `Where` via
  `_Term.where_map_frames` and forwards `sum_block_meta` through
  Expr-algebra; a re-reducing outer `Sum` still correctly drops the recipe.

### Removed
- `_skip_unbounded_over_cap` and `POLAR_HIGH_RANGES_MAX_FAMILY_ROWS` — the
  size-blind family-row cap, superseded by the bounded walk.

## [2.3.0] — 2026-05-29

### Where pushdown (added in the same release window as the prune-down work below)

`Where(expr, frame)` with a pure-filter shape (frame columns are a
subset of the expression's open dims — no map effect introducing new
dims) now defers the filter into a new `_Term.where_frames` slot
instead of inner-joining `frame` into `t.lazy` eagerly. The LHS
prune-down (`_build_lhs_pruned_plan`) then applies each recorded
frame at the Var leaf AND at every Param atomic during chain rebuild —
mirror of the existing row_index pre-prune pattern. Net effect: the
filter narrows every intermediate, not just the final result.

Pure-filter `Where` now also PRESERVES `var_source` / `param_sources` /
`coef_scalar` on its output term (today's path cleared them). This
closes the Where leg of the "Sum/Where/Lag wrapping" limitation
flagged after the 2026-05-28 audit — LHS prune-down now fires on
Where-wrapped terms in addition to bare `Var × Param × …` chains.
Sum / Lag bake `where_frames` into `t.lazy` before consuming it
(they change row identity); the Sum leg remains future work.

Behaviour-preserving on every tested scenario: the LP matrix is
byte-identical between the deferred-pushdown path and the env-var-
disabled (eager-join) path. Validated on FlexTool's DES (RETO-Africa)
scenario — 5.6M rows × 4.6M cols, same presolve reductions, same
coefficient ranges. DES itself does not exercise any of the
pushdown-eligible families (no per-process-profiles, no commodity
ladder, no investment, no reserves), so no measurable RSS delta is
expected there; richer scenarios with pure-filter Wheres over
multi-atomic chains will see the win.

### Added — Where pushdown

- `POLAR_HIGH_DISABLE_WHERE_PUSHDOWN=1` — safety fallback env var. When
  set, every `Where` call eagerly inner-joins `frame` into `t.lazy` and
  clears the leaf metadata exactly as the pre-v2.3.0 path. Use as an
  opt-out if a model surfaces unexpected drift on the pushdown path.

- `_Term.where_frames: tuple[pl.LazyFrame, ...] | None` slot — opt-in
  metadata recording pure-filter `Where` frames so they can be applied
  at the leaves during chain rebuild. Internal — no public API change.

- `_apply_where_frames(lazy, dims, where_frames)` helper — used by
  `Sum` / `Lag` to bake pending filters before consuming `t.lazy`,
  and by consumer fallback paths in `_build_canonical_matrix` /
  `_solve_streaming` / `WarmProblem._initial_build` when leaf-rebuild
  prune-down can't fire.

- `tests/test_where_pushdown_parity.py` (11 tests) — parity coverage
  for pure-filter, map-effect, nested Where, Where-after-Sum,
  Sum-after-Where, anonymous-Param-chain, scalar-fold, Where-then-mul-
  Param, disable-guard, shared-empty-extras-nonempty, and RHS-Where-
  preserved-through-negation.

### Fixed (latent — exposed by the parity sweep)

- `Where(expr, frame)` with `shared == [] and extras != ()` now
  explicitly cross-joins `frame` instead of silently claiming the
  extras columns on `_Term.dims` without ever producing them in
  `t.lazy`. Pre-fix this was a corruption waiting to happen; no
  known caller relied on the broken behaviour.

### Removed (dead code)

- The `elif isinstance(rhs, (Var, Expr))` / standalone-block negation
  patterns in `_build_canonical_matrix`, `_solve_streaming`, and
  `WarmProblem._initial_build` are deleted. `Problem.add_cstr` folds
  Var/Expr RHS into the LHS via `Expr.__sub__` before storing in
  `_CstrProto`, so `proto.rhs` only ever reaches those sites as a
  Param or scalar — the elif branches were unreachable.

### Prune-down (initial v2.3.0 work — merged earlier in the same release window)

Memory cliff fix for `Param`-chain RHS / LHS in `_build_canonical_matrix`
plus matching coverage in `_solve_streaming` and `WarmProblem._initial_build`.
On FlexTool's DES (RETO-Africa) scenario the canonicalise stage of the
first solve stalled inside `profile_flow_upper_limit` (1.5 M rows, RHS
`= profile_value × process_existing_count × process_availability`) — the
chained inner joins produced a ~2.6 billion-row Cartesian intermediate
before the row_index semi-join could prune it. The fix walks each
chain's named atomic Params and pre-prunes them against the constraint's
row_index keys (projected onto each atomic's own dim subset), bounding
the intermediate to the constraint row count.

Behaviour-preserving: every solve that completed before still produces
identical numerics (verified against the FlexTool scenario parity suite,
139 polar_high tests + the previously-failing flextool scenarios). LP
matrices are byte-identical between the prune-down path and the
original merged-lazy path on every covered chain shape; the difference
is solely intermediate peak memory.

### Added

- `POLAR_HIGH_SOLVE_PROFILE=1` — env-var-gated stderr profile lines
  covering every meaningful sub-step of `Problem._solve_streaming`
  (cold path, 27 checkpoints) and `WarmProblem._initial_build` (18
  checkpoints, including the per-family LP-build loop and the HiGHS
  handoff). Tab-separated `[solve profile] phase=… rss_gb=… delta_gb=±…
  wall_s=…` format mirroring the `POLAR_HIGH_WRITE_MPS_PROFILE`
  precedent. Zero overhead when unset.

- `POLAR_HIGH_DISABLE_PRUNE_DOWN=1` — safety fallback env var. When set,
  every multi-atomic `Param` chain in RHS / LHS handling falls through
  to the merged-lazy semi-join path. Use as an opt-out if a future model
  surfaces an unexpected numerical drift on the prune-down path.

- `Param._value_scalar` and `_Term.coef_scalar` slots — accumulate
  scalar folds (`Param * float`, `Var * float`, `Expr.__neg__`,
  `Expr.__sub__`, etc.) so the prune-down rebuild can seed its
  accumulator with the correct multiplicative constant. Without this
  tracking the rebuild would silently drop scalar factors that the
  merged-lazy path carries in the `value` / `coef` column. Internal —
  no public API change.

- Per-family and per-term checkpoints inside `_build_canonical_matrix`
  (gated by `POLAR_HIGH_WRITE_MPS_PROFILE=1`). New labels:
  `family_rhs_evaluated`, `family_rhs_l2baked`, `family_senses_built`,
  `family_rownames_built`, `family_term_plans_built`,
  `family_term_collect_start` / `family_term_collected` (per LHS
  term), `family_rhs_pruned_down` (new prune-down path),
  `family_lhs_scattered`. Each emits `family=` and `family_idx=`
  extras so per-family slicing is trivial.

- Reference tests:
  - `tests/test_canonicalise_param_chain_prune.py` (3 tests) — RHS
    prune-down parity vs merged-lazy path on synthetic 3-Param chain
    with disjoint-but-shared dims; covers `Param.__truediv__`.
  - `tests/test_lhs_param_chain_prune.py` (3 tests) — LHS prune-down
    parity at all three call sites (`_build_canonical_matrix`,
    `_solve_streaming`, `WarmProblem._initial_build`).
  - `tests/test_prune_down_scalar_anonymous_fix.py` (6 tests) —
    anonymous-Param-in-chain handling, scalar-fold tracking, sign
    propagation through `Expr.__neg__` / `__sub__`, and the disable
    env-var fallback.
  - `tests/test_lp_view.py` test for `to_csr` post-vectorisation
    (zero-copy CSR round-trip parity).

### Changed

- `_build_canonical_matrix` RHS handling (engine.py): when
  `rhs._sources` is a chain of length ≥ 2 and the composite has dim
  columns, walk the atomics one at a time. Each atomic is semi-joined
  against the running accumulator's key projection (semi-join order:
  acc keys → atomic, NOT the other way around — atomic frame is the
  pre-pruned side). Final accumulator collects via the existing
  streaming fallback chain. Single-Param / scalar / `Var`-or-`Expr`-on-RHS
  branches unchanged.

- `_build_lhs_pruned_plan` (new helper) + three LHS call sites
  (`_build_canonical_matrix` L1664-1692, `_solve_streaming` L2738-2763,
  `WarmProblem._initial_build` L3969-4002): when `term.param_sources`
  has length ≥ 2 AND `term.var_source` is set (i.e. the term has a
  direct Var anchor — not wrapped in `Sum` / `Where` / `Lag` which
  clear `var_source` to preserve safety), rebuild the LHS plan as
  `row_index ⋈ pruned_var ⋈ pruned_param_1 ⋈ pruned_param_2 …` with
  each factor pre-pruned via semi-join. Sum / Where / Lag wrapped
  terms fall back to the original path.

- `_lp_view.to_csr` row index construction: replaced a Python
  `for c in range(n_cols): col_of[a_start[c]:a_start[c+1]] = c`
  loop with `np.repeat(np.arange(n_cols), np.diff(a_start))`.
  Output identical (verified in `test_lp_view.py`); about 100× faster
  on a sparse 5 M-row LP.

- `_build_canonical_matrix` variable loop: merged the two consecutive
  loops over `self._vars.values()` (col bounds / integrality and
  `col_names` construction) into a single pass. Each
  `v.frame["col_id"].to_numpy()` materialises once instead of twice.

- ~32 `.astype(np.int64)` / `.astype(np.float64)` call sites on
  freshly-allocated numpy arrays (from `.to_numpy()`, `np.where`,
  `np.repeat`, `np.tile`, `np.concatenate`) gained `copy=False`.
  Affected: `_build_canonical_matrix` per-family scatter (`dim` and
  `scalar` branches), global / family dedup, objective-term collect,
  HiGHS bound translations in `_build_lp_arrays` / `_solve_streaming`
  / `_initial_build`, RHS `np.where` row_lb/row_ub translations,
  tracked-source scatter in `WarmProblem`, and `_lp_view.from_problem`
  bound round-trip.

### Fixed

- Anonymous `Param` instances (no `name`, no `_sources`) were silently
  dropped from `_merge_param_sources` output when participating in a
  chain with named Params. The prune-down rebuild then walked only the
  named atomics, missing the anonymous one's contribution. Fixed:
  `_sources_for_propagation` now returns `[(self, +1)]` for anonymous
  atomics so the chain rebuild walks every constituent.

- Scalar folds (`Param * float`, `Var * float`, `Expr * float`,
  `Expr.__neg__`, `Expr.__sub__`) collapsed constants into the
  `value` / `coef` column without recording them in `_sources` /
  `param_sources`. The prune-down rebuild had no way to see the
  scalars and produced numerically-different LP coefficients (DES
  scenario parity tests caught this as a ~2 % objective drift on
  `test_fullYear_roll_matches_v3320_golden`). Fixed via the new
  `Param._value_scalar` / `_Term.coef_scalar` slots; affected algebra
  ops propagate the scalar through to the prune-down accumulator.

### Performance

- Canonicalise of FlexTool DES (RETO-Africa) on a 64 GB box:
  - Before: stalled inside `profile_flow_upper_limit` RHS evaluation
    after ~10 seconds, RSS climbing from 12.7 GB toward a peak of ~38 GB
    that exceeded available memory in some configurations.
  - After: all 9 constraint families canonicalise in 27 seconds total;
    `_initial_build` exits at 49 seconds; peak RSS during the in-process
    HiGHS solve path is 23.3 GB. With `--save-memory` (subprocess
    HiGHS), peak is 15.2 GB.

- Wall-time impact of the perf quick-wins (`copy=False`, vectorised
  `to_csr`, merged var-loop): 5-15 % wall-time win on the
  canonicalise + HiGHS-handoff portion of large LPs. Memory peak win
  is small (5-10 %) — these are a separate stack from the prune-down
  fix and apply per-cell rather than per-chain.

### Notes

- The fix is behaviour-preserving on every currently-tested scenario.
  If a future model surfaces a numerical drift, set
  `POLAR_HIGH_DISABLE_PRUNE_DOWN=1` as a workaround and report the
  scenario so the engine can be fixed.
- The LHS prune-down activates only on terms with a direct Var anchor
  (not wrapped in `Sum` / `Where` / `Lag`). Sum-wrapped LHS chains
  fall back to the merged-lazy path; this is intentional for safety.
  See `specs/block_coo_evaluation_handoff.md` for the planned follow-on
  that handles those cases via a different mechanism.
- See `specs/where_pushdown_handoff.md` for the next architectural
  step (push `Where(...)` filter keys through the lazy plan tree the
  same way row_index keys are pushed today).

## [2.2.0] — 2026-05-28

GLPK-style refactor of the Layer 2 scaling pipeline and the matrix
emit path.  Two architectural changes that, together, eliminate the
"every consumer rebuilds the matrix from scratch" pattern that the
v2.1.x line was tactically patching.

### Added

- `Problem.canonicalise()` — lazily builds and caches a single
  canonical CSC representation of the LP on `Problem._matrix` (a
  new `_CanonicalMatrix` slot-dataclass carrying `col_ptr` /
  `row_idx` / `val` plus per-row `lb`/`ub`/`sense` and per-col
  `obj`/`lb`/`ub`/`integrality` and names).  Idempotent — repeat
  calls return the cached matrix unless `_canonical_dirty` is set.
  `add_var` / `add_cstr` set the dirty flag; the cached matrix is
  released by `_release_python_lp_inputs` and
  `write_mps(release=True)`.

- `Problem._layer2_col_factor` / `Problem._layer2_row_factor` —
  numpy side vectors written by `flextool`'s `apply_layer2`.  The
  col-factor vector stores `1 / cf_math` (inverse); the row-factor
  vector stores `rf_math` (forward).  At canonicalise time the
  vectors are *baked* into `_matrix.val` / `_matrix.col_obj` /
  `_matrix.row_lb` / `_matrix.row_ub` so consumers read pre-scaled
  values directly.  `Problem._layer2_locked` prevents post-Layer-2
  `add_var` / `add_cstr` that would invalidate the side-vector
  sizing.

- Regression tests `tests/test_layer2_side_vector_emit.py` and
  `tests/autoscale/test_ranges.py::test_ranges_via_streaming_honors_side_vectors`
  — exercise every emit-site branch with explicit fake side
  vectors so missed multiply sites or indexing offsets fail fast
  without depending on flextool's bit-for-bit integration test.

### Changed

- `Problem.write_mps` (Stage B1) — now calls `canonicalise()` and
  walks `_matrix.col_ptr` column-by-column.  The previous per-family
  triple-list build, group_by dedup, concat, and global sort are
  consolidated into `_build_canonical_matrix` (runs once per
  state-version, shared across all consumers).  Cross-consumer
  workflows (write_mps then solve, or any combination of the four
  canonical-consuming sinks) now family-walk exactly once.

- `Problem._build_lp_arrays` (Stage B2) — reduced to ~30 LoC.
  Reads from `_matrix` and applies the `±inf → kHighsInf`
  substitution.  Per-family LHS walk, per-family RHS, Stage A
  multiply-at-emit, global dedup + sort all moved into
  `_build_canonical_matrix`.  Back-compat shim parameters
  (`n_cols`, `col_lb`, `col_ub`) removed from the signature; the
  two callers (`LpView.from_problem`, `_ranges_via_passmodel`)
  updated.

- `WarmProblem._initial_build` (Stage B3) — bulk LP build
  (LHS / RHS / obj / bounds) now reads from `_matrix`.  Tracked-
  source bookkeeping for `WarmProblem.update_param` keeps a small
  separate walk over `_cstrs` filtered to terms with
  `param_sources` set; these terms re-collect and apply the same
  Stage A multiply-at-emit so the cached `_param_cells` factors
  remain the SCALED coef (matches the pre-refactor formula that
  `update_param` relies on).  Skipped entirely when
  `self._mutable_params` is empty.

- `LpView.from_problem` — reads `m.col_obj` from the canonical
  matrix instead of walking `_obj_terms` and applying its own
  Stage A multiply-at-emit.  After this commit, the ONLY
  remaining Stage A multiply-at-emit consumer is
  `Problem._solve_streaming` — intentional, its per-family CSR
  memory bound exists specifically to avoid full-matrix
  materialisation.

- `polar_high.autoscale._ranges._ranges_via_streaming` honours
  the side vectors.  When called post-Layer-2 it multiplies
  per-term `abs(coef)` by `|row_factor| * |col_factor|` after the
  polars collect (numpy, in place, no lazy plan modification) so
  the readout sees the same effective magnitudes the consumers
  will emit.  No-op when the side vectors are `None` (the
  pre-Layer-2 readout pattern is unchanged).

### GLPK-likeness scorecard

| Property | GLPK | Pre-v2.2.0 | Post-v2.2.0 |
|---|---|---|---|
| Matrix is canonical (one copy) | ✓ | ✗ (per-family lazy + 2-3 transient copies during emit) | ✓ |
| Scaling lives separately from coefs | ✓ | ✗ (rewrote lazy plans via flextool's `_layer2`) | ✓ |
| Objective excluded from row scaling | ✓ | n/a (no row scaling) | ✓ |
| Build/scale is O(m+n+nnz) | ✓ | ✗ (transient triple copies + dedup hash + global sort coexisted) | Mostly ✓ (canonicalise still has transient peak ~3× nnz; consumers no longer rewalk) |

The remaining "Mostly" on build/scale is the transient peak during
`_build_canonical_matrix` itself: per-family triples → global concat
→ polars `group_by` dedup → sort → final CSC.  A per-family
streaming canonicalisation (consumers process families one at a
time, dedup per-family, merge sorted CSC chunks) would close this
gap by exploiting the disjoint-row-range property each family
already has.  Future work.

## [2.1.3] — 2026-05-27

### Fixed

- `Problem._build_lp_arrays`, `Problem._solve_streaming`, and
  `WarmProblem._initial_build` now use the same semi-join + per-term
  streaming-collect pattern that v2.1.2 added to `Problem.write_mps`.
  Plain-inner-join + `pl.collect_all` previously materialised deep
  multi-Param LHS chains and multiplied the peak via parallel
  collect.  Per-term peak is now bounded by `row_count × cols-per-row`
  instead of the upstream Param-product cardinality.

- `polar_high.autoscale.detect_ranges` on a pre-solve `Problem` now
  bypasses `Problem._build_lp_arrays` entirely.  New
  `_ranges_via_streaming` walks objective + constraint terms one at
  a time with a semi-join + per-row `abs(coef)` collect (the shape
  `write_mps` proves on the same chains) and numpy-reduces to
  min/max.  Avoids materialising the full COO triple list +
  global dedup that the legacy `_ranges_via_passmodel` ran for the
  same readout.  Legacy code stays for back-compat but is no longer
  reached from production callers.

### Added

- `POLAR_HIGH_RANGES_MAX_FAMILY_ROWS` env var (default `1000000`,
  `0` to disable) — skips constraint families above the threshold
  in `_ranges_via_streaming`'s RHS + matrix readout.  Background:
  polars' streaming engine intermittently fails to push the row-key
  semi-join into deep multi-Param product chains on very large
  families, so a single term collect can allocate >30 GB before
  failing on workloads like FlexTool's DES LP
  (`profile_flow_upper_limit` at 1.5 M rows × multi-Param rhs).
  Skipping means the range report rides on the families it could
  read.

- `POLAR_HIGH_BUILD_LP_PROFILE=1` and `POLAR_HIGH_RANGES_PROFILE=1`
  diagnostic env vars — per-family / per-phase `psutil` RSS deltas
  to stderr from `Problem._build_lp_arrays` and from the autoscale
  range detectors respectively.  Zero overhead when unset.

- Regression test
  `test_detect_ranges_param_chain_does_not_explode` — synthetic
  200k-row Var × 3-Param chain.  Pre-fix peak 515 MB on this shape;
  post-fix under 300 MB (test asserts <300 MB).

## [2.1.2] — 2026-05-27

### Fixed

- `Problem.write_mps` per-term collect on LHS Param-chain terms
  (e.g. `Var * Param₁ * Param₂ * ...`) used to materialise the join
  chain's wide intermediate before the final row alignment.  On a
  9.9 M-row LP a single such family allocated +26 GB during one
  `term.lazy.collect()`, pushing `write_mps` peak to ~43 GB despite
  the spec target of 2-3 GB.  Retrofitted the same anti-explosion
  pattern the RHS path has used since v2.0.0
  (`_align_enum_join_keys` → semi-join against the row-index key
  set → `collect(engine="streaming")` with `streaming=True` and
  plain-`.collect()` fallbacks).  Synthetic 100 k-row test:
  527 MB peak → 178 MB after the fix.  Coefficients byte-identical
  with the pre-fix path; HiGHS objective unchanged.

## [2.1.1] — 2026-05-27

### Added

- `docs/guide/performance.md` — new section "Writing MPS without HiGHS"
  covering `Problem.write_mps`: API, the ~20× peak-memory advantage
  over `Highs.writeModel`, `release=True` semantics, cross-solver
  roundtrip coverage, and the `POLAR_HIGH_WRITE_MPS_PROFILE=1`
  diagnostic env var.  Cross-linked from `docs/guide/scaling.md` and
  `docs/guide/solvers.md`.

### Fixed

- Replaced a stale reference to a fictional `Problem.solve(write_mps=...)`
  kwarg in `docs/guide/solvers.md` with the real `Problem.write_mps`
  link.
- Ruff lint warnings in `tests/_bench_write_mps_parallel.py` (intentional
  late polars import) and in the `RHS` section of `Problem.write_mps`
  (unused tuple element renamed `_row_count`).

## [2.1.0] — 2026-05-27

### Added

- `Problem.write_mps(path, *, free_format=True, column_order_strict=True,
  emit_names=True, release=False, name="POLAR_HIGH")` — direct
  polars→MPS writer that bypasses `highspy.Highs.writeModel`.  Mirrors
  the per-family streaming pattern from `_solve_streaming`, performs
  one streaming sort by `(col_id, row_id)`, and chunked-streams the
  COLUMNS section with `INTORG`/`INTEND` integer markers.  Target peak
  is ~2–3 GB on a 10 M-row / 5 M-col / 20 M-nz LP — about 20× lower
  than `Highs.writeModel`'s transient.  `release=True` reuses the same
  `_release_python_lp_inputs` teardown as `solve(save_memory=True)` so
  callers driving an out-of-process solver can drop the polar-side LP
  source immediately after the write.
- `POLAR_HIGH_WRITE_MPS_PROFILE=1` env var — when set, `Problem.write_mps`
  emits per-phase and per-constraint-family `psutil` RSS deltas to
  stderr for diagnosing memory hot spots.  Zero overhead when unset
  (no `psutil` import, no closure call sites entered).
- `tests/_bench_write_mps_parallel.py` — synthetic-LP bench for
  `write_mps` peak memory across single-family / multi-family
  topologies and polars thread counts.

### Changed

- Wrapper-driven MPS roundtrip harness (`tests/test_mps_fallback_wrapper.py`)
  is now parametrized over both the legacy `LpView`-based writer and the
  new `Problem.write_mps` direct writer, so the HiGHS / Gurobi / CPLEX /
  Xpress readback tests exercise both code paths.

## [2.0.2] — 2026-05-26

### Added

- `docs/guide/scaling.md` — user-facing guide for the
  `polar_high.autoscale` package: when to use it, the typical
  `detect_ranges → recommend_scaling → apply_scaling` pattern,
  `ScalingMode` / `ScalingConfig` knobs, the precedence rules, the
  min-floor guard + geometric-centring escape branch, and migration
  from the retired `auto_user_bound_scale=True` flag.  Wired into
  `mkdocs.yml`'s Guide section between Solvers and Warm-starting.

### Changed

- Stripped proper-name callouts of specific caller-side LPs from
  source comments and CHANGELOG entries.  Replaced with generic
  scenario descriptions ("a full-year LP with RHS=(1.84e-3,
  2.02e+8)" etc.) so the technical narrative survives without
  leaking caller-side LP names.

## [2.0.1] — 2026-05-26

### Fixed

- CI `ruff check` and `ruff format --check` failures inherited from
  the v2.0.0 commits.  Sorted / removed unused imports, ran ruff
  format, and migrated `ScalingMode(str, Enum)` → `ScalingMode(
  enum.StrEnum)` to clear the UP042 hint.  Behaviour difference:
  `str(ScalingMode.OFF)` now returns `"off"` instead of
  `"ScalingMode.OFF"`; no code in `src/` or `tests/` stringifies the
  enum, so this is invisible at the API boundary.

### Changed

- Cross-solver MPS-fallback tests now have sharpened skip strings
  that distinguish *"wrapper-installed-but-CLI-binary-missing"* from
  *"solver wholly absent"*, and point at the new wrapper-driven test
  file for parallel coverage when only the Python wrapper is present.

### Added

- `tests/test_mps_fallback_wrapper.py`: for each commercial solver
  whose Python wrapper is installed (Gurobi, CPLEX, Xpress), writes
  the polar-high MPS file, reads it back into the wrapper, solves,
  and asserts the objective matches a direct in-memory HiGHS solve.
  Catches MPS-format issues end-to-end without needing the
  standalone CLI binary.  COPT is intentionally out of scope here
  due to the in-process COPT/HiGHS native-symbol conflict
  documented in `solvers/_copt.py`.

## [2.0.0] — 2026-05-26

Headline: much-improved automatic LP scaling via a new
`polar_high.autoscale` package. The previous one-shot
`auto_user_bound_scale=True` constructor flag is **retired** and
replaced by a richer caller-driven API that detects bound / cost /
RHS / matrix ranges and recommends `user_bound_scale` and
`user_objective_scale` exponents independently. The new path also
adds a min-floor guard that catches a class of false-infeasibility
results HiGHS' own `suggestScaling` can produce on wide-spread LPs.
See the [Scaling guide](https://nodal-tools.github.io/polar-high/guide/scaling/)
for the full caller story.

### Added — autoscale

- `polar_high.autoscale` package with three pieces:
  - `detect_ranges(problem_or_solution, config)` returns a
    `RangeReport` with the four `(abs_min, abs_max)` tuples
    (`matrix`, `cost`, `col_bound`, `row_bound`) plus per-category
    samples of smallest / largest contributors, usable on a built
    `Problem` *or* on a returned `Solution` (re-uses
    `Solution.streamed_lp_ranges` when available).
  - `recommend_scaling(ranges, config)` returns a `Layer3Plan` with
    `user_bound_scale` and `user_objective_scale` integer exponents,
    derived from HiGHS' own `suggestScaling` formula. Preserves the
    geometric-centering escape branch for severe asymmetric-bound
    LPs, now guarded by a min-floor check (see *Fixed*).
  - `ScalingMode` enum (`OFF` / `SOLVER_ONLY` / `BASIC` / `FULL`)
    with helper predicates so library callers can decide policy per
    mode rather than per-call kwarg.
- Precedence check: an axis whose `user_*_scale` is already set by
  the caller (via `set_solver_options` or per-call `options=`) is
  skipped by `recommend_scaling`. The caller's explicit value always
  wins.
- `Problem.set_solver_option(name, value)` and
  `Problem.get_solver_option(name)` accessors as the clean surface
  the precedence check reads from.

### Removed (breaking)

- `Problem(auto_user_bound_scale: bool = ...)` constructor option.
  The flag's one-shot, col-bound-only heuristic is superseded by
  `autoscale.recommend_scaling()`, which considers all four ranges
  independently and is configurable per-mode. Callers should:
  1. Build the `Problem` as before.
  2. Call `detect_ranges(p, config)` and then `recommend_scaling(
     ranges, config)` for the chosen `ScalingMode`.
  3. Apply the returned `Layer3Plan` via `Problem.set_solver_option`.
  See the autoscale package docstring for the migration pattern.
- The internal `_recommend_user_bound_scale` helper that backed the
  retired flag.

### Fixed

- **False-infeasibility from over-aggressive scaling.** When HiGHS'
  own `suggestScaling` looks only at the *max* of `(bound_max,
  rhs_max)`, it can pick a `user_bound_scale` exponent that crushes
  the *min* below `kExcessivelySmallBoundValue` (1e-4). HiGHS'
  presolve then mis-handles the near-zero rows and the LP comes
  back infeasible. Observed on a full-year LP with
  RHS=(1.84e-3, 2.02e+8): the formula picked N=-8 → scaled RHS min
  7.2e-6 → spurious infeasibility. The new `recommend_scaling`
  adds a min-floor guard: when the proposed delta would drag the
  scaled min below the threshold, the current scale is returned
  unchanged.
- **Duplicate-key rhs Param fan-out.** The left-join from
  `row_index` against an upstream Param with duplicate `(on=)` keys
  used to surface as an opaque
  `ValueError: operands could not be broadcast together with shapes
  (X,) (Y,)` deep inside the solver adapter. `_build_lp_arrays`
  (and the chunked / WarmProblem variants) now raise immediately
  at the join boundary, naming the offending constraint plus a
  sample of the duplicate keys.
- **`--highs-threads N>1` silently ignored.** HiGHS'
  `setOptionValue("threads", N)` is a no-op once the global Rayon
  scheduler has been initialised (which happens at default
  `threads=16`). We now call `Highs.resetGlobalScheduler(False)`
  before applying the user's options so the requested thread count
  actually takes effect.

### Notes

- The 1.5.x releases (sidecar RSS sampler, `save_memory=True`
  one-shot mode, chunked LP-range accumulator) are subsumed under
  2.0.0; their entries remain below as the detailed history.

## [1.5.1] — 2026-05-24

### Changed

- `docs/compare/benchmark.md`: trim the "how this differs from
  earlier versions" methodology paragraph (covered in the 1.5.0
  changelog entry) and minor wording cleanup.

## [1.5.0] — 2026-05-24

### Added

- `Problem.solve(save_memory: bool = False)` opt-in one-shot mode for
  benchmark-style single solves. When `True`, polar-high drops its
  Python-side LP source-of-truth (term lazy plans, Param frames,
  caller-side column-bound / cost arrays, and the `col_names` /
  `row_names` lists) once HiGHS has copied them, and then writes the
  model to a temp MPS file, clears the original `Highs` instance,
  calls `malloc_trim(0)` to return glibc arenas to the OS, and
  creates a fresh `Highs` that reads the model back before
  `h.run()`. The disk roundtrip resets HiGHS' incremental-`addRows`
  allocator slack — at N=3000 dense full-solve it drops peak RSS
  from ~38 GB to ~28 GB at the cost of ~+90 s wall time (the MPS
  write + read). A subsequent `Problem.solve()` on a Problem that
  has been released raises a clear `RuntimeError`; WarmProblem-style
  incremental updates and re-solves are unavailable after
  `save_memory=True`. Cold-start rolling-horizon loops that rebuild
  the `Problem` from scratch each iteration are unaffected and
  benefit from the per-iteration memory drop. Default `False`
  preserves the warm-restart-capable behaviour.

### Changed

- `_running_finite_nonzero_min_max` (used by the streaming
  LP-range accumulator) now scans in chunks of 1 M float64s instead
  of materialising `np.abs(arr[finite])` for the whole array. On a
  36 M-nonzero constraint family that cuts the transient temp
  allocation from ~576 MB to ~16 MB. Functionally identical output.
- `_solve_streaming` no longer concatenates `col_lb_h` with
  `col_ub_h` (or `row_lb` with `row_ub` per family) for range
  accumulation — each is scanned in place. Eliminates a 2·n_cols
  (and 2·n_rows-per-family) transient copy each.
- `_solve_streaming` drops `col_lb_h` / `col_ub_h` / `col_obj_h`
  immediately after the LP-range accumulation completes — HiGHS has
  its own internal copies from `addCols`, so the originals are not
  needed through the family loop and `h.run()`. ~432 MB at N=3000
  dense.
- Column-array construction (`col_lb` / `col_ub` / `col_obj` /
  `col_int` / `col_names`) moved from `Problem.solve()` into
  `_solve_streaming` so the caller's frame doesn't pin those arrays
  through the entire family loop and `h.run()` call. Combined with
  the drop above, this removes ~864 MB of caller-side residue at
  N=3000 dense.
- `benchmark/run_one.py` now starts a sidecar thread that samples
  `VmRSS` from `/proc/self/status` at 25 ms cadence while `solve()`
  runs, and calls `malloc_trim(0)` after `gc.collect()` at the
  post-build and post-solve checkpoints. New CSV columns:
  `rss_after_build_trim_mb`, `rss_after_solve_trim_mb`,
  `rss_solve_min_mb`, `rss_solve_p50_mb`, `rss_solve_p95_mb`,
  `rss_solve_max_mb`, `n_samples`. The `peak_rss_mb` column stays
  as before (`ru_maxrss`, the unavoidable high-water mark including
  transient HiGHS-setup scratch). Old 10-field CSV rows still parse
  through `plot.py`.
- `docs/compare/benchmark.md` rewritten: new memory-measurement
  methodology section explains `peak_rss_mb` vs `rss_solve_p50_mb`
  vs `rss_after_solve_trim_mb`; new section on the regular vs
  `save_memory` modes with a side-by-side polar-high comparison;
  headline tables updated to use `save_memory=True` for the
  cross-tool comparison (matches linopy's `io_api="lp"` file-handoff
  pattern). Threading-benefit numbers updated — speedup at N=10 000
  is now 1.18× rather than 1.33× because the MPS roundtrip is a
  serial step that doesn't scale with thread count.

## [1.4.0] — 2026-05-22

### Removed

- `Problem.peek_lp_ranges()`. The method rebuilt the full LP into
  numpy arrays via the non-streaming path purely to extract coefficient
  ranges — duplicate work the streaming solve already does. The same
  four `(abs_min, abs_max)` tuples (`matrix`, `cost`, `col_bound`,
  `row_bound`) are now populated automatically on every `solve()` and
  exposed as `Solution.streamed_lp_ranges`. Callers that needed range
  inspection should read from the `Solution` instead; there is no more
  pre-solve range-inspection API.

### Added

- `Problem(auto_user_bound_scale: bool = False)` constructor option.
  When `True`, the streaming solve accumulates LP coefficient ranges
  during the family loop (at no extra allocation cost — it walks the
  per-family arrays we already build) and applies a `user_bound_scale`
  recommendation via `setOptionValue` before `Highs.run()`, but only
  when the caller has not already set `user_bound_scale` via the
  options dict / `set_solver_options`. The embedded heuristic
  `_recommend_user_bound_scale(bound_range, rhs_range)` is a direct
  port of HiGHS' own `suggestScaling` lambda at
  `HighsSolve.cpp:570-607`: it pulls `max(bound_max, rhs_max)` into
  HiGHS' `[kExcessivelySmallBoundValue, kExcessivelyLargeBoundValue]`
  = `[1e-4, 1e+6]` comfort zone using outer-rounded log2, and
  reproduces the integer HiGHS prints in its `"Consider setting the
  user_bound_scale option to <N>"` recommendation byte-for-byte.
- `Solution.streamed_lp_ranges: dict | None` field. Populated by every
  solve that flows through `_solve_streaming` (which is the default
  path) with the four `(abs_min, abs_max) | None` range tuples. `None`
  on solves that don't go through streaming (e.g. the non-streaming
  `solve(streaming=False)` path).

### Changed

- `_solve_streaming` now performs running min/max accumulation over
  `col_obj_h`, `col_lb_h`/`col_ub_h`, and per-family `val64` /
  `row_lb` / `row_ub` numpy arrays. Cost is a handful of O(n) scans
  with no new allocations. Used to drive `auto_user_bound_scale` and
  exposed on `Solution.streamed_lp_ranges`.
- When `auto_user_bound_scale=True`, the decision is now reported on
  stdout so the run log shows what scaling (if any) was applied —
  one of: `applying user_bound_scale=N (bound …, rhs …; HiGHS' own
  kExcessively[Small|Large]BoundValue formula)`, `no scaling --
  max(bound, rhs) already within HiGHS' [1e-4, 1e+6] comfort zone
  (bound …, rhs …)`, `no scaling -- no finite bound or RHS entries
  to evaluate`, or `caller override in place (user_bound_scale=N)`.

## [1.3.0] — 2026-05-22

### Added

- Generic Enum-dtype alignment on every internal join site. When two
  frames are joined on a column that is `pl.Enum` on both sides but
  with different categorical vocabularies, polar-high now up-casts
  the narrower side to the wider Enum (provided one's categories are
  a subset of the other's). Enum-vs-`pl.Utf8` mismatches are
  resolved by casting the string side to the Enum dtype. Two Enums
  with neither-subset vocabs raise a clear `ValueError` pointing the
  caller to cast to `pl.Utf8` or build a union Enum. The behaviour
  is exposed as the internal helper `polar_high.engine._align_enum_join_keys`
  and exercised by every internal `.join` call site (operator joins,
  `Where`, `Sum`, `Lag`, constraint-emission, `WarmProblem` updates).
- `tests/test_enum_dtype_align.py`: unit + end-to-end coverage of the
  new alignment behaviour, including the disjoint-vocab raise path
  and an end-to-end `Problem.add_cstr` / `solve` with a narrower-vocab
  rhs Param.

### Changed

- README "Enum dtype handling" subsection documenting the
  subset-up-cast rule and the raise-for-no-subset behaviour. No DSL
  surface change — existing models keep building unchanged; mixed-vocab
  models that previously needed per-site casts in caller code no
  longer do.
- `engine.py`: when a constraint's rhs is a `Param` (or a chain of
  `Param * Param * ...`), pre-filter the rhs lazy plan with a semi-join
  against `row_index`'s join keys and collect via the streaming engine
  before the left-join into the constraint frame. Polars' optimiser
  doesn't always propagate the implicit row-set restriction through a
  multi-way Param product, so the intermediate buffers could blow up
  by orders of magnitude relative to the final row count. On
  FlexTool's South Africa 1-week PES-Hydro-dispatch case (a
  `p_profile_value * p_process_existing_count * p_process_availability`
  product), solver-finished ΔRSS drops from +28.77 GB to +9.40 GB
  (-67%) and the section runtime drops from 57.7 s to 17.5 s.
  Objective and total cost match the baseline byte-for-byte. Applied
  to all three rhs-Param call sites: the non-streaming
  `Problem.add_cstr` path, `_solve_streaming`, and `WarmProblem.solve`.
  Falls back to `collect(streaming=True)` on polars < 1.x.
- README: quickstart code is now inlined (GitHub/PyPI don't render
  `pymdownx.snippets` includes); the cross-product index is split into
  reusable `unit_index` / `time_index` sets; `cap` is built per-unit
  then concatenated; `v_idx` renamed to `composite_index` (the `v_`
  prefix is reserved for variables); `_idx` → `_index` throughout.
- `Problem.add_cstr` arg order in README and quickstart fixture
  reordered to `lhs_terms` before `sense` — reads more naturally as
  *lhs sense rhs*. No API change (these are keyword args).

### Removed

- **Breaking:** `Problem.peek_lp_ranges()` removed. The method rebuilt
  the full LP into numpy arrays via the non-streaming path purely to
  extract coefficient ranges — duplicate work the streaming solve
  already does. Stream-time range accumulation now populates
  `Solution.streamed_lp_ranges` with the same four `(abs_min, abs_max)`
  tuples (`matrix`, `cost`, `col_bound`, `row_bound`) at zero extra
  cost on every `solve()` that goes through `_solve_streaming` (the
  default). Callers that previously relied on `peek_lp_ranges()` for
  diagnostics should read `sol.streamed_lp_ranges` after `solve()`
  returns; the module helper `polar_high.engine._recommend_user_bound_scale`
  consumes the `(lo, hi)` of the `col_bound` entry for the
  geo-midpoint heuristic. The `top_k > 0` per-coefficient name-lookup
  variant of `peek_lp_ranges` has no streaming-time replacement; if
  needed, build the LP via the non-streaming path
  (`solve(streaming=False)`) and inspect via the solver-specific HiGHS
  diagnostics.

## [1.2.0] — 2026-05-12

### Added

- `polar_high.solvers` module: multi-solver dispatch behind a single
  `solve(problem, solver_name=..., io_api=..., env=..., **options)`
  entry point. HiGHS remains the default; **Gurobi**, **CPLEX**,
  **FICO Xpress**, and **COPT** are supported on a
  bring-your-own-license basis (we ship no binaries and no licenses).
- `polar_high.solvers.available_solvers`: runtime registry of
  installed solver Python wrappers, populated at import time. Tells
  you which wrappers are *installed*; license checks fire inside the
  adapter.
- `IOMode.MPS` file-based fallback for users with a solver's CLI
  binary on `PATH` but no matching Python wrapper. Writes a temp MPS
  via `highspy`, invokes the CLI, parses the resulting `.sol` file.
  Covers `gurobi_cl`, `cplex`, Xpress `optimizer`, and `copt_cmd`.
- `polar_high.solvers._lp_view.LpView`: frozen, solver-agnostic
  extraction surface that every adapter consumes. Engine-private
  attribute access (`Problem._build_lp_arrays` etc.) is confined to
  this single module.
- Optional install extras: `polar-high[gurobi]`, `polar-high[cplex]`,
  `polar-high[xpress]`, `polar-high[copt]`. Each pulls only the
  vendor's Python wrapper (plus `scipy` where vectorized loads need
  it).
- `docs/guide/solvers.md`: user-facing guide covering detection,
  per-solver install, the `io_api='mps'` escape hatch, the `env=`
  passthrough (Gurobi WLS example), and license troubleshooting.

### Changed

- `Problem.solve(streaming=False)` now routes through
  `polar_high.solvers._highs.run`. Behaviour and return type
  unchanged — `streaming=True` retains the existing HiGHS-only
  per-family `addRows` path.
- COPT adapter auto-routes through the `copt_cmd` CLI fallback
  whenever `highspy` is already loaded in the interpreter. COPT
  8.x's native core conflicts with HiGHS in-process (`Highs.run()`
  segfaults once `coptpy` is imported); the auto-route keeps both
  solvers usable from the same `polar-high` venv at the cost of a
  per-solve MPS write + subprocess invocation. Requires `copt_cmd`
  on PATH (not shipped by the `coptpy` pip wheel); a clean
  `SolverNotAvailableError` is raised when it is missing. Details
  in `docs/guide/solvers.md`.

## [1.1.4] — 2026-05-11

### Added

- `Problem.peek_lp_ranges()`: build the LP into numpy arrays and
  return the abs-value ranges of finite non-zero entries on each
  axis (matrix, cost, bounds, rhs) — same numbers HiGHS prints in
  its "Coefficient ranges" diagnostic, but available *before*
  `passModel()` runs. Optional `top_k` returns the worst offenders
  per axis as `(abs_value, col_name, row_name_or_side)` triples.
  Lets callers pick `user_bound_scale` / `user_cost_scale` or
  refuse to solve a catastrophically scaled LP without paying for
  a full solve. Uses `np.argpartition` so the cost is
  `O(n_nonzeros)`.
- `.github/dependabot.yml`: weekly dependency PRs for GitHub
  Actions and Python (pip) ecosystems. The initial commit
  (c3836f5) was the GitHub-provided template with an empty
  `package-ecosystem`; this release fills it in so the bot
  actually opens PRs.

### Changed

- `engine.py`: factor the non-streaming LP-build out of `solve()`
  into a private `_build_lp_arrays()` helper. `solve()` and
  `peek_lp_ranges()` now share the same arrays — diagnostics are
  byte-for-byte what HiGHS sees.
- `engine.py`: for constraint families with > 50 000 rows,
  collect term plans one at a time instead of `pl.collect_all`.
  Peak memory drops from `O(n_terms × frame)` to `O(frame)`,
  preventing stalls under memory pressure on large network
  models.
- `engine.py`: HiGHS no longer suppressed via `h.silent()` —
  solver progress and the "Coefficient ranges" line now print to
  stdout by default. Pass `options={"output_flag": False}` to
  silence.

## [1.1.3] — 2026-05-07

### Changed

- `docs/guide/debugging.md`: expanded with worked examples; doc
  snippets are now wired to test fixtures
  (`tests/fixtures/debug_example.py`,
  `tests/fixtures/lagrangian_example.py`,
  `tests/fixtures/quickstart_example.py`) so they're exercised by
  the test suite and can't silently rot.
- `mkdocs.yml`: drop `dedent_sections` from the `snippets`
  pymdownx config — incompatible with the multi-fixture snippet
  layout.

## [1.1.2] — 2026-05-05

### Added

- `docs/guide/loading-data.md`: new guide page on going from CSV /
  parquet / database tables to `Param` and `Var`, including the
  long-format vs. wide-format trade-off and how column names become
  dimension names.

### Changed

- `docs.yml`: drop the `dev` alias deploy on `main` pushes; only
  tagged releases publish a versioned doc site.

## [1.1.1] — 2026-05-05

### Fixed

- `pyproject.toml`: add Python 3.13 classifier. CI's test matrix
  already covers 3.13; the classifier was missing so the
  pyversions badge was reading "3.11 | 3.12" only.
- `release.yml`: `skip-existing: true` on the PyPI publish step.
  Re-tagging the same version now no-ops on PyPI's duplicate-file
  rejection instead of showing the run as failed.

## [1.1.0] — 2026-05-05

### Changed

- **BREAKING:** renamed package `polar-high-opt` → `polar-high`
  (Python module `polar_high_opt` → `polar_high`). All imports,
  PyPI install name, repo and docs URLs move with it.
- **BREAKING:** `Problem.solve()` defaults changed:
  `streaming=True` (per-family `addRows` instead of one big
  `passModel`; lower peak memory; numerically identical) and
  `keep_solver=False` (the live `highspy.Highs` is dropped after
  primal/dual extraction; pass `keep_solver=True` to retain it for
  post-solve inspection like `sol.highs.writeModel(...)`).
- **BREAKING:** `polar_high` sets `POLARS_MAX_THREADS=1` at import.
  Rayon coordination overhead exceeds the parallel speedup on
  typical LP-build workloads (see benchmark page). Override by
  setting the env var before `import polar_high`.
- COO row/column indices use `int32` when `nnz < 2^31`, falling
  back to `int64` only when needed. Cuts working-set memory in
  the matrix-assembly phase.
- `_Term.frame` cache is no longer populated during `Problem.solve()`
  — the lazy plan is collected into a local that goes out of scope
  per family. Re-solves rebuild from the lazy plan as before.

### Added

- Benchmark suite under `benchmark/`: dense `N×N` LP (replicates
  linopy's benchmark) and a sparse network-flow LP with irregular
  edge→node topology. Reproducible via subprocess-isolated cells in
  `benchmark/run.py`; figures rendered by `benchmark/plot.py`.
- New `docs/compare/benchmark.md` with five figures and the story
  for each (build-only headline, threads scaling at fixed N,
  threading benefit on the network LP, network LP, linopy-format
  replication).
- `Threading` section in `docs/guide/performance.md` documenting the
  default-1 choice and the override pattern.
- Tiny dispatch LP (wind + coal × 3 hours) replaces the abstract
  `i / j` placeholder in README and `docs/quickstart.md`.

## [1.0.1] — 2026-05-05

### Added

- GitHub Actions: tests on push/PR (Python 3.11–3.13), docs deploy
  on main + tag (mike), PyPI release on tag (trusted publishing).
- Ruff lint + format configured in `pyproject.toml`; `[lint]`
  optional-dependency added.
- README badges: PyPI version, Python versions, license, tests CI,
  docs CI, ruff.

### Changed

- Repo / docs URLs moved from `jkiviluo/polar-high` to
  `nodal-tools/polar-high`; documentation site is hosted at
  `https://nodal-tools.fi/polar-high/`.
- One-time `ruff format` reflow across the source tree.

### Fixed

- Dead intra-doc anchor link in `guide/performance.md` (the
  `vars-and-params.md` "Param × Param" heading slugifies to a single
  hyphen, not two).

### Removed

- Two dangling unused locals (`engine.py` and `test_warm_problem.py`).

## [1.0.0] — 2026-05-05

First public release.

### Added

- `Var`, `Param`, `Expr` — building blocks for indexed expressions
  expressed as polars DataFrames.
- `Sum`, `Where`, `Lag` — aggregation, filtering, and time-shift
  primitives that compile to LP rows efficiently.
- `Problem` — assemble an LP/MIP and solve via HiGHS (`highspy`).
- `WarmProblem` — re-solve with parameter / RHS / objective updates
  while preserving the basis.
- `LagrangianProblem` — generic dual-subgradient driver for
  Lagrangian decomposition of coupled subproblems.
- `Solution` — primal values, constraint duals, reduced costs, and a
  live `highspy.Highs` handle for advanced post-solve inspection.
- MkDocs + mike documentation site under `docs/`.

<!--changelog-end-->
