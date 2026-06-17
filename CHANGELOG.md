# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!--changelog-start-->

## [2.9.0] — 2026-06-17

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
  different `max_workers` are byte-identical. The initial build loop always
  runs sequentially on the calling thread (it may reset the global HiGHS
  scheduler). The executor is shut down on every exit path, including the
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
