# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!--changelog-start-->

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
  user_bound_scale option to <N>"` recommendation byte-for-byte (e.g.
  `N=-6` on the DES scenario, `N=-2` on the historical Rivendell-fix
  LP).
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
