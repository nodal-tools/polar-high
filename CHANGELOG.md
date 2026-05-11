# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!--changelog-start-->

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
