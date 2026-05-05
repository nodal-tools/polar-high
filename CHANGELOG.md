# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!--changelog-start-->

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
