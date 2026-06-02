# Environment variables

A reference for every environment variable `polar_high` reads at
runtime. All variables are optional — the defaults give zero-overhead
behaviour and match what `Problem.solve()` / `Problem.write_mps()` do
out of the box.

You'd set one of these to:

- **Profile / instrument.** Print per-phase RSS markers when you are
  diagnosing a memory hot spot during MPS export or autoscale range
  detection.
- **Opt out of a build optimisation.** Fall back to a pre-optimisation
  code path if you suspect it of a numerical regression.
- **Override polars threading.**

For runtime tuning on the polar-high side that is *not* an env var
(solver options, `release=True`), see [Performance](performance.md).

## Profiling / instrumentation

All three profiling switches print to stderr only when set; when
unset they short-circuit before any `psutil` call, so the overhead is
exactly zero.

| Variable | Default | Set to | What it does |
|---|---|---|---|
| `POLAR_HIGH_WRITE_MPS_PROFILE` | unset (off) | `1` | Print per-phase and per-constraint-family RSS deltas during `Problem.write_mps`. Use when an MPS write OOMs or balloons unexpectedly. Also covers `Problem.canonicalise` — when that runs inside `Problem.solve` / `WarmProblem._initial_build`, set this to see per-family checkpoints from the canonical matrix build. |
| `POLAR_HIGH_RANGES_PROFILE` | unset (off) | `1` | Print per-family and per-term RSS during autoscale range detection (`autoscale._ranges`). Use when range detection on a large LP shows a peak you cannot localise. |
| `POLAR_HIGH_SOLVE_PROFILE` | unset (off) | `1` | Print per-phase RSS markers from `Problem.solve` (`_solve_streaming` cold path, ~27 checkpoints) and from `WarmProblem._initial_build` (~18 checkpoints covering the per-family LP-build loop and the HiGHS handoff). Combine with `POLAR_HIGH_WRITE_MPS_PROFILE=1` to also see the per-family canonicalise checkpoints; the two streams interleave by wall time. |

Example — diagnose `write_mps` memory:

```bash
POLAR_HIGH_WRITE_MPS_PROFILE=1 python -m my_model.build_and_export 2> mps_profile.log
```

The output is one line per profiling checkpoint with the current RSS
and the delta since the previous checkpoint. Grep for the largest
deltas to find the phase that allocated.

## Safety / opt-out

| Variable | Default | Set to | What it does |
|---|---|---|---|
| `POLAR_HIGH_DISABLE_PRUNE_DOWN` | unset (off — prune-down active) | `1` | Disable the per-atomic prune-down for `Param` chains in `_build_canonical_matrix` (RHS), `_build_lhs_pruned_plan` (LHS), `_solve_streaming` and `WarmProblem._initial_build`. With the var set, every multi-atomic chain falls through to the original merged-lazy semi-join path. Use as a fallback if a future model surfaces a numerical drift on the prune-down path; report the affected scenario so the engine can be fixed and the env var retired. Numerics are identical between the two paths for all currently-tested scenarios. |
| `POLAR_HIGH_DISABLE_WHERE_PUSHDOWN` | unset (off — pushdown active) | `1` | Disable the pure-filter `Where(expr, frame)` deferral added in v2.3.0. With the var set, every `Where` call eagerly inner-joins `frame` into `t.lazy` and clears the leaf metadata (`var_source`, `coef_scalar`, `where_frames`) exactly as the pre-v2.3.0 path. Use as a fallback if a model surfaces a numerical drift on the pushdown path; the LP matrix is byte-identical between the two paths for all tested scenarios. |

## Polars threading

| Variable | Default | Set to | What it does |
|---|---|---|---|
| `POLARS_MAX_THREADS` | `1` (set by `polar_high` at import) | any positive integer | The number of threads polars uses across the whole process. `polar_high` calls `os.environ.setdefault` at import time, so setting the variable yourself before importing wins. |

This is a polars variable, not a `polar_high` variable, but `polar_high`
defaults it to `1` for the reasons in
[Performance — Threading](performance.md#threading). To override, set
it before `import polar_high` (or before any other module that imports
polars). The `setdefault` no-ops once a value is already present.

## See also

- [Performance](performance.md) — `release=True`, solver options, and
  the underlying build-path reasoning behind the profiling vars above.
- [FlexTool](https://github.com/irena-flextool/flextool) documents its
  own `FLEXTOOL_*` variables (the env-vars page is still on a
  not-yet-default branch, so it isn't published at a stable URL yet).
  `FLEXTOOL_SAVE_MEMORY=1` enables a subprocess solver path that
  benefits from `POLAR_HIGH_WRITE_MPS_PROFILE` when diagnosing the MPS
  write step in that path.
