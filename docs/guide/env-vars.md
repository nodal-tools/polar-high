# Environment variables

A reference for every environment variable `polar_high` reads at
runtime. All variables are optional — the defaults give zero-overhead
behaviour and match what `Problem.solve()` / `Problem.write_mps()` do
out of the box.

Two reasons to set any of them:

- **Profiling / instrumentation.** Print per-phase RSS markers when
  you are diagnosing a memory hot spot during MPS export or autoscale
  range detection.
- **Workload tuning.** Cap the work the autoscaler does on very wide
  constraint families.

For runtime tuning on the polar-high side that is *not* an env var
(threading, solver options, `release=True`), see
[Performance](performance.md).

## Profiling / instrumentation

All three profiling switches print to stderr only when set; when
unset they short-circuit before any `psutil` call, so the overhead is
exactly zero.

| Variable | Default | Set to | What it does |
|---|---|---|---|
| `POLAR_HIGH_WRITE_MPS_PROFILE` | unset (off) | `1` | Print per-phase and per-constraint-family RSS deltas during `Problem.write_mps`. Use when an MPS write OOMs or balloons unexpectedly. |
| `POLAR_HIGH_RANGES_PROFILE` | unset (off) | `1` | Print per-family and per-term RSS during autoscale range detection (`autoscale._ranges`). Use when range detection on a large LP shows a peak you cannot localise. |

Example — diagnose `write_mps` memory:

```bash
POLAR_HIGH_WRITE_MPS_PROFILE=1 python -m my_model.build_and_export 2> mps_profile.log
```

The output is one line per profiling checkpoint with the current RSS
and the delta since the previous checkpoint. Grep for the largest
deltas to find the phase that allocated.

## Workload tuning

| Variable | Default | Set to | What it does |
|---|---|---|---|
| `POLAR_HIGH_RANGES_MAX_FAMILY_ROWS` | `1000000` | positive integer; `0` disables the cap | Family-size threshold for the autoscale range readout. Families larger than this contribute their min/max via streaming but skip the full readout. `0` forces the full readout on every family regardless of size. |

This is the only knob worth changing on real workloads. On models
where autoscale range detection itself dominates the build budget,
lowering it (e.g. `200000`) trades a small amount of range-detection
accuracy on the largest families for a noticeable wall-time win.
Raising it (or setting `0`) is only useful when debugging a
range-detection regression.

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
- FlexTool's
  [Environment variables](https://irena-flextool.github.io/flextool/dev/env_vars/)
  page documents the `FLEXTOOL_*` variables. `FLEXTOOL_SAVE_MEMORY=1`
  enables a subprocess solver path that benefits from
  `POLAR_HIGH_WRITE_MPS_PROFILE` when diagnosing the MPS write step
  in that path.
