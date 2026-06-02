# Benchmark

A reproducible build / solve / memory comparison between
**polar-high**, **linopy**, **Pyomo**, and **PuLP** on the same indexed
linear program, all solving with **HiGHS**. The model and structure
follow [linopy's benchmark](https://linopy.readthedocs.io/en/latest/benchmark.html),
restricted to the linear case so polar-high can run it directly.

## The model

```
min  Σ_{i,j} (2·x[i,j] + y[i,j])
s.t. x[i,j] - y[i,j] >= i          for i,j ∈ {1,…,N}
     x[i,j] + y[i,j] >= 0          for i,j ∈ {1,…,N}
     x, y >= 0
```

Closed-form optimum: `obj = N · Σ_{i=1..N} 2i = N · N · (N+1)`. Used
as a sanity check that all four tools agree on the answer.

## What is measured

For each (tool, N) cell, in a fresh Python subprocess:

- **build_s** — wall-clock time for the model-construction code in
  the tool (everything before solver invocation).
- **solve_s** — wall-clock time for the solver call. Same HiGHS
  underneath all three tools.

Memory metrics (all MB, Linux):

- **peak_rss_mb** — `ru_maxrss` / 1024, the unavoidable high-water
  mark across the whole process. Right column for "how much RAM
  does the machine need."
- **rss_after_build_mb / rss_after_solve_mb** — point-in-time RSS
  right after `gc.collect()` at the obvious checkpoints.
- **rss_after_build_trim_mb / rss_after_solve_trim_mb** — same
  checkpoints but after `malloc_trim(0)` forces glibc to return
  freed-but-cached arenas to the OS. Diagnostic for how much
  apparent "memory still used" is glibc retention.
- **rss_solve_{min,p50,p95,max}_mb / n_samples** — a sidecar
  thread polls VmRSS at 25 ms cadence while `solve()` runs.
  `max` should align with `peak_rss_mb`; `p50` is the steady-state
  working set after transient setup spikes wash out.

Each cell runs `--repeats` times (default 3); the plot uses the
median across reps of `peak_rss_mb`. See the
[memory-measurement section in the comparison page](../docs/compare/benchmark.md#measuring-memory)
for how to read each column.

## Running it

```bash
# From the repo root (installs benchmark deps for this invocation):
uv run --with-requirements benchmark/requirements.txt python benchmark/run.py
uv run --with-requirements benchmark/requirements.txt \
  python benchmark/run.py --sizes 10 30 100 --repeats 1   # smoke run
uv run --with matplotlib python benchmark/plot.py         # writes docs/assets/benchmark.png
```

Single cell: `uv run --with-requirements benchmark/requirements.txt python benchmark/run_one.py pulp 10`

Works on Linux and Windows (`run_one.py` uses `/proc` + cgroup metrics on Linux,
`psutil` on Windows).

CSV output goes to `benchmark/results/results.csv` (gitignored).

## Tools and entry points

| Tool | Module | Solver path |
|---|---|---|
| polar-high | `benchmark/models/polar.py` | `Problem.solve()` → `highspy` |
| linopy | `benchmark/models/linopy.py` | `Model.solve(solver_name="highs")` |
| Pyomo | `benchmark/models/pyomo.py` | `pyomo.contrib.appsi.solvers.Highs` (persistent) |
| PuLP | `benchmark/models/pulp.py` | `HiGHS_CMD` (LP file + `highs` subprocess) |

The Pyomo path uses **appsi_highs** (persistent in-process), which
is the fastest Pyomo→HiGHS path. Falling back to the file-based
`SolverFactory("highs")` interface would inflate Pyomo's solve_s
unfairly via the LP-write-and-reread overhead.

PuLP uses **HiGHS_CMD**, the file-based path analogous to linopy's
``io_api="lp"``. The `highs` executable must be on ``PATH`` (or
install the system/conda HiGHS package alongside `highspy`).

## Fairness caveats

- All four build their own version of the same LP; whitespace-level
  identity isn't enforced. The closed-form objective check provides
  a cross-tool correctness anchor.
- Subprocess isolation prevents memory carry-over but adds ~50 ms
  fixed overhead per cell (irrelevant at the sizes we care about).
- The `polar-high` and `linopy` builds defer most work into
  vectorised numpy/polars/xarray ops; the `Pyomo` and `PuLP` builds
  loop in Python over each (i, j). This is *the* difference the
  benchmark is designed to surface.
- Solve time is included as the dominant component at large `N` and
  to confirm the three tools are sending equivalent LPs to HiGHS.

## Hardware

Whatever machine the runner is on — record CPU/RAM/Python version
in your published results. The CI we ship doesn't run the benchmark
(too sensitive to runner variability); it's a manual artefact.
