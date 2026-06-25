# Decomposition building blocks

polar-high does not ship a decomposition *driver*. Instead it provides the
small, domain-agnostic primitives a cutting-plane / Benders scheme needs, and
leaves the master/subproblem orchestration to the caller (where the problem
structure lives). The pieces are:

- a **persistent warm master** you grow one cut at a time
  (`WarmProblem.add_cut_row` / `add_recourse_col`),
- a **warm re-solve after a cut append** that hot-starts off the retained
  basis (`WarmProblem.solve(retry_on_unknown=True)`),
- a **deterministic parallel solver** for the independent subproblems
  (`polar_high.parallel`).

!!! note "Replaces `LagrangianProblem`"
    Up to 2.x polar-high shipped a built-in `LagrangianProblem`
    dual-subgradient driver. It was removed in **3.0.0**: its one consumer
    moved to a Benders decomposition built from the primitives below, which
    give a tight bound without a subgradient tail. If you need the old
    driver, pin `polar-high<3`.

## Persistent warm master

Build and solve the master once, then append optimality cuts to the *live*
HiGHS handle instead of rebuilding each outer iteration:

```python
m = WarmProblem(...)          # epigraph var θ + first-stage vars
m.solve()                     # initial relaxation (no cuts yet)

# ... derive a Benders optimality cut from the subproblem duals ...
row_id = m.add_cut_row(col_ids, coefs, lower)   # Σ coefs·x >= lower
sol = m.solve(retry_on_unknown=True)            # warm re-optimise
cut_dual = sol.row_dual[row_id]                 # read the appended row BY id
```

- **`WarmProblem.add_cut_row(col_ids, coefs, lower) -> int`** appends a `>=`
  row to the built model and returns its `row_id`. It is a post-build live
  edit (same class as `update_rhs` / `update_coef`) and deliberately bypasses
  the build-time DSL lock, which only guards the fixed-size autoscale side
  vectors. There is **no auto-scaling for appended rows** — keep the master
  autoscale off, or pre-scale `coefs` so the cut lives on the built columns'
  scale. The appended row's dual is read by `Solution.row_dual[row_id]`, not
  `constraint_dual` (which only knows named build-time rows).
- **`WarmProblem.add_recourse_col(name, cost, lower, upper)`** appends a lazy
  recourse column for column-generation-style schemes; its value is read by
  `Solution.col_value[col_id]`.

## Warm re-solve after a cut

`WarmProblem.solve(*, retry_on_unknown=False)` defaults to today's behaviour
(byte-identical). With `retry_on_unknown=True` the solver re-runs **warm**
after `add_cut_row` — the retained basis lets dual simplex hot-start across
the appended `>=` row — and only falls back to a single `clearSolver` cold
re-presolve if the warm run fails to certify `kOptimal` (a `kUnknown`,
transient `kSolveError`, or a spurious `kUnbounded` off a stale basis). On a
well-scaled master the warm path holds with no fallback, which removes the
super-linear cold-presolve cost that otherwise dominates the master at scale.

## Solving subproblems in parallel

Each outer iteration solves N *independent* subproblems (each its own
`WarmProblem` / HiGHS handle). Because HiGHS' `run()` releases the GIL, a
thread pool gives a real wall-clock speedup. `polar_high.parallel` wraps the
thread-safety contract so your algorithm code never touches HiGHS threading:

```python
from polar_high import (
    prewarm_global_scheduler, resolve_worker_count, solve_indexed_parallel,
)

workers = resolve_worker_count(len(subproblems), workers=None)  # 0/None → auto
prewarm_global_scheduler(1)        # pin the process-global scheduler ONCE
sols = solve_indexed_parallel(subproblems, lambda i: subproblems[i].solve(...))
```

- **`prewarm_global_scheduler(threads=1) -> bool`** pins HiGHS' process-global
  task scheduler to a single thread once, up front, so the concurrent `run()`
  calls stay single-threaded (HiGHS is non-deterministic with `threads > 1`)
  and the box is not oversubscribed `workers × cores`.
- **`solve_indexed_parallel(warmproblems, fn)`** fans `fn(i)` across the pool
  and collects results into **per-index slots**, so the outcome is identical
  regardless of thread timing; worker exceptions re-raise in index order
  (lowest failing index wins, matching the sequential loop). It requires every
  `WarmProblem` to be **already built** (the cold first build calls the
  process-global `resetGlobalScheduler` and is unsafe to run concurrently) and
  raises if asked to fan out an unbuilt one — do the first build sequentially.
- **`resolve_worker_count(n_items, workers)`** clamps a request to
  `[1, n_items]`. `workers <= 1` keeps a fully sequential path (no pool, no
  prewarm) that is byte-identical to a plain `for` loop, so the parallel and
  sequential paths can be checked against each other in a determinism gate.

## See also

- `tests/test_warm_cut_append.py` — cut-append dual/value read-back.
- `tests/test_warm_restart_retry.py` — warm vs. cold-fallback parity.
- `tests/test_parallel_solve.py` — parallel vs. sequential determinism.
