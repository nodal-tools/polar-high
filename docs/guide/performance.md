# Performance

A short list of things that move the needle on real models.

## Threading

`polar_high` defaults `POLARS_MAX_THREADS=1` at import time. On the
indexed-LP build patterns the engine drives, polars's Rayon
coordination overhead consistently exceeds the parallel speedup —
single-thread is faster *and* leaner. The benchmark page has the
numbers; the short version is "polars threading helps the matrix
assembly by maybe 10–15 % at large N, while costing per-thread
scratch memory at every N".

To override:

```bash
# before launching python
export POLARS_MAX_THREADS=8
```

```python
# or programmatically, BEFORE importing polar_high
import os
os.environ["POLARS_MAX_THREADS"] = "8"

import polar_high  # picks up your override; setdefault no-ops
```

The env var must be set before *any* import of polars (or anything
that imports polars) — polars reads it once at module load. If
you've already imported polars elsewhere in your process, the
default is locked in.

## Solver options dominate

For LPs of meaningful size, **HiGHS options matter more than
build-side micro-optimizations**. Two switches that have moved
end-to-end runtime by 2–3× on real models:

```python
p.set_solver_options({"presolve": "off"})       # for warm chains
p.set_solver_options({"solver": "simplex"})     # for LPs with simplex-friendly structure
```

- Default `presolve=on` is great for cold solves but discards basis
  information. If you are warm-starting via [`WarmProblem`](warm-starting.md),
  turning presolve off is usually a win.
- `solver=ipm` (interior point) can be much faster on large LPs
  without warm starts, but provides no basis to re-use.

Always benchmark your specific model — these are starting points,
not rules.

## Build path

The kernel goes index frames → polars joins → coordinate-format
(COO) triples → compressed-sparse-column (CSC) → HiGHS `passModel`.
The hot loops are:

1. **Constraint loop** (`Problem.solve` walks `_cstrs`, joining row
   indices with each LHS term, building COO triples).
2. **COO → CSC** (`numpy` lexsort + cumsum + populating the
   `HighsLp` struct).
3. **`passModel`** (a single C call into HiGHS).

For models in the 10⁴–10⁵ row range, the constraint loop dominates;
beyond that, HiGHS run time dominates and build cost is noise.

## `dense_axes`: block-COO arm

For a model whose Vars share a common pre-sorted trailing axis (think
`t` for time, or `(d, t)` for day-of-year × hour), the kernel ships
an opt-in evaluation path that **slices the dense suffix of each
Var's frame as a contiguous numpy view and multiplies in ufuncs**,
skipping the polars join entirely on the LHS of `Param * Var` (and
`Sum`-wrapped chains). Coefficient bytes never have to leave their
Arrow block. On a dense Sum-heavy family this typically halves the
build-side wall time and trims a few GB of peak.

You opt in once, on the `Problem` constructor:

```python
p = Problem(dense_axes=("d", "t"))
```

This is **a binding promise about every frame you pass that contains
those columns**: it must be lexicographically sorted by
`(other_dims_in_declared_order…, *dense_axes)`. In other words, the
declared dense axes are the trailing sort keys and the leading dims
form a sorted prefix. polar-high verifies this cheaply (a single-pass
monotonic scan, no re-sort) on every Var the block-COO arm fires on,
and raises a clear `ValueError` naming the Var on violation. Frames
that don't carry the dense axes (e.g. an investment Var indexed by
`("p", "d")` when `dense_axes=("d", "t")`) simply do not fire the
arm — there's no penalty for mixing.

In practice the sort usually comes for free: index frames built from
`pl.DataFrame({"d": np.repeat(...), "t": np.tile(...)})` patterns,
or a `cross`-join of `entity × timesteps`, are already in the
required order. If you build a frame in some other order, sort it
once at construction (`.sort(["entity", "d", "t"])`) before passing
it in.

Two practical knobs:

* **What you declare**: the `dense_axes` tuple is **strictly
  suffix-matched** against each Var's dims. A Var with dims
  `("e", "t")` matches `dense_axes=("t",)`; a Var with dims
  `("p", "d", "t")` matches both `("t",)` and `("d", "t")`. Pick the
  longest dense suffix your problem shares — more dense axes ⇒ more
  ufunc-friendly blocks ⇒ bigger win.
* **A/B rollback**: `POLAR_HIGH_DISABLE_BLOCK_COO=1` falls every term
  back to the polars path with no other change — useful when
  bisecting a numerical regression to confirm the arm isn't involved.

The benchmark page covers the dense and network LP cells under both
the block-COO and the legacy polars-join paths.

## polars patterns that help

- **Use `LazyFrame`s** for derived Params. The kernel internally
  stores Params lazily; if you compute composite Params with
  `Param * Param / Param`, that chain stays lazy until consumption.
- **Pre-build index frames once**, reuse them for every constraint
  that shares an index. Building the same `(n, d, t)` frame 30 times
  is cheap individually but adds up.
- **Densify Params at the boundary**, not inside the kernel. Inner-
  joins drop missing cells (see [warning](../concepts/vars-and-params.md#param-param)),
  so if you need zero-fill, do the `left_join`/`fill_null(0)` once
  before constructing the Param.

## Writing MPS without HiGHS

For very large LPs the canonical "write the model to disk and shell
out to another solver" pattern goes through HiGHS' own
`Highs.writeModel(path)` — which serialises the in-memory LP into MPS
form. On models in the ~10 M-row / 5 M-col / 20 M-nz range, that
serialiser allocates **~20×** the steady-state LP footprint in
transient buffers (~45 GB on the LP just cited), and a workstation
that comfortably runs the actual solve out-of-process can OOM on the
MPS write step alone.

`Problem.write_mps` is a direct polars→MPS writer that never
constructs a `highspy.Highs` instance. It walks the same per-family
streaming pattern the kernel uses internally, then performs one
streaming sort by `(col_id, row_id)` and emits the COLUMNS section
in chunks. Peak transient memory is **~2–3 GB** on the LP above:

```python
p = Problem()
# … add vars, params, objective, constraints …

p.write_mps("model.mps")                 # default: free format, named
p.write_mps("model.mps", emit_names=False)   # smaller file, no names
p.write_mps("model.mps", release=True)   # drops polar-side LP source
```

`release=True` calls the same internal teardown as
`Problem.solve(save_memory=True)`: every `_Term.lazy` plan and
constraint RHS reference is dropped before `write_mps` returns. The
per-`Var` `col_id` frames stay alive so a caller can later reconstruct
a `Solution` from externally-produced `col_value` / `row_dual` arrays
(via `Solution(..., vars=dict(problem._vars), ...)`). After
`release=True` the `Problem` is in `_released` state and further
`solve()` calls raise.

Intended use: drive a subprocess HiGHS (or any external solver) from
the written MPS file when the parent process must stay under a peak
RSS ceiling. The MPS is spec-compliant free-format and reads back
into HiGHS, Gurobi, CPLEX, and Xpress in the wrapper-driven roundtrip
tests under `tests/test_mps_fallback_wrapper.py`.

For diagnosing memory hot spots in `write_mps` itself, set
`POLAR_HIGH_WRITE_MPS_PROFILE=1`: per-phase and per-constraint-family
RSS deltas are emitted to stderr. Zero overhead when unset.

`Problem.solve(save_memory=True)` also writes a spill MPS during its
disk-roundtrip (item 2 above). It lands in the system temp dir
(`$TMPDIR` / `/tmp`) by default; pass `tmp_dir=` to redirect it to a
specific volume (e.g. the same filesystem as the workspace, or a
per-job scratch directory). The argument is ignored when
`save_memory=False`.

## Profiling

`Problem.solve()` is straightforward to profile with `cProfile` or
`py-spy`. The two phases worth timing separately are:

- *build* — everything before `h.run()`;
- *run* — HiGHS itself.

If HiGHS dominates, your time is best spent on solver options. If
build dominates, look at the constraint loop and at any Param chains
that are accidentally re-collected on each access.

## Releasing memory between solves

In a long-running rolling-horizon or parameter-sweep loop, intermediate
polars frames can stick around longer than you expect. The kernel does
not call `gc.collect()` for you — that would pay a full mark-and-sweep
without freeing the polars/Arrow buffers, which are released when their
owning Python references drop. The right pattern is at the call site:

```python
for window in windows:
    p = build_problem(window)
    sol = p.solve()
    write_outputs(sol)
    del p, sol      # drop refs to LP, intermediates, Solution
    gc.collect()    # optional; only useful if reference cycles linger
```

`del`-then-`gc.collect()` is most useful when you've observed RSS
growing across iterations; in plain loops the refcount drop on
re-binding is usually enough.
