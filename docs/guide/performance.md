# Performance

A short list of things that move the needle on real models.

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

## polars patterns that help

- **Use `LazyFrame`s** for derived Params. The kernel internally
  stores Params lazily; if you compute composite Params with
  `Param * Param / Param`, that chain stays lazy until consumption.
- **Pre-build index frames once**, reuse them for every constraint
  that shares an index. Building the same `(n, d, t)` frame 30 times
  is cheap individually but adds up.
- **Densify Params at the boundary**, not inside the kernel. Inner-
  joins drop missing cells (see [warning](../concepts/vars-and-params.md#param--param)),
  so if you need zero-fill, do the `left_join`/`fill_null(0)` once
  before constructing the Param.

## Profiling

`Problem.solve()` is straightforward to profile with `cProfile` or
`py-spy`. The two phases worth timing separately are:

- *build* — everything before `h.run()`;
- *run* — HiGHS itself.

If HiGHS dominates, your time is best spent on solver options. If
build dominates, look at the constraint loop and at any Param chains
that are accidentally re-collected on each access.
