# Duals and bases

polar-high exposes the full HiGHS post-solve surface, not just
the primal solution.

## Constraint duals

```python
sol = p.solve()
duals = sol.constraint_dual("balance")
# polars frame: (key, dual) — key is a comma-joined string of dim values
```

The frame's index columns mirror the `over=` frame you supplied to
`add_cstr`, so you can join duals against parameters or other
solution frames directly.

## Reduced costs (column duals)

```python
sol.col_dual              # numpy array sized to the LP's column count
```

To map this back to a specific variable's cells, join via
`var.frame["col_id"]`:

```python
import polars as pl
v_red = (v.frame
         .with_columns(reduced_cost=pl.Series(sol.col_dual[v.frame["col_id"]])))
```

## The live HiGHS handle

```python
sol.highs                  # highspy.Highs — kept alive across solves
sol.highs.getInfo()
sol.highs.getModelStatus()
sol.highs.getBasis()
sol.highs.writeModel("debug.mps")
```

Whatever `highspy` exposes is yours. This is also how
[`WarmProblem`](../guide/warm-starting.md) re-uses the basis between
solves: it holds onto the same `Highs` instance and feeds it
`changeRowBound` / `changeCoeff` / `changeObjectiveCoef` calls
between solves.

## When you need duals during build

Duals are only available after `solve()`. If you need them inside an
iterative scheme (e.g. column generation, Benders), see the
[decomposition building blocks](../guide/decomposition.md) — the
cut-append / warm-restart / parallel primitives wired to `WarmProblem`.
