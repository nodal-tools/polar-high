# Quickstart

A 2-set transportation LP — minimum-cost flow from sources `i` to
sinks `j`, with a per-source capacity.

```python
import polars as pl
from polar_high_opt import Problem, Param, Sum

p = Problem()

# Decision variable v[i, j] >= 0
v = p.add_var(
    "v", dims=("i", "j"),
    index=pl.DataFrame({"i": [1, 1, 2], "j": ["a", "b", "a"]}),
    lower=0.0,
)

# Cost parameter c[i, j]
c = Param(
    ("i", "j"),
    pl.DataFrame({"i": [1, 1, 2],
                  "j": ["a", "b", "a"],
                  "value": [3.0, 1.0, 2.0]}),
)

# Objective: min Σ_{i,j} c[i,j] * v[i,j]
p.set_objective(c * v, sense="min")

# Capacity: Σ_j v[i,j] <= 5 for each i
p.add_cstr(
    "cap",
    over=("i",),
    sense="<=",
    lhs_terms={"flow": Sum(v, over=("j",))},
    rhs_terms={"cap": 5.0},
)

# Demand: Σ_i v[i,j] >= 4 for each j
p.add_cstr(
    "demand",
    over=("j",),
    sense=">=",
    lhs_terms={"flow": Sum(v, over=("i",))},
    rhs_terms={"req": 4.0},
)

sol = p.solve()
print(f"objective: {sol.obj}")
print(sol.value(v))
```

## What just happened

1. **Index frames**. `add_var(... index=...)` registers one column per
   row of the index frame. Variables are stored as polars frames with
   columns `(*dims, col_id)`.
2. **Parameters as frames**. A `Param` is a frame `(*dims, value)`.
   Multiplying `Param * Var` does an inner-join on shared dims and
   returns an `Expr` (term frame `(*dims, col_id, coef)`).
3. **Aggregation as group-by**. `Sum(v, over=("j",))` group-by-sums
   the term over `j`, leaving `i` as the constraint's row dim.
4. **Constraints**. `add_cstr(..., over=("i",), ...)` materializes one
   LP row per cell of the `over` frame.
5. **Solve**. `Problem.solve()` builds the coordinate-format
   coefficient triples, hands HiGHS a `HighsLp` struct, and returns
   a [`Solution`](reference/api.md).

## Next steps

- The mental model: [Concepts](concepts/index.md)
- Idiomatic patterns: `Sum`, `Where`, `Lag` —
  [Expressions](concepts/expressions.md)
- Re-solve with parameter updates: [Warm-starting](guide/warm-starting.md)
- Decompose coupled subproblems: [Lagrangian](guide/lagrangian.md)
