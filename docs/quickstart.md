# Quickstart

A tiny dispatch LP. Two generation units (wind, coal) over three
hours. Wind is cheap but variable; coal is expensive but firm. Pick
how much each unit produces per hour to meet demand at minimum cost.

```python
import polars as pl
from polar_high import Problem, Param, Sum

p = Problem()

# Decision variable v_production[unit, hour] >= 0
v_idx = pl.DataFrame({
    "unit": ["wind", "wind", "wind", "coal", "coal", "coal"],
    "hour": [1, 2, 3, 1, 2, 3],
})
v_production = p.add_var(
    "v_production", dims=("unit", "hour"), index=v_idx, lower=0.0,
)

# Operating cost per unit
cost = Param(
    ("unit",),
    pl.DataFrame({"unit": ["wind", "coal"], "value": [2.0, 8.0]}),
)

# Available capacity per unit per hour (wind drops in hour 2)
cap = Param(
    ("unit", "hour"),
    pl.DataFrame({
        "unit":  ["wind", "wind", "wind", "coal", "coal", "coal"],
        "hour":  [1, 2, 3, 1, 2, 3],
        "value": [3.0, 1.0, 4.0, 10.0, 10.0, 10.0],
    }),
)

# Demand per hour
demand = Param(
    ("hour",),
    pl.DataFrame({"hour": [1, 2, 3], "value": [5.0, 6.0, 4.0]}),
)

# Minimise total cost
p.set_objective(cost * v_production, sense="min")

# v_production[unit, hour] <= cap[unit, hour]
p.add_cstr(
    "capacity",
    over=v_idx,
    sense="<=",
    lhs_terms={"production": v_production},
    rhs_terms={"cap": cap},
)

# Σ_unit v_production[unit, hour] == demand[hour]
hour_idx = v_idx.select("hour").unique().sort("hour")
p.add_cstr(
    "demand_balance",
    over=hour_idx,
    sense="==",
    lhs_terms={"production": Sum(v_production, over=("unit",))},
    rhs_terms={"demand": demand},
)

sol = p.solve()
print(f"objective: {sol.obj}")             # 72.0
print(sol.value("v_production"))            # frame: (unit, hour, value)
```

The optimum dispatches wind at full capacity in every hour (cheaper)
and uses coal to fill the gap. Hour 3 leaves coal idle because wind
alone covers demand.

## What just happened

1. **Index frames**. `add_var(... index=...)` registers one LP column
   per row of the index frame. Each Var is internally a polars frame
   `(*dims, col_id)`.
2. **Parameters as frames**. A `Param` is a frame `(*dims, value)`.
   `cost * v_production` does an inner-join on shared dims (`unit`)
   and emits an `Expr` of `(unit, hour, col_id, coef)`.
3. **Aggregation as group-by**. `Sum(v_production, over=("unit",))`
   collapses the `unit` dim, leaving `hour` for the demand
   constraint. Here `over=` is a *tuple of dim names* — the dims to
   collapse.
4. **Constraints**. `add_cstr(..., over=hour_idx, ...)` materialises
   one LP row per row of the `hour_idx` *DataFrame*. Here `over=` is
   a row-index frame (not a tuple of dim names) telling the engine
   which cells of the constraint family to instantiate.
5. **Solve**. `Problem.solve()` builds COO triples, hands HiGHS a
   `HighsLp` struct, and returns a [`Solution`](reference/api.md).

## Next steps

- The mental model: [Concepts](concepts/index.md)
- Idiomatic patterns: `Sum`, `Where`, `Lag` —
  [Expressions](concepts/expressions.md)
- Re-solve with parameter updates: [Warm-starting](guide/warm-starting.md)
- Decompose coupled subproblems: [Lagrangian](guide/lagrangian.md)
