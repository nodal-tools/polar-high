# Quickstart

A tiny dispatch LP. Two generation units (wind, coal) over three
hours. Wind is cheap but variable; coal is expensive but firm. Pick
how much each unit produces per hour to meet demand at minimum cost.

The model code below is sourced from `tests/fixtures/quickstart_example.py`
and is verified by `tests/test_quickstart_example.py`.

```python
--8<-- "tests/fixtures/quickstart_example.py:model"
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
- Decompose coupled subproblems: [Decomposition building blocks](guide/decomposition.md)
