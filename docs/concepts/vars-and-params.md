# Vars and Params

## Var

A `Var` is a polars frame with columns `(*dims, col_id)`. One row =
one LP column. The `col_id` is allocated by the `Problem` and used
internally to address the HiGHS column.

```python
v = p.add_var(
    "v",
    dims=("i", "j"),
    index=pl.DataFrame({"i": [1, 1, 2], "j": ["a", "b", "a"]}),
    lower=0.0,
    upper=float("inf"),
)
v.frame
# ┌─────┬─────┬────────┐
# │ i   ┆ j   ┆ col_id │
# │ --- ┆ --- ┆ ---    │
# │ i64 ┆ str ┆ i64    │
# ╞═════╪═════╪════════╡
# │ 1   ┆ a   ┆ 0      │
# │ 1   ┆ b   ┆ 1      │
# │ 2   ┆ a   ┆ 2      │
# └─────┴─────┴────────┘
```

- `dims` is just the list of frame columns to treat as index axes.
- `index` is a polars DataFrame whose rows enumerate the index set.
- `lower` / `upper` are either scalar bounds applied to every column,
  or a `Param` giving per-element bounds (see below).

### Per-element bounds

Pass a `Param` as `lower` and/or `upper` to bound each column
individually:

```python
cap = Param(("i",), pl.DataFrame({"i": [1], "value": [5.0]}))
v = p.add_var("v", ("i", "j"), index, upper=cap)
# columns (1,a) and (1,b) get upper 5.0; (2,a) keeps the default +inf
```

- The Param's dims must be a subset of the variable's dims (otherwise
  `ValueError`). Rows are aligned by joining on the Param's dims, so a
  lower-dim Param broadcasts over the remaining variable dims.
- Variable elements with **no matching Param row** get the default
  bound (`lower` 0.0, `upper` +inf) — a sparse bound Param only
  tightens where it has rows.
- `null` / `NaN` values also mean "default"; `+inf` / `-inf` are
  allowed (e.g. `lower=-inf` frees an element).
- Duplicate Param rows on its dims raise `ValueError`.
- The bound is resolved when `add_var` is called and stored on
  `Var.lower` / `Var.upper` as a float64 array aligned with
  `Var.frame` rows. Scalar bounds stay scalars (identical LP for
  existing callers). Read scatter-ready values with `Var.col_lower()` /
  `Var.col_upper()`; rescale finite bounds of either form with
  `Var.scale_bounds(factor)`.

## Param

A `Param` is a frame with columns `(*dims, value)`.

```python
c = Param(
    ("i", "j"),
    pl.DataFrame({"i": [1, 1, 2],
                  "j": ["a", "b", "a"],
                  "value": [3.0, 1.0, 2.0]}),
)
```

`Param.scalar(0.5)` makes a 0-dim Param. Param frames are stored as
polars `LazyFrame`s internally; they only `collect()` when you read
`.frame` or when the engine assembles the LP.

## Param × Param

Multiplying two Params does an **inner-join on shared dims** and
multiplies the value columns:

```python
a = Param(("i",),      df_a)   # value_a per i
b = Param(("i", "j"),  df_b)   # value_b per (i, j)
ab = a * b                     # Param(("i", "j"), value_a * value_b)
```

If the two Params share *no* dims, the join is cross-product. The
result is again a `Param`; chains of `Param * Param / Param` are lazy
until consumed.

!!! warning "Inner-join, not outer"

    `Param + Param` and `Param * Param` are inner-joins on shared
    dims. Cells that exist in one Param but not the other are
    **dropped**. If you want zero-fill semantics (densification),
    `left_join` your Params against the index frame and `fill_null(0)`
    before constructing the Param.

## Param × Var → Expr

Multiplying a `Param` by a `Var` gives an `Expr` whose underlying
frame has columns `(*union_dims, col_id, coef)`:

```python
term = c * v
# Each row is one (i, j, col_id, coef) — the HiGHS coefficient that
# will go into whatever LP row this term is summed into.
```

The arithmetic is a polars join, so the cost is set by polars'
join performance, not by Python loops.

## See also

- [Expressions](expressions.md) — `Sum`, `Where`, `Lag` and how dims
  flow through them.
- [Problem and Solve](problem-and-solve.md) — how Exprs become LP rows.
