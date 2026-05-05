# Expressions

An `Expr` is a list of term frames, each carrying `(*dims, col_id,
coef)`. Three primitives shape Exprs: `Sum`, `Where`, `Lag`.

## Sum

`Sum(expr, over=("j",))` group-by-sums each term over the listed
dims. The dims listed disappear; the rest become the term's *open
dims*, ready to be matched against a constraint's `over=` frame.

```python
flow_per_i = Sum(v, over=("j",))    # open dim: i
total_flow = Sum(v, over=None)      # open dims: () — collapses to a scalar
```

`Sum(..., over=None)` collapses *every* open dim — useful for
objective terms or single-row constraints.

`Sum(..., where=frame)` pre-filters term frames before the
group-by-sum (inner-join on shared columns).

## Where

`Where(expr, frame)` inner-joins an Expr with a frame, doing two
things at once:

- **Filter** — rows of the term whose join keys don't appear in
  `frame` are dropped (e.g. `Where(v_flow, wind_only)` keeps only the
  wind rows).
- **Map** — columns of `frame` that the term doesn't already carry
  become *new open dims*. This is how you remap a `(p, source, sink)`
  flow variable into a constraint indexed by `(n, t)`:

```python
# flow_to_n maps each (p, source, sink) to the destination node n
v_flow_at_n = Where(v_flow, flow_to_n)
# v_flow_at_n now has open dims (p, source, sink, n) — once Sum'd over
# (p, source, sink) it leaves just (n,) for a per-node balance.
```

## Lag

`Lag(var, lag_frame, time_dim, lag_col)` references `var` at a
shifted time index, joined via `lag_frame`:

```python
# Storage state-change: v_state[t] - v_state[t_prev]
state_diff = v_state - Lag(v_state, dt_frame,
                           time_dim="t", lag_col="t_prev")
```

The lag frame is responsible for defining the temporal mapping —
typically `(d, t, t_prev)` for within-period lag, or `(d, t,
d_prev, t_prev)` for cross-period lag. This pushes the modelling
choice (cyclic? acyclic? cross-period?) out of the engine and into
data the user supplies.

## Arithmetic

```python
expr1 + expr2     # term-list concatenation
expr * 2.5        # scale every term's coef
expr + 5.0        # adds a constant — engine handles via obj_offset / RHS shift
param * var       # see vars-and-params.md
```

`Var + Var`, `Var - Var`, `Expr + Expr` all return Exprs. Adding a
scalar to an `Expr` defers the constant to the right-hand side at
constraint-binding time.

## Open vs closed dims

- A term's **open dims** are the columns of the term frame that are
  *not* `col_id` / `coef`.
- When you call `add_cstr(..., over=row_index, lhs_terms=…,
  rhs_terms=…)`, the engine matches each term's open dims against
  `row_index`'s columns. Terms must have **a subset of `row_index`'s
  columns** as open dims.
- `Sum` reduces open dims, `Where` adds open dims via the join frame.

This is the load-bearing invariant of the modelling layer: get your open dims
right and constraints just compose. Get them wrong and you'll see
"missing dim" errors or row-count mismatches.

## See also

- [Vars and Params](vars-and-params.md) — what `Param * Var` returns.
- [Problem and Solve](problem-and-solve.md) — how `over=` matches
  open dims to row dims.
