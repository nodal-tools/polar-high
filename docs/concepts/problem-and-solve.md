# Problem and Solve

## Problem

`Problem` is the LP container. Lifecycle:

```python
p = Problem()
v = p.add_var(...)                     # register a variable
p.set_objective(c * v, sense="min")    # set an objective
p.add_cstr(name, over=…, sense=…,      # add a constraint family
           lhs_terms=…, rhs_terms=…)
sol = p.solve()                         # build LP, hand to HiGHS, return Solution
```

Constraints are stored as protos and only materialize
into LP rows when `solve()` (or `WarmProblem`) is invoked.

## Two ways to add a constraint

### Operator form

For simple expressions:

```python
p.add_cstr("cap", v <= 5.0)         # not yet — see note below
```

!!! note "Use the dict form for indexed constraints"

    The operator form is convenient for scalar constraints. For
    indexed constraint families (almost all real-world cases), use
    the labelled `lhs_terms` / `rhs_terms` form below — the engine
    can produce term-isolated diagnostics when something goes wrong.

### Labelled form

```python
p.add_cstr(
    name="balance",
    over=node_time_index,         # (n, d, t) frame; one LP row per row here
    sense="==",                   # one of "<=", ">=", "=="
    lhs_terms={                   # named contributions on the LHS
        "inflow":  Sum(v_in,  over=…),
        "outflow": Sum(v_out, over=…),
        "delta":   v_state - Lag(v_state, …),
    },
    rhs_terms={                   # named contributions on the RHS
        "demand": p_demand,
    },
)
```

Each term's open dims must be a subset of `over`'s columns. Constants
(`int`, `float`) and `Param` are accepted on either side. `Var` /
`Expr` on the RHS are moved to the LHS automatically.

## Row counts

`p.cstr_row_count("balance")` returns the number of LP rows the
constraint produced. `p.cstrs_named("balance")` returns the
[`CstrRecord`](../reference/api.md) for the constraint family — useful
when you need to look up row indices for warm updates or duals.

## Solution

`Solution` is what `Problem.solve()` returns:

```python
sol = p.solve()
sol.optimal              # bool
sol.obj                  # float — objective value (with offset applied)
sol.value(v)             # polars frame: (*dims, value)
sol.value_wide(v, …)     # wide-form frame for time-indexed vars
sol.constraint_dual("balance")   # polars frame: (*over_dims, dual)
sol.col_dual             # numpy array of reduced costs
sol.highs                # the live highspy.Highs instance — for advanced use
```

Numerics: HiGHS's optimality status is reported on `sol.optimal`.
For non-optimal terminations, inspect `sol.highs.getModelStatus()`
directly.

## Solver options

Pass HiGHS options at construction time:

```python
p.set_solver_options({"presolve": "off", "solver": "simplex"})
```

or per-solve:

```python
sol = p.solve(options={"time_limit": 60.0})
```

The per-call `options` overrides the stored options. Keys must match
HiGHS canonical names; values must be the type HiGHS expects
(`str`/`int`/`float`/`bool`).

## See also

- [Duals and bases](duals-and-bases.md) — accessing duals,
  reduced costs, and the `highs` handle.
- [Warm-starting](../guide/warm-starting.md) — re-solving with
  parameter / RHS updates while preserving the basis.
