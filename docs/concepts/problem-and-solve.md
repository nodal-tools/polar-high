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

`Problem` takes an optional **`dense_axes`** argument — a tuple of
axis names that *you* promise are the trailing sort keys on every
frame you pass that contains them. (polar-high checks the promise
once per `Var` with a single-pass scan and raises a clear error if
it's broken.) Declaring it opts in to the block-COO evaluation arm,
which slices the dense suffix of each `Var` as a contiguous numpy
view and multiplies in ufuncs instead of polars joins. The
performance guide explains the
[contract and when it pays](../guide/performance.md#dense_axes-block-coo-arm).

## Adding a constraint

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
sol.obj                  # float — objective value (with the add_obj_constant constant)
sol.value("v")             # polars frame: (*dims, value)
sol.value_wide("v", …)    # wide-form frame for time-indexed vars
sol.constraint_dual("balance")   # polars frame: (key, dual)
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

### Objective-valued options

Options that are objective values — HiGHS `objective_target`,
`objective_bound` and `mip_abs_gap`; Gurobi `BestObjStop`, `BestBdStop`,
`Cutoff`, `MIPGapAbs`, `PoolGapAbs`; CPLEX `mip tolerances
lowercutoff`/`uppercutoff`, `mip limits lowerobjstop`/`upperobjstop`,
`simplex limits lowerobj`/`upperobj`, `mip tolerances absmipgap`/
`objdifference`, `mip pool absgap`, `mip polishafter absmipgap`, `barrier
limits objrange`; Xpress `MIPABSCUTOFF`, `MIPADDCUTOFF`, `MIPABSSTOP`;
COPT `AbsGap` — are written in **your** objective units and in the sense
of your objective, including the constant from `add_obj_constant`.
polar-high maps them to what the solver compares:

- if you build the objective scaled, declare it with
  `p.set_objective_scale(factor)` (the objective you pass is `factor` ×
  your objective, e.g. `1e-6` for currency built as million currency);
  levels and absolute gaps are multiplied by `factor`;
- HiGHS' `user_objective_scale = N` multiplies them by `2**N` (HiGHS
  compares them in its scaled space);
- the objective constant is taken off a level when the solver does not
  hold the constant itself;
- HiGHS reads `objective_bound` in its minimisation space, so under
  `sense="max"` polar-high passes it negated: a bound on a maximisation is
  the lowest objective you accept.

Relative gaps (`mip_rel_gap`, `MIPGap`, …) and values that are not finite
or have `|v| >= 1e20` pass unchanged.  The stored options are never
rewritten.  This applies to every in-process HiGHS solve (`Problem.solve`,
`WarmProblem.solve`, `WarmProblem.solve_with_fallback`) and to
`polar_high.solvers.solve`.  For a solver that reads a file from
`write_mps`, translate the options yourself:

```python
from polar_high.objective_options import translate_objective_options

transform = p.objective_transform("gurobi", "mps")
options, translated = translate_objective_options("gurobi", options, transform)
```

The objective constant is never passed to HiGHS (HiGHS would not scale
it with `user_objective_scale`, which makes its MIP bound and gap wrong);
it is added to `sol.obj` and `solve_diagnostics().objective_value`.  A raw
`sol.highs.getObjectiveValue()` therefore excludes it (`sol.obj_offset`
holds it), and HiGHS' relative MIP gap is measured on the objective
without the constant.  `sol.objective_in_user_units` is `sol.obj /
factor`.

## See also

- [Duals and bases](duals-and-bases.md) — accessing duals,
  reduced costs, and the `highs` handle.
- [Warm-starting](../guide/warm-starting.md) — re-solving with
  parameter / RHS updates while preserving the basis.
