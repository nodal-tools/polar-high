# Warm-starting with WarmProblem

For rolling-horizon, parameter-sweep, or decomposition workflows you
want to **re-solve** with small parameter / RHS / coefficient updates
without rebuilding the LP. `WarmProblem` does this on top of
`Problem`.

## Lifecycle

```python
from polar_high import Problem, WarmProblem

p = Problem()
# ... add_var / add_cstr / set_objective as usual ...

wp = WarmProblem(p)
sol0 = wp.solve()        # first solve: builds the LP

# update something
wp.update_rhs("balance", new_demand_param)
sol1 = wp.solve()        # second solve: re-uses basis

wp.update_obj_coef("v", per_cell_coefs)
sol2 = wp.solve()
```

Between solves the underlying `highspy.Highs` instance is preserved
— HiGHS's basis (and post-presolve state) carry over.

The live LP keeps the labels of the `Problem` it was built from (the
`Var` frames and the HiGHS column / row names). Updates change values,
never labels: a `Param` is matched to rows and columns by those build
labels. To solve a model whose labels differ (e.g. the next window of a
rolling horizon), build a new `Problem`.

## Update kinds

| Method | What it changes | Value frame |
|---|---|---|
| `update_rhs(cstr_name, new_param, *, frame="user")` | The whole RHS of a constraint family (replaces any constant the build put into the rows). | user (default) or scaled |
| `update_obj_coef(var_name, …, *, frame="user")` | Per-cell objective coefficients. | user (default) or scaled |
| `update_obj_coef_array(var_name, dim_tuples, values, *, frame="user")` | Array-form bulk objective update. | user (default) or scaled |
| `fix_cols(var_name, dim_tuples, values)` | Fix specific variable cells (lb=ub=value). | scaled |
| `fix_col_ids` / `set_col_bounds` | Column bounds by raw col id. | scaled |
| `update_coef(row, col, value)` | Single coefficient update — rare but available. | scaled |
| `declare_mutable(*param_names)` + `update_param(name, …)` | Param-tracked auto-update: declare which Params can change, then push updates and let the engine recompute every affected matrix coefficient. | user |

**Value frames.** With Layer-2 autoscaling the built LP is scaled by
power-of-two side vectors (`Problem._layer2_row_factor` /
`_layer2_col_factor`). A **user**-frame value is in the units of the
model as written (`add_cstr` RHS terms, `set_objective` terms); the
method applies the side vectors exactly as the build did, so an update
means the same as rebuilding with the new value. A **scaled**-frame value
is written to HiGHS as given; the caller applies any transform itself.
With no side vectors both frames are identical. The small-value floor
(`Problem.coef_zero_threshold`) is always judged on the user-space
magnitude.

The Param-tracked path is the most ergonomic when many constraints
depend on the same Param chain (e.g. cost = `c_fuel * heat_rate`):
update the Param, every matrix coefficient that traces back to it is
recomputed. It updates tracked matrix cells only: the Param's
contributions to the objective, row bounds or column bounds, and cells
reached through an anonymous composite Param, are not updated. Values
are matched by the Param's dim labels, so a Param on shifted labels is
misplaced rather than re-aligned.

## Performance notes

- The first `solve()` does the same work as `Problem.solve()`. The
  speedup is on subsequent solves.
- `update_coef` is row-by-row; prefer the array forms when bulk
  updating.
- HiGHS presolve interacts subtly with warm starts. If your warm
  speedup is disappointing, try `set_solver_options({"presolve":
  "off"})` and re-measure.

## Worked example

See `tests/test_warm_problem.py` for end-to-end uses including
parameter sweeps and a small rolling-horizon chain.
