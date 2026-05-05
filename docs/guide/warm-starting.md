# Warm-starting with WarmProblem

For rolling-horizon, parameter-sweep, or Lagrangian workflows you
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

## Update kinds

| Method | What it changes |
|---|---|
| `update_rhs(cstr_name, new_param)` | RHS of an entire constraint family. |
| `update_obj_coef(var_name, …)` | Per-cell objective coefficients. |
| `update_obj_coef_array(var_name, dim_tuples, values)` | Array-form bulk objective update. |
| `fix_cols(var_name, dim_tuples, values)` | Fix specific variable cells (lb=ub=value). |
| `update_coef(row, col, value)` | Single coefficient update — rare but available. |
| `declare_mutable(*param_names)` + `update_param(name, …)` | Param-tracked auto-update: declare which Params can change, then push updates and let the engine recompute every affected coefficient. |

The Param-tracked path is the most ergonomic when many constraints
depend on the same Param chain (e.g. cost = `c_fuel * heat_rate`):
update the Param, every coefficient that traces back to it is
recomputed.

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
