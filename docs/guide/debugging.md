# Debugging models

polar-high exposes its internals as plain polars DataFrames — the same
objects you built the model with.  There is no separate inspection API
to learn: if you know how to filter, join, or group a polars frame, you
already know how to interrogate a polar-high model.

## Inspect before solve

```python
p.cstr_names()                    # list every constraint family
p.cstr_row_count("balance")       # how many LP rows did "balance" produce?
p.cstrs_named("balance")[0]       # the CstrRecord (over_index, term protos, …)
```

Row count surprises are a common bug source: an `over=` frame
that's bigger or smaller than you expected, or a `Where(...)` that
silently filtered out half the rows.

Each variable's LP column mapping is also readable before the solve:

```python
v_flow.frame           # DataFrame with (*dims, col_id)
v_flow.frame.height    # how many LP columns did this variable produce?
```

`col_id` is the 0-based column index in the HiGHS problem.  Checking
`var.frame.height` against your expected cardinality catches duplicate
rows in the index frame before they become a cryptic "matrix is
singular" error.

## Inspect after solve

```python
sol = p.solve(keep_solver=True)

sol.obj                              # scalar objective
sol.optimal                          # True / False

sol.value("v_flow")                  # DataFrame (*dims, value)
sol.constraint_dual("balance")       # DataFrame (key, dual)  or  (dual,) scalar

sol.highs.writeModel("model.lp")     # full LP in human-readable format
sol.highs.writeModel("model.mps")    # MPS for solver benchmarking
sol.highs.getModelStatus()           # HighsModelStatus enum
sol.highs.getInfo()                  # dict of solver statistics
```

`sol.value()` returns the same dimension columns that were used to
declare the variable, so you can join solution frames straight back to
input data:

```python
result = (
    sol.value("v_flow")
    .join(cap.frame, on=["node", "hour"])
    .with_columns(utilisation=pl.col("value") / pl.col("value_right"))
)
```

`sol.constraint_dual()` currently returns a `key` string column rather
than the individual dimension columns — parse it if you need to join on
component parts.

`writeModel("model.lp")` produces human-readable LP format and is the
fastest route to "what did the kernel actually send to HiGHS" — diff
against a reference build of the same model to locate mis-wired terms.

## Worked example

A small energy-balance model with one node, two hours.  The model code
below is sourced from `tests/fixtures/debug_example.py` and is verified
by `tests/test_debug_example.py`, so the API calls here stay in sync
with the test suite.

```python
--8<-- "tests/fixtures/debug_example.py:model"
```

Pre-solve audit:

```python
print(p.cstr_names())
# ['balance', 'cap']

print(p.cstr_row_count("balance"))
# 2  (one per node×hour)

print(v_flow.frame)
# ┌──────┬──────┬────────┐
# │ node ┆ hour ┆ col_id │
# │ ---  ┆ ---  ┆ ---    │
# │ str  ┆ i64  ┆ i64    │
# ╞══════╪══════╪════════╡
# │ west ┆ 1    ┆ 0      │
# │ west ┆ 2    ┆ 1      │
# └──────┴──────┴────────┘
```

Solve and inspect:

```python
sol = p.solve(keep_solver=True)
print(sol.optimal, sol.obj)
# True 200.0

print(sol.value("v_flow"))
# ┌──────┬──────┬───────┐
# │ node ┆ hour ┆ value │
# ╞══════╪══════╪═══════╡
# │ west ┆ 1    ┆ 100.0 │
# │ west ┆ 2    ┆ 100.0 │
# └──────┴──────┴───────┘

print(sol.constraint_dual("balance"))
# ┌──────────────┬───────┐
# │ key          ┆ dual  │
# ╞══════════════╪═══════╡
# │ west,1       ┆ 1.0   │
# │ west,2       ┆ 1.0   │
# └──────────────┴───────┘

sol.highs.writeModel("energy.lp")
```

The LP file shows named rows and columns (e.g. `balance[west,1]`,
`v_flow[west,1]`), so it reads as a direct map back to your model
declarations.

## Term isolation

When two builds of the same model disagree on `obj`, check the
constraint families one at a time:

1. Same `cstr_row_count` per family?
2. Same `constraint_dual` shape per family?
3. Same nonzero pattern in `model.lp`?

Because the labelled-form `add_cstr(... lhs_terms={...}, rhs_terms={...})`
preserves term names through to the LP, you can also write a
diagnostic comparator that builds two reference Exprs and compares
their resulting term frames — much sharper than "obj is off by 3.2%".

## Numerical issues

- **Infeasibility** — `sol.optimal == False` and
  `sol.highs.getModelStatus()` reports `kInfeasible`. Use HiGHS's
  irreducible-infeasible-subset support (`h.getIis(...)`) via the
  live handle if available in your `highspy` build.
- **Unboundedness** — usually a missing bound. Check that every
  `add_var` call set a finite upper bound where appropriate, and that
  no constraint family was accidentally skipped.
- **Degeneracy / cycling** — rare with default HiGHS, more common
  with hand-tuned options. Try `solver=simplex` vs `ipm` to triangulate.

## Warm-problem invariants

When using [`WarmProblem`](warm-starting.md), the first wrong-result
debugging step is always: *does a fresh `Problem.solve()` from the
same final state agree with the warm result?* If yes, the kernel is
fine and your update calls are off; if no, there's a kernel bug to
report.

## How this differs from other tools

The table below summarises the debugging workflow in comparable tools.
polar-high's distinctive property is that *the same data structure you
use to build the model is what inspection returns* — no secondary
object model to learn.

| Tool | Inspection API | Output type | Pre-solve row-count check |
|---|---|---|---|
| **polar-high** | `cstr_row_count`, `var.frame`, `value()`, `constraint_dual()` | polars DataFrame | `cstr_row_count()` before solve |
| **Linopy** | `model.constraints[name].labels`, `solution.dual` | xarray DataArray/Dataset | not directly exposed |
| **Pyomo** | `model.component_objects()`, `pyo.value()`, `model.dual[cstr]` | Pyomo ComponentData objects | not directly exposed |
| **JuMP** | `print(model)`, `variable_by_name()`, `dual(cstr_ref)` | Julia scalar / iterator | not directly exposed |
| **GAMS** | listing file (`.lst`), equation listing text block | plain text | not directly exposed |

**Linopy** is the closest relative: it is also array-native and returns
duals as labelled arrays.  The difference is the coordinate library
(xarray vs polars) and the fact that Linopy targets a wider range of
solvers via a solver-agnostic interface while polar-high is HiGHS-based,
which lets it expose the live `Highs` handle directly.

**Pyomo** and **JuMP** use a symbolic-expression graph.  Inspection
yields model objects (Pyomo's `ComponentData`, JuMP's `ConstraintRef`)
rather than data frames; you must call `.value()` or iterate to
materialise numbers, and the result is not immediately joinable to
input data.

**GAMS** has no programmatic inspection API in the traditional sense.
The listing file is the primary debugging artifact: it records generated
equation listings as formatted text, and you read it in an editor.
gamspy (the Python embedding) recovers output into pandas DataFrames
post-solve, but pre-solve shape checking is not available from Python.
