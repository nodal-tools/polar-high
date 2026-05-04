# polar-high-opt

A polars-backed LP/MIP eDSL on top of HiGHS.  Designed as a generic
optimization-modelling kernel — knows nothing about any specific
domain model.

The integration layer that drives flextool-style energy-system models
lives in the [flextool](https://github.com/irena-flextool/flextool)
repository under `flextool/engine_polars/`; this repository is the
underlying engine.

## Install

```bash
pip install -e .
pip install -e ".[test]"   # for running the tests
```

Requires Python 3.11+.

## Quick start

```python
from polar_high_opt import Problem, Param, Sum, Where
import polars as pl

p = Problem()

# Decision variable indexed by (i, j)
v = p.add_var("v", dims=("i", "j"),
              index=pl.DataFrame({"i": [1, 1, 2], "j": ["a", "b", "a"]}),
              lb=0.0)

# Cost parameter
c = Param(dims=("i", "j"),
          frame=pl.DataFrame({"i": [1, 1, 2], "j": ["a", "b", "a"], "value": [3.0, 1.0, 2.0]}))

# Objective: minimize sum_{i,j} c[i,j] * v[i,j]
p.minimize(Sum(c * v, over=("i", "j")))

# Capacity constraint: sum_j v[i,j] ≤ 5 for each i
p.add_constraint(Sum(v, over=("j",)) <= 5.0, name="cap")

sol = p.solve()
print(sol.obj, sol.value(v))
```

## What the engine provides

* `Var`, `Param`, `Expr` — building blocks for indexed expressions
  expressed as polars DataFrames.
* `Sum`, `Where`, `Lag` — aggregation and filtering primitives that
  compile to LP rows efficiently.
* `Problem` — assemble the model and solve via HiGHS.
* `WarmProblem` — re-solve with parameter / RHS updates while preserving
  the basis.
* `LagrangianProblem` — generic dual-subgradient driver for Lagrangian
  decomposition.

The aim is that the eDSL stays lean and domain-free; everything
flextool-specific (CSV/SpineDB readers, model construction, multi-solve
chain, handoff state) lives in flextool's own repo.

## Layout

```
polar-high-opt/
├── src/polar_high_opt/
│   ├── __init__.py            # public API surface
│   ├── engine.py              # core: Var, Param, Expr, Sum, Where, Problem, WarmProblem
│   └── lagrangian.py          # generic dual-subgradient driver
└── tests/
    ├── conftest.py
    ├── fixtures/              # synthetic toy models
    └── test_*.py              # engine tests
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
