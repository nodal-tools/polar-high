"""Pyomo model for the benchmark LP. Same problem as models/polar.py.

Uses the appsi_highs persistent-solver interface (the fastest Pyomo
path to HiGHS). Falls back to writing an LP file via SolverFactory
if appsi is unavailable.
"""
from __future__ import annotations

import pyomo.environ as pe


def build(N: int) -> pe.ConcreteModel:
    m = pe.ConcreteModel()
    m.I = pe.RangeSet(1, N)
    m.J = pe.RangeSet(1, N)
    m.x = pe.Var(m.I, m.J, within=pe.NonNegativeReals)
    m.y = pe.Var(m.I, m.J, within=pe.NonNegativeReals)

    m.obj = pe.Objective(
        expr=sum(2 * m.x[i, j] + m.y[i, j] for i in m.I for j in m.J),
        sense=pe.minimize,
    )
    m.c1 = pe.Constraint(m.I, m.J, rule=lambda m, i, j: m.x[i, j] - m.y[i, j] >= i)
    m.c2 = pe.Constraint(m.I, m.J, rule=lambda m, i, j: m.x[i, j] + m.y[i, j] >= 0)
    return m


def solve(
    model: pe.ConcreteModel, time_limit: float | None = None
) -> tuple[bool, float]:
    try:
        from pyomo.contrib.appsi.solvers import Highs

        solver = Highs()
        if time_limit is not None:
            solver.config.time_limit = float(time_limit)
            # In time-limited / build-only mode, HiGHS may return no
            # feasible solution and appsi would otherwise raise on load.
            solver.config.load_solution = False
        res = solver.solve(model)
        cond = str(res.termination_condition).lower()
        optimal = "optimal" in cond
    except ImportError:
        solver = pe.SolverFactory("highs")
        if time_limit is not None:
            solver.options["time_limit"] = float(time_limit)
        res = solver.solve(model, load_solutions=time_limit is None)
        optimal = str(res.solver.termination_condition).lower() == "optimal"

    try:
        obj = float(pe.value(model.obj))
    except Exception:
        obj = float("nan")
    return optimal, obj
