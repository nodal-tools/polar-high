"""Pyomo model for the sparse network-flow benchmark.

Same LP and the same numbers as ``polar_net.py``; uses the appsi_highs
persistent solver interface (the fastest Pyomo path to HiGHS).

Idiomatic Pyomo: precompute ``incoming[n]`` / ``outgoing[n]`` adjacency
dicts during build, then a ``pe.Constraint(N, T, rule=...)`` whose rule
sums the relevant edges by Python-iteration.
"""

from __future__ import annotations

import pyomo.environ as pe

from ._network_data import generate


def build(N: int) -> pe.ConcreteModel:
    data = generate(N)
    edges = data["edges"]
    cap_arr = data["cap"]
    demand_arr = data["demand"]
    T = data["T"]
    E = data["E"]

    src_list = edges["src"].to_list()
    dst_list = edges["dst"].to_list()
    cost_list = edges["cost"].to_list()

    incoming: dict[int, list[int]] = {n: [] for n in range(N)}
    outgoing: dict[int, list[int]] = {n: [] for n in range(N)}
    for e in range(E):
        incoming[dst_list[e]].append(e)
        outgoing[src_list[e]].append(e)

    m = pe.ConcreteModel()
    m.E = pe.RangeSet(0, E - 1)
    m.N = pe.RangeSet(0, N - 1)
    m.T = pe.RangeSet(0, T - 1)

    m.flow = pe.Var(m.E, m.T, within=pe.NonNegativeReals)

    m.obj = pe.Objective(
        expr=sum(cost_list[e] * m.flow[e, t] for e in range(E) for t in range(T)),
        sense=pe.minimize,
    )

    def _cap_rule(m, e, t):
        return m.flow[e, t] <= float(cap_arr[e, t])

    m.capacity = pe.Constraint(m.E, m.T, rule=_cap_rule)

    def _balance_rule(m, n, t):
        return sum(m.flow[e, t] for e in incoming[n]) - sum(
            m.flow[e, t] for e in outgoing[n]
        ) == float(demand_arr[n, t])

    m.node_balance = pe.Constraint(m.N, m.T, rule=_balance_rule)

    return m


def solve(model: pe.ConcreteModel, time_limit: float | None = None) -> tuple[bool, float]:
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
