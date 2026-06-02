"""PuLP model for the benchmark LP. Same problem as models/polar.py.

Uses HiGHS_CMD (LP file + subprocess to the ``highs`` executable), the
file-based PuLP→HiGHS path comparable to linopy's ``io_api="lp"``.
"""

from __future__ import annotations

import shutil

from pulp import (
    HiGHS_CMD,
    LpMinimize,
    LpProblem,
    LpStatusOptimal,
    lpSum_vars,
    lpSum_vars_coefs,
    value,
)

from ._network_data import generate


def build(N: int) -> LpProblem:
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

    m = LpProblem("bench_net", LpMinimize)
    all_pairs = [(e, t) for e in range(E) for t in range(T)]

    flow = m.add_variable_dicts("flow", indices=all_pairs, lowBound=0)

    m += lpSum_vars_coefs([(flow[e, t], cost_list[e]) for e, t in all_pairs])

    for e, t in all_pairs:
        m += flow[e, t] <= cap_arr[e, t]

    for n in range(N):
        for t in range(T):
            m += lpSum_vars([flow[e, t] for e in incoming[n]]) - \
                lpSum_vars([flow[e, t] for e in outgoing[n]]) == demand_arr[n, t]

    return m


def _highs_cmd(time_limit: float | None) -> HiGHS_CMD:
    kwargs: dict = {"msg": False, "mip": False}
    if time_limit is not None:
        kwargs["timeLimit"] = float(time_limit)
    path = shutil.which("highs")
    if path is not None:
        kwargs["path"] = path
    return HiGHS_CMD(**kwargs)


def solve(model: LpProblem, time_limit: float | None = None) -> tuple[bool, float]:
    # HiGHS_CMD may raise when time-limited runs leave no parseable log
    # line (build-only benchmark uses time_limit≈1e-6).
    try:
        status = model.solve(_highs_cmd(time_limit))
    except Exception:
        return False, float("nan")
    optimal = status == LpStatusOptimal
    try:
        raw = value(model.objective)
        obj = float("nan") if raw is None else float(raw)
    except Exception:
        obj = float("nan")
    return optimal, obj
