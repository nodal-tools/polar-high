"""PuLP model for the benchmark LP. Same problem as models/polar.py.

Uses HiGHS_CMD (LP file + subprocess to the ``highs`` executable), the
file-based PuLP→HiGHS path comparable to linopy's ``io_api="lp"``.
"""

from __future__ import annotations

import shutil

from pulp import HiGHS_CMD, LpMinimize, LpProblem, LpStatusOptimal, lpSum_vars_coefs, value


def build(N: int) -> LpProblem:
    m = LpProblem("bench", LpMinimize)
    idx = range(1, N + 1)
    all_pairs = [(i, j) for i in idx for j in idx]
    x = m.add_variable_dicts("x", indices=all_pairs, lowBound=0)
    y = m.add_variable_dicts("y", indices=all_pairs, lowBound=0)
    _my_list = [(x[i,j], 2) for (i, j) in all_pairs]
    _my_list.extend([(y[i,j], 1) for (i, j) in all_pairs])
    m += lpSum_vars_coefs(_my_list)


    for i, j in all_pairs:
        m += x[i,j] - y[i,j] >= i
        m += x[i,j] + y[i,j] >= 0
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
