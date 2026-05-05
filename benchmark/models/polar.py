"""polar-high model for the benchmark LP.

Indexed LP (mirrors the linopy benchmark, linearised):

    min  Σ_{i,j} (2·x[i,j] + y[i,j])
    s.t. x[i,j] - y[i,j] >= i        for i,j ∈ {1,…,N}
         x[i,j] + y[i,j] >= 0        for i,j ∈ {1,…,N}
         x, y >= 0
"""
from __future__ import annotations

import numpy as np
import polars as pl

from polar_high import Param, Problem, Sum


def build(N: int) -> Problem:
    p = Problem()

    i_arr = np.repeat(np.arange(1, N + 1, dtype=np.int64), N)
    j_arr = np.tile(np.arange(1, N + 1, dtype=np.int64), N)
    idx = pl.DataFrame({"i": i_arr, "j": j_arr})

    x = p.add_var("x", dims=("i", "j"), index=idx, lower=0.0)
    y = p.add_var("y", dims=("i", "j"), index=idx, lower=0.0)

    obj = x.to_expr() * 2.0 + y.to_expr() * 1.0
    p.set_objective(Sum(obj), sense="min")

    rhs_i = Param(
        ("i",),
        pl.DataFrame(
            {
                "i": np.arange(1, N + 1, dtype=np.int64),
                "value": np.arange(1, N + 1, dtype=np.float64),
            }
        ),
    )
    p.add_cstr(
        "c1",
        over=idx,
        sense=">=",
        lhs_terms={"x": x, "neg_y": -y.to_expr()},
        rhs_terms={"i": rhs_i},
    )
    p.add_cstr(
        "c2",
        over=idx,
        sense=">=",
        lhs_terms={"x": x, "y": y},
        rhs_terms={"zero": 0.0},
    )
    return p


def solve(model: Problem, time_limit: float | None = None) -> tuple[bool, float]:
    options = {"time_limit": float(time_limit)} if time_limit is not None else None
    sol = model.solve(options=options)
    return bool(sol.optimal), float(sol.obj)
