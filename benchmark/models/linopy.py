"""linopy model for the benchmark LP. Same problem as models/polar.py."""

from __future__ import annotations

import math

import linopy
import numpy as np
import pandas as pd
import xarray as xr


def build(N: int) -> linopy.Model:
    m = linopy.Model()

    i_idx = pd.Index(np.arange(1, N + 1), name="i")
    j_idx = pd.Index(np.arange(1, N + 1), name="j")

    x = m.add_variables(lower=0, coords=[i_idx, j_idx], name="x")
    y = m.add_variables(lower=0, coords=[i_idx, j_idx], name="y")

    m.add_objective((2 * x + y).sum())

    rhs_i = xr.DataArray(i_idx.values.astype(float), coords=[i_idx])
    m.add_constraints(x - y >= rhs_i, name="c1")
    m.add_constraints(x + y >= 0, name="c2")
    return m


def solve(model: linopy.Model, time_limit: float | None = None) -> tuple[bool, float]:
    # linopy 0.5.x routes HiGHS options as loose kwargs to ``solve()``
    # when using the default LP-file io_api. Force LP-file i/o
    # explicitly so the time_limit kwarg actually reaches HiGHS.
    kwargs: dict = {"solver_name": "highs", "io_api": "lp"}
    if time_limit is not None:
        kwargs["time_limit"] = float(time_limit)
    model.solve(**kwargs)
    try:
        obj = float(model.objective.value)
    except Exception:
        obj = float("nan")
    return math.isfinite(obj), obj
