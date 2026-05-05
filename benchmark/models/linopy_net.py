"""linopy model for the sparse network-flow benchmark.

Same LP and the same numbers as ``polar_net.py`` — both pull from
``_network_data.generate``.

Idiomatic linopy: use ``DataArray`` coordinate arrays for ``src`` / ``dst``
and ``flow.groupby(dst_arr).sum()`` to express the per-node aggregations
without ever materialising a dense ``(E, N, T)`` tensor.
"""

from __future__ import annotations

import math

import linopy
import numpy as np
import pandas as pd
import xarray as xr

from ._network_data import generate


def build(N: int) -> linopy.Model:
    data = generate(N)
    edges = data["edges"]
    cap_arr = data["cap"]
    demand_arr = data["demand"]
    T = data["T"]
    E = data["E"]

    src_np = edges["src"].to_numpy()
    dst_np = edges["dst"].to_numpy()
    cost_np = edges["cost"].to_numpy()

    e_idx = pd.Index(np.arange(E), name="e")
    t_idx = pd.Index(np.arange(T), name="t")
    n_idx = pd.Index(np.arange(N), name="n")

    m = linopy.Model()
    flow = m.add_variables(lower=0, coords=[e_idx, t_idx], name="flow")

    # ---- objective ------------------------------------------------------
    cost = xr.DataArray(cost_np, coords=[e_idx])
    m.add_objective((cost * flow).sum())

    # ---- capacity bound -------------------------------------------------
    cap = xr.DataArray(cap_arr, coords=[e_idx, t_idx])
    m.add_constraints(flow <= cap, name="capacity")

    # ---- node-balance ---------------------------------------------------
    # groupby uses the named coord on the resulting expression, so name the
    # mapping array "n" to align with the demand DataArray.
    dst_arr = xr.DataArray(dst_np, coords=[e_idx], name="n")
    src_arr = xr.DataArray(src_np, coords=[e_idx], name="n")
    inflow = flow.groupby(dst_arr).sum()
    outflow = flow.groupby(src_arr).sum()
    demand = xr.DataArray(demand_arr, coords=[n_idx, t_idx])
    m.add_constraints(inflow - outflow == demand, name="node_balance")

    return m


def solve(model: linopy.Model, time_limit: float | None = None) -> tuple[bool, float]:
    kwargs: dict = {}
    if time_limit is not None:
        kwargs["time_limit"] = float(time_limit)
    model.solve(solver_name="highs", **kwargs)
    try:
        obj = float(model.objective.value)
    except Exception:
        obj = float("nan")
    return math.isfinite(obj), obj
