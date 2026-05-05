"""polar-high model for the sparse network-flow benchmark.

LP:
    min  Σ_{e,t} cost[e] · flow[e, t]
    s.t. flow[e, t] <= cap[e, t]                                        ∀ (e, t)
         Σ_{e: dst[e]=n} flow[e,t] − Σ_{e: src[e]=n} flow[e,t] = demand[n, t]   ∀ (n, t)
         flow >= 0

Shows off polars-style join semantics: the node-balance constraint maps the
``e`` dim onto ``n`` via ``Where(flow, edges_dst_n)`` (and ``edges_src_n``),
then sums the ``e`` axis out.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from polar_high import Param, Problem, Sum, Where

from ._network_data import generate


def build(N: int) -> Problem:
    data = generate(N)
    edges: pl.DataFrame = data["edges"]
    cap_arr: np.ndarray = data["cap"]
    demand_arr: np.ndarray = data["demand"]
    T: int = data["T"]
    E: int = data["E"]

    p = Problem()

    # ---- index frames ----------------------------------------------------
    timesteps = pl.DataFrame({"t": np.arange(T, dtype=np.int64)})
    nodes = pl.DataFrame({"n": np.arange(N, dtype=np.int64)})
    et = edges.select("e").join(timesteps, how="cross")
    nt = nodes.join(timesteps, how="cross")

    # mapping frames for the node-balance constraint
    edges_dst_n = edges.select("e", n=pl.col("dst"))
    edges_src_n = edges.select("e", n=pl.col("src"))

    # ---- decision variable -----------------------------------------------
    flow = p.add_var("flow", dims=("e", "t"), index=et, lower=0.0)

    # ---- parameters ------------------------------------------------------
    cost = Param(("e",), edges.select("e", value=pl.col("cost")))

    cap_long = pl.DataFrame(
        {
            "e": np.repeat(np.arange(E, dtype=np.int64), T),
            "t": np.tile(np.arange(T, dtype=np.int64), E),
            "value": cap_arr.reshape(-1),
        }
    )
    cap = Param(("e", "t"), cap_long)

    demand_long = pl.DataFrame(
        {
            "n": np.repeat(np.arange(N, dtype=np.int64), T),
            "t": np.tile(np.arange(T, dtype=np.int64), N),
            "value": demand_arr.reshape(-1),
        }
    )
    demand = Param(("n", "t"), demand_long)

    # ---- objective -------------------------------------------------------
    p.set_objective(Sum(flow * cost), sense="min")

    # ---- capacity constraint --------------------------------------------
    p.add_cstr(
        "capacity",
        over=et,
        sense="<=",
        lhs_terms={"flow": flow},
        rhs_terms={"cap": cap},
    )

    # ---- node-balance ---------------------------------------------------
    # inflow:  for every (e, t) row of flow, remap e -> n via dst.
    # outflow: same via src.  Then Sum over e collapses to (n, t).
    inflow = Sum(Where(flow, edges_dst_n), over=("e",))
    outflow = Sum(Where(flow, edges_src_n), over=("e",))
    p.add_cstr(
        "node_balance",
        over=nt,
        sense="==",
        lhs_terms={"in": inflow, "neg_out": -outflow},
        rhs_terms={"demand": demand},
    )

    return p


def solve(model: Problem, time_limit: float | None = None) -> tuple[bool, float]:
    options = {"time_limit": float(time_limit)} if time_limit is not None else None
    sol = model.solve(options=options)
    return bool(sol.optimal), float(sol.obj)
