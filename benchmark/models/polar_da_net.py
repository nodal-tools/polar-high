"""polar-high network LP with declared ``dense_axes=("t",)``.

Same build as :mod:`models.polar_net`; only the ``Problem`` construction
differs. The network LP has Vars indexed by ``("e", "t")``; declaring
``dense_axes=("t",)`` promises every frame using ``t`` is row-sorted by
``(other_dims..., t)``.

The build below produces all such frames in that order:

* ``et`` via ``edges.select("e").join(timesteps, how="cross")`` — polars
  cross-join keeps left-then-right ordering, so rows come out (e=0,t=0),
  (e=0,t=1), ..., (e=1,t=0), ... = sorted by ``(e, t)``.
* ``cap_long`` / ``demand_long`` are built with ``np.repeat`` for the
  leading axis and ``np.tile`` for ``t`` — same sorted shape.
* ``nt`` is ``nodes.join(timesteps, how="cross")`` — sorted by ``(n, t)``.

So the dense-axes contract is satisfied without any extra sort.
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

    p = Problem(dense_axes=("t",))

    timesteps = pl.DataFrame({"t": np.arange(T, dtype=np.int64)})
    nodes = pl.DataFrame({"n": np.arange(N, dtype=np.int64)})
    et = edges.select("e").join(timesteps, how="cross")
    nt = nodes.join(timesteps, how="cross")

    edges_dst_n = edges.select("e", n=pl.col("dst"))
    edges_src_n = edges.select("e", n=pl.col("src"))

    flow = p.add_var("flow", dims=("e", "t"), index=et, lower=0.0)

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

    p.set_objective(Sum(flow * cost), sense="min")

    p.add_cstr(
        "capacity",
        over=et,
        sense="<=",
        lhs_terms={"flow": flow},
        rhs_terms={"cap": cap},
    )

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
