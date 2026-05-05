"""Toy dispatch model — five constraints + objective.

Pure constraint code: takes an LP ``Problem`` and a ``ToyData`` bag,
adds variables/constraints/objective.  Every constraint has the same
shape:

    p.add_cstr(name,
        over      = <index frame | None>,
        sense     = "<=" | ">=" | "==",
        lhs_terms = {label: term, ...},
        rhs_terms = {label: term, ...},   # optional, defaults to {}
    )

A term is either a ``Var``/``Expr`` (variable contribution) or a
``Param``/``int``/``float`` (constant contribution).  The engine
sorts them out — labels are used in row names and diagnostics.
"""

from polar_high_opt import Sum, Where


def build_dispatch(p, d):
    """Add the dispatch model to ``p`` using data bag ``d``."""

    # -- decision variables ------------------------------------------------
    v_flow = p.add_var("v_flow", ("p", "t"), d.pt, lower=0.0)
    vq_up = p.add_var("vq_up", ("t",), d.timesteps, lower=0.0)
    vq_down = p.add_var("vq_down", ("t",), d.timesteps, lower=0.0)

    # ---------------------------------------------------------------------
    # 1. Process capacity:  v_flow[p, t]  <=  cap[p]
    p.add_cstr(
        "max_flow",
        over=d.pt,
        sense="<=",
        lhs_terms={"flow": v_flow},
        rhs_terms={"capacity": d.cap},
    )

    # ---------------------------------------------------------------------
    # 2. Wind availability:  v_flow["wind", t]  <=  avail[t]
    p.add_cstr(
        "wind_available",
        over=d.wind_t,
        sense="<=",
        lhs_terms={"wind_flow": Where(v_flow, d.wind_only)},
        rhs_terms={"avail": d.avail},
    )

    # ---------------------------------------------------------------------
    # 3. Node balance:
    #     Σ_p v_flow[p, t]  +  vq_up[t]  -  vq_down[t]   ==   demand[t]
    p.add_cstr(
        "node_balance",
        over=d.timesteps,
        sense="==",
        lhs_terms={
            "generation": Sum(v_flow, over="p"),
            "slack_up": vq_up,
            "slack_down": -vq_down,
        },
        rhs_terms={"demand": d.demand},
    )

    # ---------------------------------------------------------------------
    # 4. CO2 cap (scalar — over=None):
    #     Σ_t v_flow["gas", t] · co2_factor   <=   co2_cap
    p.add_cstr(
        "co2_cap",
        sense="<=",
        lhs_terms={"gas_emissions": Sum(v_flow * d.co2_factor, where=d.gas_only)},
        rhs_terms={"co2_cap": d.co2_cap_value},
    )

    # ---------------------------------------------------------------------
    # 5. Per-process total cap:  Σ_t v_flow[p, t]  <=  max_total[p]
    p.add_cstr(
        "period_total",
        over=d.processes,
        sense="<=",
        lhs_terms={"total": Sum(v_flow, over="t")},
        rhs_terms={"max_total": d.max_total},
    )

    # -- objective ---------------------------------------------------------
    p.set_objective(
        Sum(v_flow * d.cost) + Sum(vq_up * d.slack_pen) + Sum(vq_down * d.slack_pen),
        sense="min",
    )
