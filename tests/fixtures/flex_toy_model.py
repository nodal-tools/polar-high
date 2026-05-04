"""Stage-1 flextool-flavored dispatch model.

Mirrors the *shape* of flextool's simplified dispatch case but on
synthetic data the user can hand-verify.  Validates that the engine
can express:

  * Methodful index sets (``process_source_sink_eff`` vs ``_noEff``)
  * Multi-flow node balance (sink term aggregates over (p, source))
  * Efficiency-weighted commodity cost in the objective

What we DON'T model in stage 1: storage, investment, online
status, ramps, reserves, blocks/overlap, branching, inflation,
period weights — every one of these is hardcoded to "trivial"
(zero, one, or absent).  Stage 2+ adds them.
"""

from polar_high_opt import Sum, Where


def build_flex_toy(p, d):
    """Add the dispatch model to ``p`` using data bag ``d``."""

    # -- decision variables ------------------------------------------------
    # flow per (p, source, sink, t) — in unitsize units
    v_flow = p.add_var(
        "v_flow",
        ("p", "source", "sink", "t"),
        d.pss_t,
        lower = 0.0,
    )
    # state slacks per nodeBalance node and timestep
    vq_state_up   = p.add_var("vq_state_up",   ("n", "t"), d.nodeBalance_t, lower=0.0)
    vq_state_down = p.add_var("vq_state_down", ("n", "t"), d.nodeBalance_t, lower=0.0)

    # ---------------------------------------------------------------------
    # 1. maxToSink — capacity bound on every flow
    #     v_flow[p, source, sink, t] * unitsize[p]   <=   cap[p]
    p.add_cstr(
        "maxToSink",
        over      = d.pss_t,
        sense     = "<=",
        lhs_terms = {"flow": v_flow * d.unitsize},
        rhs_terms = {"capacity": d.cap},
    )

    # ---------------------------------------------------------------------
    # 2. profile_flow_upper — wind availability per timestep
    #     v_flow[WIND, WIND, elec, t] * unitsize[WIND]   <=   wind_avail[t]
    p.add_cstr(
        "profile_flow_upper_wind",
        over      = d.wind_pss_t,
        sense     = "<=",
        lhs_terms = {
            "flow": Where(v_flow * d.unitsize, d.process_source_sink_noEff),
        },
        rhs_terms = {"wind_avail": d.wind_avail},
    )

    # ---------------------------------------------------------------------
    # 3. nodeBalance_eq — the showcase compound constraint.
    #     For each (n, t) in nodeBalance × timesteps:
    #
    #         Σ_{(p,src,n) ∈ pss} v_flow[p,src,n,t] · unitsize[p]      (sink)
    #       + vq_state_up[n,t]  −  vq_state_down[n,t]                  (slack)
    #         ==
    #         demand[n, t]
    #
    # Source-side terms (when n is a source for some flow) are zero in
    # this toy because elec is never a source.  In a real model they'd
    # appear as additional lhs_terms, with efficiency on the eff partition.
    p.add_cstr(
        "nodeBalance_eq",
        over      = d.nodeBalance_t,
        sense     = "==",
        lhs_terms = {
            "sink_flow":  Sum(Where(v_flow * d.unitsize, d.flow_to_n),
                              over=("p", "source", "sink")),
            "slack_up":    vq_state_up,
            "slack_down": -vq_state_down,
        },
        rhs_terms = {"demand": d.demand},
    )

    # -- objective ---------------------------------------------------------
    # Commodity buy:  for every (p, source=n_c, sink) in eff-flows from a
    # commodity node, the source-side draw is v_flow * unitsize * slope.
    # Cost = price[c, t] · draw · step_duration[t], summed.
    commodity_cost = Sum(
        Where(v_flow * d.unitsize * d.slope, d.flow_from_commodity_eff)
        * d.gas_price * d.step_duration,
    )

    slack_cost = (
        Sum(vq_state_up   * d.slack_pen)
      + Sum(vq_state_down * d.slack_pen)
    )

    p.set_objective(commodity_cost + slack_cost, sense="min")
