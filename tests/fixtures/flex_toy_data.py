"""Stage-1 flextool-flavored data.

A miniature dispatch case that exercises the topology features we
need before tackling real flextool input:

  * Two nodes: ``elec`` (in nodeBalance — gets a real balance
    constraint) and ``gas`` (a commodity node — supplied by the gas
    commodity at price 5 €/MWh, no balance row).
  * Two unit processes:
      - ``GT`` — gas turbine, source=``gas``, sink=``elec``,
        constant_efficiency 0.5 → ``slope = 1/0.5 = 2``,
        in ``process_source_sink_eff``.  Capacity 100 MW.
      - ``WIND`` — wind turbine, source=``WIND`` (self), sink=``elec``,
        in ``process_source_sink_noEff`` (no source-side balance term).
        Capacity 100 MW, but availability profile per timestep limits
        actual output.
  * One commodity, ``GAS_COM``, linked to the ``gas`` node via
    ``commodity_node``; price 5 €/MWh (per MWh of gas drawn).
  * Four timesteps with varying ``elec`` demand and wind availability.

Hand-verified optimum (default data):
  GT  output:  [10, 60,  0, 60]   — 130 MWh-elec total
  WIND output: [50, 30, 70, 20]   — 170 MWh-elec total
  Cost: 5 €/MWh-gas × 2 MWh-gas/MWh-elec × 130 MWh-elec = 1300 €
"""

from dataclasses import dataclass

import polars as pl

from polar_high_opt import Param


@dataclass
class FlexToyData:
    # ---- index frames ----------------------------------------------------
    nodes: pl.DataFrame  # columns: n
    nodeBalance: pl.DataFrame  # columns: n   (subset of nodes)
    processes: pl.DataFrame  # columns: p
    commodities: pl.DataFrame  # columns: c
    commodity_node: pl.DataFrame  # columns: c, n
    timesteps: pl.DataFrame  # columns: t

    # process topology
    process_source_sink: pl.DataFrame  # columns: p, source, sink
    process_source_sink_eff: pl.DataFrame  # subset where efficiency applies
    process_source_sink_noEff: pl.DataFrame  # subset where it doesn't
    pss_t: pl.DataFrame  # process_source_sink × timesteps
    nodeBalance_t: pl.DataFrame  # nodeBalance × timesteps (constraint axes)
    wind_pss_t: pl.DataFrame  # WIND-only × timesteps (profile constraint axes)

    # mapping frames (used as Where(...) targets)
    flow_to_n: pl.DataFrame
    """Columns: p, source, sink, n   (n = sink) — for "incoming flow to node n".
    Used in nodeBalance's sink term."""

    flow_from_commodity_eff: pl.DataFrame
    """Columns: p, source, sink, c   (where source is a commodity node and
    process is in process_source_sink_eff) — for the commodity buy term in
    the objective."""

    # ---- parameters ------------------------------------------------------
    unitsize: Param  # dims=(p,)
    cap: Param  # dims=(p,)        — capacity (MW)
    slope: Param  # dims=(p, t)      — 1/efficiency for eff procs, 1.0 elsewhere
    wind_avail: Param  # dims=(t,)        — wind upper limit per timestep
    demand: Param  # dims=(n, t)      — elec demand
    gas_price: Param  # dims=(c, t)      — commodity price
    step_duration: Param  # dims=(t,)
    slack_pen: Param  # scalar


def make_flex_toy_data(
    *,
    demand_values: tuple[float, float, float, float] = (60.0, 90.0, 70.0, 80.0),
    wind_avail_values: tuple[float, float, float, float] = (50.0, 30.0, 80.0, 20.0),
    gas_price_value: float = 5.0,
    gt_efficiency: float = 0.5,
    gt_capacity: float = 100.0,
    wind_capacity: float = 100.0,
    slack_pen_value: float = 1000.0,
) -> FlexToyData:
    ts = ["t1", "t2", "t3", "t4"]

    nodes = pl.DataFrame({"n": ["elec", "gas"]})
    nodeBalance = pl.DataFrame({"n": ["elec"]})
    processes = pl.DataFrame({"p": ["GT", "WIND"]})
    commodities = pl.DataFrame({"c": ["GAS_COM"]})
    commodity_node = pl.DataFrame({"c": ["GAS_COM"], "n": ["gas"]})
    timesteps = pl.DataFrame({"t": ts})

    process_source_sink = pl.DataFrame(
        {
            "p": ["GT", "WIND"],
            "source": ["gas", "WIND"],
            "sink": ["elec", "elec"],
        }
    )
    process_source_sink_eff = pl.DataFrame({"p": ["GT"], "source": ["gas"], "sink": ["elec"]})
    process_source_sink_noEff = pl.DataFrame({"p": ["WIND"], "source": ["WIND"], "sink": ["elec"]})

    pss_t = process_source_sink.join(timesteps, how="cross")
    nodeBalance_t = nodeBalance.join(timesteps, how="cross")
    wind_pss_t = process_source_sink_noEff.join(timesteps, how="cross")

    # incoming flow:  for every (p, source, sink), expose n = sink
    flow_to_n = process_source_sink.with_columns(n=pl.col("sink"))

    # commodity-buy mapping: every eff flow whose source is a commodity node
    flow_from_commodity_eff = process_source_sink_eff.join(
        commodity_node, left_on="source", right_on="n", how="inner"
    ).select("p", "source", "sink", "c")

    # parameters
    slope_value = 1.0 / gt_efficiency
    slope = Param(
        ("p", "t"),
        pl.DataFrame(
            {
                "p": ["GT"] * 4 + ["WIND"] * 4,
                "t": ts * 2,
                "value": [slope_value] * 4 + [1.0] * 4,
            }
        ),
    )

    return FlexToyData(
        nodes=nodes,
        nodeBalance=nodeBalance,
        processes=processes,
        commodities=commodities,
        commodity_node=commodity_node,
        timesteps=timesteps,
        process_source_sink=process_source_sink,
        process_source_sink_eff=process_source_sink_eff,
        process_source_sink_noEff=process_source_sink_noEff,
        pss_t=pss_t,
        nodeBalance_t=nodeBalance_t,
        wind_pss_t=wind_pss_t,
        flow_to_n=flow_to_n,
        flow_from_commodity_eff=flow_from_commodity_eff,
        unitsize=Param(("p",), pl.DataFrame({"p": ["GT", "WIND"], "value": [1.0, 1.0]})),
        cap=Param(
            ("p",), pl.DataFrame({"p": ["GT", "WIND"], "value": [gt_capacity, wind_capacity]})
        ),
        slope=slope,
        wind_avail=Param(("t",), pl.DataFrame({"t": ts, "value": list(wind_avail_values)})),
        demand=Param(
            ("n", "t"),
            pl.DataFrame(
                {
                    "n": ["elec"] * 4,
                    "t": ts,
                    "value": list(demand_values),
                }
            ),
        ),
        gas_price=Param(
            ("c", "t"),
            pl.DataFrame(
                {
                    "c": ["GAS_COM"] * 4,
                    "t": ts,
                    "value": [gas_price_value] * 4,
                }
            ),
        ),
        step_duration=Param(("t",), pl.DataFrame({"t": ts, "value": [1.0] * 4})),
        slack_pen=Param.scalar(slack_pen_value),
    )
