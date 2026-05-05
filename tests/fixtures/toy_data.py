"""Toy dataset for the dispatch model.

A dataset is just a bag of polars frames + Param objects.  Same
shape as the real flextool input layer should produce: index frames
for set membership, Param tables for parameter lookups.
"""

from dataclasses import dataclass

import polars as pl

from polar_high_opt import Param


@dataclass
class ToyData:
    # index frames -- the "sets"
    processes: pl.DataFrame
    timesteps: pl.DataFrame
    pt: pl.DataFrame  # processes × timesteps (cross product)
    gas_only: pl.DataFrame
    wind_only: pl.DataFrame
    wind_t: pl.DataFrame  # wind_only × timesteps

    # parameters
    cap: Param  # dims=(p,)
    avail: Param  # dims=(t,) — wind availability
    demand: Param  # dims=(t,)
    cost: Param  # dims=(p,)
    co2_factor: Param  # scalar
    co2_cap_value: Param  # scalar
    slack_pen: Param  # scalar
    max_total: Param  # dims=(p,)


def make_toy_data() -> ToyData:
    processes = pl.DataFrame({"p": ["gas", "wind"]})
    timesteps = pl.DataFrame({"t": ["t1", "t2", "t3", "t4"]})
    pt = processes.join(timesteps, how="cross")
    gas_only = pl.DataFrame({"p": ["gas"]})
    wind_only = pl.DataFrame({"p": ["wind"]})
    wind_t = wind_only.join(timesteps, how="cross")

    return ToyData(
        processes=processes,
        timesteps=timesteps,
        pt=pt,
        gas_only=gas_only,
        wind_only=wind_only,
        wind_t=wind_t,
        cap=Param(("p",), pl.DataFrame({"p": ["gas", "wind"], "value": [100.0, 100.0]})),
        avail=Param(
            ("t",), pl.DataFrame({"t": ["t1", "t2", "t3", "t4"], "value": [50.0, 30.0, 80.0, 20.0]})
        ),
        demand=Param(
            ("t",), pl.DataFrame({"t": ["t1", "t2", "t3", "t4"], "value": [60.0, 90.0, 70.0, 80.0]})
        ),
        cost=Param(("p",), pl.DataFrame({"p": ["gas", "wind"], "value": [50.0, 0.0]})),
        co2_factor=Param.scalar(0.4),
        co2_cap_value=Param.scalar(200.0),
        slack_pen=Param.scalar(1000.0),
        max_total=Param(("p",), pl.DataFrame({"p": ["gas", "wind"], "value": [150.0, 200.0]})),
    )
