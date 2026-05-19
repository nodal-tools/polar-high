"""Worked example from docs/quickstart.md and README.md.

Section markers are consumed by MkDocs (pymdownx.snippets) to embed
this code verbatim in the guide pages.  Module-level placement keeps
the snippet at zero indentation in the rendered docs.
"""

# --8<-- [start:model]
import polars as pl

from polar_high import Param, Problem, Sum

p = Problem()

# Index sets — declared once, reused below
unit_index = pl.DataFrame({"unit": ["wind", "coal"]})
time_index = pl.DataFrame({"hour": [1, 2, 3]})

# Decision variable v_production[unit, hour] >= 0
composite_index = unit_index.join(time_index, how="cross")
v_production = p.add_var(
    "v_production",
    dims=("unit", "hour"),
    index=composite_index,
    lower=0.0,
)

# Operating cost per unit
cost = Param(
    ("unit",),
    pl.DataFrame({"unit": ["wind", "coal"], "value": [2.0, 8.0]}),
)

# Available capacity per unit per hour — built per-unit, then concatenated
cap_wind = time_index.with_columns(
    pl.lit("wind").alias("unit"),
    pl.Series("value", [3.0, 1.0, 4.0]),  # wind drops in hour 2
)
cap_coal = time_index.with_columns(
    pl.lit("coal").alias("unit"),
    pl.Series("value", [10.0, 10.0, 10.0]),
)
cap = Param(
    ("unit", "hour"),
    pl.concat([cap_wind, cap_coal]).select("unit", "hour", "value"),
)

# Demand per hour
demand = Param(
    ("hour",),
    time_index.with_columns(pl.Series("value", [5.0, 6.0, 4.0])),
)

# Minimise total cost
p.set_objective(cost * v_production, sense="min")

# v_production[unit, hour] <= cap[unit, hour]
p.add_cstr(
    "capacity",
    over=composite_index,
    lhs_terms={"production": v_production},
    sense="<=",
    rhs_terms={"cap": cap},
)

# Σ_unit v_production[unit, hour] == demand[hour]
p.add_cstr(
    "demand_balance",
    over=time_index,
    lhs_terms={"production": Sum(v_production, over=("unit",))},
    sense="==",
    rhs_terms={"demand": demand},
)

sol = p.solve()
print(f"objective: {sol.obj}")  # 72.0
print(sol.value("v_production"))
# --8<-- [end:model]
