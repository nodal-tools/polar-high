"""Worked example from docs/quickstart.md and README.md.

Section markers are consumed by MkDocs (pymdownx.snippets) to embed
this code verbatim in the guide pages.  Module-level placement keeps
the snippet at zero indentation in the rendered docs.
"""

# --8<-- [start:model]
import polars as pl

from polar_high import Param, Problem, Sum

p = Problem()

# Decision variable v_production[unit, hour] >= 0
v_idx = pl.DataFrame(
    {
        "unit": ["wind", "wind", "wind", "coal", "coal", "coal"],
        "hour": [1, 2, 3, 1, 2, 3],
    }
)
v_production = p.add_var(
    "v_production",
    dims=("unit", "hour"),
    index=v_idx,
    lower=0.0,
)

# Operating cost per unit
cost = Param(
    ("unit",),
    pl.DataFrame({"unit": ["wind", "coal"], "value": [2.0, 8.0]}),
)

# Available capacity per unit per hour (wind drops in hour 2)
cap = Param(
    ("unit", "hour"),
    pl.DataFrame(
        {
            "unit": ["wind", "wind", "wind", "coal", "coal", "coal"],
            "hour": [1, 2, 3, 1, 2, 3],
            "value": [3.0, 1.0, 4.0, 10.0, 10.0, 10.0],
        }
    ),
)

# Demand per hour
demand = Param(
    ("hour",),
    pl.DataFrame({"hour": [1, 2, 3], "value": [5.0, 6.0, 4.0]}),
)

# Minimise total cost
p.set_objective(cost * v_production, sense="min")

# v_production[unit, hour] <= cap[unit, hour]
p.add_cstr(
    "capacity",
    over=v_idx,
    sense="<=",
    lhs_terms={"production": v_production},
    rhs_terms={"cap": cap},
)

# Σ_unit v_production[unit, hour] == demand[hour]
hour_idx = v_idx.select("hour").unique().sort("hour")
p.add_cstr(
    "demand_balance",
    over=hour_idx,
    sense="==",
    lhs_terms={"production": Sum(v_production, over=("unit",))},
    rhs_terms={"demand": demand},
)

sol = p.solve()
print(f"objective: {sol.obj}")  # 72.0
print(sol.value("v_production"))
# --8<-- [end:model]
