"""Worked example from docs/guide/debugging.md.

The ``# --8<-- [start/end:...]`` markers are consumed by MkDocs
(pymdownx.snippets) to embed sections of this file verbatim into the
guide page.  The file is kept at module level (no enclosing function)
so that the included snippet lands without indentation in the rendered
docs.
"""
# --8<-- [start:model]
import polars as pl

import polar_high as ph

nodes = pl.DataFrame({"node": ["west"]})
hours = pl.DataFrame({"hour": [1, 2]})
index = nodes.join(hours, how="cross")   # (node, hour)

demand  = ph.Param(("node", "hour"), index.with_columns(value=pl.lit(100.0)))
cap     = ph.Param(("node", "hour"), index.with_columns(value=pl.lit(120.0)))
cost    = ph.Param(("node", "hour"), index.with_columns(value=pl.lit(1.0)))
penalty = ph.Param(("node", "hour"), index.with_columns(value=pl.lit(10.0)))

p = ph.Problem()
v_flow = p.add_var("v_flow", ("node", "hour"), index, lower=0.0, upper=1e6)
v_dump = p.add_var("v_dump", ("node", "hour"), index, lower=0.0, upper=1e6)

p.add_cstr(
    "balance",
    over=index,
    sense="==",
    lhs_terms={"flow": v_flow, "dump": -v_dump},
    rhs_terms={"demand": demand},
)
p.add_cstr(
    "cap",
    over=index,
    sense="<=",
    lhs_terms={"flow": v_flow},
    rhs_terms={"cap": cap},
)
p.set_objective(ph.Sum(v_flow * cost) + ph.Sum(v_dump * penalty), sense="min")
# --8<-- [end:model]

print("constraint families:", p.cstr_names())
print("balance row count:", p.cstr_row_count("balance"))
print(v_flow.frame)

sol = p.solve(keep_solver=True)
print(f"optimal: {sol.optimal}  obj: {sol.obj}")
print(sol.value("v_flow"))
print(sol.constraint_dual("balance"))
