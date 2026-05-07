"""Worked example from docs/guide/lagrangian.md.

Two demand-floor subproblems linked by a consensus coupling: each must
supply its own local demand, and the coupling forces them to agree on
a single shared flow value.  Without coupling they choose independently;
with coupling they converge to max(demand_A, demand_B).

Section markers are consumed by MkDocs (pymdownx.snippets).
"""
# --8<-- [start:subproblems]
import polars as pl

import polar_high as ph
from polar_high import CouplingEntry, CouplingSpec, LagrangianProblem

# Sub-problem A: min flow_A  s.t.  flow_A >= 4,  0 <= flow_A <= 100
sub_a = ph.Problem()
idx_a = pl.DataFrame({"k": [0]})
x_a = sub_a.add_var("flow", "k", idx_a, lower=0.0, upper=100.0)
sub_a.add_cstr("local_demand", over=None, sense=">=",
               lhs_terms={"flow": ph.Sum(x_a, over=("k",))},
               rhs_terms={"d": 4.0})
sub_a.set_objective(ph.Sum(x_a), sense="min")

# Sub-problem B: min flow_B  s.t.  flow_B >= 2,  0 <= flow_B <= 100
sub_b = ph.Problem()
idx_b = pl.DataFrame({"k": [0]})
x_b = sub_b.add_var("flow", "k", idx_b, lower=0.0, upper=100.0)
sub_b.add_cstr("local_demand", over=None, sense=">=",
               lhs_terms={"flow": ph.Sum(x_b, over=("k",))},
               rhs_terms={"d": 2.0})
sub_b.set_objective(ph.Sum(x_b), sense="min")
# --8<-- [end:subproblems]

# --8<-- [start:coupling]
# Consensus coupling: flow_A == flow_B  (coefs +1 / -1, rhs 0)
# dim_tuples lists the variable cells that participate — one tuple per
# cell, matching the variable's dim values in order.
coupling = CouplingSpec(
    entries=[
        CouplingEntry(0, "flow", [(0,)], coef=+1.0),
        CouplingEntry(1, "flow", [(0,)], coef=-1.0),
    ],
    rhs=0.0,
)

lp = LagrangianProblem([sub_a, sub_b], [coupling])
sol = lp.solve(max_iters=200, tol=1e-9, step=0.5, min_iters=20)
print(f"best dual: {sol.total_objective}")   # 8.0
print(f"iterations: {sol.iterations}")
# --8<-- [end:coupling]
