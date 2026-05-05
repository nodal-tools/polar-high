"""Shared data generator for the sparse network-flow benchmark.

A single function ``generate(N, T=24, seed=42)`` returns the index/parameter
tables used by all four ``*_net`` model files.  Importing the same generator
guarantees identical numbers — and hence identical LPs — across polar-high,
linopy and Pyomo.

Topology:
    N nodes (0..N-1), E = 5*N edges. The first N edges form a Hamiltonian
    cycle (n → (n+1) mod N) which guarantees structural feasibility for any
    demand vector — every node has at least one incoming and one outgoing
    edge. The remaining 4*N edges are drawn (src, dst) uniformly with seed 42
    (self-loops allowed; duplicate pairs allowed — each edge has its own
    column).

Parameters:
    cost[e]      ~ uniform(1, 10)
    cap[e, t]    ~ uniform(0.5, 2.0)
    demand[n, t] ~ uniform(-1, 1) per node, then mean-subtracted per t so
                   it sums to 0 across nodes (feasibility).
"""

from __future__ import annotations

import numpy as np
import polars as pl


def generate(N: int, T: int = 168, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)

    # Hamiltonian cycle guarantees every node has >=1 incoming and >=1
    # outgoing edge -> any feasible demand can be routed.
    cycle_src = np.arange(N, dtype=np.int64)
    cycle_dst = (cycle_src + 1) % N
    rand_src = rng.integers(0, N, size=4 * N, dtype=np.int64)
    rand_dst = rng.integers(0, N, size=4 * N, dtype=np.int64)
    src = np.concatenate([cycle_src, rand_src])
    dst = np.concatenate([cycle_dst, rand_dst])
    E = src.size  # 5 * N
    cost = rng.uniform(1.0, 10.0, size=E)

    edges = pl.DataFrame(
        {
            "e": np.arange(E, dtype=np.int64),
            "src": src,
            "dst": dst,
            "cost": cost,
        }
    )

    # Random edges get small caps; the N cycle edges get a large cap so
    # they can absorb whatever the random topology can't route. Without
    # this, even a Hamiltonian cycle is infeasible at large N because
    # demand routing along the cycle scales with N.
    cap = rng.uniform(0.5, 2.0, size=(E, T))
    cap[:N, :] = float(N)

    demand = rng.uniform(-1.0, 1.0, size=(N, T))
    # subtract per-t mean so each column sums to 0 (LP feasibility)
    demand = demand - demand.mean(axis=0, keepdims=True)

    return {"edges": edges, "cap": cap, "demand": demand, "N": N, "T": T, "E": E}
