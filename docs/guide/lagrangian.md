# Lagrangian decomposition

`LagrangianProblem` is a domain-agnostic dual-subgradient driver for
N independent LP subproblems linked by linear coupling constraints

$$
\sum_i \mathrm{coef}_i \cdot \mathrm{col}_i \;=\; \mathrm{rhs}
$$

## Building blocks

```python
from polar_high import (
    CouplingEntry, CouplingSpec, LagrangianProblem,
)
```

- **`CouplingEntry`** — one participant: `(subproblem_idx, var_name,
  dim_tuples, coef)`. One entry per subproblem that owns a coupled
  cell.
- **`CouplingSpec`** — one coupling family: a list of entries plus
  an optional `rhs` (default 0). The most common shape is the 2-entry
  *consensus* coupling `x_A == x_B` with coefs `+1`/`-1`.
- **`LagrangianProblem`** — assembles the subproblems and the
  couplings, runs the subgradient loop, returns a
  `LagrangianSolution`.

## Algorithm sketch

1. Bump each entry's column cost by `coef · λ` (relaxes the coupling
   residual into the objective).
2. Solve every subproblem (warm-started after iter 1 via
   `WarmProblem`).
3. Compute residual `Σ coef_i · x_i − rhs` per cell.
4. Subgradient step `λ ← λ + (step / √k) · residual`.
5. Tail-window primal averaging → fix-and-resolve for a feasible
   primal upper bound; report the *best dual* (max `Σ obj` across
   iters) as the tight lower bound.

## Minimal example

Two subproblems each with a local demand floor, linked by a consensus
coupling that forces them to agree on a common flow value.  The model
code is sourced from `tests/fixtures/lagrangian_example.py` and is
verified by `tests/test_lagrangian_example.py`.

Build the subproblems:

```python
--8<-- "tests/fixtures/lagrangian_example.py:subproblems"
```

Assemble and solve:

```python
--8<-- "tests/fixtures/lagrangian_example.py:coupling"
```

Inspect the result:

```python
sol.total_objective       # best dual bound (== 8.0 for this LP)
sol.final_lambdas         # one array per coupling
sol.iteration_log         # per-iter diagnostics
```

`dim_tuples` lists the variable cells that participate in the coupling —
one tuple per cell, with values matching the variable's dim columns in
declaration order.  Here `"flow"` has a single dim `"k"` with one row
`(0,)`, so `dim_tuples=[(0,)]`.

## When to use it

Lagrangian is worth the iterations when:

- Subproblems have block structure (each is much faster than the
  monolith);
- Coupling is small (few λ per cell) compared to total state;
- A loose primal bound is acceptable, or you can afford the tail
  primal-recovery resolve.

If the monolith already solves quickly, just solve the monolith.

## See also

- `tests/test_lagrangian.py` for closed-form parity tests on tiny
  synthetic LPs — the cleanest place to learn the API.
