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

```python
sub_a = build_problem_a()
sub_b = build_problem_b()

coupling = CouplingSpec(
    entries=[
        CouplingEntry(0, "x", dim_tuples=[(1, "t1"), (2, "t1")], coef=+1.0),
        CouplingEntry(1, "x", dim_tuples=[(1, "t1"), (2, "t1")], coef=-1.0),
    ],
    rhs=0.0,
)

lp = LagrangianProblem(subproblems=[sub_a, sub_b],
                       couplings=[coupling])
sol = lp.solve(max_iters=50, step=1.0)

sol.total_objective       # best dual
sol.final_lambdas         # one array per coupling
sol.iteration_log         # per-iter diagnostics
```

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
