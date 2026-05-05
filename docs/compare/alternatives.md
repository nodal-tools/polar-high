# Alternatives

There are several established Python and Python-adjacent toolchains
for indexed mathematical programming. polar-high-opt's design choices
are best understood as reactions to specific limitations and as
extensions of patterns it inherited from earlier languages.

This page records the foundational decision of each alternative and
why polar-high-opt made a different choice. It then surveys two
cross-cutting features — *term-based equation formulation* and
*warm-starting / incremental modification* — where the comparison
does not fall neatly along tool boundaries.

## GNU MathProg (and its lineage to AMPL / GAMS)

**Foundational decision.** A small, expressive declarative language
for indexed mathematical programs: `set`, `param`, `var`, named
constraint blocks with explicit summations and quantifiers. AMPL and
GAMS share the same family resemblance.

**Why it inspired polar-high-opt.** MathProg is a *great language* —
the way you write a constraint reads almost exactly like the math.
The implementation, however, is showing its age: it is a
single-threaded interpreter with no in-process API, no incremental
modification, no built-in connection to modern data tooling, and a
text-only data path. polar-high-opt deliberately keeps the *spirit*
of MathProg-style indexed declarations while moving the data and
algebra into polars and the solver hand-off into a process-local
HiGHS instance.

## Pyomo

**Foundational decision.** A pure-Python algebraic-modelling layer.
Variables, parameters, and constraints are Python objects; expressions
are built by traversing those objects with operator overloading.

**Where polar-high-opt diverges.** Building large indexed programs
in Pyomo is slow because every coefficient is mediated by a Python
object. polar-high-opt keeps the entire build in **polars frames**:
multiplications are joins, summations are group-bys, and the only
Python in the hot path is the loop that iterates *constraint families*
(not coefficients). On the energy-system models that motivated this
project, build time dropped roughly an order of magnitude.

## JuMP (Julia)

**Foundational decision.** A first-class algebraic-modelling layer
written in Julia. Macros generate efficient Julia code per model,
so the build is fast even for large models.

**Where polar-high-opt diverges.** JuMP is excellent — the speed
problem of Pyomo is not present in JuMP. The hurdle is the
**Julia toolchain**: a separate language, separate package manager,
separate test/CI ecosystem, and an ahead-of-time compilation step
that complicates short-lived workflows. polar-high-opt keeps the
modeller in **plain Python** with first-class polars ergonomics, at
the cost of being a less mature modelling language than JuMP.

## linopy

**Foundational decision.** An xarray-based indexed-modelling layer.
Variables and parameters are `xarray.DataArray`s; broadcasting follows
xarray semantics.

**Where polar-high-opt diverges.** linopy and polar-high-opt make the
same architectural bet — *push the algebra into a fast columnar
backend* — but choose different backends. **xarray vs polars** is the
main axis:

- **polars** is built on Arrow, has a true lazy/columnar query
  optimizer, runs multi-threaded by default, and treats joins as
  first-class operations.
- **xarray** is built on numpy, is single-threaded by default,
  and broadcasts via aligned coordinate axes (great for dense
  numeric arrays, awkward for sparse irregular index sets that
  energy-system models tend to produce).

For the indexed-but-sparse models polar-high-opt targets, polars'
join-based semantics map more directly onto how the coefficient
matrix is actually built.

## pyoptinterface

**Foundational decision.** A thin solver-agnostic Python layer with
solver-specific backends (Gurobi, COPT, HiGHS, …).

**Where polar-high-opt diverges.** pyoptinterface stays close to the
solver API, leaving the modelling layer to the user. polar-high-opt
is the **opposite end** of that tradeoff: it commits to an opinionated
indexed-frame modelling layer, builds the matrix through HiGHS by
default, and supports MPS export for any other LP solver.

## gurobipy

**Foundational decision.** Python bindings for the Gurobi solver.
The modelling surface is shallow; performance comes from the
underlying C++ solver and from a tightly integrated build path.

**Where polar-high-opt diverges.** gurobipy is **bound to Gurobi**,
a commercial solver. Even with academic / free tiers, this is a
licensing footprint that polar-high-opt avoids — the bundled solver
is **HiGHS** (open source, Apache-licensed via `highspy`). For users
who do want a commercial solver, polar-high-opt still produces a
standard MPS file (`Solution.highs.writeModel("model.mps")`), so
Gurobi or any other LP solver can read the model.

## Cross-cutting: term-based equation formulation

Most of the alternatives above accept a constraint as a single
algebraic expression. The expression compiles down to a sparse row,
and the original named contributions are gone — when a parity check
fails, you see "row drift", not "the `inflow` term is wrong".

polar-high-opt accepts both forms:

```python
# expression form (familiar from Pyomo / JuMP / linopy)
p.add_cstr("cap", v <= 5.0)

# term form (labelled contributions preserved through to the LP)
p.add_cstr(
    name="balance",
    over=node_time_index,
    sense="==",
    lhs_terms={
        "inflow":  Sum(v_in,  over=("p",)),
        "outflow": Sum(v_out, over=("p",)),
        "delta":   v_state - Lag(v_state, dt, "t", "t_prev"),
    },
    rhs_terms={"demand": p_demand},
)
```

Whether you prefer one form over the other is largely a matter of
taste. The term form is useful when:

- a constraint is naturally a *sum of named contributions* (mass
  balance, energy balance, capacity-margin) and you want diagnostics
  to point at the right term when something disagrees;
- you anticipate term-level audits against a reference implementation
  (which is how the FlexTool migration off MathProg was validated);
- the constraint family is built up from heterogeneous sources (a
  storage block, a profile term, a slack term) and one combined
  expression would obscure where each contribution comes from.

The expression form is shorter and reads more like math when the
constraint is a single conceptual statement.

## Cross-cutting: warm-starting and incremental modification

Rolling-horizon, parameter-sweep, and decomposition workflows benefit
from re-solving with small updates instead of rebuilding. Where each
alternative stands today:

- **JuMP** and **gurobipy** support rich in-place modification —
  set RHS, set coefficient, fix variable, change bound, all on a
  live model.
- **Pyomo** offers persistent solver interfaces (e.g.
  `gurobi_persistent`, `highs_persistent`) that translate Pyomo-side
  edits into the solver's incremental APIs.
- **pyoptinterface** exposes the underlying solver's incremental API
  directly.
- **linopy** rebuilds the model per solve.
- **GNU MathProg** has no incremental API.

polar-high-opt's [`WarmProblem`](../guide/warm-starting.md) provides
the same family of incremental updates (RHS, objective coefficient,
variable bound, single-coefficient edit) on top of HiGHS, plus a
*Param-tracked auto-update* path: declare which Params are mutable,
then push a new Param value and the engine recomputes every
coefficient that traces back to that Param chain. The point of
difference is the integration with the polars-side build path —
when you update a Param, the kernel knows which constraint rows
depend on it without needing the user to bookkeep that mapping.

## Summary

| Alternative | Their bet | Our bet |
|---|---|---|
| GNU MathProg | Declarative indexed language | Same spirit, modern data path |
| Pyomo | Pure-Python flexibility | Polars columnar build path |
| JuMP | Julia compiler & macros | Plain Python ergonomics |
| linopy | xarray broadcasting | Polars joins + lazy plan |
| pyoptinterface | Solver-agnostic thin layer | Opinionated modelling layer + MPS export |
| gurobipy | Commercial-solver tight loop | Bundled HiGHS, MPS to anything |

Healthy technology choices come with tradeoffs; the tradeoffs above
are the ones polar-high-opt accepts to make large indexed models
build fast and read clearly in Python.
