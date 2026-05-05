# polar-high-opt

A Python library for building and solving large indexed linear and
mixed-integer programs.

polar-high-opt lets you describe an indexed optimization model
directly in [polars](https://pola.rs/) — variables and parameters
are DataFrames, expressions are joins and group-by-sums on those
frames, and the matrix is assembled into [HiGHS](https://highs.dev/)
without an intermediate matrix builder. The model can also be
exported in MPS form to any other LP solver. The kernel is
intentionally domain-free: it has no opinions about energy systems,
supply chains, or any specific application.

## Why this exists

If you write large indexed LPs, the bottleneck is not always the solver —
it can be the time spent translating sets and parameters into a sparse
matrix. polar-high-opt keeps every step in polars: index sets are
DataFrames, multiplications are joins, summations are group-bys, and
the final coordinate-format → compressed-sparse-column pass is a few `numpy` calls. On the typical
energy-system models that motivated the project, model build time
dropped by **~10x** vs a Pyomo equivalent and the LP shape is
identical.

You will probably like this engine if:

- you already think in **indexed sets** (the way Pyomo, JuMP, GNU MathProg,
  GAMS, AMPL users do), and
- you want a **Python-only** workflow with first-class
  **DataFrame ergonomics** (groupby, joins, `where`, lazy evaluation).

## Where to go next

- New here? [Install](installation.md) → [Quickstart](quickstart.md).
- Building a real model? Start in
  [Concepts](concepts/index.md) — the indexed-frame mental model is
  what you need before the API makes sense.
- Already have a Pyomo / linopy / JuMP model in mind? Jump to
  [Compare](compare/alternatives.md) for the design tradeoffs.
- Need raw API surface? [Reference](reference/api.md).

## Status

Version 1.0.0. The engine is heavily tested but correct functioning
is **not guaranteed** — see [About](about/about.md).

The first downstream user is the
[FlexTool](https://github.com/irena-flextool/flextool) energy-system
modelling toolkit. FlexTool's test fleet — comparing the new
polar-high-opt implementation against the earlier GNU MathProg one
across many scenarios — is the primary correctness validation for the
kernel.
