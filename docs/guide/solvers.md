# Solvers

`polar-high` is solver-agnostic. HiGHS (via `highspy`) is the
no-license-needed default and ships in the base install. Four
commercial solvers — **Gurobi**, **CPLEX**, **FICO Xpress**, and
**COPT** — are supported on a **bring-your-own-license** basis: you
install the vendor's Python wrapper and provide your own license,
`polar-high` translates the model and dispatches the solve. We do
not ship solver binaries, generate license files, or store credentials.
See [the multi-solver handoff spec](https://github.com/nodal-tools/polar-high/blob/main/specs/polar-high-multi-solver-handoff.md)
for the architectural rationale and the
["What NOT to do"](https://github.com/nodal-tools/polar-high/blob/main/specs/polar-high-multi-solver-handoff.md#what-not-to-do)
constraints baked into the design.

## Detecting what is installed

`polar_high.solvers.available_solvers` is populated at import time by
attempting each vendor wrapper's Python import. It tells you which
wrappers are **installed**, not whether their license check will pass
— the license probe only fires when an env/model is constructed
inside the adapter.

```python
from polar_high.solvers import available_solvers

print(available_solvers)
# Clean install:   ['highs']
# With Gurobi:     ['gurobi', 'highs']
# Full developer:  ['gurobi', 'cplex', 'xpress', 'copt', 'highs']
```

`solve(...)` picks the first entry as the default if you don't pass
`solver_name`:

```python
from polar_high.solvers import solve

result = solve(problem)                       # first available
result = solve(problem, solver_name="highs")  # explicit
```

## Installing a commercial solver

Each commercial adapter has a matching optional extra that pulls only
the **Python wrapper** (plus `scipy` where vectorized matrix loads
need it). The solver runtime and its license remain your
responsibility.

### Gurobi

```bash
pip install 'polar-high[gurobi]'
```

License discovery follows Gurobi's own rules — `$HOME/gurobi.lic`,
`/opt/gurobi/`, or the `GRB_LICENSE_FILE` env var. For WLS, Compute
Server, or Token Server contexts, build a `gurobipy.Env` and pass it
through:

```python
import gurobipy as gp
from polar_high.solvers import solve

env = gp.Env(empty=True)
env.setParam("WLSACCESSID", "...")
env.setParam("WLSSECRET", "...")
env.setParam("LICENSEID", 12345)
env.start()

result = solve(problem, solver_name="gurobi", env=env)
```

Vendor docs:
[Gurobi licensing](https://www.gurobi.com/documentation/current/refman/grb_license_files.html),
[gurobipy Env](https://www.gurobi.com/documentation/current/refman/py_env.html).

### CPLEX

```bash
pip install 'polar-high[cplex]'
```

License discovery uses CPLEX's own mechanisms — `ILOG_LICENSE_FILE`
or `CPLEX_STUDIO_LICENSE`, an `access.ilm` adjacent to the install,
or IBM's ILM subscription credentials. The `env=` parameter is
reserved for future CPLEX subscription contexts.

Vendor docs:
[IBM CPLEX licensing](https://www.ibm.com/docs/en/icos/22.1.2?topic=optimization-cplex-license-files),
[CPLEX Python API](https://www.ibm.com/docs/en/icos/22.1.2?topic=cplex-python-api).

### Xpress

```bash
pip install 'polar-high[xpress]'
```

The FICO Xpress wheel ships with a **free community license** built
in — small models work out of the box without any extra setup. For a
full / commercial license, point `xpress.init(...)` at your
`xpauth.xpr` file, or set the `XPRESS` / `XPAUTH_PATH` env var,
before calling `solve(...)`.

Vendor docs:
[FICO Xpress community](https://www.fico.com/fico-xpress-optimization/docs/latest/installguide/dhtml/chap2_sec_sectiondownload.html),
[Xpress Python](https://www.fico.com/fico-xpress-optimization/docs/latest/solver/optimizer/python/HTML/).

### COPT

```bash
pip install 'polar-high[copt]'
```

License discovery follows COPT's own rules — `copt.lic` in the
install dir, `COPT_LICENSE_DIR`, or the bundled non-commercial
fallback for small models. For server or subscription deployments,
build a `coptpy.Envr` and pass it via `env=`.

Vendor docs:
[COPT manual](https://www.shanshu.ai/copt).

!!! warning "COPT and HiGHS in the same Python process"

    COPT 8.x ships native code (`coptpy.coptcore`) that conflicts with
    `highspy` once both are loaded into one interpreter: `Highs.run()`
    segfaults after `coptpy` is imported. The conflict is on the solver
    side only — `highspy.Highs.writeModel` is unaffected.

    Since `polar-high` always imports `highspy` (it is the matrix
    builder), the COPT adapter cannot use its in-memory Python path
    when called from a `polar_high` process. It **transparently
    auto-routes** through the file-based `copt_cmd` CLI in that case:
    `highspy` writes the MPS, then a subprocess invokes `copt_cmd`
    (no in-process `coptpy` load).

    This requires the standalone `copt_cmd` binary on `PATH`. The
    `coptpy` pip wheel does **not** ship it; install it from the
    [full COPT distribution](https://www.shanshu.ai/copt). If
    `copt_cmd` is missing the adapter raises `SolverNotAvailableError`
    with a pointer to this section. As an alternative, invoke COPT
    from a Python process that does not import `polar_high` — for
    example, write the MPS via [`Problem.write_mps`](performance.md#writing-mps-without-highs)
    and call `coptpy` directly.

    No other commercial solver exhibits this conflict — Gurobi,
    CPLEX, and Xpress coexist with HiGHS in-process without issue.

## File-based fallback: `io_api='mps'`

If you have a solver binary on `PATH` (e.g. an IT-managed install of
`gurobi_cl` or `cplex`) but **not** the matching Python wrapper —
common on locked-down corporate Python, mismatched Python versions,
or wheels missing for your platform — drop down to the MPS file
path:

```python
from polar_high.solvers import IOMode, solve

result = solve(problem, solver_name="gurobi", io_api=IOMode.MPS)
```

`polar-high` writes a temporary MPS file via `highspy`, invokes the
solver's CLI binary on it, parses the resulting `.sol` file, and
returns the same `SolverResult` shape as the direct path. This is
slower on large models (round-trip through a file) and currently
LP-only on the writer side, but it covers cases the direct path
can't.

CLI binary names checked per solver: `gurobi_cl`, `cplex`,
`optimizer` (Xpress), `copt_cmd`. HiGHS deliberately refuses
`io_api='mps'` — the in-memory path is strictly better.

## Passing solver options

Anything after `solver_name=` / `io_api=` / `env=` is forwarded as-is
to the underlying solver via its native parameter API:

```python
result = solve(
    problem,
    solver_name="gurobi",
    TimeLimit=60,
    MIPGap=1e-4,
    OutputFlag=0,
)
```

The option keys are vendor-specific — `TimeLimit` is Gurobi's name;
CPLEX wants `timelimit`, Xpress uses `maxtime`, and so on. Consult
each vendor's reference for the supported parameter set.

## Troubleshooting

The adapters wrap raw vendor exceptions into `polar_high.solvers`
types so you never see a bare `gurobipy.GurobiError` etc. reach your
code. Distinguishing the two failure modes:

- `SolverNotAvailableError` — the Python wrapper is not importable.
  Install the corresponding optional extra.
- `LicenseError` — wrapper is installed but the license check
  failed. Fix per the messages below.
- `SolverError` — wrapper and license are fine, but the solver
  reported an error while solving (bad option, unsupported feature,
  etc.).

Common license errors, one per solver:

- **Gurobi** — `LicenseError: Gurobi license check failed (code 10009): ...`
  Place `gurobi.lic` at `$HOME` or `/opt/gurobi/`, set
  `GRB_LICENSE_FILE`, or pass a configured `gurobipy.Env` via
  `env=...`. WLS users build the env in code; the file path is not
  required.
- **CPLEX** — `LicenseError: CPLEX license check failed (code 1016): ...`
  Place `access.ilm` next to the CPLEX install, set
  `CPLEX_STUDIO_LICENSE` or `ILOG_LICENSE_FILE`, or configure your
  subscription credentials.
- **Xpress** — `LicenseError: Xpress license check failed: ...`
  Configure your Xpress license (`xpauth.xpr`, `XPRESS_LICENSE`,
  `XPAUTH_PATH` env vars), or call
  `xpress.init('<path-to-xpauth.xpr>')` before solving. The wheel's
  bundled FICO community license covers small problems and will not
  trigger this error.
- **COPT** — `LicenseError: COPT license check failed (code ...): ...`
  Place `copt.lic` in the COPT install dir, set `COPT_LICENSE_DIR`,
  or pass a configured `coptpy.Envr` via `env=...`.
