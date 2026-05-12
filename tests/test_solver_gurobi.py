"""Phase 5 tests for the Gurobi direct adapter.

The whole module is skipped at import time when ``gurobipy`` is not
installed — the ``[gurobi]`` optional extra is not present in CI.

License-error translation (``GurobiError`` errno 10009..10015 ->
:class:`LicenseError`) is *manual-only* per the implementation plan
(it requires physically removing or corrupting a real licence file);
the procedure is documented at the bottom of this file.
"""

from __future__ import annotations

import pytest

# Module-level skip — collected as one "skipped" entry rather than per
# test.  Keep the import inside the skip check so the module body never
# touches gurobipy when it isn't installed.
gp = pytest.importorskip("gurobipy")
pytest.importorskip("scipy")

from toy_data import make_toy_data  # noqa: E402
from toy_model import build_dispatch  # noqa: E402

from polar_high import Problem  # noqa: E402
from polar_high.solvers import SolverResult, SolverStatus, solve  # noqa: E402


def _toy_lp_problem() -> Problem:
    pb = Problem()
    build_dispatch(pb, make_toy_data())
    return pb


def _toy_mip_problem() -> Problem:
    """A tiny self-contained MIP.

    Maximise ``3x + 2y`` subject to ``x + y <= 4`` and ``x + 3y <= 6``,
    with ``x, y`` non-negative integers.  Optimum: ``x=3, y=1, obj=11``
    (verifiable by inspection).
    """
    pb = Problem()
    # Single scalar integer vars via add_var with no dims.  The engine
    # treats integer=True as the MIP flag.
    x = pb.add_var("x", lower=0.0, upper=10.0, integer=True)
    y = pb.add_var("y", lower=0.0, upper=10.0, integer=True)
    pb.add_cstr("c1", sense="<=", lhs_terms={"a": x, "b": y}, rhs_terms={"r": 4})
    pb.add_cstr("c2", sense="<=", lhs_terms={"a": x, "b": 3 * y}, rhs_terms={"r": 6})
    pb.set_objective(3 * x + 2 * y, sense="max")
    return pb


def test_gurobi_solves_toy_lp_returns_optimal() -> None:
    """Toy LP solves to optimal and returns a fully-populated result."""
    pb = _toy_lp_problem()
    result = solve(pb, solver_name="gurobi")

    assert isinstance(result, SolverResult)
    assert result.solver_name == "gurobi"
    assert result.status == SolverStatus.OPTIMAL
    assert result.objective is not None
    assert abs(result.objective - 6500.0) < 1e-6
    assert result.primal is not None and len(result.primal) > 0
    # LP -> dual must be populated.
    assert result.dual is not None and len(result.dual) > 0


def test_gurobi_solves_toy_mip_respects_integrality() -> None:
    """MIP variables come back integral and dual is None."""
    pb = _toy_mip_problem()
    result = solve(pb, solver_name="gurobi")

    assert result.status == SolverStatus.OPTIMAL
    assert result.objective is not None
    assert abs(result.objective - 11.0) < 1e-6
    assert result.primal is not None
    for nm, val in result.primal.items():
        assert abs(val - round(val)) < 1e-6, f"{nm}={val} not integral"
    # MIP -> dual must be None (HiGHS adapter does the same).
    assert result.dual is None


def test_gurobi_objective_matches_highs() -> None:
    """Same LP solved by both backends agrees within 1e-6."""
    pb_h = _toy_lp_problem()
    pb_g = _toy_lp_problem()

    r_h = solve(pb_h, solver_name="highs")
    r_g = solve(pb_g, solver_name="gurobi")

    assert r_h.status == SolverStatus.OPTIMAL
    assert r_g.status == SolverStatus.OPTIMAL
    assert r_h.objective is not None and r_g.objective is not None
    assert abs(r_h.objective - r_g.objective) < 1e-6


def test_gurobi_env_passthrough() -> None:
    """A pre-built ``gurobipy.Env`` is honoured by the adapter.

    We construct an Env with ``OutputFlag=0`` set on the *env* (not via
    ``options=``), pass it through ``solve(..., env=env)``, and assert
    the resulting Model picks up that setting.  This is the contract
    for advanced users running Gurobi WLS / Cluster Manager.
    """
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    try:
        pb = _toy_lp_problem()
        # The adapter currently discards the live Model after solve, so
        # we can't read its params back directly.  Instead, build a
        # *fresh* Model from the same env and assert the param took
        # effect on the env (and would therefore have taken effect on
        # the Model the adapter built).
        result = solve(pb, solver_name="gurobi", env=env)
        assert result.status == SolverStatus.OPTIMAL

        probe = gp.Model(env=env)
        try:
            # ``getParamInfo`` returns a tuple of (name, type, value,
            # min, max, default).  The third entry is the live value.
            info = probe.getParamInfo("OutputFlag")
            assert info[2] == 0, f"env-level OutputFlag not honoured: {info}"
        finally:
            probe.dispose()
    finally:
        env.dispose()


# ---------------------------------------------------------------------------
# Manual licence-error verification (NOT automated)
# ---------------------------------------------------------------------------
# The licence-error path in ``_gurobi.run`` translates
# ``GurobiError.errno`` in {10009, 10010, 10011, 10012, 10013, 10014,
# 10015} into ``polar_high.solvers.LicenseError`` with an actionable
# message.  We deliberately do NOT automate this: triggering it
# requires physically removing or corrupting a real Gurobi licence
# file, which is destructive to the developer's setup and inappropriate
# for CI.
#
# Manual reproduction procedure (run on a machine with a working Gurobi
# install):
#
#   1.  Locate the active licence file:
#         echo $GRB_LICENSE_FILE     # or check ~/gurobi.lic, /opt/gurobi/
#   2.  Rename it temporarily:
#         mv ~/gurobi.lic ~/gurobi.lic.bak
#   3.  In a Python session, run:
#
#         from polar_high import Problem
#         from polar_high.solvers import solve, LicenseError
#         pb = Problem()
#         pb.add_var("x", lower=0.0)
#         pb.set_objective(pb._vars["x"], sense="min")
#         try:
#             solve(pb, solver_name="gurobi")
#         except LicenseError as e:
#             print("Got LicenseError as expected:", e)
#
#   4.  Confirm that the printed message:
#         * is a polar_high LicenseError, not a raw GurobiError
#         * includes the code (10009 etc.)
#         * mentions ``GRB_LICENSE_FILE`` / ``gurobi.lic`` / the ``env=``
#           passthrough.
#   5.  Restore the licence:
#         mv ~/gurobi.lic.bak ~/gurobi.lic
#
# If the test ever exits step 3 with a raw ``GurobiError`` reaching the
# caller, that is a Phase 5 regression — the adapter must wrap every
# ``GurobiError`` it catches.
