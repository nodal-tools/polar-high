"""Phase 9 cross-solver integration tests.

These tests run only when at least two solver Python wrappers are
installed.  On a clean CI machine (HiGHS only) all parametrized cases
collapse to a single-solver loop and the pairwise assertions become
trivially true — pytest reports them as ``passed`` rather than skipped.

Coverage:

* :func:`test_objectives_agree_across_available_solvers` — every solver
  in :data:`polar_high.solvers.available_solvers` solves the same toy
  dispatch LP via the public :func:`polar_high.solvers.solve` entry
  point.  Pairwise objective agreement is asserted within 1e-6.
* :func:`test_objectives_agree_across_available_solvers_mip` — same
  shape but on a hand-crafted MIP :class:`LpView` consumed by each
  direct adapter's ``_run`` function.
* :func:`test_mps_fallback_matches_direct` — for every solver that has
  both a Python wrapper *and* the canonical CLI binary on ``PATH``,
  assert that :data:`IOMode.MPS` and :data:`IOMode.DIRECT` agree on the
  objective.  Skipped cleanly when no solver meets both conditions.
"""

from __future__ import annotations

import shutil
import sys

import numpy as np
import pytest
from toy_data import make_toy_data
from toy_model import build_dispatch

from polar_high import Problem
from polar_high.solvers import (
    IOMode,
    SolverStatus,
    available_solvers,
    solve,
)
from polar_high.solvers._lp_view import LpView

# Canonical CLI binary name per solver, used to gate the MPS-vs-direct
# parity test below.  Matches the lookup table in
# ``polar_high.solvers._mps_fallback``.
_CLI_BINARY = {
    "gurobi": "gurobi_cl",
    "cplex": "cplex",
    "xpress": "optimizer",
    "copt": "copt_cmd",
}


def _skip_if_copt_unreachable(solver_name: str) -> None:
    """Skip when ``solver_name='copt'`` would auto-route through a missing CLI.

    COPT/HiGHS conflict in one process: ``_copt.run`` auto-routes through
    the ``copt_cmd`` CLI whenever ``highspy`` is loaded.  If the binary
    is absent the adapter raises ``SolverNotAvailableError`` — skip the
    case rather than fail.
    """
    if solver_name == "copt" and "highspy" in sys.modules and shutil.which("copt_cmd") is None:
        pytest.skip(
            "COPT auto-routes to copt_cmd CLI when highspy is loaded (in-process "
            "COPT/HiGHS conflict); copt_cmd not on PATH in this venv.  Having "
            "the Python `coptpy` wrapper installed is insufficient for this "
            "case — the full COPT installer ships `copt_cmd`."
        )


def _toy_lp_problem() -> Problem:
    pb = Problem()
    build_dispatch(pb, make_toy_data())
    return pb


def _toy_mip_view() -> LpView:
    """Hand-built MIP :class:`LpView` matching the per-adapter test toys.

    Maximise ``3x + 2y`` subject to ``x + y <= 4`` and ``x + 3y <= 6``,
    with ``x, y`` non-negative integers in ``[0, 10]``.  Optimum:
    ``x=4, y=0, obj=12``.
    """
    col_obj = np.array([3.0, 2.0])
    col_lb = np.array([0.0, 0.0])
    col_ub = np.array([10.0, 10.0])
    integrality = np.array([1, 1], dtype=np.int8)
    row_lb = np.array([-np.inf, -np.inf])
    row_ub = np.array([4.0, 6.0])
    a_start = np.array([0, 2, 4], dtype=np.int32)
    a_index = np.array([0, 1, 0, 1], dtype=np.int32)
    a_value = np.array([1.0, 1.0, 1.0, 3.0])
    return LpView(
        n_cols=2,
        n_rows=2,
        col_obj=col_obj,
        col_lb=col_lb,
        col_ub=col_ub,
        integrality=integrality,
        row_lb=row_lb,
        row_ub=row_ub,
        a_start=a_start,
        a_index=a_index,
        a_value=a_value,
        col_names=["x", "y"],
        row_names=["c1", "c2"],
        sense="max",
        obj_offset=0.0,
    )


def _direct_adapter(solver_name: str):
    """Return the ``run`` callable for a solver's direct adapter.

    Imports are local so a missing optional wrapper raises only on the
    parametrize case that actually needs it; pytest then reports that
    case as an error rather than tainting the whole module.
    """
    if solver_name == "highs":
        from polar_high.solvers._highs import run

        return run
    if solver_name == "gurobi":
        from polar_high.solvers._gurobi import run

        return run
    if solver_name == "cplex":
        from polar_high.solvers._cplex import run

        return run
    if solver_name == "xpress":
        from polar_high.solvers._xpress import run

        return run
    if solver_name == "copt":
        from polar_high.solvers._copt import run

        return run
    raise ValueError(f"Unknown solver: {solver_name}")


# ---------------------------------------------------------------------------
# LP parity across all available solvers
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("solver_name", available_solvers)
def test_objectives_agree_across_available_solvers(solver_name: str) -> None:
    """Each available solver returns the same toy-LP objective.

    Parametrized over the runtime registry so a clean-CI machine (HiGHS
    only) executes a single trivially-passing case, while a developer
    box with several solvers installed exercises full N-way parity.

    Reference value 6500.0 comes from the existing per-solver toy-LP
    suites; the assertion against this constant ensures *every* solver
    in the registry agrees on the same anchored value, not just on a
    floating reference solve.
    """
    _skip_if_copt_unreachable(solver_name)

    pb = _toy_lp_problem()
    result = solve(pb, solver_name=solver_name)

    assert result.status == SolverStatus.OPTIMAL
    assert result.objective is not None
    # Pairwise agreement reduces to "each agrees with the anchored
    # reference" when the reference is the analytic optimum.
    assert abs(result.objective - 6500.0) < 1e-6, (
        f"{solver_name}: objective {result.objective} != 6500.0"
    )


# ---------------------------------------------------------------------------
# MIP parity across all available solvers
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("solver_name", available_solvers)
def test_objectives_agree_across_available_solvers_mip(solver_name: str) -> None:
    """Each available solver returns the same toy-MIP objective.

    Uses the direct adapter ``run`` function (not the public ``solve``)
    because the toy MIP is built as a hand-crafted :class:`LpView` —
    the public ``solve`` only accepts ``Problem`` instances and
    constructing an equivalent scalar-integer ``Problem`` via the
    public API is awkward (see the per-adapter test suites).
    """
    _skip_if_copt_unreachable(solver_name)

    view = _toy_mip_view()
    run = _direct_adapter(solver_name)
    result = run(view)

    assert result.status == SolverStatus.OPTIMAL
    assert result.objective is not None
    assert abs(result.objective - 12.0) < 1e-6, (
        f"{solver_name}: MIP objective {result.objective} != 12.0"
    )
    # MIP -> dual is None on every direct adapter by design.
    assert result.dual is None


# ---------------------------------------------------------------------------
# MPS file path vs. in-memory direct path
# ---------------------------------------------------------------------------
def _mps_parity_candidates() -> list[str]:
    """Solvers with BOTH a Python wrapper installed AND a CLI binary on PATH.

    The MPS path needs only the CLI binary; the direct path needs only
    the Python wrapper.  This parity test compares both, so we need
    both present.
    """
    out: list[str] = []
    for name in available_solvers:
        if name == "highs":
            # HiGHS refuses io_api='mps' by design (see
            # ``solvers.solve`` and ``test_mps_fallback``).  Skip it.
            continue
        cli = _CLI_BINARY.get(name)
        if cli is not None and shutil.which(cli) is not None:
            out.append(name)
    return out


_MPS_PARITY = _mps_parity_candidates()


@pytest.mark.skipif(
    not _MPS_PARITY,
    reason=(
        "no solver has BOTH the Python wrapper installed AND the corresponding "
        "CLI binary on PATH — the parity check needs both to compare them on "
        "the same machine.  Wrapper-only setups still get MPS-write coverage "
        "via tests/test_mps_fallback_wrapper.py."
    ),
)
@pytest.mark.parametrize("solver_name", _MPS_PARITY)
def test_mps_fallback_matches_direct(solver_name: str) -> None:
    """``io_api='mps'`` and ``io_api='direct'`` return the same objective."""
    pb_direct = _toy_lp_problem()
    pb_mps = _toy_lp_problem()

    r_direct = solve(pb_direct, solver_name=solver_name, io_api=IOMode.DIRECT)
    r_mps = solve(pb_mps, solver_name=solver_name, io_api=IOMode.MPS)

    assert r_direct.status == SolverStatus.OPTIMAL
    assert r_mps.status == SolverStatus.OPTIMAL
    assert r_direct.objective is not None and r_mps.objective is not None
    assert abs(r_direct.objective - r_mps.objective) < 1e-6, (
        f"{solver_name}: direct={r_direct.objective} mps={r_mps.objective}"
    )
