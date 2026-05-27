"""Wrapper-driven MPS roundtrip tests.

Companion to :mod:`tests.test_mps_fallback`, which exercises the *CLI
shell-out* path (writes MPS to disk, invokes the canonical solver
binary like ``gurobi_cl model.mps``).  Those tests legitimately skip
when the CLI binary isn't on PATH — wrapper-only installs (e.g.
``pip install gurobipy`` without the full Gurobi installer) can't run
them.

This module provides parallel coverage that works for wrapper-only
setups: write polar-high's MPS file via the same ``_write_mps`` helper,
then read it back into each commercial solver's *Python wrapper*, solve
there, and assert the objective matches an in-memory HiGHS direct
solve.  This catches MPS-format issues end-to-end (parser sees what the
writer wrote) without needing the standalone CLI binary.

COPT is intentionally out of scope here: the in-process COPT/HiGHS
native-symbol conflict (see ``solvers/_copt.py``) means we can't import
``coptpy`` alongside ``highspy`` in the same process.  Cross-solver
COPT coverage stays in ``test_mps_fallback.py`` (CLI-only) and
``test_solver_copt.py`` (whole-module gated).
"""

from __future__ import annotations

import highspy
import polars as pl
import pytest

from polar_high import Problem
from polar_high.solvers import SolverStatus, solve
from polar_high.solvers._lp_view import LpView
from polar_high.solvers._mps_fallback import _write_mps

# ---------------------------------------------------------------------------
# Writer parametrization
# ---------------------------------------------------------------------------
# Each readback test below runs through both MPS writers so we get
# wrapper-readback coverage on both code paths simultaneously:
#
#   * ``_legacy_write`` — the original HiGHS-backed writer in
#     ``solvers/_mps_fallback.py`` (still used by the CLI fallback).
#   * ``_direct_write`` — :meth:`Problem.write_mps` (added in v2.1.0),
#     which emits MPS directly from polars frames without ever
#     instantiating ``highspy.Highs``.
#
# Existing call sites that don't pass a ``writer`` get the legacy path
# unchanged.


def _legacy_write(pb: Problem, mps_path: str) -> None:
    view = LpView.from_problem(pb)
    _write_mps(view, mps_path)


def _direct_write(pb: Problem, mps_path: str) -> None:
    pb.write_mps(mps_path)


WRITER_IDS = ("legacy", "direct")
WRITERS = (_legacy_write, _direct_write)


def _tiny_lp() -> Problem:
    """Same LP shape as :func:`tests.test_mps_fallback._tiny_lp`.

    Two vars (``x``, ``y``), three constraints (one ``==``, one ``>=``,
    one ``<=``), objective ``min x + y``.  The ``x + y == 3`` equality
    fixes the optimum at 3.0.
    """
    pb = Problem()
    idx = pl.DataFrame({"i": [0]})
    x = pb.add_var("x", dims=("i",), index=idx, lower=0.0, upper=5.0)
    y = pb.add_var("y", dims=("i",), index=idx, lower=0.0, upper=5.0)
    pb.add_cstr("e1", over=idx, sense="==", lhs_terms={"x": x, "y": y}, rhs_terms={"c": 3.0})
    pb.add_cstr("g1", over=idx, sense=">=", lhs_terms={"x2": 2.0 * x, "y": y}, rhs_terms={"c": 1.0})
    pb.add_cstr("l1", over=idx, sense="<=", lhs_terms={"x": x, "y2": 2.0 * y}, rhs_terms={"c": 8.0})
    pb.set_objective(x + y, sense="min")
    return pb


def _expected_objective() -> float:
    """Direct in-memory HiGHS solve as the cross-tool truth value."""
    pb = _tiny_lp()
    result = solve(pb, solver_name="highs")
    assert result.status == SolverStatus.OPTIMAL
    assert result.objective is not None
    return float(result.objective)


def _write_lp_to_mps(tmp_path, writer=_legacy_write) -> str:
    """Shared MPS-writing prelude.  Returns the file path as a string.

    ``writer`` selects the implementation under test — the legacy
    HiGHS-backed serializer (default, for backwards compatibility with
    any caller that imports this helper) or the direct polars-only
    writer added in v2.1.0.  Tests in this module parametrize over
    both via the module-level ``WRITERS`` tuple.
    """
    pb = _tiny_lp()
    mps_path = tmp_path / "tiny.mps"
    writer(pb, str(mps_path))
    assert mps_path.is_file()
    assert mps_path.stat().st_size > 0
    return str(mps_path)


@pytest.mark.parametrize("writer", WRITERS, ids=WRITER_IDS)
def test_highs_wrapper_mps_readback(tmp_path, writer) -> None:
    """polar-high's MPS file reads back into a fresh highspy.Highs and
    solves to the same optimum.  No commercial wrapper involved — this
    is the baseline format-correctness check that runs unconditionally.

    This is a thin re-statement of test_mps_write_roundtrip in the
    wrapper-style file so the four readback tests live together.
    """
    mps_path = _write_lp_to_mps(tmp_path, writer=writer)
    h = highspy.Highs()
    try:
        h.setOptionValue("output_flag", False)
    except Exception:
        pass
    h.readModel(mps_path)
    h.run()
    assert h.getModelStatus() == highspy.HighsModelStatus.kOptimal
    assert abs(h.getObjectiveValue() - _expected_objective()) < 1e-9


@pytest.mark.parametrize("writer", WRITERS, ids=WRITER_IDS)
def test_gurobi_wrapper_mps_readback(tmp_path, writer) -> None:
    """polar-high writes MPS → ``gurobipy.read(...)`` → ``optimize()`` →
    objective matches HiGHS to 1e-6.  Runs whenever the ``gurobipy``
    wrapper is importable; does NOT need the ``gurobi_cl`` binary."""
    gp = pytest.importorskip("gurobipy")
    mps_path = _write_lp_to_mps(tmp_path, writer=writer)

    # Suppress the wrapper's default solver-log chatter so the test
    # output stays clean.  ``read`` returns a fully populated Model.
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    m = gp.read(mps_path, env=env)
    m.optimize()
    assert m.Status == gp.GRB.OPTIMAL, (
        f"Gurobi did not reach optimal on the MPS roundtrip: status={m.Status}"
    )
    assert abs(m.ObjVal - _expected_objective()) < 1e-6


@pytest.mark.parametrize("writer", WRITERS, ids=WRITER_IDS)
def test_cplex_wrapper_mps_readback(tmp_path, writer) -> None:
    """polar-high writes MPS → ``cplex.Cplex().read(...)`` → ``solve()``
    → objective matches HiGHS to 1e-6.  Wrapper-only; no ``cplex`` CLI
    binary required."""
    cplex = pytest.importorskip("cplex")
    mps_path = _write_lp_to_mps(tmp_path, writer=writer)

    c = cplex.Cplex()
    # Mute the wrapper's log streams.  set_log_stream / set_results_stream
    # accept ``None`` to discard.
    c.set_log_stream(None)
    c.set_results_stream(None)
    c.set_warning_stream(None)
    c.set_error_stream(None)
    c.read(mps_path)
    c.solve()
    # 1 = CPX_STAT_OPTIMAL for an LP solve.  We don't import the constant
    # to avoid pinning the test on CPLEX' internal layout.
    status_int = c.solution.get_status()
    assert status_int == 1, f"CPLEX did not reach optimal on the MPS roundtrip: status={status_int}"
    assert abs(c.solution.get_objective_value() - _expected_objective()) < 1e-6


@pytest.mark.parametrize("writer", WRITERS, ids=WRITER_IDS)
def test_xpress_wrapper_mps_readback(tmp_path, writer) -> None:
    """polar-high writes MPS → ``xpress.problem().read(...)`` →
    ``optimize()`` → objective matches HiGHS to 1e-6.  Wrapper-only; no
    ``optimizer`` CLI binary required."""
    xp = pytest.importorskip("xpress")
    mps_path = _write_lp_to_mps(tmp_path, writer=writer)

    m = xp.problem()
    # ``outputlog`` controls solver log emission; 0 silences it.  Some
    # Xpress builds reject the controlname via setControl with the
    # community license — fall back to silent if so.
    try:
        m.setControl("outputlog", 0)
    except Exception:
        pass
    # ``read`` was deprecated in favour of ``readProb`` in Xpress 9.8.
    # Prefer the modern name when available so the test stops emitting
    # a DeprecationWarning on current releases.
    if hasattr(m, "readProb"):
        m.readProb(mps_path)
    else:  # pragma: no cover — pre-9.8 Xpress
        m.read(mps_path)
    # ``optimize`` is the modern Xpress method; older versions used
    # ``solve``.  Prefer optimize when available.
    if hasattr(m, "optimize"):
        m.optimize()
    else:  # pragma: no cover — pre-9.x Xpress, kept as a graceful path
        m.solve()
    # ``lpstatus`` of 1 == optimal for an LP solve on Xpress 9.x.  Read
    # via ``attributes`` to avoid the legacy ``getProbStatus`` API that
    # was removed in Xpress 9.8.
    status_int = int(m.attributes.lpstatus)
    assert status_int == 1, (
        f"Xpress did not reach optimal on the MPS roundtrip: status={status_int}"
    )
    obj = float(m.attributes.lpobjval)
    assert abs(obj - _expected_objective()) < 1e-6
