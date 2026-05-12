"""Phase 4 tests for the MPS-file fallback dispatch.

These tests cover two layers:

* ``test_mps_write_roundtrip`` — the MPS *writer* alone (no external
  binary needed).  We build a small LP, write it via
  :func:`polar_high.solvers._mps_fallback._write_mps`, read it back with
  ``highspy.Highs.readModel``, solve, and compare the objective against a
  direct in-memory HiGHS solve.

* ``test_<solver>_cli_mps_path`` — the full shell-out path for each
  commercial solver, gated by ``pytest.mark.skipif`` on the absence of
  the canonical binary.  On a clean CI without licensed solvers, all
  four are skipped cleanly.
"""

from __future__ import annotations

import shutil

import highspy
import polars as pl
import pytest

from polar_high import Problem
from polar_high.solvers import IOMode, SolverStatus, solve
from polar_high.solvers._lp_view import LpView
from polar_high.solvers._mps_fallback import _write_mps


def _tiny_lp() -> Problem:
    """A tiny two-variable LP with a known analytic optimum.

    Variables (over a single-row index ``i``):
      x in [0, 5], y in [0, 5]
    Constraints:
      e1 :  x + y == 3       (E)
      g1 :  2x + y >= 1      (G)
      l1 :  x + 2y <= 8      (L)
    Objective:
      min x + y

    The equality fixes the objective at 3.0.
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


def test_mps_write_roundtrip(tmp_path) -> None:
    """Write an MPS via the helper, read it back with highspy, solve, and
    assert the objective matches a direct in-memory HiGHS solve to 1e-9.

    No binary is needed for this test — the MPS writer and highspy alone
    exercise the full file-format path.
    """
    pb = _tiny_lp()
    view = LpView.from_problem(pb)

    mps_path = tmp_path / "tiny.mps"
    _write_mps(view, str(mps_path))
    assert mps_path.is_file(), "writer did not produce the MPS file"
    assert mps_path.stat().st_size > 0, "MPS file is empty"

    # Re-read via a fresh Highs instance and solve.
    h = highspy.Highs()
    try:
        h.setOptionValue("output_flag", False)
    except Exception:
        pass
    h.readModel(str(mps_path))
    h.run()
    assert h.getModelStatus() == highspy.HighsModelStatus.kOptimal
    obj_from_mps = h.getObjectiveValue()

    # Direct in-memory solve via the new dispatch.
    pb_direct = _tiny_lp()
    result = solve(pb_direct, solver_name="highs")
    assert result.status == SolverStatus.OPTIMAL
    assert result.objective is not None

    assert abs(obj_from_mps - result.objective) < 1e-9


def test_mps_highs_routing_refused() -> None:
    """``solve(..., solver_name='highs', io_api=MPS)`` must refuse loudly.

    HiGHS has no use case for the file-based path (the direct in-memory
    adapter is strictly better), so the dispatch raises ``ValueError``
    rather than silently round-tripping or falling back.
    """
    pb = _tiny_lp()
    with pytest.raises(ValueError, match="highs"):
        solve(pb, solver_name="highs", io_api=IOMode.MPS)


def test_mps_lp_mode_not_implemented() -> None:
    """``io_api=IOMode.LP`` is reserved but still raises cleanly."""
    pb = _tiny_lp()
    # Use whatever solver is first available; the dispatch raises on
    # the io_api branch before the solver_name dispatch.
    with pytest.raises(NotImplementedError, match="lp"):
        solve(pb, io_api=IOMode.LP)


@pytest.mark.skipif(shutil.which("gurobi_cl") is None, reason="gurobi_cl not on PATH")
def test_gurobi_cl_mps_path() -> None:
    """Round-trip a tiny LP through ``gurobi_cl``.  Skipped on clean CI."""
    pb = _tiny_lp()
    result = solve(pb, solver_name="gurobi", io_api=IOMode.MPS)
    assert result.status == SolverStatus.OPTIMAL
    assert result.objective is not None
    assert abs(result.objective - 3.0) < 1e-6


@pytest.mark.skipif(shutil.which("cplex") is None, reason="cplex not on PATH")
def test_cplex_cli_mps_path() -> None:
    """Round-trip a tiny LP through CPLEX's interactive optimizer."""
    pb = _tiny_lp()
    result = solve(pb, solver_name="cplex", io_api=IOMode.MPS)
    assert result.status == SolverStatus.OPTIMAL
    assert result.objective is not None
    assert abs(result.objective - 3.0) < 1e-6


@pytest.mark.skipif(shutil.which("optimizer") is None, reason="xpress 'optimizer' not on PATH")
def test_xpress_optimizer_mps_path() -> None:
    """Round-trip a tiny LP through Xpress' ``optimizer`` console."""
    pb = _tiny_lp()
    result = solve(pb, solver_name="xpress", io_api=IOMode.MPS)
    assert result.status == SolverStatus.OPTIMAL
    assert result.objective is not None
    assert abs(result.objective - 3.0) < 1e-6


@pytest.mark.skipif(shutil.which("copt_cmd") is None, reason="copt_cmd not on PATH")
def test_copt_cmd_mps_path() -> None:
    """Round-trip a tiny LP through ``copt_cmd``."""
    pb = _tiny_lp()
    result = solve(pb, solver_name="copt", io_api=IOMode.MPS)
    assert result.status == SolverStatus.OPTIMAL
    assert result.objective is not None
    assert abs(result.objective - 3.0) < 1e-6
