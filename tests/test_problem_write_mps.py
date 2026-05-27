"""Unit tests for :meth:`polar_high.engine.Problem.write_mps`.

These tests exercise the *direct* polars → MPS writer added in v2.1.0,
the one that never instantiates :class:`highspy.Highs` and is intended
for very-large LPs where HiGHS' own ``writeModel`` OOMs.  Roundtrip
correctness is asserted by writing the MPS, reading it back into a
fresh HiGHS instance, solving, and comparing against an in-process
:meth:`Problem.solve` reference run.

Companion tests in ``test_mps_fallback_wrapper.py`` exercise the new
writer through the existing wrapper-readback harness for HiGHS,
Gurobi, CPLEX and Xpress.
"""

from __future__ import annotations

import math

import highspy
import polars as pl
import pytest

from polar_high import Problem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _solve_via_mps(path: str) -> tuple[highspy.HighsModelStatus, float, list[float]]:
    """Read ``path`` into a fresh HiGHS, solve, return (status, obj, col_value)."""
    h = highspy.Highs()
    try:
        h.setOptionValue("output_flag", False)
    except Exception:
        pass
    h.readModel(path)
    h.run()
    status = h.getModelStatus()
    obj = float(h.getObjectiveValue())
    sol = h.getSolution()
    return status, obj, list(sol.col_value)


def _tiny_lp() -> Problem:
    """Same LP shape as the wrapper-test fixture (kept local to avoid a
    cross-module test import)."""
    pb = Problem()
    idx = pl.DataFrame({"i": [0]})
    x = pb.add_var("x", dims=("i",), index=idx, lower=0.0, upper=5.0)
    y = pb.add_var("y", dims=("i",), index=idx, lower=0.0, upper=5.0)
    pb.add_cstr("e1", over=idx, sense="==", lhs_terms={"x": x, "y": y}, rhs_terms={"c": 3.0})
    pb.add_cstr("g1", over=idx, sense=">=", lhs_terms={"x2": 2.0 * x, "y": y}, rhs_terms={"c": 1.0})
    pb.add_cstr("l1", over=idx, sense="<=", lhs_terms={"x": x, "y2": 2.0 * y}, rhs_terms={"c": 8.0})
    pb.set_objective(x + y, sense="min")
    return pb


# ---------------------------------------------------------------------------
# Roundtrips
# ---------------------------------------------------------------------------
def test_roundtrip_tiny_lp(tmp_path) -> None:
    """Tiny LP written by write_mps → readModel → solve matches the
    in-process objective bit-for-bit (well, to 1e-9)."""
    pb = _tiny_lp()
    direct = pb.solve()
    assert direct.optimal

    pb2 = _tiny_lp()  # fresh copy — write_mps without release is non-mutating
    mps_path = tmp_path / "tiny.mps"
    pb2.write_mps(str(mps_path))
    assert mps_path.is_file() and mps_path.stat().st_size > 0

    status, obj, _ = _solve_via_mps(str(mps_path))
    assert status == highspy.HighsModelStatus.kOptimal
    assert abs(obj - direct.obj) < 1e-9


def test_roundtrip_mixed_senses_and_bounds(tmp_path) -> None:
    """Exercise lower<0, free var, and all three senses in one LP."""
    pb = Problem()
    idx = pl.DataFrame({"i": [0]})
    # a: bounded with negative lower
    a = pb.add_var("a", dims=("i",), index=idx, lower=-5.0, upper=10.0)
    # b: fully free
    b = pb.add_var("b", dims=("i",), index=idx, lower=-math.inf, upper=math.inf)
    # c: default bounds [0, +inf)
    c = pb.add_var("c", dims=("i",), index=idx)

    # a + b + c == 4    (E)
    pb.add_cstr("eq", over=idx, sense="==",
                lhs_terms={"a": a, "b": b, "c": c}, rhs_terms={"r": 4.0})
    # a - b >= -3       (G)  -> shifts b into negative-of-RHS
    pb.add_cstr("ge", over=idx, sense=">=",
                lhs_terms={"a": a, "negb": -1.0 * b}, rhs_terms={"r": -3.0})
    # a + c <= 12       (L)
    pb.add_cstr("le", over=idx, sense="<=",
                lhs_terms={"a": a, "c": c}, rhs_terms={"r": 12.0})

    pb.set_objective(a + b + c, sense="min")

    direct = pb.solve()
    assert direct.optimal

    pb2 = Problem()
    a = pb2.add_var("a", dims=("i",), index=idx, lower=-5.0, upper=10.0)
    b = pb2.add_var("b", dims=("i",), index=idx, lower=-math.inf, upper=math.inf)
    c = pb2.add_var("c", dims=("i",), index=idx)
    pb2.add_cstr("eq", over=idx, sense="==",
                 lhs_terms={"a": a, "b": b, "c": c}, rhs_terms={"r": 4.0})
    pb2.add_cstr("ge", over=idx, sense=">=",
                 lhs_terms={"a": a, "negb": -1.0 * b}, rhs_terms={"r": -3.0})
    pb2.add_cstr("le", over=idx, sense="<=",
                 lhs_terms={"a": a, "c": c}, rhs_terms={"r": 12.0})
    pb2.set_objective(a + b + c, sense="min")

    mps_path = tmp_path / "mixed.mps"
    pb2.write_mps(str(mps_path))

    text = mps_path.read_text()
    # Sanity checks on the section markers and bound-classification.
    assert " FR bnd  b[0]" in text, text
    assert " LO bnd  a[0]  -5" in text or " LO bnd  a[0]  -5.0" in text, text
    assert " UP bnd  a[0]  10" in text or " UP bnd  a[0]  10.0" in text, text

    status, obj, _ = _solve_via_mps(str(mps_path))
    assert status == highspy.HighsModelStatus.kOptimal
    assert abs(obj - direct.obj) < 1e-9


def test_roundtrip_integer_var(tmp_path) -> None:
    """Integer column must be flagged via INTORG/INTEND markers and the
    solver must see the column as integer on readback.

    Assertion strategy: build the SAME Problem twice; solve once
    in-process (the polar-high reference), write the second to MPS and
    solve from the MPS file via a fresh HiGHS.  Objectives must match.
    Whether HiGHS found the true global MIP optimum is *not* the
    concern here — that's a solver question — what matters is that the
    MPS we wrote describes the same LP (matrix, bounds, integrality).
    Additionally, assert the MPS contains the INTORG/INTEND markers
    and the readback LP has the right integrality flags.
    """
    def _build() -> Problem:
        pb = Problem()
        idx = pl.DataFrame({"i": [0]})
        x = pb.add_var("x", dims=("i",), index=idx, lower=0.0, upper=10.0)
        z = pb.add_var(
            "z", dims=("i",), index=idx, lower=0.0, upper=10.0, integer=True
        )
        # Choose a problem where the LP relaxation IS integer-feasible
        # so the solve happens entirely in presolve and the answer is
        # deterministic across HiGHS versions.
        pb.add_cstr("g", over=idx, sense=">=",
                    lhs_terms={"x": x, "z": z}, rhs_terms={"r": 4.5})
        pb.set_objective(2.0 * x + 3.0 * z, sense="min")
        return pb

    direct = _build().solve()
    assert direct.optimal

    mps_path = tmp_path / "mip.mps"
    _build().write_mps(str(mps_path))
    text = mps_path.read_text()
    assert "INTORG" in text and "INTEND" in text, text

    # Read back and check that HiGHS sees z as integer.
    h = highspy.Highs()
    try:
        h.setOptionValue("output_flag", False)
    except Exception:
        pass
    h.readModel(str(mps_path))
    lp = h.getLp()
    integ = list(lp.integrality_)
    assert len(integ) == 2
    # z was added second so its col_id is 1.
    assert int(integ[1]) == int(highspy.HighsVarType.kInteger), integ
    assert int(integ[0]) == int(highspy.HighsVarType.kContinuous), integ

    h.run()
    assert h.getModelStatus() == highspy.HighsModelStatus.kOptimal
    obj = float(h.getObjectiveValue())
    sol = list(h.getSolution().col_value)
    # Same objective the in-process solve returned — that's the round-
    # trip invariant.  z value must still be integer-valued.
    assert abs(obj - direct.obj) < 1e-6, (obj, direct.obj, sol)
    assert abs(round(sol[1]) - sol[1]) < 1e-6, sol


def test_emit_names_false(tmp_path) -> None:
    """``emit_names=False`` produces generic R/C names but the same LP —
    objective and column values (by index) must match the named write."""
    pb1 = _tiny_lp()
    p_named = tmp_path / "named.mps"
    pb1.write_mps(str(p_named), emit_names=True)

    pb2 = _tiny_lp()
    p_anon = tmp_path / "anon.mps"
    pb2.write_mps(str(p_anon), emit_names=False)

    text_anon = p_anon.read_text()
    # No tagged variable names left.
    assert "x[0]" not in text_anon
    assert "C0000001" in text_anon
    # Cost row name stays literal.
    assert "cost" in text_anon
    # Constraint row names are generic.
    assert "R0000002" in text_anon

    s1, obj1, cv1 = _solve_via_mps(str(p_named))
    s2, obj2, cv2 = _solve_via_mps(str(p_anon))
    assert s1 == s2 == highspy.HighsModelStatus.kOptimal
    assert abs(obj1 - obj2) < 1e-12
    assert len(cv1) == len(cv2)
    for a, b in zip(cv1, cv2):
        assert abs(a - b) < 1e-9


def test_release_true(tmp_path) -> None:
    """``release=True`` puts the Problem in the released state; subsequent
    solve must raise the same RuntimeError the save_memory path raises."""
    pb = _tiny_lp()
    mps_path = tmp_path / "tiny.mps"
    pb.write_mps(str(mps_path), release=True)
    assert pb._released is True
    with pytest.raises(RuntimeError):
        pb.solve()
    # Calling write_mps again on a released Problem should also raise.
    with pytest.raises(RuntimeError):
        pb.write_mps(str(tmp_path / "again.mps"))


def test_nan_coef_hard_errors(tmp_path) -> None:
    """A NaN coefficient anywhere in the model must produce a ValueError
    rather than a silently-corrupt MPS file."""
    pb = _tiny_lp()
    # Inject a NaN into the objective by replacing the first term's lazy
    # plan with one whose coef is NaN.  This mimics what would happen if
    # a Param had a NaN value column.
    t = pb._obj_terms[0]
    t.lazy = t.lazy.with_columns(coef=pl.col("coef") * float("nan"))

    with pytest.raises(ValueError) as excinfo:
        pb.write_mps(str(tmp_path / "nan.mps"))
    msg = str(excinfo.value)
    assert "NaN" in msg or "nan" in msg or "infinite" in msg, msg


def test_column_order_strict_false_not_implemented(tmp_path) -> None:
    """The kwarg exists on the signature for forward-compat but the
    non-strict path isn't implemented yet — should NotImplementedError."""
    pb = _tiny_lp()
    with pytest.raises(NotImplementedError):
        pb.write_mps(str(tmp_path / "x.mps"), column_order_strict=False)


def test_obj_offset_warns(tmp_path) -> None:
    """A non-zero objective offset triggers a UserWarning and is dropped
    (MPS has no portable encoding for it)."""
    pb = _tiny_lp()
    pb.add_obj_constant(7.5)
    with pytest.warns(UserWarning, match="objective offset"):
        pb.write_mps(str(tmp_path / "off.mps"))


def test_profile_env_var_emits_checkpoints(tmp_path, monkeypatch, capsys) -> None:
    """When ``POLAR_HIGH_WRITE_MPS_PROFILE=1`` is set, write_mps emits
    tab-separated `[write_mps profile]` checkpoints to stderr; when
    unset (the default) no profile lines are produced.

    Sanity guard for the optional memory-profiling instrumentation —
    keeps the off-path silent (zero output) and verifies the on-path
    fires the lifecycle checkpoints (`enter`, at least one
    `family_start`, `exit`).
    """
    # --- off-path: no env var → no profile lines on stderr ----------
    monkeypatch.delenv("POLAR_HIGH_WRITE_MPS_PROFILE", raising=False)
    _tiny_lp().write_mps(str(tmp_path / "off.mps"))
    off_err = capsys.readouterr().err
    assert "[write_mps profile]" not in off_err

    # --- on-path: env var set → checkpoints appear on stderr --------
    monkeypatch.setenv("POLAR_HIGH_WRITE_MPS_PROFILE", "1")
    _tiny_lp().write_mps(str(tmp_path / "on.mps"))
    on_err = capsys.readouterr().err

    # If psutil isn't installed in the test env the writer prints a
    # one-line warning and disables profiling — that's the documented
    # graceful-degrade behaviour, not a test failure.
    if "psutil not installed" in on_err:
        pytest.skip("psutil not installed — profiling gracefully disabled")

    lines = [
        ln for ln in on_err.splitlines() if ln.startswith("[write_mps profile]")
    ]
    assert lines, f"expected profile lines on stderr; got:\n{on_err!r}"
    phases = [
        next(
            (p.split("=", 1)[1] for p in ln.split("\t") if p.startswith("phase=")),
            None,
        )
        for ln in lines
    ]
    assert "enter" in phases
    assert "exit" in phases
    assert "family_start" in phases
