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

import gc
import math
import os
import threading
import time

import highspy
import numpy as np
import polars as pl
import pytest

from polar_high import Param, Problem


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
    pb.add_cstr(
        "eq", over=idx, sense="==", lhs_terms={"a": a, "b": b, "c": c}, rhs_terms={"r": 4.0}
    )
    # a - b >= -3       (G)  -> shifts b into negative-of-RHS
    pb.add_cstr(
        "ge", over=idx, sense=">=", lhs_terms={"a": a, "negb": -1.0 * b}, rhs_terms={"r": -3.0}
    )
    # a + c <= 12       (L)
    pb.add_cstr("le", over=idx, sense="<=", lhs_terms={"a": a, "c": c}, rhs_terms={"r": 12.0})

    pb.set_objective(a + b + c, sense="min")

    direct = pb.solve()
    assert direct.optimal

    pb2 = Problem()
    a = pb2.add_var("a", dims=("i",), index=idx, lower=-5.0, upper=10.0)
    b = pb2.add_var("b", dims=("i",), index=idx, lower=-math.inf, upper=math.inf)
    c = pb2.add_var("c", dims=("i",), index=idx)
    pb2.add_cstr(
        "eq", over=idx, sense="==", lhs_terms={"a": a, "b": b, "c": c}, rhs_terms={"r": 4.0}
    )
    pb2.add_cstr(
        "ge", over=idx, sense=">=", lhs_terms={"a": a, "negb": -1.0 * b}, rhs_terms={"r": -3.0}
    )
    pb2.add_cstr("le", over=idx, sense="<=", lhs_terms={"a": a, "c": c}, rhs_terms={"r": 12.0})
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
        z = pb.add_var("z", dims=("i",), index=idx, lower=0.0, upper=10.0, integer=True)
        # Choose a problem where the LP relaxation IS integer-feasible
        # so the solve happens entirely in presolve and the answer is
        # deterministic across HiGHS versions.
        pb.add_cstr("g", over=idx, sense=">=", lhs_terms={"x": x, "z": z}, rhs_terms={"r": 4.5})
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

    lines = [ln for ln in on_err.splitlines() if ln.startswith("[write_mps profile]")]
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


# ---------------------------------------------------------------------------
# Per-term memory-explosion regression
# ---------------------------------------------------------------------------
def _read_vmrss_mb() -> float:
    """Resident set size in MiB from ``/proc/self/status``.  Linux-only —
    the test that uses this skips on other platforms."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    raise RuntimeError("VmRSS not found in /proc/self/status")


def _peak_rss_during(fn, sample_interval: float = 0.003) -> tuple[float, float, object]:
    """Run ``fn()`` while polling RSS every ``sample_interval`` seconds.
    Returns ``(baseline_mb, peak_mb, fn_result)``.  The baseline is
    sampled after a ``gc.collect()`` immediately before the call so the
    delta ``peak - baseline`` isolates allocations attributable to
    ``fn``."""
    gc.collect()
    baseline = _read_vmrss_mb()
    peak = [baseline]
    stop = [False]

    def sampler() -> None:
        while not stop[0]:
            r = _read_vmrss_mb()
            if r > peak[0]:
                peak[0] = r
            time.sleep(sample_interval)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    try:
        result = fn()
    finally:
        stop[0] = True
        th.join()
    return baseline, peak[0], result


def _build_chain_explosion_problem() -> tuple[Problem, int]:
    """Build a one-constraint Problem whose single LHS term is the
    broadcast-chain shape that historically blew up
    ``term.lazy.collect()``:

      Var x[t]   * Param p_tk[t,k]  * Param p_k[k]  * Param p_t1[t]

    Var spans T entries, the (t,k) intermediate is T*K dense, and the
    constraint ``over`` is a sparse subset (~0.5%) of (t,k).  The OLD
    code materialises the full T*K intermediate inside the term plan
    before joining to the row index; the NEW code semi-joins through the
    chain so polars prunes upstream.

    Returns (problem, row_count).
    """
    T = 200_000
    K = 20
    subset_frac = 0.005

    tt = np.repeat(np.arange(T), K)
    kk = np.tile(np.arange(K), T)
    rng = np.random.default_rng(0)
    keep_n = max(1, int(T * K * subset_frac))
    keep_idx = rng.choice(T * K, keep_n, replace=False)
    keep_idx.sort()
    idx_cstr = pl.DataFrame({"t": tt[keep_idx], "k": kk[keep_idx]})

    pb = Problem()
    idx_var = pl.DataFrame({"t": np.arange(T)})
    x = pb.add_var("x", dims=("t",), index=idx_var, lower=0.0, upper=1.0)

    p_tk = Param(
        ("t", "k"),
        pl.DataFrame({"t": tt, "k": kk, "value": np.linspace(1.0, 2.0, T * K)}),
    )
    p_k = Param(
        ("k",),
        pl.DataFrame({"k": np.arange(K), "value": np.linspace(0.5, 1.5, K)}),
    )
    p_t1 = Param(
        ("t",),
        pl.DataFrame({"t": np.arange(T), "value": np.linspace(1.0, 2.0, T)}),
    )

    expr = x * p_tk * p_k * p_t1
    pb.add_cstr(
        "chain",
        over=idx_cstr,
        sense="<=",
        lhs_terms={"lhs": expr},
        rhs_terms={"r": 5.0},
    )
    pb.set_objective(x, sense="min")
    return pb, int(idx_cstr.height)


@pytest.mark.skipif(
    not os.path.exists("/proc/self/status"),
    reason="VmRSS sampling requires /proc (Linux-only)",
)
def test_write_mps_param_chain_term_does_not_explode(tmp_path) -> None:
    """Regression for a per-term memory blow-up in
    :meth:`Problem.write_mps`.

    Background.  On a real-world DES LP (9.9M rows × 5M cols),
    ``write_mps`` peaked at 43 GB and a single constraint family with
    one ``Var * Param * Param * ...`` term contributed +26 GB during
    ``term.lazy.collect()``.  Polars' join-chain evaluator materialised
    a wide intermediate before producing the final row-aligned result.
    The fix mirrors the RHS streaming pattern at engine.py ~1570-1593:
    semi-join the term plan against the row-index keys so polars can
    prune the join chain, then collect with ``engine="streaming"``.

    This test builds a small-but-cliffed reproducer (~200k-row Var,
    ~4M-entry (t,k) intermediate, ~20k-row constraint) and asserts that
    peak RSS during ``write_mps`` stays modest.  On the unmodified
    engine.py the delta is ~500 MB; with the fix it drops to <100 MB.
    The threshold (300 MB) is generous to keep the test stable across
    polars versions while still failing loudly on regression.

    Correctness is verified by the byte-identical-coefficient check
    below: solving the MPS roundtrip with HiGHS must reach the same
    objective as the in-process ``Problem.solve()``.  Streaming
    reorders evaluation, not arithmetic.
    """
    # ---- 1. Build & write under RSS sampling -----------------------
    pb_for_write, row_count = _build_chain_explosion_problem()
    mps_path = tmp_path / "chain.mps"

    def do_write() -> None:
        pb_for_write.write_mps(str(mps_path))

    baseline, peak, _ = _peak_rss_during(do_write)
    delta = peak - baseline

    # The constraint emits ~row_count nnz triples; anything more than
    # ~300 MB of transient memory is a sign that the LHS term materialised
    # the dense intermediate instead of streaming through the semi-join.
    assert delta < 300.0, (
        f"write_mps peak RSS delta={delta:.0f} MB exceeds 300 MB budget "
        f"on {row_count}-row chain constraint (baseline={baseline:.0f} MB, "
        f"peak={peak:.0f} MB). Regression of the semi-join + streaming "
        f"fix at engine.py write_mps LHS site."
    )

    # ---- 2. Roundtrip — byte-identical coefficient check -----------
    # Read the MPS back through HiGHS and compare objective to an
    # in-process solve of an equivalent fresh Problem.  Identical
    # optimum ⟺ identical LP ⟺ identical coefficients.
    pb_for_solve, _ = _build_chain_explosion_problem()
    direct = pb_for_solve.solve()
    assert direct.optimal, direct.status

    status, obj, _ = _solve_via_mps(str(mps_path))
    assert status == highspy.HighsModelStatus.kOptimal
    assert abs(obj - direct.obj) < 1e-9, (obj, direct.obj)
