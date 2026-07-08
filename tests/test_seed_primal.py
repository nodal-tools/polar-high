"""Acceptance tests for the partial primal-seed primitive (Phase 3 Step 3a).

The primitive is ``Problem.seed_primal(values, *, frame, missing="skip")``,
which RECORDS INTENT on ``self._primal_seed`` / ``self._primal_seed_frame``.
The solve-time hook lives in ``Problem._solve_streaming`` immediately before
``h.run()`` (just after the warm-basis block): when a seed is recorded, no
warm basis is pending, and the solve is in-process, it resolves each seeded
column *name* to its col index, converts ``user`` values to scaled space via
``_layer2_col_factor``, clamps every value to the target column's scaled
``[lb, ub]`` (critique A2 — ``setSolution`` is atomic), and calls
``h.setSolution``.  It falls back safely to a cold solve on any error, and
is mutually exclusive with a warm basis (critique A1 — the basis wins).

These tests reuse the generation-dispatch network builder from
``test_warm_start_streaming`` and add tiny dedicated LPs where a finite
upper bound (clamp test) or a known scale factor (self-scaling test) is
needed.
"""

from __future__ import annotations

import logging

import highspy
import numpy as np
import polars as pl
import pytest

import polar_high as fp
from test_warm_start_streaming import (
    _SOLVE_OPTS,
    _build_network,
    _iters,
    _obj_close,
)

_ENGINE_LOGGER = "polar_high.engine"


# ---------------------------------------------------------------------------
# setSolution spy — capture the (num, index, value) args the hook passes to
# HiGHS.  ``setSolution`` is only ever called by our seed hook, so a
# class-level wrapper (auto-restored by monkeypatch) is a clean spy.
# ---------------------------------------------------------------------------
class _SetSolutionSpy:
    def __init__(self) -> None:
        self.calls: list[tuple[int, np.ndarray, np.ndarray]] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        orig = highspy.Highs.setSolution
        calls = self.calls

        def spy(self_h, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
            # Overload 2: setSolution(num, index[int32], value[float64]).
            if len(args) == 3:
                num, index, value = args
                calls.append(
                    (int(num), np.asarray(index).copy(), np.asarray(value, dtype=float).copy())
                )
            return orig(self_h, *args, **kwargs)

        monkeypatch.setattr(highspy.Highs, "setSolution", spy)


# ---------------------------------------------------------------------------
# Tiny dedicated LPs.
# ---------------------------------------------------------------------------
def _build_bounded_lp() -> tuple[fp.Problem, float, float]:
    """Single bounded variable ``x`` in ``[0, 5]``, minimise ``-x``.

    Optimum: ``x = 5`` (upper bound binds), objective ``-5``.  The finite
    upper bound is what the clamp test needs.
    """
    p = fp.Problem()
    idx = pl.DataFrame({"i": np.arange(1, dtype=np.int64)})
    x = p.add_var("x", ("i",), idx, lower=0.0, upper=5.0)
    # A trivial, always-satisfied constraint so the model has a row and the
    # solve does non-degenerate work rather than pure presolve.
    p.add_cstr(
        "cap",
        over=idx,
        sense="<=",
        lhs_terms={"x": x},
        rhs_terms={"rhs": fp.Param(("i",), pl.DataFrame({"i": [0], "value": [10.0]}))},
    )
    p.set_objective(fp.Sum(x * -1.0), sense="min")
    return p, 5.0, -5.0


def _build_scaled_lp() -> fp.Problem:
    """Single variable ``x`` (``[0, inf]``) with one ``>=`` constraint.

    Used only to exercise the ``frame="user"`` self-scaling conversion; a
    uniform ``_layer2_col_factor`` is set by the test.
    """
    p = fp.Problem()
    idx = pl.DataFrame({"i": np.arange(1, dtype=np.int64)})
    x = p.add_var("x", ("i",), idx, lower=0.0)
    p.add_cstr(
        "demand",
        over=idx,
        sense=">=",
        lhs_terms={"x": x},
        rhs_terms={"d": fp.Param(("i",), pl.DataFrame({"i": [0], "value": [10.0]}))},
    )
    p.set_objective(fp.Sum(x * 1.0), sense="min")
    return p


# ---------------------------------------------------------------------------
# Test 1 — feasibility-safe seed: a non-optimal feasible seed never binds
# the optimum; report the cold-vs-seeded simplex-iteration counts.
# ---------------------------------------------------------------------------
def test_feasible_seed_does_not_bind_optimum():
    p_cold = _build_network(200, 20)
    sol_cold = p_cold.solve(keep_solver=True, options=_SOLVE_OPTS)
    assert sol_cold.optimal
    obj_cold = sol_cold.obj
    iters_cold = _iters(sol_cold)
    assert iters_cold > 0, f"cold solve did nothing measurable ({iters_cold} iters)"

    # A deliberately non-optimal but in-bounds seed: push a chunk of slack
    # (vq >= 0) up to 100.  No Layer 2 here, so scaled space == user space.
    n_t = 200
    seed = {f"vq[{t}]": 100.0 for t in range(n_t)}

    p_warm = _build_network(200, 20)
    p_warm.seed_primal(seed, frame="scaled")
    sol_warm = p_warm.solve(keep_solver=True, options=_SOLVE_OPTS)
    iters_warm = _iters(sol_warm)

    assert sol_warm.optimal
    # The non-optimal seed must NOT change the objective HiGHS returns.
    assert _obj_close(sol_warm.obj, obj_cold)

    # For the measurement: seeding the cold-OPTIMAL primal point (scaled ==
    # user here) instead reduces the iteration count — evidence the seed is
    # wired through and steers the solver.
    opt_seed = {nm: float(v) for nm, v in zip(sol_cold.col_names, sol_cold.col_value) if nm}
    p_opt = _build_network(200, 20)
    p_opt.seed_primal(opt_seed, frame="scaled")
    sol_opt = p_opt.solve(keep_solver=True, options=_SOLVE_OPTS)
    iters_opt = _iters(sol_opt)
    assert sol_opt.optimal
    assert _obj_close(sol_opt.obj, obj_cold)
    assert iters_opt < iters_cold, (
        f"optimal-point seed did not reduce iters: cold={iters_cold} opt={iters_opt}"
    )

    # Report the numbers (a bad seed may raise iters; a good one lowers them).
    print(
        f"[feasible-seed] cold iters={iters_cold} "
        f"bad-seed iters={iters_warm} opt-seed iters={iters_opt}"
    )


# ---------------------------------------------------------------------------
# Test 2 — bound sanitisation (A2): a value above a finite upper bound is
# CLAMPED, not passed raw (which would trip setSolution's atomic reject).
# ---------------------------------------------------------------------------
def test_seed_above_upper_bound_is_clamped(monkeypatch: pytest.MonkeyPatch):
    spy = _SetSolutionSpy()
    spy.install(monkeypatch)

    p, x_opt, obj_opt = _build_bounded_lp()
    # x's upper bound is 5.0; seed 100.0 (well above it).  A raw 100.0 would
    # make setSolution return kError (atomic) and drop the whole seed; a
    # clamped 5.0 is accepted (kOk) and lies exactly at the bound.
    p.seed_primal({"x[0]": 100.0}, frame="scaled")
    sol = p.solve(options=_SOLVE_OPTS)

    assert sol.optimal
    assert _obj_close(sol.obj, obj_opt)
    assert abs(sol.col_value[0] - x_opt) < 1e-9

    # Prove sanitisation happened: setSolution was called with the CLAMPED
    # value (5.0 == upper bound), not the raw 100.0.
    assert len(spy.calls) == 1, f"expected one setSolution call, got {len(spy.calls)}"
    num, index, value = spy.calls[0]
    assert num == 1
    assert list(index) == [0]
    assert value[0] == pytest.approx(5.0)
    assert value[0] != pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Test 3 — mutual exclusion (A1): with BOTH a warm basis and a seed set,
# the basis wins and the seed is skipped (never clobbers the basis).
# ---------------------------------------------------------------------------
def test_basis_wins_over_seed(monkeypatch: pytest.MonkeyPatch, caplog):
    spy = _SetSolutionSpy()
    spy.install(monkeypatch)

    p_cold = _build_network(200, 20)
    sol_cold = p_cold.solve(keep_solver=True, options=_SOLVE_OPTS)
    assert sol_cold.optimal
    obj_cold = sol_cold.obj
    nb = sol_cold.get_named_basis()

    p_warm = _build_network(200, 20)
    p_warm.set_named_basis(nb, policy="exact")
    p_warm.seed_primal({f"vq[{t}]": 100.0 for t in range(200)}, frame="scaled")

    with caplog.at_level(logging.INFO, logger=_ENGINE_LOGGER):
        sol_warm = p_warm.solve(keep_solver=True, options=_SOLVE_OPTS)

    assert sol_warm.optimal
    assert _obj_close(sol_warm.obj, obj_cold)
    # setSolution must NOT have been called — the seed was skipped so it
    # could not clobber the basis.
    assert spy.calls == [], "seed clobbered the basis (setSolution was called)"
    # The basis path was taken and the skip was logged.
    assert any("primal seed skipped" in r.message for r in caplog.records)
    assert any("warm-basis injected" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Test 4 — frame="user" self-scaling: the value reaching HiGHS is
# x_user / _layer2_col_factor[col_id].  Regression for the investigation.
# ---------------------------------------------------------------------------
def test_user_frame_self_scales_via_layer2(monkeypatch: pytest.MonkeyPatch):
    spy = _SetSolutionSpy()
    spy.install(monkeypatch)

    p = _build_scaled_lp()
    # Set a known, uniform Layer-2 column side-vector (inverse per-column
    # factor, positional by col_id).  factor = 8 (a power of two).
    factor = 8.0
    p._layer2_col_factor = np.full(p._next_col, factor, dtype=np.float64)

    x_user = 16.0
    p.seed_primal({"x[0]": x_user}, frame="user")
    sol = p.solve(options=_SOLVE_OPTS)

    assert sol.optimal
    # x's col_id is 0 (single scalar var, first added).
    assert len(spy.calls) == 1
    _num, index, value = spy.calls[0]
    assert list(index) == [0]
    # Forward transform: x_scaled = x_user / _layer2_col_factor[col_id].
    assert value[0] == pytest.approx(x_user / factor)  # 16 / 8 == 2.0


def test_user_frame_identity_when_no_layer2(monkeypatch: pytest.MonkeyPatch):
    """With no Layer-2 side-vector, ``frame="user"`` is the identity: the
    scaled value equals the user value."""
    spy = _SetSolutionSpy()
    spy.install(monkeypatch)

    p = _build_scaled_lp()
    assert p._layer2_col_factor is None
    p.seed_primal({"x[0]": 3.0}, frame="user")
    sol = p.solve(options=_SOLVE_OPTS)

    assert sol.optimal
    assert len(spy.calls) == 1
    _num, _index, value = spy.calls[0]
    assert value[0] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Test 5 — missing names skipped (subset behaviour).
# ---------------------------------------------------------------------------
def test_missing_names_are_skipped(monkeypatch: pytest.MonkeyPatch):
    spy = _SetSolutionSpy()
    spy.install(monkeypatch)

    p_cold = _build_network(200, 20)
    obj_cold = p_cold.solve(options=_SOLVE_OPTS).obj

    p = _build_network(200, 20)
    # One in-model name + one absent name.
    p.seed_primal({"vq[0]": 50.0, "not_a_real_column[999]": 1.0}, frame="scaled")
    sol = p.solve(options=_SOLVE_OPTS)

    assert sol.optimal
    assert _obj_close(sol.obj, obj_cold)
    # Only the in-model name was seeded (one column, not two).
    assert len(spy.calls) == 1
    num, index, _value = spy.calls[0]
    assert num == 1
    assert len(index) == 1


# ---------------------------------------------------------------------------
# Test 6 — frame is REQUIRED and validated; no silent default.
# ---------------------------------------------------------------------------
def test_frame_must_be_valid():
    p = _build_scaled_lp()
    with pytest.raises(ValueError, match="frame must be 'user' or 'scaled'"):
        p.seed_primal({"x[0]": 1.0}, frame="bogus")  # type: ignore[arg-type]


def test_frame_has_no_default():
    p = _build_scaled_lp()
    # ``frame`` is keyword-only with no default → omitting it is a TypeError.
    with pytest.raises(TypeError):
        p.seed_primal({"x[0]": 1.0})  # type: ignore[call-arg]


def test_missing_policy_validated():
    p = _build_scaled_lp()
    with pytest.raises(ValueError, match="missing must be 'skip'"):
        p.seed_primal({"x[0]": 1.0}, frame="scaled", missing="drop")
