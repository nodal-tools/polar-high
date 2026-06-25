"""Unit tests for ``polar_high.parallel`` — deterministic parallel solving.

Covers :func:`solve_indexed_parallel` (per-index deterministic collection,
parallel == sequential, built-precondition, exception propagation in index
order), :func:`resolve_worker_count`, and :func:`prewarm_global_scheduler`.
"""

from __future__ import annotations

import polars as pl
import pytest

from polar_high import (
    Param,
    Problem,
    WarmProblem,
    prewarm_global_scheduler,
    resolve_worker_count,
    solve_indexed_parallel,
)


def _make_warm(coef: float) -> WarmProblem:
    """A trivial built WarmProblem: minimize ``coef * x`` s.t. ``x >= 1``."""
    m = Problem()
    frame = pl.DataFrame({"i": ["a"]})
    x = m.add_var("x", ("i",), frame, lower=0.0)
    m.add_cstr(
        "floor",
        over=frame,
        sense=">=",
        lhs_terms={"x": x},
        rhs_terms={"one": Param(("i",), pl.DataFrame({"i": ["a"], "value": [1.0]}))},
    )
    m.set_objective(x * Param(("i",), pl.DataFrame({"i": ["a"], "value": [coef]})))
    wp = WarmProblem(m)
    wp.solve()  # cold first build (sequential)
    return wp


def test_resolve_worker_count() -> None:
    # Explicit request clamps to [1, n].
    assert resolve_worker_count(4, 2) == 2
    assert resolve_worker_count(4, 10) == 4
    assert resolve_worker_count(4, 0) == 1
    assert resolve_worker_count(4, 1) == 1
    # Auto: min(n, cpu-1), at least 1.
    auto = resolve_worker_count(3, None)
    assert 1 <= auto <= 3
    # Degenerate counts.
    assert resolve_worker_count(0, None) == 1
    assert resolve_worker_count(1, 8) == 1


def test_prewarm_global_scheduler_best_effort() -> None:
    # Best-effort: returns a bool, never raises.
    assert isinstance(prewarm_global_scheduler(1), bool)


def test_sequential_path_matches_plain_loop() -> None:
    coefs = [3.0, 5.0, 7.0]
    warms = [_make_warm(c) for c in coefs]
    objs = solve_indexed_parallel(warms, lambda i: warms[i].solve().obj, workers=1)
    # x* = 1 for each, so obj = coef.
    assert objs == coefs


def test_parallel_equals_sequential_exact() -> None:
    coefs = [2.0, 11.0, 0.5, 4.0, 9.0]
    warms_seq = [_make_warm(c) for c in coefs]
    warms_par = [_make_warm(c) for c in coefs]
    seq = solve_indexed_parallel(warms_seq, lambda i: warms_seq[i].solve().obj, workers=1)
    par = solve_indexed_parallel(warms_par, lambda i: warms_par[i].solve().obj, workers=4)
    # Bit-for-bit equality, in index order, independent of thread timing.
    assert par == seq
    assert par == coefs


def test_parallel_preserves_index_order_under_jitter() -> None:
    # Even if higher indices finish first, results stay index-ordered.
    coefs = [1.0, 2.0, 3.0, 4.0]
    warms = [_make_warm(c) for c in coefs]

    def work(i):
        # Reverse-index sleep to scramble completion order.
        import time

        time.sleep(0.005 * (len(coefs) - i))
        return warms[i].solve().obj

    out = solve_indexed_parallel(warms, work, workers=4)
    assert out == coefs


def test_unbuilt_precondition_raises() -> None:
    m = Problem()
    frame = pl.DataFrame({"i": ["a"]})
    m.add_var("x", ("i",), frame, lower=0.0)
    m.set_objective(m._vars["x"] * Param(("i",), pl.DataFrame({"i": ["a"], "value": [1.0]})))
    unbuilt = WarmProblem(m)  # never solved → _h is None
    with pytest.raises(ValueError, match="not built"):
        solve_indexed_parallel([unbuilt], lambda i: 0.0, workers=2)


def test_worker_exception_propagates_in_index_order() -> None:
    coefs = [1.0, 2.0, 3.0]
    warms = [_make_warm(c) for c in coefs]

    class Boom(RuntimeError):
        pass

    def work(i):
        if i >= 1:
            raise Boom(f"fail at {i}")
        return warms[i].solve().obj

    # Lowest failing index (1) wins, matching the sequential loop.
    with pytest.raises(Boom, match="fail at 1"):
        solve_indexed_parallel(warms, work, workers=3)
