"""Col-id-level pin primitives: :meth:`WarmProblem.fix_col_ids` and the
:meth:`get_col_bounds` / :meth:`set_col_bounds` save/restore pair.

``fix_col_ids`` is the fast-path counterpart of :meth:`fix_cols` — the same
``changeColsBounds(lo=hi=value)`` write, addressed by raw col id instead of a
per-call dim-tuple join.  Byte-equivalence with ``fix_cols`` on the same
columns is the contract, so the tests compare BOTH the resulting bounds and
the resulting solutions across two identically-built LPs.

``get_col_bounds`` / ``set_col_bounds`` back the temporary-pin pattern (save
bounds, pin, solve, restore): the round-trip must be exact, including a solve
between save and restore, and must survive infinite bounds and arbitrary /
repeated id order.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import polar_high as fp

# ----------------------------------------------------------------------------
# Shared fixture LP.


def _build_problem() -> fp.Problem:
    """A tiny dispatch LP with a hand-verifiable optimum::

        min  Σ_i cost_i · x_i
        s.t. x_i >= demand_i           (per-i "meet" row)
             0 <= x_i <= +inf          (i = 0..3)

    cost = [1, 2, 3, 4], demand = [3, 4, 5, 6] ⇒ x = demand,
    obj = 3 + 8 + 15 + 24 = 50.
    """
    p = fp.Problem()
    idx = pl.DataFrame({"i": [0, 1, 2, 3]})
    x = p.add_var("x", "i", idx, lower=0.0, upper=np.inf)
    demand = fp.Param(("i",), pl.DataFrame({"i": [0, 1, 2, 3], "value": [3.0, 4.0, 5.0, 6.0]}))
    p.add_cstr("meet", over=idx, sense=">=", lhs_terms={"x": x}, rhs_terms={"d": demand})
    cost = fp.Param(("i",), pl.DataFrame({"i": [0, 1, 2, 3], "value": [1.0, 2.0, 3.0, 4.0]}))
    p.set_objective(cost * x, sense="min")
    p.set_solver_options({"output_flag": False})
    return p


def _solved_warm() -> tuple[fp.WarmProblem, fp.Solution]:
    wp = fp.WarmProblem(_build_problem())
    sol = wp.solve()
    assert sol.optimal
    assert sol.obj == pytest.approx(50.0, abs=1e-9)
    return wp, sol


def _x_col_ids(wp: fp.WarmProblem) -> np.ndarray:
    return np.asarray([wp.col_id_of_var("x", (i,)) for i in range(4)], dtype=np.int64)


# ----------------------------------------------------------------------------
# fix_col_ids ≡ fix_cols byte-equivalence.


def test_fix_col_ids_byte_equivalent_to_fix_cols() -> None:
    """Two identically-built LPs; pin the same columns to the same values
    once via ``fix_cols`` (dim tuples), once via ``fix_col_ids`` (raw
    ids).  Resulting bounds AND solutions must be identical."""
    wp_a, _ = _solved_warm()
    wp_b, _ = _solved_warm()

    dim_tuples = [(1,), (3,)]
    pin_vals = np.array([7.0, 8.0])

    wp_a.fix_cols("x", dim_tuples, pin_vals)
    ids_b = np.asarray([wp_b.col_id_of_var("x", dt) for dt in dim_tuples], dtype=np.int64)
    wp_b.fix_col_ids(ids_b, pin_vals)

    # Bounds byte-equal on every x column (pinned and untouched alike).
    all_a = _x_col_ids(wp_a)
    all_b = _x_col_ids(wp_b)
    lo_a, hi_a = wp_a.get_col_bounds(all_a)
    lo_b, hi_b = wp_b.get_col_bounds(all_b)
    assert np.array_equal(lo_a, lo_b)
    assert np.array_equal(hi_a, hi_b)
    # The pinned columns really are lo = hi = value.
    assert np.array_equal(lo_a[[1, 3]], pin_vals)
    assert np.array_equal(hi_a[[1, 3]], pin_vals)

    # Solutions byte-equal (same LP, same pins, same solver path).
    sol_a = wp_a.solve()
    sol_b = wp_b.solve()
    assert sol_a.optimal and sol_b.optimal
    assert sol_a.obj == sol_b.obj
    assert np.array_equal(sol_a.col_value, sol_b.col_value)
    # Hand-check: x = [3, 7, 5, 8] ⇒ obj = 3 + 14 + 15 + 32 = 64.
    assert sol_a.obj == pytest.approx(64.0, abs=1e-9)


def test_fix_col_ids_length_mismatch_raises() -> None:
    wp, _ = _solved_warm()
    ids = _x_col_ids(wp)
    with pytest.raises(ValueError, match="fix_col_ids"):
        wp.fix_col_ids(ids, np.array([1.0, 2.0]))


def test_fix_col_ids_requires_built() -> None:
    wp = fp.WarmProblem(_build_problem())
    with pytest.raises(RuntimeError, match="must call solve"):
        wp.fix_col_ids(np.array([0]), np.array([1.0]))


# ----------------------------------------------------------------------------
# get_col_bounds / set_col_bounds save-restore round trip.


def test_bounds_save_restore_round_trip_with_solve_between() -> None:
    """Save bounds, pin via fix_col_ids, solve at the pin, restore, and
    verify (a) the restored bounds equal the saved ones exactly (incl. the
    +inf upper) and (b) a re-solve reproduces the pre-pin optimum."""
    wp, sol0 = _solved_warm()
    ids = _x_col_ids(wp)

    lo0, hi0 = wp.get_col_bounds(ids)
    assert np.array_equal(lo0, np.zeros(4))
    assert np.all(np.isinf(hi0)) and np.all(hi0 > 0)

    # Temporary pin: force every x above its demand.
    pin = np.array([9.0, 9.0, 9.0, 9.0])
    wp.fix_col_ids(ids, pin)
    sol_pin = wp.solve()
    assert sol_pin.optimal
    # obj = 9·(1+2+3+4) = 90 at the pin.
    assert sol_pin.obj == pytest.approx(90.0, abs=1e-9)

    # Restore and check the bounds round-tripped exactly.
    wp.set_col_bounds(ids, lo0, hi0)
    lo1, hi1 = wp.get_col_bounds(ids)
    assert np.array_equal(lo0, lo1)
    assert np.array_equal(hi0, hi1)

    # And the LP is behaviourally back: same optimum as before the pin.
    sol1 = wp.solve()
    assert sol1.optimal
    assert sol1.obj == sol0.obj
    assert np.array_equal(sol1.col_value, sol0.col_value)


def test_get_col_bounds_caller_order_and_repeats() -> None:
    """HiGHS' getCols wants an ordered duplicate-free set; the wrapper must
    accept arbitrary order and repeats and align results positionally."""
    wp, _ = _solved_warm()
    ids = _x_col_ids(wp)
    # Give the columns distinguishable bounds first.
    wp.set_col_bounds(ids, np.array([0.0, 1.0, 2.0, 3.0]), np.array([10.0, 11.0, 12.0, 13.0]))

    query = np.array([ids[2], ids[0], ids[2], ids[3]])
    lo, hi = wp.get_col_bounds(query)
    assert np.array_equal(lo, np.array([2.0, 0.0, 2.0, 3.0]))
    assert np.array_equal(hi, np.array([12.0, 10.0, 12.0, 13.0]))


def test_get_col_bounds_empty() -> None:
    wp, _ = _solved_warm()
    lo, hi = wp.get_col_bounds(np.zeros(0, dtype=np.int64))
    assert lo.size == 0 and hi.size == 0


def test_get_col_bounds_out_of_range_raises() -> None:
    wp, _ = _solved_warm()
    with pytest.raises(ValueError, match="get_col_bounds"):
        wp.get_col_bounds(np.array([10_000], dtype=np.int64))


def test_set_col_bounds_length_mismatch_raises() -> None:
    wp, _ = _solved_warm()
    ids = _x_col_ids(wp)
    with pytest.raises(ValueError, match="set_col_bounds"):
        wp.set_col_bounds(ids, np.zeros(2), np.zeros(4))


def test_bounds_accessors_require_built() -> None:
    wp = fp.WarmProblem(_build_problem())
    with pytest.raises(RuntimeError, match="must call solve"):
        wp.get_col_bounds(np.array([0]))
    with pytest.raises(RuntimeError, match="must call solve"):
        wp.set_col_bounds(np.array([0]), np.array([0.0]), np.array([1.0]))
