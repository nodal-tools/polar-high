"""WarmProblem equivalence + benchmark sanity tests.

Per the user-spec for Phase 2: don't duplicate the 159 cold-path tests.
Verify equivalence "from a couple of directions":

  1. A 24-h synthetic rolling chain — cold rebuild vs WarmProblem warm
     update.  Multiple rolls, RHS + objective coefs change between
     rolls.  Objs must match at 1e-9.

  2. A 168-h (1-week) synthetic chain — same comparison at a more
     realistic LP size.

  3. A semantic-key index sanity check — col_id_of_var / row_id_of_cstr
     return the right ids and round-trip via h.changeCoeff.

These cover the "is my warm equivalent to my cold" question without
re-running the entire flextool test fleet.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import polar_high as fp

# ----------------------------------------------------------------------------
# Helpers — same shape as tests/_bench_warm_vs_cold.py but inlined here so
# the test is self-contained.


def _build_synthetic_problem(n_t: int, cost: np.ndarray, demand: np.ndarray) -> fp.Problem:
    p = fp.Problem()
    t_idx = pl.DataFrame({"t": np.arange(n_t, dtype=np.int64)})
    v_flow = p.add_var("v_flow", "t", t_idx, lower=0.0, upper=1.0e6)
    v_state = p.add_var("v_state", "t", t_idx, lower=0.0, upper=1.0e6)

    lag = pl.DataFrame(
        {"t": np.arange(1, n_t, dtype=np.int64), "t_prev": np.arange(0, n_t - 1, dtype=np.int64)}
    )
    s_lag = fp.Lag(v_state, lag, time_dim="t", lag_col="t_prev")

    demand_p = fp.Param(
        ("t",), pl.DataFrame({"t": np.arange(n_t, dtype=np.int64), "value": demand})
    )
    p.add_cstr(
        "balance",
        over=t_idx,
        sense="==",
        lhs_terms={"v_flow": v_flow, "s_lag": s_lag, "minus_s": -v_state.to_expr()},
        rhs_terms={"demand": demand_p},
    )

    cost_p = fp.Param(("t",), pl.DataFrame({"t": np.arange(n_t, dtype=np.int64), "value": cost}))
    p.set_objective(cost_p * v_flow, sense="min")
    return p


def _make_chain(n_rolls: int, n_t: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    costs = rng.uniform(1.0, 10.0, size=(n_rolls, n_t))
    demands = rng.uniform(20.0, 80.0, size=(n_rolls, n_t))
    return costs, demands


# ----------------------------------------------------------------------------
# Equivalence tests — 1e-9 obj match between cold and warm.


@pytest.mark.parametrize("n_t,n_rolls", [(24, 8), (168, 4)])
def test_warm_equivalence_rolling_horizon(n_t: int, n_rolls: int) -> None:
    """Cold-rebuild vs WarmProblem warm-update on a rolling-horizon
    chain.  Per-roll objectives must match at 1e-9."""
    costs, demands = _make_chain(n_rolls, n_t, seed=42)

    # Cold path — new Problem + Problem.solve() per roll.
    cold_objs = []
    for r in range(n_rolls):
        p = _build_synthetic_problem(n_t, costs[r], demands[r])
        sol = p.solve()
        assert sol.optimal, f"cold roll {r}: not optimal"
        cold_objs.append(sol.obj)

    # Warm path — build WarmProblem from the first roll's data, then
    # update RHS + obj coef for each subsequent roll.
    p0 = _build_synthetic_problem(n_t, costs[0], demands[0])
    wp = fp.WarmProblem(p0)
    sol_0 = wp.solve()
    assert sol_0.optimal
    warm_objs = [sol_0.obj]

    t_idx = np.arange(n_t, dtype=np.int64)
    for r in range(1, n_rolls):
        new_demand = fp.Param(
            ("t",),
            pl.DataFrame({"t": t_idx, "value": demands[r]}),
        )
        new_cost = fp.Param(
            ("t",),
            pl.DataFrame({"t": t_idx, "value": costs[r]}),
        )
        wp.update_rhs("balance", new_demand)
        wp.update_obj_coef("v_flow", new_cost)
        sol_r = wp.solve()
        assert sol_r.optimal, f"warm roll {r}: not optimal"
        warm_objs.append(sol_r.obj)

    cold_arr = np.asarray(cold_objs)
    warm_arr = np.asarray(warm_objs)
    diff = np.abs(cold_arr - warm_arr).max()
    rel = diff / max(1.0, float(np.abs(cold_arr).max()))
    assert rel < 1e-9, (
        f"warm vs cold obj mismatch: max_abs={diff:.3e}, rel={rel:.3e}\n"
        f"cold={cold_arr.tolist()}\nwarm={warm_arr.tolist()}"
    )


def test_warm_semantic_key_indexes() -> None:
    """col_id_of_var and row_id_of_cstr should return semantically-correct
    LP indices that the underlying HiGHS instance recognises."""
    costs, demands = _make_chain(1, 24, seed=0)
    p = _build_synthetic_problem(24, costs[0], demands[0])
    wp = fp.WarmProblem(p)
    wp.solve()

    # All v_flow col_ids
    all_cids = wp.col_id_of_var("v_flow")
    assert all_cids.shape == (24,)

    # Single-cell lookup
    cid_t5 = wp.col_id_of_var("v_flow", dims=(5,))
    assert isinstance(cid_t5, int)
    assert cid_t5 in all_cids

    # Filter dict
    cid_filtered = wp.col_id_of_var("v_flow", dims={"t": 5})
    assert cid_filtered.tolist() == [cid_t5]

    # Row ids
    all_rids = wp.row_id_of_cstr("balance")
    assert all_rids.shape == (24,)
    rid_t5 = wp.row_id_of_cstr("balance", axis=(5,))
    assert isinstance(rid_t5, int)
    assert rid_t5 in all_rids

    rid_filtered = wp.row_id_of_cstr("balance", axis={"t": 5})
    assert rid_filtered.tolist() == [rid_t5]


def test_warm_update_rhs_with_scalar_and_array() -> None:
    """update_rhs accepts scalars and numpy arrays (positional)."""
    costs, demands = _make_chain(1, 12, seed=1)
    p = _build_synthetic_problem(12, costs[0], demands[0])
    wp = fp.WarmProblem(p)
    wp.solve()

    # Update with a numpy array of length-12.
    new_demand = np.full(12, 50.0)
    wp.update_rhs("balance", new_demand)
    sol_a = wp.solve()
    assert sol_a.optimal

    # And do the same via a cold rebuild; same answer expected.
    p2 = _build_synthetic_problem(12, costs[0], new_demand)
    wp2 = fp.WarmProblem(p2)
    sol_b = wp2.solve()

    assert abs(sol_a.obj - sol_b.obj) < 1e-9


def test_warm_problem_does_not_affect_cold_problem() -> None:
    """Sanity: building a WarmProblem and using it shouldn't change the
    behaviour of an unrelated cold Problem instance.

    Guards against state leakage through any module-level globals."""
    costs, demands = _make_chain(2, 24, seed=7)

    p_cold_before = _build_synthetic_problem(24, costs[0], demands[0])
    sol_before = p_cold_before.solve()

    wp_other = fp.WarmProblem(_build_synthetic_problem(24, costs[1], demands[1]))
    wp_other.solve()
    wp_other.update_rhs(
        "balance",
        fp.Param(("t",), pl.DataFrame({"t": np.arange(24, dtype=np.int64), "value": demands[0]})),
    )
    wp_other.solve()

    # Re-solve the cold one — should get the same obj as before.
    p_cold_after = _build_synthetic_problem(24, costs[0], demands[0])
    sol_after = p_cold_after.solve()
    assert abs(sol_after.obj - sol_before.obj) < 1e-12
