"""WarmProblem cut-compaction tests.

Acceptance gate for :meth:`WarmProblem.compact_cuts` and its internal
helper :meth:`WarmProblem._delete_cut_rows` — the generic cutting-plane
compaction a flextool ``BendersMaster`` drives to bound its growing master
LP.  This is polar-high-only; NO flextool import.

Correctness invariant under test: removing a cut that is strictly slack
(positive primal slack) at the incumbent optimum is LB-neutral — the master
re-solves to the SAME objective — so a subsequent solve stays optimal and
the build-time rows / their duals are untouched.

Tiny master LP (mirrors :mod:`test_warm_cut_append`):

    min  x + eta
    s.t. x >= 2          (build-time row "xlb")
         0 <= x <= 10
         0 <= eta <= 1e9

iter-0 optimum: x=2, eta=0, obj=2.  After the cut ``eta + x >= 5`` binds the
optimum is x=2, eta=3, obj=5 — at which any cut ``eta + x >= k`` with k < 5
is strictly slack (LHS = 5 > k).
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import polar_high as fp


def _build_master() -> fp.Problem:
    """min x + eta s.t. x >= 2, 0<=x<=10, 0<=eta<=1e9."""
    p = fp.Problem()
    idx = pl.DataFrame({"i": [0]})
    x = p.add_var("x", "i", idx, lower=0.0, upper=10.0)
    eta = p.add_var("eta", "i", idx, lower=0.0, upper=1.0e9)
    p.add_cstr(
        "xlb",
        over=idx,
        sense=">=",
        lhs_terms={"x": x},
        rhs_terms={"two": fp.Param((), pl.DataFrame({"value": [2.0]}))},
    )
    p.set_objective(x.to_expr() + eta.to_expr(), sense="min")
    return p


def _solve_master():
    """Return (wp, sol0, x_col, eta_col) at the iter-0 optimum."""
    wp = fp.WarmProblem(_build_master())
    sol0 = wp.solve()
    x_col = wp.col_id_of_var("x", (0,))
    eta_col = wp.col_id_of_var("eta", (0,))
    return wp, sol0, x_col, eta_col


def _build_x_only_master() -> fp.Problem:
    """min x s.t. x >= 2, 0<=x<=10 (no build-time eta; the recourse column is
    appended post-build via ``add_recourse_col`` so it registers in
    ``_recourse_cols`` and drives the dominance grouping)."""
    p = fp.Problem()
    idx = pl.DataFrame({"i": [0]})
    x = p.add_var("x", "i", idx, lower=0.0, upper=10.0)
    p.add_cstr(
        "xlb",
        over=idx,
        sense=">=",
        lhs_terms={"x": x},
        rhs_terms={"two": fp.Param((), pl.DataFrame({"value": [2.0]}))},
    )
    p.set_objective(x.to_expr(), sense="min")
    return p


# --------------------------------------------------------------------------
# Retention plumbing
# --------------------------------------------------------------------------


def test_add_cut_row_retains_cut() -> None:
    """Every appended cut is stored in ``_cut_rows`` keyed by live row id,
    and the first-cut boundary is fixed at the first append."""
    wp, _sol0, x_col, eta_col = _solve_master()
    assert wp._cut_rows == {}
    assert wp._first_cut_row is None

    r1 = wp.add_cut_row([eta_col, x_col], [1.0, 1.0], 5.0)
    assert wp._first_cut_row == r1
    assert wp._cut_rows[r1] == ([eta_col, x_col], [1.0, 1.0], 5.0)

    r2 = wp.add_cut_row([eta_col, x_col], [1.0, 1.0], 1.0)
    assert wp._first_cut_row == r1  # unchanged
    assert wp._cut_rows[r2] == ([eta_col, x_col], [1.0, 1.0], 1.0)
    assert set(wp._cut_rows) == {r1, r2}


# --------------------------------------------------------------------------
# Core compaction: drop strictly-slack, keep binding
# --------------------------------------------------------------------------


def test_compact_drops_slack_keeps_binding() -> None:
    """With a binding cut and a strictly-slack cut at the optimum, compaction
    drops the slack one, keeps the binding one, keeps the objective, and
    leaves all bookkeeping consistent."""
    wp, _sol0, x_col, eta_col = _solve_master()

    # Binding cut: eta + x >= 5  → optimum x=2, eta=3, obj=5.
    r_bind = wp.add_cut_row([eta_col, x_col], [1.0, 1.0], 5.0)
    # Strictly slack at that optimum (LHS = 5 > 1).
    wp.add_cut_row([eta_col, x_col], [1.0, 1.0], 1.0)

    sol = wp.solve()
    assert sol.obj == pytest.approx(5.0, abs=1e-9)
    n_rows_before = wp._h.getNumRow()
    assert n_rows_before == 3  # xlb + 2 cuts

    res = wp.compact_cuts(sol)
    assert res == {"kept": 1, "dropped": 1, "restored": False}

    # HiGHS row count dropped by exactly one.
    assert wp._h.getNumRow() == n_rows_before - 1
    assert wp._n_rows == n_rows_before - 1
    # Only the binding cut remains, remapped to its (unchanged) id: r_slack
    # was appended AFTER r_bind, so deleting it leaves r_bind's id fixed.
    assert set(wp._cut_rows) == {r_bind}
    assert wp._cut_rows[r_bind] == ([eta_col, x_col], [1.0, 1.0], 5.0)
    # _row_names index-aligned to surviving rows.
    assert wp._row_names is not None
    assert len(wp._row_names) == wp._h.getNumRow()
    assert wp._row_names[r_bind] == f"benders_cut_{r_bind}"

    # LB-neutral: re-solve is optimal with the SAME objective.
    sol2 = wp.solve()
    assert sol2.optimal
    assert sol2.obj == pytest.approx(5.0, abs=1e-9)
    assert sol2.col_value[x_col] == pytest.approx(2.0, abs=1e-9)
    assert sol2.col_value[eta_col] == pytest.approx(3.0, abs=1e-9)
    # Surviving cut's constraint still holds at the fresh solve.
    ax = sol2.col_value[eta_col] + sol2.col_value[x_col]
    assert ax >= 5.0 - 1e-9


def test_build_time_row_and_dual_unchanged_after_compaction() -> None:
    """Build-time row "xlb" keeps its position; its named dual reads the same
    before and after compaction (the guard against corrupting named rows)."""
    wp, _sol0, x_col, eta_col = _solve_master()
    wp.add_cut_row([eta_col, x_col], [1.0, 1.0], 5.0)
    wp.add_cut_row([eta_col, x_col], [1.0, 1.0], 1.0)
    sol = wp.solve()
    dual_before = sol.constraint_dual("xlb")

    wp.compact_cuts(sol)
    sol2 = wp.solve()
    dual_after = sol2.constraint_dual("xlb")

    assert dual_before.columns == dual_after.columns
    assert dual_before["dual"].to_list() == pytest.approx(dual_after["dual"].to_list(), abs=1e-9)


def test_compact_noop_when_all_binding() -> None:
    """When no cut is strictly slack, compaction is a no-op (no delete, no
    re-solve mutation)."""
    wp, _sol0, x_col, eta_col = _solve_master()
    # Two cuts, both binding at their combined optimum: eta+x>=5 and eta>=3.
    # At x=2, eta=3 both are tight (5 and 3 respectively).
    wp.add_cut_row([eta_col, x_col], [1.0, 1.0], 5.0)
    wp.add_cut_row([eta_col], [1.0], 3.0)
    sol = wp.solve()
    assert sol.obj == pytest.approx(5.0, abs=1e-9)
    n_rows_before = wp._h.getNumRow()
    cut_rows_before = dict(wp._cut_rows)

    res = wp.compact_cuts(sol)
    assert res == {"kept": 2, "dropped": 0, "restored": False}
    assert wp._h.getNumRow() == n_rows_before
    assert wp._cut_rows == cut_rows_before


# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------


def test_delete_build_time_row_raises() -> None:
    """_delete_cut_rows refuses to delete a build-time row id (< first cut)."""
    wp, _sol0, x_col, eta_col = _solve_master()
    wp.add_cut_row([eta_col, x_col], [1.0, 1.0], 5.0)
    wp.solve()
    # Row 0 is the build-time "xlb" row.
    with pytest.raises(AssertionError, match="build-time"):
        wp._delete_cut_rows([0])


def test_delete_before_any_cut_raises() -> None:
    """With no cut appended (_first_cut_row is None) deletion is refused."""
    wp, _sol0, _x_col, _eta_col = _solve_master()
    assert wp._first_cut_row is None
    with pytest.raises(AssertionError, match="build-time"):
        wp._delete_cut_rows([0])


def test_mutable_params_guard_raises() -> None:
    """Row deletion is refused on a WarmProblem with tracked mutable Params,
    whose stored absolute row indices the HiGHS compaction would corrupt."""
    wp, _sol0, x_col, eta_col = _solve_master()
    r = wp.add_cut_row([eta_col, x_col], [1.0, 1.0], 5.0)
    wp.solve()
    # Simulate a param-tracked master (the real _param_cells is built during
    # _initial_build; the guard only inspects non-emptiness).
    wp._mutable_params = {"p_dummy"}
    with pytest.raises(RuntimeError, match="tracked mutable Params"):
        wp._delete_cut_rows([r])
    wp._mutable_params = set()

    wp._param_cells = {"p_dummy": {}}
    with pytest.raises(RuntimeError, match="tracked mutable Params"):
        wp._delete_cut_rows([r])


# --------------------------------------------------------------------------
# Remap: interleave add / compact / add and check id consistency
# --------------------------------------------------------------------------


def test_remap_interleaved_add_compact_add() -> None:
    """Add several cuts, compact (deleting an EARLY slack cut so surviving ids
    shift down), add more, and assert ``_cut_rows`` keys equal the live row
    ids and each stored cut's constraint value matches at a fresh solve."""
    wp, _sol0, x_col, eta_col = _solve_master()

    # Append order: a strictly-slack low cut FIRST, then the binding one.
    # After compaction the binding cut must shift down by one (HiGHS compacts).
    r_low = wp.add_cut_row([eta_col, x_col], [1.0, 1.0], 1.0)  # slack at opt 5
    r_bind = wp.add_cut_row([eta_col, x_col], [1.0, 1.0], 5.0)  # binding
    assert r_bind == r_low + 1

    sol = wp.solve()
    assert sol.obj == pytest.approx(5.0, abs=1e-9)

    res = wp.compact_cuts(sol)
    assert res == {"kept": 1, "dropped": 1, "restored": False}

    # The surviving binding cut shifted from r_bind down to r_low (one deleted
    # id, r_low, was below it).
    assert set(wp._cut_rows) == {r_low}
    assert wp._cut_rows[r_low] == ([eta_col, x_col], [1.0, 1.0], 5.0)

    # Every _cut_rows key is a live HiGHS row id (< getNumRow) and >= the
    # build-time boundary.
    n_rows = wp._h.getNumRow()
    for rid in wp._cut_rows:
        assert wp._first_cut_row <= rid < n_rows

    # Add a NEW cut after compaction; it lands at the next live id.
    sol_mid = wp.solve()
    r_new = wp.add_cut_row([eta_col, x_col], [1.0, 1.0], 7.0)  # tighter → binds
    assert r_new == n_rows
    assert r_new in wp._cut_rows

    sol2 = wp.solve()
    assert sol2.optimal
    assert sol2.obj == pytest.approx(7.0, abs=1e-9)  # x=2, eta=5

    # Every stored cut's constraint holds at the fresh solve, and each key is
    # a consistent live row id.
    n_rows2 = wp._h.getNumRow()
    x_val = sol2.col_value
    for rid, (col_ids, coefs, lower) in wp._cut_rows.items():
        assert wp._first_cut_row <= rid < n_rows2
        assert len(wp._row_names) == n_rows2
        ax = sum(c * x_val[i] for i, c in zip(col_ids, coefs))
        assert ax >= lower - 1e-9

    # sol_mid was optimal at obj 5 before the tighter cut (sanity on ordering).
    assert sol_mid.obj == pytest.approx(5.0, abs=1e-9)


# --------------------------------------------------------------------------
# Verify-restore belt
# --------------------------------------------------------------------------


def test_verify_restore_rolls_back_supporting_cut() -> None:
    """Force the degenerate edge: a cut that is (wrongly) classified as slack
    but actually supports the optimum.  We drive it by hand-deleting a BINDING
    cut through the low-level path and confirming the objective would move —
    then exercise the full ``compact_cuts`` belt on a constructed case where
    an over-wide ``tol_rel`` mis-classifies the binding cut as slack, so the
    belt must fire and restore.

    The binding cut ``eta + x >= 5`` has slack 0 at the optimum; with a huge
    ``tol_rel`` the (also-present) cut ``eta + x >= 4.9999999`` — whose slack
    is ~1e-7 — is treated as binding, but if we instead make the ONLY
    supporting cut mis-classify, deleting it moves the optimum and the belt
    restores it.
    """
    wp, _sol0, x_col, eta_col = _solve_master()
    # Single supporting cut: eta + x >= 5.  Optimum obj 5.
    r_bind = wp.add_cut_row([eta_col, x_col], [1.0, 1.0], 5.0)
    sol = wp.solve()
    assert sol.obj == pytest.approx(5.0, abs=1e-9)

    # At the optimum this cut's slack is ~0.  A pathologically NEGATIVE
    # effective classification can't happen through tol, so instead force the
    # mis-classification by perturbing the solution we hand to compact_cuts so
    # the binding cut reads as strictly slack (its stored coefs vs a shifted x).
    perturbed = fp.Solution(
        optimal=True,
        obj=sol.obj,
        col_value=sol.col_value.copy(),
        row_dual=sol.row_dual,
        col_names=sol.col_names,
        row_names=sol.row_names,
        vars=sol._vars,
        highs=None,
    )
    # Inflate eta in the handed-in solution so LHS = eta+x reads far above 5,
    # making the binding cut look strictly slack → it becomes a drop candidate.
    perturbed.col_value = perturbed.col_value.copy()
    perturbed.col_value[eta_col] = perturbed.col_value[eta_col] + 100.0

    res = wp.compact_cuts(perturbed)
    # The cut was truly supporting; deleting it drops the optimum (5 → 2), so
    # the verify belt detects the drift and restores it.
    assert res == {"kept": 1, "dropped": 0, "restored": True}
    assert set(wp._cut_rows) == {r_bind}

    # After restore the master re-solves to the original optimum.
    sol2 = wp.solve()
    assert sol2.optimal
    assert sol2.obj == pytest.approx(5.0, abs=1e-9)


def test_verify_disabled_skips_resolve_belt() -> None:
    """With verify=False the belt does not fire; a normal (correct) slack drop
    still leaves the LP optimal at the same objective."""
    wp, _sol0, x_col, eta_col = _solve_master()
    wp.add_cut_row([eta_col, x_col], [1.0, 1.0], 5.0)
    wp.add_cut_row([eta_col, x_col], [1.0, 1.0], 1.0)  # strictly slack
    sol = wp.solve()

    res = wp.compact_cuts(sol, verify=False)
    assert res == {"kept": 1, "dropped": 1, "restored": False}
    sol2 = wp.solve()
    assert sol2.optimal
    assert sol2.obj == pytest.approx(5.0, abs=1e-9)


def test_deleterows_api_and_basis_reoptimises() -> None:
    """Empirically verify the highspy ``deleteRows`` call and that, after a
    delete + solve, the model returns optimal and every surviving cut holds —
    the API/basis contract the compaction relies on."""
    wp, _sol0, x_col, eta_col = _solve_master()
    ids = []
    ids.append(wp.add_cut_row([eta_col, x_col], [1.0, 1.0], 3.0))
    ids.append(wp.add_cut_row([eta_col, x_col], [1.0, 1.0], 4.0))
    ids.append(wp.add_cut_row([eta_col, x_col], [1.0, 1.0], 5.0))  # tightest
    sol = wp.solve()
    assert sol.obj == pytest.approx(5.0, abs=1e-9)

    # Delete the two looser cuts (ids[0], ids[1]) directly; both strictly slack.
    wp._delete_cut_rows([ids[0], ids[1]])
    assert wp._h.getNumRow() == 2  # xlb + tightest cut

    sol2 = wp.solve()
    assert sol2.optimal
    assert sol2.obj == pytest.approx(5.0, abs=1e-9)
    # Surviving cut (was ids[2]) remapped to ids[2]-2.
    surviving = ids[2] - 2
    assert set(wp._cut_rows) == {surviving}
    assert wp._cut_rows[surviving] == ([eta_col, x_col], [1.0, 1.0], 5.0)
    ax = sol2.col_value[eta_col] + sol2.col_value[x_col]
    assert ax >= 5.0 - 1e-9
    # np import used for the guard-array type check the primitive performs.
    assert np.asarray([surviving], dtype=np.int32).dtype == np.int32


# --------------------------------------------------------------------------
# Dominance policy
# --------------------------------------------------------------------------


def _solve_recourse_master():
    """Return (wp, sol0, x_col, eta_col) for the x-only master with a recourse
    ``η`` column appended (registered in ``_recourse_cols``).

    Master: min x + η  s.t. x >= 2.  ``η`` is free-lower (≥ -1e9) so the cuts
    alone bound it from below (mirrors a Benders recourse variable)."""
    wp = fp.WarmProblem(_build_x_only_master())
    wp.solve()  # must build once before add_recourse_col / add_cut_row
    x_col = wp.col_id_of_var("x", (0,))
    # η enters the objective with cost 1.0 (min x + η); lower −1e9 so the cuts
    # provide its only meaningful lower bound.
    eta_col = wp.add_recourse_col("eta_r", cost=1.0, lower=-1.0e9, upper=1.0e9)
    assert eta_col in wp._recourse_cols
    sol0 = wp.solve()
    return wp, sol0, x_col, eta_col


def test_add_recourse_col_registers_recourse():
    """``add_recourse_col`` records the appended col id in ``_recourse_cols``
    (the grouping key for the dominance policy)."""
    wp = fp.WarmProblem(_build_x_only_master())
    wp.solve()
    assert wp._recourse_cols == set()
    c1 = wp.add_recourse_col("eta_a", cost=1.0, lower=-1.0e9)
    c2 = wp.add_recourse_col("eta_b", cost=1.0, lower=-1.0e9)
    assert wp._recourse_cols == {c1, c2}


def test_dominance_drops_dominated_and_ties():
    """At a single trial point, dominance keeps only the OLDEST cut imposing the
    tightest bound on η per recourse group and drops BOTH the strictly-dominated
    cuts AND the redundant TIES that slack-deletion would keep.

    Cuts on recourse group η, stored coefs on [η, x] as ``η + coef_x·x >= rhs``
    (Benders form ``η − slope·x >= rhs`` with slope = −coef_x).  The bound each
    cut imposes on η at trial point x_t is ``d = rhs − coef_x·x_t`` (the recourse
    column's own term excluded).  At x=2 (the master optimum, x≥2):

      - c0: coefs [1, 1], rhs 5  → d = 5 − 2 = 3   (group max, oldest → KEEP)
      - c1: coefs [1, 1], rhs 5  → d = 3           (exact TIE with c0 → DROP)
      - c2: coefs [1, 1], rhs 4  → d = 2           (dominated, looser → DROP)

    All three cuts are primal-binding-or-slack at the optimum; slack-deletion
    keeps c0 AND c1 (the tie) — dominance keeps only c0.
    """
    wp, _sol0, x_col, eta_col = _solve_recourse_master()
    c0 = wp.add_cut_row([eta_col, x_col], [1.0, 1.0], 5.0)
    c1 = wp.add_cut_row([eta_col, x_col], [1.0, 1.0], 5.0)  # exact tie with c0
    c2 = wp.add_cut_row([eta_col, x_col], [1.0, 1.0], 4.0)  # dominated (looser)
    sol = wp.solve()
    # Optimum: x=2, η=3, obj = x + η = 5.
    assert sol.obj == pytest.approx(5.0, abs=1e-9)
    assert set(wp._cut_rows) == {c0, c1, c2}

    res = wp.compact_cuts(sol, policy="dominance")
    # Only the oldest group-max achiever (c0) survives; the tie (c1) and the
    # dominated cut (c2) are both dropped.
    assert res == {"kept": 1, "dropped": 2, "restored": False}
    assert set(wp._cut_rows) == {c0}
    assert wp._cut_rows[c0] == ([eta_col, x_col], [1.0, 1.0], 5.0)

    # LB-neutral: the master re-solves to the SAME objective.
    sol2 = wp.solve()
    assert sol2.optimal
    assert sol2.obj == pytest.approx(5.0, abs=1e-9)


def test_dominance_keeps_per_trial_point_keepers():
    """The kept set is EXACTLY the per-trial-point keepers over the window: a
    cut dominated at the latest point but the tightest at an EARLIER window
    point is KEPT (the window matters).

    Two cuts on group η with DIFFERENT slopes so the bound-tightness order flips
    with x.  The bound each imposes on η at trial x is ``d = rhs − coef_x·x``
    (recourse column term excluded); coefs on [η, x]:

      - cA coefs [1, 0], rhs 3   → d = 3     (constant, x-independent)
      - cB coefs [1, -1], rhs 0  → d = x     (grows with x)

    Trial points are (η, x) master vertices.  At a LARGE x, cB imposes the
    tighter bound (d=x > 3); at a NEGATIVE x, cA does (3 > x) — so BOTH are the
    keeper at some window point and BOTH must be kept.  (We drive the window with
    explicit ``trial_col_values``.)
    """
    wp, _sol0, x_col, eta_col = _solve_recourse_master()
    cA = wp.add_cut_row([eta_col, x_col], [1.0, 0.0], 3.0)  # d = 3
    cB = wp.add_cut_row([eta_col, x_col], [1.0, -1.0], 0.0)  # d = x
    sol = wp.solve()

    n = wp._n_cols
    # Trial point 1: x large → cB (d=8) tighter than cA (d=3) → keeper = cB.
    t_x_large = np.zeros(n, dtype=np.float64)
    t_x_large[eta_col] = 10.0
    t_x_large[x_col] = 8.0
    # Trial point 2: x negative → cA (d=3) tighter than cB (d=−8) → keeper = cA.
    t_x_neg = np.zeros(n, dtype=np.float64)
    t_x_neg[eta_col] = 10.0
    t_x_neg[x_col] = -8.0

    # Window has both points (plus the incumbent ``sol`` appended as latest);
    # cB is the keeper at t_x_large, cA at t_x_neg → both kept, none dropped.
    res = wp.compact_cuts(sol, policy="dominance", trial_col_values=[t_x_large, t_x_neg])
    assert res == {"kept": 2, "dropped": 0, "restored": False}
    assert set(wp._cut_rows) == {cA, cB}


def test_dominance_window_matters_classify_level():
    """At the LOW-LEVEL classifier, a cut that is the tightest ONLY at one window
    point (dominated at another) is retained when that point is in the window,
    and becomes a drop candidate when it is not — proving the window drives the
    selection (isolated from the incumbent-always-included belt).

    cA d=3 (constant); cB d=x.  At x large cB is tighter; at x negative cA is."""
    wp, _sol0, x_col, eta_col = _solve_recourse_master()
    cA = wp.add_cut_row([eta_col, x_col], [1.0, 0.0], 3.0)  # d = 3
    cB = wp.add_cut_row([eta_col, x_col], [1.0, -1.0], 0.0)  # d = x
    wp.solve()

    n = wp._n_cols
    t_x_large = np.zeros(n, dtype=np.float64)
    t_x_large[eta_col] = 10.0
    t_x_large[x_col] = 8.0  # cB tighter (d=8 > cA d=3)
    t_x_neg = np.zeros(n, dtype=np.float64)
    t_x_neg[eta_col] = 10.0
    t_x_neg[x_col] = -8.0  # cA tighter (d=3 > cB d=−8)

    # Build a fake solution whose col_value == t_x_neg so the incumbent (added as
    # the latest trial point) is a point where cA — not cB — is the keeper.
    sol_neg = fp.Solution(
        optimal=True,
        obj=0.0,
        col_value=t_x_neg.copy(),
        row_dual=None,
        col_names=None,
        row_names=None,
        vars=None,
        highs=None,
    )

    # Window includes t_x_large where cB is tighter → cB is a keeper → not
    # dropped; cA is the keeper at t_x_neg (and the incumbent).  Nothing dropped.
    drops_with_large = wp._classify_dominance_drops(sol_neg, [t_x_large, t_x_neg], 1e-7)
    assert drops_with_large == []

    # Window WITHOUT t_x_large → cB is dominated at every remaining trial point
    # (t_x_neg + the incumbent, both x negative) → cB is dropped, cA kept.
    drops_neg_only = wp._classify_dominance_drops(sol_neg, [t_x_neg], 1e-7)
    assert drops_neg_only == [cB]
    assert cA not in drops_neg_only


def test_dominance_singleton_no_recourse_kept():
    """A cut with NO recourse column forms its own singleton group and is
    always kept (generic non-Benders use degrades safely to 'keep all')."""
    wp, _sol0, x_col, eta_col = _solve_recourse_master()
    # Cut on x ONLY (no recourse col in col_ids).
    r_generic = wp.add_cut_row([x_col], [1.0], 2.0)
    # A recourse-group cut that is a redundant tie with itself is fine; add a
    # binding recourse cut too so the group path is exercised alongside.
    r_rec = wp.add_cut_row([eta_col, x_col], [1.0, 1.0], 5.0)
    sol = wp.solve()

    res = wp.compact_cuts(sol, policy="dominance")
    # Singleton (generic) cut kept; the single recourse cut is its group max →
    # kept. Nothing to drop.
    assert res["restored"] is False
    assert r_generic in wp._cut_rows
    assert r_rec in wp._cut_rows


def test_dominance_default_trial_uses_incumbent():
    """With ``trial_col_values=None`` the incumbent ``solution.col_value`` is
    the sole trial point."""
    wp, _sol0, x_col, eta_col = _solve_recourse_master()
    c0 = wp.add_cut_row([eta_col, x_col], [1.0, 1.0], 5.0)
    wp.add_cut_row([eta_col, x_col], [1.0, 1.0], 5.0)  # tie
    sol = wp.solve()
    res = wp.compact_cuts(sol, policy="dominance", trial_col_values=None)
    assert res == {"kept": 1, "dropped": 1, "restored": False}
    assert set(wp._cut_rows) == {c0}


def test_unknown_policy_raises():
    """An unrecognised policy is rejected."""
    wp, _sol0, x_col, eta_col = _solve_recourse_master()
    wp.add_cut_row([eta_col, x_col], [1.0, 1.0], 5.0)
    sol = wp.solve()
    with pytest.raises(ValueError, match="unknown policy"):
        wp.compact_cuts(sol, policy="bogus")
