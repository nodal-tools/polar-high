"""WarmProblem incremental cutting-plane primitive tests.

Acceptance gate for :meth:`WarmProblem.add_cut_row` (and the symmetric
:meth:`WarmProblem.add_recourse_col`) — the generic post-build append a
flextool ``BendersMaster`` drives.  Each expected value is hand-verified on
a tiny LP, so the assertions are exact, not self-fulfilling.

Tiny master LP (mirrors the Benders master shape: a trade var + a recourse
``eta``):

    min  x + eta
    s.t. x >= 2          (build-time row "xlb")
         0 <= x <= 10
         0 <= eta <= 1e9  (eta floored at 0 so iter-0 is bounded)

iter-0 optimum: x=2, eta=0, obj=2.

An optimality cut ``eta + x >= K`` (the Benders form ``eta >= K - x``) is
then appended as a NEW HiGHS row and the master re-solved warm.
"""

from __future__ import annotations

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


def test_iter0_baseline() -> None:
    """The hand-verified iter-0 optimum."""
    wp, sol0, x_col, eta_col = _solve_master()
    assert sol0.optimal
    assert sol0.obj == pytest.approx(2.0, abs=1e-9)
    assert sol0.col_value[x_col] == pytest.approx(2.0, abs=1e-9)
    assert sol0.col_value[eta_col] == pytest.approx(0.0, abs=1e-9)
    assert wp._h.getNumRow() == 1


def test_binding_cut_moves_optimum() -> None:
    """Append a cut that MUST bind (cuts off the current optimum); the
    optimum moves to the hand-verified value, the objective increases
    (monotone, min problem), getNumRow grows, the row dual reads back by
    the returned row_id, and column values still read by col_id."""
    wp, sol0, x_col, eta_col = _solve_master()
    n_rows_before = wp._h.getNumRow()

    # Cut: eta >= 5 - x  <=>  eta + x >= 5.  At (x=2, eta=0): 2 < 5 → binds.
    row_id = wp.add_cut_row([eta_col, x_col], [1.0, 1.0], 5.0)
    assert row_id == n_rows_before  # HiGHS assigns the next row index
    assert wp._h.getNumRow() == n_rows_before + 1
    assert wp._n_rows == n_rows_before + 1
    assert wp._row_names[row_id] == f"benders_cut_{row_id}"

    sol1 = wp.solve()
    assert sol1.optimal
    # New optimum: x=2 (still pinned by x>=2), eta = 5 - 2 = 3, obj = 5.
    assert sol1.obj == pytest.approx(5.0, abs=1e-9)
    assert sol1.obj > sol0.obj  # monotone increase for a binding cut (min)
    assert sol1.col_value[x_col] == pytest.approx(2.0, abs=1e-9)
    assert sol1.col_value[eta_col] == pytest.approx(3.0, abs=1e-9)
    # Binding row → unit dual (one extra unit of RHS costs one obj unit).
    assert sol1.row_dual[row_id] == pytest.approx(1.0, abs=1e-9)


def test_nonbinding_cut_leaves_optimum_and_zero_dual() -> None:
    """A cut that is already satisfied at the current optimum leaves the
    objective UNCHANGED and the new row's dual is 0 (slack)."""
    wp, _sol0, x_col, eta_col = _solve_master()

    # First make eta+x>=5 the active optimum (obj 5, eta 3).
    wp.add_cut_row([eta_col, x_col], [1.0, 1.0], 5.0)
    sol_active = wp.solve()
    assert sol_active.obj == pytest.approx(5.0, abs=1e-9)

    # Non-binding cut: eta + x >= 1 — already satisfied (current LHS = 5).
    row_id = wp.add_cut_row([eta_col, x_col], [1.0, 1.0], 1.0)
    sol = wp.solve()
    assert sol.optimal
    assert sol.obj == pytest.approx(5.0, abs=1e-9)  # unchanged
    assert sol.row_dual[row_id] == pytest.approx(0.0, abs=1e-9)  # slack


def test_bootstrap_monotone_lower_bound() -> None:
    """Bootstrap-style sequence: append two cuts across two iterations and
    assert the master objective (the Benders LOWER BOUND) is monotone
    non-decreasing — the LB-monotonicity self-check the BendersMaster
    relies on."""
    wp, sol0, x_col, eta_col = _solve_master()
    lbs = [sol0.obj]  # iter-0 LB = 2

    # Iteration 1 cut: eta + x >= 5  → LB becomes 5.
    wp.add_cut_row([eta_col, x_col], [1.0, 1.0], 5.0)
    lbs.append(wp.solve().obj)

    # Iteration 2 cut: eta + x >= 7  (tighter) → LB becomes 7 (x=2, eta=5).
    wp.add_cut_row([eta_col, x_col], [1.0, 1.0], 7.0)
    sol2 = wp.solve()
    lbs.append(sol2.obj)

    assert lbs == pytest.approx([2.0, 5.0, 7.0], abs=1e-9)
    # Monotone non-decreasing across iterations.
    for a, b in zip(lbs, lbs[1:]):
        assert b >= a - 1e-9
    assert sol2.col_value[eta_col] == pytest.approx(5.0, abs=1e-9)


def test_add_recourse_col_then_cut() -> None:
    """A lazily-appended recourse column is reachable from a later cut row;
    its value reads by col_id and the binding cut's dual reads by row_id."""
    wp, _sol0, _x_col, _eta_col = _solve_master()
    n_cols_before = wp._h.getNumCol()

    # New column z: cost 2, 0 <= z <= 3.
    z_col = wp.add_recourse_col("zextra", cost=2.0, lower=0.0, upper=3.0)
    assert z_col == n_cols_before
    assert wp._h.getNumCol() == n_cols_before + 1
    assert wp._n_cols == n_cols_before + 1
    assert wp._col_names[z_col] == "zextra"

    # Cut over the new column only: z >= 1.5  → z=1.5, obj += 2*1.5 = 3.
    row_id = wp.add_cut_row([z_col], [1.0], 1.5)
    sol = wp.solve()
    assert sol.optimal
    assert sol.col_value[z_col] == pytest.approx(1.5, abs=1e-9)
    # iter-0 obj was 2; adding 2*1.5 = 3 gives 5.
    assert sol.obj == pytest.approx(5.0, abs=1e-9)
    assert sol.row_dual[row_id] == pytest.approx(2.0, abs=1e-9)  # = cost of z


def test_length_mismatch_raises() -> None:
    """col_ids / coefs length mismatch is a hard error, not silent."""
    wp, _sol0, x_col, eta_col = _solve_master()
    with pytest.raises(ValueError, match="length mismatch"):
        wp.add_cut_row([x_col, eta_col], [1.0], 3.0)
