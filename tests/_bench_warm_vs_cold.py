"""Phase 1 benchmark — warm-LP-update vs cold-rebuild for a rolling-horizon chain.

This benchmark sizes the speedup before committing to a full ``WarmProblem``
implementation.  Per the user-spec acceptance criterion: a real warm path
must beat the cold path by at least 2x on a meaningful chain to justify the
engineering cost.

Two scenarios:

  1. **synthetic** — a clean, controlled 24-hour rolling-horizon chain
     repeated ``n_rolls`` times.  Same LP structure each roll, only the
     RHS (demand vector) shifts.  This is the cleanest possible warm-vs-cold
     comparison.

  2. **flextool** — the ``multi_fullYear_battery_nested_multi_invest``
     scenario (80 sub-solves of dispatch_fullYear_roll); cold path uses
     ``run_chain``; warm path is not implemented for the flextool case in
     Phase 1 — too much loader plumbing for a sizing exercise.  We only
     measure the cold baseline.

Re-run with::

    PYTHONPATH=src ~/venv-spi/bin/python tests/_bench_warm_vs_cold.py
"""
from __future__ import annotations

import gc
import statistics as stats
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

import highspy
import polar_high_opt as fp


# ----------------------------------------------------------------------------
# Synthetic rolling-horizon chain
#
# Each roll:
#   * 24 timesteps t = 0..23 (relative within the roll).
#   * One variable v_flow[t] (production) and v_state[t] (storage state).
#   * Balance:  v_flow[t] + s[t-1] - s[t] == demand[r, t]  (s[-1] = 0)
#   * State bounds: 0 <= s[t] <= 1e6, 0 <= v_flow[t] <= 1e6.
#   * Objective: minimize sum(c[r, t] * v_flow[t]).
#
# The "roll" index r is the absolute time offset.  Across rolls,
#   - LP STRUCTURE (cols, rows, A) is identical.
#   - RHS (demand) shifts.
#   - Objective coefficients (c) shift.
# This is exactly what a warm path can exploit.

_N_T = 24


def _make_synthetic_chain(n_rolls: int, seed: int = 0):
    """Generate per-roll cost / demand vectors.  Returns:
       costs[r, t], demands[r, t]  — both (n_rolls, _N_T) float64."""
    rng = np.random.default_rng(seed)
    costs = rng.uniform(1.0, 10.0, size=(n_rolls, _N_T))
    demands = rng.uniform(20.0, 80.0, size=(n_rolls, _N_T))
    return costs, demands


def _build_synthetic_problem(cost: np.ndarray, demand: np.ndarray) -> fp.Problem:
    """Build a single 24-hour LP for one roll."""
    p = fp.Problem()
    t_idx = pl.DataFrame({"t": np.arange(_N_T, dtype=np.int64)})
    v_flow = p.add_var("v_flow", "t", t_idx, lower=0.0, upper=1.0e6)
    v_state = p.add_var("v_state", "t", t_idx, lower=0.0, upper=1.0e6)

    lag = pl.DataFrame({"t": np.arange(1, _N_T, dtype=np.int64),
                        "t_prev": np.arange(0, _N_T - 1, dtype=np.int64)})
    s_lag = fp.Lag(v_state, lag, time_dim="t", lag_col="t_prev")

    demand_p = fp.Param(("t",), pl.DataFrame({"t": np.arange(_N_T, dtype=np.int64),
                                              "value": demand}))
    p.add_cstr(
        "balance", over=t_idx, sense="==",
        lhs_terms={"v_flow": v_flow, "s_lag": s_lag, "minus_s": -v_state.to_expr()},
        rhs_terms={"demand": demand_p},
    )

    cost_p = fp.Param(("t",), pl.DataFrame({"t": np.arange(_N_T, dtype=np.int64),
                                            "value": cost}))
    p.set_objective(cost_p * v_flow, sense="min")
    return p


# ----------------------------------------------------------------------------
# Cold path — rebuild from scratch each roll

def cold_run_synthetic(costs: np.ndarray, demands: np.ndarray) -> dict:
    """Solve every roll from scratch via Problem.solve().

    Returns dict: total_seconds, per_roll_build, per_roll_solve, objs."""
    n_rolls = costs.shape[0]
    objs: list[float] = []
    t_build = 0.0
    t_solve = 0.0

    for r in range(n_rolls):
        t0 = time.perf_counter()
        p = _build_synthetic_problem(costs[r], demands[r])
        # NB: Problem.solve() lumps build + solve.  We split using the same
        # time.perf_counter trick as _bench_tied_vs_block.  But we want a
        # legit per-roll wall-clock split.  Approach: instrument by
        # reaching into Problem internals.
        sol = p.solve()
        t1 = time.perf_counter()
        # Approximate split: the LP build is dominated by collect_all + matrix
        # assembly.  We don't separate build/solve here — the salient number
        # is total wall-clock per roll.  Reported as one combined number.
        t_solve += (t1 - t0)
        objs.append(sol.obj)
        del p, sol

    return {
        "total_seconds": t_solve,
        "n_rolls": n_rolls,
        "objs": objs,
    }


# ----------------------------------------------------------------------------
# Warm path — build LP once, hot-update RHS + cost between rolls
#
# This is a manual, surgical proof-of-concept.  We know exactly which rows
# are the demand-balance rows (one per t) and which cols are v_flow[t]
# (one per t).  Between rolls we call:
#
#   h.changeRowsBounds(_N_T, row_idx, new_lb, new_ub)
#   h.changeColsCost(_N_T, col_idx, new_cost)
#   h.run()
#
# No Problem rebuild, no LP rebuild, no passModel — just delta updates.

def warm_run_synthetic(costs: np.ndarray, demands: np.ndarray) -> dict:
    """Build the LP once (using the first roll's data), then hot-update
    RHS + cost for each subsequent roll.

    Returns dict: total_seconds, per_roll_build_first, per_roll_update,
    per_roll_solve, objs."""
    n_rolls = costs.shape[0]
    objs: list[float] = []

    inf = highspy.kHighsInf

    # ---- build initial LP from the first roll ----
    t0 = time.perf_counter()
    p = _build_synthetic_problem(costs[0], demands[0])
    # Reach into Problem internals to get the LP and a handle on the
    # row/col indices we care about.  We lift the build pipeline straight
    # out of Problem.solve() so we end up with a live highspy Highs
    # instance with the LP loaded — and we KEEP the Highs instance for
    # subsequent solves.
    h, balance_row_idx, v_flow_col_idx = _build_and_get_warm_handles(p)
    t_first = time.perf_counter() - t0

    # ---- run roll 0 ----
    t1 = time.perf_counter()
    h.run()
    t_solve_total = time.perf_counter() - t1
    objs.append(h.getObjectiveValue())

    # ---- subsequent rolls: hot-update + run ----
    t_update_total = 0.0
    for r in range(1, n_rolls):
        t_u0 = time.perf_counter()
        # Update RHS for the balance constraint.  Since the balance is an
        # equality (demand[t]), both lower and upper of those rows = demand.
        new_rhs = demands[r].astype(np.float64)
        h.changeRowsBounds(_N_T,
                           balance_row_idx.astype(np.int32),
                           new_rhs, new_rhs)
        # Update cost vector for v_flow columns.
        h.changeColsCost(_N_T,
                         v_flow_col_idx.astype(np.int32),
                         costs[r].astype(np.float64))
        t_u1 = time.perf_counter()
        t_update_total += (t_u1 - t_u0)

        t_s0 = time.perf_counter()
        h.run()
        t_solve_total += (time.perf_counter() - t_s0)
        objs.append(h.getObjectiveValue())

    total = t_first + t_update_total + t_solve_total
    return {
        "total_seconds": total,
        "first_build_seconds": t_first,
        "update_seconds": t_update_total,
        "solve_seconds": t_solve_total,
        "n_rolls": n_rolls,
        "objs": objs,
    }


def _build_and_get_warm_handles(problem: fp.Problem):
    """Run the same pipeline as Problem.solve() up to (and including)
    passModel, but DON'T call h.run().  Return:

      (h, balance_row_idx, v_flow_col_idx)

    where h is a live highspy Highs instance with the LP loaded.  The
    balance_row_idx / v_flow_col_idx arrays let the caller hot-update
    those LP cells."""
    n_cols = problem._next_col
    col_lb = np.zeros(n_cols, dtype=np.float64)
    col_ub = np.full(n_cols, np.inf, dtype=np.float64)
    col_obj = np.zeros(n_cols, dtype=np.float64)

    for v in problem._vars.values():
        ids = v.frame["col_id"].to_numpy()
        col_lb[ids] = float(v.lower)
        col_ub[ids] = float(v.upper)

    for t in problem._obj_terms:
        tf = t.frame
        np.add.at(col_obj, tf["col_id"].to_numpy(), tf["coef"].to_numpy())

    rows_lb_chunks: list[np.ndarray] = []
    rows_ub_chunks: list[np.ndarray] = []
    triple_rows: list[np.ndarray] = []
    triple_cols: list[np.ndarray] = []
    triple_vals: list[np.ndarray] = []
    next_row = 0
    pending: list[tuple] = []

    balance_row_idx = None

    for name, proto, over in problem._cstrs:
        expr, sense, rhs = proto.expr, proto.sense, proto.rhs
        if over is None:
            row_count = 1
            row_index = pl.DataFrame({"_rid": [0]})
            axis_cols: list[str] = []
        else:
            row_count = over.height
            axis_cols = list(over.columns)
            row_index = over.with_columns(
                _rid=pl.int_range(0, over.height, dtype=pl.Int64))
        base_row = next_row
        next_row += row_count
        if name == "balance":
            balance_row_idx = np.arange(base_row, base_row + row_count,
                                        dtype=np.int64)
        rhs_vec = np.zeros(row_count, dtype=np.float64)
        if isinstance(rhs, (int, float)):
            rhs_vec[:] = float(rhs)
        elif isinstance(rhs, fp.Param):
            on = list(rhs.dims)
            if on:
                j = row_index.join(rhs.frame, on=on, how="left")
                rhs_vec = (j.sort("_rid")["value"]
                            .fill_null(0.0).to_numpy().astype(np.float64))
            else:
                rhs_vec[:] = float(rhs.frame["value"][0])
        if sense == "<=":
            rows_lb_chunks.append(np.full(row_count, -np.inf, dtype=np.float64))
            rows_ub_chunks.append(rhs_vec)
        elif sense == ">=":
            rows_lb_chunks.append(rhs_vec)
            rows_ub_chunks.append(np.full(row_count, np.inf, dtype=np.float64))
        else:
            rows_lb_chunks.append(rhs_vec)
            rows_ub_chunks.append(rhs_vec)
        row_index_lf = row_index.lazy()
        for term in expr.terms:
            if term.dims:
                on = [d for d in term.dims if d in axis_cols]
                plan = (row_index_lf.join(term.lazy, on=on, how="inner")
                                    .select("_rid", "col_id", "coef"))
                pending.append(("dim", base_row, plan))
            else:
                pending.append(("scalar", base_row, row_count,
                                term.lazy.select("col_id", "coef")))

    if pending:
        plans = [p[-1] for p in pending]
        collected = pl.collect_all(plans)
        for p, j in zip(pending, collected):
            kind = p[0]
            if kind == "dim":
                _, base_row, _ = p
                if j.height == 0: continue
                triple_rows.append(base_row + j["_rid"].to_numpy().astype(np.int64))
                triple_cols.append(j["col_id"].to_numpy().astype(np.int64))
                triple_vals.append(j["coef"].to_numpy().astype(np.float64))
            else:
                _, base_row, row_count, _ = p
                cids = j["col_id"].to_numpy().astype(np.int64)
                vals = j["coef"].to_numpy().astype(np.float64)
                rs = np.repeat(np.arange(base_row, base_row + row_count,
                                         dtype=np.int64), len(cids))
                triple_rows.append(rs)
                triple_cols.append(np.tile(cids, row_count))
                triple_vals.append(np.tile(vals, row_count))

    if triple_rows:
        tr = np.concatenate(triple_rows)
        tc = np.concatenate(triple_cols)
        tv = np.concatenate(triple_vals)
        dedup = (pl.DataFrame({"r": tr, "c": tc, "v": tv})
                   .group_by(["r", "c"]).agg(pl.col("v").sum()))
        tr = dedup["r"].to_numpy().astype(np.int64)
        tc = dedup["c"].to_numpy().astype(np.int64)
        tv = dedup["v"].to_numpy().astype(np.float64)
    else:
        tr = np.zeros(0, dtype=np.int64)
        tc = np.zeros(0, dtype=np.int64)
        tv = np.zeros(0, dtype=np.float64)

    n_rows = next_row
    inf = highspy.kHighsInf
    col_lb_h = np.where(col_lb == -np.inf, -inf, col_lb).astype(np.float64)
    col_ub_h = np.where(col_ub ==  np.inf,  inf, col_ub).astype(np.float64)
    rows_lb_arr = np.concatenate(rows_lb_chunks) if rows_lb_chunks else np.zeros(0, dtype=np.float64)
    rows_ub_arr = np.concatenate(rows_ub_chunks) if rows_ub_chunks else np.zeros(0, dtype=np.float64)
    row_lb_h = np.where(rows_lb_arr == -np.inf, -inf, rows_lb_arr).astype(np.float64)
    row_ub_h = np.where(rows_ub_arr ==  np.inf,  inf, rows_ub_arr).astype(np.float64)

    if tr.size:
        order = np.lexsort((tr, tc))
        sorted_r = tr[order].astype(np.int32)
        sorted_c = tc[order].astype(np.int32)
        sorted_v = tv[order].astype(np.float64)
    else:
        sorted_r = np.zeros(0, dtype=np.int32)
        sorted_c = np.zeros(0, dtype=np.int32)
        sorted_v = np.zeros(0, dtype=np.float64)

    starts = np.zeros(n_cols + 1, dtype=np.int32)
    if sorted_c.size:
        np.add.at(starts[1:], sorted_c, 1)
    starts = np.cumsum(starts).astype(np.int32)

    lp = highspy.HighsLp()
    lp.num_col_   = int(n_cols)
    lp.num_row_   = int(n_rows)
    lp.col_cost_  = col_obj.astype(np.float64)
    lp.col_lower_ = col_lb_h
    lp.col_upper_ = col_ub_h
    lp.row_lower_ = row_lb_h
    lp.row_upper_ = row_ub_h
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.num_col_ = int(n_cols)
    lp.a_matrix_.num_row_ = int(n_rows)
    lp.a_matrix_.start_ = starts
    lp.a_matrix_.index_ = sorted_r
    lp.a_matrix_.value_ = sorted_v
    lp.sense_ = highspy.ObjSense.kMinimize

    h = highspy.Highs()
    h.silent()
    h.passModel(lp)

    # v_flow column indices — pull from the Var's frame
    v_flow = problem._vars["v_flow"]
    v_flow_col_idx = v_flow.frame.sort("t")["col_id"].to_numpy().astype(np.int64)

    return h, balance_row_idx, v_flow_col_idx


# ----------------------------------------------------------------------------
# Reference equivalence check

def warm_run_synthetic_via_class(costs: np.ndarray,
                                 demands: np.ndarray) -> dict:
    """Same as ``warm_run_synthetic`` but using the public ``WarmProblem``
    class instead of the hand-rolled POC.  Confirms the API delivers the
    same speedup as the POC."""
    n_rolls = costs.shape[0]
    objs: list[float] = []

    t0 = time.perf_counter()
    p0 = _build_synthetic_problem(costs[0], demands[0])
    wp = fp.WarmProblem(p0)
    sol_0 = wp.solve()
    t_first = time.perf_counter() - t0
    objs.append(sol_0.obj)

    t_idx = np.arange(_N_T, dtype=np.int64)
    t_update_total = 0.0
    t_solve_total = 0.0
    for r in range(1, n_rolls):
        tu0 = time.perf_counter()
        new_demand = fp.Param(("t",),
                              pl.DataFrame({"t": t_idx, "value": demands[r]}))
        new_cost = fp.Param(("t",),
                            pl.DataFrame({"t": t_idx, "value": costs[r]}))
        wp.update_rhs("balance", new_demand)
        wp.update_obj_coef("v_flow", new_cost)
        tu1 = time.perf_counter()
        t_update_total += (tu1 - tu0)
        ts0 = time.perf_counter()
        sol_r = wp.solve()
        t_solve_total += (time.perf_counter() - ts0)
        objs.append(sol_r.obj)

    return {
        "total_seconds": t_first + t_update_total + t_solve_total,
        "first_build_seconds": t_first,
        "update_seconds": t_update_total,
        "solve_seconds": t_solve_total,
        "n_rolls": n_rolls,
        "objs": objs,
    }


def _check_equivalence(cold: dict, warm: dict, tol: float = 1e-9) -> None:
    co = np.asarray(cold["objs"])
    wo = np.asarray(warm["objs"])
    diff = np.abs(co - wo).max()
    if diff > tol * max(1.0, np.abs(co).max()):
        raise AssertionError(f"obj diff cold-vs-warm: {diff:.6e}")


# ----------------------------------------------------------------------------
# Top-level

def main_synthetic(n_rolls: int = 50, repeats: int = 3) -> None:
    print(f"# Synthetic rolling-horizon chain: {n_rolls} rolls of {_N_T}h")
    print(f"# repeats={repeats}, highspy={highspy.Highs().version()}")
    print()

    # Generate once — same data across repeats so cold/warm see identical inputs.
    costs, demands = _make_synthetic_chain(n_rolls)

    cold_totals: list[float] = []
    warm_totals: list[float] = []
    warm_first: list[float] = []
    warm_update: list[float] = []
    warm_solve: list[float] = []
    wp_totals: list[float] = []
    wp_first: list[float] = []
    wp_update: list[float] = []
    wp_solve: list[float] = []
    last_cold = last_warm = last_wp = None
    for _ in range(repeats):
        gc.collect()
        c = cold_run_synthetic(costs, demands)
        cold_totals.append(c["total_seconds"])
        last_cold = c

        gc.collect()
        w = warm_run_synthetic(costs, demands)
        warm_totals.append(w["total_seconds"])
        warm_first.append(w["first_build_seconds"])
        warm_update.append(w["update_seconds"])
        warm_solve.append(w["solve_seconds"])
        last_warm = w

        gc.collect()
        wpr = warm_run_synthetic_via_class(costs, demands)
        wp_totals.append(wpr["total_seconds"])
        wp_first.append(wpr["first_build_seconds"])
        wp_update.append(wpr["update_seconds"])
        wp_solve.append(wpr["solve_seconds"])
        last_wp = wpr
    _check_equivalence(last_cold, last_warm)
    _check_equivalence(last_cold, last_wp)

    cold_med = stats.median(cold_totals)
    warm_med = stats.median(warm_totals)
    wp_med = stats.median(wp_totals)
    print(f"  cold total (median):           {cold_med*1000:9.1f} ms  "
          f"({cold_med/n_rolls*1000:7.2f} ms/roll)")
    print(f"  warm POC total (median):       {warm_med*1000:9.1f} ms")
    print(f"    first build:                 {stats.median(warm_first)*1000:9.1f} ms")
    print(f"    rhs/cost updates:            {stats.median(warm_update)*1000:9.1f} ms total "
          f"({stats.median(warm_update)/(n_rolls-1)*1000:7.3f} ms/roll)")
    print(f"    h.run() time:                {stats.median(warm_solve)*1000:9.1f} ms total "
          f"({stats.median(warm_solve)/n_rolls*1000:7.3f} ms/roll)")
    print(f"  warm WarmProblem (median):     {wp_med*1000:9.1f} ms")
    print(f"    first build:                 {stats.median(wp_first)*1000:9.1f} ms")
    print(f"    rhs/cost updates:            {stats.median(wp_update)*1000:9.1f} ms total "
          f"({stats.median(wp_update)/(n_rolls-1)*1000:7.3f} ms/roll)")
    print(f"    h.run() time:                {stats.median(wp_solve)*1000:9.1f} ms total "
          f"({stats.median(wp_solve)/n_rolls*1000:7.3f} ms/roll)")
    print(f"  speedup (POC):                 {cold_med/warm_med:6.2f}x")
    print(f"  speedup (WarmProblem):         {cold_med/wp_med:6.2f}x")
    print(f"  obj equivalence:               OK")


def main_synthetic_large(n_rolls: int = 50, n_t: int = 168,
                         repeats: int = 3) -> None:
    """A larger LP per roll (168h = 1 week) so the build/solve overhead is more
    representative of real flextool LPs.  Toggle via ``_N_T`` patch."""
    global _N_T
    saved = _N_T
    _N_T = n_t
    try:
        print(f"# Synthetic rolling-horizon chain (LARGER): "
              f"{n_rolls} rolls of {n_t}h")
        main_synthetic(n_rolls=n_rolls, repeats=repeats)
    finally:
        _N_T = saved


if __name__ == "__main__":
    n_rolls = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    repeats = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    main_synthetic(n_rolls=n_rolls, repeats=repeats)
    print()
    main_synthetic_large(n_rolls=20, n_t=168, repeats=repeats)
    print()
    main_synthetic_large(n_rolls=10, n_t=720, repeats=repeats)
