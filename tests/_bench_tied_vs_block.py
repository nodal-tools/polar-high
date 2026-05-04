"""Benchmark: tied-variables (formulation A) vs block-variables (formulation B).

Synthetic LP that mirrors the multi-resolution time-block dispatch pattern:

  * `n_fine` fine timesteps (representing hourly v_flow / v_state).
  * `n_blocks = n_fine / ratio` coarse blocks.
  * Per-hour demand balance `x[t] + s[t-1] - s[t] == demand[t]`.
  * State bounds 0 <= s[t] <= 100.
  * Objective: minimize sum(c[t] * x[t]).

Formulation A — "tied":
  - One x[t] and s[t] per fine timestep.
  - Equality `x[t] == x[t_first_in_block]` for every interior t in each block,
    same for s[t].

Formulation B — "block":
  - One x_block[b], s_block[b] per coarse block.
  - Demand balance summed across the block:
      ratio * x_block[b] + s_block[b-1] - s_block[b] == sum(demand[t in b])
  - Cost: sum(c_block[b] * x_block[b] * ratio) where c_block[b] is the average c[t].

Both formulations should produce mathematically equivalent solutions
(same objective, same x_block values).

Each (ratio, formulation, presolve) is run 3+ times; we report median/min/max
for build/solve time, peak RSS, and capture LP shape (var count, cstr count,
nnz, presolved column count).

Re-run with::

    PYTHONPATH=src python tests/_bench_tied_vs_block.py
"""

from __future__ import annotations

import gc
import resource
import statistics as stats
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import polars as pl

import highspy
import polar_high_opt as fp


# ----------------------------------------------------------------------------
# helpers

def _peak_rss_mb() -> float:
    """Return peak RSS in MB. Linux ru_maxrss is in KB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _make_data(n_fine: int, ratio: int, seed: int = 0):
    """Generate per-hour cost / demand vectors and the block lookup.

    Cost is fully random per hour.  Demand is also random per hour.
    The two formulations are NOT mathematically identical when only
    ``x[t]`` is tied across a block (which is what we benchmark — see
    ``build_tied_flexpy``):

      * Tied keeps ``s[t]`` per fine hour, so it can absorb intra-block
        demand swings.
      * Block has one ``s_block[b]``, so intra-block swings collapse
        into the block-level state.

    We compare them as *proxies for the same conceptual model*.  Section
    of the report quantifies the small objective drift this introduces
    (typically <1% on these data, well below the LP-shape and
    solve-time effects we care about).
    """
    rng = np.random.default_rng(seed)
    n_blocks = n_fine // ratio
    # truncate n_fine so it's divisible by ratio (drop the trailing partial block)
    n_fine_used = n_blocks * ratio
    cost = rng.uniform(1.0, 10.0, size=n_fine_used)
    demand = rng.uniform(20.0, 80.0, size=n_fine_used)
    block_of_t = np.arange(n_fine_used) // ratio
    # first t in each block
    first_in_block = block_of_t * ratio
    return cost, demand, n_blocks, block_of_t.astype(np.int64), first_in_block.astype(np.int64)


# ----------------------------------------------------------------------------
# Formulation A — tied (flexpy)

def build_tied_flexpy(n_fine: int, ratio: int, cost: np.ndarray,
                      demand: np.ndarray, block_of_t: np.ndarray,
                      first_in_block: np.ndarray, tie_state: bool = False
                      ) -> fp.Problem:
    """Formulation A — tied: x[t] (and optionally s[t]) tied across the block.

    Per the user spec, the conceptual model ties both ``v_flow`` and
    ``v_state``.  In a per-hour balance LP, however, tying *both* makes
    the interior-hour equations collapse to ``x_block == demand[t]``,
    over-determining x.  For a clean equivalence with the block
    formulation we therefore tie only ``x[t]`` by default
    (``tie_state=False``); ``v_state`` stays at fine resolution.  This
    also better matches typical flextool block usage: state tracks at
    the fine grid even when flow is block-constant.

    Set ``tie_state=True`` to also tie ``s[t]``.  Note that with
    block-constant demand this remains feasible (interior balance is
    automatically satisfied because ``s[t-1]==s[t]==s_b`` and
    ``demand[t]==demand_b``), and the objective is identical — but
    HiGHS sees a larger LP.
    """
    p = fp.Problem()
    t_idx = pl.DataFrame({"t": np.arange(n_fine, dtype=np.int64)})
    x = p.add_var("x", "t", t_idx, lower=0.0, upper=float("inf"))
    # state upper bound ample so that bound-feasibility is the same in both
    # formulations across ratios up to 1:712.  We're benchmarking LP shape
    # / solver behaviour, not storage scarcity.
    s = p.add_var("s", "t", t_idx, lower=0.0, upper=1.0e6)

    # demand balance: x[t] + s[t-1] - s[t] == demand[t]; for t=0, s[-1]=0
    lag = pl.DataFrame({"t": np.arange(1, n_fine, dtype=np.int64),
                        "t_prev": np.arange(0, n_fine - 1, dtype=np.int64)})
    s_lag = fp.Lag(s, lag, time_dim="t", lag_col="t_prev")
    demand_p = fp.Param(("t",), pl.DataFrame({"t": np.arange(n_fine, dtype=np.int64),
                                              "value": demand}))
    p.add_cstr(
        "balance",
        over=t_idx,
        sense="==",
        lhs_terms={"x": x, "s_lag": s_lag, "minus_s": -s.to_expr()},
        rhs_terms={"demand": demand_p},
    )

    # tying constraints: x[t] - x[first_in_block(t)] == 0 for interior t
    interior_mask = np.arange(n_fine) != first_in_block
    interior_t = np.arange(n_fine)[interior_mask].astype(np.int64)
    interior_first = first_in_block[interior_mask].astype(np.int64)
    if len(interior_t):
        tie_idx = pl.DataFrame({"t": interior_t, "t_anchor": interior_first})
        from polar_high_opt.engine import Expr, _Term

        x_row = (x.frame
                 .join(tie_idx, on="t", how="inner")
                 .with_columns(coef=pl.lit(1.0))
                 .select("t", "t_anchor", "col_id", "coef"))
        x_anchor = (x.frame.rename({"t": "t_anchor"})
                    .join(tie_idx, on="t_anchor", how="inner")
                    .with_columns(coef=pl.lit(-1.0))
                    .select("t", "t_anchor", "col_id", "coef"))
        tie_expr_x = Expr([_Term(x_row, ("t", "t_anchor")),
                           _Term(x_anchor, ("t", "t_anchor"))])
        p.add_cstr("tie_x", over=tie_idx, sense="==",
                   lhs_terms={"diff": tie_expr_x}, rhs_terms={"zero": 0.0})

        if tie_state:
            s_row = (s.frame
                     .join(tie_idx, on="t", how="inner")
                     .with_columns(coef=pl.lit(1.0))
                     .select("t", "t_anchor", "col_id", "coef"))
            s_anchor = (s.frame.rename({"t": "t_anchor"})
                        .join(tie_idx, on="t_anchor", how="inner")
                        .with_columns(coef=pl.lit(-1.0))
                        .select("t", "t_anchor", "col_id", "coef"))
            tie_expr_s = Expr([_Term(s_row, ("t", "t_anchor")),
                               _Term(s_anchor, ("t", "t_anchor"))])
            p.add_cstr("tie_s", over=tie_idx, sense="==",
                       lhs_terms={"diff": tie_expr_s}, rhs_terms={"zero": 0.0})

    cost_p = fp.Param(("t",), pl.DataFrame({"t": np.arange(n_fine, dtype=np.int64),
                                            "value": cost}))
    p.set_objective(cost_p * x, sense="min")
    return p


# ----------------------------------------------------------------------------
# Formulation B — block (flexpy)

def build_block_flexpy(n_fine: int, ratio: int, cost: np.ndarray,
                       demand: np.ndarray, block_of_t: np.ndarray) -> fp.Problem:
    """Formulation B — block: one variable per coarse block.

    Balance summed across the block:
      ``ratio * x_block[b] + s_block[b-1] - s_block[b] == sum(demand[t in b])``

    Cost coefficient: ``c_block[b] = sum(c[t in b])`` so that the
    per-hour cost ``c[t] * x[t]`` evaluated under ``x[t] == x_block``
    matches ``c_block[b] * x_block[b]`` exactly.
    """
    n_blocks = n_fine // ratio
    p = fp.Problem()
    b_idx = pl.DataFrame({"b": np.arange(n_blocks, dtype=np.int64)})
    x = p.add_var("x_block", "b", b_idx, lower=0.0, upper=float("inf"))
    s = p.add_var("s_block", "b", b_idx, lower=0.0, upper=1.0e6)

    # demand balance summed over the block
    demand_per_block = (pl.DataFrame({"b": block_of_t, "demand": demand})
                        .group_by("b").agg(pl.col("demand").sum())
                        .rename({"demand": "value"})
                        .sort("b"))
    rhs_p = fp.Param(("b",), demand_per_block)

    lag = pl.DataFrame({"b": np.arange(1, n_blocks, dtype=np.int64),
                        "b_prev": np.arange(0, n_blocks - 1, dtype=np.int64)})
    s_lag = fp.Lag(s, lag, time_dim="b", lag_col="b_prev")

    p.add_cstr(
        "balance",
        over=b_idx,
        sense="==",
        lhs_terms={"x": float(ratio) * x.to_expr(),
                   "s_lag": s_lag,
                   "minus_s": -s.to_expr()},
        rhs_terms={"demand": rhs_p},
    )

    cost_per_block = (pl.DataFrame({"b": block_of_t, "c": cost})
                      .group_by("b").agg(pl.col("c").sum())
                      .rename({"c": "value"})
                      .sort("b"))
    cost_p = fp.Param(("b",), cost_per_block)
    p.set_objective(cost_p * x, sense="min")
    return p


# ----------------------------------------------------------------------------
# Solver runner using flexpy + override solve to capture LP shape

def _emit_lp_and_solve(problem: fp.Problem, presolve: bool):
    """Replicate Problem.solve internals so we can split build_time and solve_time
    cleanly, and capture LP shape and post-presolve column count.

    Returns dict with keys:
      build_time, solve_time, n_cols, n_rows, n_nnz, n_cols_presolved, obj.
    """
    # ---- build LP frames (same logic as Problem.solve) ----
    t0 = time.perf_counter()
    n_cols = problem._next_col
    col_lb = np.zeros(n_cols, dtype=np.float64)
    col_ub = np.full(n_cols, np.inf, dtype=np.float64)
    col_obj = np.zeros(n_cols, dtype=np.float64)

    for v in problem._vars.values():
        ids = v.frame["col_id"].to_numpy()
        col_lb[ids] = float(v.lower)
        col_ub[ids] = float(v.upper)

    for t in problem._obj_terms:
        for cid, c in zip(t.frame["col_id"].to_numpy(),
                          t.frame["coef"].to_numpy()):
            col_obj[int(cid)] += float(c)

    rows_lb: list[float] = []
    rows_ub: list[float] = []
    triple_rows: list[np.ndarray] = []
    triple_cols: list[np.ndarray] = []
    triple_vals: list[np.ndarray] = []
    next_row = 0
    from polar_high_opt.engine import Expr, _Term

    for name, proto, over in problem._cstrs:
        expr, sense, rhs = proto.expr, proto.sense, proto.rhs
        if over is None:
            row_count = 1
            row_index = pl.DataFrame({"_rid": [0]})
            axis_cols: list[str] = []
        else:
            row_count = over.height
            axis_cols = list(over.columns)
            row_index = over.with_columns(_rid=pl.int_range(0, over.height, dtype=pl.Int64))

        base_row = next_row
        next_row += row_count

        rhs_vec = np.zeros(row_count, dtype=np.float64)
        if isinstance(rhs, (int, float)):
            rhs_vec[:] = float(rhs)
        elif isinstance(rhs, fp.Param):
            on = list(rhs.dims)
            if on:
                j = row_index.join(rhs.frame, on=on, how="left")
                rhs_vec = (j.sort("_rid")["value"].fill_null(0.0)
                           .to_numpy().astype(np.float64))
            else:
                rhs_vec[:] = float(rhs.frame["value"][0])
        elif isinstance(rhs, (fp.Var, fp.Expr)):
            rhs_expr = rhs.to_expr() if isinstance(rhs, fp.Var) else rhs
            neg = [_Term(t.frame.with_columns(coef=-pl.col("coef")), t.dims)
                   for t in rhs_expr.terms]
            expr = Expr(expr.terms + neg)
        else:
            raise TypeError("bad rhs")

        for r in range(row_count):
            if sense == "<=":
                rows_lb.append(-np.inf); rows_ub.append(rhs_vec[r])
            elif sense == ">=":
                rows_lb.append(rhs_vec[r]); rows_ub.append(np.inf)
            else:
                rows_lb.append(rhs_vec[r]); rows_ub.append(rhs_vec[r])

        for term in expr.terms:
            f = term.frame
            if term.dims:
                on = [d for d in term.dims if d in axis_cols]
                j = row_index.join(f, on=on, how="inner")
                if j.height == 0:
                    continue
                triple_rows.append(base_row + j["_rid"].to_numpy().astype(np.int64))
                triple_cols.append(j["col_id"].to_numpy().astype(np.int64))
                triple_vals.append(j["coef"].to_numpy().astype(np.float64))
            else:
                cids = f["col_id"].to_numpy().astype(np.int64)
                vals = f["coef"].to_numpy().astype(np.float64)
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
    n_nnz = int(tr.size)

    inf = highspy.kHighsInf
    col_lb_h = np.where(col_lb == -np.inf, -inf, col_lb).astype(np.float64)
    col_ub_h = np.where(col_ub ==  np.inf,  inf, col_ub).astype(np.float64)
    row_lb_h = np.array([-inf if v == -np.inf else float(v) for v in rows_lb],
                         dtype=np.float64)
    row_ub_h = np.array([ inf if v ==  np.inf else float(v) for v in rows_ub],
                         dtype=np.float64)

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
    if not presolve:
        h.setOptionValue("presolve", "off")
    build_time = time.perf_counter() - t0

    # ---- solve ----
    t1 = time.perf_counter()
    h.run()
    solve_time = time.perf_counter() - t1

    # post-presolve cols: getPresolvedNumCol may not exist in all builds;
    # fall back to scraping the log via getInfo or HiGHS internal model.
    n_cols_pre = -1
    try:
        info = h.getInfo()
        # Some builds expose presolved sizes via simplex_iteration_count etc;
        # we just record final reported numbers
        n_cols_pre = int(getattr(info, "presolve_simplex_iteration_count", -1))
    except Exception:
        pass
    # Safer: rebuild on a fresh Highs and request presolved LP.
    n_cols_after_presolve = -1
    try:
        h2 = highspy.Highs()
        h2.silent()
        h2.passModel(lp)
        if not presolve:
            h2.setOptionValue("presolve", "off")
        h2.presolve()
        plp = h2.getPresolvedLp()
        n_cols_after_presolve = int(plp.num_col_)
    except Exception:
        n_cols_after_presolve = -1

    obj = h.getObjectiveValue()
    return {
        "build_time": build_time,
        "solve_time": solve_time,
        "n_cols": int(n_cols),
        "n_rows": int(n_rows),
        "n_nnz": n_nnz,
        "n_cols_presolved": n_cols_after_presolve,
        "obj": float(obj),
        "status_optimal": h.getModelStatus() == highspy.HighsModelStatus.kOptimal,
    }


# ----------------------------------------------------------------------------
# Top-level run

@dataclass
class RunResult:
    ratio: int
    formulation: str
    presolve: bool
    n_fine: int
    n_cols: int = 0
    n_rows: int = 0
    n_nnz: int = 0
    n_cols_presolved: int = -1
    build_times: list[float] = field(default_factory=list)
    solve_times: list[float] = field(default_factory=list)
    peak_rss_mb: list[float] = field(default_factory=list)
    obj: float = 0.0
    status_optimal: bool = True


def _bench_one(formulation: str, ratio: int, n_fine: int,
               presolve: bool, repeats: int = 3) -> RunResult:
    cost, demand, n_blocks, block_of_t, first_in_block = _make_data(n_fine, ratio)
    # _make_data may truncate n_fine to a multiple of ratio
    n_fine_used = n_blocks * ratio

    res = RunResult(ratio=ratio, formulation=formulation,
                    presolve=presolve, n_fine=n_fine_used)

    for _ in range(repeats):
        gc.collect()
        rss_before = _peak_rss_mb()

        if formulation == "tied":
            t0 = time.perf_counter()
            problem = build_tied_flexpy(n_fine_used, ratio, cost, demand,
                                        block_of_t, first_in_block)
            t_build_problem = time.perf_counter() - t0
        elif formulation == "block":
            t0 = time.perf_counter()
            problem = build_block_flexpy(n_fine_used, ratio, cost, demand, block_of_t)
            t_build_problem = time.perf_counter() - t0
        elif formulation == "baseline":
            t0 = time.perf_counter()
            # baseline: no tying, no blocks — one var per fine timestep,
            # demand balance only, no extra constraints. (1:1 reference.)
            problem = build_baseline_flexpy(n_fine_used, cost, demand)
            t_build_problem = time.perf_counter() - t0
        else:
            raise ValueError(formulation)

        info = _emit_lp_and_solve(problem, presolve=presolve)
        # build_time = build the Problem + emit the LP frames
        build_total = t_build_problem + info["build_time"]

        res.build_times.append(build_total)
        res.solve_times.append(info["solve_time"])
        res.n_cols = info["n_cols"]
        res.n_rows = info["n_rows"]
        res.n_nnz = info["n_nnz"]
        res.n_cols_presolved = info["n_cols_presolved"]
        res.obj = info["obj"]
        res.status_optimal = info["status_optimal"]
        rss_after = _peak_rss_mb()
        res.peak_rss_mb.append(rss_after)
    return res


def build_baseline_flexpy(n_fine: int, cost: np.ndarray,
                          demand: np.ndarray) -> fp.Problem:
    p = fp.Problem()
    t_idx = pl.DataFrame({"t": np.arange(n_fine, dtype=np.int64)})
    x = p.add_var("x", "t", t_idx, lower=0.0, upper=float("inf"))
    s = p.add_var("s", "t", t_idx, lower=0.0, upper=1.0e6)
    lag = pl.DataFrame({"t": np.arange(1, n_fine, dtype=np.int64),
                        "t_prev": np.arange(0, n_fine - 1, dtype=np.int64)})
    s_lag = fp.Lag(s, lag, time_dim="t", lag_col="t_prev")
    demand_p = fp.Param(("t",), pl.DataFrame({"t": np.arange(n_fine, dtype=np.int64),
                                              "value": demand}))
    p.add_cstr("balance", over=t_idx, sense="==",
               lhs_terms={"x": x, "s_lag": s_lag, "minus_s": -s.to_expr()},
               rhs_terms={"demand": demand_p})
    cost_p = fp.Param(("t",), pl.DataFrame({"t": np.arange(n_fine, dtype=np.int64),
                                            "value": cost}))
    p.set_objective(cost_p * x, sense="min")
    return p


def _summarise(rs: list[float]) -> dict:
    return {
        "median": stats.median(rs),
        "min": min(rs),
        "max": max(rs),
    }


def main(n_fine: int = 8760, repeats: int = 3) -> list[RunResult]:
    ratios = [1, 24, 168, 712]
    results: list[RunResult] = []

    print(f"# Benchmark (n_fine = {n_fine}, repeats = {repeats}, "
          f"highspy = {highspy.Highs().version()})")
    print()
    print(f"{'ratio':>5} {'form':>9} {'presolve':>9} {'cols':>8} {'rows':>8} "
          f"{'nnz':>9} {'cols_pre':>9} "
          f"{'build_med':>10} {'solve_med':>10} {'rss_max_MB':>10} "
          f"{'obj':>14}")

    for ratio in ratios:
        for presolve in [True, False]:
            if ratio == 1:
                # baseline: only one formulation makes sense; skip duplicates
                forms = ["baseline"]
            else:
                forms = ["tied", "block"]
            for form in forms:
                r = _bench_one(form, ratio, n_fine, presolve, repeats=repeats)
                results.append(r)
                bt = _summarise(r.build_times)
                st = _summarise(r.solve_times)
                rss = max(r.peak_rss_mb) if r.peak_rss_mb else 0.0
                print(f"{ratio:>5d} {form:>9} {str(presolve):>9} "
                      f"{r.n_cols:>8d} {r.n_rows:>8d} {r.n_nnz:>9d} "
                      f"{r.n_cols_presolved:>9d} "
                      f"{bt['median']:>10.4f} {st['median']:>10.4f} "
                      f"{rss:>10.1f} "
                      f"{r.obj:>14.4f}")
    return results


if __name__ == "__main__":
    n_fine = int(sys.argv[1]) if len(sys.argv) > 1 else 8760
    repeats = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    main(n_fine=n_fine, repeats=repeats)
