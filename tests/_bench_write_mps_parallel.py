"""Profile Problem.write_mps memory peak: default vs single-thread polars.

Validates the "polars parallel sort multiplies peak by ~core-count" hypothesis
on a synthetic LP. Writes to /tmp, deletes after each run.

Run modes (controlled by argv[1]):
    baseline   — default polars, instrumented per-phase
    single     — pl.Config.set_threadpool_size(1) set BEFORE Problem.write_mps
    envthread  — relies on POLARS_MAX_THREADS=1 env-var set by parent
    chunked    — Fix #2 sketch: chunked column-id sort, default threadpool
    chunked1   — Fix #2 sketch + single-thread polars (combined)

Each subprocess prints a JSON line with peak_rss_bytes and per-phase deltas.
"""

from __future__ import annotations

import gc
import json
import os
import resource
import sys
import time
from pathlib import Path

# Set thread-pool BEFORE import polars if requested via env-var.
# (We test whether runtime set_threadpool_size works in mode "single".)
mode = sys.argv[1] if len(sys.argv) > 1 else "baseline"

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

# Now make polar_high importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from polar_high import Problem  # noqa: E402


def rss_gb() -> float:
    """Max-RSS in GB (monotonic high-water; Linux ru_maxrss is KiB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def cur_rss_gb() -> float:
    """Current RSS in GB by reading /proc/self/statm (RSS pages * page size)."""
    try:
        with open("/proc/self/statm") as f:
            rss_pages = int(f.read().split()[1])
        return rss_pages * (resource.getpagesize() / (1024**3))
    except Exception:
        return rss_gb()


def now() -> float:
    return time.monotonic()


def build_problem_multi(
    n_vars: int, n_cstrs_dim: int, nnz_per_cstr: int, n_families: int
) -> Problem:
    """Multi-family variant: split constraints into N families. Mimics realistic
    LPs where many small constraint families each contribute triples."""
    from polar_high.engine import Param
    from polar_high.engine import Sum as Sum2

    pb = Problem()
    idx_i = pl.DataFrame({"i": pl.int_range(0, n_vars, dtype=pl.Int64, eager=True)})
    x = pb.add_var("x", dims=("i",), index=idx_i, lower=0.0, upper=10.0)
    rng = np.random.default_rng(0)
    rows_per_fam = n_cstrs_dim // n_families
    for fi in range(n_families):
        idx_r = pl.DataFrame({"r": pl.int_range(0, rows_per_fam, dtype=pl.Int64, eager=True)})
        r_arr = np.repeat(np.arange(rows_per_fam, dtype=np.int64), nnz_per_cstr)
        k_arr = np.tile(np.arange(nnz_per_cstr, dtype=np.int64), rows_per_fam)
        i_arr = (r_arr * 7 + k_arr + fi) % n_vars
        v_arr = rng.uniform(0.5, 2.0, size=r_arr.size).astype(np.float64)
        coef_df = pl.DataFrame({"r": r_arr, "i": i_arr, "value": v_arr})
        p = Param(("r", "i"), coef_df)
        expr = Sum2(p * x, over=("i",))
        pb.add_cstr(
            f"c{fi}", over=idx_r, sense="<=", lhs_terms={"lhs": expr}, rhs_terms={"rhs": 100.0}
        )
    pb.set_objective(Sum2(x, over=("i",)), sense="min")
    return pb


def build_problem(n_vars: int, n_cstrs_dim: int, nnz_per_cstr: int) -> Problem:
    """Synthetic LP:
    - One Var family "x" with n_vars columns indexed by i in [0, n_vars).
    - One constraint family "c" of n_cstrs_dim rows indexed by r.
      Each row touches `nnz_per_cstr` distinct columns. We build the
      coefficient frame as a (r, i, coef) Param-like polars frame fed
      via a Sum expr.

    Total rows = n_cstrs_dim; total cols = n_vars; total nnz = n_cstrs_dim*nnz_per_cstr.
    """
    pb = Problem()
    idx_i = pl.DataFrame({"i": pl.int_range(0, n_vars, dtype=pl.Int64, eager=True)})
    idx_r = pl.DataFrame({"r": pl.int_range(0, n_cstrs_dim, dtype=pl.Int64, eager=True)})

    x = pb.add_var("x", dims=("i",), index=idx_i, lower=0.0, upper=10.0)

    # Build coefficient frame:  for each row r, pick nnz_per_cstr columns.
    # Use a deterministic pattern: column = (r*7 + k) mod n_vars.
    rng = np.random.default_rng(0)
    r_arr = np.repeat(np.arange(n_cstrs_dim, dtype=np.int64), nnz_per_cstr)
    # randomly pick columns; we want distinct (r,i) so use a permutation per row
    # but cheaper: i = (r*7 + k) mod n_vars
    k_arr = np.tile(np.arange(nnz_per_cstr, dtype=np.int64), n_cstrs_dim)
    i_arr = (r_arr * 7 + k_arr) % n_vars
    v_arr = rng.uniform(0.5, 2.0, size=r_arr.size).astype(np.float64)
    coef_df = pl.DataFrame({"r": r_arr, "i": i_arr, "value": v_arr})

    # Express the constraint as Sum_i (coef[r,i] * x[i]) <= 100.0 over r.
    # The supported pattern is to multiply Var by a Param.
    from polar_high.engine import Param, Sum

    p = Param(("r", "i"), coef_df)
    expr = Sum(p * x, over=("i",))
    pb.add_cstr("c", over=idx_r, sense="<=", lhs_terms={"lhs": expr}, rhs_terms={"rhs": 100.0})

    # Objective: minimize sum of x.
    from polar_high.engine import Sum as Sum2

    pb.set_objective(Sum2(x, over=("i",)), sense="min")
    return pb


def instrumented_write_mps(pb: Problem, path: str) -> dict:
    """Mirror Problem.write_mps phases inline with RSS snapshots.

    Only re-implements the heavy phases — not full MPS streaming — but
    is faithful enough that the peak occurs at the same point.
    """

    from polar_high.engine import _align_enum_join_keys

    phases: dict[str, float] = {}
    gc.collect()
    phases["start_rss_gb"] = rss_gb()

    n_cols = int(pb._next_col)

    row_names = ["cost"]
    families = []
    triple_frames = []
    next_row = 1

    t0 = now()
    for cname, proto, over in pb._cstrs:
        expr, sense, rhs = proto.expr, proto.sense, proto.rhs
        if over is None:
            row_count = 1
            row_index = pl.DataFrame({"_rid": [0]})
            axis_cols = []
        else:
            row_count = int(over.height)
            axis_cols = list(over.columns)
            row_index = over.with_columns(_rid=pl.int_range(0, over.height, dtype=pl.Int64))
        base_row = next_row
        next_row += row_count
        rhs_vec = np.zeros(row_count, dtype=np.float64)
        if isinstance(rhs, (int, float)):
            rhs_vec[:] = float(rhs)
        sense_char = {"<=": "L", ">=": "G", "==": "E"}[sense]
        families.append((base_row, row_count, sense_char, rhs_vec))

        # build family triples
        row_index_lf = row_index.lazy()
        term_plans = []
        for term in expr.terms:
            if term.dims:
                on = [d for d in term.dims if d in axis_cols]
                rl_a, tl_a = _align_enum_join_keys(row_index_lf, term.lazy, on)
                plan = rl_a.join(tl_a, on=on, how="inner").select("_rid", "col_id", "coef")
                term_plans.append(("dim", plan))
            else:
                term_plans.append(("scalar", term.lazy.select("col_id", "coef")))

        fam_rows_list, fam_cols_list, fam_vals_list = [], [], []
        if term_plans:
            if row_count > 50_000:
                collected = [p.collect() for _, p in term_plans]
            else:
                collected = pl.collect_all([p for _, p in term_plans])
            for (kind, _), j in zip(term_plans, collected):
                if kind == "dim":
                    if j.height == 0:
                        continue
                    fam_rows_list.append(j["_rid"].to_numpy().astype(np.int64))
                    fam_cols_list.append(j["col_id"].to_numpy().astype(np.int64))
                    fam_vals_list.append(j["coef"].to_numpy().astype(np.float64))
                else:
                    cids = j["col_id"].to_numpy().astype(np.int64)
                    vals = j["coef"].to_numpy().astype(np.float64)
                    if cids.size == 0:
                        continue
                    fam_rows_list.append(np.repeat(np.arange(row_count, dtype=np.int64), cids.size))
                    fam_cols_list.append(np.tile(cids, row_count))
                    fam_vals_list.append(np.tile(vals, row_count))
            del collected
        if fam_rows_list:
            fr = np.concatenate(fam_rows_list)
            fc = np.concatenate(fam_cols_list)
            fv = np.concatenate(fam_vals_list)
            dedup = (
                pl.DataFrame({"r": fr, "c": fc, "v": fv})
                .group_by(["r", "c"])
                .agg(pl.col("v").sum())
            )
            triple_frames.append(
                dedup.select(
                    col_id=pl.col("c").cast(pl.Int64),
                    row_id=(pl.col("r").cast(pl.Int64) + base_row),
                    coef=pl.col("v").cast(pl.Float64),
                )
            )
            del dedup, fr, fc, fv

    phases["after_term_collection_gb"] = rss_gb()
    phases["term_collection_s"] = now() - t0

    # Objective triples
    t0 = now()
    if pb._obj_terms:
        obj_cids, obj_vals = [], []
        for t in pb._obj_terms:
            f = t.lazy.collect()
            if f.height == 0:
                continue
            obj_cids.append(f["col_id"].to_numpy().astype(np.int64))
            obj_vals.append(f["coef"].to_numpy().astype(np.float64))
        if obj_cids:
            oc = np.concatenate(obj_cids)
            ov = np.concatenate(obj_vals)
            obj_df = (
                pl.DataFrame({"c": oc, "v": ov})
                .group_by("c")
                .agg(pl.col("v").sum())
                .select(
                    col_id=pl.col("c").cast(pl.Int64),
                    row_id=pl.lit(0, dtype=pl.Int64),
                    coef=pl.col("v").cast(pl.Float64),
                )
            )
            triple_frames.append(obj_df)

    if triple_frames:
        all_triples = pl.concat(triple_frames, how="vertical_relaxed")
    else:
        all_triples = pl.DataFrame(
            schema={"col_id": pl.Int64, "row_id": pl.Int64, "coef": pl.Float64}
        )
    del triple_frames

    phases["after_triple_concat_gb"] = rss_gb()
    phases["triple_concat_s"] = now() - t0
    phases["n_triples"] = int(all_triples.height)

    # Branch: baseline / single sort vs chunked
    sort_mode = phases.setdefault("sort_mode", os.environ.get("SORT_MODE", "global"))

    t0 = now()
    if sort_mode == "global":
        try:
            sorted_triples = (
                all_triples.lazy().sort(["col_id", "row_id"]).collect(engine="streaming")
            )
        except TypeError:
            sorted_triples = all_triples.lazy().sort(["col_id", "row_id"]).collect(streaming=True)
        del all_triples
        phases["after_sort_gb"] = rss_gb()
        # walk sorted to mimic emit (so the buffer stays alive through emit)
        h = int(sorted_triples.height)
        phases["sorted_h"] = h
    elif sort_mode == "chunked":
        chunk_size = int(os.environ.get("CHUNK_SIZE", "100000"))
        # Partition by col_id // chunk_size, sort each, write to /dev/null-like sink.
        n_cols_local = int(pb._next_col)
        n_chunks = (n_cols_local + chunk_size - 1) // chunk_size
        # Add partition key, then for each partition filter+sort.
        all_triples = all_triples.with_columns((pl.col("col_id") // chunk_size).alias("__part"))
        max_h = 0
        for p_idx in range(n_chunks):
            part = all_triples.filter(pl.col("__part") == p_idx).drop("__part")
            if part.height == 0:
                continue
            part_sorted = part.sort(["col_id", "row_id"])
            max_h = max(max_h, part_sorted.height)
            del part, part_sorted
        del all_triples
        phases["after_sort_gb"] = rss_gb()
        phases["max_chunk_h"] = max_h
    phases["sort_s"] = now() - t0

    phases["peak_rss_gb"] = rss_gb()
    return phases


def main():
    n_vars = int(os.environ.get("N_VARS", "500000"))
    n_cstrs = int(os.environ.get("N_CSTRS", "500000"))
    nnz_per = int(os.environ.get("NNZ_PER", "10"))
    out = {"mode": mode, "n_vars": n_vars, "n_cstrs": n_cstrs, "nnz_per_cstr": nnz_per}

    # Apply runtime threadpool change before any heavy polars op.
    if mode in ("single", "chunked1"):
        try:
            pl.Config.set_threadpool_size(1)
        except Exception as e:  # noqa: BLE001
            out["set_threadpool_size_error"] = repr(e)
    if mode == "chunked" or mode == "chunked1":
        os.environ["SORT_MODE"] = "chunked"
    else:
        os.environ["SORT_MODE"] = "global"

    out["polars_threads_after_config"] = pl.thread_pool_size()
    out["polars_max_threads_env"] = os.environ.get("POLARS_MAX_THREADS")

    n_families = int(os.environ.get("N_FAMILIES", "1"))
    if n_families > 1:
        pb = build_problem_multi(n_vars, n_cstrs, nnz_per, n_families)
    else:
        pb = build_problem(n_vars, n_cstrs, nnz_per)
    out["build_rss_gb"] = rss_gb()
    out["n_cols"] = int(pb._next_col)

    phases = instrumented_write_mps(pb, "/tmp/_bench.mps")
    out["phases"] = phases
    print(json.dumps(out))


if __name__ == "__main__":
    main()
