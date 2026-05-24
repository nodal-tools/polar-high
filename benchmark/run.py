"""Sweep the benchmark across (N, threads, tool) and write results CSV.

Each cell is run in a fresh subprocess so caches and memory don't carry
over. The orchestrator caps every subprocess to a chosen thread count
via the standard env-var bundle (POLARS_MAX_THREADS, OMP_NUM_THREADS,
OPENBLAS_NUM_THREADS, MKL_NUM_THREADS) so every tool runs in the same
N-core envelope no matter which library it imports.

Usage:
    python benchmark/run.py
    python benchmark/run.py --sizes 10 30 100
    python benchmark/run.py --tools polar linopy --repeats 5
    python benchmark/run.py --threads 1 4 16 32 --sizes 1000 --tools polar linopy
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent

DEFAULT_TOOLS = ["polar", "linopy", "pyomo"]
DEFAULT_SIZES = [10, 30, 100, 300, 1000]
DEFAULT_REPEATS = 3
DEFAULT_TIMEOUT_S = 600

THREAD_ENV_VARS = (
    "POLARS_MAX_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
)


def run_cell(
    tool: str,
    N: int,
    threads: int,
    timeout: int,
    time_limit: float | None = None,
) -> dict | None:
    cmd = [sys.executable, str(HERE / "run_one.py"), tool, str(N)]
    env = {**os.environ, **{var: str(threads) for var in THREAD_ENV_VARS}}
    if time_limit is not None:
        env["BENCH_TIME_LIMIT"] = str(time_limit)
    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
            env=env,
        )
    except subprocess.TimeoutExpired:
        print(f"[timeout] {tool} N={N} threads={threads} after {timeout}s", file=sys.stderr)
        return None
    except subprocess.CalledProcessError as e:
        msg = (e.stderr or "").strip().splitlines()
        tail = msg[-1] if msg else "(no stderr)"
        print(f"[fail]    {tool} N={N} threads={threads}: {tail[:200]}", file=sys.stderr)
        return None

    line = r.stdout.strip().splitlines()[-1]
    parts = line.split(",")
    # New 17-field schema (with sidecar sampler + malloc_trim columns).
    # Older 10-field rows are still readable by plot.py because the new
    # columns are appended at the end.
    return {
        "tool": parts[0],
        "N": int(parts[1]),
        "build_s": float(parts[2]),
        "solve_s": float(parts[3]),
        "rss_start_mb": float(parts[4]),
        "rss_after_build_mb": float(parts[5]),
        "rss_after_solve_mb": float(parts[6]),
        "peak_rss_mb": float(parts[7]),
        "rss_after_build_trim_mb": float(parts[8]),
        "rss_after_solve_trim_mb": float(parts[9]),
        "rss_solve_min_mb": float(parts[10]),
        "rss_solve_p50_mb": float(parts[11]),
        "rss_solve_p95_mb": float(parts[12]),
        "rss_solve_max_mb": float(parts[13]),
        "n_samples": int(parts[14]),
        "obj": float(parts[15]),
        "optimal": bool(int(parts[16])),
    }


def main() -> None:
    default_threads = [os.cpu_count() or 1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--tools", nargs="+", default=DEFAULT_TOOLS)
    ap.add_argument("--sizes", nargs="+", type=int, default=DEFAULT_SIZES)
    ap.add_argument(
        "--threads",
        nargs="+",
        type=int,
        default=default_threads,
        help=f"Cap on cores per cell (default: {default_threads})",
    )
    ap.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S)
    ap.add_argument(
        "--time-limit",
        type=float,
        default=None,
        help="HiGHS time-limit in seconds. With ~1e-6 the LP "
        "solve short-circuits, so solve_s captures only "
        "the per-tool model→HiGHS handoff.",
    )
    ap.add_argument("--out", default=str(HERE / "results" / "results.csv"))
    ap.add_argument(
        "--append", action="store_true", help="Append to existing CSV instead of overwriting."
    )
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "tool",
        "N",
        "threads",
        "rep",
        "build_s",
        "solve_s",
        "rss_start_mb",
        "rss_after_build_mb",
        "rss_after_solve_mb",
        "peak_rss_mb",
        "rss_after_build_trim_mb",
        "rss_after_solve_trim_mb",
        "rss_solve_min_mb",
        "rss_solve_p50_mb",
        "rss_solve_p95_mb",
        "rss_solve_max_mb",
        "n_samples",
        "obj",
        "optimal",
    ]

    write_header = not (args.append and out.exists())
    mode = "a" if args.append else "w"
    with out.open(mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()

        for N in args.sizes:
            for threads in args.threads:
                for tool in args.tools:
                    for rep in range(args.repeats):
                        print(
                            f"running {tool:8s} N={N:<5d} threads={threads:<3d} "
                            f"rep={rep + 1}/{args.repeats}",
                            file=sys.stderr,
                        )
                        res = run_cell(
                            tool,
                            N,
                            threads,
                            args.timeout,
                            time_limit=args.time_limit,
                        )
                        row: dict = {
                            "tool": tool,
                            "N": N,
                            "threads": threads,
                            "rep": rep,
                        }
                        if res is None:
                            row.update(
                                build_s=float("nan"),
                                solve_s=float("nan"),
                                rss_start_mb=float("nan"),
                                rss_after_build_mb=float("nan"),
                                rss_after_solve_mb=float("nan"),
                                peak_rss_mb=float("nan"),
                                rss_after_build_trim_mb=float("nan"),
                                rss_after_solve_trim_mb=float("nan"),
                                rss_solve_min_mb=float("nan"),
                                rss_solve_p50_mb=float("nan"),
                                rss_solve_p95_mb=float("nan"),
                                rss_solve_max_mb=float("nan"),
                                n_samples=0,
                                obj=float("nan"),
                                optimal=False,
                            )
                        else:
                            row.update(
                                build_s=res["build_s"],
                                solve_s=res["solve_s"],
                                rss_start_mb=res["rss_start_mb"],
                                rss_after_build_mb=res["rss_after_build_mb"],
                                rss_after_solve_mb=res["rss_after_solve_mb"],
                                peak_rss_mb=res["peak_rss_mb"],
                                rss_after_build_trim_mb=res["rss_after_build_trim_mb"],
                                rss_after_solve_trim_mb=res["rss_after_solve_trim_mb"],
                                rss_solve_min_mb=res["rss_solve_min_mb"],
                                rss_solve_p50_mb=res["rss_solve_p50_mb"],
                                rss_solve_p95_mb=res["rss_solve_p95_mb"],
                                rss_solve_max_mb=res["rss_solve_max_mb"],
                                n_samples=res["n_samples"],
                                obj=res["obj"],
                                optimal=res["optimal"],
                            )
                        w.writerow(row)
                        f.flush()

    print(f"wrote results to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
