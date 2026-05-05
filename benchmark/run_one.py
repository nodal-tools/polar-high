"""Single-cell benchmark worker.

Usage:  python run_one.py <tool> <N>

Imports the requested tool's model module, builds and solves the LP,
and prints one CSV line to stdout:

    tool,N,build_s,solve_s,
    rss_start_mb,rss_after_build_mb,rss_after_solve_mb,peak_rss_mb,
    obj,optimal

Memory measurement strategy:
- ``rss_start_mb``         — resident-set size at the top of main(),
                             before importing the tool's model module.
- ``rss_after_build_mb``   — RSS right after build() returns and a
                             ``gc.collect()`` has run, so retained
                             intermediates have a chance to release.
- ``rss_after_solve_mb``   — RSS right after solve() returns plus
                             another ``gc.collect()``.
- ``peak_rss_mb``          — ``ru_maxrss`` (high-water mark across
                             the whole process). Same metric as the
                             original column for backwards-compat.
"""
from __future__ import annotations

import gc
import importlib
import os
import resource
import sys
import time
from pathlib import Path

import psutil

sys.path.insert(0, str(Path(__file__).parent))


def _rss_mb() -> float:
    """Current resident-set size in MB."""
    return psutil.Process().memory_info().rss / (1024 * 1024)


def _peak_rss_mb() -> float:
    """High-water mark RSS for the process in MB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: run_one.py <tool> <N>", file=sys.stderr)
        sys.exit(2)

    tool = sys.argv[1]
    N = int(sys.argv[2])

    rss_start = _rss_mb()

    mod = importlib.import_module(f"models.{tool}")

    t0 = time.perf_counter()
    model = mod.build(N)
    build_s = time.perf_counter() - t0
    gc.collect()
    rss_after_build = _rss_mb()

    time_limit_env = os.environ.get("BENCH_TIME_LIMIT")
    solve_kwargs: dict = {}
    if time_limit_env:
        solve_kwargs["time_limit"] = float(time_limit_env)

    t1 = time.perf_counter()
    optimal, obj = mod.solve(model, **solve_kwargs)
    solve_s = time.perf_counter() - t1
    gc.collect()
    rss_after_solve = _rss_mb()

    peak_rss = _peak_rss_mb()

    print(
        f"{tool},{N},{build_s:.6f},{solve_s:.6f},"
        f"{rss_start:.2f},{rss_after_build:.2f},{rss_after_solve:.2f},"
        f"{peak_rss:.2f},"
        f"{obj:.6e},{int(optimal)}"
    )


if __name__ == "__main__":
    main()
