"""Single-cell benchmark worker.

Usage:  python run_one.py <tool> <N>

Imports the requested tool's model module, builds and solves the LP,
and prints one CSV line to stdout:

    tool,N,build_s,solve_s,
    rss_start_mb,rss_after_build_mb,rss_after_solve_mb,peak_rss_mb,
    rss_after_build_trim_mb,rss_after_solve_trim_mb,
    rss_solve_min_mb,rss_solve_p50_mb,rss_solve_p95_mb,rss_solve_max_mb,
    n_samples,
    obj,optimal

Memory measurement strategy:
- ``rss_start_mb`` / ``rss_after_build_mb`` / ``rss_after_solve_mb``  —
  point-in-time RSS at the obvious checkpoints (start, post-build,
  post-solve), each preceded by ``gc.collect()``.
- ``peak_rss_mb`` — ``ru_maxrss`` for the whole process, the unavoidable
  high-water mark including transient HiGHS-setup peaks.
- ``rss_*_trim_mb`` — RSS at the same checkpoints **after** calling
  ``malloc_trim(0)``, which forces glibc to return freed-but-cached
  arenas to the OS.  Diagnostic value: a big gap between
  ``rss_after_solve_mb`` and ``rss_after_solve_trim_mb`` means most of
  the apparent "memory still used" is glibc holding onto already-freed
  pages, not the application actually needing them.  (Cannot affect
  ``peak_rss_mb``: at the moment of peak, all memory is in use, so
  there is nothing to trim.)
- ``rss_solve_*_mb`` / ``n_samples`` — a sidecar thread samples RSS at
  ~25 ms cadence while ``solve()`` runs.  ``max`` should align with
  ``peak_rss_mb``; ``p50`` (median) is the steady-state working set
  during the bulk of the solve, useful for comparing tools after
  transient setup spikes wash out.
"""

from __future__ import annotations

import ctypes
import gc
import importlib
import os
import resource
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

_libc = ctypes.CDLL("libc.so.6", use_errno=False)


def _malloc_trim() -> None:
    """Return freed-but-cached glibc arenas to the OS.  No-op at peak
    consumption — only releases pages that have already been free()'d
    and are sitting in glibc's per-thread arenas."""
    try:
        _libc.malloc_trim(0)
    except Exception:
        pass


def _rss_mb() -> float:
    """Current resident-set size in MB, parsed from /proc/self/status
    on Linux.  Avoids the psutil import overhead in the sampler thread."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                kb = int(line.split()[1])
                return kb / 1024.0
    return 0.0


def _peak_rss_mb() -> float:
    """High-water mark RSS for the process in MB (``ru_maxrss``)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


class _RssSampler:
    """Background thread that polls VmRSS at a fixed cadence.  Used to
    capture the time-series of RSS during ``solve()`` so we can report
    not just the peak but also the steady-state value (p50) and a
    representative tail (p95)."""

    def __init__(self, interval_s: float = 0.025) -> None:
        self.interval_s = interval_s
        self.samples: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join()
        self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.samples.append(_rss_mb())
            self._stop.wait(self.interval_s)


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = (len(s) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


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
    _malloc_trim()
    rss_after_build_trim = _rss_mb()

    time_limit_env = os.environ.get("BENCH_TIME_LIMIT")
    solve_kwargs: dict = {}
    if time_limit_env:
        solve_kwargs["time_limit"] = float(time_limit_env)

    sampler = _RssSampler(interval_s=0.025)
    sampler.start()
    t1 = time.perf_counter()
    optimal, obj = mod.solve(model, **solve_kwargs)
    solve_s = time.perf_counter() - t1
    sampler.stop()

    gc.collect()
    rss_after_solve = _rss_mb()
    _malloc_trim()
    rss_after_solve_trim = _rss_mb()

    peak_rss = _peak_rss_mb()
    s = sampler.samples
    rss_solve_min = min(s) if s else 0.0
    rss_solve_p50 = _pct(s, 0.50)
    rss_solve_p95 = _pct(s, 0.95)
    rss_solve_max = max(s) if s else 0.0

    print(
        f"{tool},{N},{build_s:.6f},{solve_s:.6f},"
        f"{rss_start:.2f},{rss_after_build:.2f},{rss_after_solve:.2f},"
        f"{peak_rss:.2f},"
        f"{rss_after_build_trim:.2f},{rss_after_solve_trim:.2f},"
        f"{rss_solve_min:.2f},{rss_solve_p50:.2f},"
        f"{rss_solve_p95:.2f},{rss_solve_max:.2f},"
        f"{len(s)},"
        f"{obj:.6e},{int(optimal)}"
    )


if __name__ == "__main__":
    main()
