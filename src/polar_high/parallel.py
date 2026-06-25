"""Deterministic parallel solving of independent :class:`WarmProblem`s.

A small, domain-agnostic utility for cutting-plane / decomposition drivers
(e.g. Benders region recourse, Lagrangian subproblems) that need to solve N
*independent* subproblems each outer iteration.  Each subproblem is its own
:class:`~polar_high.engine.WarmProblem` (its own HiGHS handle), and HiGHS'
``run()`` releases the GIL, so a thread pool yields a real wall-clock speedup
over the sequential loop.

The thread-safety contract this module encapsulates (so the caller's algorithm
code never touches the HiGHS-threading detail):

* **Single-threaded HiGHS scheduler.**  Every subproblem solve runs HiGHS with
  a single-threaded scheduler.  This is required for (a) *determinism* — HiGHS
  is non-deterministic with ``threads > 1`` — and (b) to avoid
  ``workers × cores`` oversubscription.  The process-global HiGHS scheduler is
  pinned to one thread ONCE up front via :func:`prewarm_global_scheduler`; the
  per-subproblem ``run()`` calls then inherit that pinned pool.

* **Sequential cold first build.**  The FIRST ``WarmProblem.solve()`` builds the
  HiGHS model and, if it sees a ``threads`` / ``parallel`` option, calls the
  process-global ``resetGlobalScheduler`` — unsafe to run concurrently.  This
  module therefore only parallelizes solves over WarmProblems that are ALREADY
  built (``wp._h is not None``); it raises if asked to fan out an unbuilt one.
  Callers must do the cold first build sequentially (or rely on the scheduler
  pre-pin) before calling :func:`solve_indexed_parallel`.

* **Deterministic per-index collection.**  Results are collected into per-index
  slots and returned in index order, so the outcome is identical regardless of
  thread timing.  Worker exceptions are re-raised in index order (the lowest
  failing index wins, matching the sequential loop).

``workers <= 1`` keeps a fully sequential path (no pool, no prewarm) that is
byte-identical to a plain ``for`` loop — so the parallel and sequential code
paths can be exercised against each other for a determinism gate.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

import numpy as np

from polar_high.engine import WarmProblem

__all__ = [
    "prewarm_global_scheduler",
    "resolve_worker_count",
    "solve_indexed_parallel",
]

T = TypeVar("T")


def resolve_worker_count(n_items: int, workers: int | None) -> int:
    """Clamp a requested worker count to ``[1, n_items]``.

    ``workers is None`` auto-resolves to ``min(n_items, cpu_count - 1)`` (at
    least 1) — leave one core for the main thread / OS.  A non-positive request
    means sequential (1).
    """
    if n_items <= 0:
        return 1
    if workers is None:
        cpu = os.cpu_count() or 1
        auto = max(1, cpu - 1)
        return max(1, min(n_items, auto))
    return max(1, min(int(workers), n_items))


def prewarm_global_scheduler(threads: int = 1) -> bool:
    """Initialize HiGHS' process-global task scheduler ONCE, single-threaded,
    so subsequent concurrent solves on distinct ``Highs`` instances need not
    each call ``resetGlobalScheduler``.  Best-effort; returns ``False`` if any
    highspy step fails (the caller then falls back to a sequential path).

    Once this returns ``True`` the global scheduler is pinned to ``threads`` and
    a subsequent ``run()`` with NO ``threads`` option inherits that pool — so
    concurrent solves need not (and must not) pass ``threads`` per instance,
    which would re-trigger ``resetGlobalScheduler`` and is unsafe to run
    concurrently.
    """
    try:
        import highspy

        h = highspy.Highs()
        try:
            h.resetGlobalScheduler(False)
        except Exception:  # noqa: BLE001 — best-effort no-op on old highspy
            pass
        h.setOptionValue("output_flag", False)
        h.setOptionValue("threads", threads)
        # Trivial 1-col / 0-row LP forces scheduler init at the pinned thread
        # count.  Mirror the minimal HighsLp idiom WarmProblem's build uses so
        # it can't break on this highspy version.
        lp = highspy.HighsLp()
        lp.num_col_ = 1
        lp.num_row_ = 0
        lp.col_cost_ = np.array([0.0])
        lp.col_lower_ = np.array([0.0])
        lp.col_upper_ = np.array([1.0])
        lp.row_lower_ = np.array([])
        lp.row_upper_ = np.array([])
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.num_col_ = 1
        lp.a_matrix_.num_row_ = 0
        lp.a_matrix_.start_ = np.array([0, 0])
        lp.a_matrix_.index_ = np.array([], dtype=np.int32)
        lp.a_matrix_.value_ = np.array([])
        h.passModel(lp)
        h.run()
        return True
    except Exception:  # noqa: BLE001 — best-effort; caller falls back to sequential
        return False


def solve_indexed_parallel(
    warmproblems: Sequence[WarmProblem],
    fn: Callable[[int], T],
    *,
    workers: int | None = None,
) -> list[T]:
    """Run ``fn(i)`` for every index ``i`` of ``warmproblems`` deterministically,
    optionally across a thread pool, and return the results in index order.

    ``fn(i)`` is the caller's per-subproblem work: typically it pins the i-th
    :class:`WarmProblem`'s coupling columns, calls ``wp.solve()``, and extracts
    a domain result (objective + dual slopes).  Because each WarmProblem owns
    its own HiGHS handle, ``fn(i)`` for distinct ``i`` touch disjoint state and
    are safe to run concurrently — *provided* every WarmProblem is already built
    (the cold first build calls the process-global ``resetGlobalScheduler`` and
    must run sequentially up front).  This helper enforces that precondition.

    Parameters
    ----------
    warmproblems
        The per-index :class:`WarmProblem`s.  Each MUST already be built
        (``solve()`` called at least once); a ``ValueError`` is raised
        otherwise.  Only used for the built-precondition check + count — the
        actual solving is delegated to ``fn``.
    fn
        ``fn(i) -> result`` for index ``i``.  May run on a worker thread; must
        confine its mutations to the i-th WarmProblem's HiGHS handle.
    workers
        Effective worker count (see :func:`resolve_worker_count`).  ``None``
        auto-resolves to ``min(n, cpu_count - 1)``.  ``<= 1`` runs a fully
        sequential path on the calling thread (no pool, no scheduler prewarm) —
        byte-identical to a plain ``for`` loop.

    Returns
    -------
    list
        ``[fn(0), fn(1), ..., fn(n-1)]`` — always in index order, independent of
        thread timing.
    """
    n = len(warmproblems)
    for i, wp in enumerate(warmproblems):
        if wp._h is None:
            raise ValueError(
                f"solve_indexed_parallel: WarmProblem at index {i} is not built; "
                f"call its .solve() once (sequential cold build) before parallel "
                f"solving."
            )
    eff = resolve_worker_count(n, workers)

    if eff <= 1 or n <= 1:
        return [fn(i) for i in range(n)]

    # Pin the process-global HiGHS scheduler to one thread ONCE before fanning
    # out, so the concurrent run() calls inherit a single-threaded pool (no
    # oversubscription) and stay deterministic.  Best-effort: a failed prewarm
    # does not change correctness here (the WarmProblems are already built, so no
    # concurrent resetGlobalScheduler can fire), it only forgoes the explicit
    # pinning.
    prewarm_global_scheduler(1)

    with ThreadPoolExecutor(max_workers=eff) as pool:
        futs = {i: pool.submit(fn, i) for i in range(n)}
        out: list[T] = [None] * n  # type: ignore[list-item]
        try:
            for i in range(n):
                out[i] = futs[i].result()  # re-raises worker exceptions, index order
        except BaseException:
            for f in futs.values():
                f.cancel()
            raise  # `with` still does shutdown(wait=True)
        return out
