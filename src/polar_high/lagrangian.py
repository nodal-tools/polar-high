"""Generic Lagrangian decomposition for coupled :class:`Problem`s.

A domain-agnostic dual-subgradient driver for N independent LP
subproblems linked by linear coupling constraints

    Σ_i  coef_i · col_i  =  rhs

Each :class:`CouplingSpec` carries a list of
``(subproblem_idx, var_name, dim_tuple, coef)`` entries plus an
optional ``rhs`` (default 0).  The most common use is the 2-entry
consensus coupling ``x_A == x_B`` with coefs +1 / -1, rhs 0.

Algorithm:
  1. Bump each entry's column cost by ``coef · λ`` (relaxes the
     coupling residual into the objective).
  2. Solve every subproblem (warm-started after iter 1).
  3. Compute residual ``Σ coef_i · x_i − rhs`` per cell.
  4. Subgradient step ``λ ← λ + (step / √k) · residual``.
  5. Tail-window primal averaging → fix-and-resolve for a feasible
     primal upper bound; report the *best dual* (max Σ obj across
     iters) as the tight lower bound.

Knows nothing about half-flows or regions — that lives in the
flextool-side wrapper.
"""

from __future__ import annotations

import contextlib
import math
import os
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from polar_high.engine import Problem, WarmProblem

__all__ = ["CouplingEntry", "CouplingSpec", "LagrangianProblem", "LagrangianSolution"]


def _prewarm_global_scheduler(threads: int = 1) -> bool:
    """Initialize HiGHS' process-global task scheduler ONCE, single-threaded,
    so subsequent concurrent first-solves on distinct Highs instances need
    not each call resetGlobalScheduler.  Best-effort; returns False if any
    highspy step fails (caller then falls back to a sequential cold build).

    Once this returns True the global scheduler is pinned to ``threads`` and a
    subsequent ``run()`` with NO ``threads`` option inherits that pool — so the
    concurrent cold builds need not (and must not) pass ``threads`` per
    instance, which would re-trigger ``resetGlobalScheduler`` and is unsafe to
    run concurrently.
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
        # Trivial 1-col / 0-row LP to force scheduler init at the pinned thread
        # count.  Mirror the minimal HighsLp idiom used by
        # WarmProblem._initial_build (engine.py ~7290) so it can't break on this
        # highspy version.
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


@dataclass(frozen=True)
class CouplingEntry:
    """One participant in a :class:`CouplingSpec`.  ``dim_tuples`` has
    one tuple per coupling cell; entries in one CouplingSpec must
    share length (entry-i tuple-k pairs with entry-j tuple-k under
    the same λ_k)."""

    subproblem_idx: int
    var_name: str
    dim_tuples: list[tuple]
    coef: float = 1.0


@dataclass
class CouplingSpec:
    """A linear coupling family across subproblems: per cell ``k``,
    ``Σ_e coef_e · x[entries[e].cols[k]]  =  rhs[k]``.  ``rhs`` is a
    scalar or an array sized to the cell count; default 0."""

    entries: list[CouplingEntry]
    rhs: float | np.ndarray = 0.0
    key: object | None = None


@dataclass
class LagrangianSolution:
    """Result bundle from :meth:`LagrangianProblem.solve`.

    ``total_objective`` is the chosen reported total; ``report_kind``
    is ``"best_dual"`` (always for now — best LB across iters).
    ``final_lambdas`` and ``primal_recovery`` are ordered like
    ``LagrangianProblem.couplings``.  The trailing iteration_log
    entry has ``iter == -1`` and carries report_kind / dual / primal
    summary fields.
    """

    converged: bool
    iterations: int
    total_objective: float
    report_kind: str
    subproblem_objectives: list[float]
    iteration_log: list[dict]
    final_lambdas: list[np.ndarray]
    primal_recovery: list[np.ndarray] = field(default_factory=list)
    best_dual_total: float = 0.0
    recovered_total: float = 0.0
    # One numpy float64 ``col_value`` array per subproblem (region), in
    # subproblem order, each the region's FINAL recovered-primal column
    # values.  Lets callers reconstruct a whole-system primal by indexing
    # ``subproblems[i]._vars[name].frame['col_id']`` into entry ``i``.
    # Opt-in/backward-compatible: default empty list.
    subproblem_col_values: list = field(default_factory=list)


@dataclass
class _ResolvedEntry:
    subproblem_idx: int
    var_name: str
    cols: np.ndarray  # int64
    cols_i32: np.ndarray  # int32 view for HiGHS
    coef: float
    base_costs: np.ndarray  # float64, length n_cells


@dataclass
class _ResolvedCoupling:
    spec: CouplingSpec
    n_cells: int
    rhs: np.ndarray
    entries: list[_ResolvedEntry]
    lam: np.ndarray
    last_residual: np.ndarray = field(default_factory=lambda: np.zeros(0))


class LagrangianProblem:
    """Lagrangian decomposition driver.  Build N :class:`Problem`s,
    list the cross-subproblem :class:`CouplingSpec`s, then call
    ``LagrangianProblem(subproblems, couplings).solve(...)``.
    """

    def __init__(self, subproblems: Sequence[Problem], couplings: Sequence[CouplingSpec]) -> None:
        if len(subproblems) < 1:
            raise ValueError("LagrangianProblem: need at least one subproblem")
        for i, p in enumerate(subproblems):
            if not isinstance(p, Problem):
                raise TypeError(
                    f"LagrangianProblem: subproblem {i} is {type(p).__name__}, expected Problem"
                )
        self._subproblems: list[Problem] = list(subproblems)
        self._couplings: list[CouplingSpec] = list(couplings)
        self._warm: list[WarmProblem] = [WarmProblem(p) for p in self._subproblems]
        self._resolved: list[_ResolvedCoupling] = []

    @property
    def subproblems(self) -> list[Problem]:
        return list(self._subproblems)

    @property
    def couplings(self) -> list[CouplingSpec]:
        return list(self._couplings)

    @property
    def warm_problems(self) -> list[WarmProblem]:
        return list(self._warm)

    def solve(
        self,
        *,
        max_iters: int = 100,
        tol: float = 1.0,
        step: float = 1.0,
        initial_lambda: float = 0.0,
        min_iters: int = 1,
        primal_tail: int | None = None,
        progress_callback: Callable[[dict], None] | None = None,
        max_workers: int | None = None,
        subsolve_callback: Callable[[dict], None] | None = None,
    ) -> LagrangianSolution:
        """Run the dual-subgradient loop.

        ``step / √k`` is the diminishing step on iter ``k``.
        ``initial_lambda`` is a non-zero seed (breaks trivial 0-flow
        equilibria).  ``min_iters`` floors the iteration count so the
        early-termination test can't fire on iter 1.  ``primal_tail``
        defaults to ``max(20, max_iters//4)``.

        ``progress_callback`` — optional callable invoked once per outer
        iteration with that iteration's log dict (keys ``iter``,
        ``alpha_k``, ``max_abs_residual``, ``total_obj``), and once more
        at the end with the final-summary dict (``iter == -1``, carrying
        ``best_dual_total`` / ``recovered_total``).  Lets callers stream
        live progress; ``None`` (default) is a no-op and preserves the
        silent behaviour.  Callback exceptions are suppressed so a
        faulty observer can never abort the solve.

        ``max_workers`` — optional cap on the number of worker threads
        used to solve subproblems concurrently within each barrier
        (initial / per-iteration / recovery).  ``None`` (default) or
        ``1`` keeps today's fully sequential behaviour.  The effective
        worker count is clamped to ``[1, n_subproblems]``.  When >1, every
        ``h.run()`` uses a single-threaded HiGHS scheduler so it is
        deterministic (HiGHS is non-deterministic with threads>1) and the box
        is not oversubscribed.  The COLD initial build also parallelizes across
        regions: the process-global HiGHS scheduler is pre-pinned to one thread
        ONCE up front (:func:`_prewarm_global_scheduler`), after which the
        first solves fan out concurrently WITHOUT passing ``threads`` (so no
        per-instance ``resetGlobalScheduler``).  If that one-time prewarm fails
        the build falls back to a sequential cold loop on the calling thread
        (threads=1 per first solve pins the scheduler), and the warm iterations
        still parallelize.  Bit-identical to the sequential cold build.

        ``subsolve_callback`` — optional callable invoked at the start and
        finish of every individual subproblem solve, with a dict carrying
        ``event`` (``"start"`` / ``"finish"``), ``iter``, ``subproblem``
        and ``phase`` (``"initial"`` / ``"iterate"`` / ``"recovery"``);
        ``finish`` entries additionally carry ``obj`` when the subsolve
        reached optimality.  It fires from worker threads when
        ``max_workers > 1`` and MUST be thread-safe.  Exceptions are
        suppressed so a faulty observer can never abort the solve.
        ``None`` (default) is a no-op.

        When the caller uses the new functionality (``max_workers > 1`` or
        a ``subsolve_callback``) the per-subsolve HiGHS native log is
        silenced; set ``POLAR_HIGH_LAGRANGIAN_VERBOSE=1`` to force the
        verbose native log.  Plain existing callers keep today's verbose
        native log.
        """

        def _emit(entry: dict) -> None:
            if progress_callback is None:
                return
            try:
                progress_callback(entry)
            except Exception:  # noqa: BLE001 — an observer must not break the solve
                pass

        n_sp = len(self._warm)
        _eff_workers = max(1, min(max_workers if max_workers is not None else 1, n_sp))
        _parallel = _eff_workers > 1
        # Silence the per-subsolve HiGHS log when the caller uses the new
        # functionality (parallel OR a progress hook); plain existing callers
        # keep today's verbose native log. Env override forces verbose.
        _silence = (_parallel or subsolve_callback is not None) and not os.environ.get(
            "POLAR_HIGH_LAGRANGIAN_VERBOSE"
        )
        _subsolve_options: dict = {}
        if _silence:
            _subsolve_options["output_flag"] = False
        if _parallel:
            # threads=1 per subsolve: avoid N_workers x cores oversubscription
            # AND make each h.run() deterministic (HiGHS is non-deterministic
            # with threads>1). For the WARM iterations/recovery the instances
            # are already built, so passing "threads" here is moot (WarmProblem
            # ignores options after the first solve). The COLD build's thread
            # pinning is handled by the prewarm / sequential-fallback paths
            # below, so the per-iteration option dict carries threads=1 only as
            # a belt-and-suspenders no-op for any code that re-reads it.
            _subsolve_options["threads"] = 1

        # Pre-pin HiGHS' process-global scheduler to a single thread ONCE, up
        # front (single-threaded), so the COLD initial-build loop can fan out
        # across regions concurrently. Each first wp.solve() builds its HiGHS
        # model and, if it sees a "threads"/"parallel" option, calls the
        # process-global resetGlobalScheduler (engine.py ~7333) — unsafe to run
        # concurrently. With the scheduler pre-pinned we build in parallel
        # WITHOUT passing "threads", so no per-instance reset occurs and every
        # run() (cold + warm) inherits the pinned single-thread pool. Proven
        # bit-identical to the sequential cold build. Best-effort: if the
        # prewarm fails we fall back to the sequential cold build (which pins
        # the scheduler via threads=1 on each first solve).
        _cold_parallel = False
        if _parallel:
            _cold_parallel = _prewarm_global_scheduler(1)

        # Option dicts for the FIRST (build) solve:
        #  * prewarmed path — scheduler already pinned, so DO NOT pass "threads"
        #    (avoids re-triggering resetGlobalScheduler concurrently).
        #  * sequential fallback — today's behaviour: threads=1 on each first
        #    solve pins the scheduler via the sequential builds, keeping the
        #    warm phase parallel-safe even when the prewarm failed.
        _first_opts_prewarmed = {"output_flag": False} if _silence else None
        _first_opts_seq = _subsolve_options or None

        def _fire_subsolve(event: str, *, it: int, i: int, phase: str, obj=None) -> None:
            if subsolve_callback is None:
                return
            entry = {"event": event, "iter": it, "subproblem": i, "phase": phase}
            if obj is not None:
                entry["obj"] = obj
            try:
                subsolve_callback(entry)
            except Exception:  # noqa: BLE001 — an observer must not break the solve
                pass

        def _run_one(i, wp, *, it, phase, options=None):
            _fire_subsolve("start", it=it, i=i, phase=phase)
            sol = wp.solve(options=options)  # ignored after the first solve
            _fire_subsolve(
                "finish", it=it, i=i, phase=phase, obj=(sol.obj if sol.optimal else None)
            )
            return i, sol

        pool_cm = (
            ThreadPoolExecutor(max_workers=_eff_workers)
            if _parallel
            else contextlib.nullcontext(None)
        )
        with pool_cm as _pool:

            def _solve_all(phase, it, options=None):
                if _pool is None:
                    return [
                        _run_one(i, wp, it=it, phase=phase, options=options)
                        for i, wp in enumerate(self._warm)
                    ]
                futs = {
                    i: _pool.submit(_run_one, i, wp, it=it, phase=phase, options=options)
                    for i, wp in enumerate(self._warm)
                }
                out = [None] * len(self._warm)
                try:
                    for i in range(len(self._warm)):
                        out[i] = futs[i].result()  # re-raises worker exceptions, in index order
                except BaseException:
                    for f in futs.values():
                        f.cancel()
                    raise  # `with` still does shutdown(wait=True)
                return out

            # Initial solve — also builds each WarmProblem's HiGHS state.
            # Retain each region's col_value so the trivial (no-coupling)
            # early-return path can hand back a full-length
            # ``subproblem_col_values``, and so the main loop has a seeded
            # fallback for any region whose recovery solve is skipped.
            #
            # The first wp.solve() builds each WarmProblem; if it sees a
            # "threads"/"parallel" option it calls the process-global
            # resetGlobalScheduler (engine.py ~7333), which is unsafe to run
            # concurrently. Two build paths:
            #   * PARALLEL (``_cold_parallel``): the global scheduler was
            #     pre-pinned to 1 thread up front, so the first solves can fan
            #     out across regions on ``_pool``. They pass output_flag only
            #     (no "threads" => no per-instance resetGlobalScheduler).
            #     Bit-identical to the sequential cold build (proven).
            #   * SEQUENTIAL fallback (today's path): build one region at a time
            #     on the calling thread. threads=1 on each first solve pins the
            #     scheduler so the WARM iterations below stay parallel-safe even
            #     when the prewarm failed or max_workers<=1.
            first_obj: list[float] = [0.0] * n_sp
            last_col_values: list[np.ndarray] = [None] * len(self._warm)  # type: ignore[list-item]
            if _cold_parallel:
                for i, sol in _solve_all("initial", 0, options=_first_opts_prewarmed):
                    if not sol.optimal:
                        raise RuntimeError(
                            f"LagrangianProblem: initial solve for subproblem {i} did not reach optimality"
                        )
                    first_obj[i] = sol.obj
                    last_col_values[i] = sol.col_value.copy()
            else:
                for i, wp in enumerate(self._warm):
                    _, sol = _run_one(i, wp, it=0, phase="initial", options=_first_opts_seq)
                    if not sol.optimal:
                        raise RuntimeError(
                            f"LagrangianProblem: initial solve for subproblem {i} did not reach optimality"
                        )
                    first_obj[i] = sol.obj
                    last_col_values[i] = sol.col_value.copy()

            self._resolved = self._resolve_couplings(initial_lambda)
            # Snapshot per-cell base objective costs from the live LPs so
            # the per-iter ``cost = base + coef·λ`` push-down is correct
            # for variables that have a non-zero base cost.  HiGHS exposes
            # the column-cost vector via getLp(); we grab the entries we
            # need.
            for rc in self._resolved:
                for ent in rc.entries:
                    lp = self._warm[ent.subproblem_idx]._h.getLp()
                    col_cost = np.asarray(lp.col_cost_, dtype=np.float64)
                    ent.base_costs = col_cost[ent.cols].astype(np.float64, copy=True)
            if not self._resolved:
                return LagrangianSolution(
                    converged=True,
                    iterations=0,
                    total_objective=sum(first_obj),
                    report_kind="best_dual",
                    subproblem_objectives=list(first_obj),
                    iteration_log=[],
                    final_lambdas=[],
                    primal_recovery=[],
                    best_dual_total=sum(first_obj),
                    recovered_total=sum(first_obj),
                    subproblem_col_values=[cv.copy() for cv in last_col_values],
                )

            if primal_tail is None:
                primal_tail = max(20, max_iters // 4)

            iteration_log: list[dict] = []
            converged = False
            last_obj = list(first_obj)
            # Per-coupling per-entry tail accumulators.
            sum_entry_vals: list[list[np.ndarray]] = [
                [np.zeros(rc.n_cells) for _ in rc.entries] for rc in self._resolved
            ]
            tail_count = 0

            max_abs_res = float("inf")
            it = 0
            for it in range(1, max_iters + 1):
                alpha_k = step / math.sqrt(it)

                # Apply per-cell λ to every entry's column costs.
                for rc in self._resolved:
                    for ent in rc.entries:
                        new_cost = ent.base_costs + ent.coef * rc.lam
                        self._warm[ent.subproblem_idx]._h.changeColsCost(
                            int(ent.cols_i32.size),
                            ent.cols_i32,
                            new_cost,
                        )

                # Solve every subproblem (optionally in parallel; collected
                # in index order, so the raise fires on the lowest non-optimal
                # index — same as the sequential path).
                iter_obj = [0.0] * n_sp
                primal_by_sp: dict[int, np.ndarray] = {}
                for i, sol in _solve_all("iterate", it):
                    if not sol.optimal:
                        raise RuntimeError(
                            f"LagrangianProblem iter {it}: subproblem {i} did not reach optimality"
                        )
                    iter_obj[i] = sol.obj
                    primal_by_sp[i] = sol.col_value
                    last_col_values[i] = sol.col_value.copy()

                # Residual = Σ coef · x − rhs, per cell.
                max_abs_res = 0.0
                in_tail = it > max_iters - primal_tail
                for ic, rc in enumerate(self._resolved):
                    res = -rc.rhs.copy()
                    for ie, ent in enumerate(rc.entries):
                        vals = primal_by_sp[ent.subproblem_idx][ent.cols]
                        res = res + ent.coef * vals
                        if in_tail:
                            sum_entry_vals[ic][ie] += vals
                    rc.last_residual = res
                    cell_max = float(np.abs(res).max()) if res.size else 0.0
                    if cell_max > max_abs_res:
                        max_abs_res = cell_max
                if in_tail:
                    tail_count += 1

                iteration_log.append(
                    {
                        "iter": it,
                        "alpha_k": alpha_k,
                        "max_abs_residual": max_abs_res,
                        "total_obj": sum(iter_obj),
                    }
                )
                _emit(iteration_log[-1])

                last_obj = iter_obj
                if max_abs_res < tol and it >= min_iters:
                    converged = True
                    break

                for rc in self._resolved:
                    rc.lam = rc.lam + alpha_k * rc.last_residual

            # Primal recovery: tail-average then fix-and-resolve.
            recovery_obj = list(last_obj)
            primal_recovery: list[np.ndarray] = []
            best_dual_total = max(
                (log["total_obj"] for log in iteration_log), default=sum(first_obj)
            )

            if tail_count > 0:
                avg_entry_vals: list[list[np.ndarray]] = [
                    [s / tail_count for s in row] for row in sum_entry_vals
                ]
                max_avg_res = 0.0
                for ic, rc in enumerate(self._resolved):
                    res = -rc.rhs.copy()
                    for ie, ent in enumerate(rc.entries):
                        res = res + ent.coef * avg_entry_vals[ic][ie]
                    cell_max = float(np.abs(res).max()) if res.size else 0.0
                    max_avg_res = max(max_avg_res, cell_max)

                # 2-entry consensus (coefs +1/-1, rhs 0) → fix both sides
                # to ½(avg_pos + avg_neg).  Otherwise fix each entry to its
                # own tail mean.
                for ic, rc in enumerate(self._resolved):
                    if (
                        len(rc.entries) == 2
                        and rc.entries[0].coef == 1.0
                        and rc.entries[1].coef == -1.0
                        and np.allclose(rc.rhs, 0.0)
                    ):
                        consensus = 0.5 * (avg_entry_vals[ic][0] + avg_entry_vals[ic][1])
                        fix_vals = [consensus, consensus]
                    else:
                        fix_vals = avg_entry_vals[ic]
                    primal_recovery.append(fix_vals[0].copy())

                    for ie, ent in enumerate(rc.entries):
                        wp = self._warm[ent.subproblem_idx]
                        wp._h.changeColsCost(
                            int(ent.cols_i32.size), ent.cols_i32, ent.base_costs.astype(np.float64)
                        )
                        fv = fix_vals[ie].astype(np.float64)
                        wp._h.changeColsBounds(int(ent.cols_i32.size), ent.cols_i32, fv, fv)

                # Collected in index order; LENIENT (no raise — preserve the
                # ``if sol.optimal`` fallback to the most recent iterate).
                for i, sol in _solve_all("recovery", -1):
                    if sol.optimal:
                        recovery_obj[i] = sol.obj
                        last_col_values[i] = sol.col_value.copy()
                if max_avg_res < tol:
                    converged = True

            recovered_total = sum(recovery_obj)
            # For minimisation: best_dual is the tight lower bound.
            reported_total = best_dual_total
            report_kind = "best_dual"
            iteration_log.append(
                {
                    "iter": -1,
                    "report_kind": report_kind,
                    "best_dual_total": best_dual_total,
                    "recovered_total": recovered_total,
                }
            )
            _emit(iteration_log[-1])

            return LagrangianSolution(
                converged=converged,
                iterations=it,
                total_objective=reported_total,
                report_kind=report_kind,
                subproblem_objectives=list(recovery_obj),
                iteration_log=iteration_log,
                final_lambdas=[rc.lam.copy() for rc in self._resolved],
                primal_recovery=primal_recovery,
                best_dual_total=best_dual_total,
                recovered_total=recovered_total,
                subproblem_col_values=[cv.copy() for cv in last_col_values],
            )

    def _resolve_couplings(self, initial_lambda: float) -> list[_ResolvedCoupling]:
        n_sp = len(self._subproblems)
        out: list[_ResolvedCoupling] = []
        for ic, spec in enumerate(self._couplings):
            if not spec.entries:
                raise ValueError(f"CouplingSpec[{ic}]: must have at least one entry")
            n_cells = len(spec.entries[0].dim_tuples)
            entries: list[_ResolvedEntry] = []
            for ie, ent in enumerate(spec.entries):
                if ent.subproblem_idx < 0 or ent.subproblem_idx >= n_sp:
                    raise ValueError(
                        f"CouplingSpec[{ic}].entries[{ie}]: "
                        f"subproblem_idx={ent.subproblem_idx} out of range "
                        f"(n_subproblems={n_sp})"
                    )
                wp = self._warm[ent.subproblem_idx]
                if ent.var_name not in wp._p._vars:
                    raise ValueError(
                        f"CouplingSpec[{ic}].entries[{ie}]: variable "
                        f"{ent.var_name!r} not declared in subproblem "
                        f"{ent.subproblem_idx}"
                    )
                if len(ent.dim_tuples) != n_cells:
                    raise ValueError(
                        f"CouplingSpec[{ic}]: entry {ie} has "
                        f"{len(ent.dim_tuples)} dim_tuples; entry 0 has "
                        f"{n_cells}.  Cell counts must match."
                    )
                cols = wp._resolve_dim_tuples(ent.var_name, ent.dim_tuples)
                entries.append(
                    _ResolvedEntry(
                        subproblem_idx=ent.subproblem_idx,
                        var_name=ent.var_name,
                        cols=cols,
                        cols_i32=cols.astype(np.int32, copy=False),
                        coef=float(ent.coef),
                        base_costs=np.zeros(n_cells, dtype=np.float64),
                    )
                )
            if isinstance(spec.rhs, np.ndarray):
                if spec.rhs.size != n_cells:
                    raise ValueError(
                        f"CouplingSpec[{ic}]: rhs size {spec.rhs.size} != cell count {n_cells}"
                    )
                rhs_vec = spec.rhs.astype(np.float64, copy=True)
            else:
                rhs_vec = np.full(n_cells, float(spec.rhs), dtype=np.float64)
            out.append(
                _ResolvedCoupling(
                    spec=spec,
                    n_cells=n_cells,
                    rhs=rhs_vec,
                    entries=entries,
                    lam=np.full(n_cells, float(initial_lambda), dtype=np.float64),
                )
            )
        return out
