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

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from polar_high.engine import Problem, WarmProblem

__all__ = ["CouplingEntry", "CouplingSpec", "LagrangianProblem", "LagrangianSolution"]


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
    ) -> LagrangianSolution:
        """Run the dual-subgradient loop.

        ``step / √k`` is the diminishing step on iter ``k``.
        ``initial_lambda`` is a non-zero seed (breaks trivial 0-flow
        equilibria).  ``min_iters`` floors the iteration count so the
        early-termination test can't fire on iter 1.  ``primal_tail``
        defaults to ``max(20, max_iters//4)``.
        """
        # Initial solve — also builds each WarmProblem's HiGHS state.
        first_obj: list[float] = []
        for i, wp in enumerate(self._warm):
            sol = wp.solve()
            if not sol.optimal:
                raise RuntimeError(
                    f"LagrangianProblem: initial solve for subproblem {i} did not reach optimality"
                )
            first_obj.append(sol.obj)

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

            # Solve every subproblem.
            iter_obj = [0.0] * len(self._warm)
            primal_by_sp: dict[int, np.ndarray] = {}
            for i, wp in enumerate(self._warm):
                sol = wp.solve()
                if not sol.optimal:
                    raise RuntimeError(
                        f"LagrangianProblem iter {it}: subproblem {i} did not reach optimality"
                    )
                iter_obj[i] = sol.obj
                primal_by_sp[i] = sol.col_value

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

            last_obj = iter_obj
            if max_abs_res < tol and it >= min_iters:
                converged = True
                break

            for rc in self._resolved:
                rc.lam = rc.lam + alpha_k * rc.last_residual

        # Primal recovery: tail-average then fix-and-resolve.
        recovery_obj = list(last_obj)
        primal_recovery: list[np.ndarray] = []
        best_dual_total = max((log["total_obj"] for log in iteration_log), default=sum(first_obj))

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

            for i, wp in enumerate(self._warm):
                sol = wp.solve()
                if sol.optimal:
                    recovery_obj[i] = sol.obj
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
