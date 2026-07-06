"""Generic multicut Benders decomposition coordinator.

The coordinator owns the *loop* of a multicut Benders scheme — the bootstrap
subsolve pass, cut bookkeeping, recourse-floor sizing, lower/upper bound
tracking with the monotonicity and sandwich self-checks, gap/convergence, and
the parallel subproblem fan-out.  Everything problem-specific lives behind two
small adapter surfaces the caller implements:

* :class:`BendersMaster` — the master problem.  The coordinator never touches
  a master column directly; it asks the adapter to ``solve``, ``read_point``,
  ``project_point`` (feasibility projection of the coupling point),
  ``add_cut`` / ``relax_recourse`` / ``set_recourse_floor``.
* :class:`BendersSubproblem` — one recourse subproblem.  Its single method
  ``solve_at(point)`` owns *pin + solve* end to end: any pin-value transform
  (e.g. a scaling plan), the column fixing, the solve (with whatever retry
  policy the domain wants), the domain's own not-optimal diagnostics, any
  post-solve unscaling, and the aggregation of cut slopes keyed back to
  MASTER column ids.  The coordinator never pins or solves a subproblem's
  columns itself — it only hands ``solve_at`` a point and consumes the
  returned :class:`SubproblemResult`.

Scaled-space convention: the coordinator works ENTIRELY in the caller's
scale.  Objectives, bounds, cut constants and every reported/callback value
are whatever scale the adapters produce; any unscaling for presentation is
the caller's job at its own boundary.

Coupling column universe: the keys of ``initial_point`` define the full set
of master coupling column ids for the whole run.  This is domain knowledge
the protocols cannot derive (a point can only be read off the master AFTER a
master solve, but the bootstrap subsolve pass happens before one), so the
caller supplies it.  Per-subproblem column ownership is the key set of that
subproblem's :class:`SubproblemResult.slopes` — by protocol requirement it
carries a key for EVERY master column the subproblem is pinned on, zero
slopes included.

Preconditions the caller owns:

* Every subproblem's ``warm`` handle must already be BUILT (its ``solve()``
  called once, sequentially) before :func:`solve_benders_loop` — the parallel
  fan-out only re-solves warm models (see :mod:`polar_high.parallel`).
* The master must be constructed with a provisional (finite) recourse floor
  so its cut-less initial state is bounded; the coordinator replaces it with
  the tight bootstrap-sized floor via ``set_recourse_floor`` before the first
  master solve.

Not yet wired (lands in the follow-up commit; the corresponding options /
hooks below are documented inert and REJECTED loudly where they would change
behavior): in-out stabilization (``in_out_weight > 0``), the stall guard
(``stall_window`` / ``gap_floor`` / ``extra_reference_cost`` /
:class:`BendersStalled`), and periodic cut compaction (``compact_at`` /
``cut_policy`` / ``cut_window`` / ``BendersMaster.compact_cuts``).  At their
defaults the loop below is the exact (λ=0) Benders path, and the follow-up
only adds code inside blocks gated on those defaults — the λ=0 trajectory is
unchanged by it.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from polar_high.engine import Solution, WarmProblem
from polar_high.parallel import resolve_worker_count, solve_indexed_parallel

__all__ = [
    "BendersBoundInvalid",
    "BendersLoopOptions",
    "BendersLoopResult",
    "BendersMaster",
    "BendersStalled",
    "BendersSubproblem",
    "SubproblemHandle",
    "SubproblemNotOptimal",
    "SubproblemResult",
    "solve_benders_loop",
]

_logger = logging.getLogger(__name__)

# Optional monolith guard slack: the lower bound may exceed a caller-supplied
# known optimum by at most this relative slack before the bound is declared
# invalid (pure-rounding headroom on an exact comparison).
_LB_VALID_SLACK = 1e-9
# Gross band for the fail-safe bound self-checks (LB drop / sandwich): a
# relative violation within ``max(tol, _LB_GROSS_SLACK)`` is absorbed as
# numerical noise; beyond it the bound sequence is declared invalid.
_LB_GROSS_SLACK = 1e-3


# ---------------------------------------------------------------------------
# Structured exceptions.
# ---------------------------------------------------------------------------


class SubproblemNotOptimal(RuntimeError):
    """A subproblem (or master) solve failed to certify optimality.

    Raised DOMAIN-SIDE — inside an adapter's ``solve_at`` (or
    ``BendersMaster.solve``) — never by the coordinator itself; the
    coordinator propagates it untouched (including out of the parallel
    fan-out, in subproblem index order).  Provided here so adapters share one
    structured type the driving application can catch and render with its own
    diagnostics.
    """

    def __init__(self, sub_name: str, *, status: object | None = None, message: str | None = None):
        self.sub_name = sub_name
        self.status = status
        super().__init__(
            message
            if message is not None
            else f"benders: subproblem {sub_name!r} did not solve to optimality"
            + (f" (status {status!r})" if status is not None else "")
        )


class BendersBoundInvalid(RuntimeError):
    """The Benders bound sequence is invalid (beyond numerical noise).

    ``kind`` identifies the failed self-check:

    * ``"lb_drop"`` — the lower bound dropped by more than the gross band
      (cuts only tighten; a gross drop means an inconsistent master solve).
      Fields: ``lower_bound``, ``prev_lower_bound``, ``rel_drop``,
      ``gross_band``.
    * ``"sandwich"`` — the lower bound rose above the best known feasible
      cost by more than the gross band (an invalid bound).  Fields:
      ``lower_bound``, ``best_upper_bound``, ``rel_over``, ``gross_band``.
    * ``"cut_violated"`` — a just-appended cut is grossly violated at the new
      master point (a cut that failed to append, or a grossly infeasible
      master point).  Fields: ``sub_name``, ``recourse_value``, ``cut_rhs``,
      ``violation``, ``gross_tol``, ``row_scale``.
    * ``"cut_nonfinite"`` — the master returned a non-finite recourse value
      for a subproblem.  Fields: ``sub_name``, ``recourse_value``.
    * ``"monolith"`` — the OPTIONAL test-time guard: the lower bound exceeds
      the caller-supplied known monolith optimum
      (``BendersLoopOptions.monolith_objective``, caller's scale).  Fields:
      ``lower_bound``, ``monolith_objective``.

    All numeric fields are in the caller's scale.  Fields not applicable to a
    ``kind`` are ``None``.
    """

    def __init__(
        self,
        kind: str,
        message: str,
        *,
        iteration: int,
        lower_bound: float | None = None,
        prev_lower_bound: float | None = None,
        best_upper_bound: float | None = None,
        rel_drop: float | None = None,
        rel_over: float | None = None,
        gross_band: float | None = None,
        sub_name: str | None = None,
        recourse_value: float | None = None,
        cut_rhs: float | None = None,
        violation: float | None = None,
        gross_tol: float | None = None,
        row_scale: float | None = None,
        monolith_objective: float | None = None,
    ):
        self.kind = kind
        self.iteration = iteration
        self.lower_bound = lower_bound
        self.prev_lower_bound = prev_lower_bound
        self.best_upper_bound = best_upper_bound
        self.rel_drop = rel_drop
        self.rel_over = rel_over
        self.gross_band = gross_band
        self.sub_name = sub_name
        self.recourse_value = recourse_value
        self.cut_rhs = cut_rhs
        self.violation = violation
        self.gross_tol = gross_tol
        self.row_scale = row_scale
        self.monolith_objective = monolith_objective
        super().__init__(message)


class BendersStalled(RuntimeError):
    """The loop stalled: the incumbent froze far above any sane objective
    magnitude while the gap stayed far from tolerance.

    NOT YET RAISED — the stall guard is wired in the follow-up commit; the
    type is published now so the exception surface (and callers' ``except``
    clauses) is stable across that commit.  All numeric fields are in the
    caller's scale.
    """

    def __init__(
        self,
        message: str,
        *,
        iteration: int,
        gap: float,
        tol: float,
        window: int,
        reference_scale: float,
        sub_costs: dict[str, float],
        sub_reference_costs: dict[str, float],
    ):
        self.iteration = iteration
        self.gap = gap
        self.tol = tol
        self.window = window
        self.reference_scale = reference_scale
        self.sub_costs = sub_costs
        self.sub_reference_costs = sub_reference_costs
        super().__init__(message)


# ---------------------------------------------------------------------------
# Adapter surfaces.
# ---------------------------------------------------------------------------


@dataclass
class SubproblemResult:
    """One subproblem solve's contribution to the loop.

    All values are in the caller's scale.
    """

    #: Subproblem objective at the pinned point.
    cost: float
    #: ``∂cost/∂point`` keyed by MASTER column id.  PROTOCOL REQUIREMENT:
    #: carries a key for EVERY master column this subproblem is pinned on
    #: (0.0 slopes included) — the key set doubles as the subproblem's
    #: column-ownership map for the incumbent-point overlay.
    slopes: dict[int, float]
    #: Opaque domain data (e.g. the subproblem's recovered primal for a
    #: downstream handoff).  The coordinator never inspects it; it is handed
    #: to ``on_incumbent`` verbatim.
    payload: object | None = None


class BendersSubproblem(Protocol):
    """One recourse subproblem, owning its pin+solve end to end."""

    #: Stable identifier — keys the cuts, recourse values and costs.
    name: str
    #: Built solver handle.  Used ONLY for the parallel fan-out's
    #: built-precondition check; the coordinator never pins/solves it
    #: directly.
    warm: WarmProblem

    def solve_at(self, point: dict[int, float]) -> SubproblemResult:
        """Solve this subproblem with its coupling columns pinned at
        ``point`` (``{master col id -> value}``, master space).

        The adapter owns: any pin-value transform, the column fixing, the
        solve (with the domain's retry policy), the domain's not-optimal
        error (e.g. :class:`SubproblemNotOptimal`), any post-solve
        unscaling, and slope aggregation keyed back to master column ids.
        Must be safe to call from a worker thread (mutates only its own
        solver handle / state).
        """
        ...


@dataclass
class SubproblemHandle:
    """Plain-data :class:`BendersSubproblem`: a name, a built
    :class:`~polar_high.engine.WarmProblem` and a ``solve_at`` callable —
    for callers that do not want to write an adapter class."""

    name: str
    warm: WarmProblem
    solve_at_fn: Callable[[dict[int, float]], SubproblemResult]

    def solve_at(self, point: dict[int, float]) -> SubproblemResult:
        return self.solve_at_fn(point)


class BendersMaster(Protocol):
    """The master problem adapter."""

    def solve(self) -> Solution:
        """Solve the master (warm) and return the solution.  The adapter
        raises its OWN structured/diagnostic error on a non-optimal solve —
        a not-optimal master never reaches the coordinator."""
        ...

    def read_point(self, sol: Solution) -> tuple[dict[int, float], dict[str, float]]:
        """Return ``(coupling point by master col id, recourse value by
        subproblem name)`` at the given master solution."""
        ...

    def native_cost(self, sol: Solution, recourse: dict[str, float]) -> float:
        """The master's OWN cost at ``sol`` — its objective minus the
        recourse terms (``obj − Σ recourse``)."""
        ...

    def project_point(self, f: dict[int, float], sol: Solution, *, hard_fail: bool = True) -> float:
        """Project the coupling point ``f`` onto the master's feasible set
        IN PLACE (e.g. clamp values down to invested capacity) and return the
        maximum projection slack.  With ``hard_fail=True`` a GROSS violation
        raises the adapter's own diagnostic error (a real inconsistency, not
        solver rounding); ``hard_fail=False`` projects silently (used for
        interior separation points that can legitimately exceed the current
        feasible set)."""
        ...

    def add_cut(
        self,
        sub_name: str,
        gen_point: dict[int, float],
        cost: float,
        slopes: dict[int, float],
    ) -> None:
        """Append the optimality cut for ``sub_name`` generated at
        ``gen_point``: ``recourse >= cost + Σ slopes·(f − gen_point)``."""
        ...

    def relax_recourse(self, sub_name: str) -> None:
        """Relax ``sub_name``'s recourse column to free (lower −inf) once it
        has at least one cut — the cut now bounds it from below."""
        ...

    def set_recourse_floor(self, floor: float) -> None:
        """Set the lower bound of every not-yet-relaxed recourse column."""
        ...

    def compact_cuts(self, sol: Solution, *, policy: str, trial_col_values: list) -> dict:
        """OPTIONAL member — periodic cut-pool compaction (see
        :meth:`polar_high.engine.WarmProblem.compact_cuts`).  Not called by
        the coordinator yet (compaction is wired in the follow-up commit,
        gated on ``compact_at > 0``); implementations that never enable
        compaction may omit it."""
        ...


# ---------------------------------------------------------------------------
# Options / result.
# ---------------------------------------------------------------------------


@dataclass
class BendersLoopOptions:
    """Knobs for :func:`solve_benders_loop`.

    All cost-valued options are in the caller's scale.  Options marked
    *(inert)* belong to features wired in the follow-up commit; at their
    defaults they change nothing, and behavior-changing non-defaults are
    rejected loudly (``NotImplementedError``) rather than silently ignored.
    """

    #: Iteration cap.
    max_iters: int
    #: Relative gap tolerance ``(best_UB − LB)/max(1, |best_UB|)``.
    tol: float
    #: (inert) In-out separation weight λ; ``0.0`` = exact Benders.
    in_out_weight: float = 0.0
    #: Worker-thread count for the subproblem pass; ``None`` auto-resolves
    #: (see :func:`polar_high.parallel.resolve_worker_count`).
    workers: int | None = None
    #: (inert) Stall-guard trailing window.
    stall_window: int = 8
    #: (inert) Stall-guard gap floor; ``None`` derives ``max(20·tol, 0.02)``.
    gap_floor: float | None = None
    #: (inert) Cut-compaction trigger (active cut rows); ``0`` = off.
    compact_at: int = 0
    #: (inert) Cut-compaction selection policy.
    cut_policy: str = "slack"
    #: (inert) Trial-point window length for the dominance policy.
    cut_window: int = 5
    #: Bootstrap recourse-floor multiplier: the post-bootstrap floor is
    #: ``−eta_floor_mult · max(max_s |cost_s^bootstrap|, obj_scale)``.
    eta_floor_mult: float = 1.1
    #: The caller's objective scale — used ONLY as the degenerate-case guard
    #: in the recourse-floor sizing above (the coordinator itself never
    #: rescales anything).
    obj_scale: float = 1.0
    #: Gross band for the fail-safe bound self-checks (relative).
    lb_gross_slack: float = _LB_GROSS_SLACK
    #: OPTIONAL test-time guard: a known monolith optimum in the CALLER'S
    #: scale; the loop then asserts ``LB ≤ monolith_objective·(1+1e-9)``
    #: every iteration and raises ``BendersBoundInvalid("monolith")`` on
    #: failure.  ``None`` skips the guard.
    monolith_objective: float | None = None


@dataclass
class BendersLoopResult:
    """Outcome of :func:`solve_benders_loop` — everything at the incumbent,
    in the caller's scale."""

    converged: bool
    iterations: int
    lower_bound: float
    best_upper_bound: float
    gap: float
    #: Subproblem costs at the incumbent.
    sub_costs: dict[str, float] = field(default_factory=dict)
    #: Master coupling point at the incumbent.
    incumbent_point: dict[int, float] = field(default_factory=dict)
    #: Whatever ``on_incumbent`` returned at the incumbent (``None`` when no
    #: hook was given or no incumbent was recorded).
    incumbent_payload: object | None = None


# ---------------------------------------------------------------------------
# Cut self-check helpers.
# ---------------------------------------------------------------------------


def _cut_separates(
    cost: float,
    slopes: dict[int, float],
    f_out: dict[int, float],
    f_sep: dict[int, float],
    recourse_value: float,
) -> bool:
    """In-out separation test: does the cut GENERATED at the interior
    ``f_sep`` strictly separate the master vertex ``(f_out, recourse)``?

    The cut's value at the master point is ``cut_val = cost + Σ
    slope·(f_out − f_sep)``; it lower-bounds the subproblem recourse, so it
    separates iff the master under-estimated it: ``cut_val > recourse +
    tol_sep``.  ``tol_sep`` is the SAME row-scale tolerance
    :func:`_check_cuts_satisfied` uses — ``1e-6·max(1, |cut_val|,
    |recourse|, row_scale)`` with ``row_scale = |cost| + Σ|slope|·(|f_out| +
    |f_sep|)`` — reusing the identical arithmetic so the two never drift.
    The tolerance is LOAD-BEARING: a bare ``>`` reports spurious
    "separated" on round-off, the forced out-step never fires, and the loop
    livelocks on a degenerate vertex.  All quantities are homogeneous in the
    caller's scale.  (Consumed by the in-out wiring of the follow-up commit;
    published alongside :func:`_check_cuts_satisfied` so the two tolerance
    formulas live and change together.)
    """
    cut_val = cost + sum(g * (f_out[c] - f_sep[c]) for c, g in slopes.items())
    row_scale = abs(cost) + sum(abs(g) * (abs(f_out[c]) + abs(f_sep[c])) for c, g in slopes.items())
    tol_sep = 1e-6 * max(1.0, abs(cut_val), abs(recourse_value), row_scale)
    return cut_val > recourse_value + tol_sep


def _check_cuts_satisfied(
    cuts: list[tuple[str, dict[int, float], float, dict[int, float]]],
    new_point: dict[int, float],
    recourse_by_sub: dict[str, float],
    *,
    iteration: int,
) -> None:
    """Mandatory self-check: at the NEW master point each just-appended cut
    must be SATISFIED, i.e. ::

        recourse_s  >=  cost_s + Σ_col slope[col]·(new_point[col] − gen_point[col])

    Each cut is a ``(sub_name, gen_point, cost, slopes)`` 4-tuple carrying
    its OWN generation point.  We assert each recourse value is finite AND
    clears its own cut RHS — a binding cut makes this an equality, a slack
    cut an inequality.

    TOLERANCE SCALE.  The cut is a literal row ``recourse − Σ g·f ≥ cost −
    Σ g·ḡ`` in the master LP, so at the master optimum the solver already
    enforces it — the ONLY gap between row-as-solved and row-as-checked is
    the LP's feasibility tolerance.  Solvers measure that on the
    INTERNALLY-SCALED matrix, so the unscaled slack tracks the row's
    COEFFICIENT magnitude, not its (possibly heavily cancelled) rhs: when
    the subproblem cost and ``Σ g·ḡ`` are large and nearly cancel, the rhs
    collapses while the row coefficients stay huge, and a tolerance keyed
    off ``|rhs|`` alone hard-fails on pure solver round-off.  The tolerance
    is therefore keyed off the row magnitude (``|cost| + Σ|g·f|``) — absorb
    numerical noise, hard-fail only a GROSS violation (a cut that failed to
    append, or a grossly infeasible master point).
    """
    for sub_name, gen_point, cost, slopes in cuts:
        er = recourse_by_sub.get(sub_name)
        if er is None or not np.isfinite(er):
            raise BendersBoundInvalid(
                "cut_nonfinite",
                f"benders: recourse estimate for subproblem {sub_name!r} is not "
                f"a finite number ({er!r}) after the master solve at iteration "
                f"{iteration}",
                iteration=iteration,
                sub_name=sub_name,
                recourse_value=er,
            )
        rhs = cost + sum(g * (new_point[c] - gen_point[c]) for c, g in slopes.items())
        # Row conditioning: the constraint's coefficients (|cost|, |g·f|) set
        # the unscaled feasibility slack the solver may leave — they can be
        # ORDERS larger than the cancelled rhs.  Tolerance keyed off that
        # scale.
        row_scale = abs(cost) + sum(
            abs(g) * (abs(new_point[c]) + abs(gen_point[c])) for c, g in slopes.items()
        )
        tol_abs = 1e-6 * max(1.0, abs(rhs), abs(er), row_scale)
        if er >= rhs - tol_abs:
            continue
        violation = rhs - er
        # GROSS band: an un-appended cut leaves the recourse near its
        # large-negative floor, and a grossly infeasible master point
        # violates by a like amount — either dwarfs 1% of the row scale.
        # Below that, a violation is solver feasibility round-off on an
        # ill-conditioned row; warn and continue (the LB-monotonicity +
        # sandwich guards still bracket the optimum).
        gross_tol = 1e-2 * max(1.0, row_scale)
        if violation > gross_tol:
            raise BendersBoundInvalid(
                "cut_violated",
                f"benders: cut for subproblem {sub_name!r} is violated at the "
                f"new master point: recourse estimate {er:.6e} is below the cut "
                f"floor {rhs:.6e} (by {violation:.3e}, > {gross_tol:.3e} at row "
                f"scale {row_scale:.3e}) at iteration {iteration}",
                iteration=iteration,
                sub_name=sub_name,
                recourse_value=float(er),
                cut_rhs=rhs,
                violation=violation,
                gross_tol=gross_tol,
                row_scale=row_scale,
            )
        _logger.warning(
            "benders iter %d: cut for %r under-satisfied by %.3e (row scale "
            "%.3e) — solver feasibility round-off on an ill-conditioned cut "
            "row; continuing",
            iteration,
            sub_name,
            violation,
            row_scale,
        )


# ---------------------------------------------------------------------------
# The loop.
# ---------------------------------------------------------------------------


def solve_benders_loop(
    master: BendersMaster,
    subproblems: list[BendersSubproblem],
    *,
    options: BendersLoopOptions,
    initial_point: dict[int, float],
    extra_reference_cost: Callable[[], float] | None = None,
    on_iteration: Callable[[dict], None] | None = None,
    on_subsolve: Callable[[dict], None] | None = None,
    on_incumbent: Callable[[Solution, list[SubproblemResult], dict], object] | None = None,
) -> BendersLoopResult:
    """Run the multicut Benders loop over ``master`` + ``subproblems``.

    Parameters
    ----------
    master
        The :class:`BendersMaster` adapter.  Must be built with a
        provisional finite recourse floor (replaced post-bootstrap).
    subproblems
        The :class:`BendersSubproblem` adapters, each with a BUILT ``warm``
        handle (cold first solve already done, sequentially).  Names must be
        unique.
    options
        :class:`BendersLoopOptions`.
    initial_point
        The bootstrap point over the FULL coupling col-id universe
        (``{master col id -> value}``).  Its key set IS the coupling column
        universe for the whole run: the coordinator uses it for the
        bootstrap subsolve pass and as the bootstrap cuts' generation point
        (and, once in-out lands, as the stabilizer centre seed) — it never
        derives coupling columns from anything else.
    extra_reference_cost
        (inert) Called once post-bootstrap for an additional reference-cost
        term of the stall guard — wired with the stall guard in the
        follow-up commit; accepted now so caller signatures are stable.
    on_iteration
        Fired once per outer iteration (after that iteration's master +
        subproblem solves) with ``{"iter", "lower_bound", "upper_bound",
        "best_upper_bound", "gap", "converged", "sub_costs", "cut_rows"}``
        — all values in the caller's scale.
    on_subsolve
        Fired once per subproblem as its ``solve_at`` RETURNS, with
        ``{"iter", "sub", "cost"}`` (``iter`` 0 = bootstrap).  Runs on the
        worker thread — must be thread-safe.  Exceptions are swallowed (an
        observer must not break the solve).
    on_incumbent
        Fired when an iteration improves the best upper bound, with
        ``(master_solution, [SubproblemResult...], info)`` where ``info``
        carries ``{"iteration", "upper_bound", "point", "sub_costs"}``.
        Its return value is stored as :attr:`BendersLoopResult.incumbent_payload`.
        Domain adapters MUST materialize anything they keep (solver-backed
        buffers may be reused by later warm re-solves).

    Raises
    ------
    BendersBoundInvalid
        On a gross bound-sequence inconsistency (see the class docstring for
        the kinds).
    NotImplementedError
        When an option belonging to a not-yet-wired feature is set to a
        behavior-changing value (``in_out_weight > 0``, ``compact_at > 0``).
    """
    if options.in_out_weight != 0.0:
        raise NotImplementedError(
            "solve_benders_loop: in_out_weight > 0 (in-out stabilization) is "
            "not wired yet — it lands in the follow-up commit; use 0.0"
        )
    if options.compact_at > 0:
        raise NotImplementedError(
            "solve_benders_loop: compact_at > 0 (cut compaction) is not wired "
            "yet — it lands in the follow-up commit; use 0"
        )
    if not subproblems:
        raise ValueError("solve_benders_loop: no subproblems — nothing to decompose")
    names = [s.name for s in subproblems]
    if len(set(names)) != len(names):
        raise ValueError(f"solve_benders_loop: duplicate subproblem names in {names!r}")
    if not initial_point:
        raise ValueError(
            "solve_benders_loop: initial_point is empty — its key set defines "
            "the coupling column universe, so an empty point leaves nothing "
            "to decompose on"
        )

    # --- parallel subproblem pass plumbing.  The warm handles are used ONLY
    # for the fan-out's built-precondition check; each solve_at owns its own
    # pin+solve.
    warm_list = [s.warm for s in subproblems]
    eff_workers = resolve_worker_count(len(subproblems), options.workers)
    _logger.info(
        "benders: recourse pass over %d subproblem(s) with %d worker(s)",
        len(subproblems),
        eff_workers,
    )

    # Current outer-iteration index, surfaced to ``on_subsolve`` so the
    # caller can label each subproblem-finish event (bootstrap = 0).
    cur_iter = [0]

    def _solve_subs(point: dict[int, float]) -> list[SubproblemResult]:
        """Solve every subproblem at ``point`` and return the results in
        deterministic subproblem-index order (parallel when
        ``eff_workers > 1``).  Fires ``on_subsolve`` once per subproblem as
        it FINISHES (from the worker thread; the callback must be
        thread-safe)."""

        def _fn(i: int) -> SubproblemResult:
            res = subproblems[i].solve_at(point)
            if on_subsolve is not None:
                try:
                    on_subsolve({"iter": cur_iter[0], "sub": subproblems[i].name, "cost": res.cost})
                except Exception:  # noqa: BLE001 — observer must not break the solve
                    pass
            return res

        return solve_indexed_parallel(warm_list, _fn, workers=eff_workers)

    # --- BOOTSTRAP: solve every subproblem at the caller-supplied initial
    # point to (a) generate the first cuts and (b) size the TIGHT recourse
    # floor = −eta_floor_mult·max_s|cost_s^bootstrap| — a provably valid
    # global under-estimate when the initial point is the natural "no
    # coupling" point (coupling only relaxes a subproblem, so its minimum
    # achievable cost sits below the bootstrap cost with margin).  Each cut
    # carries its GENERATION POINT (the point the subproblem was solved at)
    # as the 2nd element: ``(sub_name, gen_point, cost, slopes)``; the
    # bootstrap generation point is a SNAPSHOT of ``initial_point`` so later
    # caller mutation cannot alias it.
    bootstrap_gen = dict(initial_point)
    bootstrap_cuts: list[tuple[str, dict[int, float], float, dict[int, float]]] = []
    for sub, res in zip(subproblems, _solve_subs(initial_point)):
        bootstrap_cuts.append((sub.name, bootstrap_gen, res.cost, res.slopes))
    cost_scale = max((abs(c) for _, _, c, _ in bootstrap_cuts), default=1.0)
    # Floor in the caller's scale; ``max(cost_scale, obj_scale)`` keeps it
    # from collapsing to ~0 in the degenerate all-zero-cost case.
    eta_floor = -options.eta_floor_mult * max(cost_scale, options.obj_scale)
    master.set_recourse_floor(eta_floor)

    best_ub = float("inf")
    best: dict | None = None
    lb = float("-inf")
    prev_lb = float("-inf")
    iterations = 0
    converged = False
    gap = float("inf")

    # ``pending_cuts`` are the cuts for the subproblems solved at the CURRENT
    # point; they are appended at the top of each iteration before the master
    # solve.  Iteration 1 uses the bootstrap cuts.
    pending_cuts = bootstrap_cuts
    point: dict[int, float] = initial_point
    sub_costs: dict[str, float] = {}
    # DIAGNOSTIC: running total of cut rows appended to the master, surfaced
    # in the per-iteration timing line so the master-solve cost can be read
    # against the accumulated row count.
    cut_rows = 0

    for it in range(options.max_iters):
        iterations = it + 1

        # --- append the pending cuts and (warm) re-solve the master.  Each
        # subproblem that contributes a cut has its recourse relaxed to free
        # (−inf): the cut now bounds it from below, so the bootstrap floor is
        # no longer needed and dropping it tightens the master + narrows the
        # bound range.  Each cut is added against its OWN generation point so
        # the cut constant is computed at the point it was generated at.
        for sub_name, gen_point, cost, slopes in pending_cuts:
            master.add_cut(sub_name, gen_point, cost, slopes)
            master.relax_recourse(sub_name)
            cut_rows += 1
        t_master = time.perf_counter()
        msol = master.solve()
        dt_master = time.perf_counter() - t_master
        prev_lb = lb
        lb = float(msol.obj)
        # LB monotone non-decreasing self-check.  In exact arithmetic the
        # bound can only rise (cuts only tighten), so any drop is numerical.
        # FAIL-SAFE: a drop within the gross band is treated as noise — pin
        # LB back to the (valid) previous bound and continue; only a GROSS
        # drop signals a stale basis / corrupted cut append and hard-fails.
        gross_band = max(options.tol, options.lb_gross_slack)
        if it > 0 and lb < prev_lb:
            rel_drop = (prev_lb - lb) / max(1.0, abs(prev_lb))
            if rel_drop > gross_band:
                raise BendersBoundInvalid(
                    "lb_drop",
                    f"benders: lower bound dropped {prev_lb:.6e} -> {lb:.6e} "
                    f"(by {rel_drop:.2e}, > {gross_band:.0e}) at iteration "
                    f"{iterations} — cuts only tighten, so a gross drop means "
                    f"an inconsistent master solve (stale basis / severe "
                    f"ill-conditioning)",
                    iteration=iterations,
                    lower_bound=lb,
                    prev_lower_bound=prev_lb,
                    rel_drop=rel_drop,
                    gross_band=gross_band,
                )
            if rel_drop > 1e-6:
                _logger.warning(
                    "benders iter %d: lower bound dipped %.3e (numerical) — "
                    "pinned to previous lower bound",
                    iterations,
                    rel_drop,
                )
            lb = prev_lb  # restore monotonicity (prev_lb is a valid bound)
        # OPTIONAL test-time guard: LB ≤ known monolith optimum (both in the
        # caller's scale).  Skipped when ``monolith_objective`` is None.
        if options.monolith_objective is not None and lb > options.monolith_objective * (
            1 + _LB_VALID_SLACK
        ):
            raise BendersBoundInvalid(
                "monolith",
                f"benders: lower bound {lb:.10e} exceeds the known monolith "
                f"optimum {options.monolith_objective:.10e} at iteration "
                f"{iterations} — INVALID lower bound",
                iteration=iterations,
                lower_bound=lb,
                monolith_objective=options.monolith_objective,
            )

        new_point, recourse_by_sub = master.read_point(msol)
        _check_cuts_satisfied(pending_cuts, new_point, recourse_by_sub, iteration=iterations)

        # The master's chosen coupling point must be feasible for the master
        # itself (e.g. within the capacity it invested in).  The solver returns a
        # vertex only within its feasibility tolerance, so the adapter
        # PROJECTS the point onto the feasible set (in place) — any upper
        # bound evaluated at the projected point is then a valid
        # whole-problem bound.  A gross violation hard-fails inside the
        # adapter (``hard_fail=True``).
        solver_feas = msol.max_primal_infeasibility
        max_clamp = master.project_point(new_point, msol, hard_fail=True)
        if max_clamp > max(1e-9, solver_feas):
            _logger.debug(
                "benders iter %d: projected master coupling point to "
                "feasibility (max slack %.3e, solver feasibility %.3e)",
                iterations,
                max_clamp,
                solver_feas,
            )

        # --- advance the point to the (projected) master optimum and solve
        # the subproblems there: (a) this iteration's recourse cost gives a
        # VALID upper bound (the point is feasible for the master), and (b)
        # the solves produce the next iteration's cuts.
        point = new_point
        cur_iter[0] = iterations  # label this pass's subproblem-finish events

        sub_costs = {}
        next_cuts: list[tuple[str, dict[int, float], float, dict[int, float]]] = []
        t_subs = time.perf_counter()
        sub_results = _solve_subs(point)
        dt_subs = time.perf_counter() - t_subs
        for sub, res in zip(subproblems, sub_results):
            sub_costs[sub.name] = res.cost
            next_cuts.append((sub.name, point, res.cost, res.slopes))

        # --- UB = master native cost + Σ subproblem costs at the SAME
        # (point, master solution) — all terms in the caller's scale.
        ub = master.native_cost(msol, recourse_by_sub) + sum(sub_costs.values())
        improved = ub < best_ub
        if improved:
            best_ub = ub
            incumbent_point = dict(point)
            payload: object | None = None
            if on_incumbent is not None:
                payload = on_incumbent(
                    msol,
                    list(sub_results),
                    {
                        "iteration": iterations,
                        "upper_bound": ub,
                        "point": incumbent_point,
                        "sub_costs": dict(sub_costs),
                    },
                )
            best = {
                "point": incumbent_point,
                "sub_costs": dict(sub_costs),
                "payload": payload,
            }

        # ALWAYS-ON monolith-free sandwich guard: LB ≤ optimum ≤ best_UB must
        # hold.  FAIL-SAFE: an overshoot within the gross band means LB has
        # numerically MET best_UB — the bounds have closed, the incumbent is
        # optimal — so treat it as converged and stop on the incumbent.  A
        # GROSS overshoot is the genuine invalid-lower-bound pathology and
        # hard-fails.
        if lb > best_ub:
            rel_over = (lb - best_ub) / max(1.0, abs(best_ub))
            if rel_over > gross_band:
                raise BendersBoundInvalid(
                    "sandwich",
                    f"benders: lower bound {lb:.6e} exceeds the best feasible "
                    f"cost found {best_ub:.6e} (by {rel_over:.2e}, > "
                    f"{gross_band:.0e}) at iteration {iterations} — an invalid "
                    f"lower bound (typically severe numerical "
                    f"ill-conditioning of the master)",
                    iteration=iterations,
                    lower_bound=lb,
                    best_upper_bound=best_ub,
                    rel_over=rel_over,
                    gross_band=gross_band,
                )
            _logger.warning(
                "benders iter %d: lower bound met best upper bound within "
                "%.3e (numerical) — treating as converged",
                iterations,
                rel_over,
            )
            converged = True
            gap = 0.0
            break

        gap = (best_ub - lb) / max(1.0, abs(best_ub))
        _logger.debug(
            "benders iter %d: LB=%.6e UB=%.6e bestUB=%.6e gap=%.3e",
            iterations,
            lb,
            ub,
            best_ub,
            gap,
        )
        # DIAGNOSTIC per-iteration timing: master-solve vs subproblem-pass
        # wall time against the accumulated master cut-row count — the master
        # row count grows by one-per-subproblem-per-iteration, so this line
        # makes the O(cuts) growth of the master solve directly observable;
        # the subproblem pass is ~flat (fixed-size, parallel).
        _logger.info(
            "[benders timing] iter %d: master_solve=%.3fs subproblems=%.3fs master_cut_rows=%d",
            iterations,
            dt_master,
            dt_subs,
            cut_rows,
        )

        if on_iteration is not None:
            on_iteration(
                {
                    "iter": iterations,
                    "lower_bound": lb,
                    "upper_bound": ub,
                    "best_upper_bound": best_ub,
                    "gap": gap,
                    "converged": gap <= options.tol,
                    "sub_costs": dict(sub_costs),
                    "cut_rows": cut_rows,
                }
            )

        if gap <= options.tol:
            converged = True
            break

        pending_cuts = next_cuts

    # --- assemble the result from the incumbent (fall back to the last
    # iteration's state when no iteration improved — e.g. a zero-iteration
    # run).  Everything stays in the caller's scale.
    if best is not None:
        inc_point = best["point"]
        inc_costs = best["sub_costs"]
        inc_payload = best["payload"]
    else:
        inc_point = dict(point)
        inc_costs = dict(sub_costs)
        inc_payload = None
    return BendersLoopResult(
        converged=converged,
        iterations=iterations,
        lower_bound=lb,
        best_upper_bound=best_ub,
        gap=gap,
        sub_costs=inc_costs,
        incumbent_point=inc_point,
        incumbent_payload=inc_payload,
    )
