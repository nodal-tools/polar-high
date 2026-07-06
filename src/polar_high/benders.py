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

On top of the exact (λ=0) core loop the coordinator wires three optional,
non-default mechanisms — each lives entirely inside blocks gated on its
option, so at the defaults the loop is byte-identical to plain exact Benders:

* **In-out stabilization** (``in_out_weight > 0``; Ben-Ameur & Neto 2007):
  one :class:`~polar_high.decomposition.InOutStabilizer` PER SUBPROBLEM,
  each seeded with the ``initial_point`` centre.  Every iteration each
  subproblem is solved at its OWN interior separation point ``f_sep =
  λ·centre + (1−λ)·f_out`` — re-clamped to the current feasible set via
  ``project_point(..., hard_fail=False)`` (an interior point built from an
  OLD incumbent can legitimately exceed the current feasible set; clamping
  down is routine, not a bug signal) — and its cut is GENERATED at that
  ``f_sep``.  A per-subproblem separation test then drives the stabilizer's
  register/forced-out-step logic, and the incumbent point overlays each
  subproblem's ``f_sep`` onto ONLY the master columns it owns (ownership =
  its :class:`SubproblemResult.slopes` key set).
* **Stall guard** (:class:`~polar_high.decomposition.StallMonitor`): fed
  ``(LB, best_UB)`` each iteration; its reference scale is the sum of the
  absolute bootstrap subproblem costs plus ``|extra_reference_cost()|``
  (called ONCE post-bootstrap when provided).  A frozen-blowup stall raises
  the structured :class:`BendersStalled` — the frozen incumbent is garbage,
  so returning it would hand the caller a catastrophically wrong plan.
* **Periodic cut compaction** (``compact_at > 0``): when the accumulated
  cut-row count reaches ``compact_at``, the master adapter's OPTIONAL
  ``compact_cuts`` is invoked at the end of the iteration body with the raw
  master vertex and the trailing ``cut_window`` of master vertices; a master
  lacking the member disables compaction with a warning (clean skip).
"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from polar_high.decomposition import InOutStabilizer, StallMonitor
from polar_high.engine import Solution, WarmProblem
from polar_high.parallel import resolve_worker_count, solve_indexed_parallel

__all__ = [
    "BendersBoundInvalid",
    "BendersLoopOptions",
    "BendersLoopResult",
    "BendersMaster",
    "BendersStalled",
    "BendersSubproblem",
    "PointEvaluation",
    "SubproblemHandle",
    "SubproblemNotOptimal",
    "SubproblemResult",
    "evaluate_at_point",
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

    Raised when the :class:`~polar_high.decomposition.StallMonitor`'s
    frozen-blowup conjunction fires: the best feasible cost has not improved
    across the trailing ``window`` iterations, the relative gap is still far
    from tolerance, AND the incumbent sits far above ``reference_scale``
    (the bootstrap subproblem costs plus the caller's
    ``extra_reference_cost``).  The frozen incumbent is garbage in this
    regime (it can be orders above the true optimum), so the coordinator
    fails fast instead of silently exhausting the iteration cap.
    ``sub_costs`` are the stalled iteration's subproblem costs;
    ``sub_reference_costs`` are the bootstrap (initial-point) costs — the
    pair lets the caller name the worst offender in its own diagnostics.
    All numeric fields are in the caller's scale.
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

    def native_cost_at(self, point: dict[int, float]) -> float:
        """OPTIONAL member — the master's OWN cost with its coupling columns
        PINNED at ``point`` (``{master col id -> value}``): pin the coupling
        columns to ``point``, solve the master ONCE, and return
        ``obj − Σ recourse`` at that pinned solution (restoring the pinned
        columns' bounds afterwards).

        This is the L-shaped "evaluate the first-stage cost at an ARBITRARY
        feasible coupling point" primitive — the counterpart of
        :meth:`native_cost`, which reads the cost only at the master's OWN
        vertex.  It exists so :func:`evaluate_at_point` can score the whole
        objective ``c(x̄) + Σ_r Q_r(x̄)`` at one common point ``x̄`` even when
        the master's native cost DEPENDS on the coupling flows (a
        cost-bearing master, e.g. one hosting balance/storage nodes).  A
        master whose native cost is flow-independent, or one that simply
        does not implement this, is reported subproblem-only by
        :func:`evaluate_at_point` (``master_native_cost = None``)."""
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
        :meth:`polar_high.engine.WarmProblem.compact_cuts`).  Called only
        when ``compact_at > 0`` and the accumulated cut-row count reaches
        it, with the RAW master vertex of the just-finished iteration and
        the trailing window of master vertices (for the dominance policy).
        Must return a dict with at least ``{"kept", "dropped",
        "restored"}``.  Implementations that never enable compaction may
        omit the member — the coordinator then disables compaction with a
        warning."""
        ...


# ---------------------------------------------------------------------------
# Options / result.
# ---------------------------------------------------------------------------


@dataclass
class BendersLoopOptions:
    """Knobs for :func:`solve_benders_loop`.

    All cost-valued options are in the caller's scale.  At the defaults the
    loop is exact (λ=0) Benders with the stall guard on its library
    defaults and cut compaction off.
    """

    #: Iteration cap.
    max_iters: int
    #: Relative gap tolerance ``(best_UB − LB)/max(1, |best_UB|)``.
    tol: float
    #: In-out separation weight λ on the stable interior centre
    #: (``f_sep = λ·centre + (1−λ)·f_out``); ``0.0`` = exact Benders
    #: (byte-identical off path).  Must be in ``[0, 1)``.
    in_out_weight: float = 0.0
    #: Worker-thread count for the subproblem pass; ``None`` auto-resolves
    #: (see :func:`polar_high.parallel.resolve_worker_count`).
    workers: int | None = None
    #: Stall-guard trailing window ``K`` (iterations the incumbent must be
    #: frozen before a stall can be declared).
    stall_window: int = 8
    #: Stall-guard gap floor; ``None`` derives ``max(20·tol, 0.02)`` so a
    #: loose ``tol`` never lets the floor fall below the gap it must clear.
    gap_floor: float | None = None
    #: Cut-compaction trigger (accumulated active cut rows); ``0`` = off
    #: (byte-identical to the pre-compaction path).
    compact_at: int = 0
    #: Cut-compaction selection policy (``"slack"`` | ``"dominance"``).
    cut_policy: str = "slack"
    #: Trial-point window length for the dominance policy (trailing master
    #: vertices fed to ``compact_cuts``).
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
    caller's scale.  (Kept alongside :func:`_check_cuts_satisfied` so the two
    tolerance formulas live and change together.)
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
# Single-point evaluation (the L-shaped feasible-point primitive).
# ---------------------------------------------------------------------------


@dataclass
class PointEvaluation:
    """Outcome of :func:`evaluate_at_point` — the whole two-stage objective
    scored at ONE feasible coupling point.  All costs are in the caller's
    scale (the same convention as :class:`BendersLoopResult`)."""

    #: Subproblem cost at ``point``, keyed by subproblem name.
    sub_costs: dict[str, float]
    #: ``Σ sub_costs`` — the total recourse cost at ``point``.
    sub_cost_total: float
    #: The master's native (first-stage) cost AT ``point``
    #: (``master.native_cost_at(point)``), or ``None`` when the master does
    #: not implement the optional ``native_cost_at`` member.
    master_native_cost: float | None
    #: The whole-objective value at ``point``: ``sub_cost_total +
    #: master_native_cost`` (or just ``sub_cost_total`` when the master term
    #: is unavailable).  When the master term is present and ``point`` is
    #: master-feasible this is a genuine single-point upper bound on the
    #: two-stage optimum (Van Slyke & Wets 1969).
    total_cost: float
    #: Per-subproblem "blew up vs reference" flag: ``True`` iff the
    #: subproblem's cost at ``point`` exceeds ``blowup_mult · max(1,
    #: |reference_costs[name]|)``.  Empty when no ``reference_costs`` were
    #: supplied.
    blew_up: dict[str, bool]
    #: The reference costs the blow-up test compared against (echoed for the
    #: caller's diagnostics), or ``None``.
    reference_costs: dict[str, float] | None
    #: The multiplier used for the blow-up test.
    blowup_mult: float


def evaluate_at_point(
    master: BendersMaster,
    subproblems: list[BendersSubproblem],
    point: dict[int, float],
    *,
    reference_costs: dict[str, float] | None = None,
    blowup_mult: float = 100.0,
    workers: int | None = None,
) -> PointEvaluation:
    """Evaluate the whole two-stage objective at ONE feasible coupling
    ``point`` — the L-shaped feasible-point primitive.

    For a first-stage / coupling decision ``x̄`` the two-stage objective is
    ``c(x̄) + Σ_r Q_r(x̄)`` where ``Q_r(x̄)`` is subproblem ``r`` solved to
    optimality at ``x̄`` and ``c`` is the master's own (native) cost.  This
    helper computes exactly that at ``point``:

    #. solves every subproblem at ``point`` (``solve_at(point)``, fanned out
       over a thread pool exactly like the loop's recourse pass — each
       subproblem owns its pin+solve), giving ``Q_r(point)``;
    #. reads the master's native cost AT ``point`` via the OPTIONAL
       :meth:`BendersMaster.native_cost_at` (a master lacking it is reported
       subproblem-only, ``master_native_cost = None``);
    #. flags, per subproblem, whether its cost at ``point`` "blew up" versus
       an optional per-subproblem ``reference_costs`` baseline (cost >
       ``blowup_mult · max(1, |reference|)``).

    Unlike :func:`solve_benders_loop` this runs NO iterations, appends NO
    cuts, and mutates no bound outside the master's own
    ``native_cost_at`` save/restore — it is a pure read-out at a single
    point, the building block a go/no-go pin diagnostic (and later
    consistent-point UB work) is composed from.  ``point`` must be
    master-feasible for ``total_cost`` to be a valid upper bound; the caller
    owns that (e.g. project it first, or supply a known-feasible point such
    as a monolith optimum).

    Parameters
    ----------
    master
        The :class:`BendersMaster` adapter.  Only its OPTIONAL
        ``native_cost_at`` is used (never ``solve`` / ``add_cut`` / …), so
        the master's cut pool and warm state are untouched.
    subproblems
        The :class:`BendersSubproblem` adapters, each with a BUILT ``warm``
        handle (same precondition as the loop).  Names must be unique.
    point
        ``{master col id -> value}`` over the coupling column universe every
        subproblem is pinned on (and every column ``native_cost_at`` pins).
    reference_costs
        Optional per-subproblem baseline (e.g. the stand-alone / zero-
        coupling cost) for the blow-up flag.  ``None`` leaves ``blew_up``
        empty.
    blowup_mult
        Blow-up threshold multiplier (default 100×).
    workers
        Worker-thread count for the subproblem pass; ``None`` auto-resolves
        (see :func:`polar_high.parallel.resolve_worker_count`).

    Returns
    -------
    PointEvaluation
    """
    if not subproblems:
        raise ValueError("evaluate_at_point: no subproblems — nothing to evaluate")
    names = [s.name for s in subproblems]
    if len(set(names)) != len(names):
        raise ValueError(f"evaluate_at_point: duplicate subproblem names in {names!r}")
    if not point:
        raise ValueError("evaluate_at_point: point is empty — nothing to pin")

    warm_list = [s.warm for s in subproblems]
    eff_workers = resolve_worker_count(len(subproblems), workers)

    def _fn(i: int) -> SubproblemResult:
        return subproblems[i].solve_at(point)

    results = solve_indexed_parallel(warm_list, _fn, workers=eff_workers)
    sub_costs = {sub.name: float(res.cost) for sub, res in zip(subproblems, results)}
    sub_cost_total = float(sum(sub_costs.values()))

    native_cost_at = getattr(master, "native_cost_at", None)
    master_native_cost: float | None
    if callable(native_cost_at):
        master_native_cost = float(native_cost_at(point))
    else:
        master_native_cost = None

    total_cost = sub_cost_total + (master_native_cost or 0.0)

    blew_up: dict[str, bool] = {}
    if reference_costs is not None:
        for name, cost in sub_costs.items():
            ref = reference_costs.get(name)
            if ref is None:
                continue
            threshold = blowup_mult * max(1.0, abs(float(ref)))
            blew_up[name] = cost > threshold

    return PointEvaluation(
        sub_costs=sub_costs,
        sub_cost_total=sub_cost_total,
        master_native_cost=master_native_cost,
        total_cost=total_cost,
        blew_up=blew_up,
        reference_costs=dict(reference_costs) if reference_costs is not None else None,
        blowup_mult=float(blowup_mult),
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
        bootstrap subsolve pass, as the bootstrap cuts' generation point and
        as the in-out stabilizer centre seed — it never derives coupling
        columns from anything else.
    extra_reference_cost
        Called ONCE post-bootstrap; its absolute value is added to the
        stall guard's reference scale (the sum of the absolute bootstrap
        subproblem costs) — e.g. the master's own stand-alone cost at the
        no-coupling point.  ``None`` adds nothing.
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
    BendersStalled
        When the stall guard's frozen-blowup conjunction fires (see the
        class docstring).
    """
    if not (0.0 <= options.in_out_weight < 1.0):
        raise ValueError(
            "solve_benders_loop: in_out_weight must be in [0, 1): "
            f"got {options.in_out_weight!r} (>= 1 never queries the master ⇒ "
            "non-convergent; < 0 is meaningless)"
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

    def _solve_subs(point_or_fn) -> list[SubproblemResult]:
        """Solve every subproblem and return the results in deterministic
        subproblem-index order (parallel when ``eff_workers > 1``).

        ``point_or_fn`` is EITHER a single ``{master col id -> value}`` dict
        — every subproblem is pinned at the same point (the exact-Benders /
        bootstrap path) — OR a callable ``i -> point`` returning subproblem
        ``i``'s OWN pin point (the in-out path, where each subproblem is
        pinned at its own interior ``f_sep``; a shared master column may
        carry different per-subproblem interior values, so a per-subproblem
        callable — not one merged dict — is the correct interface).  Fires
        ``on_subsolve`` once per subproblem as it FINISHES (from the worker
        thread; the callback must be thread-safe)."""
        per_sub = callable(point_or_fn)

        def _fn(i: int) -> SubproblemResult:
            pin = point_or_fn(i) if per_sub else point_or_fn
            res = subproblems[i].solve_at(pin)
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
    # Per-subproblem STAND-ALONE (initial-point) cost, keyed by name.  Its
    # absolute sum is a "sane objective magnitude" reference for the stall
    # guard, and the per-subproblem values ride the BendersStalled exception
    # so the caller can name the worst offender in its own diagnostics.
    sub_reference_costs = {name: cost for name, _, cost, _ in bootstrap_cuts}
    # Optional extra reference term (e.g. the master's own stand-alone cost
    # at the no-coupling point) — called ONCE, post-bootstrap.
    extra_ref = abs(float(extra_reference_cost())) if extra_reference_cost is not None else 0.0
    # Domain-free stall detector.  reference_scale = Σ|bootstrap cost| +
    # |extra|; the gap floor is raised to max(20·tol, 0.02) when unset so a
    # loose ``tol`` never lets the floor fall below the gap it must clear.
    stall_monitor = StallMonitor(
        sum(abs(c) for c in sub_reference_costs.values()) + extra_ref,
        window=options.stall_window,
        gap_floor=(
            options.gap_floor if options.gap_floor is not None else max(20.0 * options.tol, 0.02)
        ),
    )
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

    # --- In-out separation (Ben-Ameur & Neto 2007) — PER-SUBPROBLEM
    # stabilizer.  ``λ`` is the weight on the stable interior centre in
    # ``f_sep = λ·centre + (1−λ)·f_out``.  One stabilizer PER SUBPROBLEM (the
    # correct unit — a global λ lets a well-behaved subproblem mask a
    # degenerate one), each seeded with the caller's ``initial_point`` centre
    # (the natural no-coupling point, feasible against any projection, and
    # matching the loop bootstrap).  When ``λ == 0.0`` every in-out block
    # below is skipped and the loop is byte-identical to exact Benders.
    in_out_weight = options.in_out_weight
    in_out_on = in_out_weight > 0.0
    if in_out_on:
        _logger.info(
            "benders: in-out separation ON (weight λ=%.3f) over %d subproblem(s)",
            in_out_weight,
            len(subproblems),
        )
    stabilizers: dict[str, InOutStabilizer] = {}
    if in_out_on:
        for sub in subproblems:
            stab = InOutStabilizer(weight=in_out_weight)
            stab.set_centre(initial_point)
            stabilizers[sub.name] = stab

    # ``pending_cuts`` are the cuts for the subproblems solved at the CURRENT
    # point; they are appended at the top of each iteration before the master
    # solve.  Iteration 1 uses the bootstrap cuts.  Each entry carries its
    # own generation point (= the loop point with in-out OFF; = the interior
    # ``f_sep`` with in-out ON).
    pending_cuts = bootstrap_cuts
    point: dict[int, float] = initial_point
    sub_costs: dict[str, float] = {}
    # DIAGNOSTIC: running total of cut rows appended to the master, surfaced
    # in the per-iteration timing line so the master-solve cost can be read
    # against the accumulated row count.  With cut compaction ON it is reset
    # to the KEPT (binding) count at each compaction.
    cut_rows = 0
    # Periodic MASTER CUT COMPACTION threshold (0 = OFF = byte-identical to
    # the pre-compaction path — the whole compaction call at the bottom of
    # the loop body is guarded by ``compact_at > 0``).  Capability guard: a
    # master adapter without the OPTIONAL ``compact_cuts`` member disables
    # compaction with a clear one-time message instead of crashing mid-solve
    # with an ``AttributeError``; the run then proceeds exactly like the
    # default (OFF) path.
    compact_at = options.compact_at
    if compact_at > 0 and not callable(getattr(master, "compact_cuts", None)):
        _logger.warning(
            "benders: cut compaction was requested (compact_at=%d) but the "
            "master adapter has no compact_cuts member; continuing without "
            "compaction.",
            compact_at,
        )
        compact_at = 0
    # Bounded trailing window of recent master vertices (``msol.col_value``,
    # most-recent last) feeding the ``compact_cuts(policy="dominance")``
    # selection.  Only consulted when compaction is ON.
    cut_window: deque = deque(maxlen=max(1, options.cut_window))

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
        # Record this master vertex in the dominance-policy trial-point
        # window (most-recent last; bounded by the deque ``maxlen``).  Only
        # consulted when compaction is ON.
        cut_window.append(msol.col_value)
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

        # --- advance the point to the (projected) master optimum ``f_out``
        # (used for the LB/recourse bookkeeping and, with in-out OFF, the
        # subproblem pin) and solve the subproblems: (a) this iteration's
        # recourse cost gives a VALID upper bound (the pin point is feasible
        # for the master), and (b) the solves produce the next iteration's
        # cuts.
        point = new_point
        cur_iter[0] = iterations  # label this pass's subproblem-finish events

        # --- IN-OUT SEPARATION.  With ``λ>0`` each subproblem is solved at
        # its OWN interior separation point ``f_sep = λ·centre + (1−λ)·f_out``
        # (a per-subproblem dict, since a shared master column may carry
        # different interior values for its owning subproblems), re-projected
        # onto the CURRENT feasible set (the centre is an old incumbent point
        # feasible against a PAST projection, so ``f_sep`` can exceed the
        # current one — projecting down keeps the UB valid; hence
        # ``hard_fail=False``: this is routine, not a bug signal).  The cut
        # is then GENERATED at ``f_sep`` (its generation point), and a
        # per-subproblem separation test below decides whether the stabilizer
        # must force an exact-Benders out-step next.  With ``λ == 0.0`` (OFF)
        # this whole block is skipped and the subproblem pin / gen-point are
        # ``point`` verbatim ⇒ byte-identical to exact Benders.
        f_sep_by_sub: dict[str, dict[int, float]] = {}
        if in_out_on:
            for sub in subproblems:
                f_sep_s = stabilizers[sub.name].separation_point(point)
                if f_sep_s is not point:
                    # A genuine interior point — re-project a COPY onto the
                    # current feasible set (leaves ``point`` untouched for
                    # the verbatim out-step case, where ``separation_point``
                    # returned it as-is).
                    f_sep_s = dict(f_sep_s)
                    master.project_point(f_sep_s, msol, hard_fail=False)
                f_sep_by_sub[sub.name] = f_sep_s

        def _sub_pin(i: int, _f_sep=f_sep_by_sub) -> dict[int, float]:
            # Per-subproblem pin point: its own projected ``f_sep`` (in-out
            # ON; only called on that path).  ``_f_sep`` is bound at
            # definition so the closure pins THIS iteration's points.
            return _f_sep[subproblems[i].name]

        sub_costs = {}
        next_cuts: list[tuple[str, dict[int, float], float, dict[int, float]]] = []
        # Per-subproblem slopes recovered this pass (keyed by name), so the
        # in-out separation test / register below need not re-scan
        # ``next_cuts``.
        slopes_by_sub: dict[str, dict[int, float]] = {}
        t_subs = time.perf_counter()
        sub_results = _solve_subs(_sub_pin if in_out_on else point)
        dt_subs = time.perf_counter() - t_subs
        for sub, res in zip(subproblems, sub_results):
            gen_point = f_sep_by_sub[sub.name] if in_out_on else point
            slopes_by_sub[sub.name] = res.slopes
            sub_costs[sub.name] = res.cost
            next_cuts.append((sub.name, gen_point, res.cost, res.slopes))

        # --- UB = master native cost + Σ subproblem costs at the SAME
        # (pin, master solution) — all terms in the caller's scale.  ``pin``
        # is ``point`` (OFF) or each subproblem's projected ``f_sep`` (ON);
        # both are feasible for the master, so the UB is valid either way.
        ub = master.native_cost(msol, recourse_by_sub) + sum(sub_costs.values())
        improved = ub < best_ub
        if improved:
            best_ub = ub
            # The incumbent coupling point.  With in-out OFF it is the single
            # ``point``; with in-out ON each subproblem was solved at its own
            # ``f_sep``, so the incumbent value on a master column is the
            # separation value of the subproblem that OWNS it (ownership =
            # its ``SubproblemResult.slopes`` key set — by protocol it
            # carries a key for every pinned column).  We use ``point`` as
            # the base and overlay each subproblem's own ``f_sep`` so a
            # column reflects the value actually solved for it (a shared
            # column takes the LAST owner's value, in subproblem order).
            if in_out_on:
                incumbent_point = dict(point)
                for sub, res in zip(subproblems, sub_results):
                    fsr = f_sep_by_sub[sub.name]
                    for mc in res.slopes:
                        mc = int(mc)
                        if mc in fsr:
                            incumbent_point[mc] = fsr[mc]
            else:
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

        # --- STALL GUARD (fail fast, don't silently exhaust the iter cap).
        # Feed the domain-free monitor this iteration's (LB, best_UB); it
        # holds the best_UB window internally and returns a verdict once
        # ``window`` iterations have been seen.  A stall = incumbent frozen
        # for the window AND still blown up far above the reference scale
        # AND gap far from converged — mutually exclusive with the sandwich
        # break above (which has gap≈0).  On a stall the frozen incumbent is
        # garbage (best_UB can be orders above the true optimum), so
        # returning it would hand the caller a catastrophically wrong plan:
        # HARD-fail with the structured exception (the caller renders its
        # own domain diagnostics off the carried fields).
        verdict = stall_monitor.update(lb, best_ub)
        if verdict.stalled:
            raise BendersStalled(
                f"benders: stalled at iteration {iterations}: the best "
                f"feasible cost has not improved for {stall_monitor.window} "
                f"iterations, the relative gap is stuck at ~{gap:.2f} (far "
                f"from the {options.tol} tolerance), and the incumbent "
                f"{best_ub:.6e} still sits far above the reference cost "
                f"scale {stall_monitor.reference_scale:.6e}",
                iteration=iterations,
                gap=gap,
                tol=options.tol,
                window=stall_monitor.window,
                reference_scale=stall_monitor.reference_scale,
                sub_costs=dict(sub_costs),
                sub_reference_costs=dict(sub_reference_costs),
            )

        # --- IN-OUT: feed each subproblem's outcome back to its stabilizer.
        # The separation flag is PER-SUBPROBLEM: the MOMENT a subproblem's
        # cut fails to separate its ``point``, that subproblem's next
        # ``separation_point`` returns ``point`` VERBATIM (λ=0 → exact
        # Benders, guaranteed to separate unless already optimal).
        # ``improved`` (best_UB dropped this iteration) drives the serious
        # step (centre ← incumbent).
        if in_out_on:
            incumbent_for_register = best["point"] if best is not None else point
            for sub in subproblems:
                separated = _cut_separates(
                    sub_costs[sub.name],
                    slopes_by_sub[sub.name],
                    point,
                    f_sep_by_sub[sub.name],
                    recourse_by_sub[sub.name],
                )
                stabilizers[sub.name].register(
                    master_point=point,
                    separated=separated,
                    incumbent_point=incumbent_for_register,
                    improved=improved,
                )

        pending_cuts = next_cuts

        # --- PERIODIC MASTER CUT COMPACTION.  When the accumulated cut-row
        # count reaches ``compact_at``, the master adapter's ``compact_cuts``
        # classifies every retained cut row at the RAW master vertex
        # ``msol.col_value`` (binding iff slack ≤ tol), deletes the
        # strictly-slack rows in place, and re-solves + rolls back on any
        # objective drift (its verify belt — LB-preserving).  The adapter
        # owns all of that; the coordinator only triggers it and tracks the
        # kept count.  Guarded by ``compact_at > 0`` so the default (OFF)
        # path is byte-identical to the pre-compaction loop.
        #
        # PLACEMENT + ``msol`` SAFETY.  Called at the VERY END of the loop
        # body, AFTER ``pending_cuts = next_cuts``, so ``msol`` has already
        # been FULLY consumed by THIS iteration (``read_point`` / the LB
        # self-checks / ``_check_cuts_satisfied`` / the UB + sandwich/stall
        # guards); nothing downstream in the iteration reads it again.
        # ``msol.col_value`` is the RAW master optimum — the projection above
        # mutates ``new_point``, NOT ``msol`` — which is the correct
        # classification point.  Deleting rows here simply shrinks the master
        # for the NEXT iteration's ``solve()``.
        #
        # SELECTION POLICY (``cut_policy``): ``slack`` drops cuts strictly
        # slack at the current optimum; ``dominance`` groups cuts by recourse
        # column and, over ``cut_window`` (the last W master vertices), keeps
        # only the oldest group-max achiever at each trial point.  Both are
        # LB-safe (verify-restore belt in the adapter).
        if compact_at > 0 and cut_rows >= compact_at:
            comp = master.compact_cuts(
                msol,  # msol = raw pre-projection master vertex (latest trial point)
                policy=options.cut_policy,
                trial_col_values=list(cut_window),
            )
            cut_rows = comp["kept"]
            _logger.info(
                "[benders timing] iter %d: cut compaction kept=%d dropped=%d restored=%s",
                iterations,
                comp["kept"],
                comp["dropped"],
                comp["restored"],
            )

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
