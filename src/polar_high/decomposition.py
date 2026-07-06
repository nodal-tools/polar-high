"""Domain-agnostic tail-off / stall detection for decomposition drivers.

A small, generic utility for cutting-plane / Benders drivers (e.g. the region
recourse loop that consumes :meth:`~polar_high.engine.WarmProblem.add_cut_row`)
that need to notice when the outer iteration has stopped making progress and
bail out with a diagnostic instead of silently burning the iteration cap.

The detector knows only about the two scalars every Benders-style loop already
maintains — a monotone lower bound and a best-so-far upper bound — plus one
caller-supplied ``reference_scale``: a "sane objective magnitude" the driver
computes from its own problem (polar-high does NOT know what it means, e.g. that
in FlexTool it is a sum of stand-alone region costs). It carries NO domain
concepts (no regions, nodes, penalties): those live entirely in the caller.

Stall predicate (a CONJUNCTION, so no single signal false-positives):

1. **Far from converged.** The relative gap ``(best_ub - lower_bound) /
   max(1, |best_ub|)`` exceeds ``gap_floor``. A run that is already near
   tolerance is never a stall, however flat.
2. **Incumbent frozen.** The best upper bound has not improved by more than
   ``min_rel`` (relative, ``improvement / max(1, |best_ub|)`` — matching the gap
   formula so it is scale-stable) across the whole trailing window of ``window``
   iterations. A run whose incumbent is still dropping is making progress.
3. **Still blown up.** The best upper bound is still above
   ``blowup_mult * max(1, reference_scale)`` — i.e. the incumbent has not merely
   frozen, it froze *far above* any sane objective magnitude (the penalty /
   complete-recourse regime). A benign frozen window that has already fallen to
   ~the reference scale is not a stall.

Only when ALL THREE hold over a full ``window`` is the run declared stalled.
This is what separates a genuine tail-off (incumbent frozen high, gap ~1, for
many iterations) from the benign frozen windows a converging run can exhibit
(early blow-up that shrinks fast; short flat stretches at a sane magnitude).

The monitor is stateful (it holds a bounded ``deque`` of the recent
best-upper-bounds so the caller need not) and deterministic: feeding the same
bound sequence always yields the same verdicts. It never raises and never
mutates the caller's problem — it only *reports*.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

__all__ = [
    "InOutStabilizer",
    "StallMonitor",
    "StallVerdict",
    "TrustRegionStabilizer",
]

# Empirical defaults (validated on Benders spatial-decomposition traces up to
# N=10 regions; the caller may override every one at construction). See the
# module docstring for what each gates.
_DEFAULT_WINDOW = 8
_DEFAULT_MIN_REL = 1e-6
_DEFAULT_GAP_FLOOR = 0.02
_DEFAULT_BLOWUP_MULT = 5.0


@dataclass(frozen=True)
class StallVerdict:
    """The outcome of one :meth:`StallMonitor.update` call.

    ``stalled`` is the only field a caller must branch on; the rest are the
    measured quantities behind the decision, exposed so the caller can build a
    domain-specific diagnostic without recomputing them.
    """

    stalled: bool
    #: ``"frozen-blowup"`` when the conjunction fired, else ``None``.
    kind: str | None
    #: Relative gap ``(best_ub - lower_bound) / max(1, |best_ub|)`` this iter.
    gap: float
    #: Relative improvement of ``best_ub`` across the trailing window
    #: (``(oldest - newest) / max(1, |best_ub|)``); ``inf`` before the window
    #: fills (treated as "improving", so never stalled early).
    best_ub_rel_improvement: float
    #: ``best_ub / max(1, reference_scale)`` — how far above the sane magnitude
    #: the incumbent still sits.
    blowup_ratio: float
    #: The window length actually observed so far (``<= window`` while filling).
    window: int


class StallMonitor:
    """Stateful tail-off detector over a ``(lower_bound, best_upper_bound)``
    stream.

    Construct once per decomposition run with the problem's ``reference_scale``
    (a sane objective magnitude the caller derives) and, optionally, the four
    thresholds. Call :meth:`update` once per outer iteration with the current
    lower bound and the current *best* (incumbent) upper bound; it returns a
    :class:`StallVerdict`. The verdict can only be ``stalled`` once at least
    ``window`` iterations have been observed (the incumbent-frozen test needs a
    full trailing window).

    Parameters
    ----------
    reference_scale
        A caller-computed "sane objective magnitude". Guarded internally by
        ``max(1, |reference_scale|)`` so a zero/near-zero reference cannot make
        the blow-up gate divide by ~0.
    window
        ``K``: number of trailing iterations over which the incumbent must be
        frozen. Non-positive is clamped to 1.
    min_rel
        Relative-improvement threshold below which the incumbent counts as
        frozen over the window.
    gap_floor
        Minimum relative gap for a stall (a near-converged run is never a
        stall).
    blowup_mult
        The incumbent must exceed ``blowup_mult * max(1, reference_scale)`` to
        count as still-blown-up.
    """

    def __init__(
        self,
        reference_scale: float,
        *,
        window: int = _DEFAULT_WINDOW,
        min_rel: float = _DEFAULT_MIN_REL,
        gap_floor: float = _DEFAULT_GAP_FLOOR,
        blowup_mult: float = _DEFAULT_BLOWUP_MULT,
    ) -> None:
        self.reference_scale = float(reference_scale)
        self.window = max(1, int(window))
        self.min_rel = float(min_rel)
        self.gap_floor = float(gap_floor)
        self.blowup_mult = float(blowup_mult)
        # Bounded, deterministic history of the best upper bounds (newest last).
        # Holds up to the ``window - 1`` values PRECEDING the current update, so
        # the trailing improvement compares across a full ``window`` span
        # (oldest-held ... current) exactly when ``window`` iterations have been
        # seen — the K-th update is the earliest a stall can be declared.
        self._best_ubs: deque[float] = deque(maxlen=max(1, self.window - 1))
        self._n_seen = 0

    def update(self, lower_bound: float, upper_bound: float) -> StallVerdict:
        """Record one iteration's ``(lower_bound, best_upper_bound)`` and return
        the :class:`StallVerdict`.

        ``upper_bound`` is the *best* (incumbent) upper bound, not the raw
        per-iteration one — the caller tracks the incumbent; the monitor only
        needs the value it should test for freezing. ``update`` is pure w.r.t.
        the caller's problem (it only appends to its own bounded window).
        """
        lb = float(lower_bound)
        ub = float(upper_bound)
        ref = max(1.0, abs(self.reference_scale))

        gap = (ub - lb) / max(1.0, abs(ub))
        blowup_ratio = ub / ref

        # Trailing-window improvement of the incumbent. Measured BEFORE the new
        # value is appended: the deque holds the ``window - 1`` values preceding
        # this update, so its oldest entry together with the current ``ub``
        # spans a full ``window``-iteration trailing window — available exactly
        # once ``window`` iterations have been seen (``window == 1`` is always
        # "full", comparing the current value against itself ⇒ 0 improvement).
        if len(self._best_ubs) >= self.window - 1:
            oldest = self._best_ubs[0] if self._best_ubs else ub
            rel_improvement = (oldest - ub) / max(1.0, abs(ub))
            window_full = True
        else:
            # Window not yet full — cannot declare frozen; report "improving".
            rel_improvement = float("inf")
            window_full = False

        self._best_ubs.append(ub)
        self._n_seen += 1
        # Number of iterations in the current trailing window (capped at K).
        window_seen = min(self.window, self._n_seen)

        far_from_converged = gap > self.gap_floor
        incumbent_frozen = window_full and rel_improvement <= self.min_rel
        still_blown_up = ub > self.blowup_mult * ref

        stalled = far_from_converged and incumbent_frozen and still_blown_up

        return StallVerdict(
            stalled=stalled,
            kind="frozen-blowup" if stalled else None,
            gap=gap,
            best_ub_rel_improvement=rel_improvement,
            blowup_ratio=blowup_ratio,
            window=window_seen,
        )


# ---------------------------------------------------------------------------
# In-out separation (Ben-Ameur & Neto 2007) — generic interior cut-point picker.
# ---------------------------------------------------------------------------

# Below this the interpolation weight is treated as an exact ``0.0`` (exact
# Benders): once a region's ``weight`` has shrunk under this it is PINNED to
# ``0.0`` permanently. Guarantees the "shrink bottoms out at a forced 0"
# convergence rule terminates in finitely many null steps rather than
# asymptoting at a positive value that can never separate a degenerate vertex.
_WEIGHT_ZERO_EPS = 1e-9


class InOutStabilizer:
    """Generic in-out separation point picker for a Benders-style driver.

    Cutting-plane methods that generate each cut at the raw master vertex
    ``f_out`` tail off badly when the recourse is flat in the coupling variable
    (dual-degenerate slopes): the master wanders among cost-equivalent vertices
    and the bound closes very slowly. Ben-Ameur & Neto (2007) generate the cut
    at an INTERIOR *separation point* ``f_sep = λ·centre + (1-λ)·f_out`` — a
    convex combination of a stable centre and the master vertex — instead. The
    cuts are better centred, the master stops wandering, and the bound closes
    faster, at zero extra subproblem solves.

    This class is domain-free: it operates only on ``{col_id -> value}`` point
    dicts and scalar weights. It has NO notion of flows, regions, storage, or
    capacities — the caller (e.g. FlexTool's ``_benders.py``) owns all of that.
    Because the correct stabilisation unit is PER-REGION (a single global weight
    lets one well-behaved region mask a degenerate one), the caller constructs
    ONE ``InOutStabilizer`` PER REGION; each instance therefore holds a single
    centre, a single (per-region) weight ``λ`` and a single counter.

    Lifecycle per outer iteration, for one region::

        f_sep = stab.separation_point(f_out)     # where to solve this region
        # ... solve region at f_sep, form the cut, test separation ...
        kind = stab.register(master_point=f_out, separated=..., \
                             incumbent_point=..., improved=...)

    Convergence guarantee (Ben-Ameur & Neto): the MOMENT a region's cut fails
    to separate its ``f_out`` (``separated=False``), the next
    :meth:`separation_point` returns ``master_point`` VERBATIM (``λ=0`` ⇒ exact
    Benders, guaranteed to separate unless already optimal). On such a null step
    the weight is also SHRUNK, and the shrink bottoms out at a forced ``0`` — not
    a positive floor, since a cut arbitrarily close to ``f_out`` can still fail
    to separate a degenerate vertex; only ``λ=0`` guarantees separation.
    ``out_step_every`` is a secondary belt-and-braces cap that periodically
    forces an out-step even without a separation failure.

    Parameters
    ----------
    weight
        Initial interpolation weight ``λ`` on the interior centre. ``0.0`` is a
        verbatim no-op (exact Benders — :meth:`separation_point` returns its
        input unchanged, by construction, so byte-parity with the off path
        holds). Must be in ``[0, 1)``: ``weight >= 1`` ("never query the
        master") is non-convergent and ``weight < 0`` is meaningless — both are
        REJECTED at construction (a clear error rather than a silent clamp, so a
        config mistake surfaces).
    weight_min
        Threshold at which the geometric null-step descent stops shrinking and
        SNAPS the weight to a forced exact ``0.0``. It does NOT act as a
        positive floor: the convergence guarantee is the forced ``0`` (a cut
        arbitrarily close to ``f_out`` can still fail to separate a degenerate
        vertex), so once ``λ`` reaches ``max(weight_min, _WEIGHT_ZERO_EPS)`` it
        is pinned to ``0.0``. A larger ``weight_min`` therefore reaches the
        exact-Benders out-step SOONER, never blocks it. Must be in
        ``[0, weight]``.
    shrink
        Multiplicative weight-shrink factor applied on each null (no-separation)
        step, ``0 < shrink < 1``.
    out_step_every
        Secondary cap: force an out-step every this-many registered steps even
        if separation has not failed. ``<= 0`` disables the periodic cap
        (leaving only the on-no-separation rule). Positive integer otherwise.
    """

    def __init__(
        self,
        *,
        weight: float = 0.5,
        weight_min: float = 0.0,
        shrink: float = 0.5,
        out_step_every: int = 5,
    ) -> None:
        weight = float(weight)
        if not (0.0 <= weight < 1.0):
            raise ValueError(
                "InOutStabilizer weight must be in [0, 1): "
                f"got {weight!r} (>= 1 never queries the master ⇒ "
                "non-convergent; < 0 is meaningless)"
            )
        weight_min = float(weight_min)
        if not (0.0 <= weight_min <= weight):
            raise ValueError(
                "InOutStabilizer weight_min must be in [0, weight]: "
                f"got {weight_min!r} with weight {weight!r}"
            )
        shrink = float(shrink)
        if not (0.0 < shrink < 1.0):
            raise ValueError(f"InOutStabilizer shrink must be in (0, 1): got {shrink!r}")
        self.weight = weight
        self.weight_min = weight_min
        self.shrink = shrink
        self.out_step_every = int(out_step_every)
        # The stable interior centre (a ``{col_id -> value}`` point). ``None``
        # until the caller seeds it — the first ``separation_point`` before any
        # centre exists returns the master point verbatim (a pass-through).
        self._centre: dict[int, float] | None = None
        # When set, the NEXT ``separation_point`` returns ``master_point``
        # verbatim (the forced out-step). Armed by a no-separation register, and
        # by the periodic ``out_step_every`` cap.
        self._force_out = False
        # Count of registered steps since the last (forced or periodic)
        # out-step, for the secondary ``out_step_every`` cap.
        self._since_out = 0

    def set_centre(self, centre: dict[int, float]) -> None:
        """Explicitly seed / replace the stable interior centre.

        Optional: the centre is otherwise established lazily on the first
        :meth:`register`. Provided so a caller that knows the natural centre up
        front (e.g. FlexTool's autarky ``f̄=0``) can set it before the first
        :meth:`separation_point`. Copies the dict so later caller mutation of
        the passed point does not alias the stored centre.
        """
        self._centre = dict(centre)

    def separation_point(self, master_point: dict[int, float]) -> dict[int, float]:
        """Return the point to evaluate the subproblem at this iteration.

        ``f_sep[c] = λ·centre[c] + (1-λ)·master_point[c]`` per column, EXCEPT:

        * ``λ == 0.0`` ⇒ return ``master_point`` VERBATIM (the same dict object),
          skipping the convex-combo arithmetic entirely so byte-parity with the
          exact-Benders path holds by construction, not by float luck;
        * no centre yet (first call before any :meth:`register`) ⇒ pass-through;
        * a pending forced out-step (armed by a prior no-separation
          :meth:`register`) ⇒ pass-through this once.

        Pure: does NOT mutate any state (the forced-out flag is CONSUMED in
        :meth:`register`, not here, so re-querying is idempotent).
        """
        if self.weight == 0.0 or self._centre is None or self._force_out:
            return master_point
        w = self.weight
        centre = self._centre
        return {c: w * centre.get(c, v) + (1.0 - w) * v for c, v in master_point.items()}

    def register(
        self,
        *,
        master_point: dict[int, float],
        separated: bool,
        incumbent_point: dict[int, float] | None,
        improved: bool,
    ) -> str:
        """Record this iteration's outcome; update centre + weight; return the
        step kind taken.

        Parameters
        ----------
        master_point
            The raw master vertex ``f_out`` this iteration (establishes the
            centre on the very first register, when none exists yet).
        separated
            Whether the cut generated at ``f_sep`` actually SEPARATED
            ``f_out`` (strictly, per the caller's tolerance). ``False`` arms a
            forced out-step for the next :meth:`separation_point` and shrinks
            the weight toward a forced ``0``.
        incumbent_point
            The best-upper-bound point when ``improved`` (else ignored). On a
            serious step the centre JUMPS to it (Ben-Ameur & Neto default).
        improved
            Whether the incumbent (best UB) improved this iteration.

        Returns
        -------
        str
            ``"serious"`` (incumbent improved ⇒ centre jumped),
            ``"out"``     (cut failed to separate ⇒ forced out-step armed,
                           weight shrunk toward 0), or
            ``"null"``    (cut separated but no incumbent improvement ⇒ centre
                           and weight held).
        """
        # Seed the centre on the first register if the caller never set it.
        if self._centre is None:
            self._centre = (
                dict(incumbent_point) if incumbent_point is not None else dict(master_point)
            )

        # A forced/periodic out-step is now spent (this register follows the
        # pass-through separation_point it armed).
        was_forced = self._force_out
        self._force_out = False

        if improved and incumbent_point is not None:
            # Serious step: jump the centre to the incumbent (BAN default).
            self._centre = dict(incumbent_point)
            self._since_out = 0 if was_forced else self._since_out + 1
            self._maybe_arm_periodic_out()
            return "serious"

        if not separated:
            # Null step (no separation): arm the exact-Benders out-step for the
            # NEXT separation_point and shrink the weight toward a FORCED 0.
            self._force_out = True
            self._since_out = 0
            self._shrink_weight()
            return "out"

        # Separated but no incumbent improvement: hold centre and weight.
        self._since_out = 0 if was_forced else self._since_out + 1
        self._maybe_arm_periodic_out()
        return "null"

    def _shrink_weight(self) -> None:
        """Shrink the weight one null step, bottoming out at a FORCED 0.

        ``λ ← λ·shrink``; then, once the weight has descended to at-or-below the
        snap threshold ``max(weight_min, _WEIGHT_ZERO_EPS)``, pin it to exactly
        ``0.0`` for the rest of the run (⇒ exact Benders). The forced ``0`` —
        NOT a positive floor — is the convergence guarantee: a cut arbitrarily
        close to ``f_out`` can still fail to separate a degenerate vertex, so
        ``weight_min`` merely sets where the descent stops shrinking and snaps to
        zero, it never blocks the eventual exact-Benders out-step.
        """
        self.weight *= self.shrink
        if self.weight <= max(self.weight_min, _WEIGHT_ZERO_EPS):
            self.weight = 0.0

    def _maybe_arm_periodic_out(self) -> None:
        """Secondary belt-and-braces cap: force an out-step every
        ``out_step_every`` registered non-out steps (disabled when ``<= 0``)."""
        if self.out_step_every > 0 and self._since_out >= self.out_step_every:
            self._force_out = True
            self._since_out = 0


# ---------------------------------------------------------------------------
# Trust-region / boxstep stabilization (Ruszczyński 1986; Marsten-Hogan-
# Blankenship 1975; Göke, Schmidt & Kendziorski 2023) — generic box trust
# region on the master coupling point.
# ---------------------------------------------------------------------------

# A predicted decrease at or below this fraction of the (guarded) incumbent
# magnitude is treated as ZERO: the cutting-plane model, minimized over the
# box, offers no more improvement than the incumbent already has, so the box
# is not shrunk (there is nothing to refine toward) — the loop's own LB/UB
# sandwich, not the trust region, then certifies optimality. Keyed off the
# incumbent magnitude (not an absolute) so it is scale-stable.
_PREDICTED_ZERO_EPS = 1e-12


class TrustRegionStabilizer:
    """Generic box trust-region stabilizer for a Benders-style master.

    Where :class:`InOutStabilizer` stabilizes the subproblem *oracle query*
    (it interpolates toward a centre but leaves the master's own proposal
    unbounded), a trust region stabilizes the master *primal*: it constrains
    the master's coupling point to a box ``[x̄ − Δ, x̄ + Δ]`` (an ∞-norm
    radius Δ about an incumbent centre ``x̄``) so the coupling STEP is bounded
    directly, in variable space. That is the decisive difference for a master
    whose cut model can propose wild points a loose relaxation permits: in-out
    *damps* such a proposal toward the centre but a fixed fraction of a huge
    proposal is still huge; the box *caps* it. It is also why the upper bound
    is valid **by construction** — the master is solved WITH the box, so its
    own primal and the subproblem queries share ONE feasible iterate; the
    whole objective scored there is a genuine single-point L-shaped bound with
    no separate re-evaluation (Göke et al. 2023, Algorithm 2, Step 5; the
    coordinator's ``native_cost(boxed_iterate)`` fast path).

    This class is domain-free. It manages only a centre point
    (``{col_id -> value}``), a scalar radius, and the incumbent objective at
    the centre; it has NO notion of flows, capacities or regions — the caller
    intersects the box with the columns' original bounds (so the box never
    *widens* the feasible set) and owns the solves.

    Radius management is the standard trust-region ratio test (Ruszczyński
    1986; Linderoth & Wright 2003; Göke et al. 2023). Each iteration the
    caller solves the boxed master, evaluates the true objective at the
    resulting iterate, and calls :meth:`update` with

    * ``trial_obj``  — the TRUE whole-objective value at the boxed iterate
      (the L-shaped UB there): ``c(iterate) + Σ_r Q_r(iterate)``;
    * ``model_obj``  — the boxed master's optimal objective (the cutting-plane
      MODEL value at the iterate), a lower estimate of ``trial_obj``.

    The **predicted** decrease is ``f(x̄) − model_obj`` (what the model said
    the step would buy) and the **actual** decrease is ``f(x̄) − trial_obj``
    (what it truly bought). The ratio ``ρ = actual / predicted`` drives:

    * **serious step** (``ρ ≥ accept_ratio``): the true objective improved
      enough — accept the iterate as the new centre ``x̄`` and, when the step
      was very successful (``ρ ≥ expand_ratio``), EXPAND the radius
      (``Δ ← min(expand·Δ, max_radius)``) so the next step can reach further;
    * **null step** (``ρ < accept_ratio``): the model over-promised — keep the
      centre and SHRINK the radius (``Δ ← max(shrink·Δ, min_radius)``). The
      subproblem solved at the (rejected) iterate still returns a cut, so the
      model is refined even on a null step; over successive nulls the model
      becomes accurate near the centre and a serious step resumes.

    Because both ``model_obj`` and ``trial_obj`` are evaluated with the centre
    inside the box, the model (a valid under-estimate of the true objective)
    gives ``model_obj ≤ f(x̄)``, so the predicted decrease is ``≥ 0``; a
    predicted decrease that has collapsed to ~0 means the model has nothing
    more to offer over the box (near-optimal) — the centre is accepted if the
    trial improved on it and the radius is held (the loop's LB/UB sandwich, not
    the trust region, certifies optimality there).

    Default radius policy (documented per CLAUDE-style rationale): Δ₀ is
    ``radius · scale`` where ``scale`` is the caller's coupling-variable scale
    (so the initial box is expressed relative to the variables' own magnitude,
    never a hardcoded problem-specific number); ``expand = 2.0`` /
    ``shrink = 0.5`` are the classical geometric factors (Ruszczyński 1986;
    Linderoth & Wright 2003) and ``accept_ratio = 0.1`` / ``expand_ratio =
    0.5`` the standard sufficient-decrease / very-successful thresholds (Göke
    et al. 2023 use factors in this range). On a problem whose optimum has
    near-zero coupling flows the centre barely moves and the trust region's
    real job is to PREVENT early over-proposals, so an aggressive null-step
    shrink from a moderate Δ₀ converges quickly.

    Parameters
    ----------
    radius
        Initial box half-width in units of ``scale`` (multiplied by ``scale``
        to give Δ₀ in the coupling variables' own units). Must be ``> 0``.
    scale
        Coupling-variable scale (a caller-supplied "typical coupling
        magnitude"); ``Δ₀ = radius · scale``. Must be ``> 0`` (``1.0`` leaves
        ``radius`` as an absolute).
    expand
        Radius EXPANSION factor on a very-successful serious step. Must be
        ``> 1``.
    shrink
        Radius SHRINK factor on a null step. Must be in ``(0, 1)``.
    accept_ratio
        Sufficient-decrease threshold ``κ``: a serious step needs
        ``ρ ≥ accept_ratio``. Must be in ``[0, 1)``.
    expand_ratio
        Very-successful threshold: expand the radius when ``ρ ≥ expand_ratio``.
        Must be in ``(accept_ratio, 1]``.
    min_radius
        Lower clamp for the shrinking radius (``≥ 0``; the geometric shrink
        keeps Δ positive, so ``0.0`` is a safe default — the box never
        degenerates to a hard fix under it). Must be ``≥ 0`` and ``≤ Δ₀``.
    max_radius
        Upper clamp for the expanding radius. Must be ``≥ Δ₀`` (``inf`` leaves
        expansion uncapped).
    """

    def __init__(
        self,
        *,
        radius: float,
        scale: float = 1.0,
        expand: float = 2.0,
        shrink: float = 0.5,
        accept_ratio: float = 0.1,
        expand_ratio: float = 0.5,
        min_radius: float = 0.0,
        max_radius: float = math.inf,
    ) -> None:
        radius = float(radius)
        if not (radius > 0.0):
            raise ValueError(f"TrustRegionStabilizer radius must be > 0: got {radius!r}")
        scale = float(scale)
        if not (scale > 0.0):
            raise ValueError(f"TrustRegionStabilizer scale must be > 0: got {scale!r}")
        expand = float(expand)
        if not (expand > 1.0):
            raise ValueError(f"TrustRegionStabilizer expand must be > 1: got {expand!r}")
        shrink = float(shrink)
        if not (0.0 < shrink < 1.0):
            raise ValueError(f"TrustRegionStabilizer shrink must be in (0, 1): got {shrink!r}")
        accept_ratio = float(accept_ratio)
        if not (0.0 <= accept_ratio < 1.0):
            raise ValueError(
                f"TrustRegionStabilizer accept_ratio must be in [0, 1): got {accept_ratio!r}"
            )
        expand_ratio = float(expand_ratio)
        if not (accept_ratio < expand_ratio <= 1.0):
            raise ValueError(
                "TrustRegionStabilizer expand_ratio must be in (accept_ratio, 1]: "
                f"got {expand_ratio!r} with accept_ratio {accept_ratio!r}"
            )
        radius0 = radius * scale
        min_radius = float(min_radius)
        if not (0.0 <= min_radius <= radius0):
            raise ValueError(
                "TrustRegionStabilizer min_radius must be in [0, radius·scale]: "
                f"got {min_radius!r} with radius·scale {radius0!r}"
            )
        max_radius = float(max_radius)
        if not (max_radius >= radius0):
            raise ValueError(
                "TrustRegionStabilizer max_radius must be >= radius·scale: "
                f"got {max_radius!r} with radius·scale {radius0!r}"
            )
        self.expand = expand
        self.shrink = shrink
        self.accept_ratio = accept_ratio
        self.expand_ratio = expand_ratio
        self.min_radius = min_radius
        self.max_radius = max_radius
        #: Current box half-width Δ (in the coupling variables' own units).
        self.radius = radius0
        #: The last step taken (``"serious"`` | ``"null"`` | ``None`` before
        #: the first :meth:`update`) — diagnostics only.
        self.last_step: str | None = None
        # The incumbent centre (a ``{col_id -> value}`` point) and the TRUE
        # objective there. ``_centre_obj`` is ``inf`` until the first
        # :meth:`update` seeds it from the first boxed iterate (the caller
        # supplies only the centre POINT up front; the objective there needs a
        # solve the caller has not done yet).
        self._centre: dict[int, float] | None = None
        self._centre_obj: float = math.inf

    def set_centre(self, centre: dict[int, float], obj: float = math.inf) -> None:
        """Seed / replace the incumbent centre (and optionally its objective).

        The caller seeds the centre POINT before the first :meth:`box` (e.g.
        FlexTool's autarky / no-coupling point). The centre objective is
        normally left at its ``inf`` default and established lazily by the
        first :meth:`update` (the first boxed iterate becomes the incumbent);
        pass ``obj`` only when a genuine objective at ``centre`` is already
        known. Copies the point so later caller mutation does not alias it.
        """
        self._centre = dict(centre)
        self._centre_obj = float(obj)

    @property
    def centre(self) -> dict[int, float] | None:
        """The current incumbent centre point (a copy), or ``None`` if unseeded."""
        return dict(self._centre) if self._centre is not None else None

    @property
    def centre_obj(self) -> float:
        """The true objective at the current centre (``inf`` until seeded)."""
        return self._centre_obj

    def box(self) -> tuple[dict[int, float], float]:
        """Return ``(centre, radius)`` — the box the caller intersects with
        the coupling columns' original bounds before the boxed master solve.

        The centre is a fresh copy (the caller may mutate its own view). The
        radius is Δ in the coupling variables' own units. Raises if no centre
        has been seeded (the caller must :meth:`set_centre` first)."""
        if self._centre is None:
            raise ValueError(
                "TrustRegionStabilizer.box called before a centre was seeded "
                "(call set_centre first)"
            )
        return dict(self._centre), self.radius

    def update(
        self,
        *,
        trial_point: dict[int, float],
        trial_obj: float,
        model_obj: float,
    ) -> float:
        """Record the boxed iterate's outcome; update the centre + radius via
        the trust-region ratio test; return the NEW radius.

        Parameters
        ----------
        trial_point
            The boxed master iterate (``{col_id -> value}``) — the point the
            subproblems were queried at.
        trial_obj
            The TRUE whole-objective value at ``trial_point`` (the L-shaped UB
            there): ``c(trial) + Σ_r Q_r(trial)``.
        model_obj
            The boxed master's optimal objective at ``trial_point`` (the
            cutting-plane MODEL value, a lower estimate of ``trial_obj``).

        Returns
        -------
        float
            The updated radius Δ (also stored on :attr:`radius`).
        """
        trial_obj = float(trial_obj)
        model_obj = float(model_obj)

        # First update: no incumbent objective yet — accept the first boxed
        # iterate as the incumbent centre unconditionally (bootstrap), holding
        # the radius (no ratio to test against).
        if not math.isfinite(self._centre_obj):
            self._centre = dict(trial_point)
            self._centre_obj = trial_obj
            self.last_step = "serious"
            return self.radius

        predicted = self._centre_obj - model_obj
        actual = self._centre_obj - trial_obj

        # Predicted decrease collapsed to ~0: the model, minimized over the
        # box, is no better than the incumbent — nothing to refine toward, so
        # do NOT shrink. Accept the iterate if it genuinely improved (a
        # tie-or-better move keeps the incumbent current); hold the radius.
        if predicted <= _PREDICTED_ZERO_EPS * max(1.0, abs(self._centre_obj)):
            if actual > 0.0:
                self._centre = dict(trial_point)
                self._centre_obj = trial_obj
            self.last_step = "serious"
            return self.radius

        ratio = actual / predicted
        if ratio >= self.accept_ratio:
            # Serious step: accept the iterate as the new centre; expand the
            # radius on a very-successful step so the next step reaches further.
            self._centre = dict(trial_point)
            self._centre_obj = trial_obj
            self.last_step = "serious"
            if ratio >= self.expand_ratio:
                self.radius = min(self.radius * self.expand, self.max_radius)
        else:
            # Null step: the model over-promised — hold the centre, shrink the
            # radius (the cut just added at the rejected iterate refines the
            # model for the next attempt).
            self.last_step = "null"
            self.radius = max(self.radius * self.shrink, self.min_radius)
        return self.radius
