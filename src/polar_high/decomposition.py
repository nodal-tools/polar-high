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

from collections import deque
from dataclasses import dataclass

__all__ = [
    "InOutStabilizer",
    "StallMonitor",
    "StallVerdict",
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
