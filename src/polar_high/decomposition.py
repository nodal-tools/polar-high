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
