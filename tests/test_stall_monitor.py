"""Unit tests for ``polar_high.decomposition.StallMonitor`` — the generic
tail-off detector for Benders-style decomposition drivers.

Pure tests over synthetic ``(lower_bound, best_upper_bound)`` sequences; NO
solver, NO domain concepts. The two headline cases are the must-NOT-fire
converging trace (a real N=10 Benders run whose incumbent blows up early, has
benign short frozen windows, then closes) and the must-FIRE frozen-blow-up
stall.
"""

from __future__ import annotations

from polar_high import StallMonitor, StallVerdict

# ---------------------------------------------------------------------------
# The full N=10 (0.25x) converging trace from the plan (§3.8). Σautarky ≈
# 1.69e10 dominated by one region at 1.66e10. best_UB per outer iteration; it
# blows up to 9.62e13 (5700x reference) at iter 1 but SHRINKS, has two benign
# 4-iter frozen windows at 2x and 0.05x reference, and converges at iter 28.
# The monitor must NEVER declare this stalled.
# ---------------------------------------------------------------------------
_N10_REFERENCE = 1.69e10
# (lower_bound, best_upper_bound) per iteration, transcribed from §3.8. Where
# the trace gives ranges/"steady close", we fill a monotone-shrinking incumbent
# consistent with the reported endpoints; the frozen windows are byte-exact.
_N10_SEQUENCE: list[tuple[float, float]] = [
    (-1.80e12, 9.62e13),  # iter 1  gap 1.019  blow-up but shrinking
    (-1.08e12, 1.19e13),  # iter 2
    (-1.57e11, 3.04e12),  # iter 3
    (-2.01e10, 3.27e10),  # iter 4  ┐ frozen 4 iters (4-7) at ~1.9x ref
    (-1.24e10, 3.27e10),  # iter 5  │
    (-1.37e09, 3.27e10),  # iter 6  │
    (-1.20e09, 3.27e10),  # iter 7  ┘
    (-1.18e09, 8.68e08),  # iter 8  ┐ frozen 4 iters (8-11) at ~0.05x ref
    (-1.17e09, 8.68e08),  # iter 9  │
    (-1.16e09, 8.68e08),  # iter 10 │
    (-1.15e09, 8.68e08),  # iter 11 ┘
    (-9.00e08, 2.32e08),  # iter 12 ┐ frozen 3 iters (12-14) <1x ref
    (-6.00e08, 2.32e08),  # iter 13 │
    (-3.00e08, 2.32e08),  # iter 14 ┘
    (+1.71e08, 2.32e08),  # iter 15 LB snaps positive, gap collapses
    (+1.90e08, 2.28e08),  # iter 16 steady close ...
    (+2.00e08, 2.24e08),  # iter 17
    (+2.05e08, 2.20e08),  # iter 18
    (+2.08e08, 2.16e08),  # iter 19
    (+2.10e08, 2.13e08),  # iter 20
    (+2.11e08, 2.12e08),  # iter 21
    (+2.115e08, 2.118e08),  # iter 22
    (+2.117e08, 2.117e08),  # iter 23  converged (gap ~0)
]


def test_n10_converging_trace_never_stalls():
    """The hardest converging shape (early 5700x blow-up, two 4-iter benign
    frozen windows, non-monotone gap) must never trip the guard."""
    mon = StallMonitor(_N10_REFERENCE)  # library defaults (K=8, blowup_mult=5)
    verdicts = [mon.update(lb, ub) for lb, ub in _N10_SEQUENCE]
    assert not any(v.stalled for v in verdicts), (
        "N=10 converging trace was flagged as a stall at iters "
        + str([i + 1 for i, v in enumerate(verdicts) if v.stalled])
    )
    # The two benign frozen windows are only 4 iters (< K=8), so they never
    # satisfy the incumbent-frozen leg even in isolation.
    assert all(isinstance(v, StallVerdict) for v in verdicts)


def test_frozen_blowup_stall_fires_at_window():
    """UB frozen far above the reference for a full window (gap ~1) must be
    flagged — and exactly once the window fills, not before."""
    ref = 1.0e6
    ub = 100.0 * ref  # 100x reference: well past blowup_mult=5
    lb = -ub  # gap ~= 2 (>> gap_floor)
    mon = StallMonitor(ref)  # K=8
    verdicts = [mon.update(lb, ub) for _ in range(10)]
    # Window fills at the 8th update (index 7); the incumbent has been frozen
    # for a full window only from then on.
    assert not any(v.stalled for v in verdicts[:7]), "stall declared before the window filled"
    assert verdicts[7].stalled, "stall not declared once the window filled"
    assert verdicts[7].kind == "frozen-blowup"
    assert all(v.stalled for v in verdicts[7:])


def test_plateau_then_resume_never_stalls():
    """UB flat for K-1 iters then a real improvement — proves the K-window is
    a *trailing* window and a resume clears it (never a stall)."""
    ref = 1.0e6
    high = 100.0 * ref
    mon = StallMonitor(ref, window=8)
    # 7 frozen iters (K-1) at the blown-up level ...
    early = [mon.update(-high, high) for _ in range(7)]
    assert not any(v.stalled for v in early)
    # ... then the incumbent drops sharply (a real improvement) on iter 8.
    resume = mon.update(-high, high * 0.5)
    assert not resume.stalled, "a real improvement was mis-flagged as a stall"
    # And a couple more improving iters stay clear.
    assert not mon.update(-high, high * 0.25).stalled
    assert not mon.update(-high, high * 0.1).stalled


def test_scale_stability_of_min_rel():
    """A 1e-5 relative UB improvement is NOT 'flat'; a 1e-8 one IS."""
    ref = 1.0e6
    base = 100.0 * ref  # blown up + high gap so only the frozen leg decides

    # 1e-5 relative improvement each step over a full window: NOT frozen.
    mon_a = StallMonitor(ref, window=8)
    ub = base
    va = None
    for _ in range(9):
        va = mon_a.update(-base, ub)
        ub *= 1.0 - 1e-5
    assert not va.stalled, "1e-5 relative improvement wrongly counted as frozen"

    # 1e-8 relative improvement each step: BELOW min_rel ⇒ frozen ⇒ stall.
    mon_b = StallMonitor(ref, window=8)
    ub = base
    vb = None
    for _ in range(9):
        vb = mon_b.update(-base, ub)
        ub *= 1.0 - 1e-8
    assert vb.stalled, "1e-8 relative improvement should count as frozen"


def test_near_tolerance_flat_never_stalls():
    """Gap below the floor with a frozen UB is a (near-)converged run, not a
    stall — the gap-floor leg must veto it."""
    ref = 1.0e6
    # UB just above LB ⇒ tiny gap; keep UB blown up so only the gap leg vetoes.
    ub = 100.0 * ref
    lb = ub * (1.0 - 1e-4)  # gap ~ 1e-4 < gap_floor 0.02
    mon = StallMonitor(ref, window=8)
    verdicts = [mon.update(lb, ub) for _ in range(10)]
    assert not any(v.stalled for v in verdicts)


def test_blowup_gate_vetoes_frozen_at_sane_magnitude():
    """A frozen UB with a high gap but BELOW blowup_mult x reference is not a
    stall — the incumbent has fallen to a sane magnitude."""
    ref = 1.0e6
    ub = 2.0 * ref  # 2x reference: below blowup_mult=5
    lb = -ub  # gap ~ 2 (high) so only the blow-up leg vetoes
    mon = StallMonitor(ref, window=8)
    verdicts = [mon.update(lb, ub) for _ in range(10)]
    assert not any(v.stalled for v in verdicts), (
        "frozen UB at 2x reference (below blowup gate) wrongly flagged"
    )
    # Sanity: exactly at 5x is still not strictly greater than the gate.
    mon_eq = StallMonitor(ref, window=8, blowup_mult=5.0)
    ub_eq = 5.0 * ref
    v_eq = [mon_eq.update(-ub_eq, ub_eq) for _ in range(10)]
    assert not any(v.stalled for v in v_eq)
    # Just above the gate does fire.
    mon_gt = StallMonitor(ref, window=8, blowup_mult=5.0)
    ub_gt = 5.001 * ref
    v_gt = [mon_gt.update(-ub_gt, ub_gt) for _ in range(10)]
    assert v_gt[7].stalled


def test_window_boundary_off_by_one():
    """Exactly at ``it == K`` fires; ``it == K-1`` does not (window off-by-one).

    With ``window=K`` the incumbent-frozen test needs K observations before it
    can compare across a full trailing window, so the K-th update (1-indexed) is
    the earliest a stall can be declared.
    """
    ref = 1.0e6
    ub = 100.0 * ref
    lb = -ub
    for k in (3, 5, 8):
        mon = StallMonitor(ref, window=k)
        verdicts = [mon.update(lb, ub) for _ in range(k + 2)]
        # 1-indexed iteration k is index k-1.
        assert not verdicts[k - 2].stalled, f"K={k}: fired one iter too early"
        assert verdicts[k - 1].stalled, f"K={k}: did not fire at exactly K"


def test_reference_scale_zero_does_not_divide_by_zero():
    """A zero reference must not blow up the blow-up gate (guarded max(1,·))."""
    mon = StallMonitor(0.0, window=8)
    ub = 1.0e9
    lb = -ub
    verdicts = [mon.update(lb, ub) for _ in range(10)]
    # ub is way above max(1, 0)=1 x blowup_mult, frozen, high gap ⇒ stalls.
    assert verdicts[7].stalled
    assert all(v.blowup_ratio == ub for v in verdicts)  # ub / max(1,0)
