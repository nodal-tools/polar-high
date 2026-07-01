"""Unit tests for ``polar_high.decomposition.InOutStabilizer`` — the generic
in-out separation point picker for Benders-style decomposition drivers.

Pure tests over synthetic ``{col_id -> value}`` point streams; NO solver, NO
domain concepts. Covers every trajectory the plan (§8) calls out:

* ``λ=0`` returns the input dict VERBATIM (byte-parity guard) for every stream;
* never-separates stream → forced out-step the VERY NEXT step, ``λ`` shrinks
  monotonically to a FORCED ``0`` (the livelock guard);
* serious step jumps the centre to the incumbent point;
* first-call passthrough (no centre yet);
* per-region independence (two instances evolve ``λ`` independently);
* reject ``weight >= 1`` / ``weight < 0``;
* determinism (same stream → same outputs).
"""

from __future__ import annotations

import pytest

from polar_high import InOutStabilizer

# ---------------------------------------------------------------------------
# λ = 0 verbatim no-op (the byte-parity guard).
# ---------------------------------------------------------------------------


def test_weight_zero_separation_point_returns_input_verbatim():
    """``λ=0`` ⇒ ``separation_point`` returns the SAME dict object unchanged,
    for every point in a stream, so the off path is byte-identical."""
    stab = InOutStabilizer(weight=0.0)
    # Seed a centre so the passthrough is by the λ=0 rule, not the no-centre one.
    stab.set_centre({0: 5.0, 1: -3.0})
    for pt in ({0: 1.0, 1: 2.0}, {0: 0.0, 1: 0.0}, {0: 9.9, 1: -9.9}):
        out = stab.separation_point(pt)
        assert out is pt  # same object, verbatim


def test_weight_zero_register_never_changes_weight():
    """``λ=0`` stays exactly ``0.0`` across any register outcome."""
    stab = InOutStabilizer(weight=0.0)
    for improved, separated in ((True, True), (False, True), (False, False)):
        stab.register(
            master_point={0: 1.0},
            separated=separated,
            incumbent_point={0: 2.0},
            improved=improved,
        )
        assert stab.weight == 0.0


# ---------------------------------------------------------------------------
# First-call passthrough (no centre yet).
# ---------------------------------------------------------------------------


def test_first_call_passthrough_no_centre():
    """Before any centre is established the separation point is the master
    point verbatim (even with a positive weight)."""
    stab = InOutStabilizer(weight=0.5)
    pt = {0: 4.0, 1: 8.0}
    assert stab.separation_point(pt) is pt


def test_centre_seeded_on_first_register_from_master():
    """With no explicit centre and no incumbent, the first register seeds the
    centre from the master point; the NEXT separation point interpolates."""
    stab = InOutStabilizer(weight=0.5, out_step_every=0)
    # First register: separated, not improved, no incumbent → centre ← master.
    kind = stab.register(
        master_point={0: 10.0},
        separated=True,
        incumbent_point=None,
        improved=False,
    )
    assert kind == "null"
    # centre == {0: 10.0}; now interpolate against a new master vertex.
    sep = stab.separation_point({0: 0.0})
    assert sep[0] == pytest.approx(0.5 * 10.0 + 0.5 * 0.0)


# ---------------------------------------------------------------------------
# Separation point is the correct convex combination.
# ---------------------------------------------------------------------------


def test_separation_point_is_convex_combination():
    stab = InOutStabilizer(weight=0.25, out_step_every=0)
    stab.set_centre({0: 100.0, 1: 0.0})
    sep = stab.separation_point({0: 0.0, 1: 40.0})
    assert sep[0] == pytest.approx(0.25 * 100.0 + 0.75 * 0.0)
    assert sep[1] == pytest.approx(0.25 * 0.0 + 0.75 * 40.0)


def test_separation_point_is_pure():
    """Querying the separation point twice yields the same result and does not
    mutate weight / centre / the forced-out flag."""
    stab = InOutStabilizer(weight=0.5, out_step_every=0)
    stab.set_centre({0: 8.0})
    a = stab.separation_point({0: 2.0})
    b = stab.separation_point({0: 2.0})
    assert a == b
    assert stab.weight == 0.5


def test_missing_centre_column_falls_back_to_master_value():
    """A master column absent from the centre contributes its own value (so
    the combo is still a passthrough on that column)."""
    stab = InOutStabilizer(weight=0.5, out_step_every=0)
    stab.set_centre({0: 10.0})  # column 1 missing from centre
    sep = stab.separation_point({0: 0.0, 1: 6.0})
    assert sep[0] == pytest.approx(5.0)
    assert sep[1] == pytest.approx(6.0)  # centre.get(1, 6.0) → 6.0 ⇒ passthrough


# ---------------------------------------------------------------------------
# Serious step: jump centre to the incumbent.
# ---------------------------------------------------------------------------


def test_serious_step_jumps_centre_to_incumbent():
    stab = InOutStabilizer(weight=0.5, out_step_every=0)
    stab.set_centre({0: 0.0})
    kind = stab.register(
        master_point={0: 20.0},
        separated=True,
        incumbent_point={0: 12.0},
        improved=True,
    )
    assert kind == "serious"
    # Centre jumped to the incumbent {0: 12.0}; interpolate to confirm.
    sep = stab.separation_point({0: 0.0})
    assert sep[0] == pytest.approx(0.5 * 12.0)
    # A serious step preserves the weight (no shrink).
    assert stab.weight == 0.5


def test_serious_step_clears_pending_out():
    """An improvement clears a previously-armed forced out-step (progress
    resumes interior separation)."""
    stab = InOutStabilizer(weight=0.5, out_step_every=0)
    stab.set_centre({0: 0.0})
    # Arm a forced out-step via a no-separation register.
    stab.register(
        master_point={0: 20.0},
        separated=False,
        incumbent_point=None,
        improved=False,
    )
    assert stab.separation_point({0: 20.0}) == {0: 20.0}  # passthrough (armed)
    # Now a serious step should clear the armed out-step (interior resumes).
    # NB the earlier no-separation register shrank the weight 0.5 → 0.25, and a
    # serious step preserves the (shrunk) weight, so the combo uses 0.25.
    stab.register(
        master_point={0: 20.0},
        separated=True,
        incumbent_point={0: 10.0},
        improved=True,
    )
    sep = stab.separation_point({0: 0.0})
    assert sep[0] == pytest.approx(0.25 * 10.0)  # interior again, not passthrough


# ---------------------------------------------------------------------------
# Never-separates stream: forced out-step next step + λ → forced 0 (§1 guard).
# ---------------------------------------------------------------------------


def test_never_separates_forces_out_step_next_step():
    """The moment a cut fails to separate, the NEXT separation point returns
    the master point verbatim (the exact-Benders out-step)."""
    stab = InOutStabilizer(weight=0.5, out_step_every=0)
    stab.set_centre({0: 100.0})
    # Interior separation is active first.
    assert stab.separation_point({0: 0.0}) == {0: 50.0}
    kind = stab.register(
        master_point={0: 0.0},
        separated=False,
        incumbent_point=None,
        improved=False,
    )
    assert kind == "out"
    # NEXT separation point is a verbatim passthrough (λ=0 behaviour for it).
    master = {0: 0.0}
    assert stab.separation_point(master) is master


def test_never_separates_shrinks_weight_monotonically_to_forced_zero():
    """Feed ``separated=False`` forever: the weight shrinks monotonically and
    reaches a FORCED exact ``0.0`` (not a positive floor) — the livelock guard.
    """
    stab = InOutStabilizer(weight=0.5, shrink=0.5, out_step_every=0)
    stab.set_centre({0: 100.0})
    weights = []
    for _ in range(40):
        stab.register(
            master_point={0: 0.0},
            separated=False,
            incumbent_point=None,
            improved=False,
        )
        weights.append(stab.weight)
    # Monotone non-increasing.
    assert all(b <= a for a, b in zip(weights, weights[1:]))
    # Bottoms out at an EXACT 0.0 (forced), then stays there.
    assert stab.weight == 0.0
    assert weights[-1] == 0.0
    # Once zero, separation_point is a verbatim passthrough forever.
    master = {0: 3.0}
    assert stab.separation_point(master) is master


def test_positive_weight_min_still_reaches_forced_zero():
    """A positive ``weight_min`` only slows the descent; the forced-zero snap
    still fires below the eps, so the exact-Benders out-step is guaranteed."""
    stab = InOutStabilizer(weight=0.5, weight_min=1e-6, shrink=0.5, out_step_every=0)
    stab.set_centre({0: 100.0})
    for _ in range(60):
        stab.register(
            master_point={0: 0.0},
            separated=False,
            incumbent_point=None,
            improved=False,
        )
    assert stab.weight == 0.0


# ---------------------------------------------------------------------------
# Null step (separated, no improvement): centre + weight held.
# ---------------------------------------------------------------------------


def test_separated_no_improvement_holds_centre_and_weight():
    stab = InOutStabilizer(weight=0.5, out_step_every=0)
    stab.set_centre({0: 100.0})
    kind = stab.register(
        master_point={0: 0.0},
        separated=True,
        incumbent_point=None,
        improved=False,
    )
    assert kind == "null"
    assert stab.weight == 0.5
    # Centre unchanged: interpolation still uses {0: 100.0}.
    assert stab.separation_point({0: 0.0})[0] == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# Secondary belt-and-braces periodic cap.
# ---------------------------------------------------------------------------


def test_out_step_every_periodic_cap_forces_out_step():
    """With ``out_step_every=3`` and a stream that always separates without
    improving, the periodic cap forces an out-step: the first after 3 interior
    steps, then every 4th (the step immediately following an out-step restarts
    the interior-step counter, so 3 interior steps separate consecutive
    out-steps)."""
    stab = InOutStabilizer(weight=0.5, out_step_every=3)
    stab.set_centre({0: 100.0})
    forced = []
    for _ in range(11):
        stab.register(
            master_point={0: 0.0},
            separated=True,
            incumbent_point=None,
            improved=False,
        )
        master = {0: 0.0}
        forced.append(stab.separation_point(master) is master)
    forced_idx = [i for i, f in enumerate(forced) if f]
    assert forced_idx == [2, 6, 10]
    # There MUST be an out-step at least every out_step_every+1 registers (the
    # belt-and-braces guarantee).
    assert all(b - a <= 4 for a, b in zip(forced_idx, forced_idx[1:]))


def test_out_step_every_zero_disables_periodic_cap():
    stab = InOutStabilizer(weight=0.5, out_step_every=0)
    stab.set_centre({0: 100.0})
    for _ in range(20):
        stab.register(
            master_point={0: 0.0},
            separated=True,
            incumbent_point=None,
            improved=False,
        )
        master = {0: 0.0}
        # Never a forced passthrough — always interior.
        assert stab.separation_point(master) is not master


# ---------------------------------------------------------------------------
# Per-region independence.
# ---------------------------------------------------------------------------


def test_per_region_instances_evolve_independently():
    """Two per-region stabilizers, one always separating and one never, evolve
    their weights independently — the degenerate one is not masked."""
    good = InOutStabilizer(weight=0.5, out_step_every=0)
    bad = InOutStabilizer(weight=0.5, out_step_every=0)
    good.set_centre({0: 10.0})
    bad.set_centre({0: 10.0})
    for _ in range(10):
        good.register(
            master_point={0: 0.0},
            separated=True,
            incumbent_point=None,
            improved=False,
        )
        bad.register(
            master_point={0: 0.0},
            separated=False,
            incumbent_point=None,
            improved=False,
        )
    assert good.weight == 0.5  # untouched — not masked by the good one
    assert bad.weight < 0.5 / 100.0  # shrunk hard toward the forced zero
    # The degenerate region's next step is a forced exact-Benders out-step.
    bad_master = {0: 0.0}
    assert bad.separation_point(bad_master) is bad_master
    # The good region is still interpolating (not forced out).
    good_master = {0: 0.0}
    assert good.separation_point(good_master) is not good_master


# ---------------------------------------------------------------------------
# Constructor validation.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_weight", [1.0, 1.5, 2.0, 100.0])
def test_reject_weight_ge_one(bad_weight):
    with pytest.raises(ValueError, match="weight must be in"):
        InOutStabilizer(weight=bad_weight)


@pytest.mark.parametrize("bad_weight", [-0.1, -1.0, -1e-9])
def test_reject_weight_lt_zero(bad_weight):
    with pytest.raises(ValueError, match="weight must be in"):
        InOutStabilizer(weight=bad_weight)


def test_reject_weight_min_out_of_range():
    with pytest.raises(ValueError, match="weight_min"):
        InOutStabilizer(weight=0.3, weight_min=0.5)  # > weight
    with pytest.raises(ValueError, match="weight_min"):
        InOutStabilizer(weight=0.3, weight_min=-0.1)  # < 0


@pytest.mark.parametrize("bad_shrink", [0.0, 1.0, 1.5, -0.5])
def test_reject_shrink_out_of_range(bad_shrink):
    with pytest.raises(ValueError, match="shrink"):
        InOutStabilizer(weight=0.5, shrink=bad_shrink)


def test_boundary_weight_just_below_one_is_accepted():
    stab = InOutStabilizer(weight=0.999999)
    assert stab.weight == pytest.approx(0.999999)


# ---------------------------------------------------------------------------
# Determinism.
# ---------------------------------------------------------------------------


def _run_stream(seed_centre, stream):
    stab = InOutStabilizer(weight=0.5, shrink=0.5, out_step_every=3)
    stab.set_centre(seed_centre)
    out = []
    for master, separated, incumbent, improved in stream:
        sep = stab.separation_point(master)
        kind = stab.register(
            master_point=master,
            separated=separated,
            incumbent_point=incumbent,
            improved=improved,
        )
        out.append((tuple(sorted(sep.items())), kind, stab.weight))
    return out


def test_determinism_same_stream_same_outputs():
    stream = [
        ({0: 0.0, 1: 5.0}, True, None, False),
        ({0: 1.0, 1: 4.0}, False, None, False),
        ({0: 2.0, 1: 3.0}, True, {0: 1.5, 1: 3.5}, True),
        ({0: 1.5, 1: 3.5}, True, None, False),
        ({0: 1.4, 1: 3.4}, False, None, False),
        ({0: 1.3, 1: 3.3}, True, None, False),
    ]
    a = _run_stream({0: 100.0, 1: 0.0}, stream)
    b = _run_stream({0: 100.0, 1: 0.0}, stream)
    assert a == b
