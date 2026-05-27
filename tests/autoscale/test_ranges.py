"""Layer 1 (detect) self-test.

Constructs a small LP with hand-picked coefficient magnitudes spanning
twelve decades and verifies that :func:`detect_ranges` recovers the
four ranges plus the cross-group ratio, and that ``trigger`` flips at
the documented 9-decade threshold.

The LP is built three ways to exercise every Layer 1 entry point:

* :func:`ranges_from_arrays` — the low-level kernel.
* :func:`ranges_from_streamed` — the polar-high ``Solution`` adapter,
  fed a synthetic dict mirroring ``Solution.streamed_lp_ranges``.
* :func:`detect_ranges` end-to-end on a small polar-high ``Problem``
  (pre-solve, exercising the ``_build_lp_arrays`` fallback path).

The three reports must agree bit-for-bit — same magnitude reduction,
same trigger.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from polar_high.autoscale import (
    ScalingConfig,
    detect_ranges,
    ranges_from_arrays,
    ranges_from_streamed,
)


def _config(threshold: float = 9.0) -> ScalingConfig:
    return ScalingConfig(
        threshold_decades=threshold,
        user_bound_scale=None,
        report_yaml_path=None,
    )


def test_ranges_from_arrays_twelve_decade_span() -> None:
    """A hand-built LP with magnitudes 1e-6 … 1e6 must report the full
    twelve-decade spread and trigger at the default 9-decade threshold."""
    matrix = np.array([1e-6, 1.0, 1e3, 1e6, 0.0, np.inf, np.nan], dtype=np.float64)
    cost = np.array([1e-3, 1e2, 0.0], dtype=np.float64)
    col_lower = np.array([0.0, -np.inf, -2.0], dtype=np.float64)
    col_upper = np.array([np.inf, 5e4, np.inf], dtype=np.float64)
    row_lower = np.array([-np.inf, 10.0], dtype=np.float64)
    row_upper = np.array([1e4, np.inf], dtype=np.float64)

    report = ranges_from_arrays(
        matrix_values=matrix,
        cost=cost,
        col_lower=col_lower,
        col_upper=col_upper,
        row_lower=row_lower,
        row_upper=row_upper,
        config=_config(),
    )

    assert report.matrix == pytest.approx((1e-6, 1e6))
    assert report.cost == pytest.approx((1e-3, 1e2))
    assert report.bound == pytest.approx((2.0, 5e4))
    assert report.rhs == pytest.approx((10.0, 1e4))
    assert report.cross_group_max_ratio == pytest.approx(1e12)
    assert report.trigger is True


def test_ranges_from_arrays_below_threshold_does_not_trigger() -> None:
    """A 3-decade-spread LP must NOT trigger at 9 decades."""
    report = ranges_from_arrays(
        matrix_values=np.array([1.0, 10.0, 100.0]),
        cost=np.array([1.0, 5.0]),
        col_lower=np.array([1.0]),
        col_upper=np.array([100.0]),
        row_lower=np.array([1.0]),
        row_upper=np.array([1000.0]),
        config=_config(),
    )
    assert report.trigger is False
    assert report.cross_group_max_ratio == pytest.approx(1e3)


def test_ranges_from_arrays_empty_group_returns_nan_pair() -> None:
    """A group with no finite non-zero entries must report ``(nan, nan)``
    and be excluded from the cross-group ratio."""
    report = ranges_from_arrays(
        matrix_values=np.array([1.0]),
        cost=np.array([0.0, np.inf]),
        col_lower=np.array([1.0]),
        col_upper=np.array([10.0]),
        row_lower=np.array([2.0]),
        row_upper=np.array([20.0]),
        config=_config(),
    )
    assert math.isnan(report.cost[0]) and math.isnan(report.cost[1])
    assert report.cross_group_max_ratio == pytest.approx(20.0)
    assert report.trigger is False


def test_ranges_from_streamed_matches_arrays() -> None:
    """The streamed-ranges adapter must produce the same report as the
    low-level kernel when given equivalent inputs."""
    streamed = {
        "matrix": (1e-6, 1e6),
        "cost": (1e-3, 1e2),
        "col_bound": (2.0, 5e4),
        "row_bound": (10.0, 1e4),
    }
    via_adapter = ranges_from_streamed(streamed, _config())
    via_arrays = ranges_from_arrays(
        matrix_values=np.array([1e-6, 1e6]),
        cost=np.array([1e-3, 1e2]),
        col_lower=np.array([2.0]),
        col_upper=np.array([5e4]),
        row_lower=np.array([10.0]),
        row_upper=np.array([1e4]),
        config=_config(),
    )
    assert via_adapter == via_arrays


def test_ranges_from_streamed_handles_none_categories() -> None:
    """``None`` entries (polar-high's "no finite non-zero" sentinel)
    must map to ``(nan, nan)`` and be excluded from the cross-group
    ratio."""
    streamed = {
        "matrix": (1.0, 10.0),
        "cost": None,
        "col_bound": None,
        "row_bound": (1.0, 1000.0),
    }
    report = ranges_from_streamed(streamed, _config())
    assert math.isnan(report.cost[0])
    assert math.isnan(report.bound[0])
    assert report.cross_group_max_ratio == pytest.approx(1000.0)
    assert report.trigger is False


def test_threshold_decades_controls_trigger() -> None:
    """Trigger must fire at threshold = N when cross-group spread > 10**N."""
    arrays = dict(
        matrix_values=np.array([1.0]),
        cost=np.array([1.0]),
        col_lower=np.array([1.0]),
        col_upper=np.array([1e4]),
        row_lower=np.array([1.0]),
        row_upper=np.array([1.0]),
    )
    fires = ranges_from_arrays(**arrays, config=_config(3.0))
    doesnt = ranges_from_arrays(**arrays, config=_config(5.0))
    assert fires.trigger is True
    assert doesnt.trigger is False


def test_detect_ranges_on_polar_high_problem() -> None:
    """End-to-end: build a tiny ``polar_high.Problem``, run Layer 1's
    pre-solve fallback path, and verify the four ranges + trigger."""
    from polar_high.engine import Problem

    pb = Problem()
    idx = pl.DataFrame({"i": [0]})
    x = pb.add_var("x", "i", idx, lower=0.0, upper=5e4)
    y = pb.add_var("y", "i", idx, lower=-2.0, upper=float("inf"))
    z = pb.add_var("z", "i", idx, lower=0.0, upper=float("inf"))

    pb.set_objective(1e-3 * x + 1e2 * y + 1.0 * z, sense="min")
    pb.add_cstr(
        "c1",
        over=idx,
        sense="<=",
        lhs_terms={"x": 1e-6 * x, "y": 1.0 * y, "z": 1e3 * z},
        rhs_terms={"k": 1e4},
    )
    pb.add_cstr(
        "c2",
        over=idx,
        sense=">=",
        lhs_terms={"x": 1e6 * x, "y": 1.0 * y},
        rhs_terms={"k": 10.0},
    )

    report = detect_ranges(pb, _config())
    assert report.matrix == pytest.approx((1e-6, 1e6))
    assert report.cost == pytest.approx((1e-3, 1e2))
    assert report.bound == pytest.approx((2.0, 5e4))
    assert report.rhs == pytest.approx((10.0, 1e4))
    assert report.cross_group_max_ratio == pytest.approx(1e12)
    assert report.trigger is True


def test_detect_ranges_rejects_unrecognised_input() -> None:
    """Passing something that's neither a Problem nor a Solution must
    raise — silently degrading would hide wiring bugs."""
    with pytest.raises(TypeError):
        detect_ranges(object(), _config())


# ----------------------------------------------------------------------
# Memory regression — Param-chain LHS term must not explode inside the
# pre-solve range detection path
# ----------------------------------------------------------------------


import os as _os
import threading as _threading
import time as _time

from polar_high.engine import Param, Problem


def _read_vmrss_mb() -> float:
    """VmRSS in MB from /proc/self/status — Linux only."""
    with open("/proc/self/status", "r") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                parts = line.split()
                return float(parts[1]) / 1024.0
    raise RuntimeError("VmRSS line not in /proc/self/status")


def _peak_rss_during(fn, sample_interval: float = 0.003) -> tuple[float, float, object]:
    stop = _threading.Event()
    peak = _read_vmrss_mb()
    baseline = peak

    def sampler() -> None:
        nonlocal peak
        while not stop.is_set():
            r = _read_vmrss_mb()
            if r > peak:
                peak = r
            _time.sleep(sample_interval)

    t = _threading.Thread(target=sampler, daemon=True)
    t.start()
    try:
        result = fn()
    finally:
        stop.set()
        t.join(timeout=1.0)
    return baseline, peak, result


@pytest.mark.skipif(
    not _os.path.exists("/proc/self/status"),
    reason="VmRSS sampling requires /proc (Linux-only)",
)
def test_detect_ranges_param_chain_does_not_explode() -> None:
    """Regression for a per-term blow-up in
    :func:`_ranges_via_passmodel` → ``Problem._build_lp_arrays``.

    Background.  ``_ranges_via_passmodel`` is the pre-solve range
    detector reached when the autoscaler is enabled and the caller
    hands :func:`detect_ranges` a :class:`Problem` (FlexTool's Layer 1
    pre-solve path).  Internally it calls ``Problem._build_lp_arrays``
    which used to bind each LHS term to its row-index via a plain
    ``inner join`` (no semi-join pruning) and then collect every
    family's term plan in parallel via ``pl.collect_all``.  On a real
    9.9M-row LP with Param-product LHS terms shaped
    ``Var * Param₁ * Param₂``, this exploded above 50 GB inside
    Layer 1 detection — *before* the solve / ``write_mps`` ran at
    all.  The fix mirrors the same semi-join + streaming retrofit
    applied to ``write_mps`` in v2.1.2 and to the RHS path in v2.0.

    Reproducer mirrors the ``write_mps`` chain-explosion shape at
    smaller scale (T=200k, K=20, ~0.5% sparse constraint subset).
    Without the fix, ``_build_lp_arrays`` peaks ~500 MB+; with the
    fix, well under 300 MB.
    """
    T = 200_000
    K = 20
    subset_frac = 0.005

    tt = np.repeat(np.arange(T), K)
    kk = np.tile(np.arange(K), T)
    rng = np.random.default_rng(0)
    keep_n = max(1, int(T * K * subset_frac))
    keep_idx = rng.choice(T * K, keep_n, replace=False)
    keep_idx.sort()
    idx_cstr = pl.DataFrame({"t": tt[keep_idx], "k": kk[keep_idx]})

    pb = Problem()
    idx_var = pl.DataFrame({"t": np.arange(T)})
    x = pb.add_var("x", dims=("t",), index=idx_var, lower=0.0, upper=1.0)

    p_tk = Param(
        ("t", "k"),
        pl.DataFrame(
            {"t": tt, "k": kk, "value": np.linspace(1.0, 2.0, T * K)}
        ),
    )
    p_k = Param(
        ("k",),
        pl.DataFrame({"k": np.arange(K), "value": np.linspace(0.5, 1.5, K)}),
    )
    p_t1 = Param(
        ("t",),
        pl.DataFrame({"t": np.arange(T), "value": np.linspace(1.0, 2.0, T)}),
    )

    expr = x * p_tk * p_k * p_t1
    pb.add_cstr(
        "chain", over=idx_cstr, sense="<=",
        lhs_terms={"lhs": expr}, rhs_terms={"r": 5.0},
    )
    pb.set_objective(x, sense="min")

    def do_detect():
        return detect_ranges(pb, _config())

    baseline, peak, report = _peak_rss_during(do_detect)
    delta = peak - baseline
    # Generous ceiling so the test isn't fragile across polars
    # versions / allocator quirks.  Pre-fix peak ~500-700 MB on this
    # shape; post-fix typically <250 MB.  300 MB is the line.
    assert delta < 300.0, (
        f"_build_lp_arrays allocated {delta:.1f} MB during "
        f"detect_ranges on a Var*Param*Param*Param chain.  "
        f"Pre-fix this exceeded 500 MB and on the real DES LP "
        f"(8× this scale) crossed 50 GB.  Did the semi-join + "
        f"streaming retrofit at engine.py:_build_lp_arrays regress?"
    )
    # Sanity: the readout itself returned something coherent.
    assert report is not None
    assert report.matrix is not None


# ----------------------------------------------------------------------
# Layer 2 side-vector readout — focused unit test for
# ``_ranges_via_streaming``
# ----------------------------------------------------------------------


def test_ranges_via_streaming_honors_side_vectors() -> None:
    """Verify that :func:`_ranges_via_streaming` multiplies the
    aggregated magnitudes by ``_layer2_row_factor`` /
    ``_layer2_col_factor`` when they are present on the Problem.

    Background.  Before commit ``d4171fb`` the streaming readout
    walked ``problem._cstrs`` directly without honoring the side
    vectors — that produced a tiny ULP-level drift between
    ``scaling=full`` and ``scaling=solver_only`` on the h2_trade
    end-to-end test (Layer 3 picked different ``user_*_scale``
    exponents because it saw pre-Layer-2 magnitudes).  The patch
    threads the side vectors through the per-emit-site reduce.

    This test exercises that wiring in isolation: build a Problem
    whose unscaled matrix / cost entries are all 1.0, run the
    streaming readout, install constant side vectors, and confirm the
    re-readout produces magnitudes scaled by exactly
    ``|row_factor| * |col_factor|`` (matrix) and ``|col_factor|``
    (objective).
    """
    from polar_high.autoscale._ranges import _ranges_via_streaming
    from polar_high.engine import Problem

    pb = Problem()
    idx = pl.DataFrame({"i": [0, 1, 2]})
    x = pb.add_var("x", "i", idx, lower=0.0, upper=10.0)
    # All LHS coefficients are 1.0 (literal-coefficient term over the
    # row axis); rhs is 1.0; obj coefficient is 1.0.
    pb.add_cstr(
        "c", over=idx, sense="<=", lhs_terms={"x": x}, rhs_terms={"r": 1.0}
    )
    pb.set_objective(x, sense="min")

    n_cols = int(pb._next_col)
    # Three constraint rows; no objective row in the side vector.
    n_rows = 3

    cfg = _config()

    # Baseline: no side vectors.
    base = _ranges_via_streaming(pb, cfg)
    assert base.matrix == pytest.approx((1.0, 1.0))
    assert base.cost == pytest.approx((1.0, 1.0))

    # Install constant side vectors.  Convention (STATE.md):
    # ``_layer2_col_factor`` stores ``1 / cf_math`` (inverse forward);
    # the magnitude effect on |coef * _l2_cf| is just |_l2_cf|, so the
    # readout scales by |_l2_cf| directly regardless of which side of
    # the convention you call "forward".  Pick distinct power-of-two
    # constants per side vector so a missing multiplication is visible.
    rf = 8.0
    cf_inv = 4.0  # this is the stored side vector; readout scales by |this|
    pb._layer2_row_factor = np.full(n_rows, rf, dtype=np.float64)
    pb._layer2_col_factor = np.full(n_cols, cf_inv, dtype=np.float64)

    scaled = _ranges_via_streaming(pb, cfg)

    # Matrix entries: raw 1.0 * rf * cf_inv = 32.0 (constant) → range
    # collapses to (32.0, 32.0).
    assert scaled.matrix == pytest.approx((rf * cf_inv, rf * cf_inv)), (
        f"matrix range expected to scale by row_factor * col_factor = "
        f"{rf * cf_inv}; got {scaled.matrix}"
    )
    # RHS entries: raw 1.0 * rf = 8.0.
    assert scaled.rhs == pytest.approx((rf, rf)), (
        f"rhs range expected to scale by row_factor = {rf}; "
        f"got {scaled.rhs}"
    )
    # Objective: raw 1.0 * cf_inv = 4.0 (no row factor on the cost row,
    # per GLPK convention).
    assert scaled.cost == pytest.approx((cf_inv, cf_inv)), (
        f"cost range expected to scale by col_factor = {cf_inv} only "
        f"(no row factor on objective); got {scaled.cost}"
    )
    # Bounds are not affected by the side vectors in this readout
    # (apply_layer2 mutates Var.lower/upper directly; the side vectors
    # don't enter the bound reduce path).  Unchanged from baseline.
    assert scaled.bound == pytest.approx(base.bound)
