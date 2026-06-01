"""Byte-identity parity for the bounded coefficient-walk LHS fallback
(Phase D-5 step 2).

Background
----------
:func:`polar_high.autoscale._ranges._ranges_via_streaming` reads
``(min, max)`` of ``|post-Layer-2 LHS coef|`` per constraint term to pick
the autoscale exponents.  The inline block-COO fast path (Phase D-1) bounds
the block-evaluable ``Var × Param-chain`` terms; anything it DECLINES (a
bare ``Var`` LHS with no Param chain — block-COO classify requires a
non-empty Param list — or a map-Where / Sum-combining / sparse shape the
inline arm rejects) used to drop to a materialising ``_collect_streaming``.
Over the per-family row cap that collect was SKIPPED (loud) and the family's
coefficient range was dropped from the readout — the visible coverage gap
the DES ``maxToSink`` family hits (a bare-``Var`` LHS term).

Phase D-5 step 2 replaces that DECLINE→materialise/skip fallback with
DECLINE→:func:`bounded_coefficient_walk` (+ :class:`MinMaxAbsReducer`),
which reconstructs the IDENTICAL ``(_rid, col_id, coef)`` stream in bounded
batches via the block-COO builders' always-correct backstops and applies
the SAME side-vector scale + the SAME ``_reduce_abs`` finite/non-zero mask.
Because the walk is byte-identical to the whole-collect for any batch size,
the resulting ``RangeReport`` must match a run with the cap DISABLED
(``POLAR_HIGH_RANGES_MAX_FAMILY_ROWS=0`` — which forces the materialising
collect on the declined shapes) BYTE-for-BYTE, AND no ``ranges-stream SKIP``
line may be emitted for the declined families (they are now bounded, never
skipped).
"""

from __future__ import annotations

import io
import itertools
import os
import sys

import numpy as np
import polars as pl

from polar_high.autoscale import ScalingConfig
from polar_high.autoscale._ranges import _ranges_via_streaming
from polar_high.engine import Param, Problem, Sum, Where


def _config() -> ScalingConfig:
    return ScalingConfig(
        threshold_decades=9.0,
        user_bound_scale=None,
        report_yaml_path=None,
    )


def _clear_guard() -> None:
    os.environ.pop("POLAR_HIGH_DISABLE_BLOCK_COO", None)
    os.environ.pop("POLAR_HIGH_ENABLE_BLOCK_COO", None)
    os.environ.pop("POLAR_HIGH_BLOCK_COO_MIN_DENSE", None)
    os.environ.pop("POLAR_HIGH_BLOCK_COO_PROFILE", None)
    os.environ.pop("POLAR_HIGH_RANGES_PROFILE", None)
    os.environ.pop("POLAR_HIGH_RANGES_MAX_FAMILY_ROWS", None)


def _build_problem() -> Problem:
    """Problem reproducing the DES ``maxToSink`` family member shapes that
    DECLINE the inline block-COO LHS path, plus a map-Where + a
    Sum-combining shape, so all three declined routes are exercised:

    * ``mts`` — a BARE ``Var(p,d,t)`` LHS term (no Param chain): the inline
      block-COO classifier requires a non-empty Param list, so it DECLINES;
      this is the ``maxToSink`` LHS shape currently SKIPPED over the cap.
    * ``map`` — a ``Sum(Where(v*P_unit, map)*P_step, over=("p","s"))`` LHS
      (relabel arm via a map-introduced ``n``).  This shape FIRES the inline
      block-COO relabel fast path (bit-identical to polars' group_by), so it
      does NOT reach the walk fallback — included to confirm the walk does
      not disturb the inline route, and that the LOW cap never skips it.
    * ``comb`` — a ``Sum(Where(v*P_unit, map_ph)*P_step, over=("h",))`` LHS
      where the reduced dim ``h`` is NOT a Var dim (genuine combining): the
      inline arm declines the combining branch; the walk's whole-spine
      combining path handles it byte-identically.

    Frames are built in ``itertools.product`` order so the dense_axes sort
    contract holds.  Side vectors are installed with a multi-decade spread.
    """
    p_idx = [0, 1, 2]
    s_idx = ["s0", "s1"]
    d_idx = [10, 11]
    t_idx = [100, 101, 102]

    prob = Problem(dense_axes=("d", "t"))

    # ---- Family 1: BARE Var(p,d,t) LHS — the maxToSink shape.  The RHS is a
    # multi-Param chain (profile-like) bounded by D-2; the LHS bare-Var term
    # is what the inline block-COO path declines and the cap used to skip.
    cells = list(itertools.product(p_idx, d_idx, t_idx))
    mts_over = pl.DataFrame(
        {
            "p": [c[0] for c in cells],
            "d": [c[1] for c in cells],
            "t": [c[2] for c in cells],
        }
    )
    vm = prob.add_var("vm", ("p", "d", "t"), mts_over, lower=0.0, upper=1e6)
    dt_rows = list(itertools.product(d_idx, t_idx))
    profile = Param(
        ("d", "t"),
        pl.DataFrame(
            {
                "d": [r[0] for r in dt_rows],
                "t": [r[1] for r in dt_rows],
                "value": np.linspace(0.2, 4.0, len(dt_rows)),
            }
        ),
        name="profile",
    )
    availability = Param(
        ("p",),
        pl.DataFrame({"p": p_idx, "value": [10.0, 250.0, 3000.0]}),
        name="availability",
    )
    prob.add_cstr(
        "mts",
        over=mts_over,
        sense="<=",
        lhs_terms={"lhs": vm * 7.5},
        rhs_terms={"rhs": profile * availability},
    )

    # ---- Family 2: nodeBalance-shaped Sum (relabel via map-introduced n).
    rows = list(itertools.product(p_idx, s_idx, d_idx, t_idx))
    var_index = pl.DataFrame(
        {
            "p": [r[0] for r in rows],
            "s": [r[1] for r in rows],
            "d": [r[2] for r in rows],
            "t": [r[3] for r in rows],
        }
    )
    v = prob.add_var("v", ("p", "s", "d", "t"), var_index, lower=0.0, upper=1e6)
    P_unit = Param(
        ("p",),
        pl.DataFrame({"p": p_idx, "value": [2.0, 30.0, 400.0]}),
        name="P_unit",
    )
    P_step = Param(
        ("d", "t"),
        pl.DataFrame(
            {
                "d": [r[0] for r in dt_rows],
                "t": [r[1] for r in dt_rows],
                "value": np.linspace(0.5, 1.5, len(dt_rows)),
            }
        ),
        name="P_step",
    )
    map_rows = list(itertools.product(p_idx, s_idx))
    map_to_n = pl.DataFrame(
        {
            "p": [r[0] for r in map_rows],
            "s": [r[1] for r in map_rows],
            "n": [f"n{(r[0] + (0 if r[1] == 's0' else 1)) % 2}" for r in map_rows],
        }
    )
    nb = Sum(Where(v * P_unit, map_to_n) * P_step, over=("p", "s"))
    nb_over = nb.terms[0].frame.select(list(nb.terms[0].dims)).unique().sort(["n", "d", "t"])
    prob.add_cstr(
        "map",
        over=nb_over,
        sense="<=",
        lhs_terms={"lhs": nb},
        rhs_terms={"rhs": 0.0},
    )

    # ---- Family 3: Sum-combining (reduced dim ``h`` is NOT a Var dim).
    comb_cells = list(itertools.product(p_idx, d_idx, t_idx))
    comb_index = pl.DataFrame(
        {
            "p": [c[0] for c in comb_cells],
            "d": [c[1] for c in comb_cells],
            "t": [c[2] for c in comb_cells],
        }
    )
    vc = prob.add_var("vc", ("p", "d", "t"), comb_index, lower=0.0, upper=1e6)
    map_ph = pl.DataFrame(
        {
            "p": [p for p in p_idx for _ in range(2)],
            "h": [h for _ in p_idx for h in ("h0", "h1")],
        }
    )
    P_unit_c = Param(
        ("p",),
        pl.DataFrame({"p": p_idx, "value": [2.0, 5.0, 11.0]}),
        name="P_unit_c",
    )
    comb = Sum(Where(vc * P_unit_c, map_ph) * P_step, over=("h",))
    comb_over = comb.terms[0].frame.select(list(comb.terms[0].dims)).unique().sort(["p", "d", "t"])
    prob.add_cstr(
        "comb",
        over=comb_over,
        sense="<=",
        lhs_terms={"lhs": comb},
        rhs_terms={"rhs": 0.0},
    )

    prob.set_objective(Sum(vm) + Sum(v) + Sum(vc), sense="min")
    return prob


def _install_side_vectors(prob: Problem) -> None:
    """Install Layer-2 row/col factors with a multi-decade spread so the
    range readout's side-vector multiply is genuinely exercised."""
    n_rows = sum(over.height for _c, _p, over in prob._cstrs)
    n_cols = int(prob._next_col)
    rf = np.array([10.0 ** ((i % 5) - 2) for i in range(n_rows)], dtype=np.float64)
    cf = np.array([10.0 ** ((i % 7) - 3) for i in range(n_cols)], dtype=np.float64)
    prob._layer2_row_factor = rf
    prob._layer2_col_factor = cf


def _report_tuple(rep) -> tuple:
    """Flatten a RangeReport into a comparable tuple, NaN-safe via repr."""

    def _pair(x):
        return (repr(x[0]), repr(x[1]))

    return (
        _pair(rep.matrix),
        _pair(rep.cost),
        _pair(rep.bound),
        _pair(rep.rhs),
        repr(rep.cross_group_max_ratio),
        rep.trigger,
    )


def test_walk_fallback_byte_identical_to_cap_disabled() -> None:
    """With the bounded walk wired (and the cap set LOW to force the
    DECLINE+previously-skip path on every family), the RangeReport must be
    BYTE-IDENTICAL to a run with the cap DISABLED (which forces the
    materialising collect on the same declined shapes).  i.e. the bounded
    walk == the full collect, AND no ``ranges-stream SKIP`` line is emitted.

    Both runs use the SAME ``Problem`` instance (same constraint ``over`` row
    ordering ⇒ same ``base_row + _rid`` row-factor index, same ``col_id``
    col-factor index) so only the cap-gated evaluation route differs."""
    cfg = _config()
    _clear_guard()
    # Min-dense gate down so any block-evaluable arm in the *inline* fast
    # path still has the option to fire; the bare-Var family declines it
    # regardless (no Param chain), exercising the walk fallback.
    os.environ["POLAR_HIGH_BLOCK_COO_MIN_DENSE"] = "1"
    try:
        prob = _build_problem()
        _install_side_vectors(prob)

        # (1) Walk-wired run with a LOW cap — forces DECLINE+walk on the
        # declined shapes (any family over 10 rows).  Capture stderr to
        # assert NO SKIP line fires (the walk always bounds).
        buf = io.StringIO()
        old = sys.stderr
        os.environ["POLAR_HIGH_RANGES_MAX_FAMILY_ROWS"] = "10"
        try:
            sys.stderr = buf
            rep_walk = _ranges_via_streaming(prob, cfg)
        finally:
            sys.stderr = old
        skip_lines = buf.getvalue().count("ranges-stream SKIP")

        # (2) Cap DISABLED — forces the materialising collect on the declined
        # shapes (the reference whole-collect range).
        os.environ["POLAR_HIGH_RANGES_MAX_FAMILY_ROWS"] = "0"
        rep_full = _ranges_via_streaming(prob, cfg)
    finally:
        _clear_guard()

    assert skip_lines == 0, (
        "the bounded walk fallback must bound every declined family, so NO "
        f"'ranges-stream SKIP' line may be emitted; saw {skip_lines}.\n"
        f"stderr:\n{buf.getvalue()}"
    )
    assert _report_tuple(rep_walk) == _report_tuple(rep_full), (
        "bounded-walk fallback diverged from the cap-disabled whole-collect "
        "readout:\n"
        f"  WALK = {_report_tuple(rep_walk)}\n"
        f"  FULL = {_report_tuple(rep_full)}"
    )


def test_walk_fallback_no_skip_lines_for_declined_families() -> None:
    """Standalone assertion that the LOW-cap walk run emits ZERO SKIP lines
    for the bare-Var / map-Where / Sum-combining families — the declined
    shapes are now bounded by the walk, not skipped."""
    cfg = _config()
    _clear_guard()
    os.environ["POLAR_HIGH_BLOCK_COO_MIN_DENSE"] = "1"
    os.environ["POLAR_HIGH_RANGES_MAX_FAMILY_ROWS"] = "10"
    buf = io.StringIO()
    old = sys.stderr
    try:
        sys.stderr = buf
        prob = _build_problem()
        _install_side_vectors(prob)
        _ranges_via_streaming(prob, cfg)
    finally:
        sys.stderr = old
        _clear_guard()

    out = buf.getvalue()
    assert "ranges-stream SKIP" not in out, (
        "no family should be skipped — all declined LHS shapes route through "
        f"the bounded walk now.\nstderr:\n{out}"
    )


def test_walk_fallback_profile_signal_fires() -> None:
    """The bounded-walk LHS fallback must actually FIRE (proven via the
    ``coef_walk=1`` profile signal) for the declined families, with the cap
    set low to force the fallback route."""
    cfg = _config()
    _clear_guard()
    os.environ["POLAR_HIGH_BLOCK_COO_MIN_DENSE"] = "1"
    os.environ["POLAR_HIGH_RANGES_PROFILE"] = "1"
    os.environ["POLAR_HIGH_RANGES_MAX_FAMILY_ROWS"] = "10"
    buf = io.StringIO()
    old = sys.stderr
    try:
        sys.stderr = buf
        prob = _build_problem()
        _install_side_vectors(prob)
        _ranges_via_streaming(prob, cfg)
    finally:
        sys.stderr = old
        _clear_guard()

    out = buf.getvalue()
    n_fired = out.count("coef_walk=1")
    # bare-Var (mts) + Sum-combining (comb) = 2 declined LHS terms route
    # through the walk fallback.  The map-Where ``Sum`` (relabel arm) is
    # bit-identical to polars' group_by, so the INLINE block-COO fast path
    # (Phase D-1) fires for it (``block_coo=1``) and it never reaches the
    # fallback — exactly as intended (the walk replaces ONLY the
    # DECLINE→materialise/skip route, not the inline fast path).
    assert n_fired == 2, (
        f"expected the bounded-walk LHS fallback to fire on the 2 declined "
        f"LHS terms (bare-Var + combining); profile reported {n_fired}.\n"
        f"profile:\n{out}"
    )
    # And confirm the map-Where relabel stayed on the inline fast path.
    assert out.count("block_coo=1") == 1, (
        "the map-Where relabel Sum must stay on the inline block-COO fast "
        f"path (block_coo=1, count=1); profile:\n{out}"
    )


# ---------------------------------------------------------------------------
# RHS decline branch (Phase D-5 step 2b): a Var-LESS composite Param chain
# whose ``over`` grid is NOT dense-complete declines the inline D-2 positional
# fast path; the bounded Var-less walk replaces the materialising / cap-skip
# collect, byte-identically.


def _build_rhs_decline_problem() -> Problem:
    """A ``profile_flow_upper_limit``-shaped family whose RHS is a composite
    3-Param chain, but with a SPARSE dense ``(d, t)`` atomic so the over grid
    is NOT dense-complete — the inline D-2 positional builder DECLINES and the
    RHS routes through the Var-less coefficient walk.  A bare-Var LHS keeps
    the family well-formed."""
    p_idx = [0, 1, 2]
    s_idx = ["s0", "s1"]
    d_idx = [10, 11]
    t_idx = [100, 101, 102]

    prob = Problem(dense_axes=("d", "t"))
    rows = list(itertools.product(p_idx, s_idx, d_idx, t_idx))
    over = pl.DataFrame(
        {
            "p": [r[0] for r in rows],
            "s": [r[1] for r in rows],
            "d": [r[2] for r in rows],
            "t": [r[3] for r in rows],
        }
    )
    v = prob.add_var("v", ("p", "s", "d", "t"), over, lower=0.0, upper=1e6)

    pdt = list(itertools.product(p_idx, d_idx, t_idx))
    Pprofile = Param(
        ("p", "d", "t"),
        pl.DataFrame(
            {
                "p": [c[0] for c in pdt],
                "d": [c[1] for c in pdt],
                "t": [c[2] for c in pdt],
                "value": np.linspace(1e-3, 5e2, len(pdt)),
            }
        ),
        name="Pprofile",
    )
    ps = list(itertools.product(p_idx, s_idx))
    Pcount = Param(
        ("p", "s"),
        pl.DataFrame(
            {
                "p": [c[0] for c in ps],
                "s": [c[1] for c in ps],
                "value": np.linspace(2.0, 4e3, len(ps)),
            }
        ),
        name="Pcount",
    )
    # SPARSE Pavail on (d, t): drop a cell so the dense completeness guard
    # fails ⇒ the inline D-2 positional builder declines ⇒ the Var-less walk
    # (prune-down backstop) fires.
    dt = list(itertools.product(d_idx, t_idx))
    dt_sparse = [c for i, c in enumerate(dt) if i != 2]
    Pavail = Param(
        ("d", "t"),
        pl.DataFrame(
            {
                "d": [c[0] for c in dt_sparse],
                "t": [c[1] for c in dt_sparse],
                "value": np.linspace(0.4, 0.95, len(dt_sparse)),
            }
        ),
        name="Pavail",
    )
    prob.add_cstr(
        "profile_flow_upper_limit",
        over=over,
        sense="<=",
        lhs_terms={"lhs": v},
        rhs_terms={"rhs": Pprofile * Pcount * Pavail},
    )
    prob.set_objective(Sum(v), sense="min")
    return prob


def test_rhs_walk_fallback_byte_identical_to_cap_disabled() -> None:
    """The RHS Var-less coefficient walk (forced by a LOW cap on the declined
    chain) must produce a RangeReport BYTE-IDENTICAL to the cap-disabled run
    (which forces the materialising ``over ⋈ rhs`` collect), AND emit no
    ``ranges-stream SKIP`` line for the RHS family."""
    cfg = _config()
    _clear_guard()
    try:
        prob = _build_rhs_decline_problem()
        _install_side_vectors(prob)

        buf = io.StringIO()
        old = sys.stderr
        os.environ["POLAR_HIGH_RANGES_MAX_FAMILY_ROWS"] = "10"
        try:
            sys.stderr = buf
            rep_walk = _ranges_via_streaming(prob, cfg)
        finally:
            sys.stderr = old
        skip_lines = buf.getvalue().count("ranges-stream SKIP")

        os.environ["POLAR_HIGH_RANGES_MAX_FAMILY_ROWS"] = "0"
        rep_full = _ranges_via_streaming(prob, cfg)
    finally:
        _clear_guard()

    assert skip_lines == 0, (
        "the Var-less RHS walk must bound the declined chain, so NO "
        f"'ranges-stream SKIP' line may be emitted; saw {skip_lines}.\n"
        f"stderr:\n{buf.getvalue()}"
    )
    assert _report_tuple(rep_walk) == _report_tuple(rep_full), (
        "bounded RHS-walk fallback diverged from the cap-disabled "
        "whole-collect readout:\n"
        f"  WALK = {_report_tuple(rep_walk)}\n"
        f"  FULL = {_report_tuple(rep_full)}"
    )


def test_rhs_walk_fallback_no_skip_and_fires() -> None:
    """With a LOW cap the declined RHS chain must route through the walk
    (``rhs_walk=1`` profile signal) and emit ZERO ``ranges-stream SKIP``
    lines."""
    cfg = _config()
    _clear_guard()
    os.environ["POLAR_HIGH_RANGES_PROFILE"] = "1"
    os.environ["POLAR_HIGH_RANGES_MAX_FAMILY_ROWS"] = "10"
    buf = io.StringIO()
    old = sys.stderr
    try:
        sys.stderr = buf
        prob = _build_rhs_decline_problem()
        _install_side_vectors(prob)
        _ranges_via_streaming(prob, cfg)
    finally:
        sys.stderr = old
        _clear_guard()

    out = buf.getvalue()
    assert "ranges-stream SKIP" not in out, (
        f"the declined RHS chain must be bounded by the walk, never skipped.\nstderr:\n{out}"
    )
    assert out.count("rhs_walk=1") == 1, (
        "expected the Var-less RHS walk to fire once on the declined "
        f"composite chain; profile:\n{out}"
    )
