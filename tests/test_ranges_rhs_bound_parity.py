"""Byte-identity + bounded-memory parity for the bounded RHS Param-chain
range readout (Phase D-2).

Background
----------
:func:`polar_high.autoscale._ranges._ranges_via_streaming` reads
``(min, max)`` of ``|post-Layer-2 rhs_coef|`` per constraint family to pick
the autoscale exponents.  The dominant FlexTool DES autoscale spike is the
RHS Param-product readout — NOT the LHS: the multi-Param RHS chain (DES's
``profile_flow_upper_limit`` RHS = ``profile · existing_count ·
availability``, a 3-Param chain over ``(p, source, sink, d, t)``) is
MATERIALISED by the merged-lazy left-join because the polars streaming
engine cannot push the row-key semi-join into a 3+ Param product.

Phase D-2 bounds this the same way block-COO bounds the LHS: align the RHS
Param factors positionally on the constraint's pre-sorted dense-trailing
``over`` grid (no Var/col_id — the grid itself is the spine), multiply in
``rhs._sources`` order seeded with ``rhs._value_scalar``, and reduce to
``(min, max)`` of ``|rhs_coef × |_l2_rf[base_row + _rid]||`` — never
materialising the full product.

Because the bounded positional product reproduces the prune-down /
merged-lazy IEEE-double op sequence value-for-value (same atomics, same
multiply order), the row-factor index is the identical ``base_row + _rid``
(``_rid`` == the over grid's positional row index), and the reduction is
the identical ``_reduce_abs``, the reported ``RangeReport`` must be
BYTE-IDENTICAL between the bounded RHS path (default) and the
merged-lazy streaming path (``POLAR_HIGH_DISABLE_BLOCK_COO=1`` — the same
lever that gates the bounded RHS builder).  No tolerance.
"""

from __future__ import annotations

import itertools
import os

import numpy as np
import polars as pl

from polar_high.autoscale import ScalingConfig
from polar_high.autoscale._ranges import _ranges_via_streaming
from polar_high.engine import Param, Problem, Sum


def _config() -> ScalingConfig:
    return ScalingConfig(
        threshold_decades=9.0,
        user_bound_scale=None,
        report_yaml_path=None,
    )


def _clear_guard() -> None:
    os.environ.pop("POLAR_HIGH_DISABLE_BLOCK_COO", None)
    os.environ.pop("POLAR_HIGH_DISABLE_PRUNE_DOWN", None)
    os.environ.pop("POLAR_HIGH_BLOCK_COO_PROFILE", None)
    os.environ.pop("POLAR_HIGH_RANGES_PROFILE", None)


def _build_problem() -> Problem:
    """Problem with a ``profile_flow_upper_limit``-shaped constraint:
    RHS is a 3-Param chain ``Pprofile · Pcount · Pavail`` over a dense
    ``(p, s, d, t)`` ``over`` grid, declared dense suffix ``(d, t)``.

    The three RHS atomics exercise all three positional-alignment cases of
    the bounded builder:

    * ``Pprofile(p, d, t)`` — lead-subset (``p``) + dense (``d``, ``t``);
    * ``Pcount(p, s)``      — lead-only (no dense axis);
    * ``Pavail(d, t)``      — dense-only.

    A Var LHS keeps the family well-formed.  Frames are built in
    ``itertools.product`` order so the dense_axes sort contract holds.
    """
    p_idx = [0, 1, 2]
    s_idx = ["s0", "s1"]
    d_idx = [10, 11]
    t_idx = [100, 101, 102, 103]

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

    # ---- RHS atomics (multi-decade spreads so the readout is exercised).
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
    dt = list(itertools.product(d_idx, t_idx))
    Pavail = Param(
        ("d", "t"),
        pl.DataFrame(
            {
                "d": [c[0] for c in dt],
                "t": [c[1] for c in dt],
                "value": np.linspace(0.4, 0.95, len(dt)),
            }
        ),
        name="Pavail",
    )

    rhs_chain = Pprofile * Pcount * Pavail  # composite, _sources length 3

    prob.add_cstr(
        "profile_flow_upper_limit",
        over=over,
        sense="<=",
        lhs_terms={"lhs": v},
        rhs_terms={"rhs": rhs_chain},
    )

    prob.set_objective(Sum(v), sense="min")
    return prob


def _install_side_vectors(prob: Problem) -> None:
    """Install Layer-2 row/col factors with a multi-decade spread so the
    RHS readout's row-factor multiply is non-trivial."""
    n_rows = sum(over.height for _c, _p, over in prob._cstrs)
    n_cols = int(prob._next_col)
    rf = np.array(
        [10.0 ** ((i % 5) - 2) for i in range(n_rows)], dtype=np.float64
    )
    cf = np.array(
        [10.0 ** ((i % 7) - 3) for i in range(n_cols)], dtype=np.float64
    )
    prob._layer2_row_factor = rf
    prob._layer2_col_factor = cf


def _report_tuple(rep) -> tuple:
    """Flatten a RangeReport into a comparable tuple of raw floats / bools,
    NaN-safe (NaN compared by ``repr``)."""
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


def test_ranges_rhs_bound_byte_identical() -> None:
    """The RangeReport from the bounded RHS Param-chain path (default) must
    be BYTE-IDENTICAL to the merged-lazy streaming path
    (``POLAR_HIGH_DISABLE_BLOCK_COO=1`` declines the bounded RHS builder).

    Same ``Problem`` instance run twice — only the env-gated RHS evaluation
    path differs, so the constraint ``over`` row order (hence the
    ``base_row + _rid`` row-factor index) is identical between runs."""
    cfg = _config()

    _clear_guard()
    try:
        prob = _build_problem()
        _install_side_vectors(prob)

        # Bounded RHS path (default ON).
        rep_on = _ranges_via_streaming(prob, cfg)

        # Merged-lazy streaming path (bounded RHS declined) — SAME Problem,
        # SAME side vectors, only the code path differs.
        os.environ["POLAR_HIGH_DISABLE_BLOCK_COO"] = "1"
        rep_off = _ranges_via_streaming(prob, cfg)
    finally:
        _clear_guard()

    assert _report_tuple(rep_on) == _report_tuple(rep_off), (
        "bounded RHS range readout diverged from the merged-lazy path:\n"
        f"  ON  = {_report_tuple(rep_on)}\n"
        f"  OFF = {_report_tuple(rep_off)}"
    )


def test_ranges_rhs_bound_branch_fired() -> None:
    """The bounded RHS branch must actually FIRE for the multi-Param chain
    family (proven via the ``rhs_bound=1`` ranges-profile signal and the
    ``path=rhs_positional`` block-COO-profile signal)."""
    import io
    import sys

    cfg = _config()
    _clear_guard()
    os.environ["POLAR_HIGH_RANGES_PROFILE"] = "1"
    os.environ["POLAR_HIGH_BLOCK_COO_PROFILE"] = "1"
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
    assert out.count("rhs_bound=1") == 1, (
        "expected the bounded RHS branch to fire once on the multi-Param "
        f"chain family; profile:\n{out}"
    )
    assert "path=rhs_positional" in out, (
        "expected the rhs_positional block-COO path signal; "
        f"profile:\n{out}"
    )


def _wide_rhs_problem() -> Problem:
    """A WIDE ``profile_flow_upper_limit``-shaped Problem: 40 p × 6 s × 8 d
    × 48 t = 92160 constraint rows, RHS = ``Pprofile · Pcount · Pavail``."""
    p_idx = list(range(40))
    s_idx = [f"s{i}" for i in range(6)]
    d_idx = list(range(8))
    t_idx = list(range(48))

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
    dt = list(itertools.product(d_idx, t_idx))
    Pavail = Param(
        ("d", "t"),
        pl.DataFrame(
            {
                "d": [c[0] for c in dt],
                "t": [c[1] for c in dt],
                "value": np.linspace(0.4, 0.95, len(dt)),
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
    _install_side_vectors(prob)
    return prob


def test_ranges_rhs_bound_peak_is_bounded() -> None:
    """Memory evidence: the bounded RHS readout NEVER materialises the full
    Param-product value column, whereas the merged-lazy path collects it in
    full — on a WIDE 92160-row RHS chain.

    The merged-lazy path's RHS readout left-joins the 3-Param product onto
    the row index and collects a ``{_rid, value}`` frame of ``row_count``
    rows (the materialised product the polars streaming engine fails to
    push the semi-join into on DES — the spike).  The bounded path computes
    the same ``rhs_coef`` per ``_rid`` with a numpy product over per-atomic
    aligned slices and issues NO ``{_rid, value}`` product collect at all.

    We instrument ``polars.LazyFrame.collect`` and track the largest frame
    collected whose column signature is exactly ``{_rid, value}`` — the RHS
    merged-product collect.  That collect is the materialisation being
    eliminated, cleanly isolated from the LHS Var collect (column signature
    ``{col_id, coef}`` / ``{_rid, col_id, coef}``).  Deterministic and
    scale-independent (no RSS / allocator-retention noise)."""
    orig_collect = pl.LazyFrame.collect
    peak = {"rhs_product_rows": 0}

    def _patched(self, *a, **k):
        df = orig_collect(self, *a, **k)
        if set(df.columns) == {"_rid", "value"} and df.height > peak[
            "rhs_product_rows"
        ]:
            peak["rhs_product_rows"] = df.height
        return df

    cfg = _config()

    # --- Merged-lazy path: must collect the full 92160-row product.
    _clear_guard()
    os.environ["POLAR_HIGH_DISABLE_BLOCK_COO"] = "1"
    pl.LazyFrame.collect = _patched
    try:
        peak["rhs_product_rows"] = 0
        _ranges_via_streaming(_wide_rhs_problem(), cfg)
        merged_rows = peak["rhs_product_rows"]
    finally:
        pl.LazyFrame.collect = orig_collect
        _clear_guard()

    # --- Bounded path: must issue NO {_rid, value} product collect.
    pl.LazyFrame.collect = _patched
    try:
        peak["rhs_product_rows"] = 0
        _ranges_via_streaming(_wide_rhs_problem(), cfg)
        bounded_rows = peak["rhs_product_rows"]
    finally:
        pl.LazyFrame.collect = orig_collect
        _clear_guard()

    assert merged_rows == 92160, (
        "expected the merged-lazy path to materialise the full 92160-row "
        f"RHS product; got {merged_rows}"
    )
    assert bounded_rows == 0, (
        "bounded RHS path must NOT collect the materialised "
        f"{{_rid, value}} product; collected a {bounded_rows}-row frame"
    )
