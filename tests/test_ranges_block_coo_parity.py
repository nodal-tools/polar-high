"""Byte-identity parity for the bounded block-COO LHS range readout
(Phase D-1).

Background
----------
:func:`polar_high.autoscale._ranges._ranges_via_streaming` computes
``(min, max)`` of ``|post-Layer-2 coef|`` per LHS constraint term to pick
the autoscale exponents.  For deep ``Var × Param × Param`` chains the
polars streaming engine cannot push the join into the Param product, so it
MATERIALISES the product just to read a min/max (the memory spike).  Phase
D-1 routes the block-evaluable LHS terms through the same bounded numpy
block-COO builders the canonical matrix uses, reduces the resulting
``(_rid, col_id, coef)`` triple with the SAME side-vector scale factors and
the SAME ``_reduce_abs`` finite/non-zero mask, and skips the streaming
collect for that term.

Because block-COO is bit-identical to the polars chain (same coef values),
the row-factor index is the identical ``base_row + _rid``, the col-factor
index is the identical ``col_id``, and the reduction is the identical
``_reduce_abs``, the reported ``RangeReport`` must be BYTE-IDENTICAL
between the bounded path (default) and the streaming-collect path
(``POLAR_HIGH_DISABLE_BLOCK_COO=1``).  No tolerance.

The Problem mixes:

* a nodeBalance-shaped Sum LHS ``Sum(Where(v*P_unit, map)*P_step,
  over=("p","s"))`` — block-evaluable via the *relabel* Sum arm
  (``reduce_dims=("p","s") ⊆ var.dims``), which is bit-identical to
  polars' group_by, so the bounded path fires; and
* a non-Sum ``Var(p,d,t) × Pa(d,t) × Pb(d,t)`` LHS — block-evaluable via
  the non-Sum arm.

Layer 2 side vectors (row + col factors) are installed with a multi-decade
spread so the four-range readout is genuinely exercised.
"""

from __future__ import annotations

import itertools
import os

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


def _build_problem() -> Problem:
    """Problem with one Sum (relabel) and one non-Sum block-evaluable LHS
    family, declared dense suffix ``(d, t)``.  Frames are built in
    ``itertools.product`` order so the dense_axes sort contract holds."""
    p_idx = [0, 1]
    s_idx = ["s0", "s1"]
    d_idx = [10, 11]
    t_idx = [100, 101, 102]

    prob = Problem(dense_axes=("d", "t"))

    # ---- Family 1: nodeBalance-shaped Sum (relabel arm).
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
    P_unit = Param(("p",), pl.DataFrame({"p": p_idx, "value": [2.0, 30.0]}), name="P_unit")
    dt_rows = list(itertools.product(d_idx, t_idx))
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
    nb_over = nb.terms[0].frame.select(list(nb.terms[0].dims)).unique()
    prob.add_cstr("nb", over=nb_over, sense="<=", lhs_terms={"lhs": nb}, rhs_terms={"rhs": 0.0})

    # ---- Family 2: non-Sum Var(p,d,t) × Pa(d,t) × Pb(d,t).
    n_p, n_d, n_t = 3, 4, 5
    ps, ds, ts = list(range(n_p)), list(range(n_d)), list(range(n_t))
    cells = list(itertools.product(ps, ds, ts))
    w_over = pl.DataFrame(
        {
            "p": [c[0] for c in cells],
            "d": [c[1] for c in cells],
            "t": [c[2] for c in cells],
        }
    )
    w = prob.add_var("w", ("p", "d", "t"), w_over, lower=0.0, upper=1e6)
    dt2 = list(itertools.product(ds, ts))
    Pa = Param(
        ("d", "t"),
        pl.DataFrame(
            {
                "d": [c[0] for c in dt2],
                "t": [c[1] for c in dt2],
                "value": np.linspace(1e-3, 1e2, len(dt2)),
            }
        ),
        name="Pa",
    )
    Pb = Param(
        ("d", "t"),
        pl.DataFrame(
            {
                "d": [c[0] for c in dt2],
                "t": [c[1] for c in dt2],
                "value": np.linspace(2.0, 5e3, len(dt2)),
            }
        ),
        name="Pb",
    )
    prob.add_cstr(
        "vpp",
        over=w_over,
        sense="<=",
        lhs_terms={"lhs": w * Pa * Pb},
        rhs_terms={"rhs": 0.0},
    )

    prob.set_objective(Sum(v) + Sum(w), sense="min")
    return prob


def _install_side_vectors(prob: Problem) -> None:
    """Install Layer-2 row/col factors with a multi-decade spread so the
    range readout is genuinely exercised and the bounded path's
    side-vector multiply is non-trivial."""
    n_rows = sum(over.height for _c, _p, over in prob._cstrs)
    n_cols = int(prob._next_col)
    # Deterministic spread across several decades (powers of ten cycled).
    rf = np.array([10.0 ** ((i % 5) - 2) for i in range(n_rows)], dtype=np.float64)
    cf = np.array([10.0 ** ((i % 7) - 3) for i in range(n_cols)], dtype=np.float64)
    prob._layer2_row_factor = rf
    prob._layer2_col_factor = cf


def _report_tuple(rep) -> tuple:
    """Flatten a RangeReport into a comparable tuple of raw floats /
    bools, NaN-safe (NaN compared by ``repr``)."""

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


def test_ranges_block_coo_byte_identical() -> None:
    """The RangeReport from the bounded block-COO LHS path (default) must
    be BYTE-IDENTICAL to the streaming-collect path
    (``POLAR_HIGH_DISABLE_BLOCK_COO=1``).

    The comparison runs the SAME ``Problem`` instance (same constraint
    ``over`` row ordering, hence same ``base_row + _rid`` row-factor index,
    same ``col_id`` col-factor index) twice — only the env-gated evaluation
    path differs.  Building two separate Problems would let the nodeBalance
    ``over`` frame's ``.unique()`` row order drift between builds, which
    perturbs the position-indexed side vectors INDEPENDENTLY of block-COO
    and is not what this test pins."""
    cfg = _config()

    _clear_guard()
    try:
        os.environ["POLAR_HIGH_BLOCK_COO_MIN_DENSE"] = "1"
        prob = _build_problem()
        _install_side_vectors(prob)

        # Bounded block-COO path (default ON).
        rep_on = _ranges_via_streaming(prob, cfg)

        # Streaming-collect path (block-COO forced OFF) — SAME Problem,
        # SAME side vectors, only the code path differs.
        os.environ["POLAR_HIGH_DISABLE_BLOCK_COO"] = "1"
        rep_off = _ranges_via_streaming(prob, cfg)
    finally:
        _clear_guard()

    assert _report_tuple(rep_on) == _report_tuple(rep_off), (
        "bounded block-COO range readout diverged from the streaming path:\n"
        f"  ON  = {_report_tuple(rep_on)}\n"
        f"  OFF = {_report_tuple(rep_off)}"
    )


def test_ranges_block_coo_branch_fired() -> None:
    """The bounded block-COO LHS branch must actually FIRE for the two
    block-evaluable terms (proven via the ``block_coo=1`` profile signal).
    Two terms (the Sum relabel + the non-Sum chain) should emit it."""
    import io
    import sys

    cfg = _config()
    _clear_guard()
    os.environ["POLAR_HIGH_BLOCK_COO_MIN_DENSE"] = "1"
    os.environ["POLAR_HIGH_RANGES_PROFILE"] = "1"
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
    n_fired = out.count("block_coo=1")
    assert n_fired == 2, (
        f"expected the bounded block-COO LHS branch to fire on both "
        f"block-evaluable terms (count=2); profile reported {n_fired}.\n"
        f"profile:\n{out}"
    )
