"""Warm-path (rolling/warm-start) parity for the block-COO LHS arm.

Block-COO is dispatched at THREE sites: the non-streaming
``Problem._build_canonical_matrix`` (Site 1, ``test_block_coo_parity.py``),
the streaming ``Problem._solve_streaming`` (Site 2,
``test_block_coo_streaming_parity.py``) and the warm/rolling
``WarmProblem._initial_build`` (Site 3, pinned HERE).

Site 3 differs from Sites 1/2: its term loop filters to tracked Params
first, and its emitted frame must carry ``*term.dims`` (in addition to
``_rid, col_id, coef``) so the warm param-TRACKING machinery can re-join
each tracked Param on its dims for incremental ``update_param`` across
rolls.  The block-COO helper is therefore called with
``keep_dims=tuple(term.dims)``; this module verifies:

* On a dense-complete ``Var(p,d,t) × unit_size(p) × step_duration(d,t) ×
  efficiency(p,d,t)`` LHS over ``dense_axes=("d","t")``, the WARM build +
  an ``update_param`` roll produce bit-identical solver results with
  block-COO ON (default) vs OFF (``POLAR_HIGH_DISABLE_BLOCK_COO=1``) — at
  both the initial solve AND the updated roll.
* A pure-filter ``Where`` on ``t`` carves the grid sparse → the
  completeness guard forces the joined fallback; the warm build + roll
  stay bit-identical ON vs OFF (the keep_dims carry survives the joined
  builder too, so tracking is unaffected).
* Profile evidence (``POLAR_HIGH_BLOCK_COO_PROFILE=1``) confirms block-COO
  FIRES on the warm path (``phase_site=warm``), positional on the
  dense-complete case and joined on the sparse case.

All frames are built in ``itertools.product(lead..., d, t)`` order — the
row order the ``dense_axes`` sort contract requires.
"""

from __future__ import annotations

import io
import itertools
import os
import re
import sys as _sys

import numpy as np
import polars as pl

from polar_high import Sum, WarmProblem
from polar_high.engine import Param, Problem, Where

# --------------------------------------------------------------------- #
# Env guard helpers (mirror tests/test_block_coo_streaming_parity.py)   #
# --------------------------------------------------------------------- #


def _clear_guard() -> None:
    os.environ.pop("POLAR_HIGH_DISABLE_BLOCK_COO", None)
    os.environ.pop("POLAR_HIGH_ENABLE_BLOCK_COO", None)
    os.environ.pop("POLAR_HIGH_DISABLE_PRUNE_DOWN", None)
    os.environ.pop("POLAR_HIGH_DISABLE_WHERE_PUSHDOWN", None)
    os.environ.pop("POLAR_HIGH_BLOCK_COO_MIN_DENSE", None)


# --------------------------------------------------------------------- #
# Frame builders                                                        #
# --------------------------------------------------------------------- #

_N_P, _N_D, _N_T = 4, 6, 50
_PS, _DS, _TS = list(range(_N_P)), list(range(_N_D)), list(range(_N_T))


def _vdt_over(ps, ds, ts) -> pl.DataFrame:
    """Var(p, d, t) ``over`` frame, sorted by (p, d, t)."""
    cells = list(itertools.product(ps, ds, ts))
    return pl.DataFrame(
        {
            "p": [c[0] for c in cells],
            "d": [c[1] for c in cells],
            "t": [c[2] for c in cells],
        }
    )


def _dt_param(ds, ts, name, lo, hi) -> Param:
    """Param(d, t) over the full (d, t) grid, sorted by (d, t)."""
    cells = list(itertools.product(ds, ts))
    return Param(
        ("d", "t"),
        pl.DataFrame(
            {
                "d": [c[0] for c in cells],
                "t": [c[1] for c in cells],
                "value": np.linspace(lo, hi, len(cells)),
            }
        ),
        name=name,
    )


def _pdt_param(ps, ds, ts, name, lo, hi) -> Param:
    """Param(p, d, t) over the full (p, d, t) grid, sorted by (p, d, t)."""
    cells = list(itertools.product(ps, ds, ts))
    return Param(
        ("p", "d", "t"),
        pl.DataFrame(
            {
                "p": [c[0] for c in cells],
                "d": [c[1] for c in cells],
                "t": [c[2] for c in cells],
                "value": np.linspace(lo, hi, len(cells)),
            }
        ),
        name=name,
    )


def _eff_param(lo: float, hi: float) -> Param:
    """The tracked ``efficiency(p,d,t)`` Param (dim signature (p,d,t))."""
    return _pdt_param(_PS, _DS, _TS, "efficiency", lo, hi)


def _build_warm_problem(*, eff: Param, with_where_t: bool = False) -> Problem:
    """A SOLVABLE Problem whose single LHS family is the dense-complete
    three-case chain

        Var(p, d, t)
          × unit_size(p)          # lead-only
          × step_duration(d, t)   # dense-only
          × efficiency(p, d, t)   # lead-subset + dense  (TRACKED / mutable)

    over the declared dense ``(d, t)`` suffix.  The constraint is
    ``chain >= rhs`` and the objective minimises ``Sum(v * cost)`` — a
    bounded, feasible LP.  ``efficiency`` is the supplied ``eff`` Param so
    a roll can swap it via ``update_param``.  With ``with_where_t`` a
    pure-filter ``Where`` on ``t`` carves the grid sparse → joined
    fallback.
    """
    p = Problem(dense_axes=("d", "t"))
    over = _vdt_over(_PS, _DS, _TS)
    v = p.add_var("v", ("p", "d", "t"), over, lower=0.0, upper=1e6)

    unit_size = Param(
        ("p",),
        pl.DataFrame({"p": _PS, "value": np.linspace(10.0, 40.0, _N_P)}),
        name="unit_size",
    )
    step_duration = _dt_param(_DS, _TS, "step_duration", 0.25, 3.0)

    chain = v * unit_size * step_duration * eff
    if with_where_t:
        sel_t = _TS[: max(1, _N_T // 2)]
        chain = Where(chain, pl.DataFrame({"t": sel_t}))

    rhs = _pdt_param(_PS, _DS, _TS, "rhs", 1.0, 20.0)
    rhs_over = over.filter(pl.col("t").is_in(_TS[: max(1, _N_T // 2)])) if with_where_t else over
    p.add_cstr(
        "c",
        over=rhs_over,
        sense=">=",
        lhs_terms={"lhs": chain},
        rhs_terms={"rhs": rhs},
    )

    cost = _dt_param(_DS, _TS, "cost", 1.0, 4.0)
    p.set_objective(Sum(v * cost), sense="min")
    return p


# --------------------------------------------------------------------- #
# Warm drivers (block-COO ON / OFF), tracking efficiency across a roll  #
# --------------------------------------------------------------------- #


def _run_warm(*, disabled: bool, with_where_t: bool):
    """Build the WarmProblem (efficiency declared mutable), solve roll 0,
    then ``update_param('efficiency', ...)`` and solve roll 1.  Returns
    ``(sol_0, sol_1)``.  ``disabled`` toggles
    ``POLAR_HIGH_DISABLE_BLOCK_COO=1``."""
    _clear_guard()
    if disabled:
        os.environ["POLAR_HIGH_DISABLE_BLOCK_COO"] = "1"
    try:
        eff0 = _eff_param(0.30, 0.95)
        p = _build_warm_problem(eff=eff0, with_where_t=with_where_t)
        wp = WarmProblem(p)
        wp.declare_mutable("efficiency")
        sol_0 = wp.solve()
        # Roll: swap efficiency to a different (p,d,t) profile.
        eff1 = _eff_param(0.50, 0.80)
        wp.update_param("efficiency", eff1)
        sol_1 = wp.solve()
        return sol_0, sol_1
    finally:
        _clear_guard()


def _warm_profile(*, with_where_t: bool) -> str:
    """Run the warm build (block-COO ON) under
    ``POLAR_HIGH_BLOCK_COO_PROFILE=1`` and capture stderr."""
    _clear_guard()
    os.environ["POLAR_HIGH_BLOCK_COO_PROFILE"] = "1"
    buf = io.StringIO()
    old = _sys.stderr
    try:
        _sys.stderr = buf
        eff0 = _eff_param(0.30, 0.95)
        p = _build_warm_problem(eff=eff0, with_where_t=with_where_t)
        wp = WarmProblem(p)
        wp.declare_mutable("efficiency")
        wp.solve()
    finally:
        _sys.stderr = old
        os.environ.pop("POLAR_HIGH_BLOCK_COO_PROFILE", None)
        _clear_guard()
    return buf.getvalue()


def _assert_solution_bit_identical(sol_on, sol_off) -> None:
    assert sol_on.optimal and sol_off.optimal
    assert sol_on.obj == sol_off.obj
    assert np.array_equal(sol_on.col_value, sol_off.col_value)
    assert np.array_equal(sol_on.row_dual, sol_off.row_dual)
    assert np.array_equal(sol_on.col_dual, sol_off.col_dual)
    assert sol_on.row_names == sol_off.row_names
    assert sol_on.col_names == sol_off.col_names


# --------------------------------------------------------------------- #
# Tests                                                                 #
# --------------------------------------------------------------------- #


def test_warm_block_coo_fires_positional():
    """Site 3: block-COO must FIRE on the WARM path for the dense-complete
    chain, taking the POSITIONAL slice-multiply path.

    The warm build also runs ``p.canonicalise()`` (Site 1), which dispatches
    block-COO on the SAME term, so the profile carries two
    ``phase=block_coo_term`` lines — exactly ONE of them tagged
    ``phase_site=warm`` (Site 3, the one this test pins).  Both take the
    same path on this dense-complete grid (positional)."""
    out = _warm_profile(with_where_t=False)
    assert out.count("phase=block_coo_term\tphase_site=warm") == 1, (
        "block-COO must fire EXACTLY once on the WARM path (Site 3) for the dense-complete chain"
    )
    assert set(re.findall(r"\bpath=(\w+)", out)) == {"positional"}, (
        "dense-complete grid must take the positional path on the warm build"
    )


def test_warm_block_coo_dense_complete_parity():
    """Dense-complete chain: the warm build (initial solve) AND the
    ``update_param`` roll must be bit-identical with block-COO ON
    (default) vs OFF."""
    on_0, on_1 = _run_warm(disabled=False, with_where_t=False)
    off_0, off_1 = _run_warm(disabled=True, with_where_t=False)
    _assert_solution_bit_identical(on_0, off_0)
    _assert_solution_bit_identical(on_1, off_1)
    # The roll must actually change the optimum (else the test would be
    # vacuous w.r.t. the tracking re-join path).
    assert on_1.obj != on_0.obj


def test_warm_block_coo_falls_back_when_sparse():
    """Pure-filter ``Where`` on ``t`` → joined fallback on the warm path;
    block-COO still fires (path=joined) and the warm build + roll stay
    bit-identical ON vs OFF (the keep_dims carry survives the joined
    builder, so the tracking re-join is unaffected)."""
    out = _warm_profile(with_where_t=True)
    assert out.count("phase=block_coo_term\tphase_site=warm") == 1, (
        "block-COO must still fire EXACTLY once (then fall back) on the sparse warm case (Site 3)"
    )
    assert set(re.findall(r"\bpath=(\w+)", out)) == {"joined"}, (
        "a pure-filter Where must force the joined fallback on the warm build"
    )
    on_0, on_1 = _run_warm(disabled=False, with_where_t=True)
    off_0, off_1 = _run_warm(disabled=True, with_where_t=True)
    _assert_solution_bit_identical(on_0, off_0)
    _assert_solution_bit_identical(on_1, off_1)
    assert on_1.obj != on_0.obj
