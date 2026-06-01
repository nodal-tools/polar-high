"""Streaming-path parity for the block-COO LHS evaluation arm.

Block-COO is dispatched at TWO sites: the non-streaming
``Problem._build_canonical_matrix`` (covered by
``tests/test_block_coo_parity.py``) and the streaming
``Problem._solve_streaming`` (the DEFAULT solve path, exercised here).
This module pins that Site-2 wiring:

* On a dense-complete ``Var(p,d,t) × unit_size(p) × step_duration(d,t) ×
  efficiency(p,d,t)`` LHS over the declared ``dense_axes=("d","t")``,
  ``Problem.solve(streaming=True)`` must produce bit-identical solver
  results whether block-COO is ON (default) or OFF
  (``POLAR_HIGH_DISABLE_BLOCK_COO=1``).
* A pure-filter ``Where`` on ``t`` carves the grid sparse → the
  completeness guard forces the joined fallback; results must STILL be
  bit-identical ON vs OFF.
* Profile evidence (``POLAR_HIGH_BLOCK_COO_PROFILE=1``) confirms block-COO
  actually FIRES on the streaming path (so Site 2 is engaged), taking the
  positional path on the dense-complete case and the joined path on the
  sparse case.

All frames are built in ``itertools.product(lead..., d, t)`` order — the
row order the ``dense_axes`` sort contract requires (lead dims as a sorted
prefix, dense axes as the trailing sort keys).
"""

from __future__ import annotations

import io
import itertools
import os
import re
import sys as _sys

import numpy as np
import polars as pl

from polar_high import Sum
from polar_high.engine import Param, Problem, Where

# --------------------------------------------------------------------- #
# Env guard helpers (mirror tests/test_block_coo_parity.py)             #
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


def _build_streaming_problem(*, with_where_t: bool = False):
    """Builder for a SOLVABLE Problem whose single LHS family is the
    dense-complete three-case chain

        Var(p, d, t)
          × unit_size(p)          # lead-only
          × step_duration(d, t)   # dense-only
          × efficiency(p, d, t)   # lead-subset + dense

    over the declared dense ``(d, t)`` suffix.  The constraint is
    ``chain >= rhs`` (so the non-negative Var is pushed up off its lower
    bound) and the objective minimises ``Sum(v * cost)``, giving a bounded,
    feasible LP that solves via the streaming path.  With ``with_where_t``
    a pure-filter ``Where`` on ``t`` carves the grid sparse → joined
    fallback.
    """
    n_p, n_d, n_t = 4, 6, 50
    ps, ds, ts = list(range(n_p)), list(range(n_d)), list(range(n_t))

    def builder() -> Problem:
        p = Problem(dense_axes=("d", "t"))
        over = _vdt_over(ps, ds, ts)
        v = p.add_var("v", ("p", "d", "t"), over, lower=0.0, upper=1e6)

        unit_size = Param(
            ("p",),
            pl.DataFrame({"p": ps, "value": np.linspace(10.0, 40.0, n_p)}),
            name="unit_size",
        )
        step_duration = _dt_param(ds, ts, "step_duration", 0.25, 3.0)
        pdt_cells = list(itertools.product(ps, ds, ts))
        efficiency = Param(
            ("p", "d", "t"),
            pl.DataFrame(
                {
                    "p": [c[0] for c in pdt_cells],
                    "d": [c[1] for c in pdt_cells],
                    "t": [c[2] for c in pdt_cells],
                    "value": np.linspace(0.3, 0.95, len(pdt_cells)),
                }
            ),
            name="efficiency",
        )
        chain = v * unit_size * step_duration * efficiency
        if with_where_t:
            sel_t = ts[: max(1, n_t // 2)]
            chain = Where(chain, pl.DataFrame({"t": sel_t}))

        # rhs(p, d, t): a positive demand so v must rise off its 0 lower
        # bound → the LP is non-trivially bounded by the objective below.
        rhs = Param(
            ("p", "d", "t"),
            pl.DataFrame(
                {
                    "p": [c[0] for c in pdt_cells],
                    "d": [c[1] for c in pdt_cells],
                    "t": [c[2] for c in pdt_cells],
                    "value": np.linspace(1.0, 20.0, len(pdt_cells)),
                }
            ),
            name="rhs",
        )
        rhs_over = over.filter(pl.col("t").is_in(ts[: max(1, n_t // 2)])) if with_where_t else over
        p.add_cstr(
            "c",
            over=rhs_over,
            sense=">=",
            lhs_terms={"lhs": chain},
            rhs_terms={"rhs": rhs},
        )

        # Objective: minimise total weighted v → pushes v down against the
        # >= constraint, giving a unique interior optimum.
        cost = _dt_param(ds, ts, "cost", 1.0, 4.0)
        p.set_objective(Sum(v * cost), sense="min")
        return p

    return builder


# --------------------------------------------------------------------- #
# Profile sniffing on the STREAMING path                                #
# --------------------------------------------------------------------- #


def _streaming_profile(builder) -> str:
    """Run ``builder().solve(streaming=True)`` under the DEFAULT (block-COO
    ON) with ``POLAR_HIGH_BLOCK_COO_PROFILE=1`` and capture stderr."""
    _clear_guard()
    os.environ["POLAR_HIGH_BLOCK_COO_PROFILE"] = "1"
    buf = io.StringIO()
    old = _sys.stderr
    try:
        _sys.stderr = buf
        builder().solve(streaming=True)
    finally:
        _sys.stderr = old
        os.environ.pop("POLAR_HIGH_BLOCK_COO_PROFILE", None)
        _clear_guard()
    return buf.getvalue()


def _solve_streaming_default(builder):
    _clear_guard()
    try:
        return builder().solve(streaming=True)
    finally:
        _clear_guard()


def _solve_streaming_disabled(builder):
    _clear_guard()
    os.environ["POLAR_HIGH_DISABLE_BLOCK_COO"] = "1"
    try:
        return builder().solve(streaming=True)
    finally:
        _clear_guard()


def _assert_solution_bit_identical(sol_on, sol_off) -> None:
    """The canonical comparison test_streaming_parity.py uses, tightened
    to bit-exactness (block-COO is bit-identical to the polars path)."""
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


def test_streaming_block_coo_fires_positional():
    """Site 2: block-COO must FIRE on the streaming path for the
    dense-complete chain, taking the POSITIONAL slice-multiply path."""
    builder = _build_streaming_problem(with_where_t=False)
    out = _streaming_profile(builder)
    assert "phase=block_coo_term" in out, (
        "block-COO must fire on the STREAMING path (Site 2) for the dense-complete chain"
    )
    assert re.findall(r"\bpath=(\w+)", out) == ["positional"], (
        "dense-complete grid must take the positional path on streaming"
    )


def test_streaming_block_coo_dense_complete_parity():
    """Dense-complete chain: ``solve(streaming=True)`` must be bit-identical
    with block-COO ON (default) vs OFF."""
    builder = _build_streaming_problem(with_where_t=False)
    sol_on = _solve_streaming_default(builder)
    sol_off = _solve_streaming_disabled(builder)
    _assert_solution_bit_identical(sol_on, sol_off)


def test_streaming_block_coo_falls_back_when_sparse():
    """Pure-filter ``Where`` on ``t`` → joined fallback on the streaming
    path; block-COO still fires (path=joined) and results stay
    bit-identical ON vs OFF."""
    builder = _build_streaming_problem(with_where_t=True)
    out = _streaming_profile(builder)
    assert "phase=block_coo_term" in out, (
        "block-COO must still fire (then fall back) on the sparse case"
    )
    assert re.findall(r"\bpath=(\w+)", out) == ["joined"], (
        "a pure-filter Where must force the joined fallback on streaming"
    )
    sol_on = _solve_streaming_default(builder)
    sol_off = _solve_streaming_disabled(builder)
    _assert_solution_bit_identical(sol_on, sol_off)
