"""End-to-end solve parity for the Sum-block-COO LHS arm wired at the
STREAMING site (``Problem._solve_streaming``, Site 2) and the WARM site
(``WarmProblem._initial_build``, Site 3).

Background
----------
Phase C-3a's Sum-block-COO arm rebuilds a ``Sum``-wrapped
``Var × Param-chain`` term from its :class:`SumBlockMeta` recipe and
reduces it in-block.  ``test_sum_block_coo_parity.py`` pins the
canonical-matrix site (Site 1).  This module pins the OTHER two
dispatch sites by driving a real ``solve()`` end to end:

* STREAMING — a nodeBalance-shaped Sum model solved via
  ``solve(streaming=True)``; objective + full solution must be
  BIT-IDENTICAL with the arm ON (default) vs OFF
  (``POLAR_HIGH_DISABLE_BLOCK_COO=1``), with profile evidence the Sum arm
  fired on the streaming site (``kind=sum`` + ``phase_site=streaming``).

* WARM — a relabel-shaped Sum model whose tracked Param is keyed on the
  kept dims; the warm build + one ``update_param`` roll must be
  bit-identical ON vs OFF at BOTH the initial solve AND the updated roll,
  with profile evidence the Sum arm fired warm (``kind=sum`` +
  ``phase_site=warm``).

Warm-tracker safety
-------------------
The warm site admits the Sum-block arm ONLY for the relabel shape
(``reduce_dims ⊆ var.dims`` ⇒ single-element groups ⇒ each emitted coef is
ONE product, bit-identical to the reduced path) AND only when every
tracked Param's dims are recoverable from the kept dims (``pdims ⊆ keep``,
``keep ⊆ var.dims``).  The combining shape (coef is a SUM over a reduced
dim) and a Param keyed on a reduced/map dim would make the tracker's
``factor * new_value`` model wrong, so they fall through to the
guaranteed-correct reduced ``term.lazy`` warm path — exercised here by the
combining variant, which must stay bit-identical (it does NOT fire warm).

All frames are built in ``itertools.product(lead..., d, t)`` order — the
row order the ``dense_axes`` sort contract requires.
"""

from __future__ import annotations

import io
import itertools
import os
import sys as _sys

import numpy as np
import polars as pl

from polar_high import Sum, WarmProblem
from polar_high.engine import Param, Problem, Where

# --------------------------------------------------------------------- #
# Env guard helpers (mirror tests/test_sum_block_coo_parity.py)         #
# --------------------------------------------------------------------- #


def _clear_guard() -> None:
    os.environ.pop("POLAR_HIGH_DISABLE_BLOCK_COO", None)
    os.environ.pop("POLAR_HIGH_ENABLE_BLOCK_COO", None)
    os.environ.pop("POLAR_HIGH_DISABLE_PRUNE_DOWN", None)
    os.environ.pop("POLAR_HIGH_DISABLE_WHERE_PUSHDOWN", None)
    os.environ.pop("POLAR_HIGH_BLOCK_COO_MIN_DENSE", None)
    os.environ.pop("POLAR_HIGH_BLOCK_COO_PROFILE", None)


def _assert_solution_bit_identical(sol_on, sol_off) -> None:
    assert sol_on.optimal and sol_off.optimal
    assert sol_on.obj == sol_off.obj
    assert np.array_equal(sol_on.col_value, sol_off.col_value)
    assert np.array_equal(sol_on.row_dual, sol_off.row_dual)
    assert np.array_equal(sol_on.col_dual, sol_off.col_dual)
    assert sol_on.row_names == sol_off.row_names
    assert sol_on.col_names == sol_off.col_names


def _assert_solution_close(sol_on, sol_off, *, rtol: float) -> None:
    assert sol_on.optimal and sol_off.optimal
    np.testing.assert_allclose(sol_on.obj, sol_off.obj, rtol=rtol, atol=0.0)
    np.testing.assert_allclose(
        sol_on.col_value, sol_off.col_value, rtol=rtol, atol=0.0
    )
    np.testing.assert_allclose(
        sol_on.row_dual, sol_off.row_dual, rtol=rtol, atol=0.0
    )
    assert sol_on.row_names == sol_off.row_names
    assert sol_on.col_names == sol_off.col_names


# --------------------------------------------------------------------- #
# STREAMING builder — nodeBalance-shaped Sum (relabel)                  #
# --------------------------------------------------------------------- #

_N_P, _N_S, _N_D, _N_T = 3, 2, 2, 12
_PS = list(range(_N_P))
_SS = [f"s{j}" for j in range(_N_S)]
_DS = list(range(10, 10 + _N_D))
_TS = list(range(100, 100 + _N_T))


def _streaming_node_balance_builder(*, with_where: bool = False):
    """A SOLVABLE nodeBalance-shaped Problem:

        Sum(Where(v(p,s,d,t) * P_unit(p), map_(p,s)->n) * P_step(d,t),
            over=("p","s"))   >=  rhs(n,d,t)

    over ``dense_axes=("d","t")``.  Each flow is a distinct ``col_id``
    mapping (via the map) to one node ``n`` ⇒ every ``(n,d,t,col_id)``
    reduce group is single-element ⇒ the RELABEL fast-path fires and is
    bit-identical to the reduced path.  The objective minimises
    ``Sum(v * cost)`` against the ``>=`` demand → a bounded feasible LP that
    solves via the streaming path.  ``with_where`` adds a pure-filter
    ``Where`` on ``t`` (a where-filtered variant).
    """

    def builder() -> Problem:
        prob = Problem(dense_axes=("d", "t"))
        rows = list(itertools.product(_PS, _SS, _DS, _TS))
        var_index = pl.DataFrame(
            {
                "p": [r[0] for r in rows],
                "s": [r[1] for r in rows],
                "d": [r[2] for r in rows],
                "t": [r[3] for r in rows],
            }
        )
        v = prob.add_var(
            "v", ("p", "s", "d", "t"), var_index, lower=0.0, upper=1e6
        )
        P_unit = Param(
            ("p",),
            pl.DataFrame({"p": _PS, "value": np.linspace(1.5, 3.5, _N_P)}),
            name="P_unit",
        )
        dt_rows = list(itertools.product(_DS, _TS))
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
        map_rows = list(itertools.product(_PS, _SS))
        map_to_n = pl.DataFrame(
            {
                "p": [r[0] for r in map_rows],
                "s": [r[1] for r in map_rows],
                "n": [
                    f"n{(r[0] + (0 if r[1] == 's0' else 1)) % 2}"
                    for r in map_rows
                ],
            }
        )
        inner = Where(v * P_unit, map_to_n) * P_step
        if with_where:
            sel_t = _TS[: max(1, _N_T // 2)]
            inner = Where(inner, pl.DataFrame({"t": sel_t}))
        lhs = Sum(inner, over=("p", "s"))
        # Sort the over frame deterministically — ``.unique()`` is
        # hash-ordered, so a position-based RHS would otherwise vary
        # build-to-build and make the ON/OFF comparison meaningless.
        over_dims = list(lhs.terms[0].dims)
        over_frame = (
            lhs.terms[0].frame.select(over_dims).unique().sort(over_dims)
        )
        # Positive demand (deterministic function of the dim VALUES, not row
        # position) so v must rise off its 0 lower bound.
        rhs = Param(
            tuple(over_frame.columns),
            over_frame.with_columns(
                value=pl.lit(1.0)
                + (pl.col("d") % 3).cast(pl.Float64) * 0.3
                + (pl.col("t") % 4).cast(pl.Float64) * 0.2
            ),
            name="rhs",
        )
        prob.add_cstr(
            "nb",
            over=over_frame,
            sense=">=",
            lhs_terms={"lhs": lhs},
            rhs_terms={"rhs": rhs},
        )
        cost = _dt_cost()
        prob.set_objective(Sum(v * cost), sense="min")
        return prob

    return builder


def _dt_cost() -> Param:
    dt_rows = list(itertools.product(_DS, _TS))
    return Param(
        ("d", "t"),
        pl.DataFrame(
            {
                "d": [r[0] for r in dt_rows],
                "t": [r[1] for r in dt_rows],
                "value": np.linspace(1.0, 4.0, len(dt_rows)),
            }
        ),
        name="cost",
    )


# --------------------------------------------------------------------- #
# Streaming drivers                                                     #
# --------------------------------------------------------------------- #


def _streaming_profile(builder) -> str:
    _clear_guard()
    os.environ["POLAR_HIGH_BLOCK_COO_PROFILE"] = "1"
    os.environ["POLAR_HIGH_BLOCK_COO_MIN_DENSE"] = "1"
    buf = io.StringIO()
    old = _sys.stderr
    try:
        _sys.stderr = buf
        builder().solve(streaming=True)
    finally:
        _sys.stderr = old
        _clear_guard()
    return buf.getvalue()


def _solve_streaming(builder, *, disabled: bool):
    _clear_guard()
    if disabled:
        os.environ["POLAR_HIGH_DISABLE_BLOCK_COO"] = "1"
    os.environ["POLAR_HIGH_BLOCK_COO_MIN_DENSE"] = "1"
    try:
        return builder().solve(streaming=True)
    finally:
        _clear_guard()


def test_streaming_sum_block_fires_and_bit_identical():
    """STREAMING: the Sum-block arm fires on the streaming site and the
    full solution is bit-identical ON vs OFF."""
    builder = _streaming_node_balance_builder()
    out = _streaming_profile(builder)
    assert "kind=sum\tphase_site=streaming" in out, (
        "the Sum-block-COO arm must fire on the STREAMING site (Site 2)"
    )
    assert "path=relabel" in out, (
        "nodeBalance (reduce_dims ⊆ var.dims) must take the relabel path"
    )
    sol_on = _solve_streaming(builder, disabled=False)
    sol_off = _solve_streaming(builder, disabled=True)
    _assert_solution_bit_identical(sol_on, sol_off)


def test_streaming_sum_block_where_filtered_bit_identical():
    """STREAMING, where-filtered variant: a pure-filter ``Where`` on ``t``
    carves the grid sparse → the relabel completeness guard forces the
    reduced-term.lazy fallback, but the solution stays bit-identical ON vs
    OFF (the relabel path is bit-identical when it fires; the fallback is
    byte-identical to OFF)."""
    builder = _streaming_node_balance_builder(with_where=True)
    sol_on = _solve_streaming(builder, disabled=False)
    sol_off = _solve_streaming(builder, disabled=True)
    _assert_solution_bit_identical(sol_on, sol_off)


# --------------------------------------------------------------------- #
# WARM builder — relabel Sum with a TRACKED (d,t)-keyed Param           #
# --------------------------------------------------------------------- #

_WN_P, _WN_D, _WN_T = 4, 3, 20
_WPS = list(range(_WN_P))
_WDS = list(range(10, 10 + _WN_D))
_WTS = list(range(100, 100 + _WN_T))


def _w_dt_param(name, lo, hi) -> Param:
    cells = list(itertools.product(_WDS, _WTS))
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


def _build_warm_sum_problem(*, p_step: Param, combining: bool = False) -> Problem:
    """A SOLVABLE relabel-shaped Sum Problem whose TRACKED Param is keyed on
    the kept dims:

        Sum(v(p,d,t) * P_unit(p) * P_step(d,t), over=("p",))  >=  rhs(d,t)

    over ``dense_axes=("d","t")``.  ``reduce_dims=("p",) ⊆ var.dims`` ⇒
    relabel (single-element groups).  ``keep=("d","t") ⊆ var.dims`` (no map
    extra) and the tracked ``P_step`` is keyed on ``(d,t) ⊆ keep`` ⇒ the
    warm tracker can recover its per-cell value, so the Sum-block arm is
    admitted warm.  The objective minimises ``Sum(v * cost)`` against the
    ``>=`` demand → bounded feasible LP.

    With ``combining=True`` a one-to-many map ``p->h`` fans each var cell to
    three ``h`` rows, all summed out (``over=("p","h")``, reduce dim ``h`` ∉
    var.dims) ⇒ the COMBINING shape; the warm site must DECLINE (coef is a
    SUM, not a single product) and fall through to the reduced path.
    """
    prob = Problem(dense_axes=("d", "t"))
    rows = list(itertools.product(_WPS, _WDS, _WTS))
    var_index = pl.DataFrame(
        {
            "p": [r[0] for r in rows],
            "d": [r[1] for r in rows],
            "t": [r[2] for r in rows],
        }
    )
    v = prob.add_var("v", ("p", "d", "t"), var_index, lower=0.0, upper=1e6)
    P_unit = Param(
        ("p",),
        pl.DataFrame({"p": _WPS, "value": np.linspace(1.5, 4.0, _WN_P)}),
        name="P_unit",
    )

    if combining:
        map_rows = []
        for p in _WPS:
            map_rows.append((p, p * 10))
            map_rows.append((p, p * 10 + 1))
            map_rows.append((p, p * 10 + 2))
        map_p_to_h = pl.DataFrame(
            {"p": [r[0] for r in map_rows], "h": [r[1] for r in map_rows]}
        )
        inner = Where(v * P_unit, map_p_to_h) * p_step
        lhs = Sum(inner, over=("p", "h"))
    else:
        lhs = Sum(v * P_unit * p_step, over=("p",))

    # Sort the over frame deterministically (``.unique()`` is hash-ordered)
    # so the position-independent RHS makes the ON/OFF comparison stable.
    over_dims = list(lhs.terms[0].dims)
    over_frame = lhs.terms[0].frame.select(over_dims).unique().sort(over_dims)
    rhs = Param(
        tuple(over_frame.columns),
        over_frame.with_columns(
            value=pl.lit(1.0)
            + (pl.col("d") % 3).cast(pl.Float64) * 0.3
            + (pl.col("t") % 4).cast(pl.Float64) * 0.2
        ),
        name="rhs",
    )
    prob.add_cstr(
        "nb",
        over=over_frame,
        sense=">=",
        lhs_terms={"lhs": lhs},
        rhs_terms={"rhs": rhs},
    )
    cost = Param(
        ("d", "t"),
        pl.DataFrame(
            {
                "d": [c[0] for c in itertools.product(_WDS, _WTS)],
                "t": [c[1] for c in itertools.product(_WDS, _WTS)],
                "value": np.linspace(1.0, 4.0, _WN_D * _WN_T),
            }
        ),
        name="cost",
    )
    prob.set_objective(Sum(v * cost), sense="min")
    return prob


def _run_warm(*, disabled: bool, combining: bool = False):
    """Build the WarmProblem (P_step declared mutable), solve roll 0, then
    ``update_param('P_step', ...)`` and solve roll 1.  Returns
    ``(sol_0, sol_1)``."""
    _clear_guard()
    if disabled:
        os.environ["POLAR_HIGH_DISABLE_BLOCK_COO"] = "1"
    os.environ["POLAR_HIGH_BLOCK_COO_MIN_DENSE"] = "1"
    try:
        step0 = _w_dt_param("P_step", 0.5, 1.5)
        p = _build_warm_sum_problem(p_step=step0, combining=combining)
        wp = WarmProblem(p)
        wp.declare_mutable("P_step")
        sol_0 = wp.solve()
        step1 = _w_dt_param("P_step", 0.8, 2.2)
        wp.update_param("P_step", step1)
        sol_1 = wp.solve()
        return sol_0, sol_1
    finally:
        _clear_guard()


def _warm_profile(*, combining: bool = False) -> str:
    _clear_guard()
    os.environ["POLAR_HIGH_BLOCK_COO_PROFILE"] = "1"
    os.environ["POLAR_HIGH_BLOCK_COO_MIN_DENSE"] = "1"
    buf = io.StringIO()
    old = _sys.stderr
    try:
        _sys.stderr = buf
        step0 = _w_dt_param("P_step", 0.5, 1.5)
        p = _build_warm_sum_problem(p_step=step0, combining=combining)
        wp = WarmProblem(p)
        wp.declare_mutable("P_step")
        wp.solve()
    finally:
        _sys.stderr = old
        _clear_guard()
    return buf.getvalue()


def test_warm_sum_block_fires_and_bit_identical():
    """WARM: the Sum-block arm fires on the warm site (relabel, tracked
    Param keyed on kept dims) and the warm build + one ``update_param`` roll
    are bit-identical ON vs OFF at BOTH the initial solve and the roll."""
    out = _warm_profile()
    assert "kind=sum\tphase_site=warm" in out, (
        "the Sum-block-COO arm must fire on the WARM site (Site 3)"
    )
    on_0, on_1 = _run_warm(disabled=False)
    off_0, off_1 = _run_warm(disabled=True)
    _assert_solution_bit_identical(on_0, off_0)
    _assert_solution_bit_identical(on_1, off_1)
    # The roll must actually move the optimum (else the tracking re-join
    # path is untested).
    assert on_1.obj != on_0.obj


def test_warm_sum_block_combining_declines_but_correct():
    """WARM, combining variant: the warm site must DECLINE the Sum-block arm
    (coef is a SUM over the map-introduced reduced dim ``h`` ⇒ the
    ``factor * new_value`` tracker model is invalid) and fall through to the
    reduced ``term.lazy`` warm path — so it does NOT fire warm, yet the warm
    build + roll stay bit-identical ON vs OFF."""
    out = _warm_profile(combining=True)
    assert "kind=sum\tphase_site=warm" not in out, (
        "the combining shape must NOT fire the Sum-block arm on the WARM "
        "site (warm-tracker-unsafe ⇒ declined)"
    )
    on_0, on_1 = _run_warm(disabled=False, combining=True)
    off_0, off_1 = _run_warm(disabled=True, combining=True)
    _assert_solution_bit_identical(on_0, off_0)
    _assert_solution_bit_identical(on_1, off_1)
    assert on_1.obj != on_0.obj
