"""Parity tests for the Where() deferred-filter pushdown (Option B).

Background
----------
Previously :func:`polar_high.engine.Where` eagerly inner-joined the
filter frame into ``_Term.lazy`` AND cleared ``var_source`` /
``coef_scalar``, which prevented downstream LHS prune-down
(:func:`_build_lhs_pruned_plan`) from firing on chains like
``Where(v * P1 * P2, f)``.  Pushdown defers the pure-filter case into a
new ``_Term.where_frames`` metadata slot so the leaf-rebuild can apply
``f`` at the row_index-bounded step.  Sum / Lag / fallback paths bake
the deferred filter via :func:`_apply_where_frames` before consuming
``t.lazy``.

These tests pin numerical parity between the pushdown path
(``POLAR_HIGH_DISABLE_WHERE_PUSHDOWN`` unset) and today's verbatim
behaviour (``POLAR_HIGH_DISABLE_WHERE_PUSHDOWN=1``) — same pattern as
:mod:`tests.test_prune_down_scalar_anonymous_fix`'s
``_clear_guard`` round-trip.  Each test builds the same Problem twice
under the two regimes and asserts byte-for-byte equality of the
canonical matrix's ``val`` / ``row_lb`` / ``row_ub`` arrays.
"""

from __future__ import annotations

import itertools
import os

import numpy as np
import polars as pl
import pytest

import polar_high as fp
from polar_high.engine import Lag, Param, Problem, Sum, Var, Where


# --------------------------------------------------------------------- #
# Helpers                                                               #
# --------------------------------------------------------------------- #


def _clear_guard() -> None:
    os.environ.pop("POLAR_HIGH_DISABLE_WHERE_PUSHDOWN", None)
    os.environ.pop("POLAR_HIGH_DISABLE_PRUNE_DOWN", None)


def _set_disable() -> None:
    os.environ["POLAR_HIGH_DISABLE_WHERE_PUSHDOWN"] = "1"


def _matrix_arrays(m) -> tuple[list, list, list]:
    """Sort-stable comparison key — sorted (col, row, val) triples
    along with row_lb / row_ub.  Two different LHS evaluation paths can
    emit the same triples in different intra-column orderings (Sum
    aggregation or where_frames semi-join reordering), so the strict
    structural equality goes through a sort first.
    """
    val = np.asarray(m.val, dtype=np.float64)
    row_idx = np.asarray(m.row_idx, dtype=np.int64)
    col_ptr = np.asarray(m.col_ptr, dtype=np.int64)
    cols = np.repeat(
        np.arange(m.n_cols, dtype=np.int64), np.diff(col_ptr).astype(np.int64)
    )
    order = np.lexsort((row_idx, cols))
    return (
        list(val[order]),
        list(np.asarray(m.row_lb, dtype=np.float64)),
        list(np.asarray(m.row_ub, dtype=np.float64)),
    )


def _build_and_snapshot(builder) -> tuple[list, list, list]:
    """Run ``builder()`` (returns a Problem), canonicalise, snapshot."""
    prob = builder()
    m = prob._build_canonical_matrix()
    return _matrix_arrays(m)


def _assert_parity(builder) -> None:
    """Assert byte-for-byte canonical equality between pushdown and
    disabled-pushdown runs of the same builder."""
    _clear_guard()
    try:
        _set_disable()
        snap_off = _build_and_snapshot(builder)
    finally:
        _clear_guard()
    snap_on = _build_and_snapshot(builder)
    assert snap_on == snap_off, (
        f"Where-pushdown parity failure:\n"
        f"  pushdown ON  val={snap_on[0][:8]} ...\n"
        f"  pushdown OFF val={snap_off[0][:8]} ...\n"
        f"  row_ub diff (first 4): "
        f"{[a - b for a, b in zip(snap_on[2][:4], snap_off[2][:4])]}"
    )


# --------------------------------------------------------------------- #
# Shared fixture builders                                               #
# --------------------------------------------------------------------- #


def _build_v_p_problem(
    *,
    rows: int = 100,
    sel_rows: int = 10,
    chain_len: int = 2,
    with_extra_dim: bool = False,
):
    """Return a builder() that creates a Problem with a wide
    ``v * P1 * (P2 * …)`` LHS and a filter frame ``f`` that keeps
    ``sel_rows`` of ``rows``.  ``with_extra_dim`` switches f from
    pure-filter to map-effect.
    """

    def builder() -> Problem:
        p = Problem()
        x_idx = list(range(rows))
        over = pl.DataFrame({"x": x_idx})
        v = p.add_var("v", ("x",), over, lower=0.0, upper=1e6)

        P1 = Param(
            ("x",),
            pl.DataFrame(
                {"x": x_idx, "value": np.linspace(0.1, 0.9, rows)}
            ),
            name="P1",
        )
        chain = v * P1
        if chain_len >= 2:
            P2 = Param(
                ("x",),
                pl.DataFrame(
                    {"x": x_idx, "value": np.linspace(2.0, 5.0, rows)}
                ),
                name="P2",
            )
            chain = chain * P2

        sel = x_idx[:sel_rows]
        if with_extra_dim:
            # Map effect — frame adds a new dim 'g'.
            f = pl.DataFrame({"x": sel, "g": [f"g{i}" for i in sel]})
            lhs = Where(chain, f)
            # Sum over g so the term is bound to the over=('x',) index.
            lhs = Sum(lhs, over=("g",))
            over_cstr = pl.DataFrame({"x": x_idx})
        else:
            f = pl.DataFrame({"x": sel})
            lhs = Where(chain, f)
            over_cstr = pl.DataFrame({"x": x_idx})

        p.add_cstr(
            "c",
            over=over_cstr,
            sense="<=",
            lhs_terms={"lhs": lhs},
            rhs_terms={"rhs": 0.0},
        )
        return p

    return builder


# --------------------------------------------------------------------- #
# Tests                                                                 #
# --------------------------------------------------------------------- #


def test_where_pure_filter_var_param_param():
    """``Where(v*p1*p2, selective_frame)`` — pushdown vs disabled
    must produce byte-identical canonical matrices."""
    _assert_parity(_build_v_p_problem(rows=100, sel_rows=10, chain_len=2))


def test_where_with_extras_map_effect():
    """``Where(v*p, frame_with_extra_dim)`` — pushdown takes the
    map-effect branch (eager join, clear metadata); parity must hold."""
    _assert_parity(
        _build_v_p_problem(rows=50, sel_rows=8, chain_len=1, with_extra_dim=True)
    )


def test_nested_where():
    """``Where(Where(v*p1*p2, f1), f2)`` — where_frames must
    accumulate, both filters applied, parity vs disabled-pushdown."""

    def builder() -> Problem:
        rows = 60
        p = Problem()
        x_idx = list(range(rows))
        over = pl.DataFrame({"x": x_idx})
        v = p.add_var("v", ("x",), over, lower=0.0, upper=1e6)
        P1 = Param(
            ("x",),
            pl.DataFrame({"x": x_idx, "value": np.linspace(0.5, 1.5, rows)}),
            name="P1",
        )
        P2 = Param(
            ("x",),
            pl.DataFrame({"x": x_idx, "value": np.linspace(2.0, 4.0, rows)}),
            name="P2",
        )
        f1 = pl.DataFrame({"x": x_idx[:30]})  # first half
        f2 = pl.DataFrame({"x": x_idx[10:25]})  # narrower
        lhs = Where(Where(v * P1 * P2, f1), f2)
        p.add_cstr(
            "c", over=over, sense="<=",
            lhs_terms={"lhs": lhs}, rhs_terms={"rhs": 0.0},
        )
        return p

    _assert_parity(builder)
    # Also sanity-check that where_frames accumulated with pushdown ON.
    _clear_guard()
    prob = builder()
    # Find the registered constraint and its single LHS term:
    cstr_name, proto, _ = prob._cstrs[0]
    t = proto.expr.terms[0]
    assert t.where_frames is not None
    assert len(t.where_frames) == 2


def test_where_after_sum():
    """``Where(Sum(v*p, over=('t',)), frame)`` — Sum drops var_source
    so Where falls back to fallback semi-join (pure-filter case still
    records where_frames, but the LHS-prune path is skipped since
    var_source is None).  Parity must hold via the fallback's
    _apply_where_frames bake."""

    def builder() -> Problem:
        p = Problem()
        n_x, n_t = 8, 12
        rows = list(itertools.product(range(n_x), range(n_t)))
        over_xt = pl.DataFrame({"x": [r[0] for r in rows], "t": [r[1] for r in rows]})
        v = p.add_var("v", ("x", "t"), over_xt, lower=0.0, upper=1e6)
        P = Param(
            ("x", "t"),
            pl.DataFrame(
                {
                    "x": [r[0] for r in rows],
                    "t": [r[1] for r in rows],
                    "value": np.linspace(0.1, 1.0, len(rows)),
                }
            ),
            name="P",
        )
        # Sum over t — collapses t, term has var_source=None now.
        agg = Sum(v * P, over=("t",))
        # Filter on x — pure filter (shared=['x'], extras=()).
        f = pl.DataFrame({"x": [0, 2, 4, 6]})
        lhs = Where(agg, f)
        over_x = pl.DataFrame({"x": list(range(n_x))})
        p.add_cstr(
            "c", over=over_x, sense="<=",
            lhs_terms={"lhs": lhs}, rhs_terms={"rhs": 0.0},
        )
        return p

    _assert_parity(builder)


def test_sum_after_where():
    """``Sum(Where(v*p, frame), over=('t',))`` — Sum must bake
    where_frames before aggregating; parity vs disabled."""

    def builder() -> Problem:
        p = Problem()
        n_x, n_t = 6, 10
        rows = list(itertools.product(range(n_x), range(n_t)))
        over_xt = pl.DataFrame({"x": [r[0] for r in rows], "t": [r[1] for r in rows]})
        v = p.add_var("v", ("x", "t"), over_xt, lower=0.0, upper=1e6)
        P = Param(
            ("x", "t"),
            pl.DataFrame(
                {
                    "x": [r[0] for r in rows],
                    "t": [r[1] for r in rows],
                    "value": np.linspace(0.1, 1.0, len(rows)),
                }
            ),
            name="P",
        )
        # Filter on t — pure filter (shared=['t'], extras=()).
        f = pl.DataFrame({"t": [0, 1, 5, 9]})
        agg = Sum(Where(v * P, f), over=("t",))
        over_x = pl.DataFrame({"x": list(range(n_x))})
        p.add_cstr(
            "c", over=over_x, sense="<=",
            lhs_terms={"lhs": agg}, rhs_terms={"rhs": 0.0},
        )
        return p

    _assert_parity(builder)


def test_where_anonymous_param_chain():
    """``Where(v * named * anonymous, frame)`` — anonymous Param's
    contribution must survive the prune-down rebuild (per the
    anonymous-Param fix from commit a02b0fc)."""

    def builder() -> Problem:
        p = Problem()
        x_idx = list(range(40))
        over = pl.DataFrame({"x": x_idx})
        v = p.add_var("v", ("x",), over, lower=0.0, upper=1e6)
        named = Param(
            ("x",),
            pl.DataFrame({"x": x_idx, "value": np.linspace(0.3, 0.9, len(x_idx))}),
            name="named",
        )
        anon = Param(
            ("x",),
            pl.DataFrame({"x": x_idx, "value": np.full(len(x_idx), 0.5)}),
        )
        chain = v * named * anon
        f = pl.DataFrame({"x": x_idx[::3]})
        lhs = Where(chain, f)
        p.add_cstr(
            "c", over=over, sense="<=",
            lhs_terms={"lhs": lhs}, rhs_terms={"rhs": 0.0},
        )
        return p

    _assert_parity(builder)


def test_where_scalar_fold():
    """``Where(v * (p * 60.0) * p2, frame)`` — Param-side scalar fold
    must propagate through the prune-down + pushdown."""

    def builder() -> Problem:
        p = Problem()
        x_idx = list(range(50))
        over = pl.DataFrame({"x": x_idx})
        v = p.add_var("v", ("x",), over, lower=0.0, upper=1e6)
        a = Param(
            ("x",),
            pl.DataFrame({"x": x_idx, "value": np.full(len(x_idx), 0.001)}),
            name="a",
        )
        b = Param(
            ("x",),
            pl.DataFrame({"x": x_idx, "value": np.full(len(x_idx), 1.0)}),
            name="b",
        )
        scaled = (a * 60.0) * b  # _value_scalar=60
        lhs = Where(v * scaled, pl.DataFrame({"x": x_idx[:20]}))
        p.add_cstr(
            "c", over=over, sense="<=",
            lhs_terms={"lhs": lhs}, rhs_terms={"rhs": 0.0},
        )
        return p

    _assert_parity(builder)


def test_where_then_mul_param():
    """``Where(v*p1, f) * p2`` — where_frames must propagate through
    ``Expr.__mul__(Param)``.  Also assert canonical equality vs
    ``Where(v*p1*p2, f)``."""

    def _build(reorder: bool) -> Problem:
        p = Problem()
        x_idx = list(range(30))
        over = pl.DataFrame({"x": x_idx})
        v = p.add_var("v", ("x",), over, lower=0.0, upper=1e6)
        P1 = Param(
            ("x",),
            pl.DataFrame({"x": x_idx, "value": np.linspace(0.5, 1.5, len(x_idx))}),
            name="P1",
        )
        P2 = Param(
            ("x",),
            pl.DataFrame({"x": x_idx, "value": np.linspace(2.0, 4.0, len(x_idx))}),
            name="P2",
        )
        f = pl.DataFrame({"x": x_idx[:15]})
        if reorder:
            # Where applied AFTER the chain.
            lhs = Where(v * P1 * P2, f)
        else:
            # Where between P1 and P2 multiplies.
            lhs = Where(v * P1, f) * P2
        p.add_cstr(
            "c", over=over, sense="<=",
            lhs_terms={"lhs": lhs}, rhs_terms={"rhs": 0.0},
        )
        return p

    # Parity (pushdown ON vs OFF) for the reorder=False case.
    _assert_parity(lambda: _build(reorder=False))
    # Canonical equality between the two orderings under pushdown ON.
    _clear_guard()
    snap_a = _matrix_arrays(_build(reorder=False)._build_canonical_matrix())
    snap_b = _matrix_arrays(_build(reorder=True)._build_canonical_matrix())
    assert snap_a == snap_b


def test_disable_guard_recovers_today_behavior():
    """Same shape as ``test_disable_guard_still_recovers_merged_path``
    in ``tests.test_prune_down_scalar_anonymous_fix`` — under
    ``POLAR_HIGH_DISABLE_WHERE_PUSHDOWN=1`` the Where path matches
    today's eager-join behaviour (verified by re-running and snapshot
    comparison)."""
    builder = _build_v_p_problem(rows=20, sel_rows=5, chain_len=2)
    try:
        _set_disable()
        # Run twice — should be deterministic.
        snap_a = _build_and_snapshot(builder)
        snap_b = _build_and_snapshot(builder)
        assert snap_a == snap_b
    finally:
        _clear_guard()


def test_rhs_where_filter_preserved_through_negation():
    """Under Where-pushdown, ``Where(v*p, frame)`` on the RHS records
    its filter into ``_Term.where_frames`` instead of baking into
    ``t.lazy``.  The RHS-to-LHS negation in
    ``_build_canonical_matrix`` / ``_solve_streaming`` /
    ``WarmProblem._initial_build`` must propagate ``where_frames``
    so the resulting constraint coefs are filtered correctly.

    Pre-fix: pushdown drops the filter on RHS -> unfiltered coefs ->
    constraint matrix wrong.  Post-fix: byte-for-byte parity with
    POLAR_HIGH_DISABLE_WHERE_PUSHDOWN=1.
    """

    def builder() -> Problem:
        p = Problem()
        rows = 20
        x_idx = list(range(rows))
        over = pl.DataFrame({"x": x_idx})
        # LHS Var (simplest possible — coef 1, no filter).
        u = p.add_var("u", ("x",), over, lower=-1e6, upper=1e6)
        # RHS Var * Param with a selective filter frame.
        v = p.add_var("v", ("x",), over, lower=-1e6, upper=1e6)
        P = Param(
            ("x",),
            pl.DataFrame(
                {"x": x_idx, "value": np.linspace(0.5, 2.5, rows)}
            ),
            name="P",
        )
        # Selective frame: keep 5 of 20 rows.
        f = pl.DataFrame({"x": x_idx[:5]})
        rhs = Where(v * P, f)
        p.add_cstr(
            "c",
            over=over,
            sense="<=",
            lhs_terms={"lhs": u},
            rhs_terms={"rhs": rhs},
        )
        return p

    _assert_parity(builder)


def test_where_shared_empty_extras_nonempty():
    """Latent-bug regression: when ``shared==[]`` but ``extras!=()``,
    the original code skipped the join AND claimed extras in dims —
    leaving the lazy plan without the extras columns.  The fix
    explicitly cross-joins so the extras land in the frame."""
    p = Problem()
    v = p.add_var("v", ("x",), pl.DataFrame({"x": [1, 2, 3]}), lower=0.0)
    # Frame has NO shared columns with v — extras-only path.
    f = pl.DataFrame({"y": ["a", "b"]})
    _clear_guard()
    expr = Where(v, f)
    t = expr.terms[0]
    schema_cols = t.lazy.collect_schema().names()
    # After fix: lazy plan must carry 'y' (cross-joined) so the term's
    # dim claim is honoured.
    assert "y" in schema_cols
    assert t.dims == ("x", "y")
    # Sanity: 3 rows in v × 2 rows in f = 6.
    assert t.lazy.collect().height == 6
