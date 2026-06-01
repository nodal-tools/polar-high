"""Regression tests for the prune-down anonymous-Param + scalar-fold bug.

Background
----------
The RHS prune-down (commit 7ed01ab) and LHS prune-down (commit 1f39301)
rebuild Param chains by walking ``_sources`` one atomic at a time and
joining each atomic to a row_index-bounded accumulator.  Two
factor classes were initially lost in the rebuild — producing the
wrong coefficient compared to the merged-lazy ``Param.__mul__`` path —
which caused ~2% LP objective drift on 25 FlexTool scenario tests:

1. **Anonymous-Param drop** — ``Param._sources_for_propagation``
   returned ``None`` for an anonymous atomic Param (``name is None``),
   so a chain like ``named_a * named_b * anon_c`` registered only
   ``_sources=[(a,+1), (b,+1)]``.  The prune-down rebuilt the chain
   without ``c``'s contribution, while the merged ``.lazy`` still
   carried it.

2. **Scalar fold ignored** — ``Param.__mul__(int/float)``,
   ``Param.__truediv__(int/float)``, and ``Param.__neg__`` (alias for
   ``* -1.0``) fold a constant scalar into the value column while
   leaving ``_sources`` unchanged.  Symmetrically, ``Var.__mul__(float)``
   / ``Expr.__mul__(float)`` / ``Expr.__sub__`` / ``Expr.__neg__`` fold
   a scalar into the term's coef column without updating
   ``param_sources``.  The prune-down's per-atomic walk ignored these
   scalars and produced unsigned / unscaled coefficients.

Both classes are now tracked: ``Param._value_scalar`` accumulates
scalar Param-level folds; ``_Term.coef_scalar`` accumulates scalar
Expr/Var/Term-level folds.  The RHS prune-down seeds its accumulator
with ``rhs._value_scalar``; the LHS prune-down (``_build_lhs_pruned_plan``)
seeds with the term's ``coef_scalar``.  When a Param is multiplied into
a Var/Expr, the Param's ``_value_scalar`` is folded into the resulting
term's ``coef_scalar``.

Both flextool regressions and a focused synthetic case are covered
below.  See also: ``tests/test_canonicalise_param_chain_prune.py`` and
``tests/test_lhs_param_chain_prune.py`` for the original prune-down
parity tests.
"""

from __future__ import annotations

import os

import polars as pl
import pytest

from polar_high.engine import Param, Problem


def _clear_guard():
    os.environ.pop("POLAR_HIGH_DISABLE_PRUNE_DOWN", None)


def test_rhs_prune_down_includes_anonymous_param_in_chain():
    """RHS chain ``named_a * named_b * anonymous_c`` must use ``c``'s
    contribution under the prune-down walk."""
    _clear_guard()
    a = Param(("x",), pl.DataFrame({"x": [1, 2], "value": [2.0, 3.0]}), name="a")
    b = Param(("x",), pl.DataFrame({"x": [1, 2], "value": [5.0, 7.0]}), name="b")
    c_anon = Param(("x",), pl.DataFrame({"x": [1, 2], "value": [0.5, 0.5]}))

    chain = a * b * c_anon
    assert chain._sources is not None
    assert len(chain._sources) == 3, (
        "anonymous c must now appear in _sources for prune-down "
        f"to walk it; got {[(s.name, d) for s, d in chain._sources]}"
    )

    prob = Problem()
    v = prob.add_var("v", ("x",), pl.DataFrame({"x": [1, 2]}), lower=0.0)
    prob.add_cstr(
        "c",
        over=pl.DataFrame({"x": [1, 2]}),
        sense="<=",
        lhs_terms={"v": v},
        rhs_terms={"k": chain},
    )
    m = prob._build_canonical_matrix()
    # 2*5*0.5 = 5.0; 3*7*0.5 = 10.5
    assert list(m.row_ub) == pytest.approx([5.0, 10.5])


def test_rhs_prune_down_honours_scalar_fold():
    """``Param * 60.0`` outside the chain must contribute to the
    pruned-down RHS.  Matches the flextool ``ramp_*_constraint``
    pattern where the RHS is ``ramp_speed * 60 * step_dur * existing``."""
    _clear_guard()
    a = Param(
        ("x",),
        pl.DataFrame({"x": [1, 2], "value": [0.001, 0.001]}),
        name="ramp_speed",
    )
    b = Param(
        ("x",),
        pl.DataFrame({"x": [1, 2], "value": [1.0, 1.0]}),
        name="step_dur",
    )
    # (ramp_speed * 60) is anonymous-composite; ``_value_scalar`` carries 60.
    chain = (a * 60.0) * b
    assert chain._value_scalar == pytest.approx(60.0)

    prob = Problem()
    v = prob.add_var("v", ("x",), pl.DataFrame({"x": [1, 2]}), lower=0.0)
    prob.add_cstr(
        "c",
        over=pl.DataFrame({"x": [1, 2]}),
        sense="<=",
        lhs_terms={"v": v},
        rhs_terms={"k": chain},
    )
    m = prob._build_canonical_matrix()
    # 0.001 * 60 * 1.0 = 0.06
    assert list(m.row_ub) == pytest.approx([0.06, 0.06])


def test_rhs_prune_down_honours_negation():
    """``-Param * a * b`` (i.e. value_scalar=-1) propagates through the
    prune-down."""
    _clear_guard()
    a = Param(("x",), pl.DataFrame({"x": [1, 2], "value": [2.0, 3.0]}), name="a")
    b = Param(("x",), pl.DataFrame({"x": [1, 2], "value": [5.0, 7.0]}), name="b")
    chain = (-a) * b  # -a is a*-1 → composite, _value_scalar=-1
    assert chain._value_scalar == pytest.approx(-1.0)

    prob = Problem()
    v = prob.add_var("v", ("x",), pl.DataFrame({"x": [1, 2]}), lower=-100.0)
    prob.add_cstr(
        "c",
        over=pl.DataFrame({"x": [1, 2]}),
        sense=">=",
        lhs_terms={"v": v},
        rhs_terms={"k": chain},
    )
    m = prob._build_canonical_matrix()
    # -2*5 = -10; -3*7 = -21
    assert list(m.row_lb) == pytest.approx([-10.0, -21.0])


def test_lhs_prune_down_honours_negation():
    """``-Var * P1 * P2`` LHS prune-down must keep the negation."""
    _clear_guard()
    a = Param(("x",), pl.DataFrame({"x": [1, 2], "value": [0.5, 0.5]}), name="P1")
    b = Param(("x",), pl.DataFrame({"x": [1, 2], "value": [0.01, 0.01]}), name="P2")

    prob = Problem()
    v = prob.add_var("v_state", ("x",), pl.DataFrame({"x": [1, 2]}), lower=0.0, upper=10.0)
    # ``-v`` is ``v.to_expr() * -1.0``; ``Expr.__mul__(float)`` is the
    # path that tracks ``coef_scalar``.
    lhs = (-v) * a * b
    assert lhs.terms[0].coef_scalar == pytest.approx(-1.0)
    prob.add_cstr(
        "c",
        over=pl.DataFrame({"x": [1, 2]}),
        sense="<=",
        lhs_terms={"lhs": lhs},
        rhs_terms={"rhs": 0.0},
    )
    m = prob._build_canonical_matrix()
    # Two non-zero coefs in column 0, 1 row each: -1 * 0.5 * 0.01 = -0.005
    assert list(m.val) == pytest.approx([-0.005, -0.005])


def test_lhs_prune_down_honours_param_value_scalar():
    """``Var * (Param * 60) * Param``: prune-down must apply the Param-
    side scalar 60.  Matches the flextool storage_self_discharge-style
    case where one of the Params in the LHS chain has been scaled."""
    _clear_guard()
    a = Param(("x",), pl.DataFrame({"x": [1, 2], "value": [0.001, 0.001]}), name="a")
    b = Param(("x",), pl.DataFrame({"x": [1, 2], "value": [1.0, 1.0]}), name="b")
    scaled = (a * 60.0) * b  # _value_scalar=60, _sources=[(a,+1),(b,+1)]
    assert scaled._value_scalar == pytest.approx(60.0)

    prob = Problem()
    v = prob.add_var("v", ("x",), pl.DataFrame({"x": [1, 2]}), lower=0.0, upper=10.0)
    lhs = v * scaled
    # Var.__mul__(Param) now folds Param._value_scalar into _Term.coef_scalar
    assert lhs.terms[0].coef_scalar == pytest.approx(60.0)
    prob.add_cstr(
        "c",
        over=pl.DataFrame({"x": [1, 2]}),
        sense="<=",
        lhs_terms={"lhs": lhs},
        rhs_terms={"rhs": 0.0},
    )
    m = prob._build_canonical_matrix()
    # 60 * 0.001 * 1.0 = 0.06 per row
    assert list(m.val) == pytest.approx([0.06, 0.06])


def test_disable_guard_still_recovers_merged_path():
    """``POLAR_HIGH_DISABLE_PRUNE_DOWN=1`` continues to bypass both LHS
    and RHS prune-down so callers retain a fallback if a future
    chain pattern produces unexpected drift."""
    a = Param(("x",), pl.DataFrame({"x": [1, 2], "value": [2.0, 3.0]}), name="a")
    b = Param(("x",), pl.DataFrame({"x": [1, 2], "value": [5.0, 7.0]}), name="b")
    c_anon = Param(("x",), pl.DataFrame({"x": [1, 2], "value": [0.5, 0.5]}))
    chain = a * b * c_anon

    try:
        os.environ["POLAR_HIGH_DISABLE_PRUNE_DOWN"] = "1"
        prob = Problem()
        v = prob.add_var("v", ("x",), pl.DataFrame({"x": [1, 2]}), lower=0.0)
        prob.add_cstr(
            "c",
            over=pl.DataFrame({"x": [1, 2]}),
            sense="<=",
            lhs_terms={"v": v},
            rhs_terms={"k": chain},
        )
        m = prob._build_canonical_matrix()
        # Under the guard the merged-lazy path computes the RHS via
        # ``rhs.lazy`` directly — the anonymous-Param contribution is
        # already in the value column.
        assert list(m.row_ub) == pytest.approx([5.0, 10.5])
    finally:
        os.environ.pop("POLAR_HIGH_DISABLE_PRUNE_DOWN", None)
