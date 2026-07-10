"""Scale-invariant small-coefficient cutoff under Layer-2 autoscaling.

Regression for the commodity-ladder mis-solve: Layer 2 is a lossless
power-of-two conditioning transform, so ``coef_zero_threshold`` must
judge a coefficient's negligibility on its USER-space magnitude, not on
the scaled value.  Before the fix, a structurally-essential coefficient
of ``1.0`` whose column/row factors shrank it to ``2**-14`` was floored
to ``0.0`` — decoupling variables and silently corrupting the LP (the H2
``commodity_ladder_balance`` collapsed to ``0 == flow``, forcing the
supply passthrough to 0 and all demand onto slack, ~10,000x cost error).

The floor bakes into ``_CanonicalMatrix.val`` (non-streaming) and the
streaming ``addRows`` path identically; these tests pin the non-streaming
build.  ``matrix[i,j]`` is baked as ``user * row_factor * col_factor``
(see ``_build_canonical_matrix``), so the user magnitude is
``|scaled| / (rf*cf)``.
"""

from __future__ import annotations

import numpy as np
import polars as pl

import polar_high as fp


def _balance_problem(
    user_coef: float,
    threshold: float,
    *,
    col_pow: int | None = None,
    row_pow: int | None = None,
) -> fp.Problem:
    """``user_coef * v_flow == 10`` over a 3-step index, objective ``v_flow``.

    When ``col_pow`` / ``row_pow`` are given, install Layer-2 side vectors
    ``col_factor = 2**col_pow`` / ``row_factor = 2**row_pow`` so the baked
    matrix coefficient becomes ``user_coef * 2**(col_pow+row_pow)``.
    """
    n_t = 3
    p = fp.Problem()
    t_idx = pl.DataFrame({"t": np.arange(n_t, dtype=np.int64)})
    v = p.add_var("v_flow", "t", t_idx, lower=0.0, upper=1.0e6)
    coef_p = fp.Param(
        ("t",),
        pl.DataFrame({"t": np.arange(n_t, dtype=np.int64), "value": np.full(n_t, user_coef)}),
    )
    demand_p = fp.Param(
        ("t",),
        pl.DataFrame({"t": np.arange(n_t, dtype=np.int64), "value": np.full(n_t, 10.0)}),
    )
    p.add_cstr(
        "balance",
        over=t_idx,
        sense="==",
        lhs_terms={"coef_flow": coef_p * v},
        rhs_terms={"demand": demand_p},
    )
    p.set_objective(
        fp.Param(("t",), pl.DataFrame({"t": np.arange(n_t, dtype=np.int64), "value": np.ones(n_t)}))
        * v,
        sense="min",
    )
    p.coef_zero_threshold = threshold
    if col_pow is not None or row_pow is not None:
        n_cols = int(p._next_col)
        n_rows = int(p.cstr_row_count("balance"))
        p._layer2_col_factor = np.full(n_cols, 2.0 ** (col_pow or 0), dtype=np.float64)
        p._layer2_row_factor = np.full(n_rows, 2.0 ** (row_pow or 0), dtype=np.float64)
    return p


def test_layer2_floor_preserves_user_significant_coef():
    """A user coefficient of 1.0 scaled to 2**-14 (< 1e-4) must SURVIVE
    (user magnitude 1.0 >= threshold) — this is the exact bug class."""
    p = _balance_problem(user_coef=1.0, threshold=1e-4, col_pow=-7, row_pow=-7)
    m = p.canonicalise()
    scaled = 2.0**-14  # 1.0 * 2**-7 * 2**-7
    assert m.val.size == 3, m.val
    assert np.all(m.val != 0.0), f"structural coef floored to 0: {m.val}"
    assert np.allclose(m.val, scaled, rtol=0, atol=0), m.val


def test_layer2_floor_still_drops_user_negligible_coef():
    """A genuinely tiny user coefficient (1e-8) must still be floored even
    when scaling inflates it above the threshold — the cutoff is judged in
    user space, not scaled space."""
    p = _balance_problem(user_coef=1e-8, threshold=1e-4, col_pow=10, row_pow=10)
    m = p.canonicalise()
    # Scaled value 1e-8 * 2**20 ~= 1e-2 (>> threshold) but user 1e-8 < 1e-4.
    assert np.all(m.val == 0.0), f"user-negligible coef survived: {m.val}"


def test_floor_without_layer2_is_unchanged():
    """No Layer-2 factors ⇒ user space == scaled space: a 1.0 survives and
    a 1e-8 is floored, exactly as before the scale-invariance change."""
    m_big = _balance_problem(user_coef=1.0, threshold=1e-4).canonicalise()
    assert np.all(m_big.val == 1.0), m_big.val
    m_tiny = _balance_problem(user_coef=1e-8, threshold=1e-4).canonicalise()
    assert np.all(m_tiny.val == 0.0), m_tiny.val


def test_floor_off_is_noop_under_layer2():
    """threshold 0.0 (default) never floors, regardless of Layer-2 factors."""
    p = _balance_problem(user_coef=1e-8, threshold=0.0, col_pow=-7, row_pow=-7)
    m = p.canonicalise()
    assert np.all(m.val != 0.0), m.val
    assert np.allclose(m.val, 1e-8 * 2.0**-14), m.val
