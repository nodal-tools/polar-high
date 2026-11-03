"""Objective (cost) coefficient cutoff.

``Problem.obj_coef_zero_threshold`` (absolute) and
``Problem.obj_coef_zero_rel_threshold`` (relative to the largest user-space
``|cost|``) zero a column's summed objective coefficient whose user-space
magnitude is strictly below

    max(abs, 2**floor(log2(rel * max|c|)))

The matrix / RHS floor ``coef_zero_threshold`` never touches the cost vector,
so before this a float residue such as ``1e-17`` (an algebraic cancellation
``1/eta - slope``) reached the objective, stretched the cost range to ~17
decades and made Layer 3 centre the objective with ``user_objective_scale``
2**25+; HiGHS' dual simplex then failed ("excessive dual values").

These tests pin: default off (byte-identical), absolute and relative
semantics (power-of-two rounding, equality kept), the user-space judgement
under Layer-2 side vectors, every consumer (canonical matrix / write_mps,
streaming solve + its streamed ranges, both pre-solve range readouts, the
WarmProblem cost updates) and that Layer 3's centring ignores sub-cutoff
noise.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

import polar_high as fp
from polar_high.autoscale import ScalingConfig, detect_ranges, recommend_scaling
from polar_high.autoscale._ranges import _ranges_via_passmodel
from polar_high.engine import _objective_zero_floor

# The cost vector of the toy LP: two "real" costs, a legitimately small one
# and a float residue.  Column order = declaration order.
REAL = 2.0
SMALL = 1e-6  # legitimate (e.g. a cheap slack) — 6.3 decades below REAL
RESIDUE = 1e-17


def _problem(costs: list[float], *, dense: bool = False) -> fp.Problem:
    """``sum_t x[t] >= 1`` with ``min sum_t c[t] x[t]``, ``0 <= x <= 10``."""
    n = len(costs)
    p = fp.Problem(dense_axes=("t",)) if dense else fp.Problem()
    t_idx = pl.DataFrame({"t": np.arange(n, dtype=np.int64)})
    x = p.add_var("x", "t", t_idx, lower=0.0, upper=10.0)
    one = fp.Param(("t",), pl.DataFrame({"t": np.arange(n, dtype=np.int64), "value": np.ones(n)}))
    p.add_cstr(
        "cover",
        over=None,
        sense=">=",
        lhs_terms={"x": fp.Sum(one * x, over="t")},
        rhs_terms={"one": 1.0},
    )
    cost = fp.Param(
        ("t",),
        pl.DataFrame(
            {"t": np.arange(n, dtype=np.int64), "value": np.asarray(costs, dtype=np.float64)}
        ),
        name="p_cost",
    )
    p.set_objective(cost * x, sense="min")
    return p


# --------------------------------------------------------------------------
# The cutoff value
# --------------------------------------------------------------------------


def test_floor_value_off_absolute_relative_and_combined():
    assert _objective_zero_floor(0.0, 0.0, 5.0) == 0.0
    assert _objective_zero_floor(1e-3, 0.0, 5.0) == 1e-3
    # relative: 1e-14 * 2 = 2e-14 rounds DOWN to the power of two 2**-46.
    f = _objective_zero_floor(0.0, 1e-14, 2.0)
    assert f == 2.0**-46 and f <= 2e-14 < 2 * f
    # an exact power of two is its own floor
    assert _objective_zero_floor(0.0, 0.25, 4.0) == 1.0
    # combined: the larger part wins
    assert _objective_zero_floor(1e-3, 1e-14, 2.0) == 1e-3
    assert _objective_zero_floor(1e-20, 1e-14, 2.0) == 2.0**-46
    # no usable reference ⇒ relative part off
    assert _objective_zero_floor(0.0, 1e-14, 0.0) == 0.0
    assert _objective_zero_floor(0.0, 1e-14, math.inf) == 0.0


# --------------------------------------------------------------------------
# Canonical matrix (write_mps / passModel / WarmProblem)
# --------------------------------------------------------------------------


def test_default_off_keeps_every_cost():
    p = _problem([REAL, SMALL, RESIDUE])
    assert p.obj_coef_zero_threshold == 0.0
    assert p.obj_coef_zero_rel_threshold == 0.0
    m = p.canonicalise()
    assert m.col_obj.tolist() == [REAL, SMALL, RESIDUE]
    assert p._obj_zero_floor == 0.0


def test_relative_cutoff_drops_residue_keeps_small_real_cost():
    p = _problem([REAL, SMALL, RESIDUE])
    p.obj_coef_zero_rel_threshold = 1e-14
    m = p.canonicalise()
    assert m.col_obj.tolist() == [REAL, SMALL, 0.0]
    assert p._obj_zero_floor == 2.0**-46


def test_relative_cutoff_is_power_of_two_rounded():
    # rel * max = 3e-14 → floor 2**-45 ≈ 2.84e-14: 2.9e-14 kept, 2.8e-14 cut.
    p = _problem([3.0, 2.9e-14, 2.8e-14])
    p.obj_coef_zero_rel_threshold = 1e-14
    assert p.canonicalise().col_obj.tolist() == [3.0, 2.9e-14, 0.0]


def test_absolute_cutoff_keeps_equal_value():
    p = _problem([REAL, 1e-3, 9.9e-4, -9.9e-4])
    p.obj_coef_zero_threshold = 1e-3
    assert p.canonicalise().col_obj.tolist() == [REAL, 1e-3, 0.0, 0.0]


def test_relative_reference_is_largest_cost_not_a_constant():
    # Scaling the whole objective scales the cutoff with it: same pattern.
    for scale in (1e-6, 1.0, 1e6):
        p = _problem([REAL * scale, SMALL * scale, RESIDUE * scale])
        p.obj_coef_zero_rel_threshold = 1e-14
        assert p.canonicalise().col_obj.tolist() == [REAL * scale, SMALL * scale, 0.0]


def test_cutoff_judged_on_summed_column_cost():
    # Two terms on the same column whose sum is a residue: 0.3 + (−0.3 + 1e-17).
    n = 2
    p = fp.Problem()
    t_idx = pl.DataFrame({"t": np.arange(n, dtype=np.int64)})
    x = p.add_var("x", "t", t_idx, lower=0.0, upper=10.0)
    c1 = fp.Param(("t",), pl.DataFrame({"t": [0, 1], "value": [REAL, 0.3]}))
    c2 = fp.Param(("t",), pl.DataFrame({"t": [0, 1], "value": [0.0, -0.3 + 5.551115123125783e-17]}))
    p.set_objective(c1 * x + c2 * x, sense="min")
    p.obj_coef_zero_rel_threshold = 1e-14
    col_obj = p.canonicalise().col_obj
    assert col_obj[0] == REAL
    assert col_obj[1] == 0.0  # 0.3 − 0.3 + residue ≈ 5.6e-17 → cut


def test_write_mps_omits_cut_cost(tmp_path):
    p = _problem([REAL, SMALL, RESIDUE])
    p.obj_coef_zero_rel_threshold = 1e-14
    path = tmp_path / "lp.mps"
    p.write_mps(str(path), emit_names=True)
    lines = path.read_text().splitlines()
    obj_row = next(ln.split()[1] for ln in lines if ln.startswith(" N "))
    cost_cols = {ln.split()[0] for ln in lines if len(ln.split()) >= 3 and ln.split()[1] == obj_row}
    # The residue column carries no objective entry; the others do.
    assert cost_cols == {"x[0]", "x[1]"}


# --------------------------------------------------------------------------
# Layer-2 side vectors: judged in user space
# --------------------------------------------------------------------------


@pytest.mark.parametrize("col_pow", [-40, 40])
def test_cutoff_is_scale_invariant_under_layer2(col_pow):
    p = _problem([REAL, SMALL, RESIDUE])
    p.obj_coef_zero_rel_threshold = 1e-14
    cf = np.ones(int(p._next_col))
    # Shrink the small real cost below the cutoff / inflate the residue
    # above it in SCALED space — the user-space decision must not change.
    cf[1] = 2.0 ** min(col_pow, 0)
    cf[2] = 2.0 ** max(col_pow, 0)
    p._layer2_col_factor = cf
    m = p.canonicalise()
    assert m.col_obj[0] == REAL
    assert m.col_obj[1] == SMALL * cf[1]
    assert m.col_obj[2] == 0.0


# --------------------------------------------------------------------------
# Streaming solve + its streamed ranges
# --------------------------------------------------------------------------


def test_streaming_solve_ranges_exclude_cut_cost():
    p = _problem([REAL, SMALL, RESIDUE])
    sol_off = p.solve()
    assert sol_off.streamed_lp_ranges["cost"] == (RESIDUE, REAL)

    p = _problem([REAL, SMALL, RESIDUE])
    p.obj_coef_zero_rel_threshold = 1e-14
    sol = p.solve()
    assert sol.optimal
    assert sol.streamed_lp_ranges["cost"] == (SMALL, REAL)
    # x[2] is free of cost now; the optimum is unchanged (0 either way at
    # this precision: cover with the zero-cost column).
    assert sol.obj == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------
# Pre-solve range readouts → Layer 3 centring
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize("layer2", [False, True])
def test_presolve_readout_and_layer3_ignore_sub_cutoff_noise(dense, layer2, capfd, monkeypatch):
    def build(rel: float) -> fp.Problem:
        p = _problem([REAL, REAL, RESIDUE], dense=dense)
        p.obj_coef_zero_rel_threshold = rel
        if layer2:
            p._layer2_col_factor = np.full(int(p._next_col), 2.0**3)
        return p

    cfg = ScalingConfig()
    k = 2.0**3 if layer2 else 1.0

    r_off = detect_ranges(build(0.0), cfg)
    assert r_off.cost == (RESIDUE * k, REAL * k)
    n_off = recommend_scaling(r_off, cfg).user_objective_scale
    assert n_off >= 25  # the residue drives the centring

    r_on = detect_ranges(build(1e-14), cfg)
    assert r_on.cost == (REAL * k, REAL * k)
    if dense and layer2:
        # this case reads the objective through the bounded positional path
        captured = capfd.readouterr()
        monkeypatch.setenv("POLAR_HIGH_BLOCK_COO_PROFILE", "1")
        detect_ranges(build(1e-14), cfg)
        assert "path=obj_positional" in capfd.readouterr().err, captured
    plan = recommend_scaling(r_on, cfg)
    assert plan.user_objective_scale == 0  # in zone: centring unaffected


def test_presolve_readout_keeps_real_small_cost():
    cfg = ScalingConfig()
    p = _problem([REAL, SMALL, RESIDUE])
    p.obj_coef_zero_rel_threshold = 1e-14
    r = detect_ranges(p, cfg)
    assert r.cost == (SMALL, REAL)
    # Centred on the real band [1e-6, 2], as without the residue.
    q = _problem([REAL, SMALL])
    assert (
        recommend_scaling(r, cfg).user_objective_scale
        == recommend_scaling(detect_ranges(q, cfg), cfg).user_objective_scale
    )


def test_presolve_readout_absolute_cutoff():
    cfg = ScalingConfig()
    p = _problem([REAL, 1e-3, 9.9e-4])
    p.obj_coef_zero_threshold = 1e-3
    assert detect_ranges(p, cfg).cost == (1e-3, REAL)


def test_passmodel_readout_matches():
    cfg = ScalingConfig()
    p = _problem([REAL, SMALL, RESIDUE])
    assert _ranges_via_passmodel(p, cfg).cost == (RESIDUE, REAL)
    p = _problem([REAL, SMALL, RESIDUE])
    p.obj_coef_zero_rel_threshold = 1e-14
    assert _ranges_via_passmodel(p, cfg).cost == (SMALL, REAL)


# --------------------------------------------------------------------------
# WarmProblem
# --------------------------------------------------------------------------


class _CostSpy:
    def __init__(self, h):
        self._h = h
        self.costs: list[list[float]] = []

    def changeColsCost(self, n, idx, cost):  # noqa: N802 (mirror HiGHS API)
        self.costs.append([float(c) for c in cost])
        return self._h.changeColsCost(n, idx, cost)

    def __getattr__(self, name):
        return getattr(self._h, name)


def test_warm_initial_build_and_cost_update_floor():
    p = _problem([REAL, SMALL, RESIDUE])
    p.obj_coef_zero_rel_threshold = 1e-14
    wp = fp.WarmProblem(p)
    wp.solve()
    assert list(wp._h.getLp().col_cost_) == [REAL, SMALL, 0.0]

    spy = _CostSpy(wp._h)
    wp._h = spy
    wp.update_obj_coef_array("x", [(0,), (1,), (2,)], np.array([REAL, RESIDUE, SMALL]))
    assert spy.costs[-1] == [REAL, 0.0, SMALL]
    wp.update_obj_coef("x", 1e-18)
    assert spy.costs[-1] == [0.0, 0.0, 0.0]


def test_warm_cost_update_default_off_unchanged():
    p = _problem([REAL, SMALL, RESIDUE])
    wp = fp.WarmProblem(p)
    wp.solve()
    assert list(wp._h.getLp().col_cost_) == [REAL, SMALL, RESIDUE]
    spy = _CostSpy(wp._h)
    wp._h = spy
    wp.update_obj_coef("x", 1e-18)
    assert spy.costs[-1] == [1e-18, 1e-18, 1e-18]
