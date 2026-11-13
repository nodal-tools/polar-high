"""Objective-valued solver options in user units.

Expected values are hand-computed from ``q = f * (k * v - O) + O_solver``
(levels; negated for HiGHS ``objective_bound`` under ``max``) and
``q = f * k * g`` (differences), with ``f = 2**N`` for HiGHS only.
"""

from __future__ import annotations

import math
import sys
import types
from typing import NamedTuple

import highspy
import numpy as np
import polars as pl
import pytest

import polar_high as fp
from polar_high import solvers as ph_solvers
from polar_high.objective_options import (
    ObjectiveTransform,
    objective_option_kind,
    solver_keeps_offset,
    translate_objective_options,
)

# ---------------------------------------------------------------------------
# Pure translation (no solve)
# ---------------------------------------------------------------------------


def _scaled_problem(k: float = 1e-6, offset: float = 0.0, sense: str = "min") -> fp.Problem:
    p = fp.Problem()
    idx = pl.DataFrame({"i": [0]})
    x = p.add_var("x", "i", idx, lower=0.0, upper=1.0)
    p.set_objective(x, sense=sense)
    if offset:
        p.add_obj_constant(offset)
    p.set_objective_scale(k)
    return p


def test_highs_levels_and_gaps_scaled_by_k_and_two_to_the_n() -> None:
    p = _scaled_problem(1e-6)
    opts = {
        "objective_target": -2e9,
        "mip_abs_gap": 5e5,
        "objective_bound": math.inf,
        "mip_rel_gap": 0.01,
        "user_objective_scale": 4,
    }
    tr = p.objective_transform("highs", "direct", opts)
    out, recs = translate_objective_options("highs", opts, tr)
    assert out["objective_target"] == -32000.0  # 16 * 1e-6 * -2e9
    assert out["mip_abs_gap"] == 8.0  # 16 * 1e-6 * 5e5
    assert out["objective_bound"] == math.inf
    assert out["mip_rel_gap"] == 0.01
    assert out["user_objective_scale"] == 4
    assert list(out) == list(opts)  # order kept
    assert [r.name for r in recs] == ["objective_target", "mip_abs_gap"]
    assert recs[0].user_value == -2e9 and recs[0].solver_value == -32000.0
    # input untouched
    assert opts["objective_target"] == -2e9


def test_gurobi_mps_strings_and_pass_through() -> None:
    p = _scaled_problem(1e-6)
    # A HiGHS user_objective_scale on the Problem never reaches a Gurobi factor.
    p.set_solver_options({"user_objective_scale": 6})
    tr = p.objective_transform("gurobi", "mps")
    assert tr.solver_factor == 1.0
    out, recs = translate_objective_options(
        "gurobi",
        {"BestObjStop": "1.5e9", "MIPGapAbs": "1e5", "Cutoff": 1e100, "MIPGap": "0.01"},
        tr,
    )
    assert out["BestObjStop"] == "1500.0"
    assert out["MIPGapAbs"] == "0.1"  # 15 significant digits, not 0.09999999999999999
    assert out["Cutoff"] == 1e100
    assert out["MIPGap"] == "0.01"
    assert [r.name for r in recs] == ["BestObjStop", "MIPGapAbs"]


def test_cplex_and_key_spellings() -> None:
    p = _scaled_problem(1e-6)
    tr = p.objective_transform("cplex", "mps")
    out, _ = translate_objective_options(
        "cplex",
        {"mip.tolerances.uppercutoff": 2e9, "mip__tolerances__absmipgap": 3e6},
        tr,
    )
    assert out["mip.tolerances.uppercutoff"] == 2000.0
    assert out["mip__tolerances__absmipgap"] == 3.0
    assert objective_option_kind("gurobi", "MIP_Gap_Abs") == "diff"
    assert objective_option_kind("gurobi", "bestobjstop") == "value"
    assert objective_option_kind("xpress", "XPRS_MIPABSSTOP") == "diff"
    assert objective_option_kind("xpress", "mipabscutoff") == "value"
    assert objective_option_kind("copt", "absgap") == "diff"
    assert objective_option_kind("cplex", "mip limits upperobjstop") == "value"
    assert objective_option_kind("cplex", "mip tolerances mipgap") is None
    assert objective_option_kind("highs", "Objective_Bound") is None  # HiGHS names are exact
    assert objective_option_kind("highs", "user_objective_scale") is None


def test_sequence_of_named_tuples_keeps_type_and_fields() -> None:
    class Opt(NamedTuple):
        name: str
        value: str
        layer: str

    tr = ObjectiveTransform(objective_scale=1e-6)
    items = [Opt("TimeLimit", "60", "cli"), Opt("BestObjStop", "1.5e9", "db")]
    out, recs = translate_objective_options("gurobi", items, tr)
    assert out[0] is items[0]
    assert out[1] == Opt("BestObjStop", "1500.0", "db")
    assert isinstance(out[1], Opt)
    assert [r.name for r in recs] == ["BestObjStop"]


def test_offset_per_solver_and_io() -> None:
    # k = 1, O = 5: Gurobi / CPLEX direct keep the constant (q = v);
    # Xpress direct, every MPS solve and HiGHS do not (q = v - 5).
    p = _scaled_problem(1.0, offset=5.0)
    assert p.objective_transform("gurobi", "direct").value(100.0) == 100.0
    assert p.objective_transform("cplex", "direct").value(100.0) == 100.0
    assert p.objective_transform("xpress", "direct").value(100.0) == 95.0
    assert p.objective_transform("gurobi", "mps").value(100.0) == 95.0
    assert p.objective_transform("highs", "direct").value(100.0) == 95.0
    # COPT in-process keeps it, but routes via copt_cmd once highspy is loaded.
    assert "highspy" in sys.modules
    assert solver_keeps_offset("copt", "direct") is False
    # differences never see the constant
    assert p.objective_transform("xpress", "direct").diff(7.0) == 7.0


def test_maximisation_sign() -> None:
    # HiGHS reads objective_bound in its minimisation space, objective_target
    # in the objective's sense; Gurobi reads both kinds in the user's sense.
    p = _scaled_problem(1.0, sense="max")
    tr = p.objective_transform("highs", "direct", {"user_objective_scale": 2})
    out, _ = translate_objective_options(
        "highs", {"objective_bound": 10.0, "objective_target": 10.0, "mip_abs_gap": 1.0}, tr
    )
    assert out == {"objective_bound": -40.0, "objective_target": 40.0, "mip_abs_gap": 4.0}
    tr_g = p.objective_transform("gurobi", "direct")
    out_g, _ = translate_objective_options("gurobi", {"Cutoff": 10.0}, tr_g)
    assert out_g == {"Cutoff": 10.0}


def test_identity_mapping_passes_values_untouched() -> None:
    # k = f = 1 and no constant: the values reach the solver as written (no
    # 15-digit rounding, no string rewrite), and nothing is recorded.
    tr = ObjectiveTransform()
    gap = 0.30000000000000004
    opts = {"objective_target": "1e-7", "mip_abs_gap": gap, "objective_bound": 3}
    out, recs = translate_objective_options("highs", opts, tr)
    assert out == opts and out is not opts
    assert out["objective_target"] == "1e-7"
    assert out["mip_abs_gap"] is gap
    assert out["objective_bound"] == 3 and isinstance(out["objective_bound"], int)
    assert recs == []
    # Under max, HiGHS objective_bound is negated: not the identity; the
    # other two still pass untouched.
    out, recs = translate_objective_options("highs", opts, ObjectiveTransform(sense="max"))
    assert out["objective_bound"] == -3.0
    assert out["objective_target"] == "1e-7" and out["mip_abs_gap"] is gap
    assert [r.name for r in recs] == ["objective_bound"]
    # A constant the solver holds itself (Gurobi direct, O = O_solver = 5):
    # levels are the identity.  Under "mps" (O_solver = 0) they are not.
    p = _scaled_problem(1.0, offset=5.0)
    out, recs = translate_objective_options(
        "gurobi", {"Cutoff": "1e-7"}, p.objective_transform("gurobi", "direct")
    )
    assert out == {"Cutoff": "1e-7"} and recs == []
    out, _ = translate_objective_options(
        "gurobi", {"Cutoff": "1e-7"}, p.objective_transform("gurobi", "mps")
    )
    assert out == {"Cutoff": repr(1e-7 - 5.0)}
    # Any objective scale (k != 1) or HiGHS factor (f != 1) translates.
    assert not ObjectiveTransform(objective_scale=2.0).is_identity("diff")
    assert not ObjectiveTransform(solver_factor=2.0).is_identity("diff")


def test_set_objective_scale_validation() -> None:
    p = fp.Problem()
    assert p.objective_scale == 1.0
    for bad in (0.0, -1.0, math.inf, math.nan):
        with pytest.raises(ValueError):
            p.set_objective_scale(bad)
    with pytest.raises(ValueError):
        p.objective_transform("glpk")
    with pytest.raises(ValueError):
        p.objective_transform("highs", "lp")


# ---------------------------------------------------------------------------
# Real HiGHS: min x, 5 <= x <= 50 integer, objective constant 100 → opt 105.
# HiGHS internal objective is 2**N * x (the constant is kept out of HiGHS);
# a bound B in user units reaches HiGHS as 2**N * (B - 100).
# ---------------------------------------------------------------------------


def _one_var(n: int | None, bound: float, *, k: float = 1.0, sense: str = "min") -> fp.Problem:
    p = fp.Problem()
    idx = pl.DataFrame({"i": [0]})
    x = p.add_var("x", "i", idx, lower=5.0, upper=50.0, integer=True)
    p.add_cstr("cap", over=idx, sense="<=", lhs_terms={"x": x}, rhs_terms={"r": 50.0})
    if sense == "min":
        p.set_objective(k * x, sense="min")  # user objective x + 100, opt 105
        p.add_obj_constant(100.0 * k)
    else:
        p.set_objective(-k * x, sense="max")  # user objective 100 - x, opt 95
        p.add_obj_constant(100.0 * k)
    p.set_objective_scale(k)
    opts: dict = {"output_flag": False, "threads": 1, "objective_bound": bound}
    if n is not None:
        opts["user_objective_scale"] = n
    p.set_solver_options(opts)
    return p


@pytest.mark.parametrize("n", [None, 4])
@pytest.mark.parametrize("streaming", [True, False])
def test_highs_objective_bound_in_user_units(n: int | None, streaming: bool) -> None:
    # 106 → 2**N * 6 > 2**N * 5: optimal 105.  104 → 2**N * 4 < 2**N * 5:
    # infeasible.  Untranslated, 104 would exceed 16 * 5 = 80 under N = 4.
    sol = _one_var(n, 106.0).solve(streaming=streaming)
    assert sol.optimal and sol.obj == pytest.approx(105.0)
    sol = _one_var(n, 104.0).solve(streaming=streaming)
    assert not sol.optimal


def test_highs_objective_bound_with_objective_scale() -> None:
    # k = 1e-3: HiGHS sees 16 * 1e-3 * x = 0.08 at the optimum; 106 → 0.096
    # (optimal), 104 → 0.064 (infeasible).
    sol = _one_var(4, 106.0, k=1e-3).solve()
    assert sol.optimal
    assert sol.obj == pytest.approx(0.105)
    assert sol.objective_in_user_units == pytest.approx(105.0)
    assert not _one_var(4, 104.0, k=1e-3).solve().optimal


@pytest.mark.parametrize("n", [None, 4])
def test_highs_objective_bound_under_max(n: int | None) -> None:
    # max 100 - x, opt 95 (x = 5).  Bound 94 (below the optimum) keeps it,
    # 96 cuts it off.  HiGHS sees min 2**N * x = 2**N * 5 and gets
    # -2**N * (B - 100): 94 → 2**N * 6, 96 → 2**N * 4.
    sol = _one_var(n, 94.0, sense="max").solve()
    assert sol.optimal and sol.obj == pytest.approx(95.0)
    assert not _one_var(n, 96.0, sense="max").solve().optimal


@pytest.mark.parametrize("streaming", [True, False])
def test_highs_objective_bound_with_save_memory(streaming: bool) -> None:
    # The save_memory streaming solve (MPS round trip inside HiGHS, options
    # re-applied) gets the translated bound: 106 → 96 > 80 optimal,
    # 104 → 64 < 80 infeasible.  ``streaming=False`` ignores save_memory
    # with a warning and must translate the same way.
    def solve(bound: float):
        p = _one_var(4, bound)
        if streaming:
            return p.solve(save_memory=True)
        with pytest.warns(UserWarning, match="save_memory"):
            return p.solve(save_memory=True, streaming=False)

    sol = solve(106.0)
    assert sol.optimal and sol.obj == pytest.approx(105.0)
    assert not solve(104.0).optimal


# ---------------------------------------------------------------------------
# Real HiGHS objective_target: max sum(v_i x_i) + 100 over a 60-item binary
# knapsack, optimum 1797 (variable part 1697); the first incumbent HiGHS
# finds is 1792 (1692).  objective_target is read in the objective's own
# sense (HiGHS stops at an incumbent at least as good as the target) and
# compared in HiGHS' 2**N scaled space without the constant:
# q = 2**N * (k * T - k * 100).
# ---------------------------------------------------------------------------


def _knapsack(target: float, n: int | None, *, k: float = 1.0) -> fp.Problem:
    from polar_high.engine import Param, Sum

    rng = np.random.default_rng(0)
    m = 60
    w = rng.integers(20, 80, m).astype(float)
    v = w + rng.integers(0, 10, m)
    items = list(range(m))
    p = fp.Problem()
    x = p.add_var("x", "i", pl.DataFrame({"i": items}), lower=0, upper=1, integer=True)
    weight = Param(("i",), pl.DataFrame({"i": items, "value": w}))
    value = Param(("i",), pl.DataFrame({"i": items, "value": v * k}))
    p.add_cstr(
        "cap",
        over=None,
        sense="<=",
        lhs_terms={"l": Sum(weight * x, over=("i",))},
        rhs_terms={"r": float(np.floor(w.sum() / 2)) + 0.5},
    )
    p.set_objective(Sum(value * x, over=("i",)), sense="max")
    p.add_obj_constant(100.0 * k)
    p.set_objective_scale(k)
    opts: dict = {
        "output_flag": False,
        "threads": 1,
        "random_seed": 0,
        "mip_rel_gap": 0.0,
        "objective_target": target,
    }
    if n is not None:
        opts["user_objective_scale"] = n
    p.set_solver_options(opts)
    return p


def _target_reached(sol) -> bool:
    h = sol.highs
    return h.getModelStatus() == highspy.HighsModelStatus.kObjectiveTarget


@pytest.mark.parametrize("n", [None, 4])
@pytest.mark.parametrize("streaming", [True, False])
def test_highs_objective_target_under_max(n: int | None, streaming: bool) -> None:
    # 1750 → 2**N * 1650, below the first incumbent 2**N * 1692: reached
    # (were the constant left in, 2**N * 1750 is above the optimum 2**N *
    # 1697 and the solve would run to optimality).
    sol = _knapsack(1750.0, n).solve(keep_solver=True, streaming=streaming)
    assert _target_reached(sol) and not sol.optimal
    # 1798 and 1900 are above the optimum 1797: the solve runs to optimality.
    # Under N = 4, 1900 untranslated would be below 16 * 1697 and stop at the
    # first incumbent; negated (minimisation space) it would too.
    for target in (1798.0, 1900.0):
        sol = _knapsack(target, n).solve(keep_solver=True, streaming=streaming)
        assert sol.optimal and sol.obj == pytest.approx(1797.0)


def test_highs_objective_target_with_objective_scale() -> None:
    # k = 1e-3: the objective passed is 1e-3 x the user objective (opt
    # 1.797); targets stay in user units.
    sol = _knapsack(1750.0, 4, k=1e-3).solve(keep_solver=True)
    assert _target_reached(sol)
    sol = _knapsack(1798.0, 4, k=1e-3).solve(keep_solver=True)
    assert sol.optimal and sol.objective_in_user_units == pytest.approx(1797.0)


def test_constant_is_kept_out_of_highs_and_added_back() -> None:
    p = _one_var(4, math.inf)
    sol = p.solve(keep_solver=True)
    assert sol.optimal
    assert sol.obj == pytest.approx(105.0)
    assert sol.obj_offset == 100.0
    assert sol.highs.getLp().offset_ == 0.0
    assert sol.highs.getObjectiveValue() == pytest.approx(5.0)
    diag = sol.solve_diagnostics()
    assert diag is not None and diag.objective_value == pytest.approx(105.0)


def test_solvers_solve_highs_translates() -> None:
    res = ph_solvers.solve(_one_var(4, 104.0), solver_name="highs")
    assert res.status != fp.SolverStatus.OPTIMAL
    res = ph_solvers.solve(_one_var(4, 106.0), solver_name="highs")
    assert res.status == fp.SolverStatus.OPTIMAL
    assert res.objective == pytest.approx(105.0)


# ---------------------------------------------------------------------------
# WarmProblem: no write-back, fallback options translated with the live N
# ---------------------------------------------------------------------------


def _live(h: highspy.Highs, name: str):
    got = h.getOptionValue(name)
    return got[1] if isinstance(got, tuple) else got


def test_warm_resolve_does_not_translate_twice() -> None:
    p = _one_var(4, 106.0)
    stored = dict(p._solver_options)
    wp = fp.WarmProblem(p)
    s1 = wp.solve()
    v1 = _live(wp._h, "objective_bound")
    s2 = wp.solve()
    v2 = _live(wp._h, "objective_bound")
    assert s1.optimal and s2.optimal
    assert s1.obj == pytest.approx(105.0) and s2.obj == pytest.approx(105.0)
    assert v1 == v2 == 96.0  # 16 * (106 - 100)
    assert p._solver_options == stored


def test_warm_fallback_options_translated_with_live_n() -> None:
    # Primary 104 → 64 < 80: infeasible.  Fallback 104.5 → 72: still
    # infeasible (untranslated 104.5 would be optimal).  Fallback 106 → 96:
    # optimal.  The primary value is restored after each retry.
    wp = fp.WarmProblem(_one_var(4, 104.0))
    sol, how = wp.solve_with_fallback({"objective_bound": 104.5})
    assert how == "fallback" and not sol.optimal
    assert _live(wp._h, "objective_bound") == 64.0
    sol, how = wp.solve_with_fallback({"objective_bound": 106.0})
    assert how == "fallback" and sol.optimal
    assert sol.obj == pytest.approx(105.0)
    assert _live(wp._h, "objective_bound") == 64.0


# ---------------------------------------------------------------------------
# polar_high.solvers.solve: commercial adapters get translated options
# ---------------------------------------------------------------------------


def test_solvers_solve_gurobi_gets_translated_options(monkeypatch) -> None:
    captured: dict = {}

    def fake_run(view, *, env=None, **options):
        captured.update(options)
        return "sentinel"

    fake = types.ModuleType("polar_high.solvers._gurobi")
    fake.run = fake_run
    monkeypatch.setitem(sys.modules, "polar_high.solvers._gurobi", fake)
    monkeypatch.setattr(ph_solvers, "available_solvers", ["gurobi", "highs"])

    # k = 1e-6, O = 5e-6 (5 currency): Gurobi direct keeps ObjCon, so
    # BestObjStop 1.5e9 → 1500.0; MIPGapAbs 1e5 → 0.1; a HiGHS
    # user_objective_scale on the Problem is ignored for Gurobi.
    p = _scaled_problem(1e-6, offset=5e-6)
    p.set_solver_options({"user_objective_scale": 6})
    opts = {"BestObjStop": 1.5e9, "MIPGapAbs": "1e5", "TimeLimit": 60}
    assert ph_solvers.solve(p, solver_name="gurobi", **opts) == "sentinel"
    assert captured == {"BestObjStop": 1500.0, "MIPGapAbs": "0.1", "TimeLimit": 60}
    assert opts["BestObjStop"] == 1.5e9


_CUTOFF_OPTION = {
    ("gurobi", "min"): ("Cutoff", {"OutputFlag": 0}),
    ("gurobi", "max"): ("Cutoff", {"OutputFlag": 0}),
    ("cplex", "min"): ("mip__tolerances__uppercutoff", {}),
    ("cplex", "max"): ("mip__tolerances__lowercutoff", {}),
    ("xpress", "min"): ("MIPABSCUTOFF", {"outputlog": 0}),
    ("xpress", "max"): ("MIPABSCUTOFF", {"outputlog": 0}),
}


@pytest.mark.parametrize("sense", ["min", "max"])
@pytest.mark.parametrize("solver", ["gurobi", "cplex", "xpress"])
def test_commercial_cutoff_in_user_units(solver: str, sense: str) -> None:
    # k = 1e-3, constant 100 currency: Gurobi / CPLEX hold the constant
    # themselves, Xpress gets it added afterwards; all read the cutoff in the
    # objective's sense.  min x + 100 (opt 105): 104 cuts, 106 keeps.
    # max 100 - x (opt 95): 96 cuts, 94 keeps.
    pytest.importorskip({"gurobi": "gurobipy", "cplex": "cplex", "xpress": "xpress"}[solver])
    key, quiet = _CUTOFF_OPTION[(solver, sense)]
    cut, keep = (104.0, 106.0) if sense == "min" else (96.0, 94.0)
    try:
        res_cut = ph_solvers.solve(
            _one_var(None, math.inf, k=1e-3, sense=sense), solver_name=solver, **{key: cut}, **quiet
        )
        res_keep = ph_solvers.solve(
            _one_var(None, math.inf, k=1e-3, sense=sense),
            solver_name=solver,
            **{key: keep},
            **quiet,
        )
    except fp.LicenseError as exc:
        pytest.skip(f"{solver} licence: {exc}")
    assert res_cut.status != fp.SolverStatus.OPTIMAL
    assert res_keep.status == fp.SolverStatus.OPTIMAL
    assert res_keep.objective == pytest.approx(0.105 if sense == "min" else 0.095)
