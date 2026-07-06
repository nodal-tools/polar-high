"""Acceptance gate for :func:`polar_high.benders.solve_benders_loop` — the
generic multicut Benders coordinator (core loop).

Everything here is a HAND-BUILT synthetic 1-master / 2-subproblem LP with an
analytically known optimum — no domain vocabulary, no fixtures.

The toy decomposable problem
----------------------------

Master:  min  1·f1 + 5·f2 + eta1 + eta2,   f1, f2 in [0, 10]
Sub s:   min  gamma_s·g_s   s.t.  g_s + f_s >= d_s,   g_s >= 0
         (f_s pinned by the master; the cut slope is f_s's reduced cost,
          -gamma_s while imports still displace generation, else 0)

with  sub1: d=4, gamma=3   and   sub2: d=6, gamma=2.

Monolith optimum (hand-verified):
  sub1: import is cheaper (1 < 3)  -> f1 = 4, master pays 4, recourse 0
  sub2: import is dearer  (5 > 2)  -> f2 = 0, recourse 2·6 = 12
  total = 16.

Exact-Benders trajectory from the all-zero bootstrap (hand-verified):
  bootstrap: cost1 = 12 (slope -3), cost2 = 12 (slope -2)
             -> recourse floor = -1.1·12 = -13.2
  iter 1: master with cuts eta1 >= 12-3·f1, eta2 >= 12-2·f2
          -> f1=10, eta1=-18; f2=0, eta2=12; LB = 4
          subsolves at (10, 0): cost1 = 0 (slope 0), cost2 = 12
          UB = master native (10) + 0 + 12 = 22;  gap = 18/22
  iter 2: extra cuts eta1 >= 0, eta2 >= 12-2·f2 (duplicate)
          -> f1=4, eta1=0; f2=0, eta2=12; LB = 16
          subsolves at (4, 0): cost1 = 0, cost2 = 12
          UB = 4 + 0 + 12 = 16;  gap = 0  -> converged, 2 iterations.

Scripted-master scenarios (no LP at all) drive the bound self-checks:
the coordinator only reads ``Solution.obj`` and the adapter protocol, so a
master returning scripted objectives / points / recourse values exercises
LB-monotonicity pinning, ``lb_drop``, ``sandwich``, ``cut_violated`` and
``cut_nonfinite`` deterministically.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import polar_high as fp
from polar_high import (
    BendersBoundInvalid,
    BendersLoopOptions,
    SubproblemHandle,
    SubproblemNotOptimal,
    SubproblemResult,
    solve_benders_loop,
)

# ---------------------------------------------------------------------------
# The real toy LP: master + subproblem adapters.
# ---------------------------------------------------------------------------

_F_MAX = 10.0
_PROVISIONAL_FLOOR = -1.1e9


def _scalar_param(value: float) -> fp.Param:
    return fp.Param((), pl.DataFrame({"value": [value]}))


class _ToyMaster:
    """min f1 + 5·f2 + eta1 + eta2 over f in [0, 10], eta provisionally
    floored; implements the ``BendersMaster`` protocol on a real
    ``WarmProblem``."""

    def __init__(self, f_costs: tuple[float, float] = (1.0, 5.0)):
        p = fp.Problem()
        idx = pl.DataFrame({"i": [0]})
        f1 = p.add_var("f1", "i", idx, lower=0.0, upper=_F_MAX)
        f2 = p.add_var("f2", "i", idx, lower=0.0, upper=_F_MAX)
        eta1 = p.add_var("eta1", "i", idx, lower=_PROVISIONAL_FLOOR)
        eta2 = p.add_var("eta2", "i", idx, lower=_PROVISIONAL_FLOOR)
        # One harmless build-time row (never binding: f1+f2 <= 100 while
        # f <= 10 each) so the model has a constraint matrix.
        p.add_cstr(
            "cap",
            over=idx,
            sense="<=",
            lhs_terms={"f1": f1, "f2": f2},
            rhs_terms={"cap": _scalar_param(100.0)},
        )
        p.set_objective(
            f1.to_expr() * f_costs[0] + f2.to_expr() * f_costs[1] + eta1.to_expr() + eta2.to_expr(),
            sense="min",
        )
        self.wp = fp.WarmProblem(p)
        self.wp.solve()  # cold build (provisional floor keeps it bounded)
        self.f_col = {
            "sub1": int(self.wp.col_id_of_var("f1", (0,))),
            "sub2": int(self.wp.col_id_of_var("f2", (0,))),
        }
        self._eta_col = {
            "sub1": int(self.wp.col_id_of_var("eta1", (0,))),
            "sub2": int(self.wp.col_id_of_var("eta2", (0,))),
        }
        self._relaxed: set[str] = set()
        self.floor_set: float | None = None

    # -- BendersMaster protocol -----------------------------------------

    def solve(self) -> fp.Solution:
        sol = self.wp.solve(retry_on_unknown=True)
        if not sol.optimal:
            raise RuntimeError("toy master did not solve to optimality")
        return sol

    def read_point(self, sol: fp.Solution) -> tuple[dict[int, float], dict[str, float]]:
        f = {c: float(sol.col_value[c]) for c in self.f_col.values()}
        recourse = {name: float(sol.col_value[c]) for name, c in self._eta_col.items()}
        return f, recourse

    def native_cost(self, sol: fp.Solution, recourse: dict[str, float]) -> float:
        return float(sol.obj) - sum(recourse.values())

    def project_point(
        self, f: dict[int, float], sol: fp.Solution, *, hard_fail: bool = True
    ) -> float:
        max_slack = 0.0
        for c, v in f.items():
            slack = v - _F_MAX
            if slack > 0.0:
                f[c] = _F_MAX
                max_slack = max(max_slack, slack)
        return max_slack

    def add_cut(
        self,
        sub_name: str,
        gen_point: dict[int, float],
        cost: float,
        slopes: dict[int, float],
    ) -> None:
        # recourse >= cost + sum slopes·(f − gen)  <=>
        # recourse − sum slopes·f >= cost − sum slopes·gen
        col_ids = [self._eta_col[sub_name]]
        coefs = [1.0]
        rhs = cost
        for fcol, g in slopes.items():
            if g == 0.0:
                continue
            col_ids.append(int(fcol))
            coefs.append(-float(g))
            rhs -= g * gen_point[fcol]
        self.wp.add_cut_row(col_ids, coefs, float(rhs))

    def relax_recourse(self, sub_name: str) -> None:
        if sub_name in self._relaxed:
            return
        col = np.array([self._eta_col[sub_name]], dtype=np.int64)
        self.wp.set_col_bounds(col, np.array([-np.inf]), np.array([np.inf]))
        self._relaxed.add(sub_name)

    def set_recourse_floor(self, floor: float) -> None:
        cols = np.array(
            [c for n, c in self._eta_col.items() if n not in self._relaxed],
            dtype=np.int64,
        )
        if cols.size:
            self.wp.set_col_bounds(
                cols,
                np.full(cols.size, float(floor)),
                np.full(cols.size, np.inf),
            )
        self.floor_set = float(floor)


class _ToySub:
    """min gamma·g s.t. g + f_in >= d; ``solve_at`` pins f_in by raw col id
    and reads the cut slope off f_in's reduced cost."""

    def __init__(self, name: str, master_f_col: int, demand: float, gen_cost: float):
        p = fp.Problem()
        idx = pl.DataFrame({"i": [0]})
        f_in = p.add_var("f_in", "i", idx, lower=0.0, upper=1.0e3)
        g = p.add_var("g", "i", idx, lower=0.0)
        p.add_cstr(
            "balance",
            over=idx,
            sense=">=",
            lhs_terms={"g": g, "f_in": f_in},
            rhs_terms={"d": _scalar_param(demand)},
        )
        p.set_objective(g.to_expr() * gen_cost, sense="min")
        self.name = name
        self.warm = fp.WarmProblem(p)
        self.warm.solve()  # sequential cold build (coordinator precondition)
        self._local = int(self.warm.col_id_of_var("f_in", (0,)))
        self._master_col = int(master_f_col)
        self.seen_points: list[dict[int, float]] = []

    def solve_at(self, point: dict[int, float]) -> SubproblemResult:
        self.seen_points.append(dict(point))
        v = point[self._master_col]
        self.warm.fix_col_ids(np.array([self._local]), np.array([v], dtype=np.float64))
        sol = self.warm.solve(retry_on_unknown=True)
        if not sol.optimal:
            raise SubproblemNotOptimal(self.name)
        slope = float(sol.col_dual[self._local])
        return SubproblemResult(
            cost=float(sol.obj),
            slopes={self._master_col: slope},
            payload={"sub": self.name},
        )


def _build_toy(*, wrap_sub2_in_handle: bool = False):
    """Fresh (master, subproblems, initial_point) for one loop run."""
    master = _ToyMaster()
    sub1 = _ToySub("sub1", master.f_col["sub1"], demand=4.0, gen_cost=3.0)
    sub2 = _ToySub("sub2", master.f_col["sub2"], demand=6.0, gen_cost=2.0)
    subs: list = [sub1, sub2]
    if wrap_sub2_in_handle:
        subs[1] = SubproblemHandle(name="sub2", warm=sub2.warm, solve_at_fn=sub2.solve_at)
    initial = {master.f_col["sub1"]: 0.0, master.f_col["sub2"]: 0.0}
    return master, subs, initial


_TOY_OPTIMUM = 16.0


# ---------------------------------------------------------------------------
# Scripted (no-LP) master + subproblems for the bound self-check scenarios.
# ---------------------------------------------------------------------------


def _fake_solution(obj: float) -> fp.Solution:
    return fp.Solution(
        optimal=True,
        obj=float(obj),
        col_value=np.zeros(1, dtype=np.float64),
        row_dual=np.zeros(0, dtype=np.float64),
        col_names=[],
        row_names=[],
        vars={},
    )


class _ScriptedMaster:
    """Master returning a scripted objective / point / recourse sequence and
    a scripted (constant) native cost, decoupling the bound sequence from
    any real LP so each self-check fires deterministically."""

    def __init__(
        self,
        objs: list[float],
        recourse_seq: list[dict[str, float]],
        *,
        native: float = 0.0,
        points: list[dict[int, float]] | None = None,
    ):
        self._objs = list(objs)
        self._recourse = list(recourse_seq)
        self._native = float(native)
        self._points = points
        self._i = -1
        self.cuts: list[tuple[str, dict[int, float], float, dict[int, float]]] = []
        self.floor_set: float | None = None

    def solve(self) -> fp.Solution:
        self._i += 1
        return _fake_solution(self._objs[self._i])

    def read_point(self, sol: fp.Solution) -> tuple[dict[int, float], dict[str, float]]:
        point = dict(self._points[self._i]) if self._points is not None else {0: 0.0, 1: 0.0}
        return point, dict(self._recourse[self._i])

    def native_cost(self, sol: fp.Solution, recourse: dict[str, float]) -> float:
        return self._native

    def project_point(
        self, f: dict[int, float], sol: fp.Solution, *, hard_fail: bool = True
    ) -> float:
        return 0.0

    def add_cut(self, sub_name, gen_point, cost, slopes) -> None:
        self.cuts.append((sub_name, dict(gen_point), float(cost), dict(slopes)))

    def relax_recourse(self, sub_name: str) -> None:
        pass

    def set_recourse_floor(self, floor: float) -> None:
        self.floor_set = float(floor)


def _trivial_warm() -> fp.WarmProblem:
    p = fp.Problem()
    idx = pl.DataFrame({"i": [0]})
    x = p.add_var("x", "i", idx, lower=0.0, upper=1.0)
    p.add_cstr(
        "r",
        over=idx,
        sense=">=",
        lhs_terms={"x": x},
        rhs_terms={"z": _scalar_param(0.0)},
    )
    p.set_objective(x.to_expr(), sense="min")
    wp = fp.WarmProblem(p)
    wp.solve()
    return wp


class _ScriptedSub:
    """Subproblem returning a fixed cost + zero slope on one master column."""

    def __init__(self, name: str, master_col: int, cost: float):
        self.name = name
        self.warm = _trivial_warm()
        self._master_col = master_col
        self._cost = float(cost)

    def solve_at(self, point: dict[int, float]) -> SubproblemResult:
        return SubproblemResult(cost=self._cost, slopes={self._master_col: 0.0})


def _scripted_pair(costs: tuple[float, float] = (50.0, 50.0)) -> list[_ScriptedSub]:
    return [_ScriptedSub("a", 0, costs[0]), _ScriptedSub("b", 1, costs[1])]


_SCRIPT_INITIAL = {0: 0.0, 1: 0.0}


# ---------------------------------------------------------------------------
# Convergence on the known optimum (real toy LP).
# ---------------------------------------------------------------------------


def test_converges_to_known_optimum() -> None:
    """LB/UB/gap close on the hand-verified optimum in 2 iterations, the
    incumbent point is the analytic (f1, f2) = (4, 0), the recourse floor is
    the bootstrap-sized -1.1·12, and ``on_incumbent``'s return value comes
    back as the incumbent payload."""
    master, subs, initial = _build_toy()

    incumbent_calls: list[dict] = []

    def on_incumbent(msol, sub_results, info):
        assert isinstance(msol, fp.Solution)
        assert [type(r) for r in sub_results] == [SubproblemResult, SubproblemResult]
        assert sub_results[0].payload == {"sub": "sub1"}
        incumbent_calls.append(info)
        return {"at_iteration": info["iteration"]}

    result = solve_benders_loop(
        master,
        subs,
        options=BendersLoopOptions(max_iters=20, tol=1e-9, workers=1),
        initial_point=initial,
        on_incumbent=on_incumbent,
    )

    assert result.converged
    assert result.iterations == 2
    assert result.lower_bound == pytest.approx(_TOY_OPTIMUM, abs=1e-8)
    assert result.best_upper_bound == pytest.approx(_TOY_OPTIMUM, abs=1e-8)
    assert result.gap == pytest.approx(0.0, abs=1e-9)
    assert result.sub_costs["sub1"] == pytest.approx(0.0, abs=1e-8)
    assert result.sub_costs["sub2"] == pytest.approx(12.0, abs=1e-8)
    assert result.incumbent_point[master.f_col["sub1"]] == pytest.approx(4.0, abs=1e-7)
    assert result.incumbent_point[master.f_col["sub2"]] == pytest.approx(0.0, abs=1e-7)
    # eta-floor sizing: -eta_floor_mult · max_s|bootstrap cost| = -1.1·12.
    assert master.floor_set == pytest.approx(-13.2, abs=1e-9)
    # Both iterations improved the incumbent (UB 22 then 16); the stored
    # payload is the LAST (best) incumbent's.
    assert [c["iteration"] for c in incumbent_calls] == [1, 2]
    assert result.incumbent_payload == {"at_iteration": 2}


def test_trajectory_regression_pin() -> None:
    """The exact (LB, UB, best_UB, gap, cut_rows) per-iteration trajectory of
    the synthetic problem, human-verified against the analytic optimum (see
    the module docstring derivation) and recorded as literals — a regression
    pin for the coordinator's operation order and accumulation arithmetic."""
    master, subs, initial = _build_toy()
    trajectory: list[tuple] = []

    def on_iteration(info: dict) -> None:
        trajectory.append(
            (
                info["iter"],
                info["lower_bound"],
                info["upper_bound"],
                info["best_upper_bound"],
                info["gap"],
                info["converged"],
                info["cut_rows"],
            )
        )

    result = solve_benders_loop(
        master,
        subs,
        options=BendersLoopOptions(max_iters=20, tol=1e-9, workers=1),
        initial_point=initial,
        on_iteration=on_iteration,
    )

    expected = [
        # (iter, LB, UB, best_UB, gap, converged, cut_rows)
        (1, 4.0, 22.0, 22.0, 18.0 / 22.0, False, 2),
        (2, 16.0, 16.0, 16.0, 0.0, True, 4),
    ]
    assert len(trajectory) == len(expected)
    for got, exp in zip(trajectory, expected):
        assert got[0] == exp[0]
        assert got[1] == pytest.approx(exp[1], rel=1e-12, abs=1e-9)
        assert got[2] == pytest.approx(exp[2], rel=1e-12, abs=1e-9)
        assert got[3] == pytest.approx(exp[3], rel=1e-12, abs=1e-9)
        assert got[4] == pytest.approx(exp[4], rel=1e-12, abs=1e-9)
        assert got[5] is exp[5]
        assert got[6] == exp[6]
    assert result.iterations == 2 and result.converged


def test_subproblem_handle_adapter() -> None:
    """``SubproblemHandle`` (the plain-data adapter) is interchangeable with
    a hand-written adapter class."""
    master, subs, initial = _build_toy(wrap_sub2_in_handle=True)
    result = solve_benders_loop(
        master,
        subs,
        options=BendersLoopOptions(max_iters=20, tol=1e-9, workers=1),
        initial_point=initial,
    )
    assert result.converged
    assert result.best_upper_bound == pytest.approx(_TOY_OPTIMUM, abs=1e-8)


def test_nonzero_initial_point() -> None:
    """A non-zero ``initial_point`` is used VERBATIM as the bootstrap pin
    point and as the bootstrap cuts' generation point, and its key set is the
    coupling universe (the subproblems see exactly those keys).  The loop
    still converges to the same optimum."""
    master, subs, initial = _build_toy()
    initial = {master.f_col["sub1"]: 2.0, master.f_col["sub2"]: 1.0}

    bootstrap_costs: dict[str, float] = {}

    def on_subsolve(info: dict) -> None:
        if info["iter"] == 0:
            bootstrap_costs[info["sub"]] = info["cost"]

    result = solve_benders_loop(
        master,
        subs,
        options=BendersLoopOptions(max_iters=20, tol=1e-9, workers=1),
        initial_point=initial,
        on_subsolve=on_subsolve,
    )

    # Bootstrap solve_at received the supplied point exactly (key set AND
    # values) — for both subproblems.
    assert subs[0].seen_points[0] == initial
    assert subs[1].seen_points[0] == initial
    # Bootstrap costs at the supplied point: 3·(4−2)=6 and 2·(6−1)=10.
    assert bootstrap_costs["sub1"] == pytest.approx(6.0, abs=1e-8)
    assert bootstrap_costs["sub2"] == pytest.approx(10.0, abs=1e-8)
    # eta floor sized off the bootstrap costs at THIS point: -1.1·10.
    assert master.floor_set == pytest.approx(-11.0, abs=1e-9)
    # Same optimum as the zero bootstrap (the toy's cut lines coincide).
    assert result.converged
    assert result.best_upper_bound == pytest.approx(_TOY_OPTIMUM, abs=1e-8)
    assert result.iterations == 2


def test_determinism_workers_1_vs_2() -> None:
    """workers=1 and workers=2 produce IDENTICAL trajectories (exact float
    equality on every callback payload) — the parallel fan-out is
    deterministic and collection is index-ordered."""

    def run(workers: int) -> list[dict]:
        master, subs, initial = _build_toy()
        traj: list[dict] = []
        solve_benders_loop(
            master,
            subs,
            options=BendersLoopOptions(max_iters=20, tol=1e-9, workers=workers),
            initial_point=initial,
            on_iteration=traj.append,
        )
        return traj

    t1 = run(1)
    t2 = run(2)
    assert t1 == t2  # exact equality, floats included


def test_monolith_guard_fires_on_invalid_bound() -> None:
    """The optional test-time monolith guard raises when the LB exceeds a
    (deliberately understated) known optimum."""
    master, subs, initial = _build_toy()
    with pytest.raises(BendersBoundInvalid) as exc_info:
        solve_benders_loop(
            master,
            subs,
            options=BendersLoopOptions(max_iters=20, tol=1e-9, workers=1, monolith_objective=10.0),
            initial_point=initial,
        )
    exc = exc_info.value
    assert exc.kind == "monolith"
    assert exc.iteration == 2  # LB=16 first exceeds M=10 at iteration 2
    assert exc.lower_bound == pytest.approx(16.0, abs=1e-8)
    assert exc.monolith_objective == 10.0


# ---------------------------------------------------------------------------
# Bound self-checks (scripted master).
# ---------------------------------------------------------------------------


def test_lb_monotonicity_pins_small_dip() -> None:
    """A small (within-gross-band) LB dip is pinned back to the previous
    bound — the reported lower bound never decreases."""
    subs = _scripted_pair()
    # obj: 10, then a 1e-4-relative dip (pinned), then a genuine rise.
    master = _ScriptedMaster(
        objs=[10.0, 9.999, 20.0],
        recourse_seq=[{"a": 50.0, "b": 50.0}] * 3,
        native=0.0,
    )
    lbs: list[float] = []
    result = solve_benders_loop(
        master,
        subs,
        options=BendersLoopOptions(max_iters=3, tol=1e-9, workers=1),
        initial_point=dict(_SCRIPT_INITIAL),
        on_iteration=lambda info: lbs.append(info["lower_bound"]),
    )
    assert lbs == [10.0, 10.0, 20.0]  # dip pinned at iteration 2
    assert not result.converged  # UB stays at 100 (scripted native 0 + costs)
    assert result.lower_bound == 20.0
    assert result.best_upper_bound == 100.0


def test_bound_invalid_on_gross_lb_drop() -> None:
    """A gross LB drop (far beyond the band) raises
    ``BendersBoundInvalid(kind='lb_drop')`` with the numeric fields."""
    subs = _scripted_pair()
    master = _ScriptedMaster(
        objs=[10.0, 5.0],
        recourse_seq=[{"a": 50.0, "b": 50.0}] * 2,
        native=0.0,
    )
    with pytest.raises(BendersBoundInvalid) as exc_info:
        solve_benders_loop(
            master,
            subs,
            options=BendersLoopOptions(max_iters=5, tol=1e-9, workers=1),
            initial_point=dict(_SCRIPT_INITIAL),
        )
    exc = exc_info.value
    assert exc.kind == "lb_drop"
    assert exc.iteration == 2
    assert exc.prev_lower_bound == 10.0
    assert exc.lower_bound == 5.0
    assert exc.rel_drop == pytest.approx(0.5, rel=1e-12)
    assert exc.gross_band == pytest.approx(1e-3, rel=1e-12)


def test_bound_invalid_on_sandwich_violation() -> None:
    """LB rising grossly above the best known feasible cost raises
    ``BendersBoundInvalid(kind='sandwich')``."""
    subs = _scripted_pair()
    # iter 1: LB=10, UB = 0 + 100 = 100 (best).  iter 2: LB=200 > 100.
    master = _ScriptedMaster(
        objs=[10.0, 200.0],
        recourse_seq=[{"a": 50.0, "b": 50.0}] * 2,
        native=0.0,
    )
    with pytest.raises(BendersBoundInvalid) as exc_info:
        solve_benders_loop(
            master,
            subs,
            options=BendersLoopOptions(max_iters=5, tol=1e-9, workers=1),
            initial_point=dict(_SCRIPT_INITIAL),
        )
    exc = exc_info.value
    assert exc.kind == "sandwich"
    assert exc.iteration == 2
    assert exc.lower_bound == 200.0
    assert exc.best_upper_bound == 100.0
    assert exc.rel_over == pytest.approx(1.0, rel=1e-12)


def test_bound_invalid_on_violated_cut() -> None:
    """A recourse value grossly below a just-appended cut's RHS raises
    ``BendersBoundInvalid(kind='cut_violated')``."""
    subs = _scripted_pair(costs=(100.0, 100.0))
    # Bootstrap cuts carry cost 100 (zero slopes -> rhs = 100); the master
    # then reports recourse 0 for 'a' — a gross violation (100 > 1% of the
    # row scale 100).
    master = _ScriptedMaster(
        objs=[10.0],
        recourse_seq=[{"a": 0.0, "b": 100.0}],
        native=0.0,
    )
    with pytest.raises(BendersBoundInvalid) as exc_info:
        solve_benders_loop(
            master,
            subs,
            options=BendersLoopOptions(max_iters=5, tol=1e-9, workers=1),
            initial_point=dict(_SCRIPT_INITIAL),
        )
    exc = exc_info.value
    assert exc.kind == "cut_violated"
    assert exc.iteration == 1
    assert exc.sub_name == "a"
    assert exc.recourse_value == 0.0
    assert exc.cut_rhs == pytest.approx(100.0, rel=1e-12)
    assert exc.violation == pytest.approx(100.0, rel=1e-12)
    assert exc.row_scale == pytest.approx(100.0, rel=1e-12)


def test_bound_invalid_on_nonfinite_recourse() -> None:
    """A non-finite recourse value raises
    ``BendersBoundInvalid(kind='cut_nonfinite')``."""
    subs = _scripted_pair()
    master = _ScriptedMaster(
        objs=[10.0],
        recourse_seq=[{"a": float("nan"), "b": 50.0}],
        native=0.0,
    )
    with pytest.raises(BendersBoundInvalid) as exc_info:
        solve_benders_loop(
            master,
            subs,
            options=BendersLoopOptions(max_iters=5, tol=1e-9, workers=1),
            initial_point=dict(_SCRIPT_INITIAL),
        )
    exc = exc_info.value
    assert exc.kind == "cut_nonfinite"
    assert exc.iteration == 1
    assert exc.sub_name == "a"
    assert np.isnan(exc.recourse_value)


# ---------------------------------------------------------------------------
# Exception propagation + input validation.
# ---------------------------------------------------------------------------


class _FailingSub:
    def __init__(self, name: str):
        self.name = name
        self.warm = _trivial_warm()

    def solve_at(self, point: dict[int, float]) -> SubproblemResult:
        raise SubproblemNotOptimal(self.name, status="kInfeasible")


@pytest.mark.parametrize("workers", [1, 2])
def test_subproblem_not_optimal_propagates(workers: int) -> None:
    """A domain-side ``SubproblemNotOptimal`` raised inside ``solve_at``
    propagates through the coordinator (and the parallel fan-out) untouched,
    carrying its structured fields."""
    subs = [_ScriptedSub("a", 0, 50.0), _FailingSub("b")]
    master = _ScriptedMaster(objs=[10.0], recourse_seq=[{"a": 50.0, "b": 50.0}])
    with pytest.raises(SubproblemNotOptimal) as exc_info:
        solve_benders_loop(
            master,
            subs,
            options=BendersLoopOptions(max_iters=5, tol=1e-9, workers=workers),
            initial_point=dict(_SCRIPT_INITIAL),
        )
    assert exc_info.value.sub_name == "b"
    assert exc_info.value.status == "kInfeasible"


def test_unwired_options_rejected_loudly() -> None:
    """Options belonging to not-yet-wired features (in-out, compaction) are
    rejected with ``NotImplementedError`` at non-default values, never
    silently ignored."""
    subs = _scripted_pair()
    master = _ScriptedMaster(objs=[10.0], recourse_seq=[{"a": 50.0, "b": 50.0}])
    with pytest.raises(NotImplementedError, match="in_out_weight"):
        solve_benders_loop(
            master,
            subs,
            options=BendersLoopOptions(max_iters=5, tol=1e-9, in_out_weight=0.5),
            initial_point=dict(_SCRIPT_INITIAL),
        )
    with pytest.raises(NotImplementedError, match="compact_at"):
        solve_benders_loop(
            master,
            subs,
            options=BendersLoopOptions(max_iters=5, tol=1e-9, compact_at=100),
            initial_point=dict(_SCRIPT_INITIAL),
        )


def test_input_validation() -> None:
    """No subproblems, duplicate names, and an empty initial point are hard
    errors."""
    master = _ScriptedMaster(objs=[10.0], recourse_seq=[{"a": 50.0}])
    opts = BendersLoopOptions(max_iters=5, tol=1e-9)
    with pytest.raises(ValueError, match="no subproblems"):
        solve_benders_loop(master, [], options=opts, initial_point={0: 0.0})
    dup = [_ScriptedSub("a", 0, 1.0), _ScriptedSub("a", 1, 1.0)]
    with pytest.raises(ValueError, match="duplicate subproblem names"):
        solve_benders_loop(master, dup, options=opts, initial_point={0: 0.0})
    with pytest.raises(ValueError, match="initial_point is empty"):
        solve_benders_loop(master, [_ScriptedSub("a", 0, 1.0)], options=opts, initial_point={})
