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
    BendersStalled,
    PointEvaluation,
    SubproblemHandle,
    SubproblemNotOptimal,
    SubproblemResult,
    evaluate_at_point,
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

    def native_cost_at(self, point: dict[int, float]) -> float:
        # Pin f1/f2 at ``point`` and read obj − Σ eta (the eta cols float to
        # their floor and cancel out), so this returns f_costs·(f1, f2) at the
        # pinned coupling values — the same quantity ``native_cost`` reads at
        # the master's own vertex, but at an ARBITRARY point.
        cols = np.array(list(self.f_col.values()), dtype=np.int64)
        lo, hi = self.wp.get_col_bounds(cols)
        vals = np.array([float(point[int(c)]) for c in cols], dtype=np.float64)
        self.wp.fix_col_ids(cols, vals)
        try:
            sol = self.wp.solve(retry_on_unknown=True)
            if not sol.optimal:
                raise RuntimeError("toy master native_cost_at solve not optimal")
            eta = sum(float(sol.col_value[c]) for c in self._eta_col.values())
            return float(sol.obj) - eta
        finally:
            self.wp.set_col_bounds(cols, lo, hi)

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
    any real LP so each self-check fires deterministically.

    ``cap`` (optional) turns ``project_point`` into a real clamp-to-``cap``
    projection: with ``hard_fail=True`` an overshoot beyond
    ``hard_fail_slack`` raises (the adapter's "gross violation" error);
    with ``hard_fail=False`` ANY overshoot is silently clamped.  Every
    ``project_point`` call is recorded as ``(pre-projection point copy,
    hard_fail)`` in ``project_calls``.  Sequence lists repeat their last
    entry once exhausted.
    """

    def __init__(
        self,
        objs: list[float],
        recourse_seq: list[dict[str, float]],
        *,
        native: float = 0.0,
        points: list[dict[int, float]] | None = None,
        cap: float | None = None,
        hard_fail_slack: float = 1.0,
    ):
        self._objs = list(objs)
        self._recourse = list(recourse_seq)
        self._native = float(native)
        self._points = points
        self._cap = cap
        self._hard_fail_slack = float(hard_fail_slack)
        self._i = -1
        self.cuts: list[tuple[str, dict[int, float], float, dict[int, float]]] = []
        self.floor_set: float | None = None
        self.project_calls: list[tuple[dict[int, float], bool]] = []

    @staticmethod
    def _at(seq: list, i: int):
        return seq[min(i, len(seq) - 1)]

    def solve(self) -> fp.Solution:
        self._i += 1
        return _fake_solution(self._at(self._objs, self._i))

    def read_point(self, sol: fp.Solution) -> tuple[dict[int, float], dict[str, float]]:
        point = (
            dict(self._at(self._points, self._i)) if self._points is not None else {0: 0.0, 1: 0.0}
        )
        return point, dict(self._at(self._recourse, self._i))

    def native_cost(self, sol: fp.Solution, recourse: dict[str, float]) -> float:
        return self._native

    def project_point(
        self, f: dict[int, float], sol: fp.Solution, *, hard_fail: bool = True
    ) -> float:
        self.project_calls.append((dict(f), hard_fail))
        if self._cap is None:
            return 0.0
        max_slack = 0.0
        for c, v in f.items():
            slack = v - self._cap
            if slack > 0.0:
                if hard_fail and slack > self._hard_fail_slack:
                    raise RuntimeError(
                        f"scripted master: gross capacity violation ({v} > {self._cap})"
                    )
                f[c] = self._cap
                max_slack = max(max_slack, slack)
        return max_slack

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
    """Subproblem returning a scripted cost + zero slopes on its master
    column(s).

    ``master_col`` may be a single column id or a list (the slopes key set =
    the subproblem's column-ownership map).  ``cost`` may be a scalar or a
    per-call sequence (bootstrap first; the last entry repeats).  Every
    ``solve_at`` pin point is recorded in ``seen_points``.
    """

    def __init__(self, name: str, master_col, cost):
        self.name = name
        self.warm = _trivial_warm()
        cols = master_col if isinstance(master_col, (list, tuple)) else [master_col]
        self._cols = [int(c) for c in cols]
        self._costs = [float(c) for c in cost] if isinstance(cost, (list, tuple)) else None
        self._cost = None if self._costs is not None else float(cost)
        self._i = -1
        self.seen_points: list[dict[int, float]] = []

    def solve_at(self, point: dict[int, float]) -> SubproblemResult:
        self.seen_points.append(dict(point))
        self._i += 1
        cost = (
            self._costs[min(self._i, len(self._costs) - 1)]
            if self._costs is not None
            else self._cost
        )
        return SubproblemResult(cost=cost, slopes={c: 0.0 for c in self._cols})


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


def test_invalid_in_out_weight_rejected() -> None:
    """``in_out_weight`` outside ``[0, 1)`` is rejected loudly (>= 1 never
    queries the master ⇒ non-convergent; < 0 is meaningless)."""
    subs = _scripted_pair()
    master = _ScriptedMaster(objs=[10.0], recourse_seq=[{"a": 50.0, "b": 50.0}])
    for bad in (1.0, 1.5, -0.1):
        with pytest.raises(ValueError, match="in_out_weight"):
            solve_benders_loop(
                master,
                subs,
                options=BendersLoopOptions(max_iters=5, tol=1e-9, in_out_weight=bad),
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


# ---------------------------------------------------------------------------
# In-out stabilization (λ > 0).
#
# Hand-derived λ=0.5 trajectory on the toy LP (stabilizer centre seeded at
# the all-zero initial point; every column below in (f1, f2)):
#   iter 1: master vertex f_out = (10, 0); per-sub f_sep = 0.5·0 + 0.5·f_out
#           = (5, 0); sub1 at f1=5 -> cost 0, sub2 at f2=0 -> cost 12;
#           UB = 10 + 0 + 12 = 22 (improved); incumbent overlay writes each
#           sub's f_sep onto its own column -> incumbent (5, 0); register:
#           improved -> serious step, both centres jump to (5, 0).
#   iter 2: master vertex f_out = (4, 0); f_sep = 0.5·5 + 0.5·4 = (4.5, 0);
#           sub1 at 4.5 -> 0, sub2 at 0 -> 12; UB = 4 + 12 = 16 = LB -> gap 0,
#           converged; incumbent overlay -> (4.5, 0).
# ---------------------------------------------------------------------------


def test_in_out_converges_on_toy_lp() -> None:
    """λ=0.5 still converges to the known optimum on the toy LP, the
    subproblems are actually pinned at the INTERIOR separation points, and
    the incumbent point carries each subproblem's own ``f_sep`` value."""
    master, subs, initial = _build_toy()
    result = solve_benders_loop(
        master,
        subs,
        options=BendersLoopOptions(max_iters=20, tol=1e-9, workers=1, in_out_weight=0.5),
        initial_point=initial,
    )
    assert result.converged
    assert result.iterations == 2
    assert result.lower_bound == pytest.approx(_TOY_OPTIMUM, abs=1e-8)
    assert result.best_upper_bound == pytest.approx(_TOY_OPTIMUM, abs=1e-8)
    # Iteration-1 pin was the interior (5, 0), not the master vertex (10, 0);
    # iteration-2 pin the interior (4.5, 0), not (4, 0).
    f1, f2 = master.f_col["sub1"], master.f_col["sub2"]
    assert subs[0].seen_points[1][f1] == pytest.approx(5.0, abs=1e-7)
    assert subs[0].seen_points[1][f2] == pytest.approx(0.0, abs=1e-7)
    assert subs[0].seen_points[2][f1] == pytest.approx(4.5, abs=1e-7)
    # The incumbent point reflects the value each subproblem actually solved
    # at (its own f_sep), overlaid on its owned column.
    assert result.incumbent_point[f1] == pytest.approx(4.5, abs=1e-7)
    assert result.incumbent_point[f2] == pytest.approx(0.0, abs=1e-7)


def test_in_out_forced_out_step() -> None:
    """A no-separation iteration WITHOUT incumbent improvement arms the
    forced out-step: the next separation point is the master vertex
    VERBATIM (exact Benders for that pass), per the Ben-Ameur–Neto
    convergence rule."""
    subs = _scripted_pair()  # constant cost 50, zero slopes
    master = _ScriptedMaster(
        objs=[10.0, 20.0, 30.0],
        recourse_seq=[{"a": 50.0, "b": 50.0}],
        native=0.0,
        points=[{0: 8.0, 1: 8.0}],
    )
    result = solve_benders_loop(
        master,
        subs,
        options=BendersLoopOptions(max_iters=3, tol=1e-9, workers=1, in_out_weight=0.5),
        initial_point=dict(_SCRIPT_INITIAL),
    )
    assert not result.converged
    # Pin trajectory for sub 'a' (identical for 'b'):
    #   bootstrap at the initial (0, 0);
    #   iter 1: centre (0,0) -> f_sep = (4, 4); UB improves (inf -> 100) so
    #           the register is a SERIOUS step: centre <- incumbent (4, 4);
    #   iter 2: f_sep = 0.5·4 + 0.5·8 = (6, 6); UB stays 100 (no improvement)
    #           and the zero-slope cut (value 50) does NOT separate the
    #           recourse (50) -> out-step armed, weight shrunk;
    #   iter 3: forced out-step -> pinned at the master vertex (8, 8) VERBATIM.
    assert subs[0].seen_points == [
        {0: 0.0, 1: 0.0},
        {0: 4.0, 1: 4.0},
        {0: 6.0, 1: 6.0},
        {0: 8.0, 1: 8.0},
    ]
    assert subs[1].seen_points == subs[0].seen_points


def test_in_out_interior_reclamp_is_soft() -> None:
    """An interior separation point that exceeds the master's CURRENT
    feasible set is re-projected through ``project_point(hard_fail=False)``
    — clamped silently, even beyond the hard-fail slack — while the main
    post-master projection still runs with ``hard_fail=True``."""
    subs = _scripted_pair()
    # Centre = initial point at 20 on col 0; master vertex at the cap 10.
    # f_sep = 0.5·20 + 0.5·10 = 15 > cap, overshoot 5 > hard_fail_slack 1 —
    # a violation the MAIN projection would raise on.
    master = _ScriptedMaster(
        objs=[10.0],
        recourse_seq=[{"a": 50.0, "b": 50.0}],
        native=0.0,
        points=[{0: 10.0, 1: 0.0}],
        cap=10.0,
        hard_fail_slack=1.0,
    )
    solve_benders_loop(
        master,
        subs,
        options=BendersLoopOptions(max_iters=1, tol=1e-9, workers=1, in_out_weight=0.5),
        initial_point={0: 20.0, 1: 0.0},
    )
    # Call order: the main projection (hard_fail=True, in-cap point), then
    # one soft re-clamp per subproblem's interior point.
    assert [hard for _, hard in master.project_calls] == [True, False, False]
    for pre, hard in master.project_calls[1:]:
        assert not hard
        assert pre[0] == pytest.approx(15.0, abs=1e-9)  # pre-projection value
    # The subproblems were pinned at the CLAMPED interior point.
    assert subs[0].seen_points[1][0] == pytest.approx(10.0, abs=1e-9)
    assert subs[1].seen_points[1][0] == pytest.approx(10.0, abs=1e-9)


def test_main_projection_still_hard_fails_gross() -> None:
    """The main post-master projection keeps ``hard_fail=True``: a gross
    master-point violation raises the adapter's own error (propagated
    untouched), independent of the in-out soft path."""
    subs = _scripted_pair()
    master = _ScriptedMaster(
        objs=[10.0],
        recourse_seq=[{"a": 50.0, "b": 50.0}],
        native=0.0,
        points=[{0: 50.0, 1: 0.0}],  # 40 beyond the cap — gross
        cap=10.0,
        hard_fail_slack=1.0,
    )
    with pytest.raises(RuntimeError, match="gross capacity violation"):
        solve_benders_loop(
            master,
            subs,
            options=BendersLoopOptions(max_iters=1, tol=1e-9, workers=1, in_out_weight=0.5),
            initial_point=dict(_SCRIPT_INITIAL),
        )


def test_in_out_incumbent_overlay_ownership() -> None:
    """The λ>0 incumbent overlay writes each subproblem's ``f_sep`` onto
    exactly its slopes-key columns — including a SHARED column owned by two
    subproblems (the cross-arc shape), which takes the last owner's value —
    after the two stabilizers have genuinely DIVERGED (one forced out-step,
    one interior)."""
    # Ownership: sub 'a' owns {0, 1}, sub 'b' owns {1, 2}; column 1 shared.
    subs = [
        _ScriptedSub("a", [0, 1], [50.0, 50.0, 50.0, 40.0]),
        _ScriptedSub("b", [1, 2], [50.0, 50.0, 50.0, 40.0]),
    ]
    # iter 2: 'a' fails to separate (recourse == cut value 50) -> out-step;
    # 'b' separates (recourse 49.999 < 50, beyond the separation tolerance,
    # within the cut-satisfaction warn band) -> null step, stays interior.
    master = _ScriptedMaster(
        objs=[10.0, 20.0, 30.0],
        recourse_seq=[
            {"a": 50.0, "b": 50.0},
            {"a": 50.0, "b": 49.999},
            {"a": 50.0, "b": 50.0},
        ],
        native=0.0,
        points=[{0: 8.0, 1: 8.0, 2: 8.0}],
    )
    result = solve_benders_loop(
        master,
        subs,
        options=BendersLoopOptions(max_iters=3, tol=1e-9, workers=1, in_out_weight=0.5),
        initial_point={0: 0.0, 1: 0.0, 2: 0.0},
    )
    # iter 3 pins diverge: 'a' at the master vertex verbatim (forced out),
    # 'b' at its interior 0.5·4 + 0.5·8 = 6 (centre from the iter-1 serious
    # step's incumbent (4, 4, 4), held through the iter-2 null step).
    assert subs[0].seen_points[3] == {0: 8.0, 1: 8.0, 2: 8.0}
    assert subs[1].seen_points[3] == {0: 6.0, 1: 6.0, 2: 6.0}
    # iter 3 improves (UB 80 < 100) -> incumbent overlay: base (8, 8, 8);
    # 'a' writes 8 onto {0, 1}; 'b' then writes 6 onto {1, 2} — the shared
    # column 1 takes the LAST owner's value.
    assert result.incumbent_point == {0: 8.0, 1: 6.0, 2: 6.0}
    assert result.best_upper_bound == pytest.approx(80.0, abs=1e-9)


def test_determinism_workers_1_vs_2_in_out() -> None:
    """workers=1 and workers=2 produce IDENTICAL trajectories at λ>0 (the
    per-subproblem separation points are computed before the fan-out and
    collection is index-ordered)."""

    def run(workers: int) -> list[dict]:
        master, subs, initial = _build_toy()
        traj: list[dict] = []
        solve_benders_loop(
            master,
            subs,
            options=BendersLoopOptions(max_iters=20, tol=1e-9, workers=workers, in_out_weight=0.5),
            initial_point=initial,
            on_iteration=traj.append,
        )
        return traj

    t1 = run(1)
    t2 = run(2)
    assert t1 == t2  # exact equality, floats included


# ---------------------------------------------------------------------------
# Stall guard.
# ---------------------------------------------------------------------------


def _stall_setup(*, extra=None, max_iters: int = 6, on_iteration=None):
    """Scripted blow-up: bootstrap costs 10 each (reference scale 20), then
    the subproblem costs jump to 5000 each (UB frozen at 10000, gap ~1)."""
    subs = [
        _ScriptedSub("a", 0, [10.0, 5000.0]),
        _ScriptedSub("b", 1, [10.0, 5000.0]),
    ]
    master = _ScriptedMaster(
        objs=[float(i) for i in range(1, max_iters + 1)],
        recourse_seq=[{"a": 10.0, "b": 10.0}] + [{"a": 5000.0, "b": 5000.0}] * (max_iters - 1),
        native=0.0,
    )
    return solve_benders_loop(
        master,
        subs,
        options=BendersLoopOptions(max_iters=max_iters, tol=1e-9, workers=1, stall_window=2),
        initial_point=dict(_SCRIPT_INITIAL),
        extra_reference_cost=extra,
        on_iteration=on_iteration,
    )


def test_stall_guard_raises_with_fields() -> None:
    """A frozen, blown-up incumbent raises ``BendersStalled`` at the first
    full window, carrying the numeric fields incl. per-subproblem current
    AND bootstrap (reference) costs."""
    with pytest.raises(BendersStalled) as exc_info:
        _stall_setup()
    exc = exc_info.value
    assert exc.iteration == 2  # stall_window=2: earliest possible verdict
    assert exc.tol == 1e-9
    assert exc.window == 2
    # reference_scale = sum of |bootstrap costs| (no extra term).
    assert exc.reference_scale == pytest.approx(20.0, rel=1e-12)
    assert exc.sub_costs == {"a": 5000.0, "b": 5000.0}
    assert exc.sub_reference_costs == {"a": 10.0, "b": 10.0}
    # gap at the stalled iteration: (10000 - 2) / 10000.
    assert exc.gap == pytest.approx(0.9998, rel=1e-12)


def test_extra_reference_cost_in_reference_scale() -> None:
    """``extra_reference_cost`` is called exactly ONCE post-bootstrap and its
    ABSOLUTE value lands in the stall reference scale (asserted via the
    ``BendersStalled`` field)."""
    calls: list[float] = []

    def extra() -> float:
        calls.append(-30.0)
        return -30.0  # negative: the coordinator takes |·|

    with pytest.raises(BendersStalled) as exc_info:
        _stall_setup(extra=extra)
    assert len(calls) == 1
    assert exc_info.value.reference_scale == pytest.approx(50.0, rel=1e-12)  # 20 + |−30|


def test_extra_reference_cost_suppresses_false_stall() -> None:
    """A large extra reference cost feeds the blow-up gate: the same frozen
    trajectory is NOT declared a stall when the incumbent is within the
    (extended) sane magnitude — the loop runs to the iteration cap."""
    calls: list[float] = []

    def extra() -> float:
        calls.append(1.0e6)
        return 1.0e6

    result = _stall_setup(extra=extra, max_iters=6)
    assert len(calls) == 1  # once, not per-iteration
    assert not result.converged
    assert result.iterations == 6  # ran out the cap, no BendersStalled


# ---------------------------------------------------------------------------
# Cut compaction.
# ---------------------------------------------------------------------------


class _CompactingMaster(_ScriptedMaster):
    """Scripted master with a recording ``compact_cuts``."""

    def __init__(self, *args, kept: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self._kept = int(kept)
        self.compact_calls: list[dict] = []

    def compact_cuts(self, sol: fp.Solution, *, policy: str, trial_col_values: list) -> dict:
        self.compact_calls.append(
            {
                "obj": float(sol.obj),
                "policy": policy,
                "n_trial": len(trial_col_values),
            }
        )
        return {"kept": self._kept, "dropped": 3, "restored": False}


def test_compaction_passthrough() -> None:
    """When the accumulated cut-row count reaches ``compact_at`` the master's
    ``compact_cuts`` is invoked at the END of the iteration body with the
    RAW master vertex, the configured policy and the bounded trial-point
    window; the row count resets to the returned ``kept``."""
    subs = _scripted_pair()
    master = _CompactingMaster(
        objs=[10.0, 20.0, 30.0, 40.0],
        recourse_seq=[{"a": 50.0, "b": 50.0}],
        native=0.0,
        kept=1,
    )
    cut_rows_seen: list[int] = []
    solve_benders_loop(
        master,
        subs,
        options=BendersLoopOptions(
            max_iters=4,
            tol=1e-9,
            workers=1,
            compact_at=3,
            cut_policy="dominance",
            cut_window=2,
        ),
        initial_point=dict(_SCRIPT_INITIAL),
        on_iteration=lambda info: cut_rows_seen.append(info["cut_rows"]),
    )
    # 2 cuts/iteration: iter1 -> 2 (< 3, no compaction); iter2 -> 4 (>= 3,
    # compact, reset to kept=1); iter3 -> 3 (compact); iter4 -> 3 (compact).
    assert cut_rows_seen == [2, 4, 3, 3]
    assert [c["obj"] for c in master.compact_calls] == [20.0, 30.0, 40.0]
    assert all(c["policy"] == "dominance" for c in master.compact_calls)
    # Trial window bounded at cut_window=2 (two vertices seen by iter 2).
    assert [c["n_trial"] for c in master.compact_calls] == [2, 2, 2]


def test_compaction_clean_skip_without_member(caplog) -> None:
    """``compact_at > 0`` against a master WITHOUT ``compact_cuts`` warns
    once and proceeds exactly like the OFF path (identical trajectory)."""

    def run(compact_at: int) -> list[dict]:
        subs = _scripted_pair()
        master = _ScriptedMaster(  # has no compact_cuts member
            objs=[10.0, 20.0, 30.0],
            recourse_seq=[{"a": 50.0, "b": 50.0}],
            native=0.0,
        )
        traj: list[dict] = []
        solve_benders_loop(
            master,
            subs,
            options=BendersLoopOptions(max_iters=3, tol=1e-9, workers=1, compact_at=compact_at),
            initial_point=dict(_SCRIPT_INITIAL),
            on_iteration=traj.append,
        )
        return traj

    import logging as _logging

    with caplog.at_level(_logging.WARNING, logger="polar_high.benders"):
        t_requested = run(compact_at=3)
    assert any("no compact_cuts member" in r.message for r in caplog.records)
    t_off = run(compact_at=0)
    assert t_requested == t_off


# ---------------------------------------------------------------------------
# λ=0 default-path pin: the C3a trajectory literals are re-asserted by
# ``test_trajectory_regression_pin`` above, which runs against THIS wired
# build — proving the in-out/stall/compaction blocks leave the default
# (λ=0, compact_at=0) trajectory untouched.  This test re-runs it with the
# now-active stall guard at its DEFAULTS to pin that the guard never fires
# on a healthy converging run.
# ---------------------------------------------------------------------------


def test_lambda0_trajectory_unchanged_with_wired_features() -> None:
    """The λ=0 trajectory literals from the core-loop commit hold verbatim
    with the C3b features wired (all at defaults)."""
    master, subs, initial = _build_toy()
    trajectory: list[tuple] = []
    result = solve_benders_loop(
        master,
        subs,
        options=BendersLoopOptions(max_iters=20, tol=1e-9, workers=1),
        initial_point=initial,
        on_iteration=lambda info: trajectory.append(
            (info["iter"], info["lower_bound"], info["best_upper_bound"], info["cut_rows"])
        ),
    )
    assert result.converged and result.iterations == 2
    assert trajectory[0][0] == 1
    assert trajectory[0][1] == pytest.approx(4.0, rel=1e-12)
    assert trajectory[0][2] == pytest.approx(22.0, rel=1e-12)
    assert trajectory[0][3] == 2
    assert trajectory[1][1] == pytest.approx(16.0, rel=1e-12)
    assert trajectory[1][2] == pytest.approx(16.0, rel=1e-12)
    assert trajectory[1][3] == 4


# ---------------------------------------------------------------------------
# evaluate_at_point — the L-shaped feasible-point primitive.
#
# The toy monolith optimum is (f1=4, f2=0) with total 16 (module docstring).
# At that point:
#   sub1: demand 4, gen 3, f_in pinned 4 -> g=0  -> cost 0
#   sub2: demand 6, gen 2, f_in pinned 0 -> g=6  -> cost 12
#   master native at (4, 0) = 1·4 + 5·0 = 4
#   total = 0 + 12 + 4 = 16   (= the hand-verified monolith optimum)
# Stand-alone reference (zero coupling): sub1 = 3·4 = 12, sub2 = 2·6 = 12.
# ---------------------------------------------------------------------------


class _NoNativeMaster:
    """A master WITHOUT ``native_cost_at`` — to prove ``evaluate_at_point``
    degrades to subproblem-only (``master_native_cost = None``)."""

    def solve(self) -> fp.Solution:  # pragma: no cover — never called here
        raise AssertionError("evaluate_at_point must not call master.solve")


def test_evaluate_at_point_hand_verified_optimum() -> None:
    master, subs, initial = _build_toy()
    mono_point = {master.f_col["sub1"]: 4.0, master.f_col["sub2"]: 0.0}
    ref = evaluate_at_point(master, subs, initial, workers=1)  # stand-alone
    assert ref.sub_costs["sub1"] == pytest.approx(12.0, rel=1e-12)
    assert ref.sub_costs["sub2"] == pytest.approx(12.0, rel=1e-12)
    assert ref.master_native_cost == pytest.approx(0.0, abs=1e-9)

    res = evaluate_at_point(
        master,
        subs,
        mono_point,
        reference_costs=ref.sub_costs,
        blowup_mult=100.0,
        workers=1,
    )
    assert isinstance(res, PointEvaluation)
    assert res.sub_costs["sub1"] == pytest.approx(0.0, abs=1e-9)
    assert res.sub_costs["sub2"] == pytest.approx(12.0, rel=1e-12)
    assert res.sub_cost_total == pytest.approx(12.0, rel=1e-12)
    assert res.master_native_cost == pytest.approx(4.0, rel=1e-12)
    assert res.total_cost == pytest.approx(16.0, rel=1e-12)
    # No subproblem blew up versus its stand-alone reference.
    assert res.blew_up == {"sub1": False, "sub2": False}


def test_evaluate_at_point_master_state_untouched() -> None:
    # evaluate_at_point pins/solves inside native_cost_at but must restore the
    # master's coupling-column bounds, so a subsequent loop run is unaffected.
    master, subs, initial = _build_toy()
    mono_point = {master.f_col["sub1"]: 4.0, master.f_col["sub2"]: 0.0}
    lo0, hi0 = master.wp.get_col_bounds(np.array(list(master.f_col.values()), dtype=np.int64))
    evaluate_at_point(master, subs, mono_point, workers=1)
    lo1, hi1 = master.wp.get_col_bounds(np.array(list(master.f_col.values()), dtype=np.int64))
    assert np.array_equal(lo0, lo1) and np.array_equal(hi0, hi1)
    # And the loop still converges to 16 on the restored master.
    result = solve_benders_loop(
        master,
        subs,
        options=BendersLoopOptions(max_iters=20, tol=1e-9, workers=1),
        initial_point=initial,
    )
    assert result.converged
    assert result.best_upper_bound == pytest.approx(16.0, rel=1e-9)


def test_evaluate_at_point_blowup_flag() -> None:
    # A tiny reference makes the stand-alone costs (12, 12) read as blown up.
    master, subs, initial = _build_toy()
    res = evaluate_at_point(
        master,
        subs,
        initial,
        reference_costs={"sub1": 0.1, "sub2": 0.1},
        blowup_mult=10.0,
        workers=1,
    )
    # threshold = 10 · max(1, 0.1) = 10 ; both stand-alone costs (12) exceed it.
    assert res.blew_up == {"sub1": True, "sub2": True}
    assert res.blowup_mult == 10.0


def test_evaluate_at_point_no_native_cost_at() -> None:
    _, subs, initial = _build_toy()
    res = evaluate_at_point(_NoNativeMaster(), subs, initial, workers=1)
    assert res.master_native_cost is None
    # total falls back to the subproblem sum only.
    assert res.total_cost == pytest.approx(res.sub_cost_total, rel=1e-12)
    assert res.sub_cost_total == pytest.approx(24.0, rel=1e-12)


def test_evaluate_at_point_input_guards() -> None:
    master, subs, initial = _build_toy()
    with pytest.raises(ValueError, match="no subproblems"):
        evaluate_at_point(master, [], initial, workers=1)
    with pytest.raises(ValueError, match="point is empty"):
        evaluate_at_point(master, subs, {}, workers=1)
    dup = [subs[0], SubproblemHandle(subs[0].name, subs[0].warm, subs[0].solve_at)]
    with pytest.raises(ValueError, match="duplicate subproblem names"):
        evaluate_at_point(master, dup, initial, workers=1)
