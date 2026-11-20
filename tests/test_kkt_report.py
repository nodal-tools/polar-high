"""``kkt_report`` / ``tolerance_violation`` against HiGHS itself.

The oracle is HiGHS: ``getInfo()`` on unscaled solves, and HiGHS' own log
(captured by the logging callback) on user-scaled ones, where ``getInfo()``
holds the unscaled numbers.  No stored numbers: two seeded LP generators.
"""

from __future__ import annotations

import dataclasses
import math
import re

import highspy
import numpy as np
import pytest

import polar_high
from polar_high import _kkt, kkt_report, kkt_tolerances, tolerance_violation

_IS_115 = highspy.Highs().version().startswith("1.15.")
requires_highs_115 = pytest.mark.skipif(not _IS_115, reason="HiGHS 1.15 definitions")

INF = highspy.kHighsInf
# The 16 HighsInfo KKT fields (the first fields of KktReport).
_FIELDS = [f.name for f in dataclasses.fields(polar_high.KktReport)][:16]


def _small(seed: int, n: int = 400, m: int = 300) -> highspy.Highs:
    """400 columns x 300 ``>=`` rows, positive costs, mixed upper bounds."""
    h = highspy.Highs()
    h.setOptionValue("log_to_console", False)
    rng = np.random.default_rng(seed)
    for _ in range(n):
        h.addVar(0, rng.choice([10.0, 1e4, INF]))
    h.changeColsCost(
        n, np.arange(n, dtype=np.int32), rng.uniform(0.1, 5, n) * rng.choice([1, 1e3], n)
    )
    for _ in range(m):
        idx = rng.choice(n, 10, replace=False).astype(np.int32)
        h.addRow(rng.uniform(1, 100), INF, 10, idx, rng.uniform(0.01, 30, 10))
    return h


def _wide(seed: int, n: int = 3000, m: int = 2000) -> highspy.Highs:
    """Bounds, costs and matrix over 6-7 decades."""
    h = highspy.Highs()
    h.setOptionValue("log_to_console", False)
    rng = np.random.default_rng(seed)
    ub = np.where(rng.random(n) < 0.8, INF, 10 ** rng.uniform(0, 6, n))
    h.addVars(n, np.zeros(n), ub)
    h.changeColsCost(n, np.arange(n, dtype=np.int32), 10 ** rng.uniform(-3, 4, n))
    for _ in range(m):
        idx = rng.choice(n, 12, replace=False).astype(np.int32)
        h.addRow(10 ** rng.uniform(-1, 4), INF, 12, idx, 10 ** rng.uniform(-4, 3, 12))
    return h


def _run(h: highspy.Highs, **options) -> str:
    """Solve *h* with *options*; HiGHS' log of the run."""
    log: list[str] = []
    h.setCallback(lambda _t, msg, *_a: log.append(msg), None)
    h.startCallback(highspy.cb.kCallbackLogging)
    for k, v in options.items():
        h.setOptionValue(k, v)
    h.run()
    return "".join(log)


_NUM = r"([-+0-9.eE]+|inf)"
_ROW = re.compile(
    rf"num/max\s+(\d+) /\s*{_NUM} \(relative\s+(\d+) /\s*{_NUM}\)\s+(primal|dual) "
    r"(infeasibilities|residual errors)"
)
_PD = re.compile(rf"^\s+\d+ /\s*{_NUM}\s+P-D objective error", re.M)
_X = re.compile(
    rf'(?:no more than|to "Unknown" since relative violation of tolerances is)\s+{_NUM}'
)


def _full_lp_block(log: str) -> str:
    """The log of HiGHS' check of the full LP in the space it solved."""
    log = log.split("After solving the user-scaled model")[0]
    log = log.split("Solution optimality conditions: After removing user scaling")[0]
    return log.rsplit("Performed postsolve", 1)[-1]


def _logged(log: str) -> tuple[float | None, dict[str, float]]:
    """HiGHS' logged X (None when not printed) and its KKT table."""
    block = _full_lp_block(log)
    xs = _X.findall(block)
    table: dict[str, float] = {}
    for m in _ROW.finditer(block):
        what = m.group(5) + "_" + ("infeasibilit" if m.group(6)[0] == "i" else "residual_error")
        num_s = "ies" if what.endswith("t") else "s"
        max_s = "y" if what.endswith("t") else ""
        table[f"num_{what}{num_s}"] = int(m.group(1))
        table[f"max_{what}{max_s}"] = float(m.group(2))
        table[f"num_relative_{what}{num_s}"] = int(m.group(3))
        table[f"max_relative_{what}{max_s}"] = float(m.group(4))
    pd = _PD.findall(block)
    if pd:
        table["primal_dual_objective_error"] = float(pd[-1])
    return (float(xs[-1]) if xs else None), table


def _at_3_digits(value: float, printed: float) -> bool:
    """*value* rounds to HiGHS' ``%8.3g`` *printed* (half a unit of the 3rd digit)."""
    if printed == 0:
        return value == 0
    return abs(value - printed) <= 0.5 * 10 ** (math.floor(math.log10(abs(printed))) - 2) * 1.0001


def _residual_floors(
    h: highspy.Highs, rep
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-entry float64 residuals and their rounding floors, in the solved space."""
    lp, sol = h.getLp(), h.getSolution()
    fb, fo = 2.0**rep.user_bound_scale, 2.0**rep.user_objective_scale
    col, row, a = _kkt._matrix(lp)
    x, r = np.asarray(sol.col_value) * fb, np.asarray(sol.row_value) * fb
    y, d = np.asarray(sol.row_dual) * fo, np.asarray(sol.col_dual) * fo
    c = np.asarray(lp.col_cost_) * fo
    m, n = r.size, x.size
    k_row = np.bincount(row, minlength=m) + 1
    k_col = np.bincount(col, minlength=n) + 2
    eps = np.finfo(float).eps
    pres = np.abs(np.bincount(row, a * x[col], minlength=m) - r)
    pfloor = k_row * eps * (np.bincount(row, np.abs(a * x[col]), minlength=m) + np.abs(r))
    dres = np.abs(np.bincount(col, a * y[row], minlength=n) - c + d)
    dfloor = (
        k_col * eps * (np.bincount(col, np.abs(a * y[row]), minlength=n) + np.abs(c) + np.abs(d))
    )
    return pres, pfloor, dres, dfloor


def _residual_counts_are_robust(h, rep) -> bool:
    """No residual entry lies within its rounding floor of a count threshold."""
    pres, pfloor, dres, dfloor = _residual_floors(h, rep)
    t = rep.tolerances
    for res, floor, tol, norm in (
        (pres, pfloor, t.primal_residual, rep.bound_norm),
        (dres, dfloor, t.dual_residual, rep.cost_norm),
    ):
        for threshold in (tol, tol * (1 + norm)):
            if np.any(np.abs(res - threshold) <= floor):
                return False
    return True


def _assert_matches(rep, expected: dict[str, float], h, *, exact: bool) -> None:
    """Report fields equal *expected* (``getInfo()`` when *exact*, else the log)."""
    if not rep.basis_valid:
        robust = _residual_counts_are_robust(h, rep)
        assert robust, "a residual sits at its rounding floor: pick another fixture"
    for name, want in expected.items():
        got = getattr(rep, name)
        if name.startswith("num"):
            assert got == want, name
            continue
        if ("residual" in name or "objective" in name) and not math.isinf(want):
            # Rounding-sized values: float64 sums vs HiGHS' quad / ordered sums.
            norm = rep.bound_norm if "primal_res" in name else rep.cost_norm
            floor = 1e-12 if "relative" in name or "objective" in name else 1e-12 * (1 + norm)
            if want <= floor:
                assert got <= 10 * floor, name
                continue
        if exact:
            assert got == pytest.approx(want, rel=1e-9, abs=1e-15), name
        else:
            assert _at_3_digits(got, want), (name, got, want)


def _info(h) -> dict[str, float]:
    info = h.getInfo()
    return {f: getattr(info, f) for f in [*_FIELDS, "primal_dual_objective_error"]}


_UNSCALED = {
    "simplex": {},
    "ipm": {"solver": "ipm", "run_crossover": "off"},
    "ipm_loose": {"solver": "ipm", "run_crossover": "off", "ipm_optimality_tolerance": 1e-5},
    # Ends "Unknown" with dual infeasibilities whose relative measure uses the
    # column's own cost (HighsSolution.cpp:366-378).
    "ipm_rough": {"solver": "ipm", "run_crossover": "off", "ipm_optimality_tolerance": 1e-1},
}


@requires_highs_115
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("mode", list(_UNSCALED))
def test_unscaled_report_equals_getinfo(mode, seed):
    h = _small(seed)
    log = _run(h, **_UNSCALED[mode])
    rep = kkt_report(h)
    assert rep.basis_valid == (mode == "simplex")
    _assert_matches(rep, _info(h), h, exact=True)
    logged_x, _table = _logged(log)
    x_info, _terms = tolerance_violation(
        h.getInfo(), basis_valid=rep.basis_valid, tolerances=rep.tolerances
    )
    assert x_info == pytest.approx(rep.max_tolerance_relative_violation, rel=1e-9, abs=1e-6)
    x = rep.max_tolerance_relative_violation
    if logged_x is not None:
        assert _at_3_digits(x, logged_x)
    elif h.getModelStatus() == highspy.HighsModelStatus.kOptimal:
        assert x <= 1  # HiGHS prints X when 1 < X
    else:
        assert x > 100  # an "Unknown" with X <= 100 becomes "Optimal"
    if mode == "ipm_loose" and seed in (0, 2):
        # The forcing recipe: HiGHS keeps "Optimal" at X > 1.
        assert logged_x is not None and logged_x > 1
        assert h.getModelStatus() == highspy.HighsModelStatus.kOptimal


@requires_highs_115
@pytest.mark.parametrize("seed", [1, 2])
def test_user_scaled_ipm_equals_highs_log(seed):
    h = _small(seed)
    log = _run(
        h,
        solver="ipm",
        run_crossover="off",
        ipm_optimality_tolerance=1e-5,
        user_objective_scale=-5,
        user_bound_scale=-3,
    )
    rep = kkt_report(h)
    logged_x, table = _logged(log)
    assert (rep.user_bound_scale, rep.user_objective_scale, rep.basis_valid) == (-3, -5, False)
    assert logged_x is not None and logged_x > 1
    assert _at_3_digits(rep.max_tolerance_relative_violation, logged_x)
    assert len(table) == 17
    _assert_matches(rep, table, h, exact=False)
    # getInfo() holds the unscaled numbers: not what HiGHS judged.
    assert h.getInfo().num_primal_residual_errors == -1


@requires_highs_115
@pytest.mark.parametrize(
    "options",
    [{"solver": "simplex"}, {"solver": "ipm", "run_crossover": "on"}],
    ids=["simplex", "crossover"],
)
def test_user_scaled_basis_solve_equals_highs_log(options):
    h = _wide(0)
    log = _run(
        h,
        **options,
        primal_feasibility_tolerance=1e-4,
        dual_feasibility_tolerance=1e-4,
        kkt_tolerance=1e-10,
        user_objective_scale=-6,
        user_bound_scale=-4,
    )
    rep = kkt_report(h)
    logged_x, table = _logged(log)
    assert rep.basis_valid
    assert logged_x is not None and logged_x > 100  # status turned "Unknown"
    assert _at_3_digits(rep.max_tolerance_relative_violation, logged_x)
    _assert_matches(rep, table, h, exact=False)
    assert rep.num_primal_residual_errors == -1 and math.isinf(rep.max_primal_residual_error)


@requires_highs_115
def test_kkt_tolerance_replaces_all_five():
    h = _small(0)
    _run(h, solver="ipm", run_crossover="off", kkt_tolerance=1e-9)
    assert kkt_tolerances(h) == polar_high.KktTolerances(*[1e-9] * 5)
    rep = kkt_report(h)
    assert rep.tolerances == kkt_tolerances(h)
    _assert_matches(rep, _info(h), h, exact=True)


def test_tolerance_violation_rule():
    """The two branches of lpKktCheck on a hand-made mapping."""
    v = dict.fromkeys(_FIELDS, 0)
    v.update(num_primal_residual_errors=-1, primal_dual_objective_error=5e-7)
    v.update(num_relative_dual_infeasibilities=7, max_relative_dual_infeasibility=1.19e-7)
    tol = polar_high.KktTolerances(1e-7, 1e-7, 1e-7, 1e-7, 1e-7)
    x, terms = tolerance_violation(v, basis_valid=True, tolerances=tol)
    assert x == pytest.approx(5) and terms == (("primal_dual_objective_error", pytest.approx(5)),)
    v.update(num_primal_infeasibilities=1, max_primal_infeasibility=8e-7)
    assert tolerance_violation(v, basis_valid=True, tolerances=tol)[0] == pytest.approx(8)
    x, terms = tolerance_violation(v, basis_valid=False, tolerances=tol)
    assert x == pytest.approx(5)
    assert dict(terms)["max_relative_dual_infeasibility"] == pytest.approx(1.19)
    v["primal_dual_objective_error"] = 1e-8
    x, terms = tolerance_violation(v, basis_valid=False, tolerances=tol)
    assert x == pytest.approx(1.19) and "primal_dual_objective_error" not in dict(terms)


class _Version116:
    """A handle that says it is HiGHS 1.16."""

    def __init__(self, h):
        self._h = h

    def __getattr__(self, name):
        return getattr(self._h, name)

    def versionMinor(self):
        return 16

    def version(self):
        return "1.16.0"


def test_none_cases():
    h = _small(0)
    assert kkt_report(h) is None  # before run()
    _run(h)
    if not _IS_115:
        assert kkt_report(h) is None  # the real other version
    assert kkt_report(_Version116(h)) is None
    h.changeColIntegrality(0, highspy.HighsVarType.kInteger)
    _run(h)
    assert kkt_report(h) is None


@requires_highs_115
def test_basis_branch_reads_no_matrix(monkeypatch):
    h = _small(1)
    _run(h)

    def boom(lp):
        raise AssertionError("matrix read")

    monkeypatch.setattr(_kkt, "_matrix", boom)
    rep = kkt_report(h)
    assert rep is not None and rep.basis_valid
    assert rep.num_primal_infeasibilities == h.getInfo().num_primal_infeasibilities
