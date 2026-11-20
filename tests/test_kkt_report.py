"""``kkt_report`` / ``tolerance_violation`` against HiGHS itself.

The oracle is HiGHS: ``getInfo()`` on unscaled solves, and HiGHS' own log
(captured by the logging callback) on user-scaled ones, where ``getInfo()``
holds the unscaled numbers.  No stored numbers: seeded LP generators.

``POLAR_HIGH_REQUIRE_HIGHS_115=1`` (the CI job pinned to highspy 1.15.1)
makes a HiGHS other than 1.15.1 an error instead of skipping the oracle
tests.
"""

from __future__ import annotations

import dataclasses
import math
import os
import re

import highspy
import numpy as np
import pytest

import polar_high
from polar_high import _kkt, kkt_report, kkt_tolerances, tolerance_violation

_VERSION = highspy.Highs().version()
_IS_115 = _VERSION.startswith("1.15.") and _VERSION != "1.15.0"
if os.environ.get("POLAR_HIGH_REQUIRE_HIGHS_115") == "1" and _VERSION != "1.15.1":
    raise RuntimeError(f"POLAR_HIGH_REQUIRE_HIGHS_115=1 but HiGHS is {_VERSION}")
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


def _mixed(seed: int, n: int = 300, m: int = 200, sense: int = 1, offset: float = 0.0):
    """Free, ``(-inf, u]``, ``[l, u]`` with ``l < 0`` and ``[0, inf)`` columns;
    equality, ranged and ``<=`` rows; minimize or maximize; an offset."""
    h = highspy.Highs()
    h.setOptionValue("log_to_console", False)
    rng = np.random.default_rng(seed)
    kind = rng.integers(0, 4, n)
    lo = np.where(kind == 0, -INF, np.where(kind == 1, -rng.uniform(1, 50, n), 0.0))
    up = np.where(kind == 0, INF, np.where(kind == 3, INF, rng.uniform(1, 100, n)))
    free = rng.random(n) < 0.05
    lo[free], up[free] = -INF, INF
    cost = rng.uniform(-5, 5, n)
    cost = np.where(kind == 3, np.abs(cost), cost)
    upper_only = (kind == 0) & ~free
    up = np.where(upper_only, rng.uniform(1, 100, n), up)
    cost = np.where(upper_only, -np.abs(cost), cost)
    cost = np.where(free, 0.0, cost) * sense
    h.addVars(n, lo, up)
    h.changeColsCost(n, np.arange(n, dtype=np.int32), cost)
    for _ in range(m):
        idx = rng.choice(n, 8, replace=False).astype(np.int32)
        kind_row, b = rng.integers(0, 3), rng.uniform(-20, 20)
        lower, upper = ((b, b), (b - 30, b + 30), (-INF, b + 50))[kind_row]
        h.addRow(lower, upper, 8, idx, rng.uniform(-10, 10, 8))
    h.changeObjectiveOffset(offset)
    if sense < 0:
        h.changeObjectiveSense(highspy.ObjSense.kMaximize)
    return h


def _big_free_rows(seed: int, coefficient: float, *, free: bool) -> highspy.Highs:
    """200 columns, 120 ``<=`` rows and 20 rows with coefficients ~*coefficient*
    whose activity is far above the bound norm (free, or a far upper bound):
    plain float64 row sums err by more than the residual tolerance there."""
    h = highspy.Highs()
    h.setOptionValue("log_to_console", False)
    rng = np.random.default_rng(seed)
    n = 200
    h.addVars(n, np.zeros(n), rng.uniform(1, 10, n))
    h.changeColsCost(n, np.arange(n, dtype=np.int32), -rng.uniform(0.1, 1, n))
    for _ in range(120):
        idx = rng.choice(n, 30, replace=False).astype(np.int32)
        h.addRow(-INF, rng.uniform(50, 100), 30, idx, rng.uniform(0.1, 3, 30))
    upper = INF if free else 1e5 * coefficient
    for _ in range(20):
        idx = rng.choice(n, 100, replace=False).astype(np.int32)
        value = rng.uniform(-1, 1, 100) * coefficient * (1 + rng.random(100))
        h.addRow(-INF, upper, 100, idx, value)
    return h


_FAMILIES = {
    "small": _small,
    "mixed_min": lambda seed: _mixed(seed, offset=1e5),
    "mixed_max": lambda seed: _mixed(seed, sense=-1, offset=-3e3),
}


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
    """Per-entry residuals and how far they can be from HiGHS', in the
    solved space.  Both sides round an almost exact sum of the same float64
    products; they differ (by an ulp of the activity) only when the exact
    sum is within the error of the almost exact sums, about
    ``k**2 * eps**2 * sum |a_ij x_j|``, of a rounding boundary."""
    lp, sol = h.getLp(), h.getSolution()
    fb, fo = 2.0**rep.user_bound_scale, 2.0**rep.user_objective_scale
    col, row, a = _kkt._matrix(lp)
    x, r = np.asarray(sol.col_value) * fb, np.asarray(sol.row_value) * fb
    y, d = np.asarray(sol.row_dual) * fo, np.asarray(sol.col_dual) * fo
    c = np.asarray(lp.col_cost_) * fo
    m, n = r.size, x.size
    eps2 = np.finfo(float).eps ** 2
    k_row, k_col = np.bincount(row, minlength=m) + 1, np.bincount(col, minlength=n) + 1
    pres = np.abs(_kkt._sums(row, a * x[col], m) - r)
    pfloor = 16 * k_row**2 * eps2 * np.bincount(row, np.abs(a * x[col]), m)
    dres = np.abs(_kkt._sums(col, a * y[row], n) - c + d)
    dfloor = 16 * k_col**2 * eps2 * np.bincount(col, np.abs(a * y[row]), n)
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
            # Rounding-sized values: last bits of the sums, P-D dot products.
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
@pytest.mark.parametrize("family", list(_FAMILIES))
def test_unscaled_report_equals_getinfo(family, mode, seed):
    h = _FAMILIES[family](seed)
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
    if family == "small" and mode == "ipm_loose" and seed in (0, 2):
        # The forcing recipe: HiGHS keeps "Optimal" at X > 1.
        assert logged_x is not None and logged_x > 1
        assert h.getModelStatus() == highspy.HighsModelStatus.kOptimal


@requires_highs_115
@pytest.mark.parametrize(
    ("family", "seed", "scales"),
    [
        ("small", 1, (-3, -5)),
        ("small", 2, (-3, -5)),
        ("mixed_min", 0, (2, 3)),
        ("mixed_min", 1, (-2, 0)),
    ],
)
def test_user_scaled_ipm_equals_highs_log(family, seed, scales):
    h = _FAMILIES[family](seed)
    log = _run(
        h,
        solver="ipm",
        run_crossover="off",
        ipm_optimality_tolerance=1e-5,
        user_bound_scale=scales[0],
        user_objective_scale=scales[1],
    )
    rep = kkt_report(h)
    logged_x, table = _logged(log)
    assert (rep.user_bound_scale, rep.user_objective_scale, rep.basis_valid) == (*scales, False)
    assert logged_x is not None and logged_x > 1
    assert _at_3_digits(rep.max_tolerance_relative_violation, logged_x)
    assert len(table) == 17
    _assert_matches(rep, table, h, exact=False)
    # getInfo() holds the unscaled numbers: not what HiGHS judged, and no
    # residuals, which the no-basis rule needs.
    assert h.getInfo().num_primal_residual_errors == -1
    with pytest.raises(ValueError, match="residuals were not computed"):
        tolerance_violation(h.getInfo(), basis_valid=False, tolerances=rep.tolerances)


@requires_highs_115
@pytest.mark.parametrize(
    ("seed", "coefficient", "free", "highs_x"),
    [
        (0, 1e9, True, "> 1, Optimal"),
        (0, 1e11, True, "> 100, Unknown"),
        (0, 1e9, False, "<= 1"),
        (1, 1e11, False, "> 1, Optimal"),
    ],
)
def test_large_activity_rows_equal_getinfo(seed, coefficient, free, highs_x):
    """Rows with activity far above the bound norm: the residuals of
    float64 row sums would be wrong by more than the tolerance (X 0.025
    instead of 3.93, 3.93 instead of 0.393, 246 instead of 24.6)."""
    h = _big_free_rows(seed, coefficient, free=free)
    log = _run(h, solver="ipm", run_crossover="off", presolve="off" if free else "on")
    rep = kkt_report(h)
    assert not rep.basis_valid
    _assert_matches(rep, _info(h), h, exact=True)
    logged_x, _table = _logged(log)
    x = rep.max_tolerance_relative_violation
    status = h.getModelStatus()
    if highs_x == "<= 1":
        assert logged_x is None and x <= 1 and status == highspy.HighsModelStatus.kOptimal
    else:
        assert logged_x is not None and _at_3_digits(x, logged_x)
        want = highspy.HighsModelStatus.kOptimal if "Optimal" in highs_x else None
        if want is None:
            assert x > 100 and status == highspy.HighsModelStatus.kUnknown
        else:
            assert 1 < x <= 100 and status == want
    # What makes it hard: the rounding error bound of plain float64 row
    # sums, relative, is above the residual tolerance.
    col, row, a = _kkt._matrix(h.getLp())
    activity = np.bincount(row, np.abs(a * np.asarray(h.getSolution().col_value)[col]))
    plain_floor = 100 * np.finfo(float).eps * activity / (1 + rep.bound_norm)
    assert plain_floor.max() > rep.tolerances.primal_residual


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


def test_sums_are_almost_exact():
    rng = np.random.default_rng(3)
    key = rng.integers(0, 40, 5000)
    terms = rng.uniform(-1, 1, 5000) * 10 ** rng.uniform(-6, 12, 5000)
    terms[:4], key[:4] = [1e16, 1.0, -1e16, 1e-3], 0
    want = np.array([math.fsum(terms[key == i]) for i in range(41)])
    assert np.array_equal(_kkt._sums(key, terms, 41), want)


def test_kkt_report_is_keyword_only():
    with pytest.raises(TypeError):
        polar_high.KktReport(*range(26))


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
    with pytest.raises(ValueError, match="residuals were not computed"):
        tolerance_violation(v, basis_valid=False, tolerances=tol)
    v["num_primal_residual_errors"] = 0
    x, terms = tolerance_violation(v, basis_valid=False, tolerances=tol)
    assert x == pytest.approx(5)
    assert dict(terms)["max_relative_dual_infeasibility"] == pytest.approx(1.19)
    v["primal_dual_objective_error"] = 1e-8
    x, terms = tolerance_violation(v, basis_valid=False, tolerances=tol)
    assert x == pytest.approx(1.19) and "primal_dual_objective_error" not in dict(terms)


class _Version:
    """A handle that says it is another HiGHS version."""

    def __init__(self, h, minor: int, patch: int):
        self._h, self._minor, self._patch = h, minor, patch

    def __getattr__(self, name):
        return getattr(self._h, name)

    def versionMinor(self):
        return self._minor

    def versionPatch(self):
        return self._patch

    def version(self):
        return f"1.{self._minor}.{self._patch}"


class _NoLp:
    """A handle whose ``getLp()`` (the LP copy, with the matrix) fails."""

    def __init__(self, h):
        self._h = h

    def __getattr__(self, name):
        return getattr(self._h, name)

    def getLp(self):
        raise AssertionError("getLp() called")


def _unbounded() -> highspy.Highs:
    h = highspy.Highs()
    h.setOptionValue("log_to_console", False)
    h.addVars(2, np.zeros(2), np.full(2, INF))
    h.changeColsCost(2, np.arange(2, dtype=np.int32), np.array([-1.0, -1.0]))
    h.addRow(-INF, 1.0, 2, np.arange(2, dtype=np.int32), np.array([1.0, -1.0]))
    return h


def test_none_cases(caplog):
    caplog.set_level("DEBUG", logger="polar_high._kkt")
    h = _small(0)
    assert kkt_report(h) is None  # before run()
    _run(h)
    if not _IS_115:
        assert kkt_report(h) is None  # the real other version
    assert kkt_report(_Version(h, 16, 0)) is None
    assert kkt_report(_Version(h, 15, 0)) is None  # untested patch release
    assert "1.15.0" in caplog.text
    # A MIP solved as a relaxation has valid duals: the integrality check.
    h.changeColIntegrality(0, highspy.HighsVarType.kInteger)
    _run(h, solve_relaxation=True)
    assert h.getSolution().dual_valid
    assert kkt_report(h) is None
    if _IS_115:
        assert "not continuous" in caplog.text
    _run(h, solve_relaxation=False)
    assert kkt_report(h) is None
    # A QP: min x0^2 + x1^2 - x0 - x1, x0 + x1 >= 1.
    h = highspy.Highs()
    h.setOptionValue("log_to_console", False)
    h.addVars(2, np.zeros(2), np.full(2, 10.0))
    h.changeColsCost(2, np.arange(2, dtype=np.int32), np.array([-1.0, -1.0]))
    h.addRow(1.0, INF, 2, np.arange(2, dtype=np.int32), np.ones(2))
    hessian = highspy.HighsHessian()
    hessian.dim_, hessian.format_ = 2, highspy.HessianFormat.kTriangular
    hessian.start_ = np.arange(3, dtype=np.int32)
    hessian.index_, hessian.value_ = np.arange(2, dtype=np.int32), np.full(2, 2.0)
    h.passHessian(hessian)
    _run(h)
    assert h.getModelStatus() == highspy.HighsModelStatus.kOptimal
    assert h.getSolution().dual_valid and kkt_report(h) is None
    if _IS_115:
        assert "Hessian" in caplog.text


@requires_highs_115
@pytest.mark.parametrize("options", [{}, {"solver": "ipm", "run_crossover": "off"}])
def test_unbounded_has_no_report(options, caplog):
    """HiGHS keeps valid vectors for kUnbounded but does not judge them."""
    caplog.set_level("DEBUG", logger="polar_high._kkt")
    h = _unbounded()
    _run(h, **options)
    assert h.getModelStatus() == highspy.HighsModelStatus.kUnbounded
    sol = h.getSolution()
    assert sol.value_valid and sol.dual_valid
    assert kkt_report(h) is None
    assert "Unbounded" in caplog.text


@requires_highs_115
@pytest.mark.parametrize("family", list(_FAMILIES))
def test_basis_branch_reads_no_lp_copy(family, monkeypatch):
    h = _FAMILIES[family](1)
    _run(h)

    def boom(lp):
        raise AssertionError("matrix read")

    monkeypatch.setattr(_kkt, "_matrix", boom)
    rep = kkt_report(_NoLp(h))
    assert rep is not None and rep.basis_valid
    _assert_matches(rep, _info(h), h, exact=True)
