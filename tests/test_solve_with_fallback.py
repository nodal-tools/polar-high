""":meth:`WarmProblem.solve_with_fallback` — one-shot option-swap retry.

The primitive backs cutting-plane drivers whose primary solve can come back
non-optimal under an aggressive option set (e.g. an interior-point run kept
crossover-free for speed): solve once, and only if the primary fails to
certify ``kOptimal``, apply ``fallback_options`` via ``setOptionValue`` on
the live handle (``solve(options=...)`` is honoured on the FIRST solve only),
``clearSolver()``, re-run once, and RESTORE the prior option values.

The "demonstrably falls back" case forces a non-optimal primary with
``solver="ipm"`` under an absurd iteration limit (0) and presolve off (so
presolve cannot solve the LP outright before IPM gets its zero iterations).
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import polar_high as fp

# Hand-verified optimum of the fixture LP: min Σ cost·x, x >= demand
# with cost = [1, 2, 3], demand = [3, 4, 5] ⇒ obj = 3 + 8 + 15 = 26.
_OPT_OBJ = 26.0


def _build_problem(solver_options: dict) -> fp.Problem:
    p = fp.Problem()
    idx = pl.DataFrame({"i": [0, 1, 2]})
    x = p.add_var("x", "i", idx, lower=0.0, upper=10.0)
    demand = fp.Param(("i",), pl.DataFrame({"i": [0, 1, 2], "value": [3.0, 4.0, 5.0]}))
    p.add_cstr("meet", over=idx, sense=">=", lhs_terms={"x": x}, rhs_terms={"d": demand})
    cost = fp.Param(("i",), pl.DataFrame({"i": [0, 1, 2], "value": [1.0, 2.0, 3.0]}))
    p.set_objective(cost * x, sense="min")
    p.set_solver_options({"output_flag": False, **solver_options})
    return p


# The option set that reliably yields a NON-optimal primary: IPM with zero
# iterations allowed, presolve off so the reduction cannot solve the LP by
# itself, crossover off so nothing cleans up after the truncated IPM.
_BAD_PRIMARY = {
    "solver": "ipm",
    "ipm_iteration_limit": 0,
    "presolve": "off",
    "run_crossover": "off",
}


def _get_opt(wp: fp.WarmProblem, name: str):
    status, value = wp._h.getOptionValue(name)
    assert str(status).endswith("kOk")
    return value


# ----------------------------------------------------------------------------


def test_primary_path_on_optimal_solve() -> None:
    """An LP that solves optimally first try returns ("primary") and the
    fallback options are never applied."""
    wp = fp.WarmProblem(_build_problem({}))
    sol, path = wp.solve_with_fallback({"solver": "simplex", "presolve": "on"})
    assert path == "primary"
    assert sol.optimal
    assert sol.obj == pytest.approx(_OPT_OBJ, abs=1e-9)
    # Options untouched: still the HiGHS default, not the fallback values.
    assert _get_opt(wp, "solver") == "choose"


def test_fallback_fires_and_restores_options() -> None:
    """Force a non-optimal primary; the fallback must recover the certified
    optimum, and the prior option values must be restored afterwards."""
    wp = fp.WarmProblem(_build_problem(_BAD_PRIMARY))

    # Sanity: the primary configuration really is non-optimal on its own.
    probe = fp.WarmProblem(_build_problem(_BAD_PRIMARY))
    assert not probe.solve().optimal

    sol, path = wp.solve_with_fallback({"solver": "simplex", "presolve": "on"})
    assert path == "fallback"
    assert sol.optimal
    assert sol.obj == pytest.approx(_OPT_OBJ, abs=1e-9)
    assert np.allclose(sol.value("x").sort("i")["value"].to_numpy(), [3.0, 4.0, 5.0])

    # Prior option values restored — both the swapped keys...
    assert _get_opt(wp, "solver") == "ipm"
    assert _get_opt(wp, "presolve") == "off"
    # ...and the untouched ones.
    assert _get_opt(wp, "ipm_iteration_limit") == 0
    assert _get_opt(wp, "run_crossover") == "off"

    # Behavioural proof of the restore: a plain re-solve runs under the
    # restored (bad) primary options again and is again non-optimal.
    assert not wp.solve().optimal


def test_fallback_solution_reported_even_if_still_not_optimal() -> None:
    """A fallback that ALSO fails to certify is returned as-is (path
    "fallback", sol.optimal False) — the caller decides what that means."""
    wp = fp.WarmProblem(_build_problem(_BAD_PRIMARY))
    # Fallback keeps the absurd IPM limit — still non-optimal.
    sol, path = wp.solve_with_fallback({"ipm_iteration_limit": 1})
    assert path == "fallback"
    assert not sol.optimal
    # Restore still happened.
    assert _get_opt(wp, "ipm_iteration_limit") == 0


def test_unknown_fallback_option_raises_and_restores() -> None:
    """An unknown option name raises ValueError; options changed before the
    failure are restored."""
    wp = fp.WarmProblem(_build_problem(_BAD_PRIMARY))
    with pytest.raises(ValueError, match="unknown HiGHS option"):
        # First key is valid and gets applied; second is bogus.
        wp.solve_with_fallback({"solver": "simplex", "no_such_option": 1})
    assert _get_opt(wp, "solver") == "ipm"
