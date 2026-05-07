"""Verifies the worked example from docs/guide/lagrangian.md."""

import lagrangian_example as ex  # conftest adds tests/fixtures/ to sys.path


def test_best_dual_matches_closed_form():
    # Optimal: both agree at max(4, 2) = 4, total cost = 4 + 4 = 8.
    # Dual gap is zero for this LP so best_dual == LP optimum.
    assert abs(ex.sol.best_dual_total - 8.0) / 8.0 < 1e-6


def test_report_kind():
    assert ex.sol.report_kind == "best_dual"
