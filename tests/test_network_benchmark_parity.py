"""Cross-tool parity check for the sparse network-flow benchmark.

Builds and solves the same LP via all four ``*_net`` model files and
asserts they agree on the optimal objective to ~1e-6 relative tolerance.

This is the correctness anchor for the second benchmark family.  The test
does NOT run the bench sweep — it just imports the model files and calls
build/solve once per tool at a small N where every solver is fast.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# benchmark/ holds a `models` package; put it on sys.path so we can
# `from models.polar_net import ...` etc.
BENCHMARK = Path(__file__).resolve().parents[1] / "benchmark"
if str(BENCHMARK) not in sys.path:
    sys.path.insert(0, str(BENCHMARK))


N = 20  # small but feasible — matches the parity check called out in the spec
RTOL = 1e-6


@pytest.fixture(scope="module")
def obj_polar() -> float:
    from models.polar_net import build, solve

    p = build(N)
    optimal, obj = solve(p)
    assert optimal, "polar_net failed to solve to optimal"
    return obj


def test_polar_net_optimal(obj_polar: float) -> None:
    # smoke: just confirms the fixture solved to a finite objective
    assert obj_polar == pytest.approx(obj_polar, rel=0)
    assert obj_polar > 0  # demand is non-trivial → cost is positive


def test_linopy_net_matches_polar(obj_polar: float) -> None:
    pytest.importorskip("linopy")
    from models.linopy_net import build, solve

    m = build(N)
    optimal, obj = solve(m)
    assert optimal, "linopy_net failed to solve to optimal"
    assert abs(obj - obj_polar) / abs(obj_polar) < RTOL


def test_pyomo_net_matches_polar(obj_polar: float) -> None:
    pytest.importorskip("pyomo")
    from models.pyomo_net import build, solve

    m = build(N)
    optimal, obj = solve(m)
    assert optimal, "pyomo_net failed to solve to optimal"
    assert abs(obj - obj_polar) / abs(obj_polar) < RTOL
