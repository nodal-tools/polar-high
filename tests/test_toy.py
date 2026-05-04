"""Pure-engine validation: 5-constraint synthetic LP, hand-verifiable."""

from polar_high_opt import Problem
from toy_data import make_toy_data
from toy_model import build_dispatch


def test_toy_dispatch_obj():
    pb = Problem()
    build_dispatch(pb, make_toy_data())
    sol = pb.solve()
    assert sol.optimal
    assert abs(sol.obj - 6500.0) < 1e-6
