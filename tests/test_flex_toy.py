"""Stage-1 flex-toy: synthetic flextool-flavored topology with 5
parameter sweeps.  Hand-verifiable expected obj for each."""

import pytest

from polar_high_opt import Problem
from flex_toy_data import make_flex_toy_data
from flex_toy_model import build_flex_toy


@pytest.mark.parametrize("kwargs,expected_obj,expected_dispatch", [
    ({}, 1300.0,
     {"GT": [10.0, 60.0, 0.0, 60.0], "WIND": [50.0, 30.0, 70.0, 20.0]}),

    ({"demand_values": (80, 80, 80, 80)}, 1400.0,
     {"GT": [30.0, 50.0, 0.0, 60.0], "WIND": [50.0, 30.0, 80.0, 20.0]}),

    ({"gt_efficiency": 0.4}, 1625.0,
     {"GT": [10.0, 60.0, 0.0, 60.0], "WIND": [50.0, 30.0, 70.0, 20.0]}),

    ({"wind_avail_values": (70, 90, 70, 80)}, 0.0,
     {"GT": [0.0, 0.0, 0.0, 0.0], "WIND": [60.0, 90.0, 70.0, 80.0]}),
],
ids=["A_default", "B_flat_demand", "C_low_efficiency", "D_wind_covers"])
def test_flex_toy_obj_and_dispatch(kwargs, expected_obj, expected_dispatch):
    pb = Problem()
    build_flex_toy(pb, make_flex_toy_data(**kwargs))
    sol = pb.solve()
    assert sol.optimal
    assert abs(sol.obj - expected_obj) <= 1e-4 * max(1.0, abs(expected_obj))
    flow = sol.value("v_flow").sort("p", "t")
    for p_name, expected in expected_dispatch.items():
        rows = flow.filter(flow["p"] == p_name).sort("t")
        actual = rows["value"].to_list()
        assert all(abs(a - e) < 1e-4 for a, e in zip(actual, expected)), \
            f"{p_name}: {actual} != {expected}"


def test_flex_toy_free_gas():
    """Degenerate dispatch — only check obj == 0."""
    pb = Problem()
    build_flex_toy(pb, make_flex_toy_data(gas_price_value=0.0))
    sol = pb.solve()
    assert sol.optimal
    assert abs(sol.obj) < 1e-6
