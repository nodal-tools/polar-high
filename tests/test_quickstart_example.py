"""Verifies the worked example from docs/quickstart.md and README.md."""

import quickstart_example as ex  # conftest adds tests/fixtures/ to sys.path


def test_objective():
    assert ex.sol.optimal
    assert abs(ex.sol.obj - 72.0) < 1e-6


def test_wind_at_full_capacity():
    prod = ex.sol.value("v_production")
    wind = prod.filter(prod["unit"] == "wind").sort("hour")
    # wind capacity per hour: 3, 1, 4
    assert wind["value"].to_list() == [3.0, 1.0, 4.0]


def test_coal_fills_gap():
    prod = ex.sol.value("v_production")
    coal = prod.filter(prod["unit"] == "coal").sort("hour")
    # demand per hour: 5, 6, 4; wind: 3, 1, 4 → coal: 2, 5, 0
    assert coal["value"].to_list() == [2.0, 5.0, 0.0]
