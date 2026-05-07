"""Verifies the worked example from docs/guide/debugging.md.

The model is built by importing debug_example (module-level code runs
on import).  Tests assert the pre-solve state and post-solve values
that the guide page shows as expected outputs.
"""
import debug_example as ex  # conftest adds tests/fixtures/ to sys.path


def test_pre_solve_structure():
    assert ex.p.cstr_names() == ["balance", "cap"]
    assert ex.p.cstr_row_count("balance") == 2
    assert ex.p.cstr_row_count("cap") == 2
    assert ex.v_flow.frame.height == 2
    assert ex.v_dump.frame.height == 2


def test_solve_values():
    sol = ex.p.solve(keep_solver=True)
    assert sol.optimal
    assert abs(sol.obj - 200.0) < 1e-6

    flow = sol.value("v_flow")
    assert flow["value"].to_list() == [100.0, 100.0]

    dump = sol.value("v_dump")
    assert dump["value"].to_list() == [0.0, 0.0]

    duals = sol.constraint_dual("balance")
    assert all(abs(d - 1.0) < 1e-6 for d in duals["dual"].to_list())
