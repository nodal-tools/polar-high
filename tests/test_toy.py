"""Pure-engine validation: 5-constraint synthetic LP, hand-verifiable."""

from toy_data import make_toy_data
from toy_model import build_dispatch

from polar_high import Problem


def test_toy_dispatch_obj():
    pb = Problem()
    build_dispatch(pb, make_toy_data())
    sol = pb.solve()
    assert sol.optimal
    assert abs(sol.obj - 6500.0) < 1e-6


def test_resolve_does_not_use_stale_term_cache():
    """A Problem must be re-solvable: solve() materializes terms into
    locals only, never populating an eager cache on _Term.  Both solves
    must produce the same objective."""
    pb = Problem()
    build_dispatch(pb, make_toy_data())
    sol1 = pb.solve()
    sol2 = pb.solve()
    assert sol1.optimal and sol2.optimal
    assert abs(sol1.obj - sol2.obj) < 1e-9
    # Terms must not grow a cached eager frame as a side-effect of
    # solve(); the lazy plan stays the source of truth.
    for cstr_name in pb.cstr_names():
        for cr in pb.cstrs_named(cstr_name):
            for t in cr.proto.expr.terms:
                assert not hasattr(t, "_frame_cache")


def test_keep_solver_flag():
    """Default ``keep_solver=False`` drops the live HiGHS handle;
    ``keep_solver=True`` keeps it for post-solve inspection."""
    pb = Problem()
    build_dispatch(pb, make_toy_data())
    sol_default = pb.solve()
    assert sol_default.highs is None
    sol_keep = pb.solve(keep_solver=True)
    assert sol_keep.highs is not None
    assert abs(sol_default.obj - sol_keep.obj) < 1e-9


def test_solution_streamed_lp_ranges_populated():
    """:attr:`Solution.streamed_lp_ranges` reports per-array coefficient
    ranges captured at stream time during :meth:`Problem._solve_streaming`.

    For the toy dispatch LP the values are hand-verifiable: matrix
    coefficients sit in [0.4, 1.0], objective costs in [50, 1000],
    and row RHS in [20, 200].  No column bounds are finite (all vars
    have ``[0, +inf)``) so the col_bound entry is ``None``.
    """
    pb = Problem()
    build_dispatch(pb, make_toy_data())

    sol = pb.solve()
    assert sol.optimal
    assert abs(sol.obj - 6500.0) < 1e-6

    ranges = sol.streamed_lp_ranges
    assert isinstance(ranges, dict)
    assert set(ranges) == {"matrix", "cost", "col_bound", "row_bound"}

    mn, mx = ranges["matrix"]
    assert 0.0 < mn <= mx < float("inf")
    cmn, cmx = ranges["cost"]
    assert 0.0 < cmn <= cmx < float("inf")
    assert ranges["col_bound"] is None  # toy LP has +inf upper bounds
    rmn, rmx = ranges["row_bound"]
    assert 0.0 < rmn <= rmx < float("inf")
