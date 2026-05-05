"""Streaming-vs-passModel parity for ``Problem.solve``.

The streaming branch (``streaming=True``) emits constraint families to
HiGHS one ``addRows`` call at a time instead of building a single
``HighsLp`` and calling ``passModel``.  It must produce numerically
identical primal, duals, and objective on every existing model.
"""

import numpy as np
from flex_toy_data import make_flex_toy_data
from flex_toy_model import build_flex_toy
from toy_data import make_toy_data
from toy_model import build_dispatch

from polar_high import Problem


def _solve_both(builder, data_factory):
    p_default = Problem()
    builder(p_default, data_factory())
    p_stream = Problem()
    builder(p_stream, data_factory())
    return p_default.solve(streaming=False), p_stream.solve(streaming=True)


def test_streaming_parity_toy_dispatch():
    """5-family LP (max_flow, wind_available, node_balance, co2_cap,
    period_total): the streaming path must agree on objective, primal,
    row duals, column duals, and row/column names."""
    sol_default, sol_stream = _solve_both(build_dispatch, make_toy_data)

    assert sol_default.optimal and sol_stream.optimal
    assert abs(sol_default.obj - sol_stream.obj) < 1e-9
    assert np.allclose(sol_default.col_value, sol_stream.col_value, atol=1e-9)
    assert np.allclose(sol_default.row_dual, sol_stream.row_dual, atol=1e-9)
    assert np.allclose(sol_default.col_dual, sol_stream.col_dual, atol=1e-9)
    assert sol_default.row_names == sol_stream.row_names
    assert sol_default.col_names == sol_stream.col_names


def test_streaming_parity_flex_toy():
    """3-family flex model: same parity invariant — exercises a
    different rhs/sense mix and confirms HiGHS row indexing stays
    monotonic across ``addRows`` calls."""
    sol_default, sol_stream = _solve_both(build_flex_toy, make_flex_toy_data)

    assert sol_default.optimal and sol_stream.optimal
    assert abs(sol_default.obj - sol_stream.obj) < 1e-9
    assert np.allclose(sol_default.col_value, sol_stream.col_value, atol=1e-9)
    assert np.allclose(sol_default.row_dual, sol_stream.row_dual, atol=1e-9)
    assert np.allclose(sol_default.col_dual, sol_stream.col_dual, atol=1e-9)
    assert sol_default.row_names == sol_stream.row_names
    assert sol_default.col_names == sol_stream.col_names
