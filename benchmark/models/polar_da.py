"""polar-high dense LP with declared ``dense_axes=("j",)``.

Same build as :mod:`models.polar`; only the ``Problem`` construction
differs. Run alongside ``polar`` to show the block-COO / dense-axis
arm's effect on the same hardware in the same harness.

The dense LP has Vars indexed by ``("i", "j")``; declaring
``dense_axes=("j",)`` promises every frame using ``j`` is row-sorted by
``(other_dims..., j)``. The build below already produces frames in that
order (``np.repeat`` for ``i`` × ``np.tile`` for ``j``), so the contract
is satisfied without any extra sort.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from polar_high import Param, Problem, Sum


def build(N: int) -> Problem:
    p = Problem(dense_axes=("j",))

    i_arr = np.repeat(np.arange(1, N + 1, dtype=np.int64), N)
    j_arr = np.tile(np.arange(1, N + 1, dtype=np.int64), N)
    idx = pl.DataFrame({"i": i_arr, "j": j_arr})

    x = p.add_var("x", dims=("i", "j"), index=idx, lower=0.0)
    y = p.add_var("y", dims=("i", "j"), index=idx, lower=0.0)

    obj = x.to_expr() * 2.0 + y.to_expr() * 1.0
    p.set_objective(Sum(obj), sense="min")

    rhs_i = Param(
        ("i",),
        pl.DataFrame(
            {
                "i": np.arange(1, N + 1, dtype=np.int64),
                "value": np.arange(1, N + 1, dtype=np.float64),
            }
        ),
    )
    p.add_cstr(
        "c1",
        over=idx,
        sense=">=",
        lhs_terms={"x": x, "neg_y": -y.to_expr()},
        rhs_terms={"i": rhs_i},
    )
    p.add_cstr(
        "c2",
        over=idx,
        sense=">=",
        lhs_terms={"x": x, "y": y},
        rhs_terms={"zero": 0.0},
    )
    return p


def solve(model: Problem, time_limit: float | None = None) -> tuple[bool, float]:
    options = {"time_limit": float(time_limit)} if time_limit is not None else None
    sol = model.solve(options=options)
    return bool(sol.optimal), float(sol.obj)
