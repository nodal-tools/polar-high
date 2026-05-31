"""Pin the post-build RELEASE of the ``SumBlockMeta`` recipe.

A ``Sum``-reduced block-eligible term clears its own ``var_source`` and
*survivor-filters* ``param_sources`` (dropping the summed-out factors),
but the captured :class:`SumBlockMeta` recipe snapshots the FULL pre-Sum
chain — including the summed-out dense ``(d, t)`` Params.  That recipe is
an independent reference: it keeps every snapshotted dense Param (and its
eager source frame) alive for as long as the term object lives.

The matrix build (cold ``Problem.solve`` and warm
``WarmProblem._initial_build``) is the recipe's LAST reader — the
block-COO Sum builder consumes it, and every autoscale readout that runs
before the build consumes it too.  Once HiGHS owns the assembled LP the
recipe is dead weight.  Two release points drop it:

* cold save_memory path — :meth:`Problem._release_python_lp_inputs` nulls
  ``sum_block_meta`` on objective terms (constraint terms are dropped
  whole with ``proto.expr.terms = []``);
* warm path — the end of :meth:`WarmProblem._initial_build` nulls
  ``sum_block_meta`` on every objective + constraint term (WarmProblem
  never calls ``_release_python_lp_inputs``).

Without the release, the recipe ratchets dense Params up across rolling
solves until OOM.  These tests pin (a) that the autoscale bounded walk
itself never PINS a Param's eager ``_frame_cache`` (it reads dense Params
via the transient ``.lazy`` collect, so the cache must stay ``None``),
(b) that the cold save_memory release drops the recipe AND lets its
summed-out dense Param be garbage-collected, and (c) that the warm build
drops the recipe on every term.
"""

from __future__ import annotations

import gc
import itertools

import numpy as np
import polars as pl

import polar_high as fp
from polar_high.autoscale._coef_walk import (
    CoefWalkRecipe,
    MinMaxAbsReducer,
    bounded_coefficient_walk,
)
from polar_high.engine import Param, Problem, Sum


def _alive(target_id: int) -> bool:
    """True iff a Python object with ``id() == target_id`` is still
    reachable.  Used instead of ``weakref`` because :class:`Param` uses
    ``__slots__`` without ``__weakref__`` (not weak-referenceable)."""
    gc.collect()
    return any(id(o) == target_id for o in gc.get_objects())


def _relabel_sum(prob: Problem):
    """A block-eligible relabel ``Sum(v * P_unit * P_step, over=('p',))``
    over a Var whose dims end with the declared dense axes ``(d, t)``.

    ``P_unit`` is keyed on ``p`` only — ``over=('p',)`` sums that dim out,
    so the survivor-filtered term drops ``P_unit`` yet the recipe must
    carry it FULL.  ``P_step`` is keyed on ``(d, t)`` and survives.

    Returns ``(sum_expr, v, P_unit, P_step)``.
    """
    ps, ds, ts = [0, 1], [10, 11], [100, 101, 102]
    rows = list(itertools.product(ps, ds, ts))
    var_index = pl.DataFrame(
        {
            "p": [r[0] for r in rows],
            "d": [r[1] for r in rows],
            "t": [r[2] for r in rows],
        }
    ).sort("p", "d", "t")
    v = prob.add_var("v", ("p", "d", "t"), var_index, lower=0.0, upper=1.0e6)
    P_unit = Param(
        ("p",),
        pl.DataFrame({"p": ps, "value": [2.0, 3.0]}),
        name="P_unit",
    )
    dt = list(itertools.product(ds, ts))
    P_step = Param(
        ("d", "t"),
        pl.DataFrame(
            {
                "d": [r[0] for r in dt],
                "t": [r[1] for r in dt],
                "value": np.linspace(0.5, 1.5, len(dt)),
            }
        ),
        name="P_step",
    )
    sum_expr = Sum(v * P_unit * P_step, over=("p",))
    assert sum_expr.terms[0].sum_block_meta is not None
    return sum_expr, v, P_unit, P_step


def test_bounded_walk_does_not_pin_param_frame_cache():
    """The autoscale bounded walk over a recipe-carrying term reads dense
    Params via the transient ``.lazy`` collect, NEVER via the caching
    ``Param.frame`` property — so the Params' ``_frame_cache`` must stay
    ``None`` after the walk.  This pins that the traversal does not pin
    frames (the lever-1 guarantee)."""
    prob = Problem(dense_axes=("d", "t"))
    sum_expr, v, P_unit, P_step = _relabel_sum(prob)
    term = sum_expr.terms[0]

    # Pre-condition: no eager frame cached on either dense Param.
    assert P_unit._frame_cache is None
    assert P_step._frame_cache is None

    recipe = CoefWalkRecipe.from_term(term)
    # Column-mode spine: the Var grid carrying ``col_id`` (no ``_rid``).
    scale = (None, 0, None)
    (minmax,) = bounded_coefficient_walk(
        v.frame,
        recipe,
        scale,
        [MinMaxAbsReducer(scale)],
        batch_rows=256_000,
        dense_axes=("d", "t"),
    )
    lo, hi = minmax
    assert lo is not None and hi is not None and 0.0 < lo <= hi

    # The walk must not have populated the caching ``Param.frame`` property
    # on either dense Param — they were read through ``.lazy`` transients.
    assert P_unit._frame_cache is None, "walk pinned P_unit._frame_cache"
    assert P_step._frame_cache is None, "walk pinned P_step._frame_cache"


def _solvable_problem(dense_axes=("d", "t")):
    """A tiny but genuinely solvable LP whose single constraint family LHS
    carries a recipe-bearing relabel ``Sum`` term, plus an objective.

    Returns ``(prob, v, P_unit, P_step)``.
    """
    prob = Problem(dense_axes=dense_axes)
    sum_expr, v, P_unit, P_step = _relabel_sum(prob)

    # Constraint over the surviving (d, t) grid: Sum(...) >= 1.0 .  The Sum
    # term carries the recipe; the survivor-filtered term keeps only P_step.
    dt_rows = list(itertools.product([10, 11], [100, 101, 102]))
    over_dt = pl.DataFrame(
        {"d": [r[0] for r in dt_rows], "t": [r[1] for r in dt_rows]}
    ).sort("d", "t")
    prob.add_cstr(
        "relabel_cstr",
        over=over_dt,
        sense=">=",
        lhs_terms={"s": sum_expr},
        rhs_terms={"floor": 1.0},
    )

    # Objective: minimise the total flow (bare Var, keeps the LP bounded).
    prob.set_objective(Sum(v, over=("p", "d", "t")), sense="min")
    return prob, v, P_unit, P_step


def test_cold_save_memory_release_drops_recipe_and_summed_param():
    """``Problem.solve(save_memory=True)`` builds the LP (consuming the
    recipe), then ``_release_python_lp_inputs`` must drop the recipe so the
    summed-out dense Param (``P_unit``, referenced ONLY by the recipe after
    the build) becomes collectable."""
    prob, v, P_unit, P_step = _solvable_problem()
    id_unit = id(P_unit)

    # The constraint term carries the recipe; the recipe holds P_unit FULL.
    cstr_term = prob._cstrs[0][1].expr.terms[0]
    assert cstr_term.sum_block_meta is not None
    assert P_unit in [p for (p, _d) in cstr_term.sum_block_meta.param_sources]

    sol = prob.solve(save_memory=True, streaming=True)
    assert sol.optimal

    # After release the constraint family list is cleared (terms dropped
    # whole), so the recipe — and its sole remaining reference to P_unit —
    # is gone.  Drop the test's own handle and confirm P_unit is collectable.
    assert prob._cstrs == []
    del P_unit, cstr_term, v, P_step
    assert not _alive(id_unit), (
        "summed-out dense Param still reachable after cold save_memory "
        "release — the SumBlockMeta recipe was not dropped"
    )


def test_cold_save_memory_release_nulls_objective_recipe():
    """The objective-term release path nulls ``sum_block_meta`` explicitly
    (objective terms are not dropped via a list clear).  Pin that the
    objective term's recipe slot is ``None`` after release."""
    prob = Problem(dense_axes=("d", "t"))
    _sum_expr, v, P_unit, P_step = _relabel_sum(prob)
    # ``set_objective`` wraps its argument in ``Sum(expr, over=None)`` — so
    # passing the UN-reduced ``v * P_unit * P_step`` chain (not the already
    # block-eligible ``sum_expr``, whose nested re-Sum would drop the recipe)
    # makes that collapse capture a relabel recipe on the objective term.
    obj_chain = v * P_unit * P_step
    prob.set_objective(obj_chain, sense="min")
    # Bound the LP with a trivial constraint so it solves.
    dt_rows = list(itertools.product([10, 11], [100, 101, 102]))
    over_dt = pl.DataFrame(
        {"d": [r[0] for r in dt_rows], "t": [r[1] for r in dt_rows]}
    ).sort("d", "t")
    prob.add_cstr(
        "floor_cstr",
        over=over_dt,
        sense=">=",
        lhs_terms={"s2": Sum(v, over=("p",))},
        rhs_terms={"floor": 0.5},
    )

    obj_term = prob._obj_terms[0]
    assert obj_term.sum_block_meta is not None

    sol = prob.solve(save_memory=True, streaming=True)
    assert sol.optimal

    # _release_python_lp_inputs nulled the objective term's recipe slot.
    assert obj_term.sum_block_meta is None, (
        "objective term's SumBlockMeta recipe survived the cold "
        "save_memory release"
    )


def test_warm_initial_build_releases_recipe_on_all_terms():
    """``WarmProblem._initial_build`` is the warm path's last recipe reader
    and must null ``sum_block_meta`` on every objective + constraint term
    afterwards — WarmProblem never calls ``_release_python_lp_inputs``, so
    without this the recipe pins dense Params for the WarmProblem's whole
    lifetime (the across-rolls ratchet)."""
    prob, v, P_unit, P_step = _solvable_problem()

    cstr_term = prob._cstrs[0][1].expr.terms[0]
    assert cstr_term.sum_block_meta is not None
    id_unit = id(P_unit)

    wp = fp.WarmProblem(prob)
    sol = wp.solve()
    assert sol.optimal

    # Every term's recipe slot is cleared after the build.
    for t in prob._obj_terms:
        assert t.sum_block_meta is None
    for _name, proto, _over in prob._cstrs:
        for t in proto.expr.terms:
            assert t.sum_block_meta is None, (
                "warm build left a SumBlockMeta recipe on a constraint term"
            )

    # The warm-update machinery keys off ``param_sources`` / ``_param_cells``,
    # NOT the recipe, so the recipe drop is safe.  With the recipe gone and
    # P_unit summed OUT of the survivor-filtered term, P_unit is now reachable
    # only via the test's own handle; dropping it makes it collectable.
    del P_unit, cstr_term, v, P_step
    assert not _alive(id_unit), (
        "summed-out dense Param still reachable after warm _initial_build — "
        "the recipe was not dropped"
    )
