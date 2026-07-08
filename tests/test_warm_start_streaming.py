"""End-to-end acceptance test for the streaming warm-start injection hook
(Phase 1 Part 2, spec §9.1).

The hook lives in ``Problem._solve_streaming`` just before ``h.run()``:
when a carrier has been recorded via ``Problem.set_named_basis`` on the
in-process (``save_memory=False``) path, it fingerprints THIS model's
name-set, builds a sized ``HighsBasis`` from the carrier via
``build_highs_basis``, and injects it with ``h.setBasis`` — falling back
to a cold solve on any fingerprint mismatch, rejected ``setBasis``, or
exception.

Acceptance criterion (spec §9.1): on a dimension-identical re-solve, the
warm solve reaches the SAME optimal status + objective with a MEASURED
simplex-iteration reduction (NOT basis byte-parity).  We read the count
off the retained HiGHS handle: ``sol.highs.getInfo().simplex_iteration_count``
(attribute confirmed present on highspy 1.14).

The LP is a small generation-dispatch network sized so a cold solve takes
a non-trivial (>0) number of simplex iterations — the 3-variable toy in
``test_named_basis.py`` presolves to 0 iterations and would make the
reduction unmeasurable.
"""

from __future__ import annotations

import numpy as np
import polars as pl

import polar_high as fp
from polar_high import NamedBasis

# Silence HiGHS' per-solve banner so the test output stays clean; the
# option does not affect simplex iteration counts (verified: cold=166 both
# with and without it on the n_t=200/n_gen=20 model).
_SOLVE_OPTS = {"output_flag": False}


def _build_network(n_t: int, n_gen: int, seed: int = 0) -> fp.Problem:
    """A generation-dispatch network LP.

    ``n_gen`` generators serve a per-timestep demand over ``n_t``
    timesteps; each generator has a capacity ceiling and a linear cost, a
    high-penalty slack absorbs any shortfall.  Every constraint carries a
    stable rendered name (``max_flow[g,t]`` / ``balance[t]``), so an
    ``exact`` fingerprint round-trips across a rebuild.
    """
    rng = np.random.default_rng(seed)
    p = fp.Problem()
    t_idx = pl.DataFrame({"t": np.arange(n_t, dtype=np.int64)})
    g_idx = pl.DataFrame({"g": np.arange(n_gen, dtype=np.int64)})
    gt = g_idx.join(t_idx, how="cross")

    v_flow = p.add_var("v_flow", ("g", "t"), gt, lower=0.0)
    vq = p.add_var("vq", ("t",), t_idx, lower=0.0)

    cap = fp.Param(
        ("g",),
        pl.DataFrame({"g": np.arange(n_gen), "value": rng.uniform(20.0, 100.0, n_gen)}),
    )
    demand = fp.Param(
        ("t",),
        pl.DataFrame({"t": np.arange(n_t), "value": rng.uniform(50.0, 200.0, n_t)}),
    )
    cost = fp.Param(
        ("g",),
        pl.DataFrame({"g": np.arange(n_gen), "value": rng.uniform(1.0, 50.0, n_gen)}),
    )

    p.add_cstr(
        "max_flow",
        over=gt,
        sense="<=",
        lhs_terms={"flow": v_flow},
        rhs_terms={"cap": cap},
    )
    p.add_cstr(
        "balance",
        over=t_idx,
        sense="==",
        lhs_terms={"gen": fp.Sum(v_flow, over="g"), "slack": vq},
        rhs_terms={"demand": demand},
    )
    p.set_objective(fp.Sum(v_flow * cost) + fp.Sum(vq * 1000.0), sense="min")
    return p


def _iters(sol: fp.Solution) -> int:
    """Simplex iteration count off the retained HiGHS handle."""
    return int(sol.highs.getInfo().simplex_iteration_count)


def _obj_close(a: float, b: float) -> bool:
    return abs(a - b) < 1e-6 * (1.0 + abs(b))


# ---------------------------------------------------------------------------
# Test 1 — exact policy: fingerprint match + measured iteration reduction.
# ---------------------------------------------------------------------------
def test_warm_exact_reduces_iterations():
    # Cold solve of the model, retaining the solver so we can read the
    # iteration count and extract the optimal basis.
    p_cold = _build_network(200, 20)
    sol_cold = p_cold.solve(keep_solver=True, options=_SOLVE_OPTS)
    assert sol_cold.optimal
    obj_cold = sol_cold.obj
    iters_cold = _iters(sol_cold)
    # The model must be big enough that cold takes real simplex work, or
    # the reduction is unmeasurable (spec §9.1).
    assert iters_cold > 0, f"cold solve did nothing measurable ({iters_cold} iters)"

    nb = sol_cold.get_named_basis()

    # Rebuild the SAME model fresh and inject the basis under 'exact'.
    p_warm = _build_network(200, 20)
    p_warm.set_named_basis(nb, policy="exact")
    sol_warm = p_warm.solve(keep_solver=True, options=_SOLVE_OPTS)
    iters_warm = _iters(sol_warm)

    assert sol_warm.optimal
    assert _obj_close(sol_warm.obj, obj_cold)
    # exact injection lands the optimal basis 1:1 → strictly fewer iters.
    # (Measured: cold=166, warm=0 on highspy 1.14.)
    assert iters_warm < iters_cold


# ---------------------------------------------------------------------------
# Test 2 — alien policy: same optimum with reduction, tolerant of drift.
# ---------------------------------------------------------------------------
def test_warm_alien_reduces_iterations():
    p_cold = _build_network(200, 20)
    sol_cold = p_cold.solve(keep_solver=True, options=_SOLVE_OPTS)
    assert sol_cold.optimal
    obj_cold = sol_cold.obj
    iters_cold = _iters(sol_cold)
    assert iters_cold > 0

    nb = sol_cold.get_named_basis()

    p_warm = _build_network(200, 20)
    p_warm.set_named_basis(nb, policy="alien")
    sol_warm = p_warm.solve(keep_solver=True, options=_SOLVE_OPTS)
    iters_warm = _iters(sol_warm)

    assert sol_warm.optimal
    assert _obj_close(sol_warm.obj, obj_cold)
    # alien may repair, so assert non-strict; on this model it also lands 0.
    assert iters_warm <= iters_cold


def test_warm_alien_tolerates_mild_drift():
    """Drop one variable's name from the carrier before injecting (mild
    drift): 'alien' defaults the unmatched column and lets HiGHS repair;
    the solve must still reach the correct optimum."""
    p_cold = _build_network(200, 20)
    sol_cold = p_cold.solve(keep_solver=True, options=_SOLVE_OPTS)
    obj_cold = sol_cold.obj
    nb = sol_cold.get_named_basis()

    drifted_cols = dict(nb.col_status)
    dropped = next(iter(drifted_cols))
    del drifted_cols[dropped]
    nb_drift = NamedBasis(
        col_status=drifted_cols,
        row_status=nb.row_status,
        fingerprint=nb.fingerprint,
    )

    p_warm = _build_network(200, 20)
    p_warm.set_named_basis(nb_drift, policy="alien")
    sol_warm = p_warm.solve(keep_solver=True, options=_SOLVE_OPTS)

    assert sol_warm.optimal
    assert _obj_close(sol_warm.obj, obj_cold)


# ---------------------------------------------------------------------------
# Test 3 — safety: a warm-start NEVER breaks a solve.
# ---------------------------------------------------------------------------
def test_exact_fingerprint_mismatch_falls_back_cold():
    """A carrier whose fingerprint does not match THIS model under 'exact'
    must be ignored (cache-miss → cold), still reaching the cold optimum."""
    p_ref = _build_network(200, 20)
    sol_ref = p_ref.solve(options=_SOLVE_OPTS)
    obj_ref = sol_ref.obj

    p_cold = _build_network(200, 20)
    sol_cold = p_cold.solve(keep_solver=True, options=_SOLVE_OPTS)
    nb = sol_cold.get_named_basis()

    # Valid statuses, but a deliberately wrong fingerprint.
    nb_badfp = NamedBasis(
        col_status=nb.col_status,
        row_status=nb.row_status,
        fingerprint="0" * 16,
    )
    p_warm = _build_network(200, 20)
    p_warm.set_named_basis(nb_badfp, policy="exact")
    sol_warm = p_warm.solve(options=_SOLVE_OPTS)

    assert sol_warm.optimal
    assert _obj_close(sol_warm.obj, obj_ref)


def test_alien_garbage_statuses_still_reaches_optimum():
    """A carrier with garbage statuses (all columns basic, all rows at
    lower) under 'alien' either gets repaired by HiGHS or the setBasis is
    rejected — either way the solve must reach the correct optimum."""
    p_ref = _build_network(200, 20)
    sol_ref = p_ref.solve(options=_SOLVE_OPTS)
    obj_ref = sol_ref.obj

    p_cold = _build_network(200, 20)
    sol_cold = p_cold.solve(keep_solver=True, options=_SOLVE_OPTS)
    nb = sol_cold.get_named_basis()

    # Over-basic garbage: every column basic (an illegal over-count of
    # basic variables) and every row nonbasic at lower.
    garbage = NamedBasis(
        col_status={name: 1 for name in nb.col_status},  # kBasic
        row_status={name: 0 for name in nb.row_status},  # kLower
        fingerprint=nb.fingerprint,
    )
    p_warm = _build_network(200, 20)
    p_warm.set_named_basis(garbage, policy="alien")
    sol_warm = p_warm.solve(options=_SOLVE_OPTS)

    assert sol_warm.optimal
    assert _obj_close(sol_warm.obj, obj_ref)


def test_no_warm_basis_is_untouched_cold_path():
    """Sanity: with no carrier recorded the guard short-circuits and the
    plain cold solve is unchanged."""
    p = _build_network(200, 20)
    assert p._warm_basis is None
    sol = p.solve(keep_solver=True, options=_SOLVE_OPTS)
    assert sol.optimal
    assert _iters(sol) > 0
