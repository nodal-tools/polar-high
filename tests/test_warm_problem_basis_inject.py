"""WarmProblem warm-start basis injection (Phase 4 Step 4a, spec §4.1).

The in-process default-path arm of the basis hook lives in
``WarmProblem._initial_build`` (fired once, on the fresh build, before the
first ``h.run()``).  When a carrier has been recorded via
``WarmProblem.set_named_basis`` it fingerprints THIS model's name-set,
builds a sized ``HighsBasis`` from the carrier via ``build_highs_basis``,
and injects it with ``h.setBasis`` — falling back to a cold solve on any
fingerprint mismatch (under ``exact``), rejected ``setBasis`` status, or
exception.

Acceptance criterion (spec §9.1): on a dimension-identical re-solve the
warm solve reaches the SAME optimal status + objective with a MEASURED
simplex-iteration reduction (NOT basis byte-parity).  We read the count off
the retained HiGHS handle (a WarmProblem always retains it):
``sol.highs.getInfo().simplex_iteration_count``.

The byte-identical-when-unused guarantee is paramount here: this is the ONE
Benders-adjacent change, and the Benders master builds a WarmProblem and
never calls ``set_named_basis``.  When ``_warm_basis`` stays ``None`` the
whole inject block is skipped after one ``is not None`` check.

The LP is a small generation-dispatch network sized so a cold solve takes a
non-trivial (>0) number of simplex iterations — the same builder the
streaming test uses, so the toy-model presolve-to-0 pitfall is avoided.
"""

from __future__ import annotations

import numpy as np
import polars as pl

import polar_high as fp
from polar_high import NamedBasis

# Silence HiGHS' per-solve banner; the option does not affect iteration
# counts.
_SOLVE_OPTS = {"output_flag": False}


def _build_network(n_t: int, n_gen: int, seed: int = 0) -> fp.Problem:
    """A generation-dispatch network LP (same shape as the streaming test).

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
# Test 1 — capture -> inject round-trip (exact): measured iteration
# reduction through WarmProblem.
# ---------------------------------------------------------------------------
def test_warm_problem_exact_capture_inject_reduces_iterations():
    # Source: a WarmProblem solve (retains a live handle) we capture from.
    p_src = _build_network(200, 20)
    wp_src = fp.WarmProblem(p_src)
    sol_src = wp_src.solve(options=_SOLVE_OPTS)
    assert sol_src.optimal
    nb = sol_src.get_named_basis()

    # Cold reference: an untouched WarmProblem for the SAME model.
    p_cold = _build_network(200, 20)
    wp_cold = fp.WarmProblem(p_cold)
    sol_cold = wp_cold.solve(options=_SOLVE_OPTS)
    assert sol_cold.optimal
    obj_cold = sol_cold.obj
    iters_cold = _iters(sol_cold)
    assert iters_cold > 0, f"cold solve did nothing measurable ({iters_cold} iters)"

    # Warm: a FRESH WarmProblem for the SAME model, basis injected.
    p_warm = _build_network(200, 20)
    wp_warm = fp.WarmProblem(p_warm)
    wp_warm.set_named_basis(nb, policy="exact")
    sol_warm = wp_warm.solve(options=_SOLVE_OPTS)
    iters_warm = _iters(sol_warm)

    assert sol_warm.optimal
    assert _obj_close(sol_warm.obj, obj_cold)
    # exact injection lands the optimal basis 1:1 -> strictly fewer iters.
    assert iters_warm < iters_cold, (
        f"warm ({iters_warm}) not fewer than cold ({iters_cold}) iterations"
    )
    # The fingerprint the hook computed for the fresh build was stashed and
    # matched the carrier (so the injection actually fired, not a fallback).
    assert wp_warm._last_basis_fingerprint == nb.fingerprint
    print(f"[exact] iters_cold={iters_cold} iters_warm={iters_warm}")


# ---------------------------------------------------------------------------
# Test 2 — alien policy tolerates mild drift (a dropped column name).
# ---------------------------------------------------------------------------
def test_warm_problem_alien_tolerates_mild_drift():
    p_ref = _build_network(200, 20)
    ref = fp.WarmProblem(p_ref).solve(options=_SOLVE_OPTS)
    obj_cold = ref.obj
    nb = ref.get_named_basis()

    # Drop one variable's name from the carrier (mild drift).
    drifted_cols = dict(nb.col_status)
    dropped = next(iter(drifted_cols))
    del drifted_cols[dropped]
    nb_drift = NamedBasis(
        col_status=drifted_cols,
        row_status=nb.row_status,
        fingerprint=nb.fingerprint,
    )

    p_warm = _build_network(200, 20)
    wp_warm = fp.WarmProblem(p_warm)
    wp_warm.set_named_basis(nb_drift, policy="alien")
    sol_warm = wp_warm.solve(options=_SOLVE_OPTS)

    assert sol_warm.optimal
    assert _obj_close(sol_warm.obj, obj_cold)


# ---------------------------------------------------------------------------
# Test 3 — byte-identical when unused: an untouched WarmProblem solves
# correctly and no basis state was ever set.
# ---------------------------------------------------------------------------
def test_warm_problem_byte_identical_when_unused():
    # Two identical WarmProblems, neither touched with set_named_basis.
    p_a = _build_network(200, 20)
    wp_a = fp.WarmProblem(p_a)
    sol_a = wp_a.solve(options=_SOLVE_OPTS)

    p_b = _build_network(200, 20)
    wp_b = fp.WarmProblem(p_b)
    sol_b = wp_b.solve(options=_SOLVE_OPTS)

    assert sol_a.optimal and sol_b.optimal
    # Same objective and same iteration count -> the unused path is
    # behaviorally identical (the guarantee is structural: the inject block
    # is skipped after one ``is not None`` check).
    assert _obj_close(sol_a.obj, sol_b.obj)
    assert _iters(sol_a) == _iters(sol_b)
    # No accidental basis state was set on either.
    assert wp_a._warm_basis is None
    assert wp_a._warm_basis_policy is None
    assert wp_a._last_basis_fingerprint is None
    assert wp_b._warm_basis is None
    assert wp_b._last_basis_fingerprint is None


# ---------------------------------------------------------------------------
# Test 4 — safety: fingerprint mismatch under 'exact' falls back to cold.
# ---------------------------------------------------------------------------
def test_warm_problem_exact_fingerprint_mismatch_falls_back_cold():
    p_ref = _build_network(200, 20)
    sol_ref = fp.WarmProblem(p_ref).solve(options=_SOLVE_OPTS)
    obj_ref = sol_ref.obj
    nb = sol_ref.get_named_basis()

    # Valid statuses, deliberately wrong fingerprint.
    nb_badfp = NamedBasis(
        col_status=nb.col_status,
        row_status=nb.row_status,
        fingerprint="0" * 16,
    )
    p_warm = _build_network(200, 20)
    wp_warm = fp.WarmProblem(p_warm)
    wp_warm.set_named_basis(nb_badfp, policy="exact")
    sol_warm = wp_warm.solve(options=_SOLVE_OPTS)

    assert sol_warm.optimal
    assert _obj_close(sol_warm.obj, obj_ref)
    # The hook still stashed THIS model's fingerprint (for the orchestrator's
    # post-solve capture key) even though it declined to inject.
    assert wp_warm._last_basis_fingerprint is not None
    assert wp_warm._last_basis_fingerprint != nb_badfp.fingerprint


# ---------------------------------------------------------------------------
# Test 5 — basis_name_fingerprint accessor.
# ---------------------------------------------------------------------------
def test_basis_name_fingerprint_deterministic_and_matches_capture():
    # Deterministic across two identical builds (the orchestrator's key).
    fp1 = _build_network(200, 20).basis_name_fingerprint()
    fp2 = _build_network(200, 20).basis_name_fingerprint()
    assert fp1 is not None
    assert fp1 == fp2

    # Equals the fingerprint of a basis captured off the SAME model, so the
    # orchestrator's lookup key matches the capture key.
    p_src = _build_network(200, 20)
    sol_src = fp.WarmProblem(p_src).solve(options=_SOLVE_OPTS)
    nb = sol_src.get_named_basis()
    p_key = _build_network(200, 20)
    assert p_key.basis_name_fingerprint() == nb.fingerprint


def test_basis_name_fingerprint_none_on_empty_and_released():
    # Empty problem (no columns) -> None.
    assert fp.Problem().basis_name_fingerprint() is None

    # Released problem -> None.
    p = _build_network(50, 5)
    p._release_python_lp_inputs()
    assert p.basis_name_fingerprint() is None


# ---------------------------------------------------------------------------
# Test 6 — reuse untouched: inject fires only on the fresh build, never on a
# subsequent reuse re-run.
# ---------------------------------------------------------------------------
def test_warm_problem_inject_only_on_fresh_build_not_reuse():
    p_src = _build_network(200, 20)
    nb = fp.WarmProblem(p_src).solve(options=_SOLVE_OPTS).get_named_basis()

    p_warm = _build_network(200, 20)
    wp = fp.WarmProblem(p_warm)
    wp.set_named_basis(nb, policy="exact")

    # First solve: builds + injects.  Capture the live handle identity.
    sol_first = wp.solve(options=_SOLVE_OPTS)
    assert sol_first.optimal
    h_after_first = wp._h
    assert h_after_first is not None

    # Reuse re-run: change an RHS and solve again.  ``_initial_build`` runs
    # only while ``self._h is None`` (it is not now), so no rebuild and no
    # re-inject can occur.  Proof: the live handle object is unchanged
    # (``_initial_build`` would have replaced it with a fresh Highs).
    wp.update_rhs("balance", 100.0)
    sol_reuse = wp.solve(options=_SOLVE_OPTS)
    assert sol_reuse.optimal
    assert wp._h is h_after_first, "reuse re-ran _initial_build (handle replaced)"
