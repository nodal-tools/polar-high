"""Parity test for the Param-chain prune-down path in
``Problem._build_canonical_matrix``.

The RHS of a constraint can be a composite Param built from a chain of
``Param.__mul__`` / ``Param.__truediv__`` operations.  When the
composite's ``_sources`` list tracks the constituent atomic Params
(length >= 2), ``_build_canonical_matrix`` walks the chain one atomic
at a time and semi-joins each atomic to the running row-index key
projection (the "prune-down" path) — avoiding the wide
Cartesian-on-shared-dim intermediates that the merged-lazy plan would
otherwise materialise.

This test pins numerical parity between the prune-down path and the
fallback merged-lazy path so the optimisation is byte-for-byte safe:

  1. Build a synthetic Problem whose single constraint has a 3-atomic
     Param chain on RHS, with disjoint-but-shared dims keyed
     ``(f, d, t)`` / ``(p, d)`` / ``(p, d, t)`` (the same dim shape as
     DES's ``profile_flow_upper_limit`` cliff).
  2. Call ``canonicalise()`` — exercises the prune-down path because
     ``_sources`` has length 3.
  3. Forcibly clear ``_sources`` on the RHS Param so the next
     canonicalise falls back to the merged-lazy semi-join path.
  4. Mark the matrix dirty and re-canonicalise.
  5. Assert ``row_lb`` / ``row_ub`` (where the RHS vector lives) are
     identical between the two paths.

Single-Param RHS and anonymous chains (``_sources is None``) are
already exercised by the existing ``test_streaming_parity`` /
``test_warm_problem`` / ``test_problem_write_mps`` suites and are
unchanged by this fix.
"""

from __future__ import annotations

import itertools

import numpy as np
import polars as pl

import polar_high as fp


def _build_chain_problem() -> fp.Problem:
    """Build a small Problem with a 3-atomic Param chain on RHS.

    Dims: process ``p`` (3), flow ``f`` (2), node ``d`` (4), time ``t`` (5).
    Constraint is over ``(p, f, d, t)`` so the row index has up to
    120 rows; the atomic Params use a strict subset of those keys each.
    """
    p = fp.Problem()

    n_p, n_f, n_d, n_t = 3, 2, 4, 5
    p_idx = [f"p{i}" for i in range(n_p)]
    f_idx = [f"f{i}" for i in range(n_f)]
    d_idx = [f"d{i}" for i in range(n_d)]
    t_idx = list(range(n_t))

    # Full over frame for the constraint.
    over_rows = list(itertools.product(p_idx, f_idx, d_idx, t_idx))
    over = pl.DataFrame(
        {
            "p": [r[0] for r in over_rows],
            "f": [r[1] for r in over_rows],
            "d": [r[2] for r in over_rows],
            "t": [r[3] for r in over_rows],
        }
    )

    # LHS variable — a placeholder ``v(p, f, d, t)`` so the constraint
    # is non-degenerate.  Bounds and coefficient are irrelevant: we only
    # check row_lb / row_ub.
    v = p.add_var("v", ("p", "f", "d", "t"), over, lower=0.0, upper=1.0e6)

    # Atomic Param 1: profile_value(f, d, t).
    fdt_rows = list(itertools.product(f_idx, d_idx, t_idx))
    profile_value = fp.Param(
        ("f", "d", "t"),
        pl.DataFrame(
            {
                "f": [r[0] for r in fdt_rows],
                "d": [r[1] for r in fdt_rows],
                "t": [r[2] for r in fdt_rows],
                "value": np.linspace(0.1, 0.9, len(fdt_rows)).astype(np.float64),
            }
        ),
        name="profile_value",
    )

    # Atomic Param 2: existing_count(p, d).
    pd_rows = list(itertools.product(p_idx, d_idx))
    existing_count = fp.Param(
        ("p", "d"),
        pl.DataFrame(
            {
                "p": [r[0] for r in pd_rows],
                "d": [r[1] for r in pd_rows],
                "value": np.linspace(2.0, 5.0, len(pd_rows)).astype(np.float64),
            }
        ),
        name="existing_count",
    )

    # Atomic Param 3: availability(p, d, t).
    pdt_rows = list(itertools.product(p_idx, d_idx, t_idx))
    availability = fp.Param(
        ("p", "d", "t"),
        pl.DataFrame(
            {
                "p": [r[0] for r in pdt_rows],
                "d": [r[1] for r in pdt_rows],
                "t": [r[2] for r in pdt_rows],
                "value": np.linspace(0.5, 1.0, len(pdt_rows)).astype(np.float64),
            }
        ),
        name="availability",
    )

    # Compose the chain.  This goes through Param.__mul__ which records
    # _sources = [(profile_value, +1), (existing_count, +1), (availability, +1)].
    rhs_chain = profile_value * existing_count * availability

    # Sanity check on _sources — the test depends on the chain being
    # tracked.
    assert rhs_chain._sources is not None
    assert len(rhs_chain._sources) == 3

    p.add_cstr(
        "chain_upper",
        over=over,
        sense="<=",
        lhs_terms={"v": v},
        rhs_terms={"rhs_chain": rhs_chain},
    )

    return p, rhs_chain


def test_param_chain_prune_down_matches_merged_path():
    """Prune-down path numerically equals merged-lazy semi-join path."""
    prob, rhs_chain = _build_chain_problem()

    # First canonicalise — uses the prune-down path (sources length 3).
    m_pruned = prob.canonicalise()
    row_lb_pruned = m_pruned.row_lb.copy()
    row_ub_pruned = m_pruned.row_ub.copy()

    # Force the fallback merged-lazy path by clearing _sources on the
    # composite RHS Param, then re-canonicalise.
    rhs_chain._sources = None
    prob._matrix = None
    prob._canonical_dirty = True

    m_merged = prob.canonicalise()
    row_lb_merged = m_merged.row_lb
    row_ub_merged = m_merged.row_ub

    # rhs vector lives in row_ub for "<=" constraints; row_lb is -inf.
    assert row_lb_pruned.shape == row_lb_merged.shape
    assert row_ub_pruned.shape == row_ub_merged.shape

    # -inf rows for "<=" constraints — compare on finiteness.
    assert np.array_equal(np.isfinite(row_lb_pruned), np.isfinite(row_lb_merged))
    finite_lb = np.isfinite(row_lb_pruned)
    if finite_lb.any():
        np.testing.assert_array_equal(
            row_lb_pruned[finite_lb], row_lb_merged[finite_lb]
        )

    # row_ub is finite (the RHS vector) — must match byte-for-byte.
    np.testing.assert_array_equal(row_ub_pruned, row_ub_merged)


def test_param_chain_prune_down_division():
    """Same parity, but with a division in the chain.

    Exercises the ``direction = -1`` branch in the prune-down loop.
    """
    prob, _ = _build_chain_problem()

    # Replace the constraint with one that uses division.  Build a fresh
    # Problem rather than mutating the existing one.
    p = fp.Problem()

    n_p, n_f, n_d, n_t = 3, 2, 4, 5
    p_idx = [f"p{i}" for i in range(n_p)]
    f_idx = [f"f{i}" for i in range(n_f)]
    d_idx = [f"d{i}" for i in range(n_d)]
    t_idx = list(range(n_t))

    over_rows = list(itertools.product(p_idx, f_idx, d_idx, t_idx))
    over = pl.DataFrame(
        {
            "p": [r[0] for r in over_rows],
            "f": [r[1] for r in over_rows],
            "d": [r[2] for r in over_rows],
            "t": [r[3] for r in over_rows],
        }
    )
    v = p.add_var("v", ("p", "f", "d", "t"), over, lower=0.0, upper=1.0e6)

    fdt_rows = list(itertools.product(f_idx, d_idx, t_idx))
    profile_value = fp.Param(
        ("f", "d", "t"),
        pl.DataFrame(
            {
                "f": [r[0] for r in fdt_rows],
                "d": [r[1] for r in fdt_rows],
                "t": [r[2] for r in fdt_rows],
                "value": np.linspace(0.1, 0.9, len(fdt_rows)).astype(np.float64),
            }
        ),
        name="profile_value",
    )
    pd_rows = list(itertools.product(p_idx, d_idx))
    existing_count = fp.Param(
        ("p", "d"),
        pl.DataFrame(
            {
                "p": [r[0] for r in pd_rows],
                "d": [r[1] for r in pd_rows],
                "value": np.linspace(2.0, 5.0, len(pd_rows)).astype(np.float64),
            }
        ),
        name="existing_count",
    )
    pdt_rows = list(itertools.product(p_idx, d_idx, t_idx))
    availability = fp.Param(
        ("p", "d", "t"),
        pl.DataFrame(
            {
                "p": [r[0] for r in pdt_rows],
                "d": [r[1] for r in pdt_rows],
                "t": [r[2] for r in pdt_rows],
                # Make sure availability never crosses zero so division is
                # well-defined.
                "value": np.linspace(0.5, 1.0, len(pdt_rows)).astype(np.float64),
            }
        ),
        name="availability",
    )

    # Chain: profile_value * existing_count / availability — direction
    # = [+1, +1, -1].
    rhs_chain = profile_value * existing_count / availability
    assert rhs_chain._sources is not None
    assert len(rhs_chain._sources) == 3
    assert [d for _, d in rhs_chain._sources] == [1, 1, -1]

    p.add_cstr(
        "chain_div",
        over=over,
        sense="<=",
        lhs_terms={"v": v},
        rhs_terms={"rhs_chain": rhs_chain},
    )

    m_pruned = p.canonicalise()
    row_ub_pruned = m_pruned.row_ub.copy()

    rhs_chain._sources = None
    p._matrix = None
    p._canonical_dirty = True

    m_merged = p.canonicalise()
    row_ub_merged = m_merged.row_ub

    np.testing.assert_array_equal(row_ub_pruned, row_ub_merged)


def test_single_param_rhs_unchanged():
    """Single-Param RHS (no chain) must take the original merged-lazy
    fallback path verbatim.  Smoke-tests that the chain check
    ``len(_sources) >= 2`` correctly excludes the named-atomic case
    (where _sources == [(self, +1)] has length 1).
    """
    p = fp.Problem()
    n_t = 7
    t_idx = pl.DataFrame({"t": np.arange(n_t, dtype=np.int64)})
    v = p.add_var("v", "t", t_idx, lower=0.0, upper=1.0e6)
    rhs_atomic = fp.Param(
        ("t",),
        pl.DataFrame(
            {
                "t": np.arange(n_t, dtype=np.int64),
                "value": np.linspace(1.0, 7.0, n_t).astype(np.float64),
            }
        ),
        name="rhs_atomic",
    )
    p.add_cstr(
        "single",
        over=t_idx,
        sense="<=",
        lhs_terms={"v": v},
        rhs_terms={"r": rhs_atomic},
    )
    m = p.canonicalise()
    np.testing.assert_array_equal(
        m.row_ub, np.linspace(1.0, 7.0, n_t).astype(np.float64)
    )
