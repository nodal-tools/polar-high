"""Parity tests for the block-COO LHS evaluation arm (dense_axes contract).

Background
----------
Block-COO is a third dispatch arm in
:meth:`polar_high.engine.Problem._build_canonical_matrix` (sibling to the
polars LHS prune-down) for non-Sum ``Var × Param-chain`` terms over the
client-DECLARED dense axes.  Instead of carrying value products through
wide polars joins, it key-aligns each factor by sort + key-join and does
the final coefficient multiply in numpy on contiguous value buffers, in
the SAME left-to-right order as the polars chain rebuild (``coef_scalar``
seed, then ``*value`` / ``/value`` per atomic by direction).  Same
IEEE-double ops in the same order ⇒ the emitted coefficients are
**bit-identical** to the polars path.

Dense-axis contract
-------------------
The client declares the dense trailing axes once on the Problem
(``Problem(dense_axes=("d", "t"))``) and PROMISES that every frame it
passes carrying those columns is globally lexicographically sorted by
``(other_dims_in_declared_order..., *dense_axes)``.  Block-COO fires only
when the Var's dims END WITH the declared dense axes (the suffix
contract), verifies the sort promise cheaply, and RAISES a clear
``ValueError`` if the client breaks it.

Block-COO defaults OFF (opt-in via ``POLAR_HIGH_ENABLE_BLOCK_COO=1``;
see ``specs/block_coo_DECISIONS.md`` D3/D4).  These tests pin bit-identity
by canonicalising each builder twice: once with block-COO OFF (no opt-in
→ polars prune-down / fallback), once with ``POLAR_HIGH_ENABLE_BLOCK_COO=1``
(block-COO ON) — and the canonical matrix triples must be EQUAL with NO
tolerance.

All builders construct frames in ``itertools.product(lead..., d, t)``
order, which is exactly the row order the dense_axes contract requires
(lead dims as a sorted prefix, dense axes as trailing sort keys).
"""

from __future__ import annotations

import itertools
import os

import numpy as np
import polars as pl
import pytest

from polar_high.engine import Param, Problem, Where

# --------------------------------------------------------------------- #
# Helpers (mirror tests/test_where_pushdown_parity.py)                  #
# --------------------------------------------------------------------- #


def _clear_guard() -> None:
    os.environ.pop("POLAR_HIGH_ENABLE_BLOCK_COO", None)
    os.environ.pop("POLAR_HIGH_DISABLE_BLOCK_COO", None)
    # Keep the other prune/pushdown levers at their defaults so the
    # comparison is block-COO-on vs block-COO-off, all else equal.
    os.environ.pop("POLAR_HIGH_DISABLE_PRUNE_DOWN", None)
    os.environ.pop("POLAR_HIGH_DISABLE_WHERE_PUSHDOWN", None)
    os.environ.pop("POLAR_HIGH_BLOCK_COO_MIN_DENSE", None)


def _set_enable() -> None:
    """Opt block-COO ON (it defaults OFF — D3)."""
    os.environ["POLAR_HIGH_ENABLE_BLOCK_COO"] = "1"


def _matrix_arrays(m) -> tuple[list, list, list]:
    """Sorted (col, row, val) triple comparison key + row_lb / row_ub."""
    val = np.asarray(m.val, dtype=np.float64)
    row_idx = np.asarray(m.row_idx, dtype=np.int64)
    col_ptr = np.asarray(m.col_ptr, dtype=np.int64)
    cols = np.repeat(
        np.arange(m.n_cols, dtype=np.int64), np.diff(col_ptr).astype(np.int64)
    )
    order = np.lexsort((row_idx, cols))
    return (
        list(val[order]),
        list(np.asarray(m.row_lb, dtype=np.float64)),
        list(np.asarray(m.row_ub, dtype=np.float64)),
    )


def _build_and_snapshot(builder) -> tuple[list, list, list]:
    prob = builder()
    m = prob._build_canonical_matrix()
    return _matrix_arrays(m)


def _assert_parity(builder) -> None:
    """Assert byte-for-byte canonical equality between block-COO-on and
    block-COO-off runs of the same builder.  OFF = no opt-in (default,
    polars prune-down / fallback); ON = ``POLAR_HIGH_ENABLE_BLOCK_COO=1``."""
    _clear_guard()
    try:
        snap_off = _build_and_snapshot(builder)
    finally:
        _clear_guard()
    try:
        _set_enable()
        snap_on = _build_and_snapshot(builder)
    finally:
        _clear_guard()
    assert snap_on == snap_off, (
        "block-COO parity failure:\n"
        f"  block-COO ON  val={snap_on[0][:8]} ...\n"
        f"  block-COO OFF val={snap_off[0][:8]} ...\n"
        f"  n(on)={len(snap_on[0])} n(off)={len(snap_off[0])}"
    )


def _block_coo_fires(builder, *, enable: bool = True) -> bool:
    """Detect whether the block-COO arm fires for the builder's single
    LHS family/term by reading the PROFILE stream.

    ``enable`` opts block-COO ON (it defaults OFF — D3).  Pass
    ``enable=False`` to assert the default-off behaviour (no opt-in →
    must NOT fire)."""
    import io
    import sys as _sys

    _clear_guard()
    if enable:
        _set_enable()
    os.environ["POLAR_HIGH_BLOCK_COO_PROFILE"] = "1"
    buf = io.StringIO()
    old = _sys.stderr
    try:
        _sys.stderr = buf
        prob = builder()
        prob._build_canonical_matrix()
    finally:
        _sys.stderr = old
        os.environ.pop("POLAR_HIGH_BLOCK_COO_PROFILE", None)
        _clear_guard()
    return "phase=block_coo_term" in buf.getvalue()


def _block_coo_paths(builder) -> list[str]:
    """Return the ordered list of ``path=`` profile signals emitted by the
    block-COO builder for ``builder`` (one per fired block-COO term).

    The builder emits ``[block_coo profile]\\tpath=positional`` when the
    positional per-block slice-multiply fires and ``path=joined`` when it
    falls back to the order-preserving join backstop — gated by
    ``POLAR_HIGH_BLOCK_COO_PROFILE=1`` (same lever as the
    ``phase=block_coo_term`` line).  Mirrors :func:`_block_coo_fires`'
    profile-sniffing."""
    import io
    import re
    import sys as _sys

    _clear_guard()
    _set_enable()
    os.environ["POLAR_HIGH_BLOCK_COO_PROFILE"] = "1"
    buf = io.StringIO()
    old = _sys.stderr
    try:
        _sys.stderr = buf
        prob = builder()
        prob._build_canonical_matrix()
    finally:
        _sys.stderr = old
        os.environ.pop("POLAR_HIGH_BLOCK_COO_PROFILE", None)
        _clear_guard()
    return re.findall(r"\bpath=(\w+)", buf.getvalue())


# --------------------------------------------------------------------- #
# Frame builders — all rows in itertools.product order = sorted by      #
# (lead..., d, t), satisfying the dense_axes contract.                  #
# --------------------------------------------------------------------- #


def _vdt_over(ps, ds, ts) -> pl.DataFrame:
    """Var(p, d, t) ``over`` frame, sorted by (p, d, t)."""
    cells = list(itertools.product(ps, ds, ts))
    return pl.DataFrame(
        {
            "p": [c[0] for c in cells],
            "d": [c[1] for c in cells],
            "t": [c[2] for c in cells],
        }
    )


def _dt_param(ds, ts, name, lo, hi, *, keep=None) -> Param:
    """Param(d, t) over the full (or filtered) (d, t) grid, sorted by
    (d, t).  ``keep`` is an optional predicate on (d, t) for sparsity."""
    cells = list(itertools.product(ds, ts))
    if keep is not None:
        cells = [c for c in cells if keep(c)]
    return Param(
        ("d", "t"),
        pl.DataFrame(
            {
                "d": [c[0] for c in cells],
                "t": [c[1] for c in cells],
                "value": np.linspace(lo, hi, len(cells)),
            }
        ),
        name=name,
    )


def _build_vpp_problem(
    *,
    n_p: int = 3,
    n_d: int = 5,
    n_t: int = 40,
    with_where: bool = False,
    min_dense: int | None = None,
):
    """Builder for a Problem with a non-Sum LHS

        Var(p, d, t) × Param_a(d, t) × Param_b(d, t)

    over the declared dense ``(d, t)`` suffix.  ``with_where`` adds a
    pure-filter ``Where`` on ``t``.  ``min_dense`` overrides the firing
    threshold.
    """

    def builder() -> Problem:
        if min_dense is not None:
            os.environ["POLAR_HIGH_BLOCK_COO_MIN_DENSE"] = str(min_dense)
        p = Problem(dense_axes=("d", "t"))
        ps, ds, ts = list(range(n_p)), list(range(n_d)), list(range(n_t))
        over = _vdt_over(ps, ds, ts)
        v = p.add_var("v", ("p", "d", "t"), over, lower=0.0, upper=1e6)
        Pa = _dt_param(ds, ts, "Pa", 0.1, 0.9)
        Pb = _dt_param(ds, ts, "Pb", 2.0, 5.0)
        chain = v * Pa * Pb
        if with_where:
            sel_t = ts[: max(1, n_t // 2)]
            chain = Where(chain, pl.DataFrame({"t": sel_t}))
        p.add_cstr(
            "c",
            over=over,
            sense="<=",
            lhs_terms={"lhs": chain},
            rhs_terms={"rhs": 0.0},
        )
        return p

    return builder


# --------------------------------------------------------------------- #
# Tests                                                                 #
# --------------------------------------------------------------------- #


def test_block_coo_fires_on_declared_suffix():
    """``Var(p,d,t) × Pa(d,t) × Pb(d,t)`` with ``dense_axes=("d","t")``:
    the Var's dims end with the declared dense suffix, so block-COO fires;
    ON vs OFF must be bit-identical."""
    builder = _build_vpp_problem(n_p=3, n_d=5, n_t=40)
    assert _block_coo_fires(builder), "block-COO should fire on the declared suffix"
    _assert_parity(builder)


def test_block_coo_with_where_filter():
    """Same chain wrapped in a pure-filter ``Where(chain, t-frame)`` —
    block-COO must honour where_frames; parity ON vs OFF must hold."""
    builder = _build_vpp_problem(n_p=3, n_d=5, n_t=40, with_where=True)
    assert _block_coo_fires(builder), (
        "block-COO should fire on the pure-filter Where shape"
    )
    _assert_parity(builder)


def test_block_coo_below_threshold_falls_back():
    """When the dense-card upper bound is BELOW the perf threshold the
    block-COO arm must NOT fire (fall back to polars); parity holds
    trivially."""
    # Var height = 2*2*5 = 20 < 100 → classifier returns None.
    builder = _build_vpp_problem(n_p=2, n_d=2, n_t=5)
    assert not _block_coo_fires(builder), (
        "block-COO must NOT fire below the dense-card threshold"
    )
    _assert_parity(builder)


def test_block_coo_single_param_chain():
    """Chain length 1 (``Var(p,d,t) × Pa(d,t)``) — block-COO accepts
    single-Param chains (unlike the prune-down's >=2 gate); parity must
    hold vs the disabled path."""

    def builder() -> Problem:
        n_p, n_d, n_t = 3, 5, 40
        p = Problem(dense_axes=("d", "t"))
        ps, ds, ts = list(range(n_p)), list(range(n_d)), list(range(n_t))
        over = _vdt_over(ps, ds, ts)
        v = p.add_var("v", ("p", "d", "t"), over, lower=0.0, upper=1e6)
        Pa = _dt_param(ds, ts, "Pa", 0.1, 0.9)
        p.add_cstr(
            "c",
            over=over,
            sense="<=",
            lhs_terms={"lhs": v * Pa},
            rhs_terms={"rhs": 0.0},
        )
        return p

    assert _block_coo_fires(builder)
    _assert_parity(builder)


def test_block_coo_param_introduces_on_dim_falls_back():
    """``Var(p, d) × Pa(d, t)`` over a ``(p, d, t)`` constraint axis: the
    Param introduces ``t`` into ``on`` but ``t ∉ var.dims``, so the
    block-COO seed (var dims only) cannot supply the ``t`` join key.  The
    classifier must fall back; parity with the polars path must hold."""

    def builder() -> Problem:
        n_p, n_d, n_t = 4, 5, 30
        p = Problem(dense_axes=("d", "t"))
        ps, ds, ts = list(range(n_p)), list(range(n_d)), list(range(n_t))
        over = _vdt_over(ps, ds, ts)
        # Var spans (p, d) only — NOT t.  Sorted by (p, d).
        pd_cells = list(itertools.product(ps, ds))
        v_over = pl.DataFrame(
            {"p": [c[0] for c in pd_cells], "d": [c[1] for c in pd_cells]}
        )
        v = p.add_var("v", ("p", "d"), v_over, lower=0.0, upper=1e6)
        Pa = _dt_param(ds, ts, "Pa", 0.1, 0.9)
        p.add_cstr(
            "c",
            over=over,
            sense="<=",
            lhs_terms={"lhs": v * Pa},
            rhs_terms={"rhs": 0.0},
        )
        return p

    assert not _block_coo_fires(builder), (
        "block-COO must fall back when a join key comes from a Param"
    )
    _assert_parity(builder)


def test_block_coo_scalar_fold_and_division():
    """Param-side scalar fold + a denominator Param exercise the
    ``coef_scalar`` seed and the ``/value`` direction branch; block-COO
    must reproduce both bit-identically."""

    def builder() -> Problem:
        n_p, n_d, n_t = 3, 5, 40
        p = Problem(dense_axes=("d", "t"))
        ps, ds, ts = list(range(n_p)), list(range(n_d)), list(range(n_t))
        over = _vdt_over(ps, ds, ts)
        v = p.add_var("v", ("p", "d", "t"), over, lower=0.0, upper=1e6)
        Pa = _dt_param(ds, ts, "Pa", 0.5, 1.5)
        Pb = _dt_param(ds, ts, "Pb", 2.0, 4.0)
        # Scalar fold on Pa (×60) and a division by Pb.
        chain = (v * (Pa * 60.0)) / Pb
        p.add_cstr(
            "c",
            over=over,
            sense="<=",
            lhs_terms={"lhs": chain},
            rhs_terms={"rhs": 0.0},
        )
        return p

    assert _block_coo_fires(builder)
    _assert_parity(builder)


def test_block_coo_default_off_does_not_fire():
    """With NO opt-in (``POLAR_HIGH_ENABLE_BLOCK_COO`` unset), block-COO
    must NOT fire even on a shape the classifier would otherwise accept."""
    builder = _build_vpp_problem(n_p=3, n_d=5, n_t=40)
    assert _block_coo_fires(builder, enable=True), (
        "shape must be block-COO-eligible when opted in"
    )
    assert not _block_coo_fires(builder, enable=False), (
        "block-COO must NOT fire without POLAR_HIGH_ENABLE_BLOCK_COO=1"
    )


def test_block_coo_low_dim_param_now_fires():
    """``Var(p,d,t) × Pa(d,t) × Pb(p)`` — the broadcast shape.  Under the
    DECLARED-suffix contract (``dense_axes=("d","t")``) the dense set is
    the declared suffix, NOT the factor intersection, so the low-dim /
    broadcast ``Pb(p)`` no longer empties it: block-COO NOW fires (the
    join-based builder broadcasts ``Pb`` correctly).  This proves the
    contract enables the broadcast case.  ON vs OFF must be
    bit-identical."""

    def builder() -> Problem:
        n_p, n_d, n_t = 3, 5, 40
        p = Problem(dense_axes=("d", "t"))
        ps, ds, ts = list(range(n_p)), list(range(n_d)), list(range(n_t))
        over = _vdt_over(ps, ds, ts)
        v = p.add_var("v", ("p", "d", "t"), over, lower=0.0, upper=1e6)
        Pa = _dt_param(ds, ts, "Pa", 0.1, 0.9)
        # Pb is genuinely low-dim: spans only p.
        Pb = Param(
            ("p",),
            pl.DataFrame({"p": ps, "value": np.linspace(2.0, 5.0, n_p)}),
            name="Pb",
        )
        chain = v * Pa * Pb
        p.add_cstr(
            "c",
            over=over,
            sense="<=",
            lhs_terms={"lhs": chain},
            rhs_terms={"rhs": 0.0},
        )
        return p

    assert _block_coo_fires(builder), (
        "block-COO must fire on the broadcast shape under the dense_axes "
        "contract"
    )
    _assert_parity(builder)


def test_block_coo_no_dense_axes_declared_does_not_fire():
    """Same broadcast shape as above but ``Problem()`` with NO dense_axes
    declared → block-COO must NOT fire (the contract is the only firing
    trigger now); parity holds trivially."""

    def builder() -> Problem:
        n_p, n_d, n_t = 3, 5, 40
        p = Problem()  # no dense_axes
        ps, ds, ts = list(range(n_p)), list(range(n_d)), list(range(n_t))
        over = _vdt_over(ps, ds, ts)
        v = p.add_var("v", ("p", "d", "t"), over, lower=0.0, upper=1e6)
        Pa = _dt_param(ds, ts, "Pa", 0.1, 0.9)
        Pb = Param(
            ("p",),
            pl.DataFrame({"p": ps, "value": np.linspace(2.0, 5.0, n_p)}),
            name="Pb",
        )
        chain = v * Pa * Pb
        p.add_cstr(
            "c",
            over=over,
            sense="<=",
            lhs_terms={"lhs": chain},
            rhs_terms={"rhs": 0.0},
        )
        return p

    assert not _block_coo_fires(builder), (
        "block-COO must NOT fire when no dense_axes are declared"
    )
    _assert_parity(builder)


def test_block_coo_var_without_dense_suffix_falls_back():
    """An investment-like ``Var(p, d) × Pa(p, d)`` with
    ``dense_axes=("d","t")``: ``var.dims = ("p","d")`` does NOT end in
    ``("d","t")``, so the suffix contract fails and block-COO falls back
    (correctly — the Var lacks the dense axes).  Parity holds."""

    def builder() -> Problem:
        n_p, n_d = 30, 20
        p = Problem(dense_axes=("d", "t"))
        ps, ds = list(range(n_p)), list(range(n_d))
        pd_cells = list(itertools.product(ps, ds))
        over = pl.DataFrame(
            {"p": [c[0] for c in pd_cells], "d": [c[1] for c in pd_cells]}
        )
        v = p.add_var("v", ("p", "d"), over, lower=0.0, upper=1e6)
        Pa = Param(
            ("p", "d"),
            pl.DataFrame(
                {
                    "p": [c[0] for c in pd_cells],
                    "d": [c[1] for c in pd_cells],
                    "value": np.linspace(0.1, 0.9, len(pd_cells)),
                }
            ),
            name="Pa",
        )
        p.add_cstr(
            "c",
            over=over,
            sense="<=",
            lhs_terms={"lhs": v * Pa},
            rhs_terms={"rhs": 0.0},
        )
        return p

    assert not _block_coo_fires(builder), (
        "block-COO must fall back when the Var does not end in the "
        "declared dense suffix"
    )
    _assert_parity(builder)


def test_block_coo_verification_raises_on_misordered_input():
    """A Var ``over`` frame deliberately NOT sorted by (p, d, t) violates
    the dense_axes sort contract.  With block-COO enabled,
    ``_build_canonical_matrix`` must RAISE ``ValueError`` naming the
    contract — polar-high's own ``add_var`` does NOT sort, so the unsorted
    frame stays unsorted and trips the verifier."""
    n_p, n_d, n_t = 3, 5, 40
    ps, ds, ts = list(range(n_p)), list(range(n_d)), list(range(n_t))
    over = _vdt_over(ps, ds, ts)
    # Shuffle rows deterministically so it is NOT sorted by (p, d, t).
    rng = np.random.default_rng(12345)
    perm = rng.permutation(over.height)
    over_shuffled = over[perm.tolist()]
    assert not (
        over_shuffled.select(pl.struct(["p", "d", "t"]).alias("k"))
        .to_series()
        .is_sorted()
    ), "shuffled over frame must NOT be sorted (test precondition)"

    def builder() -> Problem:
        p = Problem(dense_axes=("d", "t"))
        v = p.add_var("v", ("p", "d", "t"), over_shuffled, lower=0.0, upper=1e6)
        Pa = _dt_param(ds, ts, "Pa", 0.1, 0.9)
        Pb = _dt_param(ds, ts, "Pb", 2.0, 5.0)
        p.add_cstr(
            "c",
            over=over,  # constraint axis can be sorted; the Var seed is not
            sense="<=",
            lhs_terms={"lhs": v * Pa * Pb},
            rhs_terms={"rhs": 0.0},
        )
        return p

    _clear_guard()
    try:
        _set_enable()
        prob = builder()
        with pytest.raises(ValueError, match="dense_axes contract violated"):
            prob._build_canonical_matrix()
    finally:
        _clear_guard()


def test_block_coo_sparse_param_no_crash_and_parity():
    """A SPARSE Param (coefficients defined on only a subset of its shared
    keys) must NOT crash block-COO and must stay bit-identical to the
    polars prune-down (whose inner join silently DROPS the unmatched
    cells)."""

    def builder() -> Problem:
        n_p, n_d, n_t = 3, 5, 40
        p = Problem(dense_axes=("d", "t"))
        ps, ds, ts = list(range(n_p)), list(range(n_d)), list(range(n_t))
        over = _vdt_over(ps, ds, ts)
        v = p.add_var("v", ("p", "d", "t"), over, lower=0.0, upper=1e6)
        dt_cells = list(itertools.product(ds, ts))
        # Pa: keep only cells where (d + t) is even → sparse on (d,t).
        Pa = _dt_param(ds, ts, "Pa", 0.1, 0.9, keep=lambda c: (c[0] + c[1]) % 2 == 0)
        assert 0 < Pa.frame.height < len(dt_cells), "Pa must be sparse"
        # Pb: full (d,t) axis but sparse on a DIFFERENT subset.
        Pb = _dt_param(
            ds, ts, "Pb", 2.0, 5.0, keep=lambda c: not (c[0] == 0 and c[1] < 7)
        )
        assert 0 < Pb.frame.height < len(dt_cells), "Pb must be sparse"
        chain = v * Pa * Pb
        p.add_cstr(
            "c",
            over=over,
            sense="<=",
            lhs_terms={"lhs": chain},
            rhs_terms={"rhs": 0.0},
        )
        return p

    assert _block_coo_fires(builder), (
        "block-COO should fire on the sparse dense-axis chain"
    )
    _assert_parity(builder)


# --------------------------------------------------------------------- #
# Positional path: all three Param cases (lead-only, dense-only,         #
# lead-subset+dense) over a dense-complete grid.                          #
# --------------------------------------------------------------------- #


def _build_three_case_problem(*, with_where_t: bool = False):
    """Builder for a non-Sum LHS exercising ALL THREE positional Param
    alignment cases at once:

        Var(p, d, t)
          × unit_size(p)          # lead-only  (shared = [p] ⊆ non_dense)
          × step_duration(d, t)   # dense-only (shared = [d, t] == dense)
          × efficiency(p, d, t)   # lead-subset + dense (shared = [p,d,t])

    over the declared dense ``(d, t)`` suffix.  All frames are built in
    ``itertools.product`` order (lead prefix, dense trailing) so the
    dense_axes sort contract holds.  With ``with_where_t`` a pure-filter
    ``Where`` on ``t`` makes the grid sparse → forces the joined fallback.
    """
    n_p, n_d, n_t = 4, 6, 50
    ps, ds, ts = list(range(n_p)), list(range(n_d)), list(range(n_t))

    def builder() -> Problem:
        p = Problem(dense_axes=("d", "t"))
        over = _vdt_over(ps, ds, ts)
        v = p.add_var("v", ("p", "d", "t"), over, lower=0.0, upper=1e6)

        # lead-only Param(p)
        unit_size = Param(
            ("p",),
            pl.DataFrame({"p": ps, "value": np.linspace(10.0, 40.0, n_p)}),
            name="unit_size",
        )
        # dense-only Param(d, t)
        step_duration = _dt_param(ds, ts, "step_duration", 0.25, 3.0)
        # lead-subset + dense Param(p, d, t)
        pdt_cells = list(itertools.product(ps, ds, ts))
        efficiency = Param(
            ("p", "d", "t"),
            pl.DataFrame(
                {
                    "p": [c[0] for c in pdt_cells],
                    "d": [c[1] for c in pdt_cells],
                    "t": [c[2] for c in pdt_cells],
                    "value": np.linspace(0.3, 0.95, len(pdt_cells)),
                }
            ),
            name="efficiency",
        )
        chain = v * unit_size * step_duration * efficiency
        if with_where_t:
            sel_t = ts[: max(1, n_t // 2)]
            chain = Where(chain, pl.DataFrame({"t": sel_t}))
        p.add_cstr(
            "c",
            over=over,
            sense="<=",
            lhs_terms={"lhs": chain},
            rhs_terms={"rhs": 0.0},
        )
        return p

    return builder


def test_block_coo_positional_dense_complete_parity():
    """``Var(p,d,t) × unit_size(p) × step_duration(d,t) × efficiency(p,d,t)``
    over a dense-complete grid (no Where): all three positional Param cases
    (lead-only, dense-only, lead-subset+dense) fire on the POSITIONAL path,
    and ON vs OFF must be bit-identical."""
    builder = _build_three_case_problem(with_where_t=False)
    assert _block_coo_fires(builder), "three-case chain should fire block-COO"
    # Positional path must be taken on the dense-complete grid.
    assert _block_coo_paths(builder) == ["positional"], (
        "dense-complete grid must take the positional slice-multiply path"
    )
    _assert_parity(builder)


def test_block_coo_falls_back_when_sparse():
    """Same three-case chain with a pure-filter ``Where`` on ``t`` carves
    the grid sparse → the completeness guard forces the JOINED fallback;
    bit-parity ON vs OFF must still hold."""
    builder = _build_three_case_problem(with_where_t=True)
    assert _block_coo_fires(builder), "sparse three-case chain should still fire"
    # where_frames present ⇒ completeness guard falls back to joined.
    assert _block_coo_paths(builder) == ["joined"], (
        "a pure-filter Where must force the joined fallback path"
    )
    _assert_parity(builder)
