"""Parity tests for the block-COO LHS evaluation arm (Phase A/B/D/F).

Background
----------
Block-COO is a third dispatch arm in
:meth:`polar_high.engine.Problem._build_canonical_matrix` (sibling to the
polars LHS prune-down) for non-Sum ``Var × Param-chain`` terms over a
dense axis.  Instead of carrying value products through wide polars joins,
it key-aligns each factor by sort + key-join and does the final
coefficient multiply in numpy on contiguous value buffers, in the SAME
left-to-right order as the polars chain rebuild (``coef_scalar`` seed,
then ``*value`` / ``/value`` per atomic by direction).  Same IEEE-double
ops in the same order ⇒ the emitted coefficients are **bit-identical** to
the polars path.

Block-COO defaults OFF (opt-in via ``POLAR_HIGH_ENABLE_BLOCK_COO=1``;
see ``specs/block_coo_DECISIONS.md`` D3/D4).  These tests pin bit-identity
by canonicalising each builder twice: once with block-COO OFF (no opt-in
→ polars prune-down / fallback), once with ``POLAR_HIGH_ENABLE_BLOCK_COO=1``
(block-COO ON) — and the canonical matrix triples must be EQUAL with NO
tolerance.  Same harness as :mod:`tests.test_where_pushdown_parity`.
"""

from __future__ import annotations

import itertools
import os

import numpy as np
import polars as pl

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


# --------------------------------------------------------------------- #
# Synthetic builders — non-Sum Var(p,d,t) × Param_a(d,t) × Param_b(d,t) #
# --------------------------------------------------------------------- #


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

    over a dense ``(d, t)`` axis present in EVERY factor (so the
    classifier's dense set ``D = var ∩ Pa ∩ Pb = {d, t}`` is non-empty).
    The dense-card upper bound used by the classifier is the Var height
    ``n_p * n_d * n_t``.

    ``with_where`` adds a pure-filter ``Where`` on ``t`` (no new dim →
    where_frames path).  ``min_dense`` overrides the firing threshold.
    """

    def builder() -> Problem:
        if min_dense is not None:
            os.environ["POLAR_HIGH_BLOCK_COO_MIN_DENSE"] = str(min_dense)
        p = Problem()
        ps = list(range(n_p))
        ds = list(range(n_d))
        ts = list(range(n_t))
        cells = list(itertools.product(ps, ds, ts))
        over = pl.DataFrame(
            {
                "p": [c[0] for c in cells],
                "d": [c[1] for c in cells],
                "t": [c[2] for c in cells],
            }
        )
        v = p.add_var("v", ("p", "d", "t"), over, lower=0.0, upper=1e6)

        dt_cells = list(itertools.product(ds, ts))
        Pa = Param(
            ("d", "t"),
            pl.DataFrame(
                {
                    "d": [c[0] for c in dt_cells],
                    "t": [c[1] for c in dt_cells],
                    "value": np.linspace(0.1, 0.9, len(dt_cells)),
                }
            ),
            name="Pa",
        )
        Pb = Param(
            ("d", "t"),
            pl.DataFrame(
                {
                    "d": [c[0] for c in dt_cells],
                    "t": [c[1] for c in dt_cells],
                    "value": np.linspace(2.0, 5.0, len(dt_cells)),
                }
            ),
            name="Pb",
        )
        chain = v * Pa * Pb

        if with_where:
            sel_t = ts[: max(1, n_t // 2)]
            f = pl.DataFrame({"t": sel_t})
            chain = Where(chain, f)

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


def test_block_coo_matches_polars():
    """Dense ``Var(p,d,t) × Pa(d,t) × Pb(d,t)`` — both Params carry the
    full ``(d,t)`` axis, so the dense set ``D = var ∩ Pa ∩ Pb = {d,t}`` is
    non-empty and block-COO fires.  ON vs OFF must produce a bit-identical
    canonical matrix.  Dense cardinality (Var height = 3*5*40 = 600) is
    well above the 100 threshold."""
    builder = _build_vpp_problem(n_p=3, n_d=5, n_t=40)
    assert _block_coo_fires(builder), "block-COO should fire on this shape"
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
    """When the dense axis is BELOW the threshold the block-COO arm must
    NOT fire (fall back to polars); parity must still hold trivially."""
    # Var height = 2*2*5 = 20 < 100 → classifier returns None.
    builder = _build_vpp_problem(n_p=2, n_d=2, n_t=5)
    assert not _block_coo_fires(builder), (
        "block-COO must NOT fire below the dense-card threshold"
    )
    _assert_parity(builder)


def test_block_coo_single_param_chain():
    """Chain length 1 (``Var(p,d,t) × Pa(d,t)``) — block-COO accepts
    single-Param chains (unlike the prune-down's >=2 gate); parity must
    hold vs the disabled path (which uses the fallback semi-join)."""

    def builder() -> Problem:
        n_p, n_d, n_t = 3, 5, 40
        p = Problem()
        ps, ds, ts = list(range(n_p)), list(range(n_d)), list(range(n_t))
        cells = list(itertools.product(ps, ds, ts))
        over = pl.DataFrame(
            {
                "p": [c[0] for c in cells],
                "d": [c[1] for c in cells],
                "t": [c[2] for c in cells],
            }
        )
        v = p.add_var("v", ("p", "d", "t"), over, lower=0.0, upper=1e6)
        dt_cells = list(itertools.product(ds, ts))
        Pa = Param(
            ("d", "t"),
            pl.DataFrame(
                {
                    "d": [c[0] for c in dt_cells],
                    "t": [c[1] for c in dt_cells],
                    "value": np.linspace(0.1, 0.9, len(dt_cells)),
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

    assert _block_coo_fires(builder)
    _assert_parity(builder)


def test_block_coo_param_introduces_on_dim_falls_back():
    """``Var(p, d) × Pa(d, t)`` over a ``(p, d, t)`` constraint axis: the
    Param introduces ``t`` into ``on`` but ``t ∉ var.dims``, so the
    block-COO seed (var dims only) cannot supply the ``t`` join key.  The
    classifier must fall back; parity with the polars path must hold."""

    def builder() -> Problem:
        n_p, n_d, n_t = 4, 5, 30
        p = Problem()
        ps, ds, ts = list(range(n_p)), list(range(n_d)), list(range(n_t))
        cells = list(itertools.product(ps, ds, ts))
        over = pl.DataFrame(
            {
                "p": [c[0] for c in cells],
                "d": [c[1] for c in cells],
                "t": [c[2] for c in cells],
            }
        )
        # Var spans (p, d) only — NOT t.
        pd_cells = list(itertools.product(ps, ds))
        v_over = pl.DataFrame(
            {"p": [c[0] for c in pd_cells], "d": [c[1] for c in pd_cells]}
        )
        v = p.add_var("v", ("p", "d"), v_over, lower=0.0, upper=1e6)
        dt_cells = list(itertools.product(ds, ts))
        Pa = Param(
            ("d", "t"),
            pl.DataFrame(
                {
                    "d": [c[0] for c in dt_cells],
                    "t": [c[1] for c in dt_cells],
                    "value": np.linspace(0.1, 0.9, len(dt_cells)),
                }
            ),
            name="Pa",
        )
        # term dims = (p, d, t); on = (p, d, t); t not in var dims.
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
        p = Problem()
        ps, ds, ts = list(range(n_p)), list(range(n_d)), list(range(n_t))
        cells = list(itertools.product(ps, ds, ts))
        over = pl.DataFrame(
            {
                "p": [c[0] for c in cells],
                "d": [c[1] for c in cells],
                "t": [c[2] for c in cells],
            }
        )
        v = p.add_var("v", ("p", "d", "t"), over, lower=0.0, upper=1e6)
        dt_cells = list(itertools.product(ds, ts))
        Pa = Param(
            ("d", "t"),
            pl.DataFrame(
                {
                    "d": [c[0] for c in dt_cells],
                    "t": [c[1] for c in dt_cells],
                    "value": np.linspace(0.5, 1.5, len(dt_cells)),
                }
            ),
            name="Pa",
        )
        Pb = Param(
            ("d", "t"),
            pl.DataFrame(
                {
                    "d": [c[0] for c in dt_cells],
                    "t": [c[1] for c in dt_cells],
                    "value": np.linspace(2.0, 4.0, len(dt_cells)),
                }
            ),
            name="Pb",
        )
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
    must NOT fire even on a shape the classifier would otherwise accept —
    it defaults OFF (D3) so it cannot silently affect real solves."""
    builder = _build_vpp_problem(n_p=3, n_d=5, n_t=40)
    # Sanity: with opt-in this exact shape DOES fire (guards against the
    # test passing because the shape is simply non-firing).
    assert _block_coo_fires(builder, enable=True), (
        "shape must be block-COO-eligible when opted in"
    )
    # Default: no opt-in → must not fire.
    assert not _block_coo_fires(builder, enable=False), (
        "block-COO must NOT fire without POLAR_HIGH_ENABLE_BLOCK_COO=1"
    )


def test_block_coo_low_dim_param_does_not_fire():
    """``Var(p,d,t) × Pa(d,t) × Pb(p)`` — a genuinely low-dim Param
    ``Pb(p)`` makes the dense set ``D = var ∩ Pa ∩ Pb = {p} ∩ {d,t}`` ...
    actually ``{d,t} ∩ {p} = ∅``, so the dense set is empty and block-COO
    does NOT fire (the classifier falls back).  This documents honestly
    that broadcast/low-dim shapes are out of scope for the current
    non-Sum arm; generalizing them is deferred to the Phase C reshape
    (D4).  Parity with the polars path must still hold."""

    def builder() -> Problem:
        n_p, n_d, n_t = 3, 5, 40
        p = Problem()
        ps, ds, ts = list(range(n_p)), list(range(n_d)), list(range(n_t))
        cells = list(itertools.product(ps, ds, ts))
        over = pl.DataFrame(
            {
                "p": [c[0] for c in cells],
                "d": [c[1] for c in cells],
                "t": [c[2] for c in cells],
            }
        )
        v = p.add_var("v", ("p", "d", "t"), over, lower=0.0, upper=1e6)
        dt_cells = list(itertools.product(ds, ts))
        Pa = Param(
            ("d", "t"),
            pl.DataFrame(
                {
                    "d": [c[0] for c in dt_cells],
                    "t": [c[1] for c in dt_cells],
                    "value": np.linspace(0.1, 0.9, len(dt_cells)),
                }
            ),
            name="Pa",
        )
        # Pb is genuinely low-dim: spans only p.
        Pb = Param(
            ("p",),
            pl.DataFrame(
                {"p": ps, "value": np.linspace(2.0, 5.0, n_p)}
            ),
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
        "block-COO must NOT fire when a low-dim Param empties the dense set"
    )
    _assert_parity(builder)


def test_block_coo_sparse_param_no_crash_and_parity():
    """A SPARSE Param (coefficients defined on only a subset of its shared
    keys) must NOT crash block-COO and must stay bit-identical to the
    polars prune-down (whose inner join silently DROPS the unmatched
    cells).  Two sparsity sources are combined:

    * ``Pa(d,t)`` missing roughly half its ``(d,t)`` cells, and
    * a broadcast-shaped ``Pb(d,t)`` (full ``(d,t)`` axis so it remains in
      the dense set / fires) that is ALSO sparse on a different cell set.

    The surviving LP rows are the intersection of both Params' key sets,
    exactly as the polars inner-join chain produces — block-COO drops the
    same rows in numpy, never raising on the missing coefficients."""

    def builder() -> Problem:
        n_p, n_d, n_t = 3, 5, 40
        p = Problem()
        ps, ds, ts = list(range(n_p)), list(range(n_d)), list(range(n_t))
        cells = list(itertools.product(ps, ds, ts))
        over = pl.DataFrame(
            {
                "p": [c[0] for c in cells],
                "d": [c[1] for c in cells],
                "t": [c[2] for c in cells],
            }
        )
        v = p.add_var("v", ("p", "d", "t"), over, lower=0.0, upper=1e6)
        dt_cells = list(itertools.product(ds, ts))
        # Pa: keep only cells where (d + t) is even → sparse on (d,t).
        pa_cells = [c for c in dt_cells if (c[0] + c[1]) % 2 == 0]
        assert 0 < len(pa_cells) < len(dt_cells), "Pa must be sparse"
        Pa = Param(
            ("d", "t"),
            pl.DataFrame(
                {
                    "d": [c[0] for c in pa_cells],
                    "t": [c[1] for c in pa_cells],
                    "value": np.linspace(0.1, 0.9, len(pa_cells)),
                }
            ),
            name="Pa",
        )
        # Pb: full (d,t) axis (so it stays in the dense set) but sparse on
        # a DIFFERENT subset (drop the first 7 t-steps for d==0).
        pb_cells = [c for c in dt_cells if not (c[0] == 0 and c[1] < 7)]
        assert 0 < len(pb_cells) < len(dt_cells), "Pb must be sparse"
        Pb = Param(
            ("d", "t"),
            pl.DataFrame(
                {
                    "d": [c[0] for c in pb_cells],
                    "t": [c[1] for c in pb_cells],
                    "value": np.linspace(2.0, 5.0, len(pb_cells)),
                }
            ),
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

    # Must fire (dense set {d,t} non-empty, Var height 600 >= 100) and
    # must NOT crash on the sparse Params.
    assert _block_coo_fires(builder), (
        "block-COO should fire on the sparse dense-axis chain"
    )
    _assert_parity(builder)
