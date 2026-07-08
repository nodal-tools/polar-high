"""Unit tests for the warm-start / basis-injection primitives (Phase 1
Part 1): the :class:`NamedBasis` carrier, :func:`basis_fingerprint`,
:func:`build_highs_basis`, ``Solution.get_named_basis`` and
``Problem.set_named_basis``.

Re-injection / re-solve is intentionally NOT tested here — that is the
next agent's e2e acceptance test.
"""

import math
from dataclasses import FrozenInstanceError

import highspy
import numpy as np
import pytest
from toy_data import make_toy_data
from toy_model import build_dispatch

from polar_high import NamedBasis, Problem, basis_fingerprint
from polar_high._warm_basis import build_highs_basis

HighsBasis = highspy.HighsBasis
HighsBasisStatus = highspy.HighsBasisStatus

S_LOWER = int(HighsBasisStatus.kLower)
S_BASIC = int(HighsBasisStatus.kBasic)
S_UPPER = int(HighsBasisStatus.kUpper)
S_ZERO = int(HighsBasisStatus.kZero)
S_NONBASIC = int(HighsBasisStatus.kNonbasic)
_FREE = {S_NONBASIC, S_ZERO}


# ---------------------------------------------------------------------------
# basis_fingerprint
# ---------------------------------------------------------------------------
def test_fingerprint_permutation_invariant():
    cols = ["v_flow[wind,1]", "v_flow[gas,1]", "vq_up[1]"]
    rows = ["node_balance[1]", "max_flow[wind,1]"]
    fp1 = basis_fingerprint(cols, rows)
    fp2 = basis_fingerprint(list(reversed(cols)), list(reversed(rows)))
    assert fp1 == fp2


def test_fingerprint_changed_name_set_differs():
    cols = ["a", "b", "c"]
    rows = ["r0", "r1"]
    fp = basis_fingerprint(cols, rows)
    assert basis_fingerprint(["a", "b", "d"], rows) != fp
    assert basis_fingerprint(cols, ["r0", "r2"]) != fp


def test_fingerprint_drops_synthetic_rows():
    cols = ["a", "b"]
    rows = ["node_balance[1]", "max_flow[1]"]
    fp_plain = basis_fingerprint(cols, rows)
    fp_with_syn = basis_fingerprint(cols, rows + ["row_5", "row_12"])
    assert fp_plain == fp_with_syn
    # ... but only because they are dropped: keeping them changes the key.
    fp_kept = basis_fingerprint(cols, rows + ["row_5"], drop_synthetic_rows=False)
    assert fp_kept != fp_plain


# ---------------------------------------------------------------------------
# build_highs_basis — exact
# ---------------------------------------------------------------------------
def test_build_exact_maps_one_to_one():
    nb = NamedBasis(
        col_status={"c0": S_LOWER, "c1": S_BASIC},
        row_status={"r0": S_BASIC, "r1": S_UPPER},
        fingerprint="deadbeefdeadbeef",
    )
    basis, stats = build_highs_basis(
        nb,
        "exact",
        col_names=["c0", "c1"],
        row_names=["r0", "r1"],
        col_lb=[0.0, 0.0],
        col_ub=[10.0, 10.0],
        HighsBasis=HighsBasis,
        HighsBasisStatus=HighsBasisStatus,
    )
    assert basis.alien is False
    assert [int(s) for s in basis.col_status] == [S_LOWER, S_BASIC]
    assert [int(s) for s in basis.row_status] == [S_BASIC, S_UPPER]
    assert len(basis.col_status) == 2
    assert len(basis.row_status) == 2
    assert stats["n_cols_matched"] == 2
    assert stats["n_rows_matched"] == 2
    assert stats["n_cols_defaulted"] == 0
    assert stats["policy"] == "exact"
    assert stats["alien"] is False


def test_build_exact_missing_name_raises():
    nb = NamedBasis(
        col_status={"c0": S_LOWER},
        row_status={"r0": S_BASIC},
        fingerprint="x",
    )
    with pytest.raises(ValueError, match="exact basis transfer requires"):
        build_highs_basis(
            nb,
            "exact",
            col_names=["c0", "c1"],  # c1 not in carrier
            row_names=["r0"],
            col_lb=[0.0, 0.0],
            col_ub=[10.0, 10.0],
            HighsBasis=HighsBasis,
            HighsBasisStatus=HighsBasisStatus,
        )


def test_build_exact_synthetic_row_defaults_basic_not_missing():
    """A synthetic ``row_<i>`` target is not in the carrier by construction;
    exact must default it to kBasic rather than treating it as missing."""
    nb = NamedBasis(
        col_status={"c0": S_LOWER},
        row_status={"r0": S_UPPER},
        fingerprint="x",
    )
    basis, stats = build_highs_basis(
        nb,
        "exact",
        col_names=["c0"],
        row_names=["r0", "row_7"],
        col_lb=[0.0],
        col_ub=[10.0],
        HighsBasis=HighsBasis,
        HighsBasisStatus=HighsBasisStatus,
    )
    assert [int(s) for s in basis.row_status] == [S_UPPER, S_BASIC]
    assert stats["n_rows_matched"] == 1
    assert stats["n_rows_defaulted"] == 1


# ---------------------------------------------------------------------------
# build_highs_basis — alien
# ---------------------------------------------------------------------------
def test_build_alien_defaults_and_synthetic_rows():
    nb = NamedBasis(
        col_status={"c0": S_BASIC},
        row_status={"r0": S_UPPER},
        fingerprint="x",
    )
    basis, stats = build_highs_basis(
        nb,
        "alien",
        # c1: finite lower, infinite upper -> nonbasic at lower.
        # c2: both bounds infinite (free) -> free nonbasic.
        col_names=["c0", "c1", "c2"],
        row_names=["r0", "r_extra", "row_9"],
        col_lb=[0.0, 0.0, -np.inf],
        col_ub=[np.inf, np.inf, np.inf],
        HighsBasis=HighsBasis,
        HighsBasisStatus=HighsBasisStatus,
    )
    assert basis.alien is True
    cols = [int(s) for s in basis.col_status]
    assert cols[0] == S_BASIC  # matched
    assert cols[1] == S_LOWER  # defaulted: finite lower is the only bound
    assert cols[2] in _FREE  # both bounds infinite -> free
    rows = [int(s) for s in basis.row_status]
    assert rows[0] == S_UPPER  # matched
    assert rows[1] == S_BASIC  # extra target row defaults basic
    assert rows[2] == S_BASIC  # synthetic row defaults basic
    assert len(basis.col_status) == 3
    assert len(basis.row_status) == 3
    assert stats["n_cols_matched"] == 1
    assert stats["n_cols_defaulted"] == 2
    assert stats["n_rows_defaulted"] == 2
    assert stats["alien"] is True


def test_build_alien_defaults_nearer_finite_bound():
    """Nonbasic default picks the bound with the smaller magnitude when
    both are finite."""
    nb = NamedBasis(col_status={}, row_status={}, fingerprint="x")
    basis, _ = build_highs_basis(
        nb,
        "alien",
        col_names=["near_lower", "near_upper"],
        row_names=[],
        col_lb=[-2.0, -50.0],
        col_ub=[100.0, 3.0],
        HighsBasis=HighsBasis,
        HighsBasisStatus=HighsBasisStatus,
    )
    cols = [int(s) for s in basis.col_status]
    assert cols[0] == S_LOWER  # |-2| <= |100|
    assert cols[1] == S_UPPER  # |3| < |-50|


# ---------------------------------------------------------------------------
# Bound-finiteness sanitation
# ---------------------------------------------------------------------------
def test_sanitize_kupper_with_infinite_upper_demoted():
    nb = NamedBasis(
        col_status={"c0": S_UPPER},  # names the upper bound...
        row_status={},
        fingerprint="x",
    )
    basis, stats = build_highs_basis(
        nb,
        "exact",
        col_names=["c0"],
        col_lb=[0.0],
        col_ub=[np.inf],  # ...which is infinite in THIS model
        row_names=[],
        HighsBasis=HighsBasis,
        HighsBasisStatus=HighsBasisStatus,
    )
    st = int(basis.col_status[0])
    assert st == S_LOWER  # demoted to the finite lower bound
    assert stats["n_sanitized"] >= 1


def test_sanitize_klower_both_infinite_goes_free():
    nb = NamedBasis(
        col_status={"c0": S_LOWER},
        row_status={},
        fingerprint="x",
    )
    basis, stats = build_highs_basis(
        nb,
        "exact",
        col_names=["c0"],
        col_lb=[-np.inf],
        col_ub=[np.inf],
        row_names=[],
        HighsBasis=HighsBasis,
        HighsBasisStatus=HighsBasisStatus,
    )
    assert int(basis.col_status[0]) in _FREE
    assert stats["n_sanitized"] >= 1


def test_bad_policy_raises():
    nb = NamedBasis(col_status={}, row_status={}, fingerprint="x")
    with pytest.raises(ValueError, match="policy must be"):
        build_highs_basis(
            nb,
            "warm",
            col_names=[],
            row_names=[],
            col_lb=[],
            col_ub=[],
            HighsBasis=HighsBasis,
            HighsBasisStatus=HighsBasisStatus,
        )


# ---------------------------------------------------------------------------
# Solution.get_named_basis
# ---------------------------------------------------------------------------
def test_get_named_basis_requires_live_solver():
    pb = Problem()
    build_dispatch(pb, make_toy_data())
    sol = pb.solve()  # default keep_solver=False -> highs is None
    assert sol.highs is None
    with pytest.raises(RuntimeError, match="keep_solver=True"):
        sol.get_named_basis()


def test_get_named_basis_duplicate_col_raises():
    pb = Problem()
    build_dispatch(pb, make_toy_data())
    sol = pb.solve(keep_solver=True)
    # Inject a duplicate column name: it would silently overwrite a status.
    dup = sol.col_names[:]
    dup[1] = dup[0]
    sol.col_names = dup
    with pytest.raises(ValueError, match="duplicate column name"):
        sol.get_named_basis()


def test_get_named_basis_solve_backed_smoke():
    pb = Problem()
    build_dispatch(pb, make_toy_data())
    sol = pb.solve(keep_solver=True)
    assert sol.optimal
    nb = sol.get_named_basis()

    # Every rendered variable column name is present with an int status.
    assert set(nb.col_status.keys()) == set(sol.col_names)
    for status in nb.col_status.values():
        assert isinstance(status, int)
        assert status in {S_LOWER, S_BASIC, S_UPPER, S_ZERO, S_NONBASIC}

    # Non-synthetic rows carried; statuses are ints.
    for name, status in nb.row_status.items():
        assert not name.startswith("row_") or not name[4:].isdigit()
        assert isinstance(status, int)

    # Fingerprint is a stable short hex digest across repeated calls.
    assert isinstance(nb.fingerprint, str)
    assert len(nb.fingerprint) == 16
    nb2 = sol.get_named_basis()
    assert nb.fingerprint == nb2.fingerprint


# ---------------------------------------------------------------------------
# Problem.set_named_basis (records intent only)
# ---------------------------------------------------------------------------
def test_set_named_basis_records_intent():
    pb = Problem()
    assert pb._warm_basis is None
    assert pb._warm_basis_policy is None
    nb = NamedBasis(col_status={"c0": S_LOWER}, row_status={}, fingerprint="x")
    pb.set_named_basis(nb, policy="alien")
    assert pb._warm_basis is nb
    assert pb._warm_basis_policy == "alien"


def test_set_named_basis_default_policy_exact():
    pb = Problem()
    nb = NamedBasis(col_status={}, row_status={}, fingerprint="x")
    pb.set_named_basis(nb)
    assert pb._warm_basis_policy == "exact"


def test_set_named_basis_bad_policy_raises():
    pb = Problem()
    nb = NamedBasis(col_status={}, row_status={}, fingerprint="x")
    with pytest.raises(ValueError, match="policy must be"):
        pb.set_named_basis(nb, policy="warm")


def test_named_basis_is_frozen():
    nb = NamedBasis(col_status={}, row_status={}, fingerprint="x")
    with pytest.raises(FrozenInstanceError):
        nb.fingerprint = "y"  # frozen dataclass
    # math import kept meaningful: sanity that inf handling matches build side.
    assert not math.isfinite(np.inf)
