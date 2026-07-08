"""Tests for ``Problem.write_mps`` stashing a structural basis-fingerprint
on ``self._last_mps_fingerprint`` (warm-start Phase 1, Part 2).

The fingerprint keys a downstream ``save_memory`` subprocess warm-start
cache by the exact column / constraint-row name-set the MPS carries.  It
must be computed BEFORE ``release`` drops ``_cstrs`` (row names are then
unrecoverable), only when real names are emitted, and must line up with
an independently computed :func:`polar_high.basis_fingerprint` over the
same model's names.
"""

import dataclasses

import polars as pl
from toy_data import ToyData, make_toy_data
from toy_model import build_dispatch

from polar_high import Param, Problem, basis_fingerprint


def _make_toy_data_variant() -> ToyData:
    """A STRUCTURALLY different toy dataset: five timesteps instead of
    four, so the emitted column / row name-set differs (extra
    ``v_flow[*,t5]``, ``vq_up[t5]``, ``node_balance[t5]`` ... names)."""
    base = make_toy_data()
    timesteps = pl.DataFrame({"t": ["t1", "t2", "t3", "t4", "t5"]})
    pt = base.processes.join(timesteps, how="cross")
    wind_t = base.wind_only.join(timesteps, how="cross")
    return dataclasses.replace(
        base,
        timesteps=timesteps,
        pt=pt,
        wind_t=wind_t,
        avail=Param(
            ("t",),
            pl.DataFrame(
                {"t": ["t1", "t2", "t3", "t4", "t5"], "value": [50.0, 30.0, 80.0, 20.0, 40.0]}
            ),
        ),
        demand=Param(
            ("t",),
            pl.DataFrame(
                {"t": ["t1", "t2", "t3", "t4", "t5"], "value": [60.0, 90.0, 70.0, 80.0, 55.0]}
            ),
        ),
    )


def test_write_mps_stashes_fingerprint(tmp_path):
    """A named write_mps (release=True) leaves a non-empty fingerprint str,
    computed before the source is released."""
    pb = Problem()
    build_dispatch(pb, make_toy_data())
    assert pb._last_mps_fingerprint is None  # nothing written yet

    pb.write_mps(tmp_path / "toy.mps", release=True)

    fp = pb._last_mps_fingerprint
    assert isinstance(fp, str)
    assert fp  # non-empty
    assert len(fp) == 16  # phbasis-v1 digest width
    # Released: the row source is gone, but the stashed fingerprint survives.
    assert pb._released is True


def test_fingerprint_deterministic_same_model(tmp_path):
    """Two independently built copies of the SAME model produce the same
    fingerprint (order-independent set hash)."""
    pb1 = Problem()
    build_dispatch(pb1, make_toy_data())
    pb1.write_mps(tmp_path / "a.mps")

    pb2 = Problem()
    build_dispatch(pb2, make_toy_data())
    pb2.write_mps(tmp_path / "b.mps")

    assert pb1._last_mps_fingerprint == pb2._last_mps_fingerprint


def test_fingerprint_differs_for_different_structure(tmp_path):
    """A structurally different model (different timestep dims) yields a
    different fingerprint."""
    pb = Problem()
    build_dispatch(pb, make_toy_data())
    pb.write_mps(tmp_path / "base.mps")

    pb_var = Problem()
    build_dispatch(pb_var, _make_toy_data_variant())
    pb_var.write_mps(tmp_path / "variant.mps")

    assert pb._last_mps_fingerprint != pb_var._last_mps_fingerprint


def test_generic_names_yield_none(tmp_path):
    """``emit_names=False`` emits positional C/R names — useless as a
    cross-run key — so the fingerprint is reset to None."""
    pb = Problem()
    build_dispatch(pb, make_toy_data())
    pb.write_mps(tmp_path / "generic.mps", emit_names=False)
    assert pb._last_mps_fingerprint is None


def test_generic_names_reset_after_named(tmp_path):
    """A named write stashes a fingerprint; a subsequent generic-name write
    on a fresh Problem resets it to None (no stale key carries over)."""
    pb = Problem()
    build_dispatch(pb, make_toy_data())
    pb.write_mps(tmp_path / "named.mps")
    assert isinstance(pb._last_mps_fingerprint, str)

    pb2 = Problem()
    build_dispatch(pb2, make_toy_data())
    pb2.write_mps(tmp_path / "gen.mps", emit_names=False)
    assert pb2._last_mps_fingerprint is None


def test_fingerprint_matches_independent_computation(tmp_path):
    """The stashed fingerprint equals :func:`basis_fingerprint` computed
    independently over the model's column names and CONSTRAINT-row names.

    The independent name-set comes from a ``solve(keep_solver=True)`` of
    the same (deterministic) model: ``Solution.col_names`` are the LP
    columns and ``Solution.row_names`` are the constraint rows (no "cost"
    sentinel — that is a write_mps-only prefix).  ``basis_fingerprint``
    drops synthetic ``row_<i>`` names on both sides, so any positional
    rows (there are none in the fully-named toy model) would not perturb
    the comparison.
    """
    # Solve one copy to harvest the authoritative name-set.
    pb_solve = Problem()
    build_dispatch(pb_solve, make_toy_data())
    sol = pb_solve.solve(keep_solver=True)
    assert sol.optimal

    expected = basis_fingerprint(sol.col_names, sol.row_names)

    # A second, independently built copy writes the MPS and stashes its
    # fingerprint — deterministic, so it must equal the independent hash.
    pb_write = Problem()
    build_dispatch(pb_write, make_toy_data())
    pb_write.write_mps(tmp_path / "toy.mps")

    assert pb_write._last_mps_fingerprint == expected
