"""``highs_names``: the LP names go to HiGHS always, on demand, or never.

Relational checks on a small network LP, for the streaming and
non-streaming ``Problem.solve`` and for ``WarmProblem``:

* the solution arrays are bitwise equal in every mode, and so are the
  Python-side ``Solution.col_names`` / ``row_names``;
* under ``"auto"`` (no name-writing option) and ``"never"`` HiGHS holds no
  names; after ``pass_names_to_highs()`` it holds exactly the ``"always"``
  names, and a HiGHS-written MPS / solution file is byte-identical to the
  ``"always"`` one;
* ``"auto"`` passes the names when a name-writing option is on;
* a ``save_memory`` solve passes them whatever the mode (its MPS round trip
  goes through HiGHS);
* ``WarmProblem`` rows / columns appended while names are out of HiGHS get
  their names once they are passed, and a named-basis warm start works
  without names in HiGHS.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import polar_high as fp
from polar_high.engine import HIGHS_NAME_OPTIONS, highs_options_need_names

_OPTS = {"output_flag": False}


def _build(n_t: int = 40, n_gen: int = 6, seed: int = 3) -> fp.Problem:
    rng = np.random.default_rng(seed)
    p = fp.Problem()
    t_idx = pl.DataFrame({"t": np.arange(n_t, dtype=np.int64)})
    g_idx = pl.DataFrame({"g": np.arange(n_gen, dtype=np.int64)})
    gt = g_idx.join(t_idx, how="cross")
    v_flow = p.add_var("v_flow", ("g", "t"), gt, lower=0.0)
    vq = p.add_var("vq", ("t",), t_idx, lower=0.0)
    cap = fp.Param(
        ("g",), pl.DataFrame({"g": np.arange(n_gen), "value": rng.uniform(20.0, 100.0, n_gen)})
    )
    demand = fp.Param(
        ("t",), pl.DataFrame({"t": np.arange(n_t), "value": rng.uniform(50.0, 200.0, n_t)})
    )
    cost = fp.Param(
        ("g",), pl.DataFrame({"g": np.arange(n_gen), "value": rng.uniform(1.0, 50.0, n_gen)})
    )
    p.add_cstr("max_flow", over=gt, sense="<=", lhs_terms={"flow": v_flow}, rhs_terms={"cap": cap})
    p.add_cstr(
        "balance",
        over=t_idx,
        sense="==",
        lhs_terms={"gen": fp.Sum(v_flow, over="g"), "slack": vq},
        rhs_terms={"demand": demand},
    )
    p.set_objective(fp.Sum(v_flow * cost) + fp.Sum(vq * 1000.0), sense="min")
    return p


def _solve(kind: str, mode: str, options: dict | None = None) -> tuple[fp.Solution, object]:
    opts = dict(_OPTS, **(options or {}))
    p = _build()
    if kind == "warm":
        wp = fp.WarmProblem(p, highs_names=mode)
        return wp.solve(options=opts), wp
    sol = p.solve(options=opts, keep_solver=True, streaming=(kind == "streaming"), highs_names=mode)
    return sol, None


def _highs_names(sol: fp.Solution) -> tuple[list, list]:
    lp = sol.highs.getLp()
    return list(lp.col_names_), list(lp.row_names_)


def _arrays_bits(sol: fp.Solution) -> tuple[bytes, ...]:
    hs = sol.highs.getSolution()
    return (
        np.asarray(hs.col_value, dtype=np.float64).tobytes(),
        np.asarray(hs.row_value, dtype=np.float64).tobytes(),
        np.asarray(hs.col_dual, dtype=np.float64).tobytes(),
        np.asarray(hs.row_dual, dtype=np.float64).tobytes(),
        np.float64(sol.obj).tobytes(),
    )


KINDS = ["streaming", "nonstreaming", "warm"]


@pytest.mark.parametrize("kind", KINDS)
def test_modes_solve_identically_and_names_come_on_demand(kind: str, tmp_path) -> None:
    ref, _ = _solve(kind, "always")
    ref_cols, ref_rows = _highs_names(ref)
    assert ref.highs_names_passed
    assert ref_cols and ref_rows and all(ref_cols) and all(ref_rows)
    ref.highs.writeModel(str(tmp_path / "always.mps"))
    ref.highs.writeSolution(str(tmp_path / "always.sol"), 1)

    for mode in ("auto", "never"):
        sol, _wp = _solve(kind, mode)
        assert sol.optimal
        assert _arrays_bits(sol) == _arrays_bits(ref)
        assert sol.col_names == ref.col_names
        assert sol.row_names == ref.row_names
        assert not sol.highs_names_passed
        assert _highs_names(sol) == ([], [])

        sol.pass_names_to_highs()
        sol.pass_names_to_highs()  # idempotent
        assert sol.highs_names_passed
        assert _highs_names(sol) == (ref_cols, ref_rows)
        sol.highs.writeModel(str(tmp_path / f"{mode}.mps"))
        sol.highs.writeSolution(str(tmp_path / f"{mode}.sol"), 1)
        assert (tmp_path / f"{mode}.mps").read_bytes() == (tmp_path / "always.mps").read_bytes()
        assert (tmp_path / f"{mode}.sol").read_bytes() == (tmp_path / "always.sol").read_bytes()


@pytest.mark.parametrize("kind", KINDS)
def test_auto_passes_names_when_an_option_writes_them(kind: str, tmp_path) -> None:
    ref, _ = _solve(kind, "always")
    ref_names = _highs_names(ref)
    for opt in (
        {"write_solution_to_file": True, "solution_file": str(tmp_path / f"{kind}.sol")},
        {"write_basis_file": str(tmp_path / f"{kind}.bas")},
    ):
        sol, _ = _solve(kind, "auto", opt)
        assert sol.highs_names_passed
        assert _highs_names(sol) == ref_names
        assert _arrays_bits(sol) == _arrays_bits(ref)
    # The solution file HiGHS wrote during run() carries the LP names.
    assert "v_flow[0,0]" in (tmp_path / f"{kind}.sol").read_text()
    # "never" ignores the option: HiGHS writes its own c<i> / r<i> names.
    sol, _ = _solve(kind, "never", {"write_basis_file": str(tmp_path / "never.bas")})
    assert not sol.highs_names_passed
    assert "v_flow[" not in (tmp_path / "never.bas").read_text()


def test_auto_reads_options_set_on_the_live_handle() -> None:
    sol, wp = _solve("warm", "auto")
    assert not sol.highs_names_passed
    h = sol.highs
    assert not highs_options_need_names(h)
    assert set(HIGHS_NAME_OPTIONS) >= {"write_solution_to_file", "write_model_to_file"}
    h.setOptionValue("write_iis_model_file", "")
    h.setOptionValue("write_model_to_file", True)
    h.setOptionValue("write_model_file", "/dev/null")
    assert highs_options_need_names(h)
    sol2 = wp.solve()
    assert sol2.highs_names_passed
    assert list(h.getLp().col_names_) == [n for n in sol2.col_names]


@pytest.mark.parametrize("mode", ["auto", "never"])
def test_save_memory_always_passes_names(mode: str) -> None:
    ref = _build().solve(options=_OPTS, keep_solver=True, save_memory=True)
    sol = _build().solve(options=_OPTS, keep_solver=True, save_memory=True, highs_names=mode)
    assert sol.highs_names_passed
    assert sol.col_names == ref.col_names and sol.row_names == ref.row_names
    assert _highs_names(sol) == _highs_names(ref)
    names = _highs_names(sol)
    assert "v_flow[0,0]" in names[0] and any(n.startswith("balance") for n in names[1])
    assert _arrays_bits(sol) == _arrays_bits(ref)


def test_build_only_mps_identical_in_every_mode(tmp_path) -> None:
    _build().build_only(str(tmp_path / "ref.mps"), options=_OPTS)
    p = _build()
    # build_only goes through the save_memory path, which forces "always".
    p._solve_streaming(
        options=_OPTS,
        keep_solver=False,
        save_memory=True,
        _mps_out_path=str(tmp_path / "never.mps"),
        _build_only=True,
        highs_names="never",
    )
    assert (tmp_path / "never.mps").read_bytes() == (tmp_path / "ref.mps").read_bytes()


def test_warm_appended_rows_and_cols_get_names_when_passed() -> None:
    def run(mode: str):
        wp = fp.WarmProblem(_build(), highs_names=mode)
        wp.solve(options=_OPTS)
        eta = wp.add_recourse_col("eta", 1.0, lower=0.0)
        wp.add_cut_row([0, eta], [1.0, 1.0], 0.5)
        sol = wp.solve()
        return wp, sol

    wp_ref, ref = run("always")
    wp, sol = run("never")
    assert _arrays_bits(sol) == _arrays_bits(ref)
    assert sol.col_names == ref.col_names and sol.row_names == ref.row_names
    assert _highs_names(sol) == ([], [])
    wp.pass_names_to_highs()
    assert _highs_names(sol) == _highs_names(ref)
    assert _highs_names(ref)[0][-1] == "eta"
    # Appended after the names are in: passed straight away, as under "always".
    wp.add_cut_row([1], [1.0], 0.0)
    wp_ref.add_cut_row([1], [1.0], 0.0)
    assert _highs_names(wp.solve()) == _highs_names(wp_ref.solve())


def test_kept_solution_does_not_keep_the_warm_problem_alive() -> None:
    import gc
    import weakref

    wp = fp.WarmProblem(_build(), highs_names="never")
    sol = wp.solve(options=_OPTS)
    ref = weakref.ref(wp)
    del wp
    gc.collect()
    assert ref() is None
    sol.pass_names_to_highs()
    assert sol.highs_names_passed
    cols, rows = _highs_names(sol)
    assert cols == list(sol.col_names) and rows == list(sol.row_names)


def test_named_basis_warm_start_without_names_in_highs() -> None:
    src = fp.WarmProblem(_build(), highs_names="never").solve(options=_OPTS)
    nb = src.get_named_basis()
    cold = fp.WarmProblem(_build()).solve(options=_OPTS)
    wp = fp.WarmProblem(_build(), highs_names="never")
    wp.set_named_basis(nb, policy="exact")
    warm = wp.solve(options=_OPTS)
    assert wp._last_basis_fingerprint == nb.fingerprint
    assert warm.optimal and abs(warm.obj - cold.obj) <= 1e-9 * (1 + abs(cold.obj))
    assert (
        warm.highs.getInfo().simplex_iteration_count < cold.highs.getInfo().simplex_iteration_count
    )


def test_invalid_mode_rejected() -> None:
    with pytest.raises(ValueError):
        fp.WarmProblem(_build(), highs_names="sometimes")
    with pytest.raises(ValueError):
        _build().solve(highs_names="sometimes")
