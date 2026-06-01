"""Focused regression test for the Layer 2 side-vector emit path.

Stage A's end-to-end safety net
(``flextool/tests/engine_polars/autoscale/test_layer2_roundtrip.py``)
proves that ``unscale_solution(solve(scaled_problem))`` matches
``solve(raw_problem)`` bit-for-bit.  That test, however, exercises
both the solve path and the MPS path through the same side vectors:
if both were wrong in the same way, it would still pass.

This test exercises :meth:`polar_high.engine.Problem.write_mps`
directly.  It sets fake side vectors with distinct power-of-two
factors per row and per column (so that any indexing offset or
missed multiply site changes a coefficient visibly), writes two MPS
files (one without side vectors, one with), parses both, and asserts
the per-entry algebra:

* matrix entry  ``A'[i,j]  =  A[i,j] * row_factor[i] / col_factor_math[j]``
* RHS entry     ``b'[i]    =  b[i]   * row_factor[i]``
* objective     ``c'[j]    =  c[j]   / col_factor_math[j]``
* bounds        unchanged  (side vectors do NOT touch ``Var.lower/upper``;
                bound mutation lives in ``apply_layer2`` proper, which
                this test deliberately does NOT call).

Convention reminder (STATE.md, 2026-05-28).  Consumers multiply
``coef`` by ``_layer2_col_factor[col_id]``, but the math the consumers
implement is ``A / cf_math``.  Therefore ``_layer2_col_factor`` stores
``1 / cf_math`` (the inverse forward factor).  ``_layer2_row_factor``
stores ``rf_math`` forward (no inversion).  This test uses ``cf_math``
in the assertions and installs ``1 / cf_math`` on the Problem.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from polar_high import Param, Problem


# ---------------------------------------------------------------------------
# MPS parsing — tiny hand-rolled reader for the subset we emit
# ---------------------------------------------------------------------------
def _parse_mps(
    path: str,
) -> tuple[
    dict[tuple[str, str], float],  # matrix[(row_name, col_name)] = coef
    dict[str, float],  # rhs[row_name]                 = rhs
    dict[str, float],  # obj[col_name]                 = coef
    list[tuple[str, ...]],  # bound lines (verbatim tokens)
]:
    matrix: dict[tuple[str, str], float] = {}
    rhs: dict[str, float] = {}
    obj: dict[str, float] = {}
    bounds: list[tuple[str, ...]] = []
    section: str | None = None
    with open(path) as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            # Section headers start in column 0.
            stripped = line.lstrip()
            if line[:1] not in (" ", "\t"):
                head = stripped.split()[0]
                if head in (
                    "NAME",
                    "OBJSENSE",
                    "ROWS",
                    "COLUMNS",
                    "RHS",
                    "RANGES",
                    "BOUNDS",
                    "ENDATA",
                ):
                    section = head
                    continue
                # OBJSENSE value line (also column-0 in our emitter
                # but indented — fall through).
            tokens = stripped.split()
            if section == "COLUMNS":
                if len(tokens) >= 3 and tokens[1] == "'MARKER'":
                    # MARKER INTORG/INTEND — skip; integer markers don't
                    # affect coefficient values.
                    continue
                # Lines look like: "<colname> <rowname> <value> [<rowname2> <value2>]"
                if len(tokens) >= 3:
                    col = tokens[0]
                    pairs = tokens[1:]
                    for i in range(0, len(pairs) - 1, 2):
                        row = pairs[i]
                        val = float(pairs[i + 1])
                        if row == "cost":
                            obj[col] = val
                        else:
                            matrix[(row, col)] = val
            elif section == "RHS":
                if len(tokens) >= 3:
                    # "rhs <row1> <val1> [<row2> <val2>]"
                    pairs = tokens[1:]
                    for i in range(0, len(pairs) - 1, 2):
                        rhs[pairs[i]] = float(pairs[i + 1])
            elif section == "BOUNDS":
                bounds.append(tuple(tokens))
    return matrix, rhs, obj, bounds


# ---------------------------------------------------------------------------
# Small but representative LP fixture
# ---------------------------------------------------------------------------
def _build_problem() -> tuple[Problem, dict[str, np.ndarray], list[str]]:
    """Build a small LP exercising every emit-site branch.

    Returns the Problem, a dict ``var_name -> col_id array``, and the
    ordered list of constraint row names (matching the order
    ``Problem._cstrs`` produces).
    """
    pb = Problem()

    # Var families:
    #   x over (i,): 3 columns (col_id 0..2) — used in dim LHS terms
    #   y dimless: 1 column   (col_id 3)    — used in scalar LHS terms
    #     (truly scalar variables, dims=(); they broadcast across the
    #     constraint's ``over`` rows via the scalar-term branch)
    #   z dimless: 1 column   (col_id 4)
    idx_i = pl.DataFrame({"i": [0, 1, 2]})
    idx_dummy = pl.DataFrame({"_dummy": [0]})
    x = pb.add_var("x", "i", idx_i, lower=0.0, upper=10.0)
    y = pb.add_var("y", (), idx_dummy, lower=-1.0, upper=5.0)
    z = pb.add_var("z", (), idx_dummy, lower=0.0, upper=math.inf)

    # Param-coefficient LHS family (dim term): coef varies per row.
    # a_i * x_i  ==  rhs(i)  — 3 rows.
    a = Param(
        ("i",),
        pl.DataFrame({"i": [0, 1, 2], "value": [3.0, 5.0, 7.0]}),
    )
    rhs_param = Param(
        ("i",),
        pl.DataFrame({"i": [0, 1, 2], "value": [11.0, 13.0, 17.0]}),
    )
    pb.add_cstr(
        "dim_eq",
        over=idx_i,
        sense="==",
        lhs_terms={"ax": a * x},
        rhs_terms={"r": rhs_param},
    )

    # Literal-coefficient LHS family (scalar term, dim over=idx_i):
    # 2.0 * y  +  x_i  <=  19.0  — 3 rows; the y term is scalar
    # (no dims) and broadcasts across the over rows, while the x term
    # is dim.
    pb.add_cstr(
        "scalar_le",
        over=idx_i,
        sense="<=",
        lhs_terms={"yc": 2.0 * y, "xx": x},
        rhs_terms={"k": 19.0},
    )

    # Scalar-rhs single-row family (no over) — exercises the
    # row_count=1 / over=None path on both LHS and RHS.
    pb.add_cstr(
        "scalar_row",
        over=None,
        sense=">=",
        lhs_terms={"yz": y + z},
        rhs_terms={"k": -2.0},
    )

    # Var-on-RHS family — moved into LHS as a negated term.  This
    # exercises the "folded into LHS" path: the rhs Var becomes a
    # negative coefficient on its column in the constraint row.
    # x_i + (- y) >= 0  (originally x_i >= y, but written as
    # ``lhs={x}, rhs={y}``).
    pb.add_cstr(
        "var_rhs",
        over=idx_i,
        sense=">=",
        lhs_terms={"x": x},
        rhs_terms={"y": y},  # gets folded into LHS as -y
    )

    # Objective: dim term (b_i * x_i) + scalar term (1.5 * y) + scalar
    # term (3.0 * z).  Three obj terms in self._obj_terms; the dim
    # term contributes 3 entries (col_id 0..2), each scalar term
    # contributes 1.
    b = Param(
        ("i",),
        pl.DataFrame({"i": [0, 1, 2], "value": [0.25, 0.5, 1.25]}),
    )
    pb.set_objective(b * x + 1.5 * y + 3.0 * z, sense="min")

    col_ids = {
        "x": x.frame.sort("i")["col_id"].to_numpy(),
        "y": y.frame["col_id"].to_numpy(),
        "z": z.frame["col_id"].to_numpy(),
    }

    # Row name layout matches what write_mps emits (see engine.py
    # around line 1670):
    #   dim_eq       -> "dim_eq[0]", "dim_eq[1]", "dim_eq[2]"
    #   scalar_le    -> "scalar_le[0]", "scalar_le[1]", "scalar_le[2]"
    #   scalar_row   -> "scalar_row"
    #   var_rhs      -> "var_rhs[0]", "var_rhs[1]", "var_rhs[2]"
    # Total: 10 constraint rows.
    row_names = (
        [f"dim_eq[{i}]" for i in range(3)]
        + [f"scalar_le[{i}]" for i in range(3)]
        + ["scalar_row"]
        + [f"var_rhs[{i}]" for i in range(3)]
    )
    return pb, col_ids, row_names


# ---------------------------------------------------------------------------
# Test #1 — write_mps + side vectors roundtrip
# ---------------------------------------------------------------------------
def test_write_mps_with_layer2_side_vectors(tmp_path) -> None:
    """Set fake power-of-two side vectors directly on the Problem
    (bypassing ``apply_layer2``) and verify ``write_mps`` emits every
    matrix / rhs / objective coefficient with the right per-(row,col)
    factor.  Bounds must be unchanged (side vectors don't touch bounds;
    ``Var.lower/upper`` mutation belongs to ``apply_layer2`` proper,
    deliberately not exercised here)."""
    pb, col_ids, row_names = _build_problem()
    n_cols = pb._next_col
    n_rows = len(row_names)

    path_raw = tmp_path / "raw.mps"
    pb.write_mps(str(path_raw))
    mat_raw, rhs_raw, obj_raw, bnd_raw = _parse_mps(str(path_raw))

    # Sanity checks on the unscaled MPS — make sure the fixture
    # actually populated the four branches we want to scale.
    assert mat_raw, "raw MPS produced no matrix entries"
    assert rhs_raw, "raw MPS produced no rhs entries"
    assert obj_raw, "raw MPS produced no objective entries"
    # Var-on-RHS folding: y's column should appear in the var_rhs[*]
    # rows as a negative coefficient.
    y_col = "y"
    for ri in range(3):
        rn = f"var_rhs[{ri}]"
        assert (rn, y_col) in mat_raw, (
            f"folded Var-on-RHS missing entry ({rn!r}, {y_col!r}); "
            f"emit path may have skipped the negation step"
        )
        assert mat_raw[(rn, y_col)] < 0, (
            f"folded Var-on-RHS entry ({rn!r}, {y_col!r}) should be "
            f"negative; got {mat_raw[(rn, y_col)]}"
        )

    # ------------------------------------------------------------------
    # Fake side vectors — distinct power-of-two factors per col / row.
    # Powers of two make the multiplications exact in IEEE float64,
    # so any non-trivial discrepancy must be a real bug.
    # ------------------------------------------------------------------
    # col_factor_math[j]: cycles through 0.5, 1.0, 2.0 (per j)
    col_factor_math = np.array([2.0 ** ((j % 3) - 1) for j in range(n_cols)], dtype=np.float64)
    # row_factor[i]: cycles 0.25, 0.5, 1.0, 2.0, 4.0 (per i)
    row_factor = np.array([2.0 ** ((i % 5) - 2) for i in range(n_rows)], dtype=np.float64)

    # Convention: consumers do coef *= _layer2_col_factor[col_id], but
    # the math is coef /= col_factor_math[col_id], so the side vector
    # stores 1 / col_factor_math (see STATE.md 2026-05-28 convention
    # asymmetry note).
    pb._layer2_col_factor = 1.0 / col_factor_math
    pb._layer2_row_factor = row_factor
    # Stage B1: assigning side vectors after canonicalise() requires
    # marking the cached _matrix stale so the next write_mps rebuilds
    # with the baked vectors.  In production, ``apply_layer2`` flips
    # this flag itself; the test mimics that responsibility.
    pb._canonical_dirty = True
    # Deliberately do NOT set _layer2_locked = True and do NOT mutate
    # Var.lower/upper.  This test exercises the read path in isolation.

    path_scl = tmp_path / "scaled.mps"
    pb.write_mps(str(path_scl))
    mat_scl, rhs_scl, obj_scl, bnd_scl = _parse_mps(str(path_scl))

    # ------------------------------------------------------------------
    # Index maps from MPS names back to integer ids.
    # ------------------------------------------------------------------
    name_to_col: dict[str, int] = {}
    # x[i] for i in 0..2
    for i in range(3):
        name_to_col[f"x[{i}]"] = int(col_ids["x"][i])
    name_to_col["y"] = int(col_ids["y"][0])
    name_to_col["z"] = int(col_ids["z"][0])
    name_to_row = {name: i for i, name in enumerate(row_names)}

    # ------------------------------------------------------------------
    # Matrix: every entry must be raw * row_factor[r] / col_factor_math[c].
    # Set of (row, col) keys must match exactly between the two files
    # — scaling never adds or removes nonzeros.
    # ------------------------------------------------------------------
    assert set(mat_raw.keys()) == set(mat_scl.keys()), (
        f"matrix support set changed after scaling — "
        f"only_in_raw={set(mat_raw) - set(mat_scl)}, "
        f"only_in_scaled={set(mat_scl) - set(mat_raw)}"
    )
    for (rn, cn), v_raw in mat_raw.items():
        assert rn in name_to_row, f"unrecognised row name {rn!r} in MPS"
        assert cn in name_to_col, f"unrecognised col name {cn!r} in MPS"
        r = name_to_row[rn]
        c = name_to_col[cn]
        expected = v_raw * row_factor[r] / col_factor_math[c]
        v_scl = mat_scl[(rn, cn)]
        assert v_scl == expected or math.isclose(v_scl, expected, rel_tol=1e-12, abs_tol=0.0), (
            f"matrix entry ({rn!r}, {cn!r}): expected "
            f"{v_raw} * rf[{r}]={row_factor[r]} / cf[{c}]={col_factor_math[c]} "
            f"= {expected}, got {v_scl}"
        )

    # ------------------------------------------------------------------
    # RHS: every entry must be raw * row_factor[r].
    # ------------------------------------------------------------------
    assert set(rhs_raw.keys()) == set(rhs_scl.keys()), (
        f"rhs support set changed after scaling — "
        f"only_in_raw={set(rhs_raw) - set(rhs_scl)}, "
        f"only_in_scaled={set(rhs_scl) - set(rhs_raw)}"
    )
    for rn, v_raw in rhs_raw.items():
        r = name_to_row[rn]
        expected = v_raw * row_factor[r]
        v_scl = rhs_scl[rn]
        assert v_scl == expected or math.isclose(v_scl, expected, rel_tol=1e-12, abs_tol=0.0), (
            f"rhs entry ({rn!r}): expected {v_raw} * rf[{r}]={row_factor[r]} "
            f"= {expected}, got {v_scl}"
        )

    # ------------------------------------------------------------------
    # Objective: every entry must be raw / col_factor_math[c]
    # (no row factor — objective row "cost" is not in row_factor).
    # ------------------------------------------------------------------
    assert set(obj_raw.keys()) == set(obj_scl.keys()), (
        f"obj support set changed after scaling — "
        f"only_in_raw={set(obj_raw) - set(obj_scl)}, "
        f"only_in_scaled={set(obj_scl) - set(obj_raw)}"
    )
    for cn, v_raw in obj_raw.items():
        c = name_to_col[cn]
        expected = v_raw / col_factor_math[c]
        v_scl = obj_scl[cn]
        assert v_scl == expected or math.isclose(v_scl, expected, rel_tol=1e-12, abs_tol=0.0), (
            f"obj entry ({cn!r}): expected {v_raw} / cf[{c}]={col_factor_math[c]} "
            f"= {expected}, got {v_scl}"
        )

    # ------------------------------------------------------------------
    # Bounds: side vectors don't touch Var.lower/upper, so the BOUNDS
    # section must be byte-identical between the two MPS files.
    # ------------------------------------------------------------------
    assert bnd_raw == bnd_scl, (
        "BOUNDS section changed after side vectors were installed — "
        "side-vector write_mps must NOT mutate bounds.  apply_layer2 "
        "is the only legitimate bound mutator, and this test does "
        "not call it."
    )
