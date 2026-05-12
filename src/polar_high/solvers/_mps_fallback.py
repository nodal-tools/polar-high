"""File-based (MPS) fallback dispatch for commercial solvers.

This module implements the slow-but-universal escape hatch documented in
``specs/polar-high-multi-solver-handoff.md`` Step 3.  When a user has a
commercial solver binary on their machine but **not** the matching Python
wrapper (corporate Python, no admin rights, mismatched ABI), the direct
adapters in ``_gurobi.py`` / ``_cplex.py`` / ``_xpress.py`` / ``_copt.py``
all fail to import.  This module is the alternative path:

1. Build a fresh MPS file via a transient :class:`highspy.Highs` instance
   (we never call ``h.run()`` — HiGHS here is *only* an MPS writer).
2. Locate the solver's command-line binary on PATH (or in a small set of
   conventional install dirs).
3. ``subprocess.run`` the binary with a per-solver script that reads the
   MPS, optimizes, and writes a ``.sol`` file.
4. Parse the ``.sol`` back into a :class:`SolverResult`.

The contract matches the rest of ``polar_high.solvers``: **no warm
starts, no incremental edits**.  Every call writes a fresh MPS,
sub-processes the solver from scratch, parses the result, and cleans up.

Licensing stays the solver's concern.  If the solver's stderr mentions
"license", we upgrade the resulting :class:`SolverError` to a
:class:`LicenseError`; otherwise we simply surface the captured stdout +
stderr to the caller so they can see what the binary said.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import highspy
import numpy as np

from ._base import IOMode, LicenseError, SolverError, SolverResult, SolverStatus
from ._lp_view import LpView


# ---------------------------------------------------------------------------
# MPS writer (transient HiGHS)
# ---------------------------------------------------------------------------
def _write_mps(view: LpView, path: str) -> None:
    """Write the LP described by ``view`` to ``path`` as an MPS file.

    Implementation mirrors the ``passModel`` block of
    :func:`polar_high.solvers._highs.run`, but **never calls** ``h.run()``;
    HiGHS is used purely as the MPS writer here.
    """
    n_cols = int(view.n_cols)
    n_rows = int(view.n_rows)

    inf = highspy.kHighsInf
    col_lb_h = np.where(view.col_lb == -np.inf, -inf, view.col_lb).astype(np.float64)
    col_ub_h = np.where(view.col_ub == np.inf, inf, view.col_ub).astype(np.float64)
    row_lb_h = np.where(view.row_lb == -np.inf, -inf, view.row_lb).astype(np.float64)
    row_ub_h = np.where(view.row_ub == np.inf, inf, view.row_ub).astype(np.float64)

    lp = highspy.HighsLp()
    lp.num_col_ = n_cols
    lp.num_row_ = n_rows
    lp.col_cost_ = view.col_obj.astype(np.float64)
    lp.col_lower_ = col_lb_h
    lp.col_upper_ = col_ub_h
    lp.row_lower_ = row_lb_h
    lp.row_upper_ = row_ub_h
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.num_col_ = n_cols
    lp.a_matrix_.num_row_ = n_rows
    lp.a_matrix_.start_ = view.a_start
    lp.a_matrix_.index_ = view.a_index
    lp.a_matrix_.value_ = view.a_value
    lp.sense_ = highspy.ObjSense.kMaximize if view.sense == "max" else highspy.ObjSense.kMinimize
    if view.obj_offset:
        lp.offset_ = float(view.obj_offset)
    if view.integrality is not None:
        kCont = highspy.HighsVarType.kContinuous
        kInt = highspy.HighsVarType.kInteger
        integ_arr = np.where(view.integrality.astype(bool), kInt, kCont)
        lp.integrality_ = integ_arr.tolist()

    h = highspy.Highs()
    # Silence HiGHS chatter during the write — we don't want it printing
    # to stderr or fighting for the user's terminal.
    try:
        h.setOptionValue("output_flag", False)
    except Exception:
        pass
    h.passModel(lp)
    for i, n in enumerate(view.col_names):
        if n is not None:
            h.passColName(i, n)
    for i, n in enumerate(view.row_names):
        if n is not None:
            h.passRowName(i, n)

    status = h.writeModel(path)
    ok_status = getattr(highspy.HighsStatus, "kOk", None)
    if ok_status is not None and status != ok_status:
        raise SolverError(f"highspy failed to write MPS to {path!r} (status={status!r})")


# ---------------------------------------------------------------------------
# Binary lookup
# ---------------------------------------------------------------------------
# Canonical binary name per solver_name.
_BINARY_NAMES: dict[str, str] = {
    "gurobi": "gurobi_cl",
    "cplex": "cplex",
    "xpress": "optimizer",
    "copt": "copt_cmd",
}

# Conventional POSIX install dirs to scan when PATH doesn't have the binary.
# Each entry is a directory; for each binary we check ``<dir>/<binary>``
# and (where appropriate) ``<dir>/bin/<binary>``.
_POSIX_INSTALL_DIRS: dict[str, list[str]] = {
    "gurobi": [
        "/opt/gurobi/bin",
        "/opt/gurobi/linux64/bin",
        "/Library/gurobi/bin",
        os.path.expanduser("~/gurobi/bin"),
    ],
    "cplex": [
        "/opt/ibm/ILOG/CPLEX_Studio/cplex/bin/x86-64_linux",
        "/opt/ibm/ILOG/CPLEX_Studio/cplex/bin",
        "/opt/cplex/bin",
        os.path.expanduser("~/cplex/bin"),
    ],
    "xpress": [
        "/opt/xpressmp/bin",
        "/opt/fico/xpress/bin",
        os.path.expanduser("~/xpressmp/bin"),
    ],
    "copt": [
        "/opt/copt/bin",
        "/opt/copt71/bin",
        os.path.expanduser("~/copt/bin"),
    ],
}


def _find_solver_binary(solver_name: str) -> Path | None:
    """Return the absolute path to the solver's CLI binary, or ``None``.

    Lookup order:

    1. ``shutil.which(<canonical_name>)`` — honours ``$PATH`` on every
       platform.
    2. On POSIX (``os.name == 'posix'``), check the conventional install
       directories listed in :data:`_POSIX_INSTALL_DIRS` for the solver.

    Returns ``None`` if neither locates the binary.  Callers translate
    ``None`` to a :class:`SolverError` with a hint about installing the
    solver and/or putting its ``bin/`` directory on ``$PATH``.
    """
    bin_name = _BINARY_NAMES.get(solver_name)
    if bin_name is None:
        return None

    found = shutil.which(bin_name)
    if found is not None:
        return Path(found)

    if os.name == "posix":
        for d in _POSIX_INSTALL_DIRS.get(solver_name, []):
            candidate = Path(d) / bin_name
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate

    return None


# ---------------------------------------------------------------------------
# Per-solver command builders
# ---------------------------------------------------------------------------
def _gurobi_script(binary: Path, mps_path: Path, sol_path: Path) -> list[str]:
    """``gurobi_cl ResultFile=<sol> <mps>`` — see Gurobi CL reference."""
    return [str(binary), f"ResultFile={sol_path}", str(mps_path)]


def _cplex_script(binary: Path, mps_path: Path, sol_path: Path) -> tuple[list[str], str]:
    """CPLEX interactive optimizer: pipe commands on stdin.

    Returns ``(argv, stdin_text)`` — the caller forwards ``stdin_text``
    to ``subprocess.run(input=...)``.  We use ``read``/``optimize``/
    ``write`` rather than the ``-c`` form because the ``-c`` form quotes
    differently across CPLEX versions.
    """
    cmds = "\n".join(
        [
            f"read {mps_path}",
            "optimize",
            f"write {sol_path} sol",
            "quit",
            "",
        ]
    )
    return [str(binary)], cmds


def _xpress_script(binary: Path, mps_path: Path, sol_path: Path) -> tuple[list[str], str]:
    """Xpress optimizer script via stdin.

    Xpress' console reads command lines from stdin; we ask it to
    ``readprob``, ``maxim``/``minim`` (we use ``optimize`` which dispatches
    on the MPS file's sense), then ``writesol`` and ``quit``.
    """
    cmds = "\n".join(
        [
            f"readprob {mps_path}",
            "lpoptimize",
            f"writesol {sol_path}",
            "quit",
            "",
        ]
    )
    return [str(binary)], cmds


def _copt_script(binary: Path, mps_path: Path, sol_path: Path) -> tuple[list[str], str]:
    """COPT's ``copt_cmd`` command-line is similar to Gurobi.

    ``copt_cmd`` accepts a ``-c "..."`` script form; we pipe commands on
    stdin instead for parity with the CPLEX/Xpress branches.
    """
    cmds = "\n".join(
        [
            f"read {mps_path}",
            "optimize",
            f"write {sol_path}",
            "quit",
            "",
        ]
    )
    return [str(binary)], cmds


# ---------------------------------------------------------------------------
# Sol-file parsers
# ---------------------------------------------------------------------------
def _parse_gurobi_sol(
    path: Path,
) -> tuple[SolverStatus, float | None, dict[str, float] | None, dict[str, float] | None]:
    """Parse a Gurobi ``.sol`` file.

    Gurobi's ``.sol`` is key=value text:

    ::

        # Objective value = 6500.0
        x[0] 1.0
        y[0] 2.0

    Lines starting with ``#`` are comments.  If the file is missing, the
    solve almost certainly failed; return :data:`SolverStatus.OTHER`.
    """
    if not path.is_file():
        return SolverStatus.OTHER, None, None, None
    objective: float | None = None
    primal: dict[str, float] = {}
    with path.open("r") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                # "# Objective value = <value>"
                m = re.search(r"[Oo]bjective\s+value\s*=\s*([-\d.eE+inf]+)", line)
                if m:
                    try:
                        objective = float(m.group(1))
                    except ValueError:
                        objective = None
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    primal[parts[0]] = float(parts[1])
                except ValueError:
                    continue
    status = SolverStatus.OPTIMAL if primal else SolverStatus.OTHER
    return status, objective, (primal or None), None


def _parse_copt_sol(
    path: Path,
) -> tuple[SolverStatus, float | None, dict[str, float] | None, dict[str, float] | None]:
    """COPT's ``.sol`` format matches Gurobi's key=value layout closely."""
    return _parse_gurobi_sol(path)


def _parse_cplex_sol(
    path: Path,
) -> tuple[SolverStatus, float | None, dict[str, float] | None, dict[str, float] | None]:
    """Parse a CPLEX XML ``.sol`` file via ``xml.etree.ElementTree``.

    The structure (CPLEX Studio 12+) looks like:

    .. code-block:: xml

       <CPLEXSolution>
         <header solutionStatusString="optimal" objectiveValue="3.0" .../>
         <variables>
           <variable name="x[0]" value="1.0"/>
           ...
         </variables>
         <linearConstraints>
           <constraint name="c1" dual="0.5"/>
           ...
         </linearConstraints>
       </CPLEXSolution>
    """
    if not path.is_file():
        return SolverStatus.OTHER, None, None, None

    try:
        tree = ET.parse(path)
    except ET.ParseError:
        return SolverStatus.OTHER, None, None, None
    root = tree.getroot()

    # Some CPLEX builds wrap multiple solutions in <CPLEXSolutions>; pick
    # the first child if so.
    if root.tag == "CPLEXSolutions":
        children = list(root)
        if not children:
            return SolverStatus.OTHER, None, None, None
        root = children[0]

    objective: float | None = None
    status = SolverStatus.OTHER

    header = root.find("header")
    if header is not None:
        obj_str = header.get("objectiveValue")
        if obj_str is not None:
            try:
                objective = float(obj_str)
            except ValueError:
                pass
        status_str = (header.get("solutionStatusString") or "").lower()
        if "optimal" in status_str:
            status = SolverStatus.OPTIMAL
        elif "infeasible" in status_str:
            status = SolverStatus.INFEASIBLE
        elif "unbounded" in status_str:
            status = SolverStatus.UNBOUNDED
        elif "time" in status_str:
            status = SolverStatus.TIME_LIMIT

    primal: dict[str, float] = {}
    vars_el = root.find("variables")
    if vars_el is not None:
        for v in vars_el.findall("variable"):
            name = v.get("name")
            val_str = v.get("value")
            if name is None or val_str is None:
                continue
            try:
                primal[name] = float(val_str)
            except ValueError:
                continue

    dual: dict[str, float] = {}
    cons_el = root.find("linearConstraints")
    if cons_el is not None:
        for c in cons_el.findall("constraint"):
            name = c.get("name")
            dual_str = c.get("dual")
            if name is None or dual_str is None:
                continue
            try:
                dual[name] = float(dual_str)
            except ValueError:
                continue

    # Status fallback: if we recovered primal values but the header
    # string was missing, treat it as OPTIMAL.
    if status == SolverStatus.OTHER and primal:
        status = SolverStatus.OPTIMAL

    return status, objective, (primal or None), (dual or None)


def _parse_xpress_sol(
    path: Path,
) -> tuple[SolverStatus, float | None, dict[str, float] | None, dict[str, float] | None]:
    """Parse Xpress' tabular ``.sol`` ASCII output.

    Xpress' ``writesol`` produces a header block followed by a column
    listing.  Across versions the exact column count varies, but the
    invariants we rely on are:

    * One ``Objective`` line of the form ``Objective <value>`` (or
      ``Objective function value: <value>``) somewhere near the top.
    * A ``Variables`` section, then per-variable rows of the form
      ``<index> <name> <value> ...``.

    This parser is intentionally lenient — if we can't pin down a value
    column we walk the row looking for the first finite float after the
    variable name.
    """
    if not path.is_file():
        return SolverStatus.OTHER, None, None, None
    objective: float | None = None
    primal: dict[str, float] = {}
    in_vars = False

    with path.open("r") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            low = line.lower()
            if objective is None and "objective" in low:
                m = re.search(r"objective.*?[:=]?\s*([-\d.eE+]+)", line)
                if m:
                    try:
                        objective = float(m.group(1))
                    except ValueError:
                        pass
                continue
            if low.startswith("variables") or low.startswith("columns"):
                in_vars = True
                continue
            if low.startswith("rows") or low.startswith("constraints"):
                in_vars = False
                continue
            if not in_vars:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            # Try to find the name (first non-numeric token) and the
            # first float after it.
            name = None
            for i, tok in enumerate(parts):
                try:
                    float(tok)
                    continue
                except ValueError:
                    name = tok
                    rest = parts[i + 1 :]
                    break
            if name is None:
                continue
            for tok in rest:
                try:
                    primal[name] = float(tok)
                    break
                except ValueError:
                    continue

    status = SolverStatus.OPTIMAL if primal else SolverStatus.OTHER
    return status, objective, (primal or None), None


_PARSERS = {
    "gurobi": _parse_gurobi_sol,
    "cplex": _parse_cplex_sol,
    "xpress": _parse_xpress_sol,
    "copt": _parse_copt_sol,
}


# ---------------------------------------------------------------------------
# License-error heuristic
# ---------------------------------------------------------------------------
# Substring patterns (case-insensitive) that, when found in stderr or
# stdout, signal a license problem rather than a model/solver error.
# Kept deliberately small — we only want very strong signals here, not a
# fuzzy keyword match.
_LICENSE_HINTS = (
    "license",
    "licence",  # UK spelling, appears in Xpress messages
    "no token",  # FlexLM / xpress
    "token server",
    "wls",  # gurobi web license service
)


def _looks_like_license_error(*texts: str) -> bool:
    blob = "\n".join(t for t in texts if t).lower()
    return any(h in blob for h in _LICENSE_HINTS)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def run_via_file(
    view: LpView,
    solver_name: str,
    io_api: IOMode,
    **options: Any,
) -> SolverResult:
    """Solve ``view`` by writing an MPS and shelling out to a CLI binary.

    Parameters
    ----------
    view
        Pre-built :class:`LpView`.  The dispatch in
        :mod:`polar_high.solvers` builds this once before calling.
    solver_name
        One of ``"gurobi"``, ``"cplex"``, ``"xpress"``, ``"copt"``.
        HiGHS is intentionally **not** supported via this path — see the
        note in :mod:`polar_high.solvers.__init__` and the
        ``ValueError`` raised below.
    io_api
        :data:`IOMode.MPS` is the only currently-supported value.
        :data:`IOMode.LP` raises :class:`NotImplementedError` per the
        Phase 4 "out of scope" decision.
    **options
        ``time_limit`` (seconds, ``float``) is forwarded to
        :func:`subprocess.run` as its timeout; other options are
        ignored in this phase (commercial solvers expose hundreds of
        knobs and the MPS-file path can't pass them in cleanly).

    Returns
    -------
    SolverResult
        Populated from the solver's ``.sol`` output.

    Raises
    ------
    NotImplementedError
        For ``io_api=IOMode.LP``.
    SolverError
        If the binary is not found, the subprocess exits non-zero, or
        the ``.sol`` file cannot be parsed.  ``SolverError`` is upgraded
        to :class:`LicenseError` when the captured stderr/stdout contains
        a strong license-related substring (see :data:`_LICENSE_HINTS`).
    """
    if io_api == IOMode.LP:
        raise NotImplementedError(
            "io_api='lp' is reserved but not implemented in Phase 4. "
            "Use io_api='mps' for the file-based fallback, or io_api='direct' "
            "for the in-memory adapter."
        )
    if io_api != IOMode.MPS:
        raise ValueError(f"_mps_fallback.run_via_file got unexpected io_api={io_api!r}")

    if solver_name == "highs":
        # HiGHS already has an excellent in-memory path; routing it
        # through a temp MPS would be a strict regression in correctness
        # (file-format round-trips lose names, can hit precision quirks)
        # and performance.  Refuse loudly rather than do something subtle.
        raise ValueError(
            "io_api='mps' is not supported for solver_name='highs'. "
            "HiGHS always uses the in-memory direct path; use "
            "io_api=IOMode.DIRECT (the default) for HiGHS."
        )

    if solver_name not in _BINARY_NAMES:
        raise ValueError(
            f"_mps_fallback.run_via_file does not know solver {solver_name!r}; "
            f"expected one of {sorted(_BINARY_NAMES)}."
        )

    binary = _find_solver_binary(solver_name)
    if binary is None:
        raise SolverError(
            f"{solver_name!r} CLI binary "
            f"({_BINARY_NAMES[solver_name]!r}) was not found on $PATH "
            f"or in the conventional install directories. Install the "
            f"solver and ensure its 'bin' directory is on $PATH."
        )

    time_limit = options.get("time_limit")

    with tempfile.TemporaryDirectory(prefix="polar_high_mps_") as tmpdir:
        tmp = Path(tmpdir)
        mps_path = tmp / "model.mps"
        sol_path = tmp / "model.sol"

        _write_mps(view, str(mps_path))

        stdin_text: str | None = None
        if solver_name == "gurobi":
            argv = _gurobi_script(binary, mps_path, sol_path)
        elif solver_name == "cplex":
            argv, stdin_text = _cplex_script(binary, mps_path, sol_path)
        elif solver_name == "xpress":
            argv, stdin_text = _xpress_script(binary, mps_path, sol_path)
        elif solver_name == "copt":
            argv, stdin_text = _copt_script(binary, mps_path, sol_path)
        else:  # pragma: no cover — guarded above.
            raise AssertionError(f"unreachable: {solver_name!r}")

        try:
            completed = subprocess.run(
                argv,
                input=stdin_text,
                capture_output=True,
                text=True,
                check=False,
                timeout=time_limit,
            )
        except subprocess.TimeoutExpired as exc:
            raise SolverError(
                f"{solver_name!r} CLI exceeded time_limit={time_limit!r}s. "
                f"Captured stderr: {exc.stderr!r}"
            ) from exc

        stdout = completed.stdout or ""
        stderr = completed.stderr or ""

        if completed.returncode != 0:
            msg = (
                f"{solver_name!r} CLI exited with returncode="
                f"{completed.returncode}.\n"
                f"--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
            )
            if _looks_like_license_error(stdout, stderr):
                raise LicenseError(msg)
            raise SolverError(msg)

        parser = _PARSERS[solver_name]
        status, objective, primal, dual = parser(sol_path)

        if status == SolverStatus.OTHER and not primal:
            # Subprocess succeeded (rc==0) but we couldn't read a
            # solution.  This is almost always a license issue that
            # gurobi_cl etc. surfaces on stdout while still exiting 0.
            if _looks_like_license_error(stdout, stderr):
                raise LicenseError(
                    f"{solver_name!r} CLI returned no usable solution "
                    f"and its output mentions licensing.\n"
                    f"--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
                )
            raise SolverError(
                f"{solver_name!r} CLI produced no solution file we could "
                f"parse at {sol_path}.\n--- stdout ---\n{stdout}\n"
                f"--- stderr ---\n{stderr}"
            )

        result = SolverResult(
            status=status,
            objective=objective,
            primal=primal,
            dual=dual,
            solver_name=solver_name,
            raw_status=completed.returncode,
        )
        return result


__all__ = ["run_via_file"]
