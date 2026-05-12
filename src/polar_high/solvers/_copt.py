"""COPT direct adapter for ``polar_high.solvers.solve``.

Phase 6 of ``specs/polar-high-multi-solver-implementation-plan.md``.

This adapter consumes a fully-extracted
:class:`~polar_high.solvers._lp_view.LpView` and pushes it into a fresh
``coptpy.Model`` in memory.  Vectorized load via
``scipy.sparse.csc_matrix`` + ``Model.addMVars`` + ``Model.addMConstrs``,
mirroring the Phase 5 Gurobi adapter — COPT's API is Gurobi-shaped, so
the structure is intentionally identical.

License model — bring-your-own
------------------------------
``polar_high`` never inspects, constructs, or validates a COPT license.
The caller may pass a pre-constructed ``coptpy.Envr`` via the ``env=``
kwarg (e.g. configured for a floating-licence server).  When
``env is None`` a fresh ``cp.Envr()`` is created, which picks up the
licence via COPT's own discovery (``COPT_LICENSE_DIR``, ``copt.lic``,
WLS tokens, etc.).

Any :class:`coptpy.CoptError` is caught and re-raised as either
:class:`~polar_high.solvers._base.LicenseError` (when the errno is in
the provisional license-code list OR the message contains a licence
keyword) or :class:`~polar_high.solvers._base.SolverError`
(everything else).  The raw vendor exception never reaches the caller.

Dependency note
---------------
``coptpy`` and ``scipy`` are *optional* — they are pulled in by the
``polar-high[copt]`` extra.  Both imports happen inside :func:`run`,
not at module load, so this module is always importable.  A missing
wrapper raises :class:`~polar_high.solvers._base.SolverNotAvailableError`
with an install pointer.

Out of scope (matching the plan)
--------------------------------
- Callbacks, lazy constraints, MIP starts, multi-objective
- Quadratic / SOCP / nonlinear models
- Solver-switch warm starts (every call is a fresh ``Model``)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._base import (
    LicenseError,
    SolverError,
    SolverNotAvailableError,
    SolverResult,
    SolverStatus,
)
from ._lp_view import LpView

# COPT license-error codes.  PROVISIONAL: the published ``coptpy``
# error-code table is not as cleanly documented as Gurobi's; the values
# below are the codes most commonly cited by users hitting licence
# problems (no license file, expired licence, token-server unreachable,
# WLS / floating-license check failed).  Anyone with a real COPT
# installation should tighten this list against ``coptpy``'s actual
# error-code table.  The string-keyword fallback below makes the
# precision of this list non-critical for UX — the keyword match catches
# anything the integer list misses.
_LICENSE_ERROR_CODES = frozenset(
    {
        # provisional — replace with verified values from coptpy docs
        2,  # COPT_RETCODE_LICENSE (generic licence-check failure)
        7,  # token / floating-licence acquisition failed
        8,  # licence expired
        9,  # licence host / hostid mismatch
    }
)

# Keywords matched (case-insensitive) against the ``CoptError`` message
# when the errno does not appear in ``_LICENSE_ERROR_CODES``.  Belt-and-
# braces because the errno table above is provisional.
_LICENSE_MESSAGE_TOKENS = ("license", "licence", "no token", "expired", "wls")

_INSTALL_HINT = (
    "Install the optional extra:  pip install 'polar-high[copt]'  "
    "(this pulls coptpy and scipy together)."
)


def run(
    view: LpView,
    *,
    env: Any = None,
    **options: Any,
) -> SolverResult:
    """Solve ``view`` with COPT via the in-memory direct API.

    Parameters
    ----------
    view
        A fully-extracted :class:`LpView`.  Built once by
        :func:`polar_high.solvers.solve`.
    env
        Optional pre-constructed ``coptpy.Envr``.  Pass-through only —
        ``polar_high`` does not inspect or construct it.  When ``None``,
        a fresh ``cp.Envr()`` is created and COPT performs its own
        licence discovery.
    **options
        Forwarded one-by-one to ``Model.setParam`` (e.g. ``TimeLimit=60``,
        ``RelGap=0.01``).  Unknown keys raise ``CoptError`` which is
        wrapped into :class:`SolverError`.

    Returns
    -------
    SolverResult
        ``status`` is mapped from ``Model.status``.  ``primal`` is keyed
        by variable name whenever a solution is available.  ``dual`` is
        populated for LP solves only (``Model.ismip`` falsy and a
        solution exists); ``None`` otherwise.  ``raw_status`` carries
        the integer COPT status code for debugging.

    Raises
    ------
    SolverNotAvailableError
        ``coptpy`` or ``scipy`` is not importable.
    LicenseError
        ``CoptError`` whose errno is in the provisional license-code
        list OR whose message contains a licence keyword.
    SolverError
        Any other ``CoptError`` from model construction or solve.
    """
    # ------------------------------------------------------------------
    # Lazy imports — keep the module importable without [copt] extra.
    # ------------------------------------------------------------------
    try:
        import coptpy as cp
        from coptpy import COPT
    except ImportError as exc:  # pragma: no cover — covered manually
        raise SolverNotAvailableError("coptpy is not installed.  " + _INSTALL_HINT) from exc

    try:
        import scipy.sparse as sp
    except ImportError as exc:  # pragma: no cover — covered manually
        raise SolverNotAvailableError(
            "scipy is not installed (required by the COPT adapter for "
            "vectorized matrix load).  " + _INSTALL_HINT
        ) from exc

    n_cols = int(view.n_cols)

    # We need access to ``CoptError`` for the except clause; resolve it
    # outside the try-body so the handler can reference it.
    CoptError = cp.CoptError

    # The env construction itself can raise a licence error (when the
    # default ``Envr()`` ctor probes the licence).  Wrap the whole
    # body in a single try so both paths funnel into the same handler.
    try:
        # --------------------------------------------------------------
        # Env / Model construction.  COPT differs from Gurobi here:
        # ``Envr`` is the env type, and the Model is created from the
        # env via ``env.createModel(name)``, not ``Model(env=env)``.
        # --------------------------------------------------------------
        if env is None:
            env = cp.Envr()
        m = env.createModel("polar_high")

        # --------------------------------------------------------------
        # Variable types.
        #
        # LpView.integrality is int8 (1 = integer, 0 = continuous) or
        # ``None`` for pure LP.  Same policy as the Gurobi adapter: no
        # binary shortcut — an integer var with bounds [0, 1] stays
        # typed integer, and we let COPT presolve detect the binary
        # form.
        # --------------------------------------------------------------
        if view.integrality is None:
            vtype: Any = COPT.CONTINUOUS
        else:
            vtype = np.where(view.integrality.astype(bool), COPT.INTEGER, COPT.CONTINUOUS)

        x = m.addMVars(
            n_cols,
            lb=view.col_lb,
            ub=view.col_ub,
            obj=view.col_obj,
            vtype=vtype,
        )

        # --------------------------------------------------------------
        # Variable names.  ``LpView.col_names`` may contain ``None``
        # entries (the engine doesn't always populate them); skip those.
        # --------------------------------------------------------------
        for i, nm in enumerate(view.col_names):
            if nm is not None:
                x[i].name = nm

        # --------------------------------------------------------------
        # Constraint matrix and RHS — same shape as Phase 5: split
        # ranged rows upstream, then a single vectorized ``addMConstrs``
        # call with a scipy CSC matrix.
        # --------------------------------------------------------------
        load_view = view.split_ranged_rows()
        load_n_rows = int(load_view.n_rows)

        a = sp.csc_matrix(
            (load_view.a_value, load_view.a_index, load_view.a_start),
            shape=(load_n_rows, n_cols),
        )

        senses_arr, rhs_arr, _range_arr = load_view.row_sense_rhs()
        sense_map = {"E": COPT.EQUAL, "L": COPT.LESS_EQUAL, "G": COPT.GREATER_EQUAL}
        sense_chars = np.array([sense_map[s] for s in senses_arr.tolist()], dtype=object)

        if load_n_rows > 0:
            constrs = m.addMConstrs(a, x, sense_chars, rhs_arr)
            for i, nm in enumerate(load_view.row_names):
                # row_names may contain empty strings on the lo/hi
                # halves of an originally anonymous ranged row.
                if nm:
                    constrs[i].name = nm

        # --------------------------------------------------------------
        # Objective.  ``addMVars(..., obj=...)`` already set the linear
        # coefficients; we just need the sense and any offset.  COPT
        # exposes the constant via the ObjConst parameter.
        # --------------------------------------------------------------
        m.setObjSense(COPT.MAXIMIZE if view.sense == "max" else COPT.MINIMIZE)
        if view.obj_offset:
            m.setParam(COPT.Param.ObjConst, float(view.obj_offset))

        # --------------------------------------------------------------
        # Per-call options.  Pass-through; unknown keys surface as
        # ``CoptError`` and are wrapped below.
        # --------------------------------------------------------------
        for k, v in options.items():
            m.setParam(k, v)

        # COPT's solve method is ``solve``, not ``optimize``.
        m.solve()

        # --------------------------------------------------------------
        # Result assembly.  Map ``Model.status`` to SolverStatus.
        # Matching the Gurobi adapter, INF_OR_UNBD collapses to
        # UNBOUNDED so callers see one consistent terminal status.
        # --------------------------------------------------------------
        status_map = {
            COPT.OPTIMAL: SolverStatus.OPTIMAL,
            COPT.INFEASIBLE: SolverStatus.INFEASIBLE,
            COPT.UNBOUNDED: SolverStatus.UNBOUNDED,
            COPT.INF_OR_UNBD: SolverStatus.UNBOUNDED,
            COPT.NUMERICAL: SolverStatus.OTHER,
            COPT.NODELIMIT: SolverStatus.OTHER,
            COPT.TIMEOUT: SolverStatus.TIME_LIMIT,
            COPT.UNFINISHED: SolverStatus.OTHER,
            COPT.INTERRUPTED: SolverStatus.INTERRUPTED,
        }
        status = status_map.get(m.status, SolverStatus.OTHER)

        # A solution is available when COPT reports OPTIMAL.  Other
        # statuses (TIMEOUT etc.) may also carry incumbents in MIP runs,
        # but COPT's standard guard is the ``hasmipsol`` /
        # ``haslpsol`` attribute set; for simplicity and to match the
        # Gurobi adapter's contract we treat OPTIMAL as the canonical
        # "solution present" signal.
        has_solution = m.status == COPT.OPTIMAL
        is_mip = bool(getattr(m, "ismip", 0))

        primal: dict[str, float] | None
        dual: dict[str, float] | None
        objective: float | None

        if has_solution:
            primal = {v.name: float(v.x) for v in m.getVars() if v.name}
            objective = float(m.objval)
            if is_mip:
                dual = None
            else:
                dual = {c.name: float(c.pi) for c in m.getConstrs() if c.name}
        else:
            primal = None
            dual = None
            objective = None

        return SolverResult(
            status=status,
            objective=objective,
            primal=primal,
            dual=dual,
            solver_name="copt",
            raw_status=m.status,
        )

    except CoptError as exc:
        errno = getattr(exc, "retcode", None)
        if errno is None:
            errno = getattr(exc, "errno", None)
        msg_lower = str(exc).lower()
        is_license = errno in _LICENSE_ERROR_CODES or any(
            tok in msg_lower for tok in _LICENSE_MESSAGE_TOKENS
        )
        if is_license:
            raise LicenseError(
                f"COPT license check failed (code {errno}): {exc}.  "
                "Place copt.lic in the COPT install dir, set "
                "COPT_LICENSE_DIR, or pass a configured coptpy.Envr "
                "via env=..."
            ) from exc
        raise SolverError(f"COPT error (code {errno}): {exc}") from exc


__all__ = ["run"]
