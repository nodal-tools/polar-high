"""Xpress direct adapter for ``polar_high.solvers.solve``.

Phase 8 of ``specs/polar-high-multi-solver-implementation-plan.md``.

This adapter consumes a fully-extracted
:class:`~polar_high.solvers._lp_view.LpView` and pushes it into a fresh
``xpress.problem()`` via the column-oriented ``loadproblem`` entry
point.  Unlike CPLEX's row-oriented ``SparsePair`` interface, Xpress'
``loadproblem`` accepts CSC arrays directly, so the LpView's
``a_start`` / ``a_index`` / ``a_value`` triple flows through with no
CSR conversion.

Ranged constraints — locked decision (Phase 3 + Phase 8 plan)
-------------------------------------------------------------
Xpress has a native range concept (``rowtype="R"`` + ``rng`` array);
we use it directly.  We do **not** call
:meth:`LpView.split_ranged_rows` here — that helper is reserved for
solvers without native range support.

**Empirically-verified range convention (Xpress 9.8.1)**:

* ``rhs[i]``  is the row's **upper** bound (``row_ub``).
* ``rng[i]``  is the row's width: ``abs(row_ub - row_lb)``.  Per the
  installed Xpress 9.8 docs string for ``loadproblem``, "the sign of
  the range value is ignored — the absolute value is used in all
  cases".  Lower bound is therefore ``rhs - abs(rng)`` = ``row_ub -
  (row_ub - row_lb)`` = ``row_lb``.

This was verified at adapter-development time by loading a single
ranged row ``5 <= x + y <= 10`` and confirming that both ``max x+y``
hit ``10`` and ``min x+y`` hit ``5``.  See
``tests/test_solver_xpress.py::test_xpress_ranged_constraint_native``.

License model — bring-your-own
------------------------------
``polar_high`` never inspects, constructs, or validates an Xpress
license.  The ``env`` parameter is *reserved for future use*: the
public ``xpress.problem()`` constructor today does not accept an
``env`` argument (unlike ``gurobipy.Model(env=...)``).  ``xpress.init``
takes a licence file path globally rather than a per-problem env, so
there is no natural object to pass through here.  If FICO exposes a
runtime context object in a future ``xpress`` release we will route it
through here without changing the call site.  For now ``env`` is
accepted (so the dispatch layer's pass-through contract holds) and
otherwise unused.

Any :class:`xpress.SolverError`, :class:`xpress.ModelError`, or
:class:`xpress.InterfaceError` is caught and re-raised as either
:class:`~polar_high.solvers._base.LicenseError` (when the message
contains a licence keyword) or
:class:`~polar_high.solvers._base.SolverError` (everything else).
The raw vendor exception never reaches the caller.

Options forwarding
------------------
Xpress controls are set via ``p.setControl(name, value)``.  Each
``key=value`` in ``**options`` is forwarded as
``p.setControl(key, value)``; unknown names raise the vendor
exception, which we then wrap as ``SolverError``.

Dependency note
---------------
``xpress`` is *optional* — it is pulled in by the
``polar-high[xpress]`` extra.  The import happens inside :func:`run`,
not at module load, so this module is always importable.  A missing
wrapper raises :class:`~polar_high.solvers._base.SolverNotAvailableError`
with an install pointer.

Out of scope (matching the plan)
--------------------------------
- Callbacks, lazy constraints, MIP starts, multi-objective
- Quadratic / SOCP / nonlinear models
- Solver-switch warm starts (every call is a fresh ``xpress.problem``)
"""

from __future__ import annotations

import math
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

# Keywords matched (case-insensitive) against the exception message to
# decide LicenseError vs SolverError.  Xpress doesn't expose a stable
# numeric error-code attribute on its Python exceptions in 9.x, so we
# rely entirely on the message text for license classification.
_LICENSE_MESSAGE_TOKENS = ("license", "licence", "oem", "expired", "no token")

_INSTALL_HINT = (
    "Install the optional extra:  pip install 'polar-high[xpress]'  "
    "(this pulls FICO's official xpress Python wrapper)."
)


def _to_xpress_inf(bounds: np.ndarray, xpr_inf: float) -> list[float]:
    """Map ``±math.inf`` entries in ``bounds`` to ``±xpress.infinity``.

    Xpress accepts ``xpress.infinity`` (``1e+20``) as its unbounded
    sentinel.  Raw ``math.inf`` propagates through but the docstring
    explicitly recommends the finite sentinel; we honour that.
    Finite bounds pass through unchanged.
    """
    out: list[float] = []
    for b in bounds.tolist():
        if math.isinf(b):
            out.append(xpr_inf if b > 0 else -xpr_inf)
        else:
            out.append(float(b))
    return out


def run(
    view: LpView,
    *,
    env: Any = None,
    **options: Any,
) -> SolverResult:
    """Solve ``view`` with Xpress via the in-memory direct API.

    Parameters
    ----------
    view
        A fully-extracted :class:`LpView`.  Built once by
        :func:`polar_high.solvers.solve`.
    env
        Reserved for future use — see the module docstring.  Currently
        ignored: ``xpress.problem()`` takes no ``env`` argument.
        Passing a non-None value is *not* an error.
    **options
        Forwarded as ``p.setControl(key, value)`` after model load.
        Unknown names raise :class:`SolverError`.

    Returns
    -------
    SolverResult
        ``status`` is mapped from ``p.attributes.lpstatus`` (LP) or
        ``p.attributes.mipstatus`` (MIP).  ``primal`` is keyed by
        variable name whenever a solution is available.  ``dual`` is
        populated for LP solves only; ``None`` for MIP or when no
        solution exists.  ``raw_status`` carries the integer status
        code for debugging.

    Raises
    ------
    SolverNotAvailableError
        ``xpress`` is not importable.
    LicenseError
        Any caught Xpress exception whose message contains a licence
        keyword (``"license"`` / ``"licence"`` / ``"oem"`` /
        ``"expired"`` / ``"no token"``).
    SolverError
        Any other vendor exception from model construction or solve.
    """
    # ------------------------------------------------------------------
    # Lazy import — keep the module importable without [xpress] extra.
    # ------------------------------------------------------------------
    try:
        import xpress
    except ImportError as exc:  # pragma: no cover — covered manually
        raise SolverNotAvailableError("xpress is not installed.  " + _INSTALL_HINT) from exc

    # ``env`` is reserved for a future Xpress runtime context object —
    # see module docstring.  Bind it locally to silence linters.
    _ = env

    n_cols = int(view.n_cols)
    n_rows = int(view.n_rows)
    xpr_inf = xpress.infinity

    # Collect every Xpress-side exception class we want to wrap.  Older
    # / newer xpress releases occasionally rename one of these, so we
    # probe defensively with getattr.
    _xpr_exc_classes = tuple(
        cls
        for cls in (
            getattr(xpress, "SolverError", None),
            getattr(xpress, "ModelError", None),
            getattr(xpress, "InterfaceError", None),
        )
        if cls is not None
    ) or (Exception,)

    try:
        # --------------------------------------------------------------
        # Model construction.
        # --------------------------------------------------------------
        p = xpress.problem()

        # Silence the solver log by default; callers can re-enable via
        # an options['outputlog'] = 1.  This matches the quiet defaults
        # the other direct adapters provide.
        try:
            p.setControl("outputlog", 0)
        except _xpr_exc_classes:  # pragma: no cover — best-effort
            pass

        # --------------------------------------------------------------
        # Translate LpView -> loadproblem arguments.
        # --------------------------------------------------------------
        senses_arr, rhs_arr, range_arr = view.row_sense_rhs()
        # ``row_sense_rhs`` produces CPLEX-convention range values
        # (``row_lb - row_ub``, i.e. negative).  Xpress' rng uses the
        # absolute width: see the module docstring for the empirical
        # verification.  ``np.abs`` handles both signs uniformly.
        rng_xpr = np.abs(range_arr).astype(np.float64).tolist()
        rowtype = senses_arr.tolist()  # list[str] of single-char senses
        rhs_list = rhs_arr.astype(np.float64).tolist()

        obj_list = view.col_obj.astype(np.float64).tolist()

        # CSC passthrough — loadproblem accepts start of length ncol+1
        # plus rowind/rowcoef when ``collen=None``.  Cast to plain
        # Python lists for maximum cross-version compatibility (xpress
        # 9.x accepts numpy too, but the deprecated loadproblem path
        # was originally validated against lists).
        start_list = view.a_start.astype(np.int64).tolist()
        rowind_list = view.a_index.astype(np.int64).tolist()
        rowcoef_list = view.a_value.astype(np.float64).tolist()

        lb_list = _to_xpress_inf(view.col_lb, xpr_inf)
        ub_list = _to_xpress_inf(view.col_ub, xpr_inf)

        # Column / row names: Xpress requires non-None entries.  Fill
        # gaps with synthetic placeholders (also a defensive belt-and-
        # braces guard against ``split_ranged_rows`` empty-string
        # fallbacks, even though we use native ranges here).
        col_names = [
            (nm if nm is not None and nm != "" else f"x{i}") for i, nm in enumerate(view.col_names)
        ]
        row_names: list[str] = []
        for i in range(n_rows):
            nm = view.row_names[i] if i < len(view.row_names) else ""
            row_names.append(nm if nm else f"c{i}")

        # Integer column types passed at load time.  ``loadproblem``'s
        # ``coltype`` argument is optional and applies only to the
        # listed entries via ``entind`` — but in practice xpress 9.x
        # also accepts a full-length coltype string covering every
        # column when ``entind=None``.  To stay portable, we instead
        # call ``p.chgcoltype`` after load with the indices of the
        # integer columns only.
        load_kwargs: dict[str, Any] = {
            "probname": "polar_high",
            "rowtype": rowtype,
            "rhs": rhs_list,
            "rng": rng_xpr if any(rt == "R" for rt in rowtype) else None,
            "objcoef": obj_list,
            "start": start_list,
            "collen": None,  # CSC: start has ncol+1 entries
            "rowind": rowind_list,
            "rowcoef": rowcoef_list,
            "lb": lb_list,
            "ub": ub_list,
            "colnames": col_names,
            "rownames": row_names,
        }
        p.loadproblem(**load_kwargs)

        # Objective sense.  ``chgobjsense`` is the canonical way to
        # flip the sense post-load; alternatively we could have
        # negated ``obj_list`` and adjusted the returned objective —
        # ``chgobjsense`` is cleaner and keeps the duals' sign
        # convention straightforward.
        if view.sense == "max":
            p.chgObjSense(xpress.maximize)
        else:
            p.chgObjSense(xpress.minimize)

        # Integrality.  ``chgcoltype`` takes (indices, types) pairs.
        if view.integrality is not None:
            int_idx = np.flatnonzero(view.integrality).astype(np.int64)
            if int_idx.size:
                p.chgColType(int_idx.tolist(), ["I"] * int(int_idx.size))

        # Objective offset.  Xpress 9.8.1 does not expose
        # ``setObjOffset`` on the problem object; we therefore apply
        # the offset after-the-fact when reading back the objective.
        obj_offset = float(view.obj_offset)

        # --------------------------------------------------------------
        # Per-call options.
        # --------------------------------------------------------------
        for k, v in options.items():
            p.setControl(k, v)

        # --------------------------------------------------------------
        # Solve.  ``solve`` and ``optimize`` are both present in
        # xpress 9.8; ``optimize`` is the modern entry point.  We pick
        # ``optimize`` for forward compatibility and document the
        # ``solve`` alias here for the curious reader.
        # --------------------------------------------------------------
        p.optimize()

        # --------------------------------------------------------------
        # Determine MIP-ness from the view (the engine never produces
        # quadratic / nonlinear models, so integrality is sufficient).
        # --------------------------------------------------------------
        is_mip = view.integrality is not None and bool(view.integrality.any())

        # --------------------------------------------------------------
        # Status mapping.
        # --------------------------------------------------------------
        if is_mip:
            raw = int(p.attributes.mipstatus)
            status = _MIP_STATUS_MAP.get(raw, SolverStatus.OTHER)
        else:
            raw = int(p.attributes.lpstatus)
            status = _LP_STATUS_MAP.get(raw, SolverStatus.OTHER)

        # --------------------------------------------------------------
        # Solution extraction.
        # --------------------------------------------------------------
        primal: dict[str, float] | None = None
        dual: dict[str, float] | None = None
        objective: float | None = None

        try:
            values = p.getSolution()
            if values is not None and len(values) == n_cols:
                primal = {col_names[i]: float(values[i]) for i in range(n_cols)}
                # Pick the right objective attribute based on MIP/LP.
                obj_attr = p.attributes.mipobjval if is_mip else p.attributes.lpobjval
                objective = float(obj_attr) + obj_offset
        except _xpr_exc_classes:
            primal = None
            objective = None

        if primal is not None and not is_mip and n_rows > 0:
            try:
                duals = p.getDuals()
                if duals is not None and len(duals) == n_rows:
                    dual = {row_names[i]: float(duals[i]) for i in range(n_rows)}
            except _xpr_exc_classes:
                dual = None

        return SolverResult(
            status=status,
            objective=objective,
            primal=primal,
            dual=dual,
            solver_name="xpress",
            raw_status=raw,
        )

    except _xpr_exc_classes as exc:
        msg_lower = str(exc).lower()
        is_license = any(tok in msg_lower for tok in _LICENSE_MESSAGE_TOKENS)
        if is_license:
            raise LicenseError(
                f"Xpress license check failed: {exc}.  "
                "Configure your Xpress licence (xpauth.xpr / "
                "XPRESS_LICENSE / XPAUTH_PATH env vars), or call "
                "xpress.init('<path-to-xpauth.xpr>') before "
                "dispatching.  The env= parameter is reserved for "
                "future Xpress runtime contexts."
            ) from exc
        raise SolverError(f"Xpress error: {exc}") from exc


# ---------------------------------------------------------------------------
# Status code maps.  Computed once at import time; keys are the integer
# values exposed by xpress.enums.{LPStatus,MIPStatus} in 9.6+ (also
# available as the legacy ``xpress.lp_optimal`` etc. attributes).
# ---------------------------------------------------------------------------
_LP_STATUS_MAP: dict[int, SolverStatus] = {
    0: SolverStatus.OTHER,  # UNSTARTED
    1: SolverStatus.OPTIMAL,  # OPTIMAL
    2: SolverStatus.INFEASIBLE,  # INFEAS
    3: SolverStatus.OTHER,  # CUTOFF
    4: SolverStatus.INTERRUPTED,  # UNFINISHED
    5: SolverStatus.UNBOUNDED,  # UNBOUNDED
    6: SolverStatus.OTHER,  # CUTOFF_IN_DUAL
    7: SolverStatus.OTHER,  # UNSOLVED
    8: SolverStatus.OTHER,  # NONCONVEX
}

_MIP_STATUS_MAP: dict[int, SolverStatus] = {
    0: SolverStatus.OTHER,  # NOT_LOADED
    1: SolverStatus.OTHER,  # LP_NOT_OPTIMAL
    2: SolverStatus.OTHER,  # LP_OPTIMAL (root LP done, MIP not started)
    3: SolverStatus.INTERRUPTED,  # NO_SOL_FOUND (search stopped, no feas)
    4: SolverStatus.INTERRUPTED,  # SOLUTION (search stopped, feasible)
    5: SolverStatus.INFEASIBLE,  # INFEAS
    6: SolverStatus.OPTIMAL,  # OPTIMAL
    7: SolverStatus.UNBOUNDED,  # UNBOUNDED
}


__all__ = ["run"]
