"""CPLEX direct adapter for ``polar_high.solvers.solve``.

Phase 7 of ``specs/polar-high-multi-solver-implementation-plan.md``.

This adapter consumes a fully-extracted
:class:`~polar_high.solvers._lp_view.LpView` and pushes it into a fresh
``cplex.Cplex`` instance in memory.  CPLEX's Python API is meaningfully
different from Gurobi/COPT: bulkier, sense+rhs based, and row-oriented
(``SparsePair`` per row).  We therefore feed it via
:meth:`LpView.to_csr` (row-major) and
:meth:`LpView.row_sense_rhs` (CPLEX-style senses + rhs + range_values).

Ranged constraints — locked decision (Phase 3 + Phase 7 plan)
-------------------------------------------------------------
CPLEX has a native range concept (``senses="R"`` with ``range_values``);
we use it directly.  We do **not** call
:meth:`LpView.split_ranged_rows` here — that helper is reserved for
solvers without native range support.

License model — bring-your-own
------------------------------
``polar_high`` never inspects, constructs, or validates a CPLEX
license.  The ``env`` parameter is *reserved for future use*: the public
``cplex.Cplex()`` constructor today does not accept an ``env`` argument
(unlike ``gurobipy.Model(env=...)``).  If IBM exposes a subscription /
runtime context object in a future ``cplex`` release, we will route it
through here without changing the call site.  For now ``env`` is
accepted (so the dispatch layer's pass-through contract holds) and
otherwise unused.

Any :class:`cplex.exceptions.CplexSolverError` or
:class:`cplex.exceptions.CplexError` is caught and re-raised as either
:class:`~polar_high.solvers._base.LicenseError` (when the numeric code
is in the provisional license-code list, or the message contains a
licence keyword) or :class:`~polar_high.solvers._base.SolverError`
(everything else).  The raw vendor exception never reaches the caller.

Options forwarding
------------------
CPLEX parameters live under a *nested* attribute namespace
(``c.parameters.<group>.<sub>...<name>``).  There is no single
``setParam(string, value)`` like Gurobi.  This adapter therefore
expects callers to pass option keys as **dotted parameter paths**
relative to ``c.parameters``:

* ``timelimit=60``                          → ``c.parameters.timelimit``
* ``mip__tolerances__mipgap=0.01``          → ``c.parameters.mip.tolerances.mipgap``
* ``"mip.tolerances.mipgap"`` as kwargs name — not a valid Python
  identifier, so use the double-underscore convention above, or pass
  options via ``**{"mip.tolerances.mipgap": 0.01}`` style.

Either ``a.b.c`` or ``a__b__c`` is accepted; we normalise.

Dependency note
---------------
``cplex`` is *optional* — it is pulled in by the
``polar-high[cplex]`` extra.  The import happens inside :func:`run`,
not at module load, so this module is always importable.  A missing
wrapper raises :class:`~polar_high.solvers._base.SolverNotAvailableError`
with an install pointer.

Out of scope (matching the plan)
--------------------------------
- Callbacks, lazy constraints, MIP starts, multi-objective
- Quadratic / SOCP / nonlinear models
- Solver-switch warm starts (every call is a fresh ``cplex.Cplex``)
"""

from __future__ import annotations

import math
from typing import Any

from ._base import (
    LicenseError,
    SolverError,
    SolverNotAvailableError,
    SolverResult,
    SolverStatus,
)
from ._lp_view import LpView

# CPLEX license-error codes — PROVISIONAL.
# Documented values in IBM ILOG CPLEX:
#   1016   CPXERR_NO_LICENSE        — no licence found / readable
#   32024  ILM / CPLEX subscription error (CPXERR_LICENSE_*)
# In the field, other ILM-family codes (32000..32099) also surface as
# licence failures.  We list the two values the spec calls out by name
# and rely on the keyword fallback below to catch anything else.  Anyone
# with a real CPLEX installation should tighten this list against the
# live ``cplex.exceptions`` error-code table.
_LICENSE_ERROR_CODES = frozenset(
    {
        1016,  # CPXERR_NO_LICENSE
        32024,  # ILM / subscription failure
    }
)

# Keywords matched (case-insensitive) against the exception message when
# the numeric code does not appear in ``_LICENSE_ERROR_CODES``.
_LICENSE_MESSAGE_TOKENS = ("license", "licence", "ilm", "no license", "no licence")

_INSTALL_HINT = (
    "Install the optional extra:  pip install 'polar-high[cplex]'  "
    "(this pulls IBM's official cplex Python wrapper)."
)


def _to_cplex_inf(bound: float, cplex_inf: float) -> float:
    """Map ``±math.inf`` in an LpView bound to ``±cplex.infinity``.

    CPLEX rejects raw ``math.inf`` in ``variables.add``; it wants its own
    ``cplex.infinity`` sentinel (a large finite double).  Finite bounds
    pass through unchanged.
    """
    if math.isinf(bound):
        return cplex_inf if bound > 0 else -cplex_inf
    return float(bound)


def _resolve_param(parameters_root: Any, key: str) -> Any:
    """Walk a dotted parameter path under ``c.parameters`` and return the leaf.

    ``key`` may use ``.`` or ``__`` as the separator (Python kwargs cannot
    contain a dot, so the double-underscore form is the practical input).
    Raises :class:`SolverError` if any segment does not resolve.
    """
    normalised = key.replace("__", ".")
    node = parameters_root
    for segment in normalised.split("."):
        try:
            node = getattr(node, segment)
        except AttributeError as exc:
            raise SolverError(
                f"CPLEX option {key!r} does not resolve at "
                f"segment {segment!r}: no such parameter under "
                "c.parameters.  Use dotted paths like 'timelimit' or "
                "'mip__tolerances__mipgap' (== c.parameters.mip.tolerances.mipgap)."
            ) from exc
    return node


def run(
    view: LpView,
    *,
    env: Any = None,
    **options: Any,
) -> SolverResult:
    """Solve ``view`` with CPLEX via the in-memory direct API.

    Parameters
    ----------
    view
        A fully-extracted :class:`LpView`.  Built once by
        :func:`polar_high.solvers.solve`.
    env
        Reserved for future use — see the module docstring.  Currently
        ignored: ``cplex.Cplex()`` takes no ``env`` argument.  Passing a
        non-None value is *not* an error (the dispatch layer must be
        able to forward ``env`` to any adapter without inspection).
    **options
        Dotted CPLEX parameter paths.  Each ``key=value`` resolves to
        ``c.parameters.<dotted.path>.set(value)``.  Use ``__`` in kwargs
        keys (e.g. ``mip__tolerances__mipgap=0.01``) or pass via
        ``**{"mip.tolerances.mipgap": 0.01}``.  Unknown keys raise
        :class:`SolverError`.

    Returns
    -------
    SolverResult
        ``status`` is mapped from ``c.solution.get_status()``.
        ``primal`` is keyed by variable name whenever a solution is
        available.  ``dual`` is populated for LP solves only;
        ``None`` for MIP or when no solution exists.  ``raw_status``
        carries the integer CPLEX status code for debugging.

    Raises
    ------
    SolverNotAvailableError
        ``cplex`` is not importable.
    LicenseError
        ``CplexSolverError`` / ``CplexError`` whose code is in the
        provisional license-code list OR whose message contains a
        licence keyword.
    SolverError
        Any other vendor exception from model construction or solve.
    """
    # ------------------------------------------------------------------
    # Lazy imports — keep the module importable without [cplex] extra.
    # ------------------------------------------------------------------
    try:
        import cplex
        from cplex.exceptions import CplexError, CplexSolverError
    except ImportError as exc:  # pragma: no cover — covered manually
        raise SolverNotAvailableError("cplex is not installed.  " + _INSTALL_HINT) from exc

    # ``env`` is reserved for a future CPLEX subscription / runtime
    # context object — see module docstring.  We do NOT inspect it.  The
    # local reference below is purely to document intent and silence
    # "unused argument" linters.
    _ = env

    n_rows = int(view.n_rows)
    cplex_inf = cplex.infinity

    try:
        # --------------------------------------------------------------
        # Model construction.  CPLEX takes no env argument today.
        # --------------------------------------------------------------
        c = cplex.Cplex()

        # --------------------------------------------------------------
        # Objective sense + offset.
        # --------------------------------------------------------------
        c.objective.set_sense(
            c.objective.sense.maximize if view.sense == "max" else c.objective.sense.minimize
        )
        if view.obj_offset:
            c.objective.set_offset(float(view.obj_offset))

        # --------------------------------------------------------------
        # Variables.  CPLEX expects plain Python lists, not numpy
        # arrays, and uses its own ``cplex.infinity`` sentinel for
        # unbounded entries.
        # --------------------------------------------------------------
        lb_list = [_to_cplex_inf(float(b), cplex_inf) for b in view.col_lb.tolist()]
        ub_list = [_to_cplex_inf(float(b), cplex_inf) for b in view.col_ub.tolist()]
        obj_list = [float(o) for o in view.col_obj.tolist()]

        # ``view.col_names`` may contain None entries (the engine doesn't
        # always populate them).  CPLEX requires non-None names, so fill
        # the gaps with synthetic ``x{i}`` placeholders.
        col_names = [(nm if nm is not None else f"x{i}") for i, nm in enumerate(view.col_names)]

        add_kwargs: dict[str, Any] = {
            "lb": lb_list,
            "ub": ub_list,
            "obj": obj_list,
            "names": col_names,
        }
        if view.integrality is not None:
            # CPLEX vartypes: 'C' continuous, 'I' integer, 'B' binary.
            # Same policy as the Gurobi/COPT adapters: no binary
            # shortcut — let CPLEX presolve detect the [0,1] form.
            add_kwargs["types"] = "".join("I" if int(v) else "C" for v in view.integrality.tolist())

        c.variables.add(**add_kwargs)

        # --------------------------------------------------------------
        # Constraints.  CPLEX wants row-oriented input:
        # ``lin_expr=[SparsePair(ind=..., val=...) for each row]``,
        # plus a senses string, an rhs list, and (for ranged rows) a
        # ``range_values`` list using CPLEX's own convention.
        # --------------------------------------------------------------
        if n_rows > 0:
            row_start, row_index, row_value = view.to_csr()
            lin_expr = [
                cplex.SparsePair(
                    ind=row_index[row_start[i] : row_start[i + 1]].tolist(),
                    val=row_value[row_start[i] : row_start[i + 1]].tolist(),
                )
                for i in range(n_rows)
            ]

            senses_arr, rhs_arr, range_arr = view.row_sense_rhs()
            senses_str = "".join(senses_arr.tolist())
            rhs_list = [float(r) for r in rhs_arr.tolist()]
            range_list = [float(r) for r in range_arr.tolist()]

            row_names = [(nm if nm else f"c{i}") for i, nm in enumerate(view.row_names)]
            # In rare cases row_names may be shorter than n_rows; pad.
            while len(row_names) < n_rows:
                row_names.append(f"c{len(row_names)}")

            c.linear_constraints.add(
                lin_expr=lin_expr,
                senses=senses_str,
                rhs=rhs_list,
                range_values=range_list,
                names=row_names,
            )

        # --------------------------------------------------------------
        # Per-call options.  Walk dotted paths under c.parameters.
        # --------------------------------------------------------------
        for k, v in options.items():
            param = _resolve_param(c.parameters, k)
            try:
                param.set(v)
            except AttributeError as exc:
                raise SolverError(
                    f"CPLEX option {k!r} resolved but has no .set(value) — "
                    "the dotted path did not reach a leaf parameter."
                ) from exc

        # --------------------------------------------------------------
        # Solve.
        # --------------------------------------------------------------
        c.solve()

        # --------------------------------------------------------------
        # Status mapping.  ``c.solution.get_status()`` returns an int;
        # the symbolic constants live on ``c.solution.status``.
        # --------------------------------------------------------------
        # Symbolic status constants live on ``c.solution.status``.  The
        # ``getattr(..., None)`` dance below is defensive: the CPLEX
        # version-to-version namespace has shifted attribute names over
        # the years (e.g. ``MIP_abort_time_limit`` -> ``MIP_time_limit_*``);
        # querying with ``getattr`` keeps the adapter forward-compatible.
        sol_status = c.solution.status
        raw = c.solution.get_status()

        def _maybe(name: str) -> int | None:
            return getattr(sol_status, name, None)

        status_map: dict[int, SolverStatus] = {}
        for sname, mapped in (
            ("optimal", SolverStatus.OPTIMAL),
            ("optimal_tolerance", SolverStatus.OPTIMAL),
            ("MIP_optimal", SolverStatus.OPTIMAL),
            ("MIP_optimal_infeasible", SolverStatus.OPTIMAL),
            ("infeasible", SolverStatus.INFEASIBLE),
            ("MIP_infeasible", SolverStatus.INFEASIBLE),
            ("unbounded", SolverStatus.UNBOUNDED),
            ("MIP_unbounded", SolverStatus.UNBOUNDED),
            ("infeasible_or_unbounded", SolverStatus.UNBOUNDED),
            ("MIP_infeasible_or_unbounded", SolverStatus.UNBOUNDED),
            ("abort_time_limit", SolverStatus.TIME_LIMIT),
            ("abort_dettime_limit", SolverStatus.TIME_LIMIT),
            ("MIP_time_limit_feasible", SolverStatus.TIME_LIMIT),
            ("MIP_time_limit_infeasible", SolverStatus.TIME_LIMIT),
            ("MIP_dettime_limit_feasible", SolverStatus.TIME_LIMIT),
            ("MIP_dettime_limit_infeasible", SolverStatus.TIME_LIMIT),
            ("abort_user", SolverStatus.INTERRUPTED),
            ("MIP_abort_feasible", SolverStatus.INTERRUPTED),
            ("MIP_abort_infeasible", SolverStatus.INTERRUPTED),
        ):
            code = _maybe(sname)
            if code is not None:
                status_map[int(code)] = mapped
        status = status_map.get(int(raw), SolverStatus.OTHER)

        # --------------------------------------------------------------
        # Solution extraction.  Guard with try/except around the values
        # call: when CPLEX has not produced a solution at all, the
        # accessors raise.
        # --------------------------------------------------------------
        primal: dict[str, float] | None = None
        dual: dict[str, float] | None = None
        objective: float | None = None

        try:
            values = c.solution.get_values()
            objective = float(c.solution.get_objective_value())
            primal = {name: float(val) for name, val in zip(col_names, values)}
        except CplexSolverError:
            primal = None
            objective = None

        # Determine MIP-ness via problem_type (an integer index into
        # ``c.problem_type``'s reverse map).  Continuous LP duals only.
        is_mip = False
        try:
            ptype_id = c.get_problem_type()
            ptype_name = c.problem_type[ptype_id]
            is_mip = ptype_name in {"MILP", "MIQP", "MIQCP"}
        except CplexError:
            # If we can't read the problem type, fall back to whether
            # the view declared integrality.
            is_mip = view.integrality is not None

        if primal is not None and not is_mip and n_rows > 0:
            try:
                duals = c.solution.get_dual_values()
                # Re-derive the row names we registered earlier; they
                # always exist after the linear_constraints.add call.
                names_for_dual = c.linear_constraints.get_names()
                dual = {name: float(val) for name, val in zip(names_for_dual, duals)}
            except CplexSolverError:
                dual = None

        return SolverResult(
            status=status,
            objective=objective,
            primal=primal,
            dual=dual,
            solver_name="cplex",
            raw_status=raw,
        )

    except (CplexSolverError, CplexError) as exc:
        code = _extract_cplex_code(exc)
        msg_lower = str(exc).lower()
        is_license = code in _LICENSE_ERROR_CODES or any(
            tok in msg_lower for tok in _LICENSE_MESSAGE_TOKENS
        )
        if is_license:
            raise LicenseError(
                f"CPLEX license check failed (code {code}): {exc}.  "
                "Place access.ilm next to the CPLEX install, set the "
                "CPLEX_STUDIO_LICENSE / ILOG_LICENSE_FILE env var, or "
                "configure your subscription credentials.  The env= "
                "parameter is reserved for future CPLEX subscription "
                "contexts."
            ) from exc
        raise SolverError(f"CPLEX error (code {code}): {exc}") from exc


def _extract_cplex_code(exc: BaseException) -> int | None:
    """Best-effort numeric code extractor for CPLEX exceptions.

    ``CplexSolverError`` historically exposes the code via
    ``args[2]`` (the third positional argument); newer versions also
    set an ``args[0]`` int.  We probe both, falling back to ``None``.
    """
    args = getattr(exc, "args", ())
    # Common shape: (status_str, env, code).
    if len(args) >= 3 and isinstance(args[2], int):
        return args[2]
    if args and isinstance(args[0], int):
        return args[0]
    return None


__all__ = ["run"]
