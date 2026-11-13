"""Objective-valued solver options, read in user units.

Some solver options are objective values: a target or bound that stops the
solve (HiGHS ``objective_target``, Gurobi ``BestObjStop``), a cutoff, or an
absolute gap.  The solver compares them with the objective it works on,
which is not always the objective the caller thinks in:

* the caller may build the objective scaled (``Problem.set_objective_scale``:
  the objective handed to polar-high is ``factor`` times the objective in
  user units, e.g. currency × 1e-6);
* HiGHS ``user_objective_scale = N`` multiplies the costs by ``2**N`` and
  compares ``objective_target`` / ``objective_bound`` / ``mip_abs_gap`` in
  that scaled space;
* the objective constant (``Problem.add_obj_constant``) may or may not reach
  the solver;
* HiGHS compares ``objective_bound`` in its internal minimisation space, so
  for a maximisation the bound has the opposite sign.

*User units* are the units of the objective as the caller thinks of it:
the objective given to polar-high divided by the objective scale ``k``.
This module maps a value the caller writes, in user units and in the sense
of the objective, to the value the solver needs.  Two kinds of option:

``"value"``
    An objective level (target, bound, cutoff, objective limit).  Affine:
    ``q = f * (k * v - O) + O_solver`` (negated for an option the solver
    reads in its minimisation space when the objective is maximised).
``"diff"``
    An objective difference (absolute gap, cutoff increment, objective
    range).  Factor only: ``q = f * k * g``.

``k`` is the objective scale, ``f`` the solver's own objective factor
(``2**N`` for HiGHS, 1 otherwise), ``O`` the objective constant and
``O_solver`` the constant the solver itself adds to its objective.  A value
that is not finite, or whose magnitude is ``1e20`` or more (every solver's
"infinity" or "off"), passes unchanged.  Relative gaps and tolerances are
not objective values and are never changed.  When the mapping is the
identity for an option (``k = f = 1``; for a level also ``O = O_solver`` and
no minimisation-space negation), its value passes untouched (same object,
no rounding).

Generic: no knowledge of any caller's model.  ``Problem.objective_transform``
builds the transform for a :class:`~polar_high.engine.Problem`.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple

#: Solvers whose objective-valued options this module knows.
OBJECTIVE_OPTION_SOLVERS: tuple[str, ...] = ("highs", "gurobi", "cplex", "xpress", "copt")

#: A value at or above this magnitude is the solver's "infinity" / "off"
#: (HiGHS ``inf``, CPLEX 1e75, Gurobi 1e100, Xpress 1e40) and passes
#: unchanged.
PASS_THROUGH_MAGNITUDE = 1e20

# The option names as each solver documents them.  Folded with
# :func:`option_key` to build :data:`OBJECTIVE_OPTIONS`.
_OPTION_TABLE: dict[str, dict[str, str]] = {
    # highspy 1.15 / 1.16: the only objective-valued options.
    "highs": {
        "objective_target": "value",
        "objective_bound": "value",
        "mip_abs_gap": "diff",
    },
    # Gurobi 13: PDHGAbsTol is a tolerance of the PDHG method, not an
    # objective value, and is left out.
    "gurobi": {
        "BestObjStop": "value",
        "BestBdStop": "value",
        "Cutoff": "value",
        "MIPGapAbs": "diff",
        "PoolGapAbs": "diff",
    },
    # CPLEX 22.1: ``barrier limits objrange`` is a magnitude (factor only).
    "cplex": {
        "mip tolerances lowercutoff": "value",
        "mip tolerances uppercutoff": "value",
        "mip limits lowerobjstop": "value",
        "mip limits upperobjstop": "value",
        "simplex limits lowerobj": "value",
        "simplex limits upperobj": "value",
        "mip tolerances absmipgap": "diff",
        "mip tolerances objdifference": "diff",
        "mip pool absgap": "diff",
        "mip polishafter absmipgap": "diff",
        "barrier limits objrange": "diff",
    },
    # Xpress 9.8: MIPABSGAPNOTIFY* only feed a callback and are left out.
    "xpress": {
        "MIPABSCUTOFF": "value",
        "MIPADDCUTOFF": "diff",
        "MIPABSSTOP": "diff",
    },
    # coptpy 8.0: no objective-level option.
    "copt": {
        "AbsGap": "diff",
    },
}

# Options a solver compares in its internal minimisation space whatever the
# objective sense (probe, highspy 1.15.1 and 1.16.0: ``max x``, 5 <= x <= 50
# integer, is infeasible under ``objective_bound=-51`` and optimal under
# ``-49``).  HiGHS ``objective_target`` is read in the objective's own sense.
_MIN_SPACE_OPTIONS: dict[str, frozenset[str]] = {
    "highs": frozenset({"objective_bound"}),
}


def option_key(solver: str, name: str) -> str:
    """The comparison key of an option name: one key for every spelling the
    solver reads as the same option.

    Every solver: surrounding white space stripped.  HiGHS: names are exact.
    Gurobi: case-folded, ``_`` after the first character removed (the
    library reads ``MIP_Gap_Abs`` as ``MIPGapAbs``).  CPLEX: case-folded,
    ``.`` and ``__`` read as a space (``mip.tolerances.absmipgap`` and the
    adapter's keyword form ``mip__tolerances__absmipgap`` are
    ``mip tolerances absmipgap``), runs of white space collapsed.  Xpress:
    case-folded, one leading ``xprs_`` removed.  COPT: case-folded.
    """
    key = str(name).strip()
    if solver == "highs":
        return key
    if solver == "cplex":
        key = key.replace("__", " ").replace(".", " ")
    key = " ".join(key.split()).casefold()
    if solver == "gurobi":
        key = key[:1] + key[1:].replace("_", "")
    elif solver == "xpress" and key.startswith("xprs_"):
        key = key[len("xprs_") :]
    return key


#: ``{solver: {option_key(solver, name): "value" | "diff"}}``.
OBJECTIVE_OPTIONS: dict[str, dict[str, Literal["value", "diff"]]] = {
    solver: {option_key(solver, name): kind for name, kind in table.items()}  # type: ignore[misc]
    for solver, table in _OPTION_TABLE.items()
}


def _check_solver(solver: str) -> str:
    if solver not in OBJECTIVE_OPTIONS:
        raise ValueError(
            f"unknown solver {solver!r} for objective options; "
            f"expected one of {', '.join(OBJECTIVE_OPTION_SOLVERS)}"
        )
    return solver


def objective_option_kind(solver: str, name: str) -> Literal["value", "diff"] | None:
    """``"value"`` (an objective level), ``"diff"`` (an objective difference)
    or ``None`` (not an objective-valued option) for option ``name`` of
    ``solver``."""
    return OBJECTIVE_OPTIONS[_check_solver(solver)].get(option_key(solver, name))


def _min_space(solver: str, name: str) -> bool:
    return option_key(solver, name) in _MIN_SPACE_OPTIONS.get(solver, frozenset())


def solver_keeps_offset(solver: str, io: str) -> bool:
    """Whether the solver itself adds the objective constant to the objective
    it compares the options with, for a solve through polar-high.

    ``io="mps"``: never (``Problem.write_mps`` and the file-based fallback do
    not write the constant).  ``io="direct"``: HiGHS never (polar-high keeps
    the constant out of HiGHS and adds it to ``Solution.obj``); Gurobi
    (``ObjCon``) and CPLEX (``set_offset``) yes; Xpress no (the adapter adds
    it after the solve); COPT yes in-process (``ObjConst``), no when the
    adapter routes the solve through the ``copt_cmd`` file path because
    ``highspy`` is loaded.
    """
    _check_solver(solver)
    if io not in ("direct", "mps"):
        raise ValueError(f"io must be 'direct' or 'mps'; got {io!r}")
    if io == "mps" or solver in ("highs", "xpress"):
        return False
    if solver == "copt":
        from ._copt import routes_via_file

        return not routes_via_file()
    return True


def _finite_user_value(v: float) -> bool:
    return math.isfinite(v) and abs(v) < PASS_THROUGH_MAGNITUDE


@dataclass(frozen=True)
class ObjectiveTransform:
    """How a value in user units maps to a solver's value.

    ``objective_scale`` (``k``): the objective given to polar-high is ``k``
    times the objective in user units.  ``solver_factor`` (``f``): the solver's own
    objective multiplier (HiGHS ``2**user_objective_scale``).  ``offset``
    (``O``): the objective constant in polar-high's units.
    ``solver_offset`` (``O_solver``): the constant the solver adds itself
    (``O`` or 0).  ``sense``: ``"min"`` or ``"max"``.
    """

    objective_scale: float = 1.0
    solver_factor: float = 1.0
    offset: float = 0.0
    solver_offset: float = 0.0
    sense: str = "min"

    def __post_init__(self) -> None:
        for name in ("objective_scale", "solver_factor"):
            val = getattr(self, name)
            if not (math.isfinite(val) and val > 0.0):
                raise ValueError(f"ObjectiveTransform.{name} must be finite and > 0; got {val!r}")
        if self.sense not in ("min", "max"):
            raise ValueError(f"sense must be 'min' or 'max'; got {self.sense!r}")

    def value(self, v: float, *, min_space: bool = False) -> float:
        """The solver value of objective level ``v`` (user units and
        sense).  ``min_space``: the solver reads the option in its internal
        minimisation space (negated under ``sense="max"``).  A non-finite
        ``v`` or ``|v| >= 1e20`` is returned unchanged."""
        v = float(v)
        if not _finite_user_value(v):
            return v
        q = self.solver_factor * (self.objective_scale * v - self.offset) + self.solver_offset
        return -q if (min_space and self.sense == "max") else q

    def diff(self, g: float) -> float:
        """The solver value of objective difference ``g`` (user units).
        A non-finite ``g`` or ``|g| >= 1e20`` is returned unchanged."""
        g = float(g)
        if not _finite_user_value(g):
            return g
        return self.solver_factor * self.objective_scale * g

    def is_identity(self, kind: str, *, min_space: bool = False) -> bool:
        """Whether the mapping of an option of ``kind`` (``"value"`` or
        ``"diff"``) is the identity: ``k = f = 1`` and, for a level, ``O =
        O_solver`` and no negation (``min_space`` under ``sense="max"``)."""
        if self.objective_scale != 1.0 or self.solver_factor != 1.0:
            return False
        if kind == "diff":
            return True
        return self.offset == self.solver_offset and not (min_space and self.sense == "max")


class Translated(NamedTuple):
    """One translated option: its name as given, its kind, the value the
    caller wrote and the value the solver gets (both as floats)."""

    name: str
    kind: str
    user_value: float
    solver_value: float


def _as_float(val: Any) -> float | None:
    """``val`` as a float, or ``None`` when it is not a number (booleans
    included) or a string that does not parse as one."""
    if isinstance(val, bool):
        return None
    if isinstance(val, numbers.Real):
        return float(val)
    if isinstance(val, str):
        try:
            return float(val.strip())
        except ValueError:
            return None
    return None


def _round15(x: float) -> float:
    """``x`` rounded to 15 significant digits, so a value the caller wrote
    exactly reaches the solver without binary residue (``1e5 * 1e-6`` is
    ``0.1``, not ``0.09999999999999999``)."""
    return float(f"{x:.15g}")


def _translate_one(solver: str, name: str, val: Any, transform: ObjectiveTransform):
    """``(new_value, Translated | None)`` for one option."""
    kind = objective_option_kind(solver, str(name))
    if kind is None:
        return val, None
    user = _as_float(val)
    if user is None or not _finite_user_value(user):
        return val, None
    min_space = _min_space(solver, str(name))
    if transform.is_identity(kind, min_space=min_space):
        # Nothing to map: the value reaches the solver exactly as written.
        return val, None
    if kind == "value":
        q = transform.value(user, min_space=min_space)
    else:
        q = transform.diff(user)
    q = _round15(q)
    new_val: Any = repr(q) if isinstance(val, str) else q
    return new_val, Translated(str(name), kind, user, q)


def has_objective_options(solver: str, options: Mapping | Iterable | None) -> bool:
    """Whether ``options`` (a dict, or a sequence of ``(name, value, ...)``
    items) holds an objective-valued option of ``solver``."""
    if not options:
        return False
    table = OBJECTIVE_OPTIONS[_check_solver(solver)]
    names = options.keys() if isinstance(options, Mapping) else (item[0] for item in options)
    return any(option_key(solver, str(n)) in table for n in names)


def translate_objective_options(
    solver: str,
    options: Mapping | Iterable | None,
    transform: ObjectiveTransform,
) -> tuple[Any, list[Translated]]:
    """Map the objective-valued options in ``options`` to ``solver``'s values.

    ``options`` is a dict (a new dict is returned, same order) or a sequence
    of ``(name, value, ...)`` tuples (a new list is returned; a named tuple
    keeps its type and its other fields).  The input is never changed.  A
    translated string value comes back as the ``repr`` of the float; a
    translated number comes back as a float.  Translated values are rounded
    to 15 significant digits.  Options that are not objective-valued, values
    that are not numbers, non-finite values, values with ``|v| >= 1e20``
    and options whose mapping is the identity
    (:meth:`ObjectiveTransform.is_identity`) pass unchanged.  The second
    element lists the translated options in order.
    """
    _check_solver(solver)
    if options is None:
        return None, []
    records: list[Translated] = []
    if isinstance(options, Mapping):
        out: dict = {}
        for name, val in options.items():
            new_val, rec = _translate_one(solver, name, val, transform)
            out[name] = new_val
            if rec is not None:
                records.append(rec)
        return out, records
    out_items: list = []
    for item in options:
        name, val = item[0], item[1]
        new_val, rec = _translate_one(solver, name, val, transform)
        if rec is None:
            out_items.append(item)
            continue
        records.append(rec)
        rest = tuple(item[2:])
        if hasattr(item, "_make"):
            out_items.append(item._make((name, new_val, *rest)))
        else:
            out_items.append((name, new_val, *rest))
    return out_items, records


__all__ = [
    "OBJECTIVE_OPTIONS",
    "OBJECTIVE_OPTION_SOLVERS",
    "PASS_THROUGH_MAGNITUDE",
    "ObjectiveTransform",
    "Translated",
    "has_objective_options",
    "objective_option_kind",
    "option_key",
    "solver_keeps_offset",
    "translate_objective_options",
]
