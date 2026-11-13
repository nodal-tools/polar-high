"""Objective-valued solver options in user units.

Public home of :mod:`polar_high.solvers._objective_options`; see there for
the rules.  Typical use for a solver polar-high does not drive itself (e.g.
a commercial solver reading a file from ``Problem.write_mps``)::

    from polar_high.objective_options import translate_objective_options

    transform = problem.objective_transform("gurobi", "mps")
    options, translated = translate_objective_options("gurobi", options, transform)
"""

from polar_high.solvers._objective_options import (
    OBJECTIVE_OPTION_SOLVERS,
    OBJECTIVE_OPTIONS,
    PASS_THROUGH_MAGNITUDE,
    ObjectiveTransform,
    Translated,
    has_objective_options,
    objective_option_kind,
    option_key,
    solver_keeps_offset,
    translate_objective_options,
)

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
