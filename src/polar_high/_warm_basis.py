"""Warm-start / basis-injection primitives (spec §4.1, §4.5).

Generic, flextool-free.  A :class:`NamedBasis` carries a HiGHS basis
keyed by *rendered LP names* (``"v_flow[wind,3]"`` for columns,
``"node_balance[3]"`` for rows), plus helpers to fingerprint a
name-set (:func:`basis_fingerprint`) and to materialise a HiGHS basis
sized to a *specific* model from a name-keyed carrier
(:func:`build_highs_basis`).

Landmine A (spec): basis statuses are **scale-immune** — a variable
sits at its lower bound / upper bound / basic regardless of any column
or row scaling factor applied to the LP.  So there is **no arithmetic**
in this module; it only copies / permutes integer status codes by name.
Do not add scaling here.

Statuses are stored as the *integer* value of
``highspy.HighsBasisStatus`` (kLower=0, kBasic=1, kUpper=2, kZero=3,
kNonbasic=4 on highspy 1.14) so a carrier is portable across enum
re-imports and processes (e.g. the ``save_memory`` subprocess path).
This module never imports ``highspy`` at top level — the caller passes
the ``HighsBasis`` / ``HighsBasisStatus`` classes in.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass

# Format-version tag baked into every fingerprint payload.  Bumping this
# string changes every fingerprint, invalidating carriers persisted
# under an older name-set layout (spec §4.5).
_FINGERPRINT_VERSION = "phbasis-v1"

# Synthetic positional row names (spec §4.5): a constraint row emitted
# without a caller name is rendered as ``row_<i>`` where ``i`` is its
# positional index.  That index is NOT stable across models, so these
# names are excluded both from fingerprints and from name-keyed
# transfer.
_SYNTHETIC_ROW_RE = re.compile(r"^row_\d+$")


def is_synthetic_row(name: str) -> bool:
    """True for a positional ``row_<i>`` name (spec §4.5)."""
    return _SYNTHETIC_ROW_RE.match(name) is not None


@dataclass(frozen=True)
class NamedBasis:
    """A HiGHS basis keyed by rendered LP names (spec §4.1).

    ``col_status`` / ``row_status`` map a rendered name to the integer
    value of a ``highspy.HighsBasisStatus`` member.  ``fingerprint`` is
    the :func:`basis_fingerprint` of the name-set this basis came from;
    the set-side hook compares it against the target model's fingerprint
    to decide whether an ``"exact"`` transfer is legal.
    """

    col_status: dict[str, int]
    row_status: dict[str, int]
    fingerprint: str


def status_from_int(value: int, HighsBasisStatus):
    """Reconstruct a ``HighsBasisStatus`` enum member from its int value.

    The carrier stores plain ints for portability; a HiGHS basis needs
    the enum members.  ``highspy``'s pybind11 enum constructs directly
    from its integer value.
    """
    return HighsBasisStatus(int(value))


def basis_fingerprint(
    col_names,
    row_names,
    *,
    drop_synthetic_rows: bool = True,
) -> str:
    """Cheap, order-independent fingerprint of an LP's name-set (spec §4.5).

    The digest is taken over ``(sorted(set(col_names)),
    sorted(set(row_names)))`` prefixed with a format-version tag, so it
    is:

      * identical for permuted input orders (we sort the sets), and
      * different whenever the *set* of names differs.

    Synthetic ``row_<i>`` names are dropped before hashing when
    ``drop_synthetic_rows`` is True (the default) — they are positional,
    not stable, and must not perturb the key.  The set-side and get-side
    must pass the same flag so their fingerprints line up.
    """
    cols = sorted(set(col_names))
    if drop_synthetic_rows:
        rows = sorted({r for r in row_names if not is_synthetic_row(r)})
    else:
        rows = sorted(set(row_names))
    h = hashlib.sha256()
    h.update(_FINGERPRINT_VERSION.encode("utf-8"))
    h.update(b"\x00cols\x00")
    for c in cols:
        h.update(c.encode("utf-8"))
        h.update(b"\x00")
    h.update(b"\x00rows\x00")
    for r in rows:
        h.update(r.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def build_highs_basis(
    nb: NamedBasis,
    policy: str,
    *,
    col_names,
    row_names,
    col_lb,
    col_ub,
    HighsBasis,
    HighsBasisStatus,
) -> tuple[object, dict]:
    """Materialise a ``HighsBasis`` for THIS model from a name-keyed carrier.

    ``col_names`` / ``row_names`` are the target model's rendered names
    in LP (column / row) order; ``col_lb`` / ``col_ub`` are its
    per-column bounds in the same order (``±inf`` for unbounded).  The
    ``HighsBasis`` / ``HighsBasisStatus`` highspy classes are passed in
    so this module needs no top-level highspy import.

    Returns ``(basis, stats)``.  ``stats`` counts (for logging):
    ``n_cols_matched``, ``n_cols_defaulted``, ``n_rows_matched``,
    ``n_rows_defaulted``, ``n_sanitized``, ``policy``, ``alien``.

    Policies (spec §4.1):

    * ``"exact"`` — the caller has already confirmed the fingerprints
      match, so every non-synthetic name is present in ``nb``.  Map 1:1
      by name and set ``basis.alien = False``.  A missing non-synthetic
      target col/row name raises :class:`ValueError` (the set-side hook
      catches it and falls back to a cold solve) — we never silently
      mis-size the arrays.  Synthetic ``row_<i>`` rows are NOT in ``nb``
      by construction (they are dropped on extraction); they default to
      ``kBasic`` and are not treated as missing.

    * ``"alien"`` — map shared names by name, set ``basis.alien = True``,
      and let HiGHS repair the rest.  Target columns not in ``nb`` default
      to nonbasic at the nearer finite bound (``kLower`` if ``lower`` is
      finite and (``upper`` infinite or ``|lower| <= |upper|``); elif
      ``upper`` finite → ``kUpper``; else free → ``kNonbasic``/``kZero``).
      Target rows not in ``nb`` (or synthetic) default to ``kBasic``.

    Bound-finiteness sanitation (BOTH policies, every transferred
    *nonbasic column* status): a ``kLower`` needs a finite target
    ``lower``; a ``kUpper`` needs a finite target ``upper``.  If the
    transferred status names a bound that is infinite in THIS model,
    demote it to a legal status (``kLower`` → ``kUpper`` if upper finite
    else free; ``kUpper`` → ``kLower`` if lower finite else free) and
    count it in ``n_sanitized``.  ``kBasic`` / ``kZero`` / ``kNonbasic``
    need no sanitation.  Landmine A: statuses are scale-immune, so this
    is purely a legality repair, never arithmetic.

    Both policies emit full-length status arrays (one per target column /
    row): ``setBasis`` with ``alien=False`` ``kError``s on a wrong count.
    """
    if policy not in ("exact", "alien"):
        raise ValueError(f"policy must be 'exact' or 'alien', got {policy!r}")

    col_names = list(col_names)
    row_names = list(row_names)
    col_lb = list(col_lb)
    col_ub = list(col_ub)
    if not (len(col_names) == len(col_lb) == len(col_ub)):
        raise ValueError(
            "col_names, col_lb and col_ub must be equal length "
            f"({len(col_names)}, {len(col_lb)}, {len(col_ub)})"
        )

    s_lower = int(HighsBasisStatus.kLower)
    s_upper = int(HighsBasisStatus.kUpper)
    s_basic = int(HighsBasisStatus.kBasic)
    # Free nonbasic status — prefer kNonbasic, fall back to kZero if a
    # future highspy drops the member.
    s_free = int(getattr(HighsBasisStatus, "kNonbasic", HighsBasisStatus.kZero))

    stats = {
        "n_cols_matched": 0,
        "n_cols_defaulted": 0,
        "n_rows_matched": 0,
        "n_rows_defaulted": 0,
        "n_sanitized": 0,
        "policy": policy,
        "alien": policy == "alien",
    }

    def _default_col(lb: float, ub: float) -> int:
        """Nonbasic at the nearer finite bound (alien-missing column)."""
        if math.isfinite(lb) and (not math.isfinite(ub) or abs(lb) <= abs(ub)):
            return s_lower
        if math.isfinite(ub):
            return s_upper
        return s_free

    def _sanitize_col(status: int, lb: float, ub: float) -> int:
        """Demote a nonbasic status that names an infinite target bound."""
        if status == s_lower and not math.isfinite(lb):
            stats["n_sanitized"] += 1
            return s_upper if math.isfinite(ub) else s_free
        if status == s_upper and not math.isfinite(ub):
            stats["n_sanitized"] += 1
            return s_lower if math.isfinite(lb) else s_free
        return status

    missing_cols: list[str] = []
    col_status_out: list[int] = []
    for name, lb, ub in zip(col_names, col_lb, col_ub):
        if name in nb.col_status:
            col_status_out.append(_sanitize_col(nb.col_status[name], float(lb), float(ub)))
            stats["n_cols_matched"] += 1
        elif policy == "exact":
            missing_cols.append(name)
            col_status_out.append(s_basic)  # placeholder; we raise below
        else:
            col_status_out.append(_default_col(float(lb), float(ub)))
            stats["n_cols_defaulted"] += 1

    missing_rows: list[str] = []
    row_status_out: list[int] = []
    for name in row_names:
        synthetic = is_synthetic_row(name)
        if not synthetic and name in nb.row_status:
            row_status_out.append(nb.row_status[name])
            stats["n_rows_matched"] += 1
        elif policy == "exact" and not synthetic:
            missing_rows.append(name)
            row_status_out.append(s_basic)  # placeholder; we raise below
        else:
            # Alien-missing row, or a synthetic row under either policy.
            row_status_out.append(s_basic)
            stats["n_rows_defaulted"] += 1

    if policy == "exact" and (missing_cols or missing_rows):
        raise ValueError(
            "exact basis transfer requires every non-synthetic target name "
            "in the carrier, but "
            f"{len(missing_cols)} column(s) and {len(missing_rows)} row(s) "
            "were missing "
            f"(e.g. cols={missing_cols[:5]}, rows={missing_rows[:5]}); "
            "the fingerprints should have matched — refusing to mis-size the basis"
        )

    basis = HighsBasis()
    basis.col_status = [status_from_int(v, HighsBasisStatus) for v in col_status_out]
    basis.row_status = [status_from_int(v, HighsBasisStatus) for v in row_status_out]
    basis.alien = policy == "alien"
    return basis, stats
