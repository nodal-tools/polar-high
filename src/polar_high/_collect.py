"""Collect lazy join chains in the order they are written.

polars 2.x added a join-order optimisation (``QueryOptFlags.join_order``,
on by default) that reorders a run of inner equi-joins by estimated
cardinality.  On polar-high's coefficient chains the estimate can pick a
join on a non-unique key subset first: in the Sum relabel emit
(``result ⋈ map(p, source, sink → n) ⋈ row_index(n, d, t)``) it joins
``row_index`` to ``result`` on ``(d, t)`` alone, a many-to-many product of
tens of millions of rows, before the map join narrows it.  The result is
the same rows, but the peak memory grows by several GB per call.

The chains here are written so that each join narrows or relabels the
accumulator before the next one (the order polars 1.x always ran), so they
are collected with join reordering off.  On polars 1.x, which has no such
flag, the helpers are a plain collect.

Every collect of a plan that polar-high composes goes through these helpers,
not only the relabel emit: Param x Param and Var x Param products, ``Where``
frames, the ``Sum`` emits and the row-index attach are all inner joins, and a
``Param.lazy`` or term plan handed to a later left join or sort can carry such
a chain inside it.  A plan with no run of inner joins is collected exactly as
before.  A ``Var`` frame the caller passes in lazily is the caller's own plan
and is collected with the caller's polars defaults.
"""

from __future__ import annotations

import inspect

import polars as pl


def _no_join_reorder_flags() -> pl.QueryOptFlags | None:
    """Default optimisation flags with ``join_order`` off, or ``None`` when
    the installed polars has no ``join_order`` flag (polars 1.x)."""
    flags_cls = getattr(pl, "QueryOptFlags", None)
    if flags_cls is None:
        return None
    try:
        params = inspect.signature(flags_cls).parameters
    except (TypeError, ValueError):
        return None
    if "join_order" not in params:
        return None
    return flags_cls(join_order=False)


#: ``QueryOptFlags(join_order=False)`` on polars 2.x, ``None`` on 1.x.
NO_JOIN_REORDER: pl.QueryOptFlags | None = _no_join_reorder_flags()


def _opt_kwargs() -> dict:
    return {} if NO_JOIN_REORDER is None else {"optimizations": NO_JOIN_REORDER}


def collect_in_join_order(plan: pl.LazyFrame) -> pl.DataFrame:
    """``plan.collect()`` with the joins run in the order they are written."""
    return plan.collect(**_opt_kwargs())


def collect_streaming_in_join_order(plan: pl.LazyFrame) -> pl.DataFrame:
    """Collect ``plan`` with the streaming engine and joins in written order.

    Falls back to the legacy ``streaming=True`` keyword and then to the
    in-memory engine when the streaming engine is unavailable or rejects the
    plan, as the streaming collects in polar-high always have.
    """
    kw = _opt_kwargs()
    try:
        return plan.collect(engine="streaming", **kw)
    except TypeError:
        try:
            return plan.collect(streaming=True, **kw)
        except TypeError:
            return plan.collect(**kw)
    except Exception:
        return plan.collect(**kw)
