"""Route HiGHS solver log output through Python's ``sys.stdout``.

HiGHS (via ``highspy``) writes its log to the C-level standard-output file
descriptor (fd 1), not Python's ``sys.stdout``.  Most consoles forward fd 1
just fine, but some do not:

* **Jupyter / Spine-Toolbox Basic Console on Windows.**  ``ipykernel`` only
  redirects the low-level fd on POSIX; on Windows the C-runtime ``stdout`` of
  the ``highspy`` extension is never captured, so the entire HiGHS log
  vanishes while Python-level ``print`` output is still visible.
* Anything relying on ``contextlib.redirect_stdout`` or ``pytest``'s
  ``capsys`` — those patch ``sys.stdout``, which the native fd-1 writes
  bypass.

:func:`route_highs_log_to_stdout` handles those cases by installing a HiGHS
*logging callback* that re-emits each message through ``sys.stdout`` and
suppressing the native console write (``log_to_console=False``), so the log
appears exactly once and integrates with Python stream redirection.

It does this **only when ``sys.stdout`` is not the native fd-1 sink** — i.e.
when re-routing actually changes where the log lands.  When ``sys.stdout`` is
backed by fd 1 (a terminal, a pipe such as the FlexTool GUI reading a
subprocess, or a file opened on fd 1), HiGHS' native log already reaches that
exact sink, so the function leaves native logging untouched.  Suppressing the
native write there would be a bet that the callback fires, and some
``highspy`` builds register the ``kCallbackLogging`` callback but never
deliver a message — suppress-native + silent-callback then loses the log
entirely.  Routing only when ``sys.stdout`` diverges from fd 1 keeps the
robust native path on the common case and confines callback-dependence to
environments where the native write is unreachable anyway.

Set ``POLAR_HIGH_NATIVE_LOG=1`` to opt out and keep HiGHS' native fd-1
logging untouched everywhere.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import highspy

# Attribute stamped on a ``Highs`` instance once its log has been routed, so
# the install is idempotent (WarmProblem reuses one instance across solves).
_ROUTED_ATTR = "_polar_high_log_routed"
# Backref keeping the callback closure alive for the instance's lifetime.
# ``highspy.setCallback`` already retains a reference, but stamping it here is
# cheap belt-and-braces against any future highspy that does not.
_CB_ATTR = "_polar_high_log_cb"


def _sink_is_native_stdout(stream: Any) -> bool:
    """Return True when the log sink is backed by the native stdout fd (fd 1).

    In that case HiGHS' own native log already reaches the sink, so the
    ``sys.stdout`` callback is redundant and suppressing the native write
    would risk losing the log on ``highspy`` builds whose logging callback is
    silent.  A stream with no real fd — ``io.StringIO``, an ``ipykernel``
    ``OutStream``, a ``pytest`` capture buffer, or a closed stream — is
    treated as *not* native stdout, so the callback is used to surface the
    log.  (``io.UnsupportedOperation`` subclasses both ``OSError`` and
    ``ValueError``, so the broad ``except`` covers "no usable fileno".)
    """
    target = stream if stream is not None else sys.stdout
    try:
        return target.fileno() == 1
    except (AttributeError, OSError, ValueError):
        return False


def route_highs_log_to_stdout(h: Any, *, stream: Any = None) -> None:
    """Make ``Highs`` instance *h* log through Python ``sys.stdout``.

    No-op when:

    * ``POLAR_HIGH_NATIVE_LOG`` is set (operator opted out),
    * the instance was already routed (idempotent),
    * HiGHS output is disabled (``output_flag`` is false — a silent solve
      stays silent),
    * the sink is already backed by fd 1 (terminal / pipe / file on fd 1):
      HiGHS' native log reaches it directly, so native logging is left
      untouched rather than suppressed in favour of a callback that may be
      silent on some ``highspy`` builds.

    Defensive throughout: any ``highspy`` error leaves the native logging
    path untouched rather than risking a lost log or a broken solve.  The
    native console write is suppressed *only after* the callback is known to
    be registered, so a registration failure can never leave the solve with
    no log at all.

    Parameters
    ----------
    h
        A live ``highspy.Highs`` instance, after solver options have been
        applied and before ``run()``.
    stream
        Sink for log messages; defaults to the *current* ``sys.stdout`` at
        each call (so it follows any later redirection, e.g. ipykernel's).
    """
    if os.environ.get("POLAR_HIGH_NATIVE_LOG"):
        return
    if getattr(h, _ROUTED_ATTR, False):
        return

    # ``getOptionValue`` returns ``(status, value)`` in highspy 1.x.
    try:
        result = h.getOptionValue("output_flag")
        output_flag = result[1] if isinstance(result, tuple) else result
    except Exception:
        return
    if not output_flag:
        return

    # When the sink is the native stdout fd (fd 1), HiGHS' own log already
    # lands exactly where the callback would write it.  Re-emitting would
    # only duplicate it, and suppressing the native write to avoid that
    # duplication bets the callback fires — some highspy builds register the
    # callback but never deliver a message, which would then lose the log
    # (native suppressed + callback silent).  Leave native logging untouched:
    # it is the robust path and needs no callback.  Do *not* stamp
    # ``_ROUTED_ATTR`` here, so a later solve whose ``sys.stdout`` has been
    # redirected away from fd 1 (e.g. ``capsys``) is still re-evaluated.
    if _sink_is_native_stdout(stream):
        return

    def _log_callback(
        callback_type: Any,
        message: str,
        data_out: Any,
        data_in: Any,
        user_data: Any,
    ) -> None:
        # Resolve the sink lazily so redirection installed after registration
        # (common under Jupyter) is still honoured.  HiGHS messages already
        # carry their own trailing newline.
        out = stream if stream is not None else sys.stdout
        try:
            out.write(message)
            out.flush()
        except Exception:
            # A broken sink must never crash the solve.
            pass

    try:
        h.setCallback(_log_callback, None)
        h.startCallback(highspy.cb.kCallbackLogging)
    except Exception:
        # Could not register — leave HiGHS' native logging in place.
        return

    # Registered successfully: now silence the duplicate native console write
    # so the log appears exactly once (it would otherwise double up on
    # platforms where fd 1 *is* captured, e.g. POSIX Jupyter).
    try:
        h.setOptionValue("log_to_console", False)
    except Exception:
        pass

    try:
        setattr(h, _CB_ATTR, _log_callback)
        setattr(h, _ROUTED_ATTR, True)
    except Exception:
        # Pybind instance that rejects attributes — the callback ref held by
        # highspy itself still keeps it alive; we just lose idempotency.
        pass
