"""Tests for :mod:`polar_high._log_routing`.

The hardening these lock in: ``route_highs_log_to_stdout`` must never
*suppress* HiGHS' native log unless ``sys.stdout`` is genuinely a different
sink from the native fd-1 write.  When the sink already is fd 1 (terminal /
pipe / file on fd 1), native logging is left intact — suppressing it and
betting on the logging callback loses the log on highspy builds whose
callback registers but never delivers (the symptom this guards against).
"""

from __future__ import annotations

import contextlib
import io

from polar_high import _log_routing
from polar_high._log_routing import (
    _sink_is_native_stdout,
    route_highs_log_to_stdout,
)


class _FakeHighs:
    """Records the option/callback mutations ``route_*`` performs."""

    def __init__(self, output_flag: bool = True) -> None:
        self._opts = {"output_flag": output_flag, "log_to_console": True}
        self.callbacks_set = 0
        self.callbacks_started: list[object] = []
        self._cb = None

    def getOptionValue(self, name: str):  # noqa: N802 — highspy spelling
        # highspy 1.x returns ``(status, value)``.
        return (0, self._opts[name])

    def setOptionValue(self, name: str, value) -> None:  # noqa: N802
        self._opts[name] = value

    def setCallback(self, cb, data) -> None:  # noqa: N802
        self.callbacks_set += 1
        self._cb = cb

    def startCallback(self, kind) -> None:  # noqa: N802
        self.callbacks_started.append(kind)


class _FakeStream:
    """Stream whose ``fileno`` is controllable; ``fd=None`` => no usable fd."""

    def __init__(self, fd: int | None) -> None:
        self._fd = fd
        self.buf: list[str] = []

    def fileno(self) -> int:
        if self._fd is None:
            raise io.UnsupportedOperation("stream has no fileno")
        return self._fd

    def write(self, s: str) -> None:
        self.buf.append(s)

    def flush(self) -> None:
        pass


# --------------------------------------------------------------------------
# _sink_is_native_stdout
# --------------------------------------------------------------------------


def test_sink_is_native_stdout_classification():
    assert _sink_is_native_stdout(_FakeStream(fd=1)) is True
    assert _sink_is_native_stdout(_FakeStream(fd=5)) is False
    # No real fd: StringIO, an object without ``fileno`` — both non-native.
    assert _sink_is_native_stdout(io.StringIO()) is False
    assert _sink_is_native_stdout(object()) is False


# --------------------------------------------------------------------------
# route_highs_log_to_stdout — gating
# --------------------------------------------------------------------------


def test_fd1_sink_leaves_native_logging_untouched():
    h = _FakeHighs(output_flag=True)
    route_highs_log_to_stdout(h, stream=_FakeStream(fd=1))
    # No callback registered, native console write NOT suppressed.
    assert h.callbacks_set == 0
    assert h._opts["log_to_console"] is True
    # Not stamped routed: a later non-fd1 solve must be re-evaluated.
    assert not getattr(h, _log_routing._ROUTED_ATTR, False)


def test_non_fd1_sink_routes_and_suppresses_native():
    h = _FakeHighs(output_flag=True)
    stream = _FakeStream(fd=None)  # like StringIO / ipykernel OutStream
    route_highs_log_to_stdout(h, stream=stream)
    assert h.callbacks_set == 1
    assert h._opts["log_to_console"] is False
    assert getattr(h, _log_routing._ROUTED_ATTR, False) is True
    # The registered callback re-emits through the sink.
    h._cb(None, "Running HiGHS\n", None, None, None)
    assert "".join(stream.buf) == "Running HiGHS\n"


def test_fd_other_than_one_routes():
    # A sink backed by a real-but-non-stdout fd still diverges from native
    # fd-1, so it must be routed.
    h = _FakeHighs(output_flag=True)
    route_highs_log_to_stdout(h, stream=_FakeStream(fd=5))
    assert h.callbacks_set == 1
    assert h._opts["log_to_console"] is False


def test_silent_solve_is_noop():
    h = _FakeHighs(output_flag=False)
    route_highs_log_to_stdout(h, stream=_FakeStream(fd=None))
    assert h.callbacks_set == 0
    assert h._opts["log_to_console"] is True


def test_native_log_env_opts_out(monkeypatch):
    monkeypatch.setenv("POLAR_HIGH_NATIVE_LOG", "1")
    h = _FakeHighs(output_flag=True)
    route_highs_log_to_stdout(h, stream=_FakeStream(fd=None))
    assert h.callbacks_set == 0
    assert h._opts["log_to_console"] is True


def test_already_routed_is_idempotent():
    h = _FakeHighs(output_flag=True)
    setattr(h, _log_routing._ROUTED_ATTR, True)
    route_highs_log_to_stdout(h, stream=_FakeStream(fd=None))
    assert h.callbacks_set == 0


# --------------------------------------------------------------------------
# Real solves — exercise the actual highspy callback on each path
# --------------------------------------------------------------------------


def _trivial_problem():
    import polars as pl

    from polar_high import Problem

    p = Problem()
    df = pl.DataFrame({"i": [0, 1, 2]})
    v = p.add_var("x", dims=("i",), index=df, lower=0.0, upper=10.0)
    p.set_objective(v.to_expr())
    return p


def test_real_solve_routes_to_non_fd1_sink():
    # sys.stdout redirected to a StringIO (no fileno) => callback path; the
    # HiGHS banner must reach the buffer.
    from polar_high.solvers import solve

    buf = io.StringIO()
    p = _trivial_problem()
    with contextlib.redirect_stdout(buf):
        solve(p, solver_name="highs")
    assert "Running HiGHS" in buf.getvalue()


def test_real_solve_uses_native_when_fd1(capfd):
    # ``capfd`` redirects the OS-level fd 1 but leaves ``sys.stdout.fileno()``
    # == 1, so the native path is taken and HiGHS' own fd-1 log is captured.
    from polar_high.solvers import solve

    p = _trivial_problem()
    solve(p, solver_name="highs")
    assert "Running HiGHS" in capfd.readouterr().out
