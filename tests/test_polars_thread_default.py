"""polar_high's ``POLARS_MAX_THREADS`` default applies only before polars is imported.

polars reads the variable once, at import.  Setting it after polars has loaded does not change the
running process and only leaks into child processes, so polar_high must leave it alone then.
Each case runs in its own interpreter, because the import order is process-global.
"""

from __future__ import annotations

import os
import subprocess
import sys

_PROBE = "import os, sys; {pre}import polar_high; print(os.environ.get('POLARS_MAX_THREADS'))"


def _run(pre: str) -> str:
    env = {k: v for k, v in os.environ.items() if k != "POLARS_MAX_THREADS"}
    out = subprocess.run(
        [sys.executable, "-c", _PROBE.format(pre=pre)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    return out.stdout.strip().splitlines()[-1]


def test_default_set_when_polars_not_yet_imported() -> None:
    assert _run("") == "1"


def test_no_default_when_polars_already_imported() -> None:
    assert _run("import polars; ") == "None"
