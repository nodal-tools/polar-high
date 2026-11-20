"""Manual check of ``polar_high.kkt_report`` on large LPs (HiGHS 1.15 only).

Solves each model file with the given HiGHS options, captures HiGHS' log,
and compares the report's X with the X HiGHS logged for the full LP ("no
more than X" / "changed ... since relative violation of tolerances is X").
Prints the report's time next to HiGHS' run time.  Not part of the test
suite: the files are too big to commit.

    python benchmark/kkt_report_check.py model.mps [more.mps ...] \\
        --opt solver=ipm --opt run_crossover=off --opt user_objective_scale=5

Exit code 1 when an X disagrees with the log.
"""

from __future__ import annotations

import argparse
import math
import re
import resource
import sys
import time

import highspy

from polar_high import kkt_report

_X = re.compile(
    r'(?:no more than|to "Unknown" since relative violation of tolerances is)\s+([-+0-9.eE]+|inf)'
)


def _logged_x(log: str) -> float | None:
    """The X HiGHS logged for the full LP in the space it solved."""
    block = log.split("Solution optimality conditions: After removing user scaling")[0]
    block = block.rsplit("Performed postsolve", 1)[-1]
    found = _X.findall(block)
    return float(found[-1]) if found else None


def _agrees(x: float, logged: float | None, status: str) -> bool:
    if logged is None:  # HiGHS prints X only for 1 < X from "Optimal"
        return x <= 1 if status == "kOptimal" else True
    return abs(x - logged) <= 0.5001 * 10 ** (math.floor(math.log10(abs(logged))) - 2)


def _option(text: str):
    key, value = text.split("=", 1)
    for cast in (int, float):
        try:
            return key, cast(value)
        except ValueError:
            pass
    return key, value


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--opt", action="append", default=[], type=_option, help="HiGHS option k=v")
    args = ap.parse_args()
    bad = 0
    for path in args.files:
        h = highspy.Highs()
        h.setOptionValue("log_to_console", False)
        log: list[str] = []
        h.setCallback(lambda _t, msg, *_a, log=log: log.append(msg), None)
        h.startCallback(highspy.cb.kCallbackLogging)
        h.readModel(path)
        for key, value in args.opt:
            h.setOptionValue(key, value)
        h.run()
        rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        t0 = time.perf_counter()
        rep = kkt_report(h)
        took = time.perf_counter() - t0
        grew = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - rss0) / 1024
        status = h.getModelStatus().name
        if rep is None:
            print(f"{path}: no report (HiGHS {h.version()}, status {status})")
            bad += 1
            continue
        x, logged = rep.max_tolerance_relative_violation, _logged_x("".join(log))
        ok = _agrees(x, logged, status)
        bad += not ok
        run = h.getRunTime()
        print(
            f"{path}: {h.getNumCol()} cols x {h.getNumRow()} rows, {status}, "
            f"basis={rep.basis_valid}, X={x:.4g}, logged={logged}, {'OK' if ok else 'MISMATCH'}; "
            f"report {took:.2f} s vs HiGHS {run:.1f} s ({100 * took / run:.2f} %), "
            f"peak RSS +{grew:.0f} MB"
        )
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
