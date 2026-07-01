"""polar-high — Python library for building indexed linear and mixed-integer programs in polars."""

# Default polars to a single thread for the LP-build workload.
# Rayon's coordination overhead consistently exceeds the parallel
# speedup on typical indexed-LP build patterns (see the benchmark
# page); single-thread is faster *and* leaner. Users who want more
# threads can set POLARS_MAX_THREADS before importing polar_high.
import os as _os

_os.environ.setdefault("POLARS_MAX_THREADS", "1")

from polar_high.decomposition import (
    StallMonitor,
    StallVerdict,
)
from polar_high.engine import (
    CstrRecord,
    Expr,
    Lag,
    Param,
    Problem,
    Solution,
    Sum,
    Var,
    WarmProblem,
    Where,
)
from polar_high.parallel import (
    prewarm_global_scheduler,
    resolve_worker_count,
    solve_indexed_parallel,
)
from polar_high.solvers import (
    IOMode,
    LicenseError,
    SolverError,
    SolverNotAvailableError,
    SolverResult,
    SolverStatus,
    solve,
)

__all__ = [
    "Var",
    "Param",
    "Expr",
    "Sum",
    "Where",
    "Lag",
    "Problem",
    "WarmProblem",
    "Solution",
    "CstrRecord",
    "StallMonitor",
    "StallVerdict",
    "prewarm_global_scheduler",
    "resolve_worker_count",
    "solve_indexed_parallel",
    "solve",
    "IOMode",
    "SolverStatus",
    "SolverResult",
    "SolverNotAvailableError",
    "LicenseError",
    "SolverError",
]
