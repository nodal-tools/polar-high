"""polar-high — Python library for building indexed linear and mixed-integer programs in polars."""

# Default polars to a single thread for the LP-build workload.
# Rayon's coordination overhead consistently exceeds the parallel
# speedup on typical indexed-LP build patterns (see the benchmark
# page); single-thread is faster *and* leaner. Users who want more
# threads can set POLARS_MAX_THREADS before importing polar_high.
import os as _os

_os.environ.setdefault("POLARS_MAX_THREADS", "1")

from polar_high._warm_basis import (
    NamedBasis,
    basis_fingerprint,
)
from polar_high.benders import (
    BendersBoundInvalid,
    BendersLoopOptions,
    BendersLoopResult,
    BendersMaster,
    BendersStalled,
    BendersSubproblem,
    PointEvaluation,
    SubproblemHandle,
    SubproblemNotOptimal,
    SubproblemResult,
    evaluate_at_point,
    solve_benders_loop,
)
from polar_high.decomposition import (
    InOutStabilizer,
    StallMonitor,
    StallVerdict,
    TrustRegionStabilizer,
)
from polar_high.engine import (
    CstrRecord,
    Expr,
    Lag,
    Param,
    Problem,
    Solution,
    SolveDiagnostics,
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
    "BendersBoundInvalid",
    "BendersLoopOptions",
    "BendersLoopResult",
    "BendersMaster",
    "BendersStalled",
    "BendersSubproblem",
    "PointEvaluation",
    "SubproblemHandle",
    "SubproblemNotOptimal",
    "SubproblemResult",
    "evaluate_at_point",
    "solve_benders_loop",
    "Var",
    "Param",
    "Expr",
    "Sum",
    "Where",
    "Lag",
    "Problem",
    "WarmProblem",
    "Solution",
    "SolveDiagnostics",
    "NamedBasis",
    "basis_fingerprint",
    "CstrRecord",
    "InOutStabilizer",
    "StallMonitor",
    "StallVerdict",
    "TrustRegionStabilizer",
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
