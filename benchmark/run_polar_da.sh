#!/usr/bin/env bash
# Run polar (regular, no dense_axes) and polar_da (dense_axes) at all
# headline cells, on whichever polar_high editable install is active
# (currently the worktree-block-coo branch).
set -e

cd "$(dirname "$0")/.."
OUT_DIR="benchmark/results/cgroup"
PY=python
LOG="$OUT_DIR/run_polar_da.log"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
run() {
    echo "[$(ts)] $*" | tee -a "$LOG"
    "$@" >> "$LOG" 2>&1
}

echo "[$(ts)] === A. dense FULL HiGHS solve (polar + polar_da) ===" | tee -a "$LOG"
run $PY benchmark/run.py --tools polar polar_da \
    --sizes 10 30 100 300 1000 3000 \
    --threads 1 --repeats 3 --timeout 900 \
    --append --out "$OUT_DIR/dense_fullsolve.csv"

echo "[$(ts)] === B. dense build-only (polar + polar_da) ===" | tee -a "$LOG"
run $PY benchmark/run.py --tools polar polar_da \
    --sizes 10 30 100 300 1000 3000 \
    --threads 1 --repeats 3 --timeout 900 --time-limit 1e-6 \
    --append --out "$OUT_DIR/dense_buildonly.csv"

echo "[$(ts)] === C. network LP (polar_net + polar_da_net) ===" | tee -a "$LOG"
run $PY benchmark/run.py --tools polar_net polar_da_net \
    --sizes 100 300 1000 3000 10000 \
    --threads 1 --repeats 3 --timeout 900 --time-limit 1e-6 \
    --append --out "$OUT_DIR/network.csv"

echo "[$(ts)] === D. threads sweep on dense LP at N=300 (polar + polar_da) ===" | tee -a "$LOG"
run $PY benchmark/run.py --tools polar polar_da \
    --sizes 300 \
    --threads 1 4 16 32 --repeats 3 --timeout 600 --time-limit 1e-6 \
    --append --out "$OUT_DIR/threads_dense.csv"

echo "[$(ts)] === E. network LP threading (polar_net + polar_da_net @1 & @32) ===" | tee -a "$LOG"
run $PY benchmark/run.py --tools polar_net polar_da_net \
    --sizes 100 300 1000 3000 10000 \
    --threads 1 32 --repeats 3 --timeout 900 --time-limit 1e-6 \
    --append --out "$OUT_DIR/network_threads.csv"

echo "[$(ts)] === DONE ===" | tee -a "$LOG"
