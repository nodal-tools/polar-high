#!/usr/bin/env bash
# Overnight benchmark sweep, cgroup-instrumented harness.
# Each cell is wrapped in its own systemd-run --user --scope so
# memory.peak reflects only that cell's allocations.
set -e

cd "$(dirname "$0")/.."
OUT_DIR="benchmark/results/cgroup"
mkdir -p "$OUT_DIR"

PY=python
LOG="$OUT_DIR/run.log"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

# Cells skipped on purpose (will not be re-run):
#   - pyomo at N >= 3000 dense: known timeout at 10 min in docs
#   - pyomo_net at N >= 3000: known timeout
# Everything else gets a 15 min per-cell timeout so a one-off bad cell
# doesn't stall the whole night.

run() {
    echo "[$(ts)] $*" | tee -a "$LOG"
    "$@" >> "$LOG" 2>&1
}

echo "[$(ts)] === A. dense FULL HiGHS solve ===" | tee -a "$LOG"
# polar (save_memory=False, the production default) + linopy at all sizes
run $PY benchmark/run.py --tools polar polar_sm linopy \
    --sizes 10 30 100 300 1000 3000 \
    --threads 1 --repeats 3 --timeout 900 \
    --out "$OUT_DIR/dense_fullsolve.csv"
# pyomo capped at N=1000 (timeout at larger sizes is established)
run $PY benchmark/run.py --tools pyomo \
    --sizes 10 30 100 300 1000 \
    --threads 1 --repeats 3 --timeout 900 \
    --append --out "$OUT_DIR/dense_fullsolve.csv"

echo "[$(ts)] === B. dense build-only (HiGHS short-circuit) ===" | tee -a "$LOG"
run $PY benchmark/run.py --tools polar polar_sm linopy \
    --sizes 10 30 100 300 1000 3000 \
    --threads 1 --repeats 3 --timeout 900 --time-limit 1e-6 \
    --out "$OUT_DIR/dense_buildonly.csv"
run $PY benchmark/run.py --tools pyomo \
    --sizes 10 30 100 300 1000 \
    --threads 1 --repeats 3 --timeout 900 --time-limit 1e-6 \
    --append --out "$OUT_DIR/dense_buildonly.csv"

echo "[$(ts)] === C. network LP (build-only) ===" | tee -a "$LOG"
run $PY benchmark/run.py --tools polar_net polar_sm_net linopy_net \
    --sizes 100 300 1000 3000 10000 \
    --threads 1 --repeats 3 --timeout 900 --time-limit 1e-6 \
    --out "$OUT_DIR/network.csv"
run $PY benchmark/run.py --tools pyomo_net \
    --sizes 100 300 1000 \
    --threads 1 --repeats 3 --timeout 900 --time-limit 1e-6 \
    --append --out "$OUT_DIR/network.csv"

echo "[$(ts)] === D. threads sweep on dense LP at N=300 ===" | tee -a "$LOG"
run $PY benchmark/run.py --tools polar polar_sm linopy pyomo \
    --sizes 300 \
    --threads 1 4 16 32 --repeats 3 --timeout 600 --time-limit 1e-6 \
    --out "$OUT_DIR/threads_dense.csv"

echo "[$(ts)] === E. network LP threading: polar @1 & @32 ===" | tee -a "$LOG"
run $PY benchmark/run.py --tools polar_net polar_sm_net linopy_net \
    --sizes 100 300 1000 3000 10000 \
    --threads 1 32 --repeats 3 --timeout 900 --time-limit 1e-6 \
    --out "$OUT_DIR/network_threads.csv"

echo "[$(ts)] === DONE ===" | tee -a "$LOG"
