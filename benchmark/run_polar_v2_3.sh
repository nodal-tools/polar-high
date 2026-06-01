#!/usr/bin/env bash
# Re-run polar / polar_sm / polar_net / polar_sm_net at v2.3.0 with the
# cgroup-instrumented harness, appending into the cgroup CSVs (which
# have had pre-v2.3.0 polar* rows pruned ahead of time).
# linopy and pyomo cells are unchanged from the overnight sweep.
set -e

cd "$(dirname "$0")/.."
OUT_DIR="benchmark/results/cgroup"
PY=python
LOG="$OUT_DIR/run_polar_v2_3.log"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
run() {
    echo "[$(ts)] $*" | tee -a "$LOG"
    "$@" >> "$LOG" 2>&1
}

# polar_sm / polar_sm_net data was captured today after the perf commits
# already landed, so it stays. Only polar / polar_net (regular) needs the
# v2.3.0 re-run.

echo "[$(ts)] === A. dense FULL HiGHS solve (polar) ===" | tee -a "$LOG"
run $PY benchmark/run.py --tools polar \
    --sizes 10 30 100 300 1000 3000 \
    --threads 1 --repeats 3 --timeout 900 \
    --append --out "$OUT_DIR/dense_fullsolve.csv"

echo "[$(ts)] === B. dense build-only (polar) ===" | tee -a "$LOG"
run $PY benchmark/run.py --tools polar \
    --sizes 10 30 100 300 1000 3000 \
    --threads 1 --repeats 3 --timeout 900 --time-limit 1e-6 \
    --append --out "$OUT_DIR/dense_buildonly.csv"

echo "[$(ts)] === C. network LP (polar_net) ===" | tee -a "$LOG"
run $PY benchmark/run.py --tools polar_net \
    --sizes 100 300 1000 3000 10000 \
    --threads 1 --repeats 3 --timeout 900 --time-limit 1e-6 \
    --append --out "$OUT_DIR/network.csv"

echo "[$(ts)] === D. threads sweep on dense LP at N=300 (polar) ===" | tee -a "$LOG"
run $PY benchmark/run.py --tools polar \
    --sizes 300 \
    --threads 1 4 16 32 --repeats 3 --timeout 600 --time-limit 1e-6 \
    --append --out "$OUT_DIR/threads_dense.csv"

echo "[$(ts)] === E. network LP threading (polar_net @1 & @32) ===" | tee -a "$LOG"
run $PY benchmark/run.py --tools polar_net \
    --sizes 100 300 1000 3000 10000 \
    --threads 1 32 --repeats 3 --timeout 900 --time-limit 1e-6 \
    --append --out "$OUT_DIR/network_threads.csv"

echo "[$(ts)] === DONE ===" | tee -a "$LOG"
