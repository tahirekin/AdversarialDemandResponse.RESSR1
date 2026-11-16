#!/bin/zsh
# Full rerun after the review fixes (16 Nov 2026). Logs in logs/rerun_*.log
set -e
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONUNBUFFERED=1
echo "== calibration $(date)"
for E in 2016 2017 2023 2024 2025; do python3 calibrate.py $E > logs/rerun_calib_$E.log 2>&1 & done; wait
echo "== acceptance $(date)"
python3 tests/acceptance.py > logs/rerun_acceptance.log 2>&1 || true
grep -E "PASS|FAIL" logs/rerun_acceptance.log
echo "== version B $(date)"
python3 run_all.py B --procs 8 > logs/rerun_B.log 2>&1
grep -c ERROR logs/rerun_B.log || true
echo "== long chain $(date)"
python3 tests/longchain_check.py > logs/rerun_longchain.log 2>&1
echo "== tables $(date)"
python3 tables.py B > /dev/null 2>&1
python3 tables_tex.py 2>&1 | tail -1
python3 tests/design_checks.py > logs/rerun_design_checks.log 2>&1
echo "== done $(date)"
