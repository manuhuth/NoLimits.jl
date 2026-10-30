#!/bin/bash
# Full package test suite as a job array, one TEST_GROUPS entry of test/runtests.jl per
# task (same grouping and -O0 subprocess flags as CI):
#   sbatch --array=1-28 jobs/test_groups.sbatch
# Task 0 only instantiates/precompiles the test environment (run it first, alone).
#SBATCH --partition=intelsr_short
#SBATCH --account=ag_irumls_hasenauer
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --job-name=aog_testgrp
#SBATCH --output=logs/%x_%A_%a.log
set -u
B=${BT_ROOT:-$SLURM_SUBMIT_DIR}
[ -f "$B/lib/render.jl" ] || { echo "submit from dev/aog_battletest (or set BT_ROOT); got B=$B" >&2; exit 2; }
R=$(cd "$B/../.." && pwd)
cd $R/test
if [ "$SLURM_ARRAY_TASK_ID" = "0" ]; then
  JULIA_NUM_PRECOMPILE_TASKS=4 julia --project=$R/test -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
  echo "exit=$?"; exit 0
fi
echo "group=$SLURM_ARRAY_TASK_ID host=$(hostname)"; date
NL_TEST_GROUP=$SLURM_ARRAY_TASK_ID julia --project=$R/test --check-bounds=yes $R/test/runtests.jl
rc=$?
date; echo "exit=$rc"; exit $rc
