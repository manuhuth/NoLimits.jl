#!/bin/bash
# Run selected test files of the package test suite in one batch (test env).
#   TESTS="plotting_functions_tests.jl integration_plotting.jl" sbatch jobs/tests.sbatch
#SBATCH --partition=intelsr_short
#SBATCH --account=ag_irumls_hasenauer
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --job-name=aog_tests
#SBATCH --output=logs/%x_%j.log
# Paths derive from the submit directory: run sbatch from dev/aog_battletest of the
# worktree you want to test (or set BT_ROOT). Logs go to ./logs (relative to it).
B=${BT_ROOT:-$SLURM_SUBMIT_DIR}
[ -f "$B/lib/render.jl" ] || { echo "submit from dev/aog_battletest (or set BT_ROOT); got B=$B" >&2; exit 2; }
R=$(cd "$B/../.." && pwd)
cd $R/test
julia --project=$R/test -t $SLURM_CPUS_PER_TASK -e 'using Pkg; Pkg.instantiate()'
julia --project=$R/test -t $SLURM_CPUS_PER_TASK $R/test/run_batch.jl ${TESTS:-plotting_functions_tests.jl integration_plotting.jl}
echo "exit=$?"
