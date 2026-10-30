#!/bin/bash
# Run selected test files of the package test suite in one batch (test env).
#   TESTS="plotting_functions_tests.jl integration_plotting.jl" sbatch jobs/tests.sbatch
#SBATCH --partition=intelsr_short
#SBATCH --account=ag_irumls_hasenauer
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --job-name=aog_tests
#SBATCH --output=/home/mhuth1_hpc/NoLimits-aog/dev/aog_battletest/logs/%x_%j.log
R=/home/mhuth1_hpc/NoLimits-aog
cd $R/test
julia --project=$R/test -t $SLURM_CPUS_PER_TASK -e 'using Pkg; Pkg.instantiate()'
julia --project=$R/test -t $SLURM_CPUS_PER_TASK $R/test/run_batch.jl ${TESTS:-plotting_functions_tests.jl integration_plotting.jl}
echo "exit=$?"
