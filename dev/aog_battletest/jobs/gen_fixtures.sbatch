#!/bin/bash
# One fixture per array task. Submit with
#   sbatch --array=0-$(( $(ls fixtures/defs/*.jl | wc -l) - 1 )) jobs/gen_fixtures.sbatch
# or a single fixture: FIXTURES="name1 name2" sbatch jobs/gen_fixtures.sbatch
#SBATCH --partition=intelsr_short
#SBATCH --account=ag_irumls_hasenauer
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --job-name=aog_fixture
#SBATCH --output=/home/mhuth1_hpc/NoLimits-aog/dev/aog_battletest/logs/%x_%A_%a.log
set -u
B=/home/mhuth1_hpc/NoLimits-aog/dev/aog_battletest
cd $B
if [ -z "${FIXTURES:-}" ]; then
  mapfile -t ALL < <(ls fixtures/defs/*.jl | xargs -n1 basename | sed 's/\.jl$//' | sort)
  FIXTURES=${ALL[$SLURM_ARRAY_TASK_ID]}
fi
julia --project=$B/envs/${MODE:-new} -t $SLURM_CPUS_PER_TASK $B/lib/gen_fixture.jl $FIXTURES
