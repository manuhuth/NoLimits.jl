#!/bin/bash
# Load-check fixtures with another environment, one fixture per array task:
#   MODE=baseline BT_FIXTURE_DATA=... sbatch --array=0-22 jobs/check_load.sbatch
#SBATCH --partition=intelsr_short
#SBATCH --account=ag_irumls_hasenauer
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:45:00
#SBATCH --job-name=aog_loadcheck
#SBATCH --output=logs/%x_%A_%a.log
set -u
B=${BT_ROOT:-$SLURM_SUBMIT_DIR}
[ -f "$B/lib/render.jl" ] || { echo "submit from dev/aog_battletest (or set BT_ROOT); got B=$B" >&2; exit 2; }
cd $B
if [ -z "${FIXTURES:-}" ]; then
  mapfile -t ALL < <(ls fixtures/defs/*.jl | xargs -n1 basename | sed 's/\.jl$//' | sort)
  FIXTURES=${ALL[$SLURM_ARRAY_TASK_ID]}
fi
julia --project=$B/envs/${MODE:-baseline} -t $SLURM_CPUS_PER_TASK $B/lib/check_load.jl ${MODE:-baseline} $FIXTURES
