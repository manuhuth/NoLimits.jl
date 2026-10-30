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
#SBATCH --output=logs/%x_%A_%a.log
set -u
# Paths derive from the submit directory: run sbatch from dev/aog_battletest of the
# worktree you want to test (or set BT_ROOT). Logs go to ./logs (relative to it).
B=${BT_ROOT:-$SLURM_SUBMIT_DIR}
[ -f "$B/lib/render.jl" ] || { echo "submit from dev/aog_battletest (or set BT_ROOT); got B=$B" >&2; exit 2; }
R=$(cd "$B/../.." && pwd)
cd $B
if [ -z "${FIXTURES:-}" ]; then
  mapfile -t ALL < <(ls fixtures/defs/*.jl | xargs -n1 basename | sed 's/\.jl$//' | sort)
  FIXTURES=${ALL[$SLURM_ARRAY_TASK_ID]}
fi
julia --project=$B/envs/${MODE:-new} -t $SLURM_CPUS_PER_TASK $B/lib/gen_fixture.jl $FIXTURES
