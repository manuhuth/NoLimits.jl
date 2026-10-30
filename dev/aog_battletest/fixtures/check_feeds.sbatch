#!/bin/bash
# Check fixture `feeds` against a plotting implementation (default: baseline = origin/main).
#   cd dev/aog_battletest && sbatch --array=0-18 fixtures/check_feeds.sbatch
#   FIXTURES="a b" MODE=new sbatch fixtures/check_feeds.sbatch
#SBATCH --partition=intelsr_short
#SBATCH --account=ag_irumls_hasenauer
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --job-name=aog_checkfeeds
#SBATCH --output=logs/%x_%A_%a.log
set -u
# Paths derive from the submit directory: run sbatch from dev/aog_battletest of the
# worktree you want to test (or set BT_ROOT). Logs go to ./logs (relative to it).
B=${BT_ROOT:-$SLURM_SUBMIT_DIR}
[ -f "$B/lib/render.jl" ] || { echo "submit from dev/aog_battletest (or set BT_ROOT); got B=$B" >&2; exit 2; }
R=$(cd "$B/../.." && pwd)
cd $B
export MODE=${MODE:-baseline}
if [ -z "${FIXTURES:-}" ]; then
  mapfile -t ALL < <(ls fixtures/defs/*.jl | xargs -n1 basename | sed 's/\.jl$//' | sort)
  FIXTURES=${ALL[$SLURM_ARRAY_TASK_ID]}
fi
julia --project=$B/envs/$MODE -t $SLURM_CPUS_PER_TASK $B/fixtures/check_feeds.jl $FIXTURES
