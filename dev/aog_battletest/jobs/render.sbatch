#!/bin/bash
# Render the case matrix in chunks. Examples (from dev/aog_battletest):
#   MODE=baseline sbatch --array=1-40 jobs/render.sbatch            # NCHUNKS defaults to 40
#   MODE=new NCHUNKS=10 FILTER='^plot_(fits|data)__' sbatch --array=1-10 jobs/render.sbatch
#   FORCE=1 re-renders cases that already have a status file.
#SBATCH --partition=intelsr_short
#SBATCH --account=ag_irumls_hasenauer
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:30:00
#SBATCH --job-name=aog_render
#SBATCH --output=logs/%x_%A_%a.log
set -u
# Paths derive from the submit directory: run sbatch from dev/aog_battletest of the
# worktree you want to test (or set BT_ROOT). Logs go to ./logs (relative to it).
B=${BT_ROOT:-$SLURM_SUBMIT_DIR}
[ -f "$B/lib/render.jl" ] || { echo "submit from dev/aog_battletest (or set BT_ROOT); got B=$B" >&2; exit 2; }
R=$(cd "$B/../.." && pwd)
MODE=${MODE:-new}
cd $B
echo "mode=$MODE chunk=$SLURM_ARRAY_TASK_ID/${NCHUNKS:-40} filter=${FILTER:-} host=$(hostname)"
julia --project=$B/envs/$MODE -t $SLURM_CPUS_PER_TASK $B/lib/render.jl $MODE $SLURM_ARRAY_TASK_ID ${NCHUNKS:-40} "${FILTER:-}"
