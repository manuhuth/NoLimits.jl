#!/bin/bash
# Build out/summary.csv, side-by-side images (out/compare/) and contact sheets (out/sheets/).
#   sbatch jobs/compare.sbatch                                  # everything, 1 task
#   FILTER='^plot_fits__' NCHUNKS=8 sbatch --array=1-8 jobs/compare.sbatch
#SBATCH --partition=intelsr_short
#SBATCH --account=ag_irumls_hasenauer
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --job-name=aog_compare
#SBATCH --output=logs/%x_%A_%a.log
set -u
# Paths derive from the submit directory: run sbatch from dev/aog_battletest of the
# worktree you want to test (or set BT_ROOT). Logs go to ./logs (relative to it).
B=${BT_ROOT:-$SLURM_SUBMIT_DIR}
[ -f "$B/lib/render.jl" ] || { echo "submit from dev/aog_battletest (or set BT_ROOT); got B=$B" >&2; exit 2; }
R=$(cd "$B/../.." && pwd)
cd $B
julia --project=$B/envs/new $B/lib/compare.jl "${FILTER:-}" ${SLURM_ARRAY_TASK_ID:-1} ${NCHUNKS:-1}
