#!/bin/bash
#SBATCH --partition=intelsr_devel
#SBATCH --account=ag_irumls_hasenauer
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --job-name=aog_smoke
#SBATCH --output=logs/%x_%j.log
# Paths derive from the submit directory: run sbatch from dev/aog_battletest of the
# worktree you want to test (or set BT_ROOT). Logs go to ./logs (relative to it).
B=${BT_ROOT:-$SLURM_SUBMIT_DIR}
[ -f "$B/lib/render.jl" ] || { echo "submit from dev/aog_battletest (or set BT_ROOT); got B=$B" >&2; exit 2; }
R=$(cd "$B/../.." && pwd)
cd $B && julia --project=$B/envs/${MODE:-new} $B/lib/smoke.jl
