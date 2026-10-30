#!/bin/bash
# Precompile the harness environments (new = this worktree, baseline = origin/main worktree).
# Run once after any dependency change; render jobs then only load cached pkgimages.
#SBATCH --partition=intelsr_short
#SBATCH --account=ag_irumls_hasenauer
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --job-name=aog_precompile
#SBATCH --output=logs/%x_%j.log
set -u
# Paths derive from the submit directory: run sbatch from dev/aog_battletest of the
# worktree you want to test (or set BT_ROOT). Logs go to ./logs (relative to it).
B=${BT_ROOT:-$SLURM_SUBMIT_DIR}
[ -f "$B/lib/render.jl" ] || { echo "submit from dev/aog_battletest (or set BT_ROOT); got B=$B" >&2; exit 2; }
R=$(cd "$B/../.." && pwd)
export JULIA_NUM_PRECOMPILE_TASKS=8
for m in ${MODES:-new baseline}; do
  echo "=== $m"; date
  julia --project=$B/envs/$m -t 8 -e '
    using Pkg; Pkg.precompile()
    t = @elapsed @eval using NoLimits, CairoMakie, AlgebraOfGraphics, DataFrames, Distributions, JLD2, Serialization
    println("load time: ", t)
    import Turing
    println("ext loaded: ", Base.get_extension(NoLimits, :NoLimitsMakieExt) !== nothing)'
  date
done
