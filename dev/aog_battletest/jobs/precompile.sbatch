#!/bin/bash
# Precompile the harness environments (new = this worktree, baseline = origin/main worktree).
# Run once after any dependency change; render jobs then only load cached pkgimages.
#SBATCH --partition=intelsr_short
#SBATCH --account=ag_irumls_hasenauer
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --job-name=aog_precompile
#SBATCH --output=/home/mhuth1_hpc/NoLimits-aog/dev/aog_battletest/logs/%x_%j.log
set -u
B=/home/mhuth1_hpc/NoLimits-aog/dev/aog_battletest
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
